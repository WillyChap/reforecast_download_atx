"""
Bias per calendar day (1-365), smoothed along lead, the fields atx_meteo.bias reads:

    {FINAL_BASE}/{var}/daily_hourly_gaussian/{var}_bias_climo_{MM-DD}_hourly_gaussian.zarr
        {var}_bias (lead_hour 0..1104 hourly, latitude, longitude)

Method (unchanged from the 49r1 bias fields):
  1. start-date biases from step 2 within ±15 days, cyclic Gaussian weights (sigma 7.5 days),
     NaN-skipping weighted mean
  2. smoothing along lead per hour of day (00, 06, 12, 18 UTC): local linear fit with
     Epanechnikov weights, half-width 0 up to day 5, rising to 7 days at day 28
  3. linear interpolation of leads from 6-hourly to hourly

A day is rebuilt only when the start dates in its window, or their step-2 builds,
have changed. Days with no start date within ±15 days are skipped; atx_meteo.bias
then uses the nearest built day. The record of what each day used is
{FINAL_BASE}/{var}/days.json.

Run:  python step3_bias.py                 (or --vars t2m, --days 05-21)
"""

import argparse

import numpy as np
import xarray as xr

import common as C

LEAD_SMOOTH_START = 5.0   # days: no smoothing up to here
LEAD_SMOOTH_FULL = 28.0   # days: full half-width from here on
LEAD_SMOOTH_MAX = 7.0     # days: maximum half-width


def half_width(day):
    ramp = (day - LEAD_SMOOTH_START) / (LEAD_SMOOTH_FULL - LEAD_SMOOTH_START)
    return LEAD_SMOOTH_MAX * np.clip(ramp, 0.0, 1.0)


def local_linear_operator(days):
    """Matrix S with smoothed = S @ values, for one hour-of-day series (days ascending)."""
    n = days.size
    S = np.eye(n)
    for i, d0 in enumerate(days):
        h = half_width(d0)
        if h < 1.0:                       # fewer than 3 points: leave as is
            continue
        u = (days - d0) / (h + 1.0)       # points at distance h keep a non-zero weight
        w = np.where(np.abs(u) < 1, 0.75 * (1 - u ** 2), 0.0)
        X = np.stack([np.ones(n), days - d0], axis=1)
        XtW = X.T * w
        S[i] = np.linalg.solve(XtW @ X, XtW)[0]
    return S


def lead_operator(lead_hours):
    """Block operator over all 6-hourly leads, acting within each hour-of-day group."""
    lead_hours = np.asarray(lead_hours)
    S = np.zeros((lead_hours.size, lead_hours.size))
    for hod in np.unique(lead_hours % 24):
        idx = np.where(lead_hours % 24 == hod)[0]
        S[np.ix_(idx, idx)] = local_linear_operator(lead_hours[idx] / 24.0)
    return S


def smooth_leads(da, S):
    vals = da.transpose("lead_hour", ...).values.copy()
    nan = np.isnan(vals)
    if nan.any():
        # fill a missing lead from its neighbours so it does not spread, then restore it
        flat = vals.reshape(vals.shape[0], -1)
        x = np.arange(flat.shape[0])
        for j in np.where(np.isnan(flat).any(axis=0))[0]:
            ok = ~np.isnan(flat[:, j])
            flat[:, j] = np.interp(x, x[ok], flat[ok, j]) if ok.any() else 0.0
    out = np.tensordot(S, vals, axes=(1, 0))
    out[nan] = np.nan
    return da.transpose("lead_hour", ...).copy(data=out.astype(np.float32))


def window(source_doys, target_doy):
    doys = np.array(sorted(source_doys))
    if doys.size == 0:
        return doys, doys, doys
    d = np.abs(doys - target_doy)
    d = np.minimum(d, C.N_DAYS - d)
    keep = d <= C.WINDOW_RADIUS_DAYS
    w = np.exp(-0.5 * (d[keep] / C.SIGMA_DAYS) ** 2)
    return doys[keep], d[keep], w / w.sum() if keep.any() else w


def build_day(var, doy, sources, cache, S):
    doys, dist, w = window(list(sources), doy)
    stack = xr.concat([cache[d] for d in doys], dim=xr.DataArray(doys, dims="source_dayofyear"))
    weights = xr.DataArray(w, dims="source_dayofyear", coords={"source_dayofyear": doys})
    daily_6h = smooth_leads(stack.weighted(weights).mean("source_dayofyear"), S)
    hours = np.arange(0, int(daily_6h.lead_hour.max()) + 1)
    hourly = daily_6h.interp(lead_hour=hours, method="linear").astype(np.float32)
    ds = hourly.transpose("lead_hour", "latitude", "longitude").to_dataset(name=f"{var}_bias")
    md = C.dayofyear_to_month_day(doy)
    ds.attrs.update({
        "bias_definition": "EC46 reforecast minus ERA5", "ifs_cycle": "50r1",
        "variable": var, "units": C.VARIABLES[var]["units"],
        "calendar_month_day": md, "calendar_dayofyear": int(doy),
        "calendar_estimation_method": "cyclic Gaussian weights over start dates, NaN-skipping",
        "gaussian_sigma_days": C.SIGMA_DAYS, "gaussian_window_radius_days": C.WINDOW_RADIUS_DAYS,
        "number_of_contributing_dates": int(doys.size),
        "source_dates_used": ",".join(sources[d] for d in doys),
        "source_distance_days": ",".join(str(int(x)) for x in dist),
        "source_weights_used": ",".join(f"{x:.6f}" for x in w),
        "lead_smoothing": "local linear, Epanechnikov, per hour of day; half-width 0 d up to "
                          f"day {LEAD_SMOOTH_START:g}, ramping to {LEAD_SMOOTH_MAX:g} d at day {LEAD_SMOOTH_FULL:g}",
        "lead_interpolation_method": "linear_6hourly_to_hourly (after lead smoothing)",
        "built": C.now(),
    })
    return ds.chunk({"lead_hour": -1, "latitude": 42, "longitude": 42})


def run_var(var, days_arg, force):
    src_manifest = C.read_json(C.source_manifest_path(var), {})
    sources = {C.month_day_to_dayofyear(md): md for md in src_manifest
               if C.done(C.source_mean_path(var, md))}
    stamps = {md: src_manifest[md]["built"] for md in sources.values()}
    manifest = C.read_json(C.final_manifest_path(var), {})
    days = [C.month_day_to_dayofyear(d) for d in days_arg] if days_arg else range(1, C.N_DAYS + 1)

    todo = []
    for doy in days:
        doys, _, _ = window(list(sources), doy)
        if doys.size == 0:
            continue
        md = C.dayofyear_to_month_day(doy)
        used = {sources[d]: stamps[sources[d]] for d in doys}
        if force or manifest.get(md) != used or not C.done(C.final_path(var, md)):
            todo.append((doy, md, doys, used))
    print(f"step3 {var}: {len(sources)} start dates, {len(todo)} days to build", flush=True)
    if not todo:
        return
    S, cache = None, {}
    for doy, md, doys, used in todo:          # in day order, so the cache holds about one window
        for d in list(cache):
            if d not in doys:
                del cache[d]
        for d in doys:
            if d not in cache:
                cache[d] = C.open_zarr(C.source_mean_path(var, sources[d]))[f"{var}_bias"].load()
        if S is None:
            S = lead_operator(cache[doys[0]].lead_hour.values)
        C.write_zarr(build_day(var, doy, sources, cache, S), C.final_path(var, md))
        manifest[md] = used
        C.write_json(manifest, C.final_manifest_path(var))
    print(f"step3 {var}: built {', '.join(t[1] for t in todo)}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--vars", nargs="+", default=list(C.VARIABLES), choices=list(C.VARIABLES))
    p.add_argument("--days", nargs="+", help="MM-DD (default: all 365)")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    a = p.parse_args()
    for var in a.vars:
        run_var(var, a.days, a.force)


if __name__ == "__main__":
    main()
