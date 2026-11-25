"""
Bias per reforecast start date (calendar MM-DD) and variable:

    bias(lead) = mean over past years of [ mean over 11 members of reforecast(year, lead)
                                           - ERA5(start + lead) ]

    {BIAS_BASE}/{var}/source_date_means_6h/{var}_bias_climo_{MM-DD}_6h.zarr
        {var}_bias   (lead_hour 0..1104 every 6 h, latitude, longitude)
        n_init_years (lead_hour)  past years with ERA5 at that lead

Each start date uses the newest reforecast folder of that calendar day. A lead
without ERA5 is NaN for that year, and the mean uses the other years. RH is
computed per member from t2m and d2m. ssrd and tp have no value at lead 0.

A start date is rebuilt when it is new, when a newer folder of the same calendar
day arrives, or when ERA5 months it lacked have since been added. The record of
what each bias used is {BIAS_BASE}/{var}/source_dates.json.

Run:  python step2_bias.py                 (or --month-days 05-21 05-23, --vars t2m)
"""

import argparse
from functools import lru_cache

import dask
import numpy as np
import pandas as pd
import xarray as xr

import common as C


@lru_cache(maxsize=4)
def era5_month(ym):
    path = f"{C.ERA5_GRID_DIR}/{ym}.zarr"
    return C.open_zarr(path) if C.done(path) else None


def era5_months_available():
    return {n[:7] for n in C.ls(C.ERA5_GRID_DIR) if n.endswith(".zarr") and C.done(f"{C.ERA5_GRID_DIR}/{n}")
            and int(C.open_zarr(f"{C.ERA5_GRID_DIR}/{n}").attrs.get("complete", 0))}


def era5_at(var, valid_times):
    """ERA5 on the reforecast grid at the valid times (NaN where missing) and the months lacking it."""
    months = sorted({t.strftime("%Y-%m") for t in valid_times})
    parts, lacking = [], []
    for ym in months:
        ds = era5_month(ym)
        if ds is None or var not in ds:
            lacking.append(ym)
        else:
            parts.append(ds[var])
            if not int(ds.attrs.get("complete", 0)):
                lacking.append(ym)
    if not parts:
        return None, lacking
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    return da.reindex(time=valid_times), lacking


def year_bias(hind, i, var):
    init = pd.Timestamp(hind.time.values[i])
    steps = pd.to_timedelta(hind.step.values)
    valid = pd.DatetimeIndex(init + steps)
    if var == "rh":
        fc = C.rh_from_t_td(hind["t2m"].isel(time=i).astype(np.float32), hind["d2m"].isel(time=i).astype(np.float32))
    else:
        fc = hind[var].isel(time=i).astype(np.float32)
    member_mean = fc.mean("number", skipna=True).transpose("step", "latitude", "longitude")
    obs, lacking = era5_at(var, valid)
    if obs is None:
        return None, lacking
    if not (np.allclose(obs.latitude, hind.latitude) and np.allclose(obs.longitude, hind.longitude)):
        raise ValueError("ERA5 grid from step 1 does not match the reforecast grid")
    bias = xr.DataArray(member_mean.data - obs.data, dims=("lead_hour", "latitude", "longitude"),
                        coords={"lead_hour": (steps / pd.Timedelta("1h")).astype(int),
                                "latitude": hind.latitude.values, "longitude": hind.longitude.values})
    return bias, lacking


def source_date_bias(md, var, folder):
    hind = C.open_zarr(C.hindcast_path(folder))
    years, biases, lacking_all = [], [], set()
    for i, init in enumerate(pd.DatetimeIndex(hind.time.values)):
        b, lacking = year_bias(hind, i, var)
        lacking_all.update(lacking)
        if b is None:
            continue
        years.append(int(init.year))
        biases.append(b)
    if not biases:
        return None, sorted(lacking_all)
    stack = xr.concat(biases, dim=xr.DataArray(years, dims="init_year", name="init_year"))
    ds = xr.Dataset({f"{var}_bias": stack.mean("init_year", skipna=True).astype(np.float32),
                     "n_init_years": stack.notnull().any(("latitude", "longitude")).sum("init_year").astype(np.int16)})
    ds.attrs.update({"bias_definition": "EC46 reforecast member mean minus ERA5, mean over past years",
                     "variable": var, "calendar_month_day": md, "source_folder": folder,
                     "init_years_used": ",".join(map(str, years)), "units": C.VARIABLES[var]["units"],
                     "era5_months_lacking": ",".join(sorted(lacking_all)), "built": C.now()})
    return ds.chunk({"lead_hour": -1, "latitude": 42, "longitude": 42}), sorted(lacking_all)


def needs_build(var, md, folder, manifest, available):
    rec = manifest.get(md)
    if rec is None or not C.done(C.source_mean_path(var, md)) or rec["folder"] != folder:
        return True
    return bool(set(rec.get("era5_lacking", [])) & available)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--vars", nargs="+", default=list(C.VARIABLES), choices=list(C.VARIABLES))
    p.add_argument("--month-days", nargs="+", help="MM-DD (default: every start date)")
    p.add_argument("--force", action="store_true", help="rebuild even if up to date")
    a = p.parse_args()

    folders = C.hindcast_folders()
    if a.month_days:
        folders = folders[folders.month_day.isin(a.month_days)]
    manifests = {v: C.read_json(C.source_manifest_path(v), {}) for v in a.vars}
    available = era5_months_available()
    plan = [(row.month_day, row.folder,
             [v for v in a.vars if a.force or needs_build(v, row.month_day, row.folder, manifests[v], available)])
            for row in folders.itertuples()]
    plan = [x for x in plan if x[2]]
    print(f"step2: {len(folders)} start dates, {len(plan)} to build", flush=True)
    client = C.start_cluster() if plan else None
    for md, folder, todo in plan:
        built = {v: source_date_bias(md, v, folder) for v in todo}
        writes = {v: C.write_zarr(ds, C.source_mean_path(v, md), compute=False)
                  for v, (ds, _) in built.items() if ds is not None}
        dask.compute(*writes.values())          # one graph: shared reforecast chunks are read once
        for v, (ds, lacking) in built.items():
            if ds is None:
                print(f"  {md} {v}: no ERA5 for any lead yet, skipped", flush=True)
                continue
            C.mark_done(C.source_mean_path(v, md))
            manifests[v][md] = {"folder": folder, "built": ds.attrs["built"], "era5_lacking": lacking}
            C.write_json(manifests[v], C.source_manifest_path(v))
        print(f"  {md}: {', '.join(todo)} from {folder}", flush=True)
    if client:
        client.close()


if __name__ == "__main__":
    main()
