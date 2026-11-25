"""
ERA5 at 00/06/12/18 UTC on the reforecast grid, one store per month:

    {BIAS_BASE}/era5_6h_on_ec46grid/{YYYY-MM}.zarr
        t2m, rh, tcc, u10, v10, u100, v100: value at the valid time
        ssrd, tp: amount over the 6 h ending at the valid time (sum of six hourly values)

Only the months that the downloaded reforecasts need are built (each past year
from its start to 46 days later). ERA5 is bilinearly interpolated; it covers
20-72N, so reforecast grid rows outside that band are NaN. A month with no ERA5
yet is skipped and tried again on the next run; a month with some ERA5 missing
is written with NaN there and rebuilt on later runs until it is complete.

Run:  python step1_era5.py                 (or --months 2025-05 2025-06)
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr

import common as C


def months_needed():
    months = set()
    for row in C.hindcast_folders().itertuples():
        hind = C.open_zarr(C.hindcast_path(row.folder))
        for t in pd.DatetimeIndex(hind.time.values):
            for d in pd.date_range(t.normalize(), t + pd.Timedelta(hours=int(C.STEPS[-1])), freq="D"):
                months.add(d.strftime("%Y-%m"))
    return sorted(months)


def era5_hourly(kind, name, hours):
    """ERA5 variable at the given hours (NaN where missing), from the yearly stores around them."""
    parts = []
    for y in sorted({h.year for h in hours} | {h.year - 1 for h in hours}):
        path = C.era5_store(kind, y)
        if path is None:
            continue
        da = C.open_zarr(path)[name]
        sel = hours[np.isin(hours, da.time.values)]
        if sel.size:
            parts.append(da.sel(time=sel))
    if not parts:
        return None
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.isel(time=np.unique(da.time.values, return_index=True)[1])
    return da.reindex(time=hours)


def era5_month(ym, lat, lon):
    start = pd.Timestamp(f"{ym}-01")
    times = pd.date_range(start, start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=18), freq="6h")
    out, missing = {}, {}
    for var, spec in C.VARIABLES.items():
        kind, name, factor = spec["era5"]
        if spec.get("accumulated"):
            hours = pd.date_range(times[0] - pd.Timedelta(hours=5), times[-1], freq="h")
            da = era5_hourly(kind, name, hours)
            if da is not None:
                da = da.rolling(time=6).sum(skipna=False).reindex(time=times)
        else:
            da = era5_hourly(kind, name, times)
        if da is None:
            missing[var] = times.size
            continue
        da = da.assign_coords(longitude=((da.longitude + 180) % 360) - 180).sortby("longitude")
        da = da.transpose("time", "latitude", "longitude").interp(latitude=lat, longitude=lon, method="linear")
        out[var] = (da * factor).astype(np.float32)
        out[var].attrs = {"era5_variable": name, "factor": factor, "units": spec["units"]}
    if not out:
        return None, missing
    ds = xr.Dataset(out)
    ds = ds.drop_vars([c for c in ds.coords if c not in ("time", "latitude", "longitude")])
    for var in out:                                   # count times with no ERA5 at all
        n = int(ds[var].isnull().all(("latitude", "longitude")).sum())
        if n:
            missing[var] = n
    ds.attrs.update({"description": "ERA5 at 00/06/12/18 UTC on the EC46 reforecast grid",
                     "interpolation": "bilinear", "complete": int(not missing),
                     "missing_times": str(missing), "built": C.now()})
    return ds.chunk({"time": 4, "latitude": -1, "longitude": -1}), missing


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("--months", nargs="+", help="YYYY-MM (default: every month the reforecasts need)")
    a = p.parse_args()

    folders = C.hindcast_folders()
    if folders.empty:
        print("step1: no complete reforecast folders yet")
        return
    grid = C.open_zarr(C.hindcast_path(folders.folder.iloc[0]))
    lat, lon = grid.latitude.values, grid.longitude.values
    months = a.months or months_needed()
    todo = []
    for ym in months:
        path = f"{C.ERA5_GRID_DIR}/{ym}.zarr"
        if C.done(path) and int(C.open_zarr(path).attrs.get("complete", 0)):
            continue
        todo.append(ym)
    print(f"step1: {len(months)} months needed, {len(todo)} to build", flush=True)
    client = C.start_cluster() if todo else None
    for ym in todo:
        ds, missing = era5_month(ym, lat, lon)
        if ds is None:
            print(f"  {ym}: no ERA5 yet, skipped", flush=True)
            continue
        C.write_zarr(ds, f"{C.ERA5_GRID_DIR}/{ym}.zarr")
        print(f"  {ym}: written" + (f" (times without ERA5: {missing})" if missing else ""), flush=True)
    if client:
        client.close()


if __name__ == "__main__":
    main()
