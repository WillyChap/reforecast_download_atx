"""
Download the EC46 reforecasts of IFS Cycle 50r1 from MARS into the atx S3 bucket.

    {HINDCAST_ROOT}/{date}/ec46_{date}.zarr
    dims time (10 past years, newest first), number (11: control + 10), step (185: 0-1104 h by 6),
         latitude (167: 74.76..15.0), longitude (487: -130..44.96), 0.36° grid
    t2m, d2m (K), tcc (0-1), u10, v10, u100, v100 (m/s)       instantaneous
    ssrd (W m-2, mean over the previous 6 h), tp (mm in the previous 6 h), NaN at step 0
    stored as 16-bit integers with scale and offset (GRIB itself packs at 16 bits)

Reforecast dates are the odd days of each month from 21 May 2026, the first 50r1 reforecast. Each date takes 20 MARS requests (control and perturbed members for each of 10 past years). Years are downloaded, checked, written and deleted one at a time, so local disk stays near 3 GB. A store is complete when its .done marker exists; a re-run skips complete dates and restarts an interrupted one.

Usage:
    python download.py --dry-run            # dates still to do
    python download.py --probe 2026-05-21   # every field of the control at steps 0-12, one year
    python download.py                      # all dates from 2026-05-21 to today
    python download.py --start 2026-06-01 --end 2026-06-30
"""

import argparse
import os
import shutil
import sys
import time

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

import common as C

FIRST_50R1 = pd.Timestamp("2026-05-21")
N_YEARS = 10
# MARS param -> store name (cfgrib uses the store names)
PARAMS = {"2t": "t2m", "2d": "d2m", "tcc": "tcc", "10u": "u10", "10v": "v10",
          "100u": "u100", "100v": "v100", "ssrd": "ssrd", "tp": "tp"}
ACCUMULATED = {"ssrd": 1 / 21600.0, "tp": 1000.0}    # running total -> 6 h mean W m-2 / 6 h total mm
# name: (attributes, packing range); values outside the range are clipped and counted
FIELDS = {
    "t2m": ({"units": "K", "long_name": "2 m temperature"}, (150.0, 350.0)),
    "d2m": ({"units": "K", "long_name": "2 m dew point temperature"}, (150.0, 350.0)),
    "tcc": ({"units": "(0 - 1)", "long_name": "total cloud cover"}, (0.0, 1.0)),
    "u10": ({"units": "m s**-1", "long_name": "10 m eastward wind"}, (-150.0, 150.0)),
    "v10": ({"units": "m s**-1", "long_name": "10 m northward wind"}, (-150.0, 150.0)),
    "u100": ({"units": "m s**-1", "long_name": "100 m eastward wind"}, (-150.0, 150.0)),
    "v100": ({"units": "m s**-1", "long_name": "100 m northward wind"}, (-150.0, 150.0)),
    "ssrd": ({"units": "W m**-2", "long_name": "surface solar radiation downwards, mean over the previous 6 h"},
             (0.0, 1500.0)),
    "tp": ({"units": "mm", "long_name": "total precipitation in the previous 6 h"}, (0.0, 1000.0)),
}
GRID, AREA = "0.36/0.36", "74.76/-130/15/44.96"
CHUNKS = {"time": 1, "number": 11, "step": 185, "latitude": 50, "longitude": 50}
DIMS = ("time", "number", "step", "latitude", "longitude")


def encoding():
    """int16 with scale and offset per field; -32768 marks missing."""
    enc = {}
    for v, (_, (lo, hi)) in FIELDS.items():
        enc[v] = {"dtype": "int16", "scale_factor": (hi - lo) / 65000.0, "add_offset": (hi + lo) / 2.0,
                  "_FillValue": np.int16(-32768), "chunks": tuple(CHUNKS.values())}
    return enc


def reforecast_dates(start, end):
    return [d for d in pd.date_range(start, end, freq="D") if d.day % 2 == 1]


def hdates(date, n_years):
    out = []
    for k in range(1, n_years + 1):
        try:
            out.append(date.replace(year=date.year - k))
        except ValueError:                     # 29 February in a non-leap year
            out.append(pd.Timestamp(date.year - k, 2, 28))
    return out


def request(date, hdate, kind):
    r = {"class": "od", "expver": "1", "stream": "eefh", "type": kind, "levtype": "sfc",
         "param": "/".join(PARAMS), "time": "00:00:00", "step": "0/to/1104/by/6",
         "date": date.strftime("%Y-%m-%d"), "hdate": hdate.strftime("%Y-%m-%d"),
         "grid": GRID, "area": AREA}
    if kind == "pf":
        r["number"] = "1/to/10"
    return r


def retrieve(server, req, target):
    if os.path.exists(target):
        return
    part = target + ".part"
    for attempt in range(1, 4):
        try:
            server.execute(req, part)
            os.replace(part, target)
            return
        except Exception as e:                 # MARS queue errors, network
            print(f"    MARS attempt {attempt} failed: {e}", flush=True)
            if attempt == 3:
                raise
            time.sleep(300 * attempt)


def read_grib(path):
    """
    {name: DataArray(number, step, latitude, longitude)}, lazily read. The file mixes
    level types and instantaneous and accumulated fields, so cfgrib returns several
    datasets; level coordinates are dropped.
    """
    out = {}
    for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
        for v in ds.data_vars:
            da = ds[v]
            if "number" not in da.dims:                 # control: number is a scalar coordinate (0)
                da = da.expand_dims(number=[int(da.number) if "number" in da.coords else 0])
            keep = {"number", "step", "latitude", "longitude", "time"}
            da = da.drop_vars([c for c in da.coords if c not in keep])
            out[v] = da.transpose("number", "step", "latitude", "longitude")
    missing = set(PARAMS.values()) - set(out)
    if missing:
        raise ValueError(f"{path}: fields missing: {sorted(missing)}")
    return out


def as_array(v, da, steps):
    """float32 (number, step, lat, lon); a running total becomes 6 h amounts, NaN at step 0."""
    a = da.values.astype(np.float32)
    if v in ACCUMULATED:
        if steps[0] != 0 or not np.all(np.diff(steps) == 6):
            raise ValueError("accumulated fields need steps 0, 6, 12, ...")
        d = np.diff(a, axis=1) * np.float32(ACCUMULATED[v])
        if d.size and float(np.nanmin(d)) < -0.01 * max(float(np.nanmax(d)), 1e-6):   # beyond packing noise
            raise ValueError(f"{v}: running total decreases by {float(np.nanmin(d)):.3g}")
        a = np.concatenate([np.full_like(a[:, :1], np.nan), np.maximum(d, 0)], axis=1)
    return a


def check_field(v, da, hdate):
    t = pd.Timestamp(da.time.values)
    if t != hdate:
        raise ValueError(f"{v}: GRIB start {t:%Y-%m-%d} != requested {hdate:%Y-%m-%d}")
    if list(da.number.values) != list(range(11)):
        raise ValueError(f"{v}: members {list(da.number.values)}")
    if not np.array_equal(pd.to_timedelta(da.step.values) / pd.Timedelta("1h"), C.STEPS):
        raise ValueError(f"{v}: steps differ from 0..1104 h by 6")
    if not (np.allclose(da.latitude, C.LAT, atol=1e-6) and np.allclose(da.longitude, C.LON, atol=1e-6)):
        raise ValueError(f"{v}: grid differs from the 0.36° reforecast grid")


def template(date, years):
    import dask.array as dsa
    shape = (len(years), 11, C.STEPS.size, C.LAT.size, C.LON.size)
    coords = {"time": ("time", pd.DatetimeIndex(years).values, {"long_name": "reforecast start (past year)"}),
              "number": np.arange(11),
              "step": ("step", pd.to_timedelta(C.STEPS, "h").values, {"long_name": "lead time"}),
              "latitude": ("latitude", C.LAT, {"units": "degrees_north"}),
              "longitude": ("longitude", C.LON, {"units": "degrees_east"}),
              "init_date": date.to_datetime64()}
    data = {v: (DIMS, dsa.full(shape, np.nan, dtype=np.float32, chunks=tuple(CHUNKS.values())), attrs)
            for v, (attrs, _) in FIELDS.items()}
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update({"ifs_cycle": "50r1", "grid": "0.36° regular, as the 49r1 reforecasts",
                     "source": f"MARS class=od stream=eefh, grid {GRID}, area {AREA}",
                     "written_by": "reforecast_download_atx/src/download.py",
                     "downloaded": C.now()})
    return ds


def write_year(path, i, cf_path, pf_path, hdate):
    """Check, convert and write one past year, one field at a time (peak memory ~2 GB)."""
    cf, pf = read_grib(cf_path), read_grib(pf_path)
    for v in FIELDS:
        da = xr.concat([cf[v], pf[v]], dim="number").sortby("number")
        check_field(v, da, hdate)
        a = as_array(v, da, C.STEPS)
        first = 1 if v in ACCUMULATED else 0
        if np.isnan(a[:, first:]).all(axis=(2, 3)).any():
            raise ValueError(f"{v}: a field is entirely missing")
        lo, hi = FIELDS[v][1]
        n_out = int(((a < lo) | (a > hi)).sum())
        if n_out:
            print(f"    {v}: {n_out} values outside {lo}..{hi} clipped", flush=True)
            a = np.clip(a, lo, hi)
        reg = xr.Dataset({v: (DIMS, a[None])})
        reg.to_zarr(C.url(path), storage_options=C.STORAGE_OPTIONS, region={"time": slice(i, i + 1)},
                    zarr_format=3, consolidated=False)


def process(server, date, n_years, work, keep_grib):
    folder = date.strftime("%Y-%m-%d")
    path = C.hindcast_path(folder)
    if C.done(path):
        return
    years = hdates(date, n_years)
    wdir = os.path.join(work, folder)
    os.makedirs(wdir, exist_ok=True)
    print(f"{folder}: {C.now()} start", flush=True)
    for i, hd in enumerate(years):
        files = {}
        for kind in ("cf", "pf"):
            files[kind] = os.path.join(wdir, f"{kind}_{hd:%Y-%m-%d}.grib")
            retrieve(server, request(date, hd, kind), files[kind])
        if i == 0:                             # the first year is checked inside write_year before data lands
            C.write_zarr(template(date, years), path, encoding=encoding(), compute=False)
        write_year(path, i, files["cf"], files["pf"], hd)
        if not keep_grib:
            for f in files.values():
                os.remove(f)
        print(f"  {folder} {hd:%Y}: written", flush=True)

    chk = C.open_zarr(path)
    t = float(chk.t2m.isel(time=-1, number=10, step=-1).mean())
    sw = float(chk.ssrd.isel(time=-1, number=10, step=slice(1, 5)).mean())
    want = {"time": len(years), "number": 11, "step": C.STEPS.size, "latitude": C.LAT.size,
            "longitude": C.LON.size}
    if dict(chk.sizes) != want or not (180 < t < 320) or not (0 <= sw < 1400):
        raise RuntimeError(f"{folder}: written store failed its check ({dict(chk.sizes)}, "
                           f"t2m mean {t:.1f} K, ssrd mean {sw:.1f} W m-2)")
    C.mark_done(path)
    if not keep_grib:
        shutil.rmtree(wdir, ignore_errors=True)
    print(f"{folder}: {C.now()} complete", flush=True)


def probe(server, date, work):
    r = request(date, hdates(date, 1)[0], "cf")
    r["step"] = "0/6/12"
    os.makedirs(work, exist_ok=True)
    target = os.path.join(work, f"probe_{date:%Y-%m-%d}.grib")
    if os.path.exists(target):
        os.remove(target)
    server.execute(r, target)
    f = read_grib(target)
    t = f["t2m"]
    grid_ok = np.allclose(t.latitude, C.LAT, atol=1e-6) and np.allclose(t.longitude, C.LON, atol=1e-6)
    print(f"probe: start {pd.Timestamp(t.time.values):%Y-%m-%d}, grid {t.latitude.size} x {t.longitude.size}, "
          f"matches the reforecast grid: {grid_ok}")
    for v in FIELDS:
        a = as_array(v, f[v], np.array([0, 6, 12]))
        cells = [f"{np.nanmin(a[0, k]):9.2f}..{np.nanmax(a[0, k]):9.2f}" if np.isfinite(a[0, k]).any()
                 else f"{'NaN (step 0)':>20s}" for k in range(3)]
        print(f"  {v:5s} {FIELDS[v][0]['units']:8s} steps 0/6/12: " + "  ".join(cells))
    if not grid_ok:
        sys.exit("probe: grid mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=FIRST_50R1.strftime("%Y-%m-%d"))
    p.add_argument("--end", default=pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    p.add_argument("--years", type=int, default=N_YEARS, help="past years per date")
    p.add_argument("--work", default=os.path.expanduser("~/reforecast_grib"), help="local GRIB staging")
    p.add_argument("--keep-grib", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list the dates still to do")
    p.add_argument("--probe", metavar="DATE", help="request every field of the control at steps 0-12 and exit")
    a = p.parse_args()

    dates = reforecast_dates(a.start, a.end)
    if any(d < FIRST_50R1 for d in dates):
        sys.exit(f"dates before {FIRST_50R1:%Y-%m-%d} are not 50r1")

    if a.dry_run:
        todo = [d for d in dates if not C.done(C.hindcast_path(d.strftime("%Y-%m-%d")))]
        print(f"{len(dates)} reforecast dates {a.start}..{a.end}; {len(todo)} to download, "
              f"{len(todo) * a.years * 2} MARS requests")
        print(" ".join(d.strftime("%Y-%m-%d") for d in todo))
        return

    from ecmwfapi import ECMWFService
    server = ECMWFService("mars")
    if a.probe:
        probe(server, pd.Timestamp(a.probe), a.work)
        return

    todo = [d for d in dates if not C.done(C.hindcast_path(d.strftime("%Y-%m-%d")))]
    print(f"{C.now()} {len(todo)} of {len(dates)} reforecast dates to download", flush=True)
    failed = []
    for d in todo:
        try:
            process(server, d, a.years, a.work, a.keep_grib)
        except Exception as e:
            print(f"{d:%Y-%m-%d}: FAILED: {e}", flush=True)
            failed.append(d.strftime("%Y-%m-%d"))
    if failed:
        sys.exit(f"{len(failed)} dates failed: {' '.join(failed)}")


if __name__ == "__main__":
    main()
