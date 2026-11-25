"""
End-to-end test of the bias steps on synthetic data, without S3 or MARS.

Builds small reforecast stores (the download format: 16-bit packing, 11 members,
185 leads) whose member mean is ERA5 plus a known bias, and small ERA5 stores in
the bucket's naming, in a temporary directory standing in for the bucket. Runs
steps 1-3 and checks the bias comes out as set. Then adds a reforecast date and
checks that only that start date and the days within ±15 days of it are rebuilt.

    conda activate reforecast
    python tests/test_pipeline.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

ROOT = tempfile.mkdtemp(prefix="refcast_test_")
os.environ["REFCAST_LOCAL_ROOT"] = ROOT
os.environ.setdefault("REFCAST_WORKERS", "1")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import common as C     # noqa: E402
import download as D   # noqa: E402

# small piece of the real grid
C.LAT, C.LON = C.LAT[60:63], C.LON[100:104]
BIAS = {"t2m": 1.5, "tcc": 0.05, "u10": 0.3, "v10": -0.2, "u100": 0.4, "v100": -0.3, "ssrd": 10.0, "tp": 0.5}
E_LAT, E_LON = np.arange(72.0, 19.0, -2.0), np.arange(-132.0, 49.0, 2.0)   # coarse ERA5 stand-in


def era5_value(var, t, lat, lon):
    """Synthetic ERA5 (bias units), linear in lat/lon so bilinear interpolation is exact."""
    h = np.sin(2 * np.pi * t.hour / 24.0)
    return {"t2m": 280 + 0.1 * lat + 0.05 * lon + 3 * h, "tcc": 0.5 + 0.001 * lat, "u10": 2 + 0.01 * lat,
            "v10": 1 - 0.01 * lon, "u100": 4 + 0.01 * lat, "v100": 3 - 0.01 * lon,
            "rh": 60 + 0.1 * lat + 5 * h, "ssrd": 200.0 + 0 * lat, "tp": 2.16 + 0 * lat}[var]


def write_era5(year):
    hours = pd.date_range(f"{year}-04-25", f"{year}-08-31 23:00", freq="h")
    la, lo = np.meshgrid(E_LAT, E_LON, indexing="ij")
    def field(var, scale):
        return (("time", "latitude", "longitude"),
                np.stack([era5_value(var, t, la, lo) / scale for t in hours]).astype(np.float32))
    coords = {"time": hours, "latitude": E_LAT, "longitude": E_LON}
    base = f"{C.BUCKET}/ERA5_reanalysis"
    stores = {
        f"zarr_store_{year}.zarr": {"VAR_2T": field("t2m", 1), "TCC": field("tcc", 1), "VAR_10U": field("u10", 1),
                                    "VAR_10V": field("v10", 1), "VAR_100U": field("u100", 1),
                                    "VAR_100V": field("v100", 1)},
        f"combined_vars_{year}.zarr": {"VAR_RH": field("rh", 100.0)},
        # hourly J m-2 whose 6 h sum / 21600 is the 6 h mean; hourly mean rate whose 6 h sum x 3600 is mm
        f"zarr_store_{year}_accum.zarr": {"SSRD": field("ssrd", 1 / 3600.0)},
        f"zarr_store_{year}_mtpr.zarr": {"MTPR": field("tp", 6 * 3600.0)},
    }
    for name, data in stores.items():
        xr.Dataset(data, coords=coords).to_zarr(C.url(f"{base}/{name}"), mode="w", zarr_format=2)


def write_reforecast(date, n_years=2):
    date = pd.Timestamp(date)
    years = D.hdates(date, n_years)
    path = C.hindcast_path(date.strftime("%Y-%m-%d"))
    C.write_zarr(D.template(date, years), path, encoding=D.encoding(), compute=False)
    la, lo = np.meshgrid(C.LAT, C.LON, indexing="ij")
    members = (np.arange(11) - 5) * 0.01                       # mean zero over members
    for i, hd in enumerate(years):
        valid = hd + pd.to_timedelta(C.STEPS, "h")
        data = {}
        for v in D.FIELDS:
            base = v if v != "d2m" else "t2m"
            x = np.stack([era5_value(base, t, la, lo) for t in valid]) + BIAS.get(base, 0.0)
            if v == "d2m":
                x = x - 5.0
            if v in D.ACCUMULATED:
                x[0] = np.nan
            a = x[None] + members[:, None, None, None]
            data[v] = (D.DIMS, a[None].astype(np.float32))
        xr.Dataset(data).to_zarr(C.url(path), region={"time": slice(i, i + 1)}, zarr_format=3, consolidated=False)
    C.mark_done(path)


def run(script):
    r = subprocess.run([sys.executable, script], cwd=SRC, env=os.environ, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr[-3000:])
        raise SystemExit(f"{script} failed")
    return r.stdout


def expected_rh_bias(folder):
    """Mean over years and members of RH(reforecast t2m, d2m) - ERA5 RH, per lead, at one point."""
    hind = C.open_zarr(C.hindcast_path(folder))
    out = []
    for i, init in enumerate(pd.DatetimeIndex(hind.time.values)):
        fc = C.rh_from_t_td(hind.t2m.isel(time=i, latitude=0, longitude=0),
                            hind.d2m.isel(time=i, latitude=0, longitude=0)).mean("number").values
        obs = np.array([era5_value("rh", init + pd.Timedelta(hours=int(s)), C.LAT[0], C.LON[0]) for s in C.STEPS])
        out.append(fc - obs)
    return np.mean(out, axis=0)


def main():
    print(f"test bucket: {ROOT}")
    for y in (2024, 2025):
        write_era5(y)
    for d in ("2026-05-21", "2026-05-23"):
        write_reforecast(d)
    run("step1_era5.py")
    run("step2_bias.py")
    run("step3_bias.py")

    ok = True
    for var in C.VARIABLES:
        b2 = C.open_zarr(C.source_mean_path(var, "05-21"))[f"{var}_bias"].load()
        b3 = C.open_zarr(C.final_path(var, "05-22"))[f"{var}_bias"].load()
        lead0_nan = C.VARIABLES[var].get("accumulated", False)
        if var == "rh":
            # RH bias varies with place and lead: check one point against a direct calculation, and
            # day 05-22 (equal weights on 05-21 and 05-23) against their mean up to day 5 (no lead smoothing)
            err2 = float(np.nanmax(np.abs(b2.isel(latitude=0, longitude=0) - expected_rh_bias("2026-05-21"))))
            b2b = C.open_zarr(C.source_mean_path(var, "05-23"))[f"{var}_bias"].load()
            mean2 = (0.5 * (b2 + b2b)).sel(lead_hour=slice(0, 120))
            err3 = float(np.nanmax(np.abs(b3.sel(lead_hour=mean2.lead_hour) - mean2)))
        else:
            err2 = float(np.nanmax(np.abs(b2 - BIAS[var])))
            err3 = float(np.nanmax(np.abs(b3 - BIAS[var])))
        nan_ok = bool(np.isnan(b2.isel(lead_hour=0)).all()) == lead0_nan and not np.isnan(b2.isel(lead_hour=slice(1, None))).any()
        tol = 0.03 if var in ("ssrd", "rh") else 0.01
        good = err2 < tol and err3 < tol and nan_ok
        ok &= good
        print(f"  {var:5s} step2 max error {err2:.4f}, step3 max error {err3:.4f}, lead-0 NaN as expected: {nan_ok}"
              f"  {'PASS' if good else 'FAIL'}")

    n_days = len(C.read_json(C.final_manifest_path("t2m")))
    print(f"  calendar days built from 05-21 and 05-23: {n_days} (expected 33)")
    ok &= n_days == 33

    write_reforecast("2026-05-25")
    out2 = run("step2_bias.py")
    out3 = run("step3_bias.py")
    rebuilt = [l for l in out2.splitlines() if l.strip().startswith("05-")]
    days3 = [int(l.split(",")[1].split()[0]) for l in out3.splitlines() if l.startswith("step3 t2m:") and "days to build" in l]
    print(f"  after adding 05-25: start dates rebuilt {[l.split(':')[0].strip() for l in rebuilt]}, "
          f"t2m days rebuilt {days3} (expected ['05-25'] and [31])")
    ok &= [l.split(":")[0].strip() for l in rebuilt] == ["05-25"] and days3 == [31]

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
