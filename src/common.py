"""
Shared settings for the EC46 (IFS 50r1) reforecast download and bias build.

Storage
    Only the bucket atx-weather-consulting (us-east-1) is ever read or written.
    REFCAST_LOCAL_ROOT=/some/dir replaces the bucket by a local directory (tests).

Layout (paths inside the bucket)
    EC_Hindcast/hindcast_50r1/{date}/ec46_{date}.zarr               reforecasts (download.py)
    EC46_Bias_50r1/era5_6h_on_ec46grid/{YYYY-MM}.zarr               ERA5 on the reforecast grid (step1)
    EC46_Bias_50r1/{var}/source_date_means_6h/...                    bias per start date (step2)
    EC46_Bias_50r1_leadsmooth/{var}/daily_hourly_gaussian/...        bias per calendar day (step3),
                                                                     the fields atx_meteo.bias reads
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
warnings.filterwarnings("ignore", message=".*does not have a Zarr V3 specification.*")

BUCKET = "atx-weather-consulting"
LOCAL_ROOT = os.environ.get("REFCAST_LOCAL_ROOT")
if LOCAL_ROOT:
    import fsspec
    fs = fsspec.filesystem("file")
    STORAGE_OPTIONS = None
else:
    import s3fs
    STORAGE_OPTIONS = {"anon": False, "client_kwargs": {"region_name": "us-east-1"}}
    fs = s3fs.S3FileSystem(**STORAGE_OPTIONS)

HINDCAST_ROOT = os.environ.get("REFCAST_HINDCAST_ROOT", f"{BUCKET}/EC_Hindcast/hindcast_50r1")
BIAS_BASE = os.environ.get("REFCAST_BIAS_BASE", f"{BUCKET}/EC46_Bias_50r1")
FINAL_BASE = os.environ.get("REFCAST_FINAL_BASE", f"{BUCKET}/EC46_Bias_50r1_leadsmooth")
ERA5_GRID_DIR = f"{BIAS_BASE}/era5_6h_on_ec46grid"
ERA5_DIRS = [f"{BUCKET}/ERA5_reanalysis_z", f"{BUCKET}/ERA5_reanalysis"]   # searched in this order

# Reforecast grid: the 0.36 grid of the 49r1 reforecasts behind the current bias fields
LAT = np.round(74.76 - 0.36 * np.arange(167), 2)
LON = np.round(-130.0 + 0.36 * np.arange(487), 2)
STEPS = np.arange(0, 1105, 6)                      # lead hours

# Bias variables: reforecast field(s), ERA5 store kind, ERA5 variable, factor to bias units.
# Accumulated variables are the amount over the 6 h ending at the valid time, built
# from six hourly ERA5 values (sum x factor).
VARIABLES = {
    "t2m":  {"era5": ("instant", "VAR_2T", 1.0), "units": "K"},
    "rh":   {"era5": ("rh", "VAR_RH", 100.0), "units": "%"},          # reforecast: from t2m and d2m
    "tcc":  {"era5": ("instant", "TCC", 1.0), "units": "fraction (0-1)"},
    "u10":  {"era5": ("instant", "VAR_10U", 1.0), "units": "m/s"},
    "v10":  {"era5": ("instant", "VAR_10V", 1.0), "units": "m/s"},
    "u100": {"era5": ("instant", "VAR_100U", 1.0), "units": "m/s"},
    "v100": {"era5": ("instant", "VAR_100V", 1.0), "units": "m/s"},
    "ssrd": {"era5": ("ssrd", "SSRD", 1 / 21600.0), "units": "W m-2 (mean over previous 6 h)",
             "accumulated": True},                                     # J m-2 per hour -> 6 h mean W m-2
    "tp":   {"era5": ("mtpr", "MTPR", 3600.0), "units": "mm (previous 6 h)",
             "accumulated": True},                                     # hourly mean kg m-2 s-1 -> mm
}

# ERA5 store names per kind; the first that exists is used
ERA5_NAMES = {
    "instant": ["zarr_store_{y}.zarr"],
    "rh": ["combined_vars_{y}.zarr"],
    "ssrd": ["zarr_store_{y}_accum.zarr", "zarr_store_ssrd_{y}.zarr"],
    "mtpr": ["zarr_store_{y}_mtpr.zarr"],
}

# Calendar weighting and lead smoothing (unchanged from the 49r1 bias fields)
N_DAYS = 365
WINDOW_RADIUS_DAYS = 15
SIGMA_DAYS = 7.5


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------

def check_bucket(path):
    bucket = path.replace("s3://", "").split("/")[0]
    if bucket != BUCKET:
        raise PermissionError(f"refusing to access bucket {bucket!r}; only {BUCKET!r} is allowed")


def url(path):
    """Bucket path -> what xarray and fsspec open (s3:// or the local stand-in)."""
    check_bucket(path)
    return os.path.join(LOCAL_ROOT, path) if LOCAL_ROOT else f"s3://{path}"


def _fs_path(path):
    check_bucket(path)
    return os.path.join(LOCAL_ROOT, path) if LOCAL_ROOT else path


def exists(path):
    return fs.exists(_fs_path(path))


def ls(path):
    p = _fs_path(path)
    if not fs.exists(p):
        return []
    return sorted(x.rstrip("/").split("/")[-1] for x in fs.ls(p))


def done(path):
    """A store finished writing (the marker is written after the data)."""
    return fs.exists(_fs_path(path) + ".done")


def mark_done(path):
    fs.pipe(_fs_path(path) + ".done", b"")


def remove(path):
    for p in (_fs_path(path), _fs_path(path) + ".done"):
        if fs.exists(p):
            fs.rm(p, recursive=True)


def open_zarr(path, **kw):
    return xr.open_zarr(url(path), storage_options=STORAGE_OPTIONS, **kw)


def write_zarr(ds, path, encoding=None, compute=True):
    """
    Write a zarr v3 store, replacing any previous one. With compute=True the
    .done marker follows the data and a failed write is removed; with
    compute=False the caller fills the store and then calls mark_done(path).
    """
    remove(path)
    for v in ds.variables:
        ds[v].encoding = {}
    try:
        out = ds.to_zarr(url(path), storage_options=STORAGE_OPTIONS, mode="w", zarr_format=3,
                         consolidated=False, compute=compute, encoding=encoding)
    except Exception:
        remove(path)
        raise
    if compute:
        mark_done(path)
    return out


def read_json(path, default=None):
    p = _fs_path(path)
    return json.loads(fs.cat(p)) if fs.exists(p) else default


def write_json(obj, path):
    fs.pipe(_fs_path(path), json.dumps(obj, indent=1, sort_keys=True).encode())


# ---------------------------------------------------------------------------
# reforecast folders
# ---------------------------------------------------------------------------

def hindcast_path(folder_date):
    return f"{HINDCAST_ROOT}/{folder_date}/ec46_{folder_date}.zarr"


def hindcast_folders():
    """
    Complete reforecast folders, one row per folder: folder (YYYY-MM-DD), month_day.
    Only the newest folder of each calendar day is kept, so each start date uses
    the most recent past years and no year is counted twice.
    """
    rows = []
    for name in ls(HINDCAST_ROOT):
        try:
            ts = pd.Timestamp(name)
        except ValueError:
            continue
        if (ts.month, ts.day) == (2, 29) or not done(hindcast_path(name)):
            continue
        rows.append({"folder": name, "month_day": ts.strftime("%m-%d")})
    df = pd.DataFrame(rows, columns=["folder", "month_day"])
    return df.sort_values("folder").groupby("month_day").tail(1).sort_values("month_day").reset_index(drop=True)


# ---------------------------------------------------------------------------
# ERA5 sources
# ---------------------------------------------------------------------------

def era5_store(kind, year):
    for d in ERA5_DIRS:
        for pattern in ERA5_NAMES[kind]:
            p = f"{d}/{pattern.format(y=year)}"
            if exists(p):
                return p
    return None


def rh_from_t_td(t_k, td_k):
    """RH (%) from temperature and dew point in K, the Magnus form used by Open-Meteo."""
    a, b = 17.625, 243.04
    t, td = t_k - 273.15, td_k - 273.15
    return (100.0 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))).clip(0, 100)


# ---------------------------------------------------------------------------
# bias paths and calendar
# ---------------------------------------------------------------------------

def source_mean_path(var, md):
    return f"{BIAS_BASE}/{var}/source_date_means_6h/{var}_bias_climo_{md}_6h.zarr"


def source_manifest_path(var):
    return f"{BIAS_BASE}/{var}/source_dates.json"


def final_path(var, md):
    return f"{FINAL_BASE}/{var}/daily_hourly_gaussian/{var}_bias_climo_{md}_hourly_gaussian.zarr"


def final_manifest_path(var):
    return f"{FINAL_BASE}/{var}/days.json"


def dayofyear_to_month_day(doy):
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(doy) - 1)).strftime("%m-%d")


def month_day_to_dayofyear(md):
    return pd.Timestamp(f"2001-{md}").dayofyear


def now():
    return pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def start_cluster():
    """Small local dask cluster; size with REFCAST_WORKERS and REFCAST_WORKER_MEMORY."""
    from dask.distributed import Client, LocalCluster
    n = int(os.environ.get("REFCAST_WORKERS", "2"))
    mem = os.environ.get("REFCAST_WORKER_MEMORY", "6GB")
    return Client(LocalCluster(n_workers=n, threads_per_worker=2, memory_limit=mem,
                               dashboard_address=None))
