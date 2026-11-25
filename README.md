# reforecast_download_atx

Downloads the ECMWF EC46 reforecasts (IFS Cycle 50r1) from MARS and builds the lead-time bias fields used by `atx-meteo raw --bias-correct`. All data live in `s3://atx-weather-consulting` (us-east-1).

## What it does

`ops/run.sh` runs four steps. Each step skips work that is already finished. A run can stop at any time and the next run continues from there.

| Step | Script | Writes (under `s3://atx-weather-consulting/`) |
|---|---|---|
| 1. Download reforecasts from MARS | `src/download.py` | `EC_Hindcast/hindcast_50r1/{date}/ec46_{date}.zarr` |
| 2. ERA5 on the reforecast grid | `src/step1_era5.py` | `EC46_Bias_50r1/era5_6h_on_ec46grid/{YYYY-MM}.zarr` |
| 3. Bias per reforecast start date | `src/step2_bias.py` | `EC46_Bias_50r1/{var}/source_date_means_6h/` |
| 4. Bias per calendar day, smoothed in lead | `src/step3_bias.py` | `EC46_Bias_50r1_leadsmooth/{var}/daily_hourly_gaussian/` |

ECMWF publishes a reforecast for every odd day of the month, 10 past years each, starting 21 May 2026. Step 1 downloads every date not yet on S3. Steps 2–4 then update only the bias fields that the new dates change. The bias covers every variable that ERA5 has: 2 m temperature, relative humidity, cloud cover, 10 m and 100 m wind, solar radiation and precipitation.

## Setup (once, on the instance)

1. Clone the repository and create the environment:
   ```bash
   git clone https://github.com/WillyChap/reforecast_download_atx.git
   cd reforecast_download_atx
   conda env create -f environment.yml
   conda activate reforecast
   ```
2. ECMWF key for MARS. Log in at https://api.ecmwf.int/v1/key/ with the company ECMWF account, copy the three lines shown (`url`, `key`, `email`) into `~/.ecmwfapirc`, then:
   ```bash
   chmod 600 ~/.ecmwfapirc
   ```
3. S3 access. The instance needs read and write access to `s3://atx-weather-consulting`. **Question!!!:** how to give it credentials that do not expire (for example an IAM instance role). Check with `aws s3 ls s3://atx-weather-consulting/`. Do you have a good solution for this? Like a permanent credential?? I have limited aws s3 experience. 

   **Problem with `aws sso login`:** an SSO login expires after a few hours, and every S3 read or write then fails with `Token has expired and refresh failed`. This happened repeatedly while building this repo. With SSO, `ops/run.sh` pauses and checks every 5 minutes until someone runs `aws sso login` again, so nothing is lost, but the backfill stalls each time and cron runs stall until a login. Credentials that do not expire avoid this. **We should prioritize the newest hindcast/reforecasts so we can bias correct the real-time forecast ASAP.**
4. Test both:
   ```bash
   cd src
   python download.py --probe 2026-05-21   # MARS: prints nine fields on a 167 x 487 grid
   python download.py --dry-run            # S3: lists the dates still to download
   ```

## Run

First run (the backfill, about 90 dates; expect several days). Use `tmux` so it survives a disconnect:
```bash
tmux new -s reforecast
bash ops/run.sh
```
Detach with `Ctrl-b d`, reattach with `tmux attach -t reforecast`.

Afterwards, every two weeks by cron (`crontab -e`):
```
0 6 1,15 * * /bin/bash $HOME/reforecast_download_atx/ops/run.sh
```
A lock file stops a second run from starting while one is active.

## Check progress

```bash
bash ops/status.sh          # dates downloaded, bias fields built, last log lines
```
Logs are in `logs/run_{date}.log`. A run that ends with an error says which step and date failed; running `ops/run.sh` again retries it.

## Notes

- **Instance.** 2 vCPU and 16 GB of memory are enough (for example `r6i.large`), with 50 GB of disk for GRIB staging. Run it in us-east-1 so S3 transfer is free.
- **Size.** Each date is stored as 16-bit integers, about 10–25 GB. About 90 dates now and 180 by May 2027 means roughly 1–4 TB on S3.
- **ERA5.** The bias compares each reforecast with ERA5, which is in the bucket up to 1 January 2026. Reforecast years after that are left out of the bias until ERA5 for 2026 is added under `ERA5_reanalysis_z/`.
- **Testing without S3.** `REFCAST_LOCAL_ROOT=/some/dir` makes every script use a local directory instead of the bucket.
