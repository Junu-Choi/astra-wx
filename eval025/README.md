# Astra: 0.25° evaluation against ERA5 (WeatherBench 2, 2020)

Evaluation code and per-initialisation scores for the 0.25° results of
"Efficient probabilistic weather forecasting at quarter-degree resolution with two-sided transport attention"
(arXiv:2610.35545). The package:

1. scores 0.25° ensemble forecasts against ERA5 with the WeatherBench 2 (WB2) metrics: fair CRPS, ensemble-mean
   squared error and ensemble variance (M − 1 in the denominator), area-weighted with WB2's cell-area latitude weights,
   per initialisation and lead time (`astra_eval025/scorer.py`, `astra_eval025/score_zarr.py`);
2. aggregates over the 732 initialisations of 2020 (00 and 12 UTC): CRPS, unbiased ensemble-mean RMSE
   `sqrt(mean(mse) − mean(sp2)/M)`, spread–skill ratio, and compares them with the WB2 result files of GenCast and
   IFS ENS (`astra_eval025/scorecard.py`);
3. runs the significance tests of the paper's Methods: circular block bootstrap over the time-ordered
   initialisations, blocks of 14, 10,000 resamples, seed 20201231; paired tests where per-initialisation reference
   scores exist (IFS ENS at 24 h, 8 variables), otherwise the widened fixed-reference test (both half-widths × κ,
   κ = median ratio of paired to fixed-reference interval widths over the 16 paired tests, at least 1); in the
   scorecard, `significant` is the paper's verdict (the paired test where paired, otherwise the widened
   fixed-reference test), and `p_unwidened` and `significant_unwidened` are before widening;
4. reproduces the paper's 0.25° score-based numbers from the stored scores (`astra_eval025/reproduce.py`).

## Requirements

Python 3.12 with the versions pinned in `requirements.txt`: numpy, scipy, xarray (netCDF via the scipy engine), jax
(CPU is enough; only for `scorer.py`, `score_zarr.py` and `weights.py`), pytest. Scoring raw forecast stores also
needs zarr, and gcsfs (with aiohttp) for remote stores, pinned in `requirements-forecast.txt`.

## Data (`data/`)

| file | contents |
|---|---|
| `astra_2020_025_per_init_scores.npz` | Astra, 50 members, 732 initialisations × 30 lead times (12–360 h) × 24 rows: per-initialisation global fair CRPS (`crps`), squared error of the ensemble mean (`mse`) and ensemble variance (`sp2`), physical units; `init_times` in hours since 1970-01-01; NaN = undefined (tp24h at 12 h) |
| `ifs_ens_2020_24h_per_init_scores.npz` | IFS ENS at 24 h, per initialisation, 8 variables (z, t at 500/700/850 hPa, t2m, msl): fair CRPS, unbiased squared error of the mean, squared error, variance; computed with WeatherBench-X from the WB2 IFS ENS forecasts (they reproduce the published scores to 8 × 10⁻⁸) |
| `wb2/*_1440x721_2020.nc` | WB2 result files (`gs://weatherbench2/benchmark_results/`); `scripts/fetch_wb2_results.sh` downloads them and checks their sha256 |
| `expected/` | reference outputs on these inputs, and a synthetic scorer fixture (inputs and outputs), for the tests |

The WB2 result files are redistributed from WeatherBench 2 (`gs://weatherbench2/benchmark_results/`; Rasp et al.,
J. Adv. Model. Earth Syst. 16, e2023MS004019, 2024).

Rows: z, t, q, u, v and wind speed (ws) at 500, 700, 850 hPa; msl, t2m, u10, v10, ws10, tp24h. The paper's 20
targets are z, t, q, u and v at 500, 700 and 850 hPa, and t2m, msl, u10, v10 and ws10.

## Astra's 0.25° results in the WeatherBench 2 layout (`results/`)

`results/astra_vs_era5_1440x721_2020.nc` holds Astra's 2020 scores at 0.25° against ERA5 in the layout of WB2's
`*_vs_era5_1440x721_2020.nc` result files. `scripts/make_scorecard.py` builds it from
`data/astra_2020_025_per_init_scores.npz` (the command under "Score a forecast store", applied to that file, writes the
same file byte for byte).

- Region `global`; `lead_time` 12–360 h every 12 h; `level` 500, 700 and 850 hPa. Only the global region is available,
  because the per-initialisation scores are global means.
- Variables: `geopotential`, `temperature`, `specific_humidity`, `u_component_of_wind`, `v_component_of_wind` and
  `wind_speed` on the three levels; `2m_temperature`, `mean_sea_level_pressure`, `10m_u_component_of_wind`,
  `10m_v_component_of_wind`, `10m_wind_speed` and `total_precipitation_24hr`.
- Metrics, with the names and definitions of WB2's GenCast and IFS ENS files: `crps` (fair), `mean_rmse` (RMSE of the
  50-member ensemble mean), `unbiased_mean_rmse`, `spread_skill` and `unbiased_spread_skill` (each an RMSE divided by
  the spread, as in those files); also `spread`.

`results/ratio_1p5_vs_025.md` compares these scores with the WB2 evaluation of the same forecasts at 1.5°
(`hf://datasets/JunuChoi/Astra-WB2-2020-240x121`): the 1.5° / 0.25° ratios of CRPS and ensemble-mean RMSE agree with
GenCast's own ratios to 0.30% and 0.16% (medians over 8 variables × 5 lead times).

## Reproduce the paper's 0.25° score-based numbers

```bash
python scripts/reproduce_paper_025.py          # paper value vs reproduced value, exit status 0 if all match
python -m pytest -q tests                      # under a minute on CPU
```

## Score a forecast store

```bash
python scripts/score_forecasts.py --forecast FC.zarr \
    --era5 https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr \
    --out scores.npz --ens-dim number
python scripts/make_scorecard.py scores.npz --wb2-dir data/wb2 --out out/ \
    --ens-states data/ifs_ens_2020_24h_per_init_scores.npz
```

The forecast store needs the variables of `astra_eval025/score_zarr.CORE` on the 721 × 1440 grid and
`total_precipitation_6hr` every 6 h (tp24h is the sum of the four 6 h accumulations ending at the lead time); wind
speeds are computed from each member's own components. `out/scorecard.json` holds every comparison (ratio, 95 %
interval, widened interval, flags); `out/astra_vs_era5_1440x721_2020.nc` holds the year scores in the layout of the
WB2 result files.

## Check the scorer against WeatherBench 2

```bash
pip install git+https://github.com/google-research/weatherbench2   # only weatherbench2.metrics and .regions are used
python scripts/check_scorer_against_wb2.py                        # exit status 0 when the scores agree
```

The script scores GenCast's public 0.25° ensemble (read anonymously from WB2's bucket) against WB2's 0.25° ERA5 twice,
on the same arrays: with this package's scorer and with `weatherbench2.metrics` (`CRPS`, `EnsembleMeanMSE`,
`EnsembleVariance`). On the default sample (initialisation 2020-01-02T00; leads 24 and 120 h; z500, t850, t2m, msl and
tp24h; 56 and 50 members) the CRPS, ensemble-mean MSE and ensemble variance agree with weatherbench2 0.2.1 to within
2.3 × 10⁻⁶ relative. The run reads about 2.6 GB and takes about 16 min on 4 CPU threads.

## Definitions

* Fair CRPS: `(1/M) Σ_i ⟨|x_i − y|⟩ − 1/(M(M−1)) Σ_{i<j} ⟨|x_i − x_j|⟩`, ⟨·⟩ the area-weighted mean over grid points.
* Unbiased ensemble-mean RMSE: `sqrt(mean_i mse_i − mean_i sp2_i / M)` (WB2 `unbiased_mean_rmse`).
* Spread–skill ratio: `sqrt(mean_i sp2_i) / unbiased RMSE`; for GenCast and IFS ENS, the reciprocal of WB2's
  `unbiased_spread_skill`.
* Every score is averaged over grid points and initialisations before square roots and ratios are taken.

## Licence

Apache License 2.0 (`LICENSE`).
