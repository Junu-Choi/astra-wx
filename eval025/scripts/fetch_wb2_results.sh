#!/bin/bash
# Downloads the WeatherBench 2 result files used by the 0.25 deg comparison (public, about 13 MB) to a temporary
# directory, checks their sha256 against those of the files used for the paper and moves them into data/wb2/
# (on a mismatch: non-zero exit status, nothing moved).
set -euo pipefail
DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/wb2}"
BASE=https://storage.googleapis.com/weatherbench2/benchmark_results
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -f "$TMP"/*_1440x721_2020.nc; rmdir "$TMP"' EXIT
for f in gencast_vs_era5 ens_vs_era5 ens_vs_hres_t0 hres_vs_era5 graphcast_vs_era5 pangu_vs_era5 fuxi_vs_era5 \
         excarta_vs_era5 baguan_vs_era5; do
  curl -fsSL -o "$TMP/${f}_1440x721_2020.nc" "$BASE/${f}_1440x721_2020.nc"
done
(cd "$TMP" && sha256sum -c -) <<'SUMS'
9af592f59cf72a335d04268b0570e1878028df27c67a34a19365076636a4ab5f  gencast_vs_era5_1440x721_2020.nc
60c2919a5a563ffc6805313a8ad5b7a4bba3e838ffb662230a3958bb61bc84af  ens_vs_era5_1440x721_2020.nc
7c2035805258c38ec3f7c0fe28f92308a95d67173f5cd5748f3701b3bc590117  ens_vs_hres_t0_1440x721_2020.nc
0dbae230b826650668168c50f3cab7140ee1370ae7a3f6aaf48ec944ec22cdc6  hres_vs_era5_1440x721_2020.nc
8373adf4009a87384d6b4baa19d226a2c88d8bb2260c7424cba225cbc578152c  graphcast_vs_era5_1440x721_2020.nc
79ae1be6b497492352a730b4d0f11c7270ade14c8a6fa2a813d84ba3bdeee07e  pangu_vs_era5_1440x721_2020.nc
52fc4c306e4bcb23c14bb110710fe40dbb54f134a98bd59318f6aed6855c9114  fuxi_vs_era5_1440x721_2020.nc
8fd88be9959a4ada46843382b395cf1ab4acabe6ac65b9d628fd670bb91f25e1  excarta_vs_era5_1440x721_2020.nc
d81f543231e830b99770a3bc0faf3b7e13687a2e2d0baa4a33d6615c7daa7088  baguan_vs_era5_1440x721_2020.nc
SUMS
mv "$TMP"/*_1440x721_2020.nc "$DEST"/
