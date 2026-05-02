#!/usr/bin/env bash
# Download dreeb/mapper/paga graph files for all SCD-* samples from the cluster.
# Only grabs the files needed for the GNN; skips images, point clouds, and
# redundant graph formats.
#
# Usage:
#   bash download_samples.sh
#
# Replace NETID with your Yale NetID before running.

set -euo pipefail

NETID="zmw6"   # <-- replace with your NetID (e.g. zaw4)
REMOTE_HOST="bouchet.ycrc.yale.edu"
REMOTE_BASE="/nfs/roberts/pi/pi_sk2433/shared/Geomancer_2026_singlecell/algorithm_run_outputs"
LOCAL_DEST="$(dirname "$0")/data"

mkdir -p "$LOCAL_DEST"

rsync -avz --progress \
  --include="SCD-*/"              \
  --include="k25/"                \
  --include="k50/"                \
  --include="k50_ks100/"                \
  --include="dreeb/"              \
  --include="mapper/"             \
  --include="paga/"               \
  --include="nodes.csv"           \
  --include="edges.csv"           \
  --include="h0_finite.csv"       \
  --include="h1_essential.csv"    \
  --include="payload.json"        \
  --include="run_manifest.json"   \
  --include="completion_metadata.json" \
  --exclude="*"                   \
  "${NETID}@${REMOTE_HOST}:${REMOTE_BASE}/" \
  "${LOCAL_DEST}/"
