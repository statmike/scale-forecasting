#!/bin/bash
# Customization script for the custom GPU Dataproc VM image (run once, at image-build time, on the
# temporary builder VM that `generate_custom_image` boots from the 2.2-debian12 base). It bakes the
# NVIDIA driver into the disk image so a GPU cluster booting from that image already has the driver
# present — no per-cluster-create compile, which is what races (and loses to) Dataproc's cluster
# create window.
#
# It runs the SAME stock Dataproc GPU-driver init action we would otherwise run at create time, so
# the baked driver is exactly the one Dataproc supports for this image line. The cuDNN + NCCL source
# builds are skipped (the deep-learning wheel bundles its own): `generate_custom_image` passes
# `cudnn-version=` as builder-VM metadata, which the init action reads as empty and treats as "skip".
#
# Requirements at build time: the builder VM needs egress to the NVIDIA driver mirrors (the driver is
# fetched, not bundled). `generate_custom_image` gives the builder VM an external IP by default; if
# the project's org policy blocks external IPs, run the build on a subnet with Cloud NAT instead.
set -euo pipefail

readonly INIT_ACTION="gs://goog-dataproc-initialization-actions-us-central1/gpu/install_gpu_driver.sh"
readonly LOCAL="/tmp/install_gpu_driver.sh"

echo "custom-gpu-image: fetching stock GPU-driver init action"
gsutil cp "${INIT_ACTION}" "${LOCAL}"
chmod +x "${LOCAL}"

echo "custom-gpu-image: installing NVIDIA driver (cuDNN/NCCL skipped via cudnn-version= metadata)"
"${LOCAL}"

echo "custom-gpu-image: driver install complete"
