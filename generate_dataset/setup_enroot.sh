#!/bin/bash
# Build the 3 os-std sandbox images as Enroot squashfs files (no Docker, no root).
# Run inside a CPU allocation:  srun -p lrz-cpu --qos=cpu -c 4 --mem=16G -t 01:00:00 --pty bash
set -euo pipefail
IMG_DIR="${ENROOT_SANDBOX_DIR:-$HOME/enroot_images}"
DF_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/os_interaction/res/dockerfiles"
mkdir -p "$IMG_DIR"; cd "$IMG_DIR"

[ -f ubuntu_base.sqsh ] || enroot import -o ubuntu_base.sqsh docker://ubuntu

for name in default packages ubuntu; do
  [ -f "$name.sqsh" ] && continue
  enroot remove -f "build_$name" 2>/dev/null || true
  enroot create -n "build_$name" ubuntu_base.sqsh
  # replay the Dockerfile's RUN lines
  { echo 'echo "APT::Sandbox::User \"root\";" > /etc/apt/apt.conf.d/99sandbox'
    grep '^RUN ' "$DF_DIR/$name" | sed 's/^RUN //'; } > "/tmp/build_$name.sh"
  enroot start --rw --root --mount "/tmp/build_$name.sh:/build.sh" "build_$name" bash /build.sh
  enroot export -o "$name.sqsh" "build_$name"
  enroot remove -f "build_$name"
done
ls -lh "$IMG_DIR"
