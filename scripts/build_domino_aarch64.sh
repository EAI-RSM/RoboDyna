#!/usr/bin/env bash
# Reproducible DOMINO deploy on NVIDIA GB10 / aarch64 / CUDA 13 (DGX Spark-class box).
# DOMINO = RoboTwin 2.0 + dynamic-scene extensions. Its stock script/_install.sh FAILS here
# (pins torch==2.4.1, sapien==3.0.0b1, mplib==0.2.1 — none have working aarch64 builds).
# This is the working recipe (adapted from jack's RoboTwin aarch64 build). Benchmark/data-
# collection only; PUMA/VLA policy stack is intentionally skipped.
set -eo pipefail

CONDA=/shared_work/markhsp/miniforge3
ENV=/shared_work/markhsp/envs/domino
REPO=/shared_work/markhsp/DOMINO
PY=$ENV/bin/python
export PIP_CACHE_DIR=/shared_work/markhsp/.pip_cache
export HF_HOME=/shared_work/markhsp/.hf_cache

source $CONDA/etc/profile.d/conda.sh

echo "== 1/9 env (py3.10) =="
[ -d "$ENV" ] || conda create -p "$ENV" python=3.10 -y
conda activate "$ENV"

echo "== 2/9 torch 2.9.1 cu130 (GB10 sm_121 runs sm_120 cubin) + numpy<2 + ninja =="
$PY -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu130
$PY -m pip install 'numpy<2' ninja

echo "== 3/9 SAPIEN 3.0.3 aarch64 wheel (the 3.0.0b1 pin has no arm64 build) =="
WHL=/shared_work/markhsp/wheels/sapien-3.0.3-cp310-cp310-linux_aarch64.whl
mkdir -p /shared_work/markhsp/wheels
[ -f "$WHL" ] || cp /shared_work/jack/wheels/sapien-3.0.3-cp310-cp310-linux_aarch64.whl "$WHL" 2>/dev/null \
  || curl -fsSL -o "$WHL" https://github.com/haosulab/SAPIEN/releases/download/3.0.3/sapien-3.0.3-cp310-cp310-linux_aarch64.whl
$PY -m pip install "$WHL"

echo "== 4/9 glibc-2.39 librt ABI fix (symlink vendored librt -> system librt.so.1) =="
SLIBS="$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")/sapien.libs"
for V in $(ls "$SLIBS" 2>/dev/null | grep -E '^librt-[0-9].*\.so$'); do
  ln -sf /lib/aarch64-linux-gnu/librt.so.1 "$SLIBS/$V"
done

echo "== 5/9 open3d-before-sapien fix (sitecustomize preloads open3d; else sapien renderer segfaults) =="
SP="$($PY -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")"
cat > "$SP/sitecustomize.py" <<'PY'
try:
    import open3d  # must load before sapien on aarch64 (clashing bundled libs)
except Exception:
    pass
PY

echo "== 6/9 curobo v0.7.7 from source (arch 12.0) =="
export CUDA_HOME=/usr/local/cuda-13.0 PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=12.0 FORCE_CUDA=1 MAX_JOBS=8 NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++'
CUROBO=$REPO/envs/curobo
[ -d "$CUROBO/.git" ] || git clone --depth 1 --branch v0.7.7 https://github.com/NVlabs/curobo.git "$CUROBO"
# clear_cache patch (avoids cache["mesh"] KeyError on update_world)
$PY - "$CUROBO/src/curobo/geom/sdf/world_mesh.py" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
old='        if self._env_mesh_names is not None:\n            self._env_mesh_names = [\n                [None for _ in range(self.cache["mesh"])] for _ in range(self.n_envs)\n            ]'
new='        if self._env_mesh_names is not None:\n            for i in range(len(self._env_mesh_names)):\n                for j in range(len(self._env_mesh_names[i])):\n                    self._env_mesh_names[i][j] = None'
open(p,'w').write(s.replace(old,new))
PY
$PY -m pip install setuptools setuptools_scm tomli wheel 'warp-lang==1.12.1'
$PY -m pip install --no-build-isolation -e "$CUROBO"

echo "== 7/9 RoboTwin/DOMINO runtime deps =="
$PY -m pip install transforms3d 'gymnasium==0.29.1' 'open3d==0.18.0' imageio-ffmpeg h5py \
  termcolor av matplotlib 'pyglet<2' toppra pydantic 'huggingface_hub==0.25.0' zarr openai moviepy wandb
FF=$($PY -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())") && ln -sf "$FF" "$ENV/bin/ffmpeg"

echo "== 8/9 SOURCE PATCHES (mplib has no aarch64 build; guarded out — CuroboPlanner does TOPP via toppra) =="
echo "   already applied in-repo: envs/robot/planner.py (try/except mplib, _HAS_MPLIB),"
echo "                            envs/robot/robot.py  (import _HAS_MPLIB; gate MplibPlanner)"

echo "== 9/9 assets (embodiments 0.2G + objects 3.7G; background_texture 11G only for demo_random) =="
cd "$REPO/assets"
$PY -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TianxingChen/RoboTwin2.0', allow_patterns=['embodiments.zip','objects.zip'], local_dir='.', repo_type='dataset', resume_download=True)"
unzip -q -o embodiments.zip; unzip -q -o objects.zip
cd "$REPO"; $PY script/update_embodiment_config_path.py

echo "== DONE. Runtime env REQUIRED before collection: =="
echo "   export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY"
echo "   conda activate $ENV && bash scripts/collect_data.sh adjust_bottle demo_clean_dynamic 0"
