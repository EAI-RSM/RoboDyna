echo "Installing the necessary packages ..."
pip install -r script/requirements.txt

echo "Installing pytorch3d ..."
# cd third_party/pytorch3d_simplified
# pip install -e .
# cd ../..
# NOTE: importing torch.utils.cpp_extension (done by this build at the metadata
# stage) crashes nondeterministically on some machines (CPython sre/heap bug hit
# by torch's hipify regex). The crash is fail-stop, so retrying is safe.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation && break
    echo "pytorch3d build attempt $attempt failed (flaky cpp_extension crash), retrying ..."
done
python -c "import pytorch3d" || echo "ERROR: pytorch3d still not importable after retries"

echo "Adjusting code in sapien/wrapper/urdf_loader.py ..."
# location of sapien, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/sapien"
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
# Adjust some code in wrapper/urdf_loader.py
URDF_LOADER=$SAPIEN_LOCATION/wrapper/urdf_loader.py
# ----------- before -----------
# 667         with open(urdf_file, "r") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + "srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
# ----------- after  -----------
# 667         with open(urdf_file, "r", encoding="utf-8") as f:
# 668             urdf_string = f.read()
# 669 
# 670         if srdf_file is None:
# 671             srdf_file = urdf_file[:-4] + ".srdf"
# 672         if os.path.isfile(srdf_file):
# 673             with open(srdf_file, "r", encoding="utf-8") as f:
# 674                 self.ignore_pairs = self.parse_srdf(f.read())
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' $URDF_LOADER


echo "Adjusting code in mplib/planner.py ..."
# location of mplib, like "~/.conda/envs/RoboTwin/lib/python3.10/site-packages/mplib"
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib

# Adjust some code in planner.py
# ----------- before -----------
# 807             if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
# ----------- after  ----------- 
# 807             if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
# 808                 return {"status": "screw plan failed"}
PLANNER=$MPLIB_LOCATION/planner.py
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

echo "Installing Curobo ..."
cd envs
if [ ! -d curobo ]; then
    git clone https://github.com/NVlabs/curobo.git
fi
cd curobo
# Pin v0.7.8: RoboDyna targets the classic curobo API (curobo.types.math, curobo.wrap.reacher, ...).
# curobo HEAD (>= 0.8.0) restructured to curobo/_src and removed these modules.
git checkout v0.7.8
# Same flaky cpp_extension crash as pytorch3d above; retry until the build lands.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    pip install -e . --no-build-isolation && break
    echo "curobo build attempt $attempt failed (flaky cpp_extension crash), retrying ..."
done
python -c "import curobo" || echo "ERROR: curobo still not importable after retries"
cd ../..

echo "Re-pinning packages that curobo's install drags to incompatible versions ..."
# warp-lang > 1.4.x does not auto-expose warp.torch, which curobo 0.7.8 uses
# (geom/sdf/world_mesh.py: wp.torch.device_from_torch). scipy must stay at the
# requirements.txt pin; curobo's editable install silently upgrades it.
pip install warp-lang==1.4.2 scipy==1.10.1

echo "Installation basic environment complete!"
echo -e "You need to:"
echo -e "    1. \033[34m\033[1m(Important!)\033[0m Download assets from huggingface."
echo -e "    2. Install requirements for running baselines. (Optional)"
