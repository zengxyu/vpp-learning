@echo off
if not exist P3DModuleBuilder/.git (
	git clone https://github.com/Eruvae/P3DModuleBuilder
)

cd P3DModuleBuilder
git fetch
git checkout p3d_voxgrid
python build.py --clean
cd ..
rm p3d_voxgrid.pdb