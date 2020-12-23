#!/bin/bash
if [ ! -d P3DModuleBuilder/.git ]; then
  git clone https://github.com/Eruvae/P3DModuleBuilder
fi

cd P3DModuleBuilder
git fetch
git checkout p3d_voxgrid
python build.py --clean
cd ..
rm p3d_voxgrid.pdb