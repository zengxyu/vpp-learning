@echo off
:: if not exist P3DModuleBuilder/.git (
:: 	git clone https://github.com/Eruvae/P3DModuleBuilder
::)
git submodule update --init

cd pybind11_modules
if not exist build (
    mkdir build
)
cd build
cmake ..
cmake --build . --config Release
cp Release/*.pyd ../..
