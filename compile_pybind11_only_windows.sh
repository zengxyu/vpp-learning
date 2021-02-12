git submodule update --init

cd pybind11_modules
mkdir -p build
cd build
cmake ..
cmake --build . --config Release
cp Release/*.pyd ../..
