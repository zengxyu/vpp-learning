
it submodule update --init

cd pybind11_modules
mkdir -p build
cd build
cmake ..
make
cp *.so ../..
