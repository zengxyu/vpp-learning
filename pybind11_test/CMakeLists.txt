# CMakeList.txt : CMake project for p3d_cpp, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("p3d_cpp")

# Add source to this project's executable.
include_directories("C:/Panda3D-1.10.7-x64/include")
link_directories("C:/Panda3D-1.10.7-x64/lib")
add_subdirectory(pybind11)
pybind11_add_module(p3d_voxgrid p3d_voxgrid.cpp)
target_link_libraries(p3d_voxgrid PRIVATE
	libp3assimp
	libp3converter
	libp3cvscopy
	libp3daeegg
	libp3direct
	libp3dtool
	libp3dtoolconfig
	libp3dxf
	libp3dxfegg
	libp3eggbase
	libp3eggcharbase
	libp3ffmpeg
	libp3flt
	libp3fltegg
	libp3fmod_audio
	libp3framework
	libp3imagebase
	libp3interrogatedb
	libp3lwo
	libp3lwoegg
	libp3objegg
	libp3openal_audio
	libp3palettizer
	libp3pandatoolbase
	libp3progbase
	libp3pstatserver
	libp3ptloader
	libp3pystub
	libp3tinydisplay
	libp3vision
	libp3vrml
	libp3vrmlegg
	libp3vrpn
	libp3windisplay
	libp3xfile
	libp3xfileegg
	libpanda
	libpandaai
	libpandabullet
	libpandaegg
	libpandaexpress
	libpandafx
	libpandagl
	libpandaode
	libpandaphysics
	libpandaskel
)

install(TARGETS p3d_voxgrid DESTINATION .)

# TODO: Add tests and install targets if needed.
