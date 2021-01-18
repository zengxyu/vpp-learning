// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pandabase.h>
#include <geom.h>
#include <geomVertexData.h>
#include <geomVertexWriter.h>
#include <geomTriangles.h>
#include <geomPrimitive.h>
#include <iostream>
#include <py_panda.h>
#include <dtoolbase.h>
#include <dtoolsymbols.h>

namespace py = pybind11;

//from panda3d.core import Point3, GeomNode, Geom, GeomPrimitive, GeomVertexData, GeomTriangles, GeomTristrips, GeomVertexWriter, GeomVertexFormat, DirectionalLight, AmbientLight

/*Geom createCube(const Point3 &min, const Point3 &max, visible_faces = [1, 1, 1, 1, 1, 1]):
    GeomVertexData vertices("vertices", GeomVertexFormat.get_v3c4(), Geom.UHStatic)
    vertices.setNumRows(8)
    vertex = GeomVertexWriter(vertices, 'vertex')
    color = GeomVertexWriter(vertices, 'color')
    vertex.addData3(min[0], min[1], min[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(max[0], min[1], min[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(max[0], max[1], min[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(min[0], max[1], min[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(min[0], max[1], max[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(max[0], max[1], max[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(max[0], min[1], max[2])
    color.addData4(0, 0, 1, 1)
    vertex.addData3(min[0], min[1], max[2])
    color.addData4(0, 0, 1, 1)

    Geom geom(vertices)
    triangles = [
        [0, 2, 1],
	    [0, 3, 2],
	    [2, 3, 4],
	    [2, 4, 5],
	    [1, 2, 5],
	    [1, 5, 6],
	    [0, 7, 4],
	    [0, 4, 3],
	    [5, 4, 7],
	    [5, 7, 6],
	    [0, 6, 7],
	    [0, 1, 6]
        ]
    tri_prim = GeomTriangles(Geom.UHStatic)
    for tri in triangles:
        tri_prim.addVertices(tri[0], tri[1], tri[2])

    geom.addPrimitive(tri_prim)
    return geom*/

py::object createVoxelGrid(const py::array_t<bool>& arr, double scale, const py::array_t<float>& min_col, const py::array_t<float>& max_col)
{
    LVecBase4 min_col_c(min_col.at(0), min_col.at(1), min_col.at(2), min_col.at(3));
    LVecBase4 max_col_c(max_col.at(0), max_col.at(1), max_col.at(2), max_col.at(3));
    const py::ssize_t *arr_shape = arr.shape();
    const size_t vertex_shape[3] = { (size_t)arr_shape[0] + 1, (size_t)arr_shape[1] + 1, (size_t)arr_shape[2] + 1 };
    size_t y_stride = vertex_shape[0];
    size_t z_stride = vertex_shape[0] * vertex_shape[1];
    size_t num_vertices = vertex_shape[0] * vertex_shape[1] * vertex_shape[2];
    PT(GeomVertexData) vertices(new GeomVertexData("vertices", GeomVertexFormat::get_v3c4(), Geom::UH_static));
    vertices->set_num_rows(num_vertices);
    GeomVertexWriter vertex(vertices, "vertex");
    GeomVertexWriter color(vertices, "color");

    std::cout << "Generating vertices..." << std::endl;

    for (size_t z = 0; z < vertex_shape[2]; z++)
    {
        for (size_t y = 0; y < vertex_shape[1]; y++)
        {
            for (size_t x = 0; x < vertex_shape[1]; x++)
            {
                double prog = (double)z / (double)vertex_shape[2]; //(x / vertex_shape[0] + y / vertex_shape[1] + z / vertex_shape[2]) / 3
                LVecBase4 col = min_col_c * (1 - prog) + max_col_c * prog;
                vertex.add_data3(x * scale, y * scale, z * scale);
                color.add_data4(col[0], col[1], col[2], col[3]);
            }
        }
    }

    PT(Geom) geom(new Geom(vertices));
    PT(GeomTriangles) tri_prim(new GeomTriangles(Geom::UH_static));

    std::cout << "Generating faces..." << std::endl;

    for (py::ssize_t z = 0; z < arr_shape[2]; z++)
    {
        for (py::ssize_t y = 0; y < arr_shape[1]; y++)
        {
            for (py::ssize_t x = 0; x < arr_shape[0]; x++)
            {
                if (arr.at(x, y, z))
                {
                    size_t coord[8] = {
                        x + y * y_stride + z * z_stride,
                        (x + 1) + y * y_stride + z * z_stride,
                        (x + 1) + (y + 1) * y_stride + z * z_stride,
                        x + (y + 1) * y_stride + z * z_stride,
                        x + (y + 1) * y_stride + (z + 1) * z_stride,
                        (x + 1) + (y + 1) * y_stride + (z + 1) * z_stride,
                        (x + 1) + y * y_stride + (z + 1) * z_stride,
                         x + y * y_stride + (z + 1) * z_stride,
                    };

                    size_t triangles[12][3] = {
                        {coord[0], coord[2], coord[1]},
                        {coord[0], coord[3], coord[2]},
                        {coord[2], coord[3], coord[4]},
                        {coord[2], coord[4], coord[5]},
                        {coord[1], coord[2], coord[5]},
                        {coord[1], coord[5], coord[6]},
                        {coord[0], coord[7], coord[4]},
                        {coord[0], coord[4], coord[3]},
                        {coord[5], coord[4], coord[7]},
                        {coord[5], coord[7], coord[6]},
                        {coord[0], coord[6], coord[7]},
                        {coord[0], coord[1], coord[6]}
                    };

                    for (const auto& tri : triangles)
                        tri_prim->add_vertices(tri[0], tri[1], tri[2]);
                }
            }
        }
    }

    std::cout << "Adding primitive..." << std::endl;

    geom->add_primitive(tri_prim);

    py::object py_geom = py::cast(geom);
    return py_geom;
}

PYBIND11_MODULE(p3d_voxgrid, m) {
    m.doc() = "p3d_voxgrid plugin"; // Optional module docstring
    m.def("createVoxelGrid", &createVoxelGrid, "A function that creates a VoxelGrid for panda3d");
}