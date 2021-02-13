
// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

class Vec3D
{
public:
	double x, y, z;

	Vec3D() : x(0), y(0), z(0) {};

	Vec3D(double x, double y, double z) : x(x), y(y), z(z) {};

	double abs() const
    {
        return std::sqrt(x*x + y*y + z*z);
    }

    double dot(const Vec3D &r) const
    {
        return x*r.x + y*r.y + z*r.z;
    }

    Vec3D cross(const Vec3D &r) const
    {
        return Vec3D(y*r.z - z*r.y, z*r.x - x*r.z, x*r.y - y*r.x);
    }

    void clip(const Vec3D &min, const Vec3D &max)
    {
        if (x < min.x) x = min.x;
        if (y < min.y) y = min.y;
        if (z < min.z) z = min.z;
        if (x > max.x) x = max.x;
        if (y > max.y) y = max.y;
        if (z > max.z) z = max.z;
    }

	double distanceFrom(const Vec3D &other) const
    {
        return (other - *this).abs();
    }

	Vec3D normalized() const
    {
        return *this / abs();
    }

	Vec3D& operator+= (const Vec3D &r)
    {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }

	Vec3D& operator-= (const Vec3D &r)
    {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }

	Vec3D& operator*= (const double &r)
    {
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }

	Vec3D& operator/= (const double &r)
    {
        x /= r;
        y /= r;
        z /= r;
        return *this;
    }

    Vec3D operator+ (const Vec3D& r) const
    {
        return Vec3D(x + r.x, y + r.y, z + r.y);
    }

    Vec3D operator- (const Vec3D& r) const
    {
        return Vec3D(x - r.x, y - r.y, z - r.y);
    }

    Vec3D operator* (const double &r) const
    {
        return Vec3D(x * r, y * r, z * r);
    }

    Vec3D operator/ (const double &r) const
    {
        return Vec3D(x / r, y / r, z / r);
    }
};

Vec3D operator* (const double &a, Vec3D b)
{
    return b *= a;
}

std::ostream& operator<< (std::ostream& output, const Vec3D& r)
{
	output << "(" << r.x << ", " << r.y << ", " << r.z << ")";
	return output;
}

std::pair<Vec3D, double> line_plane_intersection(const Vec3D &p0, const Vec3D &nv, const Vec3D &l0, const Vec3D &lv)
{
    /** return intersection of a line with a plane

    Parameters:
        p0: Point in plane
        nv: Normal vector of plane
        l0: Point on line
        lv: Direction vector of line

    Returns:
        The intersection point
    */
    double denom = lv.dot(nv);
    if (denom == 0) // No intersection or line contained in plane
        return std::make_pair(Vec3D(), 0);
    
    double d = (p0 - l0).dot(nv) / denom;
    return std::make_pair(l0 + (lv * d), d);
}

bool point_in_rectangle(const Vec3D &p, const Vec3D &p0, const Vec3D &v1, const Vec3D &v2)
{
    /** check if point is within reactangle

    Parameters:
        p: Point
        p0: Corner point of rectangle
        v1: Side vector 1 starting from p0
        v2: Side vector 2 starting from p0

    Returns:
        True if within rectangle
    */
    double v1_len = v1.abs();
    double v2_len = v2.abs();
    double v1_proj_length = (p - p0).dot(v1 / v1_len);
    double v2_proj_length = (p - p0).dot(v2 / v2_len);
    return ((v1_proj_length >= 0) && (v1_proj_length <= v1_len) && (v2_proj_length >= 0) && (v2_proj_length <= v2_len));
}

std::pair<Vec3D, Vec3D> get_bb_points(const std::vector<Vec3D> &points, const ssize_t *shape)
{
    Vec3D min = points[0];
    Vec3D max = points[0];
    for (size_t i = 1; i < points.size(); i++)
    {
        if (points[i].x < min.x) min.x = points[i].x;
        if (points[i].y < min.y) min.y = points[i].y;
        if (points[i].z < min.z) min.z = points[i].z;
        if (points[i].x > max.x) max.x = points[i].x;
        if (points[i].y > max.y) max.y = points[i].y;
        if (points[i].z > max.z) max.z = points[i].z;
    }
    Vec3D min_bound(0, 0, 0);
    Vec3D max_bound(shape[0], shape[1], shape[2]);
    min.clip(min_bound, max_bound);
    max.clip(min_bound, max_bound);
    return std::make_pair(min, max);
}

inline bool in_range(int val, int size)
{
    return val >= 0 && val < size;
}

int count_unknown(const py::array_t<int> &known_map, const Vec3D &start, const Vec3D &dir_vec, const double &step, const double &len)
{
    int unkown = 0;
    for (double frac=0.0; frac < len; frac += step)
    {
        Vec3D cur = start + frac * dir_vec;
        int x = (int)cur.x;
        if (!in_range(x, known_map.shape()[0])) break;
        int y = (int)cur.y;
        if (!in_range(y, known_map.shape()[1])) break;
        int z = (int)cur.z;
        if (!in_range(z, known_map.shape()[2])) break;
        int cell_val = *known_map.data(x, y, z);
        if (cell_val == 0)
            unkown++;
    }
    return unkown;
}

/*py::array_t<int> generate_camera_image(const py::array_t<int> &map, const Vec3D& cam_pos, const Vec3D& ep_left_down, const Vec3D& ep_left_up, const Vec3D& ep_right_down, Vec3D& ep_right_up, int xres=640, int yres=480)
{
    Vec3D left_right = ep_right_up - ep_left_up;
    Vec3D up_down = ep_left_down - ep_left_up;
    for (int x = 0; x < xres; x++)
    {
        double xfac = (double)x / (double)xres;
        for (int y = 0; y < yres; y++)
        {
            double yfac = (double)y / (double)yres;
            Vec3D cur_target = ep_left_up + xfac * left_right + yfac * up_down;
            Vec3D unit_dir = (cur_target - cam_pos).normalized();
        }
    }
}*/

std::tuple<py::array_t<int>, int, std::vector<int>, std::vector<int>> update_grid_inds_in_view(py::array_t<int> &known_map, const py::array_t<int> &global_map, const Vec3D& cam_pos, const Vec3D& ep_left_down, const Vec3D& ep_left_up, const Vec3D& ep_right_down, Vec3D& ep_right_up)
{
    std::vector<Vec3D> points = {cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up};
    auto[bb_min, bb_max] = get_bb_points(points, known_map.shape());
    Vec3D v1 = ep_right_up - ep_right_down;
    Vec3D v2 = ep_left_down - ep_right_down;
    Vec3D plane_normal = v1.cross(v2);
    int found_targets = 0;
    std::vector<int> coords, values;
    for (size_t z = (size_t)bb_min.z; z < (size_t)bb_max.z; z++)
    {
        for (size_t y = (size_t)bb_min.y; y < (size_t)bb_max.y; y++)
        {
            for (size_t x = (size_t)bb_min.x; x < (size_t)bb_max.x; x++)
            {
                Vec3D point = Vec3D(x, y, z);
                if (*known_map.data(x, y, z) != 0)
                    continue;
                auto[p_proj, rel_dist] = line_plane_intersection(ep_right_down, plane_normal, cam_pos, (point - cam_pos));
                if (rel_dist < 1.0) // if point lies behind projection or there is no projection, skip
                    continue;
                if (point_in_rectangle(p_proj, ep_right_down, v1, v2))
                {
                    *known_map.mutable_data(x, y, z) = *global_map.data(x, y, z);
                    // for now, occupied cells are targets, change later
                    if (*known_map.data(x, y, z) == 2)
                        found_targets += 1;
                     //if (!headless) {}
                    coords.push_back(x);
                    coords.push_back(y);
                    coords.push_back(z);
                    values.push_back(*known_map.data(x, y, z) + 3);
                }
            }
        }
    }
    return std::make_tuple(known_map, found_targets, coords, values);
}

void test()
{
    std::cout << "Test" << std::endl; 
}

PYBIND11_MODULE(field_env_3d_helper, m) {
    m.doc() = "field env 3d helper plugin"; // Optional module docstring
    py::class_<Vec3D>(m, "Vec3D")
        .def(py::init<double, double, double>())
        .def("dot", &Vec3D::dot)
        .def("abs", &Vec3D::abs);
    m.def("update_grid_inds_in_view", &update_grid_inds_in_view, "Update grid indices");
    m.def("count_unknown", &count_unknown, "Count unknown cells on ray");
    m.def("test", &test, "Print test");
}