from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
# from direct.actor.Actor import Actor
# from direct.interval.IntervalGlobal import Sequence
from panda3d.core import GeomNode, Geom, GeomVertexData, GeomTriangles, GeomVertexWriter, GeomVertexFormat  # , Point3, GeomPrimitive, GeomTristrips
from panda3d.core import LVecBase3i, LineSegs, PTA_int, PTA_float  # ,DirectionalLight, AmbientLight, LVecBase4, BitArray, TextNode

from p3d_voxgrid import VoxelGrid
from field_env_3d import Field, Action

import numpy as np
import binvox_rw

import time


def createEdgedCube(min, max):
    lines = LineSegs()
    lines.moveTo(min[0], min[1], min[2])
    lines.drawTo(max[0], min[1], min[2])
    lines.drawTo(max[0], max[1], min[2])
    lines.drawTo(min[0], max[1], min[2])
    lines.drawTo(min[0], min[1], min[2])
    lines.drawTo(min[0], min[1], max[2])
    lines.drawTo(max[0], min[1], max[2])
    lines.drawTo(max[0], min[1], min[2])

    lines.moveTo(max[0], max[1], min[2])
    lines.drawTo(max[0], max[1], max[2])
    lines.drawTo(max[0], min[1], max[2])

    lines.moveTo(max[0], max[1], max[2])
    lines.drawTo(min[0], max[1], max[2])
    lines.drawTo(min[0], min[1], max[2])

    lines.moveTo(min[0], max[1], max[2])
    lines.drawTo(min[0], max[1], min[2])

    lines.setThickness(4)
    node = lines.create()
    return node


def createCube(min, max, visible_faces=[1, 1, 1, 1, 1, 1]):
    vertices = GeomVertexData('vertices', GeomVertexFormat.get_v3c4(), Geom.UHStatic)
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

    geom = Geom(vertices)
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
    return geom


def createVoxelGrid(arr, scale, min_col, max_col):
    arr_shape = np.shape(arr)
    vertex_shape = (arr_shape[0] + 1, arr_shape[1] + 1, arr_shape[2] + 1)
    y_stride = vertex_shape[0]
    z_stride = vertex_shape[0] * vertex_shape[1]
    num_vertices = vertex_shape[0] * vertex_shape[1] * vertex_shape[2]
    vertices = GeomVertexData('vertices', GeomVertexFormat.get_v3c4(), Geom.UHStatic)
    vertices.setNumRows(num_vertices)
    vertex = GeomVertexWriter(vertices, 'vertex')
    color = GeomVertexWriter(vertices, 'color')

    print("Generating vertices...")

    for z in range(vertex_shape[2]):
        for y in range(vertex_shape[1]):
            for x in range(vertex_shape[0]):
                prog = z / vertex_shape[2]  # (x / vertex_shape[0] + y / vertex_shape[1] + z / vertex_shape[2]) / 3
                col = np.asarray(min_col) * (1 - prog) + np.asarray(max_col) * prog
                vertex.addData3(x * scale, y * scale, z * scale)
                color.addData4(col[0], col[1], col[2], col[3])

    geom = Geom(vertices)
    tri_prim = GeomTriangles(Geom.UHStatic)

    print("Generating faces...")

    for z in range(arr_shape[2]):
        for y in range(arr_shape[1]):
            for x in range(arr_shape[0]):
                if arr[x, y, z]:
                    coord = [
                        x + y * y_stride + z * z_stride,
                        (x + 1) + y * y_stride + z * z_stride,
                        (x + 1) + (y + 1) * y_stride + z * z_stride,
                        x + (y + 1) * y_stride + z * z_stride,
                        x + (y + 1) * y_stride + (z + 1) * z_stride,
                        (x + 1) + (y + 1) * y_stride + (z + 1) * z_stride,
                        (x + 1) + y * y_stride + (z + 1) * z_stride,
                        x + y * y_stride + (z + 1) * z_stride,
                        ]
                    triangles = [
                        [coord[0], coord[2], coord[1]],
                        [coord[0], coord[3], coord[2]],
                        [coord[2], coord[3], coord[4]],
                        [coord[2], coord[4], coord[5]],
                        [coord[1], coord[2], coord[5]],
                        [coord[1], coord[5], coord[6]],
                        [coord[0], coord[7], coord[4]],
                        [coord[0], coord[4], coord[3]],
                        [coord[5], coord[4], coord[7]],
                        [coord[5], coord[7], coord[6]],
                        [coord[0], coord[6], coord[7]],
                        [coord[0], coord[1], coord[6]]
                        ]

                    for tri in triangles:
                        tri_prim.addVertices(tri[0], tri[1], tri[2])

    geom.addPrimitive(tri_prim)
    return geom


def line_plane_intersection(p0, nv, l0, lv):
    """ return intersection of a line with a plane

    Parameters:
        p0: Point in plane
        nv: Normal vector of plane
        l0: Point on line
        lv: Direction vector of line

    Returns:
        The intersection point
    """
    denom = np.dot(lv, nv)
    if denom == 0:  # No intersection or line contained in plane
        return None, None

    d = np.dot((p0 - l0), nv) / denom
    return l0 + lv * d, d


def point_in_rectangle(p, p0, v1, v2):
    """ check if point is within reactangle

    Parameters:
        p: Point
        p0: Corner point of rectangle
        v1: Side vector 1 starting from p0
        v2: Side vector 2 starting from p0

    Returns:
        True if within rectangle
    """
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    v1_proj_length = np.dot((p - p0), v1 / v1_len)
    v2_proj_length = np.dot((p - p0), v2 / v2_len)
    return (v1_proj_length >= 0 and v1_proj_length <= v1_len and v2_proj_length >= 0 and v2_proj_length <= v2_len)


def get_bb_points(points):
    return np.amin(points, axis=0), np.amax(points, axis=0)


def get_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, shape):
    bb_min, bb_max = get_bb_points([cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up])
    bb_min, bb_max = np.clip(np.rint(bb_min), [0, 0, 0], shape).astype(int), np.clip(np.rint(bb_max), [0, 0, 0], shape).astype(int)
    v1 = ep_right_up - ep_right_down
    v2 = ep_left_down - ep_right_down
    plane_normal = np.cross(v1, v2)
    grid_inds = []
    for z in range(bb_min[2], bb_max[2] + 1):
        for y in range(bb_min[1], bb_max[1] + 1):
            for x in range(bb_min[0], bb_max[0] + 1):
                point = np.array([x, y, z])
                p_proj, rel_dist = line_plane_intersection(ep_right_down, plane_normal, cam_pos, (point - cam_pos))
                if p_proj is None or rel_dist < 1.0:  # if point lies behind projection, skip
                    continue
                if point_in_rectangle(p_proj, ep_right_down, v1, v2):
                    grid_inds.extend([x, y, z])
    return grid_inds


class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.scale = 0.05

        # Load the environment model.
        # self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        # self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        # self.scene.setScale(0.25, 0.25, 0.25)
        # self.scene.setPos(-8, 42, -10)

        # Add the spinCameraTask procedure to the task manager.
        # self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        # self.pandaActor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        # self.pandaActor.setScale(0.005, 0.005, 0.005)
        # self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        # self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        # posInterval1 = self.pandaActor.posInterval(13, Point3(0, -10, 0), startPos=Point3(0, 10, 0))
        # posInterval2 = self.pandaActor.posInterval(13, Point3(0, 10, 0), startPos=Point3(0, -10, 0))
        # hprInterval1 = self.pandaActor.hprInterval(3, Point3(180, 0, 0), startHpr=Point3(0, 0, 0))
        # hprInterval2 = self.pandaActor.hprInterval(3, Point3(0, 0, 0), startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        # self.pandaPace = Sequence(posInterval1, hprInterval1, posInterval2, hprInterval2, name="pandaPace")
        # self.pandaPace.loop()

        # cube = GeomNode("cube")
        # cube.addGeom(createCube([0, 0, 0], [1, 1, 1]))

        self.voxgrid_node = GeomNode("voxgrid")
        self.fov_node = GeomNode("fov")
        # voxgrid.addGeom(createVoxelGrid(
        #    np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 0], [1, 1, 1]]]),
        #    (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))

        with open('VG07_6.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        grid_array = np.transpose(model.data, (2, 0, 1))

        print(grid_array.shape)

        self.test_cube = createEdgedCube([0, 0, 0], np.asarray(grid_array.shape) * self.scale)
        self.render.attachNewNode(self.test_cube)

        self.env = Field(shape=grid_array.shape, target_count=100, sensor_range=50.0, hfov=90.0, vfov=60.0, scale=self.scale, max_steps=1000, headless=False)

        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.env.compute_fov()

        # debug vec positions
        # ld_text = TextNode('ld text')
        # ld_text.setText("Left-Down")
        # lu_text = TextNode('ld text')
        # lu_text.setText("Left-Up")
        # rd_text = TextNode('ld text')
        # rd_text.setText("Right-Down")
        # ru_text = TextNode('ld text')
        # ru_text.setText("Right-Up")
        # ldTextNodePath = self.render.attachNewNode(ld_text)
        # ldTextNodePath.setPos(tuple(ep_left_down * self.scale))
        # luTextNodePath = self.render.attachNewNode(lu_text)
        # luTextNodePath.setPos(tuple(ep_left_up * self.scale))
        # rdTextNodePath = self.render.attachNewNode(rd_text)
        # rdTextNodePath.setPos(tuple(ep_right_down * self.scale))
        # ruTextNodePath = self.render.attachNewNode(ru_text)
        # ruTextNodePath.setPos(tuple(ep_right_up * self.scale))

        # self.fov_geom = self.env.create_fov_geom(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        # self.fov_node.addGeom(self.fov_geom)
        self.fov_node = self.env.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)

        self.accept('q', self.keyboardInput, ['q'])
        self.accept('w', self.keyboardInput, ['w'])
        self.accept('e', self.keyboardInput, ['e'])
        self.accept('a', self.keyboardInput, ['a'])
        self.accept('s', self.keyboardInput, ['s'])
        self.accept('d', self.keyboardInput, ['d'])
        self.accept('u', self.keyboardInput, ['u'])
        self.accept('i', self.keyboardInput, ['i'])
        self.accept('o', self.keyboardInput, ['o'])
        self.accept('j', self.keyboardInput, ['j'])
        self.accept('k', self.keyboardInput, ['k'])
        self.accept('l', self.keyboardInput, ['l'])

        # self.taskMgr.add(self.moveCameraTask, "MoveCameraTask")

        model_flat = grid_array.flatten()

        # print('Converting numpy array to BitArray')

        # barr = BitArray()
        # for i in range(len(model_flat)):
        #     barr.set_bit_to(i, model_flat[i])

        # print('Done')

        # geom = createVoxelGrid(model.data, 0.01, (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0))
        self.shape = LVecBase3i(model.data.shape[0], model.data.shape[1], model.data.shape[2])
        # print('Types: {}, {}, {}, {}, {}'.format(type(barr), type(shape), type(0.01),
        # type(LVecBase4(0.0, 1.0, 0.0, 1.0)), type(LVecBase4(0.0, 0.0, 1.0, 1.0))))
        # geom = p3d_voxgrid.create_voxel_grid(barr, shape, self.scale, LVecBase4(0.0, 1.0, 0.0, 1.0), LVecBase4(0.0, 0.0, 1.0, 1.0))
        # print('Geom: {}'.format(type(geom)))

        print(np.max(model_flat.astype(int)))

        colors = PTA_float([220/255, 20/255, 60/255, 1.0, 199/255, 21/255, 133/255, 1.0])
        self.voxgrid = VoxelGrid(PTA_int(model_flat.astype(int).tolist()), self.shape, colors, self.scale)
        # self.voxgrid = VoxelGrid(self.shape, PTA_float([220/255, 20/255, 60/255, 1, 199/255, 21/255, 133/255, 1]), self.scale)

        self.voxgrid_node.addGeom(self.voxgrid.getGeom())

        self.render.attachNewNode(self.voxgrid_node)
        self.render.attachNewNode(self.fov_node)

        # alight = AmbientLight('alight')
        # alight.setColor((0.2, 0.2, 0.2, 1))
        # alnp = render.attachNewNode(alight)
        # self.render.setLight(alnp)

        # dlight = DirectionalLight('vg_light')
        # dlight.setColor((1, 1, 1, 1))
        # dlight.setDirection((0, 0, -1))
        # dlight.setShadowCaster(True, 512, 512)
        # dlnp = render.attachNewNode(dlight)
        # dlnp.setHpr(0, -60, 0)
        # self.render.setLight(dlnp)
        # self.render.setShaderAuto()

        # shader = self.loader.loadShader("test_shader.sha")
        # self.render.setShader(shader)
        # self.render.setShaderInput("light", dlnp)

    def keyboardInput(self, char):
        if char == 'a':
            act = Action.MOVE_LEFT
        elif char == 'd':
            act = Action.MOVE_RIGHT
        elif char == 'w':
            act = Action.MOVE_FORWARD
        elif char == 's':
            act = Action.MOVE_BACKWARD
        elif char == 'e':
            act = Action.MOVE_DOWN
        elif char == 'q':
            act = Action.MOVE_UP
        elif char == 'j':
            act = Action.ROTATE_YAW_N
        elif char == 'l':
            act = Action.ROTATE_YAW_P
        elif char == 'i':
            act = Action.ROTATE_PITCH_P
        elif char == 'k':
            act = Action.ROTATE_PITCH_N
        elif char == 'o':
            act = Action.ROTATE_ROLL_P
        elif char == 'u':
            act = Action.ROTATE_ROLL_N
        else:
            return

        self.env.step(act)
        self.fov_node.removeGeom(0)
        cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.env.compute_fov()
        # self.fov_geom = self.env.create_fov_geom(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        # self.fov_node.addGeom(self.fov_geom)
        self.fov_node = self.env.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.render.attachNewNode(self.fov_node)

        print("Computing indices")
        time_start = time.perf_counter()
        inds = get_grid_inds_in_view(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, self.shape)
        print("Done in {} s".format(time.perf_counter() - time_start))
        print("Drawing indices")
        time_start = time.perf_counter()
        self.voxgrid.updateValues(PTA_int(inds), 2)
        print("Done in {} s".format(time.perf_counter() - time_start))

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont

    # def moveCameraTask(self, task):
    #    action = self.player.get_action(None, None)
    #    if action == Action.DO_NOTHING:
    #        return Task.cont
    #
    #    self.fov_node.removeGeom(self.fov_geom)
    #    self.env.step(action)
    #    cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up = self.env.compute_fov()
    #    self.fov_geom = self.env.create_fov_geom(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
    #    self.fov_node.addGeom(self.fov_geom)


app = MyApp()
app.run()
