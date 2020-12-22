from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, GeomNode, Geom, GeomPrimitive, GeomVertexData, GeomTriangles, GeomTristrips, GeomVertexWriter, GeomVertexFormat, DirectionalLight, AmbientLight, LVecBase3i, LVecBase4, BitArray

import p3d_voxgrid

import numpy as np
import binvox_rw

def createCube(min, max, visible_faces = [1, 1, 1, 1, 1, 1]):
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
                prog = z / vertex_shape[2] #(x / vertex_shape[0] + y / vertex_shape[1] + z / vertex_shape[2]) / 3
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

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        #self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        #self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        #self.scene.setScale(0.25, 0.25, 0.25)
        #self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        #self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        #self.pandaActor = Actor("models/panda-model", {"walk": "models/panda-walk4"})
        #self.pandaActor.setScale(0.005, 0.005, 0.005)
        #self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        #self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        #posInterval1 = self.pandaActor.posInterval(13, Point3(0, -10, 0), startPos=Point3(0, 10, 0))
        #posInterval2 = self.pandaActor.posInterval(13, Point3(0, 10, 0), startPos=Point3(0, -10, 0))
        #hprInterval1 = self.pandaActor.hprInterval(3, Point3(180, 0, 0), startHpr=Point3(0, 0, 0))
        #hprInterval2 = self.pandaActor.hprInterval(3, Point3(0, 0, 0), startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        #self.pandaPace = Sequence(posInterval1, hprInterval1, posInterval2, hprInterval2, name="pandaPace")
        #self.pandaPace.loop()

        #cube = GeomNode("cube")
        #cube.addGeom(createCube([0, 0, 0], [1, 1, 1]))

        voxgrid = GeomNode("voxgrid")
        #voxgrid.addGeom(createVoxelGrid(
        #    np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 0, 0], [1, 1, 1]]]),
        #    (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)))

        with open('VG07_6_fruits.binvox', 'rb') as f:
            model = binvox_rw.read_as_3d_array(f)

        print(model.data.shape)
        
        model_flat = model.data.flatten()
        
        print('Converting numpy array to BitArray')
        
        barr = BitArray()
        for i in range(len(model_flat)):
            barr.set_bit_to(i, model_flat[i])
            
        print('Done')
        
        #geom = createVoxelGrid(model.data, 0.01, (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0))
        shape = LVecBase3i(model.data.shape[0], model.data.shape[1], model.data.shape[2])
        print('Types: {}, {}, {}, {}, {}'.format(type(barr), type(shape), type(0.01), type(LVecBase4(0.0, 1.0, 0.0, 1.0)), type(LVecBase4(0.0, 0.0, 1.0, 1.0))))
        geom = p3d_voxgrid.create_voxel_grid(barr, shape, 0.1, LVecBase4(0.0, 1.0, 0.0, 1.0), LVecBase4(0.0, 0.0, 1.0, 1.0))
        print('Geom: {}'.format(type(geom)))
        voxgrid.addGeom(geom)


        self.render.attachNewNode(voxgrid)

        #alight = AmbientLight('alight')
        #alight.setColor((0.2, 0.2, 0.2, 1))
        #alnp = render.attachNewNode(alight)
        #self.render.setLight(alnp)
        
        #dlight = DirectionalLight('vg_light')
        #dlight.setColor((1, 1, 1, 1))
        #dlight.setDirection((0, 0, -1))
        #dlight.setShadowCaster(True, 512, 512)
        #dlnp = render.attachNewNode(dlight)
        #dlnp.setHpr(0, -60, 0)
        #self.render.setLight(dlnp)
        #self.render.setShaderAuto()

        #shader = self.loader.loadShader("test_shader.sha")
        #self.render.setShader(shader)
        #self.render.setShaderInput("light", dlnp)

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


app = MyApp()
app.run()