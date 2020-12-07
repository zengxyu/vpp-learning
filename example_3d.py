from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3, GeomNode, Geom, GeomPrimitive, GeomVertexData, GeomTriangles, GeomTristrips, GeomVertexWriter, GeomVertexFormat

def createCube(min, max):
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

    geom = Geom(vertices)

    tri_prim = GeomTriangles(Geom.UHStatic)
    for tri in triangles:
        tri_prim.addVertices(tri[0], tri[1], tri[2])

    geom.addPrimitive(tri_prim)
    return geom

class MyApp(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        posInterval1 = self.pandaActor.posInterval(13,
                                                   Point3(0, -10, 0),
                                                   startPos=Point3(0, 10, 0))
        posInterval2 = self.pandaActor.posInterval(13,
                                                   Point3(0, 10, 0),
                                                   startPos=Point3(0, -10, 0))
        hprInterval1 = self.pandaActor.hprInterval(3,
                                                   Point3(180, 0, 0),
                                                   startHpr=Point3(0, 0, 0))
        hprInterval2 = self.pandaActor.hprInterval(3,
                                                   Point3(0, 0, 0),
                                                   startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(posInterval1, hprInterval1,
                                  posInterval2, hprInterval2,
                                  name="pandaPace")
        self.pandaPace.loop()

        cube = GeomNode("cube")
        cube.addGeom(createCube([0, 0, 0], [1, 1, 1]))

        nodePath = self.render.attachNewNode(cube)

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


app = MyApp()
app.run()