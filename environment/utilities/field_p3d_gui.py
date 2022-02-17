#!/usr/bin/environment python

import numpy as np
from enum import IntEnum
from direct.showbase.ShowBase import ShowBase
from panda3d.core import GeomNode, Geom, GeomVertexData, GeomTriangles, GeomVertexWriter, GeomVertexFormat, LineSegs
from panda3d.core import LVecBase3i, PTA_int, PTA_float
from p3d_voxgrid import VoxelGrid
from direct.stdpy import threading


class GuiFieldValues(IntEnum):
    FREE_UNSEEN = 0,
    OCCUPIED_UNSEEN = 1,
    TARGET_UNSEEN = 2,
    FREE_SEEN = 3,
    OCCUPIED_SEEN = 4,
    TARGET_SEEN = 5


class FieldGUI(ShowBase):

    def __init__(self, env, scale):
        ShowBase.__init__(self)
        self.env = env
        self.scale = scale

        # Color constants
        self.FOV_ALPHA = 1.0  # not working
        self.FOV_UP_COLOR = (220 / 255, 20 / 255, 60 / 255, self.FOV_ALPHA)  # Crimson red
        self.FOV_DOWN_COLOR = (199 / 255, 21 / 255, 133 / 255, self.FOV_ALPHA)  # MediumVioletRed
        self.FOV_LEFT_COLOR = (255 / 255, 69 / 255, 0 / 255, self.FOV_ALPHA)  # OrangeRed
        self.FOV_RIGHT_COLOR = (255 / 255, 215 / 255, 0 / 255, self.FOV_ALPHA)  # Gold
        self.FOV_FRONT_COLOR = (218 / 255, 112 / 255, 214 / 255, self.FOV_ALPHA)  # Orchid

        self.OCCUPIED_UNSEEN_COLOR = (34 / 255, 139 / 255, 34 / 255, 1.0)  # ForestGreen
        self.TARGET_UNSEEN_COLOR = (199 / 255, 21 / 255, 133 / 255, 1.0)  # MediumVioletRed
        self.FREE_SEEN_COLOR = (255 / 255, 215 / 255, 0 / 255, 1.0)  # Gold
        self.OCCUPIED_SEEN_COLOR = (0 / 255, 255 / 255, 255 / 255, 1.0)  # OrangeRed
        self.TARGET_SEEN_COLOR = (218 / 255, 112 / 255, 214 / 255, 1.0)  # Orchid
        self.VISIT_COLOR = (139 / 255, 69 / 255, 19 / 255, 1.0)  # Saddle Brown
        self.X_COLOR = (1, 0, 0, 1)
        self.Y_COLOR = (0, 1, 0, 1)
        self.Z_COLOR = (0, 0, 1, 1)

        self.voxgrid_node = GeomNode("voxgrid")
        self.fov_node = None
        self.fov_node_path = None

        self.colors = PTA_float(self.OCCUPIED_UNSEEN_COLOR + self.TARGET_UNSEEN_COLOR + self.FREE_SEEN_COLOR +
                                self.OCCUPIED_SEEN_COLOR + self.TARGET_SEEN_COLOR + self.VISIT_COLOR)
        self.bbox_fields = []
        self.fields = []
        self.voxgrid = None
        self.field_border = None

        self.gui_done = threading.Event()

        self.accept('reset', self.reset)
        self.accept('update_cell', self.updateSeenCell)
        self.accept('update_fov', self.updateFov)
        self.accept('update_fov_and_cells', self.updateFovAndCells)

    def reset(self):
        self.voxgrid = VoxelGrid(self.env.global_map.shape, self.colors, self.scale)
        self.field_border = self.create_edged_cube([0, 0, 0], np.asarray(self.env.global_map.shape) * self.scale)
        self.reset_fields()

        gui_map = self.env.global_map - 1  # GUI map is shifted by one for unseen cells
        self.voxgrid.reset(PTA_int(gui_map.flatten().tolist()))
        if self.env.parser_args.env_config["draw_bbox"]:
            self.reset_bbox_fields()
        self.gui_done.set()

    def reset_fields(self):
        # reset field border
        if self.fields:
            for field in self.fields:
                field.removeNode()
        self.fields.append(self.render.attachNewNode(self.field_border))
        self.fields.append(self.render.attachNewNode(self.voxgrid_node))

        # reset voxgrid
        if self.voxgrid_node:
            self.voxgrid_node.removeAllGeoms()

        self.voxgrid_node.addGeom(self.voxgrid.getGeom())

    def reset_bbox_fields(self):
        # remove the nodes from self.bbox_fields
        if self.bbox_fields:
            for bbox in self.bbox_fields:
                bbox.removeNode()
        # add new bbox
        for bbox in self.env.bounding_boxes:
            bbox_field = self.create_edged_cube(np.asarray(bbox[0]) * self.scale, np.asarray(bbox[1]) * self.scale)
            self.bbox_fields.append(self.render.attachNewNode(bbox_field))

    def create_edged_cube(self, min, max):
        lines = LineSegs()
        lines.setColor(self.X_COLOR)
        lines.moveTo(min[0], min[1], min[2])
        lines.drawTo(max[0], min[1], min[2])
        lines.setColor(self.Y_COLOR)
        lines.moveTo(max[0], min[1], min[2])
        lines.drawTo(max[0], max[1], min[2])
        lines.setColor(self.X_COLOR)
        lines.moveTo(max[0], max[1], min[2])
        lines.drawTo(min[0], max[1], min[2])
        lines.setColor(self.Y_COLOR)
        lines.moveTo(min[0], max[1], min[2])
        lines.drawTo(min[0], min[1], min[2])
        lines.setColor(self.Z_COLOR)
        lines.moveTo(min[0], min[1], min[2])
        lines.drawTo(min[0], min[1], max[2])
        lines.setColor(self.X_COLOR)
        lines.moveTo(min[0], min[1], max[2])
        lines.drawTo(max[0], min[1], max[2])
        lines.setColor(self.Z_COLOR)
        lines.moveTo(max[0], min[1], max[2])
        lines.drawTo(max[0], min[1], min[2])

        lines.setColor(self.Z_COLOR)
        lines.moveTo(max[0], max[1], min[2])
        lines.drawTo(max[0], max[1], max[2])
        lines.setColor(self.Y_COLOR)
        lines.moveTo(max[0], max[1], max[2])
        lines.drawTo(max[0], min[1], max[2])

        lines.setColor(self.X_COLOR)
        lines.moveTo(max[0], max[1], max[2])
        lines.drawTo(min[0], max[1], max[2])
        lines.setColor(self.Y_COLOR)
        lines.moveTo(min[0], max[1], max[2])
        lines.drawTo(min[0], min[1], max[2])

        lines.setColor(self.Z_COLOR)
        lines.moveTo(min[0], max[1], max[2])
        lines.drawTo(min[0], max[1], min[2])

        lines.setThickness(4)
        node = lines.create()
        return node

    def create_fov_geom(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        vertices = GeomVertexData('vertices', GeomVertexFormat.get_v3c4(), Geom.UHStatic)
        vertices.setNumRows(16)
        vertex = GeomVertexWriter(vertices, 'vertex')
        color = GeomVertexWriter(vertices, 'color')

        cam_pos_scaled = np.asarray(cam_pos) * self.scale
        ep_left_up_scaled = np.asarray(ep_left_up) * self.scale
        ep_left_down_scaled = np.asarray(ep_left_down) * self.scale
        ep_right_up_scaled = np.asarray(ep_right_up) * self.scale
        ep_right_down_scaled = np.asarray(ep_right_down) * self.scale

        # left (0-2)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_LEFT_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_LEFT_COLOR)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_LEFT_COLOR)

        # up (3-5)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_UP_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_UP_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_UP_COLOR)

        # right (6-8)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_RIGHT_COLOR)

        # down (9-11)
        vertex.addData3(tuple(cam_pos_scaled))
        color.addData4(self.FOV_DOWN_COLOR)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_DOWN_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_DOWN_COLOR)

        # front (12-15)
        vertex.addData3(tuple(ep_left_down_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_left_up_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_right_down_scaled))
        color.addData4(self.FOV_FRONT_COLOR)
        vertex.addData3(tuple(ep_right_up_scaled))
        color.addData4(self.FOV_FRONT_COLOR)

        geom = Geom(vertices)
        tri_prim = GeomTriangles(Geom.UHStatic)
        tri_prim.add_consecutive_vertices(0, 12)
        tri_prim.add_vertices(13, 15, 12)
        tri_prim.add_vertices(14, 12, 15)

        geom.add_primitive(tri_prim)
        return geom

    def create_fov_lines(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        cam_pos_scaled = np.asarray(cam_pos) * self.scale
        ep_left_up_scaled = np.asarray(ep_left_up) * self.scale
        ep_left_down_scaled = np.asarray(ep_left_down) * self.scale
        ep_right_up_scaled = np.asarray(ep_right_up) * self.scale
        ep_right_down_scaled = np.asarray(ep_right_down) * self.scale

        lines = LineSegs()
        lines.moveTo(tuple(cam_pos_scaled))
        lines.drawTo(tuple(ep_left_down_scaled))
        lines.draw_to(tuple(ep_left_up_scaled))
        lines.draw_to(tuple(cam_pos_scaled))
        lines.draw_to(tuple(ep_right_down_scaled))
        lines.draw_to(tuple(ep_right_up_scaled))
        lines.draw_to(tuple(cam_pos_scaled))

        lines.move_to(tuple(ep_left_down_scaled))
        lines.draw_to(tuple(ep_right_down_scaled))

        lines.move_to(tuple(ep_left_up_scaled))
        lines.draw_to(tuple(ep_right_up_scaled))

        lines.setThickness(4)
        node = lines.create()
        return node

    def updateSeenCell(self, coord):
        # seen values 3 higher than unseen
        self.voxgrid.updateValue(LVecBase3i(coord), self.env.global_map[coord] + 3)
        self.gui_done.set()

    def updateSeenCells(self, coords, values):
        self.voxgrid.updateValues(coords, values)
        self.gui_done.set()

    def updateFov(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up):
        if self.fov_node_path:
            self.fov_node_path.removeNode()

        self.fov_node = self.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.fov_node_path = self.render.attachNewNode(self.fov_node)
        self.gui_done.set()

    def updateFovAndCells(self, cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up, coords, values):
        if self.fov_node_path:
            self.fov_node_path.removeNode()

        self.fov_node = self.create_fov_lines(cam_pos, ep_left_down, ep_left_up, ep_right_down, ep_right_up)
        self.fov_node_path = self.render.attachNewNode(self.fov_node)
        # print(coords)
        self.voxgrid.updateValues(coords, values)
        self.gui_done.set()
