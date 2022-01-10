import sys
import os
import numpy as np
from numpy.lib import recfunctions as rfn

class Pointcloud:
    def __init__(self, data):
        self.data = data
        if self.data.dtype.names:
            self.num_points = len(self.data)
            self.point_dims = len(self.data.dtype.names)
        else:
            self.num_points = self.data.shape[0]
            self.point_dims = self.data.shape[1]
        if self.point_dims == 3:
            self.h_fields = ['x', 'y', 'z']
            self.h_size = [4, 4, 4]
            self.h_type = ['F', 'F', 'F']
            self.h_count = [1, 1, 1]
        elif self.point_dims == 4:
            self.h_fields = ['x', 'y', 'z', 'label']
            self.h_size = [4, 4, 4, 4]
            self.h_type = ['F', 'F', 'F', 'U']
            self.h_count = [1, 1, 1, 1]

    def writeToFile(self, filename):
        with open(filename, 'wb') as file:
            header =  bytes((
                f"# .PCD v0.7 - Point Cloud Data file format\n"
                f"VERSION 0.7\n"
                f"FIELDS {' '.join(map(str, self.h_fields))}\n"
                f"SIZE {' '.join(map(str, self.h_size))}\n"
                f"TYPE {' '.join(map(str, self.h_type))}\n"
                f"COUNT {' '.join(map(str, self.h_count))}\n"
                f"WIDTH {self.num_points}\n"
                f"HEIGHT 1\n"
                f"VIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {self.num_points}\n"
                f"DATA binary\n"),
                encoding='ascii')
            file.write(header)
            file.write(self.data.tobytes())


def convert_file(infile_name):
    infile_name = infile_name
    outfile_name = os.path.splitext(infile_name)[0] + '.pcd'
    outfile_name1 = os.path.splitext(infile_name)[0] + '_label1.pcd'
    outfile_name2 = os.path.splitext(infile_name)[0] + '_label2.pcd'

    with open(infile_name, 'r') as infile:
        points = [tuple(line.rstrip().split(' ')) for line in infile.readlines()]
        if len(points[0]) == 3:
            points_array = np.array(points, dtype=np.float32)
            pointcloud = Pointcloud(points_array)
            pointcloud.writeToFile(outfile_name)
        elif len(points[0]) == 4:
            points_array = np.array(points, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('label', np.uint32)])
            pointcloud = Pointcloud(points_array)
            pointcloud.writeToFile(outfile_name)
        elif len(points[0]) == 5:
            points_array = np.array(points, dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('label1', np.uint32), ('label2', np.uint32)])
            points_array1 = rfn.repack_fields(points_array[['x', 'y', 'z', 'label1']])
            points_array2 = rfn.repack_fields(points_array[['x', 'y', 'z', 'label2']])
            pointcloud1 = Pointcloud(points_array1)
            pointcloud2 = Pointcloud(points_array2)
            pointcloud1.writeToFile(outfile_name1)
            pointcloud2.writeToFile(outfile_name2)


if len(sys.argv) < 2:
    print('Please provide a file or folder to convert')

if os.path.isfile(sys.argv[1]):
    print('Converting single file')
    convert_file(sys.argv[1])
    print('Done')

elif os.path.isdir(sys.argv[1]):
    print('Converting folder recursively')
    for subdir, _, files in os.walk(sys.argv[1]):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                filepath = os.path.join(subdir, file)
                print('Converting file ', file)
                convert_file(filepath)
                print('Done')
else:
    print('No valid path given')