import numpy as np

z = np.zeros((2, 2, 6), dtype='U2')
o = np.ones((2, 1), dtype='O')
print("z\n", z)
print("o\n", o)
c = np.hstack([[o,z], z])
print(c)
