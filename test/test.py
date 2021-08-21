import numpy as np


class Parent(object):
    def __init__(self):
        print("Parent init")

    def say(self):
        print("I am parent")
        self.test()

    def test(self):
        print("pppp")


class Children(Parent):
    def __init__(self):
        super(Children, self).__init__()
        print("Children init")

    def say(self):
        super(Children, self).say()

    def test(self):
        print("cccccc")


if __name__ == '__main__':
    arr = np.array([0, 0, 0, 1, 1])
    print(np.sum(arr == 1))
# if __name__ == '__main__':
#     child = Children()
#     child.say()
# import torch
#
# a = torch.rand(4, 3, 28, 28)
# ind = torch.tensor([0, 2])
# c = a.index_select(0, ind)
# print(c.shape)
