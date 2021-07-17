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
    child = Children()
    child.say()
