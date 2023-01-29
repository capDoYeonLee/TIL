from math import hypot

class vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self, other):
        return 'Vector(%r, %r)' % (self.x, self.y), 'other.x' %(other.x)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return vector(x,y), other.x, other.y

    def __mul__(self, other):
        return vector(self.x * scalar, self.y * scalar)


v1 = vector(2,4)
v2 = vector(2,1)
print(v1+ v2)