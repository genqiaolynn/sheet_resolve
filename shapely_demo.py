# -*- coding:utf-8 -*-
__author__ = 'lynn'
__date__ = '2021/2/3 16:52'


import math
from shapely.geometry import Point
from shapely.geometry import LineString

point = Point(0, 0)
point_2 = Point((0, 0))
point_3 = Point(point)
print(point.area)  # 0
print(point.length)  # 0
print(point.coords)
print(list(point.coords))
# <shapely.coords.CoordinateSequence object at 0x7efd31647d30>
# [(0.0, 0.0)]
print(Point(0, 0).distance(Point(0, 1)))
# 1.0

# h*w-h2*w2/

line = LineString([(0, 0), (1, 1), (1, 2)])
print(line.area)
# 0.0
print(line.coords)
print(list(line.coords))
# <shapely.coords.CoordinateSequence object at 0x7f41d6c77d68>
# [(0.0, 0.0), (1.0, 1.0), (1.0, 2.0)]

print(line.bounds)
# (0.0, 0.0, 1.0, 2.0)
print(line.length)
# 2.414213562373095=sqrt(2)+1
print(math.sqrt(2))
# 1.4142135623730951
print(line.geom_type)
# LineString

