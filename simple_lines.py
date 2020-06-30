import math


def dx_dy(start, end):
    deltaX = end[0] - start[0]
    deltaY = end[1] - start[1]
    return deltaX, deltaY


def slope(start, end):
    dxdy = dx_dy(start, end)
    dx = dxdy[0]
    dy = dxdy[1]
    if dx == 0:
        return float('nan')
    return dy / dx


def slope_from_dxdy(dx, dy):
    return dy / dx if dx != 0 else 0


def normal(dx, dy):
    if dy == 0:
        return float('nan')
    return - dx / dy


def line_direction(start, end):
    dxdy = dx_dy(start, end)
    dx = dxdy[0]
    dy = dxdy[1]
    return 1 if (dx == 0 and dy > 0) or dx > 0 else -1


def line_length(start, end):
    return math.hypot(end[0] - start[0], end[1] - start[1])


def path_length(line, start, end):
    d = 0
    for i in range(line.index(start), line.index(end) - 1):
        d += line_length(line[i], line[i + 1])


def line_intersection(line1, line2):
    if not intersect(line1[0], line1[1], line2[0], line2[1]):
        return None
    start1 = line1[0]
    end1 = line1[1]
    xdiff = (start1[0] - end1[0], line2[0][0] - line2[1][0])
    ydiff = (start1[1] - end1[1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def ccw(A, B, C):  # "counter clock wise"?
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def path_intersection(path1, path2):
    for i in range(0, len(path1) - 1):
        for j in range(0, len(path2) - 1):
            path1_segment = [path1[i], path1[i + 1]]
            path2_segment = [path2[j], path2[j + 1]]
            intersection = line_intersection(path1_segment, path2_segment)
            if intersection is not None:
                return PathIntersection(intersection, path1_segment, path2_segment)
    return None


class PathIntersection:
    def __init__(self, intersection_point, path1_segment, path2_segment) -> None:
        super().__init__()
        self.intersection_point = intersection_point
        self.path1_segment = path1_segment
        self.path2_segment = path2_segment


def transpose(point, slope, distance):
    if math.isnan(slope):
        return point[0], point[1] + distance
    # https://math.stackexchange.com/questions/656500/given-a-point-slope-and-a-distance-along-that-slope-easily-find-a-second-p
    unit_length_vector = math.sqrt(1 + (slope * slope))
    return point[0] + distance / unit_length_vector, point[1] + ((distance * slope) / unit_length_vector)


def midpoint(start, end):
    deltaX = end[0] - start[0]
    deltaY = end[1] - start[1]
    return start[0] + (deltaX / 2), start[1] + (deltaY / 2)