import math
from PIL import Image
import svgwrite
import random
from enum import Enum
import time

import simple_lines as sl

# TODO Fail and retry mechanism if main trenches touch
# TODO better parallel line variation (rear trenches and communication) (?)
# TODO more chaotic lines in back (?)
# TODO More build param variation
# TODO Line object with memory and helpers such as remembering each iteration of midpoint displacement
# TODO clean joins between pulsed and non-pulsed lines
# TODO Apply templates, wire to more lines
# TODO Supervision distance not constant across trench - pos. floating point problems?
# TODO Unit tests
# TODO Separate trench model from map drawing
# TODO Unlock initial slope
# TODO Barbed wire smoothing
# TODO Friendly line - options for hide, full trenches, variability.
# TODO Templates are ugly
# TODO something something tensor fields

BLACK = (0, 0, 0)
SEPIA = (222, 222, 155)
CRIMSON = (200, 0, 0)
BLUE = (0, 150, 200)
GREYISH = (200, 195, 185)

sign = lambda a: (a > 0) - (a < 0)


def scale(atuple, scalar):
    return tuple([round(scalar * x) for x in atuple])


class TrenchParams:
    def __init__(self, map_width=800, map_height=800,
                 trench_color=CRIMSON, bg_color=SEPIA,
                 support_range=(100, 200), reserve_range=(100, 200),
                 supervision_range=(20, 40),
                 front_line_segment_length=50,
                 first_pass_depth=1,
                 template_dimensions=10,
                 comm_interval_range=(80, 280), comm_length_range=(400, 1600), comm_segment_length=12,
                 variance_factor_initial=5, variance_decay_factor=20 / 9,
                 sap_chance=0.5, sap_range=(25, 100),
                 zig_zag_chance=0.2,
                 wire_interval=16, wire_cross_length=6,
                 distance_scaling=1.0) -> None:
        self.map_width = map_width
        self.map_height = map_height
        self.trench_color = trench_color
        self.bg_color = bg_color
        self.sap_chance = sap_chance
        self.sap_range = scale(sap_range, distance_scaling)
        self.zig_zag_chance = zig_zag_chance
        self.support_range = scale(support_range, distance_scaling)
        self.reserve_range = scale(reserve_range, distance_scaling)
        self.front_line_segment_length = front_line_segment_length * distance_scaling
        self.first_pass_depth = first_pass_depth
        self.template_dimensions = template_dimensions * distance_scaling
        self.comm_interval_range = scale(comm_interval_range, distance_scaling)
        self.comm_length_range = scale(comm_length_range, distance_scaling)
        self.comm_segment_length = comm_segment_length * distance_scaling
        self.variance_factor_initial = variance_factor_initial
        self.variance_decay_factor = variance_decay_factor
        self.wire_interval = wire_interval
        self.wire_cross_length = wire_cross_length
        self.supervision_range = scale(supervision_range, distance_scaling)


def line_trench(params=TrenchParams()):
    initial_spread = (params.map_width / 4) - 1
    no_mans_land_range = (30, 300)

    second_pass_variance_factor = params.variance_factor_initial * (
            params.first_pass_depth * params.variance_decay_factor)

    # random.seed("war is hell")

    # draw bg
    img = Image.new('RGB', (params.map_width, params.map_height), "black")  # Create a new black image
    pixels = bg(img, params.bg_color)

    # first trench now
    x_center = params.map_width / 3.0

    start = (x_center + random.randrange(-initial_spread, initial_spread), 0.0)
    end = (x_center + random.randrange(-initial_spread, initial_spread), params.map_height - 1.0)
    front_line = [start, end]

    divide_line(front_line, start, end, params.front_line_segment_length,
                sl.line_length(start, end) / params.variance_factor_initial, params.first_pass_depth,
                jitter_decay=params.variance_decay_factor)

    # friendly
    friendly = copy_line(front_line, 0, -150)

    # rear lines
    rear_lines = []
    support_line = copy_line(front_line, 0, 20 + random.randrange(*params.support_range))
    rear_lines.append(support_line)
    rear_lines.append(copy_line(support_line, 0, random.randrange(*params.reserve_range)))

    for i in range(len(front_line) - 1, 0, -1):
        divide_line(front_line, front_line[i - 1], front_line[i], params.front_line_segment_length,
                    sl.line_length(start, end) / second_pass_variance_factor,
                    jitter_decay=params.variance_decay_factor)

    for rear_line in rear_lines:
        for i in range(len(rear_line) - 1, 0, -1):
            divide_line(rear_line, rear_line[i - 1], rear_line[i], params.front_line_segment_length,
                        sl.line_length(start, end) / second_pass_variance_factor,
                        jitter_decay=params.variance_decay_factor)

    for i in range(len(friendly) - 1, 0, -1):
        divide_line(friendly, friendly[i - 1], friendly[i], params.front_line_segment_length,
                    sl.line_length(start, end) / second_pass_variance_factor, jitter_decay=params.variance_decay_factor)

    # Add comm trenches
    comm_trenches = []
    d = 0
    comm_interval = random.randrange(*params.comm_interval_range)
    for i in range(0, len(front_line) - 2):
        comm_distance = random.randrange(*params.comm_length_range)
        p1 = front_line[i]
        p2 = front_line[i + 2]
        midpoint = front_line[i + 1]
        d += sl.line_length(p1, midpoint)
        if d >= comm_interval:
            normal_slope = sl.normal(p2[0] - p1[0], p2[1] - p1[1])
            new_point = sl.transpose(midpoint, normal_slope, comm_distance)
            new_trench = [midpoint, new_point]
            new_length = sl.line_length(*new_trench)
            divide_line(new_trench, midpoint, new_point, params.comm_segment_length, new_length / 10)
            comm_trenches.append(new_trench)
            # reset counters
            comm_interval = random.randrange(*params.comm_interval_range)
            d = 0
            # Add sap sometimes
            if random.random() < params.sap_chance:
                sap_length = random.randrange(*params.sap_range)
                listening_post = sl.transpose(midpoint, normal_slope, -sap_length)
                new_sap = [listening_post, midpoint]
                divide_line(new_sap, listening_post, midpoint, params.comm_segment_length, sap_length / 10)
                last_comm_trench_index = len(comm_trenches) - 1
                comm_trenches[last_comm_trench_index] = new_sap + comm_trenches[last_comm_trench_index]

        d += sl.line_length(midpoint, p2)

    # handle comm trench intersections
    randomized_comm_trenches = comm_trenches
    random.shuffle(randomized_comm_trenches)
    for i in range(0, len(randomized_comm_trenches)):
        for j in range(i + 1, len(randomized_comm_trenches)):
            path1 = randomized_comm_trenches[i]
            path2 = randomized_comm_trenches[j]
            intersection = sl.path_intersection(path1, path2)
            if intersection is not None:
                # Truncate one of the lines past the intersection
                shorten_path1 = False
                line_to_truncate = path1 if shorten_path1 else path2
                target_segment = intersection.path1_segment if line_to_truncate is path1 else intersection.path2_segment
                last_point = target_segment[0]
                last_index = line_to_truncate.index(last_point)
                del (line_to_truncate[last_index:])  # clear values past start of intersecting section
                line_to_truncate.append(intersection.intersection_point)  # Intersection point new endpoint

    # barbed wire
    barbed_wire_lines = []
    barbed_wire_lines.append(copy_line(front_line, 0, -20))

    barbed_wire_crosses = []
    for barbed_wire_line in barbed_wire_lines:
        barbed_wire_crosses = barbed_wire_crosses + apply_template(barbed_wire_line, Templates.CROSSES,
                                                                   params.wire_cross_length, params.wire_interval)

    # apply templates
    rear_lines.append(
        copy_line(front_line, 0, (params.template_dimensions / 2) + random.randrange(*params.supervision_range)))
    template = Templates.ZIGZAG if random.random() < params.zig_zag_chance else Templates.PULSE
    front_line = apply_template(front_line, template, params.template_dimensions * 0.75, params.template_dimensions)

    # draw lines
    draw_line(front_line, pixels, params.trench_color, img)

    for rear_trench in rear_lines:
        draw_line(rear_trench, pixels, params.trench_color, img)

    for comm_trench in comm_trenches:
        draw_line(comm_trench, pixels, params.trench_color, img)

    for wire_line in barbed_wire_crosses:
        draw_line(wire_line, pixels, params.trench_color, img)

    draw_line(friendly, pixels, BLUE, img)

    img.show()
    img.save("out/trench" + str(time.time()) + ".bmp")
    print("hey hey hey")


def find_map_edge(point, slope, direction, height, width):
    # gotta be a better way to do this
    map_edges = [[(0, 0), (width - 1, 0)], [(width - 1, 0), (width - 1, height - 1)],
                 [(width - 1, height - 1), (0, height - 1)], [(0, height - 1), (0, 0)]]
    for edge in map_edges:
        n = sl.line_intersection([point, sl.transpose(point, slope, 10000 * direction)], edge)
        if n is not None:
            return n
    Exception()


def copy_line(line, direction, distance, mutator=lambda p: 0):
    new_line = []
    for point in line:
        new_line.append(sl.transpose(point, direction, distance + mutator(point)))
    return new_line


def divide_line(line, start, end, end_length, jitter_max, depth_limit=math.inf, depth=0, jitter_decay=2):
    line_length = sl.line_length(start, end)
    if depth < depth_limit and line_length > end_length:
        # find midpoint
        deltaX = end[0] - start[0]
        deltaY = end[1] - start[1]
        midpoint = (start[0] + (deltaX / 2), start[1] + (deltaY / 2))
        # apply jitter
        normal_slope = sl.normal(deltaX, deltaY)
        jitter = random.randrange(round(-jitter_max), round(jitter_max)) if round(jitter_max) > 0 else 0
        jittered_midpoint = sl.transpose(midpoint, normal_slope, jitter)
        line.insert(line.index(start) + 1, jittered_midpoint)
        # recurse
        divide_line(line, start, jittered_midpoint, end_length, jitter_max / jitter_decay, depth_limit, depth + 1)
        divide_line(line, jittered_midpoint, end, end_length, jitter_max / jitter_decay, depth_limit, depth + 1)


def vector_test():
    dwg = svgwrite.Drawing('test.svg', profile='tiny')
    dwg.add(dwg.line((0, 0), (10, 0), stroke=svgwrite.rgb(10, 10, 16, '%')))
    dwg.add(dwg.text('Test', insert=(0, 2), fill='red'))
    dwg.save()


def test():
    img = Image.new('RGB', (255, 255), "black")  # Create a new black image
    pixels = img.load()  # Create the pixel map
    for i in range(img.size[0]):  # For every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = (i, j, 100)  # Set the colour accordingly
    img.show()


def template_test(trench_params=TrenchParams()):
    # draw bg
    img = Image.new('RGB', (trench_params.map_width, trench_params.map_height), "black")  # Create a new black image
    pixels = bg(img, trench_params.bg_color)

    # first trench now
    x_center = trench_params.map_width / 2.0

    start = (x_center, 0.0)
    end = (x_center, trench_params.map_height - 1.0)
    line = [start, end]
    zzl = apply_template(line, Templates.ZIGZAG, 10, 10)
    draw_line(zzl, pixels, trench_params.trench_color, img)
    draw_line(apply_template([(100, 100), (200, 200), (50, 300), (300, 250), (250, 200)], Templates.ZIGZAG, 12, 12),
              pixels, (200, 0, 200), img)
    draw_line(apply_template([(500, 250), (700, 250)], Templates.ZIGZAG, 5, 5), pixels, (0, 200, 0), img)
    draw_line(apply_template([(700, 300), (500, 300)], Templates.ZIGZAG, 5, 5), pixels, (0, 0, 200), img)
    draw_line([(700, 300), (500, 300)], pixels, (0, 0, 200), img)
    draw_line(apply_template([(450, 500), (450, 250)], Templates.ZIGZAG, 5, 10), pixels, (200, 100, 0), img)
    img.show()


def transpose_test(trench_params=TrenchParams()):
    # draw bg
    img = Image.new('RGB', (trench_params.map_width, trench_params.map_height), "black")  # Create a new black image
    pixels = bg(img, trench_params.bg_color)
    p0 = (400, 400)

    def go(p1, c):
        draw_line([p0, sl.transpose(p0, sl.slope(p0, p1), 100 * sl.line_direction(p0, p1))], pixels, c, img)

    go((500, 500), (255, 0, 0))
    go((500, 400), (222, 111, 0))
    go((500, 300), (222, 222, 0))
    go((400, 300), (0, 222, 0))
    go((300, 300), (0, 222, 222))
    go((300, 400), (0, 0, 222))
    go((300, 500), (222, 0, 222))
    go((400, 500), (0, 0, 0))
    img.show()


def draw_line(line, pixels, color, img):
    for i in range(0, len(line) - 1):
        draw(pixels, line[i], line[i + 1], color, img)


class Templates(Enum):
    PULSE = 0
    ZIGZAG = 1
    DOTS = 2
    CROSSES = 3


def apply_template(line, template, width, height):
    zig_zag_line = []
    i = 0
    zig_or_zag = 1
    last_placed = line[0]
    if template is not Templates.CROSSES:
        zig_zag_line.append(last_placed)
    carryover = 0
    while i + 1 < len(line):
        cur_point = line[i]
        next_point = line[i + 1]
        slope = sl.slope(cur_point, next_point)
        dxdy = sl.dx_dy(cur_point, next_point)
        normal = sl.normal(*dxdy)
        direction = sl.line_direction(cur_point, next_point)
        segment_length = sl.line_length(cur_point, next_point)
        dist = carryover
        while dist < segment_length:
            dist += height
            next_output_point = sl.transpose(cur_point, slope, dist * direction)
            displace_dist = (width / 2) * zig_or_zag
            if template is Templates.DOTS:
                zig_zag_line.append(next_output_point)
            elif template is Templates.PULSE:
                zig_zag_line.append(sl.transpose(last_placed, normal, displace_dist))
                zig_zag_line.append(sl.transpose(next_output_point, normal, displace_dist))
                zig_or_zag = zig_or_zag * -1
                zig_zag_line.append(next_output_point)
            elif template is Templates.ZIGZAG:
                midpoint = sl.midpoint(last_placed, next_output_point)
                midpoint = sl.transpose(midpoint, normal, displace_dist)
                zig_zag_line.append(midpoint)
                zig_zag_line.append(next_output_point)
                zig_or_zag = zig_or_zag * -1
            elif template is Templates.CROSSES:
                left_side = sl.transpose(next_output_point, normal, displace_dist * -1)
                right_side = sl.transpose(next_output_point, normal, displace_dist)
                zig_zag_line.append(
                    [sl.transpose(left_side, slope, displace_dist), sl.transpose(right_side, slope, displace_dist * -1)])
                zig_zag_line.append(
                    [sl.transpose(left_side, slope, displace_dist * -1), sl.transpose(right_side, slope, displace_dist)])
            else:
                Exception()

            last_placed = zig_zag_line[len(zig_zag_line) - 1]
        i += 1
        carryover = dist - segment_length
    return zig_zag_line


# Instead of applying a template, take a lambda and execute at each interval.
def for_interval_in_path(line, do, interval):
    out_line = []
    i = 0
    carryover = 0
    while i + 1 < len(line):
        cur_point = line[i]
        next_point = line[i + 1]
        slope = sl.slope(cur_point, next_point)
        dxdy = sl.dx_dy(cur_point, next_point)
        direction = sl.line_direction(cur_point, next_point)
        segment_length = sl.line_length(cur_point, next_point)
        dist = carryover
        while dist < segment_length:
            dist += interval
            point = sl.transpose(cur_point, slope, dist * direction)
            do(point, dxdy, out_line)
        i += 1
        carryover = dist - segment_length
    return out_line


def bg(img, color=(0, 0, 0)):
    pixels = img.load()  # Create the pixel map
    for i in range(img.size[0]):  # For every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = color  # Set the colour accordingly
    return pixels


def draw(pixels, p1, p2, color, img):
    dX = p2[0] - p1[0]
    dY = p2[1] - p1[1]
    margin = 0.0001

    # vertical special case
    if -margin < dX < margin:
        x = p1[0]
        y1 = int(p1[1])
        y2 = int(p2[1])
        y_sign = -1 if y2 < y1 else 1
        for y in range(y1, y2, y_sign):
            pixels[x, y] = color
        return pixels

    slope = dY / dX

    def f(a):
        deltaX = a - p1[0]
        deltaY = deltaX * slope
        return p1[1] + deltaY

    xStart = int(p1[0])
    xEnd = int(p2[0])
    x_sign = -1 if xStart > xEnd else 1

    for x in range(xStart, xEnd + x_sign, x_sign):
        # fill all points of line on this 'x'
        fx = int(f(x))
        fx1 = int(f(x + 0.999))
        y_sign = -1 if fx > fx1 else 1
        for y in range(fx, fx1 + y_sign, y_sign):
            if int(min(p1[1], p2[1])) <= y <= round(max(p1[1], p2[1])) and 0 <= x < img.size[0] and 0 <= y < img.size[
                1]:
                pixels[x, y] = color

    return pixels


def line_test():
    img = Image.new('RGB', (400, 400), "black")  # Create a new black image
    pixels = bg(img, (255, 255, 255))
    pixels = draw(
        draw(
            draw(pixels, (100, 100), (200, 200), BLACK, img), (100, 50), (200, 100), (0, 100, 0), img), (50, 100),
        (100, 200), (255, 0, 0), img)
    draw(pixels, (100, 200), (300, 200), (0, 0, 255), img)
    draw(pixels, (200, 100), (200, 300), (255, 0, 255), img)
    img.show()


line_trench(TrenchParams(distance_scaling=0.5, first_pass_depth=2))
# line_test()
# trench()
# vector_test()
# template_test()
# transpose_test()
