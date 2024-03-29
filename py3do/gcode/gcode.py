"""Gcode parsing utility."""

from contextlib import nullcontext

import math
import numpy as np

def _numbered_line_reader(f):
    for li, l in enumerate(f):
        yield li+1, l.strip()
def _get_arg(args, a, conv=None):
    for ai in args:
        if ai.startswith(a):
            ai = ai[len(a):]
            if conv is not None:
                ai = conv(ai)
            return ai
    return None

def planArc(currentPos, targetPos, offset, clockwise,
            alpha_axis, beta_axis, helical_axis,
            mm_per_arc_segment=1):
    """Copied directly from klipper for compatibility"""
    # todo: sometimes produces full circles

    # Radius vector from center to current location
    r_P = -offset[0]
    r_Q = -offset[1]

    # Determine angular travel
    center_P = currentPos[alpha_axis] - r_P
    center_Q = currentPos[beta_axis] - r_Q
    rt_Alpha = targetPos[alpha_axis] - center_P
    rt_Beta = targetPos[beta_axis] - center_Q
    angular_travel = math.atan2(r_P * rt_Beta - r_Q * rt_Alpha,
                                r_P * rt_Alpha + r_Q * rt_Beta)
    if angular_travel < 0.:
        angular_travel += 2. * math.pi
    if clockwise:
        angular_travel -= 2. * math.pi

    if (angular_travel == 0.
        and currentPos[alpha_axis] == targetPos[alpha_axis]
        and currentPos[beta_axis] == targetPos[beta_axis]):
        # Make a circle if the angular rotation is 0 and the
        # target is current position
        angular_travel = 2. * math.pi

    # Determine number of segments
    linear_travel = targetPos[helical_axis] - currentPos[helical_axis]
    radius = math.hypot(r_P, r_Q)
    flat_mm = radius * angular_travel
    if linear_travel:
        mm_of_travel = math.hypot(flat_mm, linear_travel)
    else:
        mm_of_travel = math.fabs(flat_mm)
    segments = max(1., math.floor(mm_of_travel / mm_per_arc_segment))

    # Generate coordinates
    theta_per_segment = angular_travel / segments
    linear_per_segment = linear_travel / segments
    coords = []
    for i in range(1, int(segments)):
        dist_Helical = i * linear_per_segment
        cos_Ti = math.cos(i * theta_per_segment)
        sin_Ti = math.sin(i * theta_per_segment)
        r_P = -offset[0] * cos_Ti + offset[1] * sin_Ti
        r_Q = -offset[0] * sin_Ti - offset[1] * cos_Ti

        # Coord doesn't support index assignment, create list
        c = [None, None, None]
        c[alpha_axis] = center_P + r_P
        c[beta_axis] = center_Q + r_Q
        c[helical_axis] = currentPos[helical_axis] + dist_Helical
        coords.append(c)

    coords.append(targetPos)
    return coords

class GCode:
    def __init__(self, f, mm_per_arc_segment=1):
        self.parse_gcode(f)
        self.pos = np.array(self.pos)
        # index line number -> pos array
        self.idx = np.full(self.n_lines, -1)
        prev_li = 0
        for i, li in enumerate(self.line_nos):
            self.idx[prev_li:li] = i-1
            prev_li = li
        for j in range(li, self.n_lines):
            self.idx[j] = i
        self.split_layers()

    def split_layers(self):
        self.layers = list()
        self.layers.append(0) # 0-th layer contains initial moves and purge line
        if self.slicer != "Cura":
            print("Splitting into layers only implemented for Cura gcode")
            return
        for li, ev in self.events:
            if ev[0] == "Layer Start":
                self.layers.append(self.idx[li])
                print(li, self.idx[li])
            elif ev[0] == "End Main Print":
                self.layers.append(self.idx[li])
        self.n_layers = len(self.layers)
        self.layers.append(self.pos.shape[0])
    def parse_gcode(self, f):
        def _make_move():
            nonlocal x, y, z, e
            nonlocal feed_rate
            dist = np.sqrt((next_coord[0] - x)**2 + (next_coord[1] - y)**2 + (next_coord[2] - z)**2)
            de = next_coord[3] - e
            if feed_rate is None:
                dt = None
            else:
                if dist == 0:
                    dt = abs(de) / feed_rate * 60
                else:
                    dt = dist / feed_rate * 60
            if de == 0 and cmd[1] == "1":
                self.events.append((li, ["Warning: G1 move without extrusion"]))
                print(li, "Warning: G1 move without extrusion", len(self.pos))
            if de != 0 and cmd[1] == "0":
                self.events.append((li, ["Warning: G0 move with extrusion"]))
            # make the move
            x, y, z, e = next_coord
            self.line_nos.append(li)
            self.pos.append([x, y, z])
            self.extruder.append(e)
            self.dts.append(dt)
            self.dists.append(dist)
            self.des.append(de)

        if hasattr(f, 'read'):
            f_ctx = nullcontext(f)
        else:
            f_ctx = open(f, 'r')
        # state
        self.slicer = None
        x = y = z = 100.0
        e = 0.0
        feed_rate = None
        rel_extrude = False
        rel_coord = False
        speed_factor = 1.0
        extrude_factor = 1.0
        x_offset = 0.0 # offsets to handle G92
        y_offset = 0.0
        z_offset = 0.0
        e_offset = 0.0
        # end state
        self.events = []
        self.line_nos = [] # line numbers for each position 
        self.pos = [] # head positions
        self.extruder = [] # extruder position
        self.dts = []
        self.ds = []
        self.dists = []
        self.des = [] # extruder distances
        with f_ctx as fl:
            for li, l in _numbered_line_reader(fl):
                l = l.strip()
                if l == "":
                    continue
                # remove comments
                _split_line = l.split(";", maxsplit=1)
                cmd = _split_line[0].upper()
                cmt = _split_line[1] if len(_split_line) > 1 else None
                if cmt is not None and cmt.startswith("Generated with Cura_SteamEngine"):
                    self.slicer = "Cura"
                # parse Cura comments
                if self.slicer == "Cura" and cmt is not None:
                    if cmt.startswith("LAYER:"):
                        layer_no = int(cmt[6:])
                        self.events.append((li, ["Layer Start", layer_no]))
                    #if cmt.startswith("TIME_ELAPSED:"):
                    #    self.events.append((li, ["End Main Print", layer_no]))                        
                    if cmt.startswith("TYPE:"):
                        self.events.append((li, ["Start", cmt[5:]]))
                    if cmt.startswith("MESH:"):
                        pass
                # parse command
                if cmd == "":
                    continue
                _split_cmd = cmd.split()
                cmd = _split_cmd[0]
                args = _split_cmd[1:]
                if cmd == "G28":
                    if len(args) == 0:
                        next_coord = [0, 0, 0, e]
                    else:
                        next_coord = [x, y, z, e]
                        if "X" in args:
                            next_coord[0] = 0
                        if "Y" in args:
                            next_coord[1] = 0
                        if "Z" in args:
                            next_coord[2] = 0
                    _make_move()
                elif cmd == "M82":
                    rel_extrude = False
                elif cmd == "M83":
                    rel_extrude = True
                elif cmd == "G90":
                    rel_coord = False
                elif cmd == "G91":
                    rel_coord = True
                elif cmd == "M220": # speed factor
                    a = _get_arg(args, "S", float)
                    if a:
                        speed_factor = a / 100
                elif cmd == "M221": # extrude factor
                    a = _get_arg(args, "S", float)
                    if a:
                        extrude_factor = a / 100
                elif cmd == "G92":
                    for a in args:
                        f = float(a[1:])
                        if a[0] == "E":
                            e_offset += e - f
                            e = f
                        elif a[0] == "X":
                            x_offset += x - f
                            x = f
                        elif a[0] == "Y":
                            y_offset += y - f
                            y = f
                        elif a[0] == "Z":
                            z_offset += z - f
                            z = f
                        else:
                            print("Unknown G92 coordinate")
                elif cmd == "G0" or cmd == "G1":
                    if rel_coord:
                        next_coord = [0, 0, 0, e]
                    else:
                        next_coord = [x, y, z, e]
                    if rel_extrude:
                        next_coord[3] = 0
                    # parse args
                    for a in args:
                        ci = "XYZE".find(a[0])
                        if ci > -1:
                            move_made = True
                            next_coord[ci] = float(a[1:])
                        elif a[0] == "F":
                            feed_rate = float(a[1:])
                        else:
                            raise RuntimeError("Wrong argument for G0/G1: " + a[0])
                    # compute new coords and move time
                    if rel_coord:
                        next_coord[0] += x
                        next_coord[1] += y
                        next_coord[2] += z
                    if rel_extrude:
                        next_coord[3] += e
                    if move_made:
                        _make_move()
                elif cmd == "G2" or cmd == "G3":
                    if rel_coord:
                        self.events.append((li, ["Warning: G2/G3 do not support relative coordinates", cmd, args]))
                        continue
                    #self.events.append((li, ["Warning: Unimplemented arc Gcode", cmd, args]))
                    clockwise = (cmd[1] == "2")
                    next_coord = [x, y, z, e]
                    I = J = None
                    # parse args
                    for a in args:
                        ci = "XYZE".find(a[0])
                        if ci > -1:
                            move_made = True
                            next_coord[ci] = float(a[1:])
                        elif a[0] == "F":
                            feed_rate = float(a[1:])
                        elif a[0] == "I":
                            I = float(a[1:])
                        elif a[0] == "J":
                            J = float(a[1:])
                        else:
                            raise RuntimeError("Wrong argument for G2/G3: " + a[0])
                    de = next_coord[3] - e
                    # compute arc coords
                    if I is None or J is None:
                        raise RuntimeError("Arc center not provided")
                    coords = planArc([x, y, z], next_coord[:3], [I, J, 0], clockwise, 0, 1, 2)
                    de_move = de / len(coords)
                    for c in coords:
                        next_coord = c + [e + de_move]
                        _make_move()                        
                else:
                    self.events.append((li, ["Warning: Unhandled Gcode", cmd, args]))
            self.n_lines = li
    # plot:
    # 3d
    # layers
    # move lengths
    # move length histograms
    # split into layers for non-Cura gcode
