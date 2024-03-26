"""Gcode parsing utility."""

from contextlib import nullcontext

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

class GCode:
    def __init__(self, f):
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
        if hasattr(f, 'read'):
            f_ctx = nullcontext(f)
        else:
            f_ctx = open(f, 'r')
        # state
        self.slicer = None
        x = y = z = None
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
                move_made = False
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
                        x = y = z = 0
                    else:
                        if "X" in args:
                            x = 0
                        if "Y" in args:
                            y = 0
                        if "Z" in args:
                            z = 0
                    move_made = True
                    dt = 5 # arbitrary value
                    dist = 5 # arbitrary value
                    de = 0
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
                        a_coord = [0, 0, 0]
                        a_e = 0
                    else:
                        a_coord = [x, y, z]
                        a_e = e
                    # parse args
                    for a in args:
                        ci = "XYZ".find(a[0])
                        if ci > -1:
                            move_made = True
                            a_coord[ci] = float(a[1:])
                        elif a[0] == "E":
                            move_made = True
                            a_e = float(a[1:])
                        elif a[0] == "F":
                            feed_rate = float(a[1:])
                    # compute new coords and move time
                    if not rel_coord:
                        a_coord[0] -= x
                        a_coord[1] -= y
                        a_coord[2] -= z
                    d = a_coord
                    if rel_extrude:
                        de = a_e
                    else:
                        de = a_e - e
                    x += d[0]
                    y += d[1]
                    z += d[2]
                    e += de
                    # compute move parameters
                    dist = np.sqrt(sum(c*c for c in d))
                    if dist == 0:
                        dt = abs(de) / feed_rate * 60
                    else:
                        dt = dist / feed_rate * 60
                    if de == 0 and cmd[1] == "1":
                        self.events.append((li, ["Warning: G1 move without extrusion"]))
                        print(li, "Warning: G1 move without extrusion", len(self.pos))
                    if de != 0 and cmd[1] == "0":
                        self.events.append((li, ["Warning: G0 move with extrusion"]))
                    #make_move(li, x, y, x, e, dt, dist, de)
                else:
                    self.events.append((li, ["Warning: Unhandled Gcode", cmd, args]))
                # add new extrusion point
                if move_made:
                    self.line_nos.append(li)
                    self.pos.append([x, y, z])
                    self.extruder.append(e)
                    self.dts.append(dt)
                    self.dists.append(dist)
                    self.des.append(de)
            self.n_lines = li
    # plot:
    # 3d
    # layers
    # move lengths
    # move length histograms
    # split into layers for non-Cura gcode
