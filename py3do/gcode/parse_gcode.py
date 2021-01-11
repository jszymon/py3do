"""Gcode parsing utility."""

from contextlib import nullcontext

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

def parse_gcode(f):
    if hasattr(f, 'read'):
        f_ctx = nullcontext(f)
    else:
        f_ctx = open(f, 'r')
    # state
    slicer = None
    x = y = z = e = None
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
    events = []
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
                slicer = "Cura"
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
                    if a[0] == "E":
                        e_offset = e
                        e = 0
                    elif a[0] == "X":
                        x_offset = e
                        x = 0
                    elif a[0] == "Y":
                        y_offset = e
                        y = 0
                    elif a[0] == "Z":
                        z_offset = e
                        z = 0
                    else:
                        print("Unknown G92 coordinate")
            elif cmd == "G0" or cmd == "G1":
                for a in args:
                    if a[0] == "X":
                        ax = float(a[1:])
                    elif a[0] == "Y":
                        ay = float(a[1:])
                    # ...............
                if rel_coord:
                    pass
            else:
                events.append((li, ["Unhandled Gcode", cmd, args]))
            # parse Cura comments
            if slicer == "Cura" and cmt is not None:
                if cmt.startswith("LAYER:"):
                    pass
                if cmt.startswith("TYPE:"):
                    pass
                if cmt.startswith("MESH:"):
                    pass

class GCode:
    def __init__(self, f):
        self.parsed = parse_gcode(f)
    # plot
    # split into layers
