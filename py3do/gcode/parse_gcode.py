"""Gcode parsing utility."""

from contextlib import nullcontext

def _numbered_line_reader(f):
    for li, l in enumerate(f):
        yield li+1, l.strip()

def parse_gcode(f):
    if hasattr(f, 'read'):
        f_ctx = nullcontext(f)
    else:
        f_ctx = open(f, 'r')
    # state
    x = y = z = e = None
    fr = None # feed rate
    rel_extrude = False
    rel_coord = False
    with f_ctx as fl:
        for li, l in _numbered_line_reader(fl):
            l = l.strip()
            if l == "":
                continue
            # remove comments
            _split_line = l.split(";", maxsplit=1)
            cmd = _split_line[0].upper()
            cmt = _split_line[1] if len(_split_line) > 1 else None
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
                pass
            elif cmd == "M221": # extrude factor
                pass
            
            
