from contextlib import nullcontext

def _numbered_line_reader(f):
    for li, l in enumerate(f):
        yield li+1, l.strip()

def _match_line(f, match):
    for li, l in f:
        if li != "": break
    else:
        raise RuntimeError("Unexpected end of file, expected " + str(match))
    if l != match:
        raise RuntimeError("Expected " + str(match) + " on line " + str(li))
def _parse_vector(f, match, raise_on_nonmatch=True):
    for li, l in f:
        if li != "": break
    else:
        if raise_on_nonmatch:
            raise RuntimeError("Unexpected end of file, expected " + str(match))
        return None, li, l
    if not l.startswith(match):
        if raise_on_nonmatch:
            raise RuntimeError("Expected '" + str(match) + "' on line " + str(li))
        return None, li, l
    toks = l[len(match):].strip().split()
    if len(toks) != 3:
        raise RuntimeError("A vector must have exactly 3 coordinates"
                               " (line " + str(li) + ")")
    vec = [float(x) for x in toks]
    return vec, li, l

def read_ascii_stl(fname):
    if hasattr(fname, 'read'):
        f_ctx = nullcontext(fname)
    else:
        f_ctx = open(fname, 'r')
    with f_ctx as fl:
        f = _numbered_line_reader(fl)
        li, header = next(f)
        if header[0:6].lower() != "solid ":
            raise RuntimeError("Wrong ASCII STL header")
        name = header[6:].strip()
        print("read stl", name)
        while True:
            normal, li, l = _parse_vector(f, "facet normal",
                                              raise_on_nonmatch=False)
            if normal is None and l.startswith("endsolid"):
                if name != l[9:]:
                    print("Warning: different names in 'solid'"
                                           " and 'endsolid'")
                break
            _match_line(f, "outer loop")
            v1, li, l = _parse_vector(f, "vertex")
            v2, li, l = _parse_vector(f, "vertex")
            v3, li, l = _parse_vector(f, "vertex")
            _match_line(f, "endloop")
            _match_line(f, "endfacet")
            print("facet")
            print("  normal", normal)
            print("  v1", v1)
            print("  v2", v2)
            print("  v3", v3)
        else:
            raise RuntimeError("Missing 'endsolid'")
        for li, l in f:
            if l != "":
                raise RuntimeError("Content after 'endsolid'")

