"""The Path class."""

import numpy as np

from .geom_2d import two_segment_fillet

class Path:
    """A connected path."""
    def __init__(self, start_x=0, start_y=0, start_angle=0,
                 closed=True, round_r=None, round_n=None):
        """Create a path.

        If closed is True: the path will be automatically closed on
        render.

        round_r is the default rounding radius for corners.  If None
        no default rounding is done.

        """
        self.points = [(start_x, start_y)]
        self.angle = start_angle
        self.closed = closed
        self.round_r = round_r
        self.round_n = round_n
        self.corner_mods = {} # corner modifications, e.g. rounding
    def copy(self):
        p = Path()
        p.round_r = self.round_r
        p.round_n = self.round_n
        p.closed = self.closed
        p.points = self.points[:]
        p.angle = self.angle
        p.corner_mods = self.corner_mods.copy()
        return p
    def to(self, x=None, y=None):
        px, py = self.points[-1]
        if x is None:
            x = px
            dx = 0
        else:
            dx = x - px
        if y is None:
            y = py
            dy = 0
        else:
            dy = y - py
        self.points.append((x, y))
        self.angle = np.arctan2(dy, dx)
        return self
    def dxy(self, dx, dy):
        px, py = self.points[-1]
        self.points.append((x+dx, y+dy))
        self.angle = np.arctan2(dy, dx)
        return self
    def dx(self, dx):
        px, py = self.points[-1]
        x = px + dx
        self.points.append((x, py))
        self.angle = np.pi * (dx < 0)
        return self
    def dy(self, dy):
        px, py = self.points[-1]
        y = py + dy
        self.points.append((px, y))
        self.angle = np.pi/2 * np.sign(dy)
        return self

    def fd(self, l):
        """Move forward at current angle"""
        px, py = self.points[-1]
        x = px + l * np.cos(self.angle)
        y = py + l * np.sin(self.angle)
        self.points.append((x, y))
        return self
    def lt(self, a):
        self.angle += np.deg2rad(a)
        return self
    def rt(self, a):
        return self.lt(-a)
    def round(self, r):
        """Round last corner."""
        self.corner_mods[len(self.points)-1] = ("round", r)
        return self

    def render(self, round_n=None):
        """Render path: return coordinate list."""
        if round_n is None:
            round_n = self.round_n
        if round_n is None:
            round_n_args = []
        else:
            round_n_args = [round_n]
        rp = self.points[:]
        if self.closed:
            if rp[-1] != rp[0]:
                rp.append(rp[0])
        if self.round_r is None and len(self.corner_mods) == 0:
            return rp
        # apply rounding / chamfering etc.
        rp2 = []
        if self.closed:
            # round the closing point
            mod = self.corner_mods.get(0, ("round", self.round_r))
            assert mod[0] == "round"
            r = mod[1]
            if r is None:
                rp2.append(rp[0])
            else:
                xy, c, a = two_segment_fillet(*rp[-2], *rp[0], *rp[1], r, *round_n_args)
                rp2.extend(xy)
                rp[0] = xy[-1]
                rp[-1] = xy[0]
        else:
            rp2.append(rp[0])
        for i in range(1, len(rp)-1):
            mod = self.corner_mods.get(i, ("round", self.round_r))
            assert mod[0] == "round"
            r = mod[1]
            if r is None:
                rp2.append(rp[i])
            else:
                xy, c, a = two_segment_fillet(*rp[i-1], *rp[i], *rp[i+1], r, *round_n_args)
                # TODO: test a=0
                rp2.extend(xy)
                rp[i] = xy[-1]
        rp2.append(rp[-1])
        return rp2
    def plot(self):
        import matplotlib.pyplot as plt
        pts = self.render()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys)
        plt.gca().set_aspect("equal")
        plt.show()
