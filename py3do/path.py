"""The Path class."""

import numpy as np

class Path:
    """A connected path."""
    def __init__(self, start_x=0, start_y=0, start_angle=0, closed=True):
        self.points = [(start_x, start_y)]
        self.angle = start_angle
        self.closed = closed
    def copy(self):
        p = Path()
        p.points = self.points[:]
        p.angle = self.angle
    def to(self, x=None, y=None, dx=None, dy=None):
        px, py = self.points[-1]
        if x is None:
            x = px + dx
        else:
            dx = x - px
        if y is None:
            y = py + dy
        else:
            dy = y - py
        self.points.append((x, y))
        self.angle = np.arctan2(dy, dx)
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
    def r(self, r):
        """Round last corner."""
        return self
