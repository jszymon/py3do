"""The Path class."""

import numpy as np

class Path:
    """A connected path."""
    def __init__():
        self.points = []
        self.angle = 0
    def copy(self)
        p = Path()
        p.points = self.points[:]
        p.angle = self.angle
    def to(self, x=None, y=None, dx=None, dy=None):
        if len(self.points) > 0:
            px, py = self.points[-1]
        if x is None:
            x = px + dx
        else:
            dx = x - px
        if y is None:
            y = py + dy
        self.points.append(x, y)
        self.angle = np.arctan2(dy, dx)
    def forward(self, l):
        """Move forward at current angle"""
        pass
