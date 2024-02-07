"""The union-find data structure."""

import numpy as np

from .array_utils import arg_split

class UnionFind:
    def __init__(self, n):
        """Initialize to n singletons."""
        self.n_elem = n
        self.n = n
        self.parents = np.arange(n)
    def find(self, i):
        # skip bound checking for speed
        ip = self.parents[i]
        if (i != ip).any():
            ip = self.find(ip)
            self.parents[i] = ip
        return ip
    def _find_scalar(self, i):
        """Fast version for scalars, avoids call to .any()"""
        # skip bound checking for speed
        ip = self.parents[i]
        if i != ip:
            ip = self._find_scalar(ip)
            self.parents[i] = ip
        return ip
    def union(self, i, j):
        # skip bound checking for speed
        i = self._find_scalar(i)
        j = self._find_scalar(j)
        if i != j:
            self.n = self.n-1
            self.parents[j] = i
        return i
    def sets(self):
        """Return the set each element belongs numbered consecutively
        from 0."""
        _ = self.find(np.arange(self.n_elem))
        return np.unique(self.parents, return_inverse=True)[1]
    def set_elements(self):
        """Return a list of arrays of elements of each set."""
        sets = self.sets()
        return arg_split(sets, self.n)
    def __str__(self):
        set_elems = self.set_elements()
        strs = ["{" + ",".join(str(i) for i in s) + "}" for s in set_elems]
        return ",".join(strs)
