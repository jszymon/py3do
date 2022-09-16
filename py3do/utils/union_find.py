"""The union-find data structure."""

import numpy as np


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
    def union(self, i, j):
        self.n = self.n-1
        i = self.find(i)
        self.parents[j] = i
        return i
    def sets(self):
        """Return the set each element belongs numbered consecutively
        from 0."""
        _ = self.find(np.arange(self.n_elem))
        return np.unique(self.parents, return_inverse=True)[1]
    def set_elements(self):
        """Return a list of arrays of elements of each set."""
        set_elems = []
        sets = self.sets()
        # split array into lists
        elem_idx = np.argsort(sets)
        sets = sets[elem_idx]
        set_idx = np.unique(sets, return_index=True)[1]
        prev_si = 0
        for si in set_idx[1:]:
            set_elems.append(elem_idx[prev_si:si])
            prev_si = si
        set_elems.append(elem_idx[si:])
        #for si in range(self.n):
        #    set_elems.append(np.nonzero(sets == si)[0])
        return set_elems
    def __str__(self):
        set_elems = self.set_elements()
        strs = ["{" + ",".join(str(i) for i in s) + "}" for s in set_elems]
        return ",".join(strs)
