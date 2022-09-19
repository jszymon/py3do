import numpy as np

def arg_split(set_ids, n):
    """Return a list of arrays of indices of elements of set_ids equal to i for each 0<=i<n set_ids."""
    set_elems = []
    elem_idx = np.argsort(set_ids)
    set_idx = np.searchsorted(set_ids, np.arange(n+1), sorter=elem_idx)
    for i in range(n):
        set_elems.append(elem_idx[set_idx[i]:set_idx[i+1]])
    return set_elems
