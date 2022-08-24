import numpy as np

from py3do.utils import UnionFind

def test_union_find():
    uf = UnionFind(5)
    assert uf.n == 5
    assert uf.n_elem == 5
    assert uf.find(1) == 1
    assert str(uf) == "{0},{1},{2},{3},{4}"
    uf.union(0,1)
    uf.union(1,4)
    assert uf.n == 3
    assert uf.n_elem == 5
    assert str(uf) == "{0,1,4},{2},{3}"
    uf.union(0,2)
    assert uf.n == 2
    assert uf.n_elem == 5
    assert str(uf) == "{0,1,2,4},{3}"
    se = uf.set_elements()
    assert len(se) == 2
    assert np.array_equal(uf.sets(), [0,0,0,1,0])
