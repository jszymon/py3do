from py3do import cube
from py3do import is_isomorphic

def test_rot90_identity():
    m = cube()
    m_orig = m.clone()
    m.rot90("xxxX")
    assert is_isomorphic(m, m_orig)
    m.rot90("yyyY")
    assert is_isomorphic(m, m_orig)
    m.rot90("zzzZ")
    assert is_isomorphic(m, m_orig)
