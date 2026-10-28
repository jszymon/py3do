"""Triangular mesh."""

import copy
import functools

import numpy as np

from .geom import normals_cross

def _as_3col(x, *, fl=True):
    """Convert x to 3 column array.

    Works for empty lists"""
    if len(x) == 0:
        return np.empty((0,3), dtype=np.double if fl else np.int64)
    if fl:
        x = np.asarray(x, dtype=float)
    else:
        x = np.asarray(x)
    return np.asarray(x)
def _check_points_array(x, name):
    """Make sure x is a 3 column matrix.

    name is used for error reporting"""
    if len(x.shape) != 2:
        raise RuntimeError(name + " must be a 2d array")
    if x.shape[1] != 3:
        raise RuntimeError(name + " must have 3 coordinates")
    if not np.isfinite(x).all():
        raise RuntimeError("Inf or Nan in " + name)

@functools.lru_cache
def _split_triangle_template(k0, k1, k2):
    """Triangulate a triangle with sides split into k0, k1, k2 parts.

    Vertices are numbered along the boundary starting from the first
    corner: the corner, then the k0-1 points on side 0, the second
    corner, etc.  Returns triangles in local vertex numbers.
    Triangles are cut off as ears, original corners first, then in
    round robin fashion, never producing degenerate triangles.

    """
    ks = (k0, k1, k2)
    # sides of the original triangle each vertex lies on
    sides = []
    for s in range(3):
        sides.append({(s - 1) % 3, s})
        sides.extend({s} for _ in range(ks[s] - 1))
    poly = list(range(len(sides)))
    tris = []
    def clip(p):
        """Cut off vertex at position p if possible."""
        u, v, w = poly[p - 1], poly[p], poly[(p + 1) % len(poly)]
        if sides[u] & sides[v] & sides[w]:
            return False  # collinear
        if set.intersection(*(sides[x] for x in poly if x != v)):
            return False  # remaining vertices collinear
        tris.append((u, v, w))
        del poly[p]
        return True
    for c in (0, k0, k0 + k1):
        if len(poly) > 3:
            clip(poly.index(c))
    p = 0
    while len(poly) > 3:
        p %= len(poly)
        if not clip(p):
            p += 1
    tris.append(tuple(poly))
    return tuple(tris)

@functools.lru_cache
def _grid_triangle_template(k):
    """Subdivide a triangle with all sides split into k parts into a
    regular grid of k^2 triangles.

    Boundary vertices are numbered as in _split_triangle_template,
    interior vertices follow.  Returns triangles in local vertex
    numbers and barycentric coordinates of interior vertices.

    """
    # grid point (i, j) is at a + i/k (b-a) + j/k (c-a)
    idx = {}
    for i in range(k + 1):
        idx[i, 0] = i  # side 0: a -> b
    for j in range(k + 1):
        idx[k - j, j] = k + j  # side 1: b -> c
    for j in range(1, k + 1):
        idx[0, j] = 3 * k - j  # side 2: c -> a
    weights = []
    for i in range(1, k):
        for j in range(1, k - i):
            idx[i, j] = 3 * k + len(weights)
            weights.append(((k - i - j) / k, i / k, j / k))
    tris = []
    for i in range(k):
        for j in range(k - i):
            tris.append((idx[i, j], idx[i + 1, j], idx[i, j + 1]))
            if i + j < k - 1:
                tris.append((idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]))
    return tuple(tris), tuple(weights)

class Mesh:
    def __init__(self, vertices, faces, /, normals=None, *,
                 fix_nan_normals=False):
        """Create a mesh.

        Setting fix_nan_normals=True replaces Nan's in normals with
        zeros.  Useful for meshes with bad normals, e.g. collinear
        triangles.
        """
        self.vertices = _as_3col(vertices, fl=True)
        self.faces = _as_3col(faces, fl=False)
        if normals is not None:
            self.normals = _as_3col(normals, fl=True)
        else:
            self.normals, _ = normals_cross(self)
        if fix_nan_normals:
            self.normals[np.isnan(self.normals)] = 0
        self.check_faces_and_vertices()
    def check_faces_and_vertices(self):
        """Basic checks of consistency of faces and vertices."""
        _check_points_array(self.vertices, "vertices")
        if not np.issubdtype(self.faces.dtype, np.integer):
            raise RuntimeError("faces must be an integer typed array, got "\
                               + str(self.faces.dtype))
        if len(self.faces.shape) != 2:
            raise RuntimeError("faces must be a 2d array")
        if self.faces.shape[1] != 3:
            raise RuntimeError("faces must have 3 vertices")
        if (self.faces < 0).any():
            raise RuntimeError("Negative indices in faces")
        n_vert = self.vertices.shape[0]
        if (self.faces >= n_vert).any():
            raise RuntimeError("faces indices out of bounds")
        _check_points_array(self.normals, "normals")

    def clone(self):
        return copy.deepcopy(self)

    def round_coords(self, decimals, *, inplace=False, merge=False):
        """Round coordinates of all points.

        merge identical points if requested."""
        if merge:
            raise NotImplementedError("round_coords: Merging is not implemented")
        if inplace:
            np.around(self.vertices, decimals, out=self.vertices)
            if self.normals is not None:
                self.normals, _ = normals_cross(self)
            return None
        return Mesh(np.around(self.vertices, decimals),
                    self.faces, self.normals)

    def extents(self):
        """Size of the bounding box."""
        M = self.vertices.max(axis=0)
        m = self.vertices.min(axis=0)
        return M - m

    def delete_edge(self, i, j):
        """Delete an edge i--j.

        vertex i is removed, all edges pointing to it now point at j.
        Results in an usused vertex.

        """
        i_mask = (self.faces == i).any(axis=1)
        j_mask = (self.faces == j).any(axis=1)
        f_mask = ~(i_mask & j_mask)
        self.faces = self.faces[f_mask]
        self.normals = self.normals[f_mask]
        self.faces[self.faces == i] = j

    def delete_vertices(self, vs):
        """Delete vertices with given numbers.

        Remaining vertices may be renumbered."""
        vs = np.unique(vs)
        # remove affected faces
        f_mask = ~np.isin(self.faces, vs).any(axis=1)
        self.faces = self.faces[f_mask]
        self.normals = self.normals[f_mask]
        # renumber vertices
        old_n = self.vertices.shape[0]
        v_mask = np.full(old_n, True)
        v_mask[vs] = False
        new_verts_idx = np.full(old_n, -1)
        new_verts_idx[v_mask] = np.arange(old_n - vs.shape[0])
        self.faces = new_verts_idx[self.faces]
        self.vertices = self.vertices[v_mask]

    def split_edges(self, edges, k=2, *, method="ear"):
        """Split given edges into k equal segments.

        edges is an m x 2 array-like of vertex index pairs.  k is
        either a single number or a sequence with the number of
        segments for each edge.  k-1 new vertices are added on each
        edge and all faces adjacent to it are split.  Edge
        orientation is ignored, repeated edges must have equal k.
        New faces inherit normals of the original faces.  The first
        part of each split face keeps its original index, the
        remaining parts are appended at the end.

        method determines how split faces are triangulated:

        "ear": the polygon formed by face corners and new edge
        vertices is triangulated without adding interior vertices.
        Triangles may become long and thin for large k.

        "grid": faces with all three sides split into the same k
        parts are subdivided into a regular grid of k^2 triangles,
        adding (k-1)(k-2)/2 interior vertices per face.  Other faces
        are triangulated as in "ear".

        Returns an array with the index of the first new vertex of
        each given edge.  The k-1 vertices of an edge (i, j) have
        consecutive indices and are ordered from min(i, j) to max(i,
        j).  Interior vertices are added after all edge vertices.
        Pairs which are not edges of the mesh result in unused
        vertices.

        """
        if method not in ("ear", "grid"):
            raise ValueError("split_edges: unknown method " + repr(method))
        n = self.vertices.shape[0]
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        k = np.broadcast_to(np.asarray(k, dtype=np.int64), edges.shape[:1])
        if edges.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if ((edges < 0) | (edges >= n)).any():
            raise ValueError("split_edges: vertex indices out of bounds")
        if (k < 1).any():
            raise ValueError("split_edges: k must be at least 1")
        # edges as sorted (i,j) records to allow for lexicographic
        # sorting and searching
        pair_dt = np.dtype([("i", np.int64), ("j", np.int64)])
        def as_pairs(e1, e2):
            p = np.empty(e1.shape, dtype=pair_dt)
            p["i"] = np.minimum(e1, e2)
            p["j"] = np.maximum(e1, e2)
            return p
        upairs, uidx, inv = np.unique(as_pairs(edges[:,0], edges[:,1]),
                                      return_index=True, return_inverse=True)
        uk = k[uidx]
        if (uk[inv] != k).any():
            raise ValueError("split_edges: different k for repeated edge")
        # add new vertices of unique edges
        n_new = uk - 1
        ustart = n + np.cumsum(n_new) - n_new
        e = np.repeat(np.arange(len(upairs)), n_new)
        t = ((np.arange(len(e)) + n - ustart[e] + 1) / uk[e])[:,None]
        new_vertices = (self.vertices[upairs["i"][e]] * (1 - t)
                        + self.vertices[upairs["j"][e]] * t)
        # find split face edges, edge s is f[s] -- f[s+1]
        f = self.faces.astype(np.int64)
        f_next = np.roll(f, -1, axis=1)
        fpairs = as_pairs(f, f_next)
        pos = np.minimum(np.searchsorted(upairs, fpairs), len(upairs) - 1)
        fk = np.where(upairs[pos] == fpairs, uk[pos], 1)
        fstart = ustart[pos]
        fwd = (f < f_next)
        # split faces with the same pattern of k's at once
        out = f.copy()
        new_faces = [out]
        new_parents = []
        new_vertices = [new_vertices]
        n_vert = n + n_new.sum()
        patterns, pat_inv = np.unique(fk, axis=0, return_inverse=True)
        pat_inv = pat_inv.reshape(-1)
        for p, ks in enumerate(patterns):
            if (ks == 1).all():
                continue
            fi = np.nonzero(pat_inv == p)[0]
            # polygon formed by corners and new vertices on sides
            cols = []
            for s in range(3):
                cols.append(f[fi, s, None])
                j = np.arange(ks[s] - 1)
                st = fstart[fi, s, None]
                cols.append(np.where(fwd[fi, s, None], st + j,
                                     st + ks[s] - 2 - j))
            if method == "grid" and (ks == ks[0]).all():
                template, weights = _grid_triangle_template(int(ks[0]))
                # add interior vertices
                weights = np.array(weights).reshape(-1, 3)
                n_int = len(weights)
                corners = self.vertices[f[fi]]
                new_vertices.append(np.einsum("pc,fcx->fpx", weights,
                                              corners).reshape(-1, 3))
                interior = n_vert + np.arange(len(fi) * n_int)
                cols.append(interior.reshape(len(fi), n_int))
                n_vert += len(fi) * n_int
            else:
                template = _split_triangle_template(*ks)
            poly = np.concatenate(cols, axis=1)
            tris = poly[:, template]
            out[fi] = tris[:, 0]
            new_faces.append(tris[:, 1:].reshape(-1, 3))
            new_parents.append(np.repeat(fi, tris.shape[1] - 1))
        self.vertices = np.concatenate([self.vertices] + new_vertices)
        self.faces = np.concatenate(new_faces).astype(self.faces.dtype)
        self.normals = np.concatenate([self.normals]
                                      + [self.normals[p] for p in new_parents])
        return ustart[inv]

    def merge(self, other):
        """Merge another mesh into this one in place.

        Vertices, faces and normals of other are appended.  No
        union is performed: vertices are not shared and
        intersections are not resolved.

        Returns the offset added to vertex indices of other.

        """
        offset = self.vertices.shape[0]
        self.vertices = np.concatenate([self.vertices, other.vertices])
        self.faces = np.concatenate([self.faces,
                                     other.faces + offset]).astype(self.faces.dtype)
        self.normals = np.concatenate([self.normals, other.normals])
        return offset

    def rot90(self, rotations=""):
        """Rotate model in place clockwise 90 degrees in turn around
        given axes.

        rotations must be a string of xyzXYZ.  For example "XY"
        rotates first around the X axis, then around the Y axis.

        """
        for axis in rotations.upper():
            ai = "XYZ".find(axis)
            if ai < 0:
                raise ValueError("Error: rotations argument must be a string of XYZxyz")
            a0 = 1 if ai == 0 else 0
            a1 = 1 if ai == 2 else 2
            self.vertices[:,(a0,a1)] = self.vertices[:,(a1,a0)]
            self.vertices[:,a1] = -self.vertices[:,a1]

    def get_submesh(self, vertex_mask):
        """Return a submesh formed of vertices selected with
        vertex_mask.

        The submesh contains only faces for which all vertices are in
        the submesh.

        Also returned is an index which allows for mapping submesh
        vertices back to original mesh.  This functionality is useful
        for applying transformations only to subset of a mesh.

        """
        # compute maps between vertices
        n = self.vertices.shape[0]
        map_submesh_mesh = np.arange(n)[vertex_mask]
        n_sub = len(map_submesh_mesh)
        map_mesh_submesh = np.full(n, -1)
        map_mesh_submesh[map_submesh_mesh] = np.arange(n_sub)
        submesh_faces = map_mesh_submesh[self.faces]
        # only include faces for which all vertices are present
        faces_mask = np.all(submesh_faces > -1, axis=1)
        submesh_faces = submesh_faces[faces_mask]
        submesh = Mesh(self.vertices[vertex_mask], submesh_faces,
                       normals=self.normals[faces_mask])
        return submesh, map_submesh_mesh

    def set_submesh_vertices(self, submesh, map_submesh_mesh):
        self.vertices[map_submesh_mesh] = submesh.vertices
        # recompute normals
        self.normals, _ = normals_cross(self)
