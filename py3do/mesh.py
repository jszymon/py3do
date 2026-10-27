"""Triangular mesh."""

import copy

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

    def split_edges(self, edges):
        """Split given edges in the middle.

        edges is a k x 2 array-like of vertex index pairs.  A new
        vertex is added at the midpoint of each edge and all faces
        adjacent to it are split.  Edge orientation and repeated
        edges are ignored.  A face with several split edges is split
        into 3 or 4 triangles.  New faces inherit normals of the
        original faces.  The first part of each split face keeps its
        original index, the remaining parts are appended at the end.

        Returns an array with the index of the new vertex for each
        given edge.  Pairs which are not edges of the mesh result in
        unused vertices.

        """
        n = self.vertices.shape[0]
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if edges.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if ((edges < 0) | (edges >= n)).any():
            raise ValueError("split_edges: vertex indices out of bounds")
        # edges as sorted (i,j) records to allow for lexicographic
        # sorting and searching
        pair_dt = np.dtype([("i", np.int64), ("j", np.int64)])
        def as_pairs(e1, e2):
            p = np.empty(e1.shape, dtype=pair_dt)
            p["i"] = np.minimum(e1, e2)
            p["j"] = np.maximum(e1, e2)
            return p
        # add midpoints of unique edges
        upairs, inv = np.unique(as_pairs(edges[:,0], edges[:,1]),
                                return_inverse=True)
        mids = (self.vertices[upairs["i"]] + self.vertices[upairs["j"]]) / 2
        # find midpoints of face edges, edge k is f[k] -- f[k+1]
        f = self.faces.astype(np.int64)
        fpairs = as_pairs(f, np.roll(f, -1, axis=1))
        pos = np.minimum(np.searchsorted(upairs, fpairs), len(upairs) - 1)
        found = (upairs[pos] == fpairs)
        fmid = n + pos
        n_split = found.sum(axis=1)
        # rotate faces such that for one split edge it is edge 0, and
        # for two split edges the unsplit edge is edge 2
        r = np.zeros(f.shape[0], dtype=np.int64)
        r[n_split == 1] = np.argmax(found[n_split == 1], axis=1)
        r[n_split == 2] = (np.argmin(found[n_split == 2], axis=1) + 1) % 3
        idx = (np.arange(3)[None,:] + r[:,None]) % 3
        f = np.take_along_axis(f, idx, axis=1)
        fmid = np.take_along_axis(fmid, idx, axis=1)
        a, b, c = f.T.copy()  # copy since f is modified below
        m0, m1, m2 = fmid.T
        # replace split faces and append new ones
        new_faces = [f]
        new_parents = []
        def add(mask, *tris):
            f[mask] = np.column_stack(tris[0])[mask]
            for t in tris[1:]:
                new_faces.append(np.column_stack(t)[mask])
                new_parents.append(np.nonzero(mask)[0])
        add(n_split == 1, (a, m0, c), (m0, b, c))
        add(n_split == 2, (a, m0, m1), (m0, b, m1), (a, m1, c))
        add(n_split == 3, (a, m0, m2), (m0, b, m1), (m2, m1, c), (m0, m1, m2))
        self.vertices = np.concatenate([self.vertices, mids])
        self.faces = np.concatenate(new_faces).astype(self.faces.dtype)
        self.normals = np.concatenate([self.normals]
                                      + [self.normals[p] for p in new_parents])
        return n + inv

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
