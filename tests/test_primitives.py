from py3do import cube, cone_pipe

# cylinder empty top/bottom
cone_pipe(1, 1, n=10)
# cylinder with top/bottom
cone_pipe(1, 1, n=10, close_top=True, close_bottom=True)

# need to flip normals:
cone_pipe(1, 1, 1, 1, 2, n=10, connect_top_bottom=True)
