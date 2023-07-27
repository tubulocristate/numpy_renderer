import numpy as np
from camera import Camera
from read_form_obj import read_from_obj
from render_triangle import to_homogeneous, rasterize_triangle_with_depth, show_image
from render_triangle import normalize_z, scale_points

WIDTH, HEIGHT = 256, 256
# cube = np.array(
#         [
#             [0, 0, 0], # 0
#             [1, 0, 0], # 1
#             [1, 1, 0], # 2
#             [0, 1, 0], # 3
#             [0, 0, 1], # 4
#             [1, 0, 1], # 5
#             [1, 1, 1], # 6
#             [0, 1, 1], # 7
#         ])
# faces = np.array(
#         [
#             [0, 1, 2],
#             [0, 2, 3],
#             [0, 4, 7],
#             [0, 7, 3],
#             [0, 4, 5],
#             [0, 5, 1],
#             [1, 5, 2],
#             [2, 5, 6],
#             [2, 6, 3],
#             [3, 7, 6],
#             [4, 5, 6],
#             [4, 6, 7]
#
#         ])
#
# colors = np.array(
#         [
#             [1., 0.5, 0.],
#             [0., 1., 0.],
#             [0., 0., 1.],
#             [0., 0., 1.],
#             [1., 0., 1.],
#             [1., 0., 0.],
#             [1., 0.5, 0.],
#             [0., 1., 0.],
#         ])

vertices, faces = read_from_obj("bench/model.obj")


def main(vertices, faces):
    camera_position = np.array([1, 2, -3])
    at = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    fov = 90
    z_near = 0.1
    z_far = 100


    camera = Camera(camera_position, at, up, fov, z_near, z_far)
    vertices = vertices - np.mean(vertices)
    cube_homogeneous = to_homogeneous(vertices)
    projected_cube = camera(cube_homogeneous)
    normalize_z(projected_cube)
    scale_points(projected_cube)

    canvas = np.zeros(shape=(256, 256, 3))
    z_buffer = np.zeros(shape=(256, 256, 3))
    z_buffer.fill(-float("inf"))

    for face in faces:
        to_rasterizer = projected_cube[face]
        canvas, z_buffer = rasterize_triangle_with_depth(to_rasterizer, canvas=canvas, z_buffer=z_buffer)
    show_image(canvas)
    return 0

if __name__ == "__main__":
    main(vertices, faces)
    print("DONE!")
