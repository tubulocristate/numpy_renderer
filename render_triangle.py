import numpy as np
import matplotlib.pyplot as plt


def all_positive_or_negative(a, b, c):
    all_positive = a > 0 and b > 0 and c > 0
    all_negative = a < 0 and b < 0 and c < 0
    return all_positive or all_negative


def is_inside(a, b, c):
    all_positive = np.logical_and(np.logical_and(a >= 0, b >= 0), c >= 0)
    all_negative = np.logical_and(np.logical_and(a < 0, b < 0), c < 0)
    return np.logical_or(all_positive, all_negative)


def make_perspective_matrix(fow: float, aspect: float, znear: float, zfar: float):
    perspective_matrix = np.zeros(shape=(4, 4))
    perspective_matrix[0][0] = aspect * (1 / np.tan(fow / 2))
    perspective_matrix[1][1] = 1 / np.tan(fow / 2)
    perspective_matrix[2][2] = zfar / (zfar - znear)
    perspective_matrix[2][3] = -(zfar * znear) / (zfar - znear)
    perspective_matrix[3][2] = 1.0
    return perspective_matrix


def make_perspective(perspective_matrix, vectors):
    perspective = perspective_matrix @ vectors
    perspective = np.where(perspective[-1] != 0, perspective / perspective[-1], perspective)
    return perspective

def normalize_z(points):
    min_z = np.min(points[:, 2])
    points[:, 2] -= min_z - 0.001
    max_z = np.max(points[:, 2])
    min_z = np.min(points[:, 2])
    points[:, 2] *= (1/(max_z+min_z))

def scale_points(points):
    scale_x = 64 / np.max(points[:, 0])
    scale_y = 64 / np.max(points[:, 1])
    points[:, 0] *= scale_x
    points[:, 1] *= scale_y

def rasterize_triangle_Z(vertices, colors, canvas, z_buffer):
    WIDTH, HEIGHT, CHANNELS = canvas.shape

    local_z_buffer = np.zeros(shape=(WIDTH, HEIGHT, CHANNELS))
    local_z_buffer.fill(-float("inf"))

    upper_z, left_z, right_z = vertices[1, 2], vertices[0, 2], vertices[2, 2]
    vertices = vertices[:, :2]

    center = [int(WIDTH / 2), int(HEIGHT / 2)]
    min_x = int(np.floor(np.min(np.min(vertices, axis=0))))
    min_y = int(np.floor(np.min(np.min(vertices, axis=1))))
    max_x = int(np.ceil(np.max(np.max(vertices, axis=0))))
    max_y = int(np.ceil(np.max(np.max(vertices, axis=1))))

    upper, left, right = vertices[1], vertices[0], vertices[2]
    upper_color, left_color, right_color = colors[1], colors[0], colors[2]
    AB = upper - left
    BC = right - upper
    CA = left - right

    coordinates = np.array(np.meshgrid(np.arange(min_x, max_x),
                                       np.arange(min_y, max_y))).T.reshape(-1, 2)

    cross = lambda point, edge: np.cross(point, edge)
    area = np.cross(right - left, upper - left)

    mask = is_inside(cross(coordinates - upper, BC), cross(coordinates - right, CA), cross(coordinates - left, AB))

    triangle_points = coordinates[mask]
    n_points, _ = triangle_points.shape
    alphas = np.expand_dims(np.cross(triangle_points - upper, BC) / area, axis=1)
    betas = np.expand_dims(np.cross(triangle_points - right, CA) / area, axis=1)
    gamas = np.expand_dims(np.cross(triangle_points - left, AB) / area, axis=1)

    upper_color = np.broadcast_to(upper_color, shape=(n_points, 3))
    left_color = np.broadcast_to(left_color, shape=(n_points, 3))
    right_color = np.broadcast_to(right_color, shape=(n_points, 3))

    x_coord = np.clip(triangle_points[:, 0], -center[1], center[1]) + center[0] - 1
    y_coord = np.clip(-triangle_points[:, 1], -center[0], center[0]) - center[1]
    local_z_buffer[y_coord, x_coord] = left_z * alphas + upper_z * betas + right_z * gamas
    color_value = left_color * alphas + upper_color * betas + right_color * gamas
    canvas[y_coord, x_coord] = np.where(z_buffer[y_coord, x_coord] < local_z_buffer[y_coord, x_coord], color_value,
                                        canvas[y_coord, x_coord])
    # canvas[y_coord, x_coord] = left_color*alphas + upper_color*betas + right_color*gamas
    z_buffer = np.where(z_buffer < local_z_buffer, local_z_buffer, z_buffer)

    return canvas, z_buffer


def rasterize_triangle(vertices, color, canvas):
    WIDTH, HEIGHT, CHANNELS = canvas.shape

    vertices = vertices[:, :2]

    center = [int(WIDTH / 2), int(HEIGHT / 2)]
    min_x = int(np.floor(np.min(np.min(vertices, axis=0))))
    min_y = int(np.floor(np.min(np.min(vertices, axis=1))))
    max_x = int(np.ceil(np.max(np.max(vertices, axis=0))))
    max_y = int(np.ceil(np.max(np.max(vertices, axis=1))))

    upper, left, right = vertices[1], vertices[0], vertices[2]
    AB = upper - left
    BC = right - upper
    CA = left - right

    coordinates = np.array(np.meshgrid(np.arange(min_x, max_x),
                                       np.arange(min_y, max_y))).T.reshape(-1, 2)

    cross = lambda point, edge: np.cross(point, edge)

    mask = is_inside(cross(coordinates - upper, BC), cross(coordinates - right, CA), cross(coordinates - left, AB))

    triangle_points = coordinates[mask]
    n_points, _ = triangle_points.shape

    x_coord = np.clip(triangle_points[:, 0], -center[1], center[1]) + center[0] - 1
    y_coord = np.clip(-triangle_points[:, 1], -center[0], center[0]) - center[1]
    canvas[y_coord, x_coord] = color
    return canvas

def rasterize_triangle_with_depth(vertices, canvas, z_buffer):
    WIDTH, HEIGHT, CHANNELS = canvas.shape

    local_z_buffer = np.zeros(shape=(WIDTH, HEIGHT, CHANNELS))
    local_z_buffer.fill(-float("inf"))

    upper_z, left_z, right_z = vertices[1, 2], vertices[0, 2], vertices[2, 2]
    vertices = vertices[:, :2]

    center = [int(WIDTH / 2), int(HEIGHT / 2)]
    min_x = int(np.floor(np.min(np.min(vertices, axis=0))))
    min_y = int(np.floor(np.min(np.min(vertices, axis=1))))
    max_x = int(np.ceil(np.max(np.max(vertices, axis=0))))
    max_y = int(np.ceil(np.max(np.max(vertices, axis=1))))

    upper, left, right = vertices[1], vertices[0], vertices[2]
    AB = upper - left
    BC = right - upper
    CA = left - right

    coordinates = np.array(np.meshgrid(np.arange(min_x, max_x),
                                       np.arange(min_y, max_y))).T.reshape(-1, 2)

    cross = lambda point, edge: np.cross(point, edge)
    area = np.cross(right - left, upper - left)

    mask = is_inside(cross(coordinates - upper, BC), cross(coordinates - right, CA), cross(coordinates - left, AB))

    triangle_points = coordinates[mask]
    n_points, _ = triangle_points.shape
    alphas = np.expand_dims(np.cross(triangle_points - upper, BC) / area, axis=1)
    betas = np.expand_dims(np.cross(triangle_points - right, CA) / area, axis=1)
    gamas = np.expand_dims(np.cross(triangle_points - left, AB) / area, axis=1)

    x_coord = np.clip(triangle_points[:, 0], -center[1], center[1]) + center[0] - 1
    y_coord = np.clip(-triangle_points[:, 1], -center[0], center[0]) - center[1]
    local_z_buffer[y_coord, x_coord] = left_z * alphas + upper_z * betas + right_z * gamas
    left_z = 1 - left_z
    right_z = 1 - right_z
    upper_z = 1 - upper_z
    color_value = left_z* alphas + upper_z* betas + right_z* gamas
    canvas[y_coord, x_coord] = np.where(z_buffer[y_coord, x_coord] < local_z_buffer[y_coord, x_coord], color_value,
                                        canvas[y_coord, x_coord])
    z_buffer = np.where(z_buffer > local_z_buffer, local_z_buffer, z_buffer)
    return canvas, z_buffer

def to_homogeneous(points):
    n_points, _ = points.shape
    ones = np.array([1 for i in range(n_points)]).reshape(-1, 1)
    homogeneous = np.hstack([points, ones])
    return homogeneous


def show_image(image):
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
