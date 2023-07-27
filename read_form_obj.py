import numpy as np

def read_from_obj(path_to_obj):
    with open(path_to_obj, "r") as f:
        lines = f.readlines()

    count_vertices = 0
    count_textures = 0
    count_faces = 0
    for line in lines:
        if line.startswith("v "):
            count_vertices+=1
        elif line.startswith("vt"):
            count_textures+=1
        elif line.startswith("f "):
            count_faces+=1

    vertices = np.zeros(shape=(count_vertices, 3))
    faces = np.zeros(shape=(count_faces, 3))
    vertices_inserted = 0
    faces_inserted = 0
    for line in lines:
        if line.startswith("v "):
            vertices[vertices_inserted] = np.array(line.split()[1:])
            vertices_inserted+=1
        elif line.startswith("f "):
            intermediate = line.split(" ")[1:]
            vertices_indexes = []
            for component in intermediate:
                vertices_indexes.append(int(component.split("/")[0]))
            faces[faces_inserted] = np.array(vertices_indexes)
            faces_inserted+=1
    faces = faces.astype(int) - 1
    return vertices, faces
