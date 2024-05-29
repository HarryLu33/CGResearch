import numpy as np
from numpy import ndarray
from pytorch3d.structures import Meshes


def radius_ratio_array(meshes: Meshes):
    faces = meshes.faces_packed()
    verts = meshes.verts_packed()
    radius_ratios = []
    for face in faces:
        A = verts[face[0]].numpy()
        B = verts[face[1]].numpy()
        C = verts[face[2]].numpy()
        radius_ratios.append(radius_ratio(A, B, C))
    return radius_ratios


def radius_ratio(A: ndarray, B: ndarray, C: ndarray):
    # Calculate the edge vectors AB, AC, and BC
    AB = B - A
    AC = C - A
    BC = C - B

    # Calculate the lengths of the sides
    a = np.linalg.norm(BC)
    b = np.linalg.norm(AC)
    c = np.linalg.norm(AB)

    # Calculate the area of the triangle using the cross product
    area_vector = np.cross(AB, AC)
    area = np.linalg.norm(area_vector) / 2

    # Calculate the circumradius
    circumradius = (a * b * c) / (4 * area)

    # Calculate the semi-perimeter
    s = (a + b + c) / 2

    # Calculate the inradius
    inradius = area / s

    return 2 * inradius / circumradius
