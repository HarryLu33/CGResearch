import numpy as np
from numpy import ndarray
from pytorch3d.structures import Meshes


# Define the three points
# A = np.array([0, 9, 0])
# B = np.array([9, 0, 0])
# C = np.array([0, 0, 9])

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
    # Calculate the vectors AB, AC, and BC
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

    # Calculate the circumcenter
    # Using the formula for circumcenter in 3D space
    D = 2 * (np.linalg.norm(np.cross(B - A, C - A)) ** 2)
    Ux = (np.dot(np.cross(B - A, B - C), np.cross(B - A, C - A)) * np.dot(C - A, C - A) +
          np.dot(np.cross(C - A, C - B), np.cross(C - A, A - B)) * np.dot(A - B, A - B) +
          np.dot(np.cross(A - B, A - C), np.cross(A - B, B - C)) * np.dot(B - C, B - C)) / D
    circumcenter = A + Ux * (B - A)

    # Calculate the semi-perimeter
    s = (a + b + c) / 2

    # Calculate the inradius
    inradius = area / s

    # Calculate the incenter
    # Using the formula for incenter in 3D space
    incenter = (a * A + b * B + c * C) / (a + b + c)

    return 2 * inradius / circumradius
