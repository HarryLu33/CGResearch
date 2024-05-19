import torch
from pytorch3d.structures import Meshes


def laplacian_coordinates(vertices, meshes: Meshes, constant_weight=True):
    num_vertices = len(vertices)
    laplacian = torch.zeros((num_vertices, 3), dtype=torch.float)

    i = 0
    for vi in vertices:
        all_neighbour = getNeighbourVertex(vertices, meshes.edges_packed(), i)
        if constant_weight:
            weight = 1.0 / len(all_neighbour)
        else:
            # Calculate weight based on the cotangent of the angle opposite to the edge
            weight = 1.0

        delta = torch.tensor([0, 0, 0], dtype=torch.float)
        for vj in all_neighbour:
            delta += weight * (vi - vj)

        laplacian[i, 0] = delta[0]
        laplacian[i, 1] = delta[1]
        laplacian[i, 2] = delta[2]
        i += 1

    return laplacian


def getNeighbourVertex(verts, edges, index):
    vi = []
    for edge in edges:
        if edge[0] == index:
            vi.append(verts[edge[1]])
        elif edge[1] == index:
            vi.append(verts[edge[1]])
    return vi
