import torch
from pytorch3d.structures import Meshes


def laplacian_coordinates(verts, meshes: Meshes, constant_weight=True):
    num_vertices = len(verts)
    laplacian = torch.zeros((num_vertices, 3), dtype=torch.float)
    all_neighbours_indexes = get_all_neighbours(verts, meshes.edges_packed())

    i = 0
    for vi in verts:
        neighbour_indexes = all_neighbours_indexes[i]
        if constant_weight:
            weight = 1.0 / len(neighbour_indexes)
        else:
            w_ij = 0.0
            w_ik_sum = 0.0
            for k in neighbour_indexes:
                alpha_beta = []
                for n in neighbour_indexes:
                    if len(alpha_beta) == 2:
                        break
                    if n in all_neighbours_indexes[i] and n in all_neighbours_indexes[k]:
                        alpha_beta.append(n)

                vk = verts[k]
                alpha = verts[alpha_beta[0]]
                beta = verts[alpha_beta[1]]

            weight = w_ij / w_ik_sum

        delta = torch.tensor([0, 0, 0], dtype=torch.float)
        for j in neighbour_indexes:
            vj = verts[j]
            delta += weight * (vi - vj)

        laplacian[i, 0] = delta[0]
        laplacian[i, 1] = delta[1]
        laplacian[i, 2] = delta[2]
        i += 1

    return laplacian


def get_neighbour_vertex(verts, edges, index):
    vi = []
    for edge in edges:
        if edge[0] == index:
            vi.append(edge[1])
        elif edge[1] == index:
            vi.append(edge[0])
    return vi


def get_all_neighbours(verts, edges):
    neighbour_dict = {}
    i = 0
    while i < len(verts):
        vi = []
        for edge in edges:
            if edge[0] == i:
                vi.append(verts[edge[1]])
            elif edge[1] == i:
                vi.append(verts[edge[0]])
        neighbour_dict[i] = vi
        i += 1
    return neighbour_dict
