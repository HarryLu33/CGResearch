import torch
from torch import Tensor


def hc_algorithm(old_verts, new_verts, all_neighbours_indexes: dict[int, list], B):
    fixed_verts: Tensor = torch.empty(len(old_verts), 3)
    i = 0
    while i < len(old_verts):
        q_i = old_verts[i]
        p_i = new_verts[i]
        b_i = p_i - q_i
        neighbour_indexes = all_neighbours_indexes[i]
        d_i = B * b_i
        for j in neighbour_indexes:
            q_j = old_verts[j]
            p_j = new_verts[j]
            b_j = p_j - q_j
            d_i += (1 - B) / len(neighbour_indexes) * b_j
        p_i = p_i - d_i
        fixed_verts[i] = p_i
        i += 1
    return fixed_verts
