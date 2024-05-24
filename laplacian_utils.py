import torch


def laplacian_coordinates(verts, all_neighbours_indexes, constant_weight=True):
    num_vertices = len(verts)
    laplacian = torch.zeros((num_vertices, 3), dtype=torch.float)

    i = 0
    for vi in verts:
        neighbour_indexes = all_neighbours_indexes[i]
        delta = torch.tensor([0, 0, 0], dtype=torch.float)
        if constant_weight:
            weight = 1.0 / len(neighbour_indexes)
            for j in neighbour_indexes:
                vj = verts[j]
                delta += weight * (vi - vj)
        else:
            w_ik_sum = 0.0
            w_ij_dict = {}
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
                cot_alpha = calculate_cotangent(alpha, vi, vk)
                cot_beta = calculate_cotangent(beta, vi, vk)
                w_ij = cot_beta + cot_alpha
                w_ij_dict[k] = w_ij
                w_ik_sum += w_ij

            for j in neighbour_indexes:
                w_ij = w_ij_dict[j]
                vj = verts[j]
                delta += w_ij / w_ik_sum * (vi - vj)

        laplacian[i, 0] = delta[0]
        laplacian[i, 1] = delta[1]
        laplacian[i, 2] = delta[2]
        i += 1

    return laplacian


def calculate_cotangent(A, B, C):
    AB = B - A
    AC = C - A

    magnitude_AB = torch.norm(AB)
    magnitude_AC = torch.norm(AC)

    dot_product = torch.dot(AB, AC)
    cosine_angle_A = dot_product / (magnitude_AB * magnitude_AC)

    cross_product = torch.cross(AB, AC)
    sine_angle_A = torch.norm(cross_product) / (magnitude_AB * magnitude_AC)

    return cosine_angle_A / sine_angle_A
