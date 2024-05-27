import statistics

import torch
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from get_neighbours_utils import get_all_neighbours
from laplacian_utils import laplacian_coordinates
from radius_ratio import radius_ratio_array


def laplacian_optimization(filename, weight, us_HC: bool, B, turns):
    laplacian_operation(filename, "optimization", weight, us_HC, B, turns)


def laplacian_smoothing(filename, weight, us_HC: bool, B, turns):
    laplacian_operation(filename, "smoothing", weight, us_HC, B, turns)


def laplacian_operation(filename, option, weight, us_HC: bool, B, turns):
    # constants
    folder = "Models"
    extension = ".obj"

    # load meshes
    verts, faces, aux = load_obj(folder + "/" + filename + extension)
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    num_verts = verts.shape[0] if isinstance(verts, torch.Tensor) else len(verts)

    # load vertices neighbour relations
    all_neighbours_indexes = None
    if (option == "optimization") | us_HC:
        all_neighbours_indexes = get_all_neighbours(verts, meshes.laplacian_packed().to_dense())

    t = 1
    while t <= turns:
        # get the Laplacian matrix for the whole mesh
        L_u = meshes.laplacian_packed().to_dense()
        # use uniform weights W_p = I
        I_m = torch.eye(num_verts)
        # concatenate L and W_p to get A, use W_l = I
        A = torch.cat((L_u, I_m), dim=0)

        f = None
        # set f = delta_dc in detail preserving optimization
        if option == "optimization":
            # use cotangent weight
            delta_dc = laplacian_coordinates(verts, all_neighbours_indexes, weight)
            f = delta_dc
        # set f = 0 in mesh smoothing
        elif option == "smoothing":
            zero_matrix = torch.zeros(num_verts, 3)
            f = zero_matrix

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]
        # use all vertices as anchors in V_d
        V_d = torch.stack((x, y, z), dim=1)
        # concatenate f and V_d to get A, use W_l = I, W_p = I
        b = torch.cat((f, V_d), dim=0)

        # solve matrix equations A * V'd = b, V'd is the smoothed/optimised vertices
        new_verts = torch.inverse(A.t() @ A) @ A.t() @ b

        if us_HC:
            # use HC algorithm to push the modified points back towards the previous point
            b_i = new_verts - verts
            b_i_meshes = Meshes(verts=[b_i], faces=[faces.verts_idx])
            L_u_hc = b_i_meshes.laplacian_packed().to_dense()
            A = torch.cat((L_u_hc, I_m), dim=0)
            b = torch.cat((torch.zeros(num_verts, 3), b_i), dim=0)
            b_j_mean = torch.inverse(A.t() @ A) @ A.t() @ b
            final_verts_hc = new_verts - (B * b_i + (1 - B) * b_j_mean)

            meshes = Meshes(verts=[final_verts_hc], faces=[faces.verts_idx])
            save_to = folder + "/" + option + "/" + filename + "_" + option + "_hc_" + str(
                B) + "_" + weight + "_" + str(t) + extension
            IO().save_mesh(meshes, save_to)
            verts = final_verts_hc
        else:
            final_verts = new_verts
            meshes = Meshes(verts=[final_verts], faces=[faces.verts_idx])
            save_to = folder + "/" + option + "/" + filename + "_" + option + "_" + weight + "_" + str(t) + extension
            IO().save_mesh(meshes, save_to)
            verts = final_verts

        radius_ratios = radius_ratio_array(meshes)
        log = (save_to + "\n" +
               "maximum radius ratio: " + str(max(radius_ratios)) + "\n" +
               "median radius ratio: " + str(statistics.median(radius_ratios)) + "\n" +
               "mean radius ratio: " + str(statistics.mean(radius_ratios)) + "\n" +
               "minimum radius ratio: " + str(min(radius_ratios)) + "\n")
        print(log)
        t += 1
