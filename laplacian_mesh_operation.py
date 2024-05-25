import torch
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from get_neighbours_utils import get_all_neighbours
from hc_utils import hc_algorithm
from laplacian_utils import laplacian_coordinates


def laplacian_optimization(filename, us_HC: bool, turns=1):
    laplacian_operation(filename, "optimization", us_HC, turns)


def laplacian_smoothing(filename, us_HC: bool, turns=1):
    laplacian_operation(filename, "smoothing", us_HC, turns)


def laplacian_operation(filename, option, us_HC: bool, turns):
    # check cuda
    print(f"CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # constants
    file_path = "Models"
    file_extension = ".obj"

    # load meshes
    verts, faces, aux = load_obj(file_path + "/" + filename + file_extension)
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    num_verts = verts.shape[0] if isinstance(verts, torch.Tensor) else len(verts)

    # load vertices neighbour relations
    all_neighbours_indexes = None
    if (option == "optimization") | us_HC:
        all_neighbours_indexes = get_all_neighbours(verts, meshes.edges_packed())

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
            delta_dc = laplacian_coordinates(verts, all_neighbours_indexes, False)
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
            final_verts_hc = hc_algorithm(verts, new_verts, all_neighbours_indexes, 0.8)
            meshes = Meshes(verts=[final_verts_hc], faces=[faces.verts_idx])
            save_to = file_path + "/" + option + "/" + filename + "_" + option + "_hc" + "_" + str(t) + file_extension
            IO().save_mesh(meshes, save_to)
            verts = final_verts_hc
        else:
            final_verts = new_verts
            meshes = Meshes(verts=[final_verts], faces=[faces.verts_idx])
            save_to = file_path + "/" + option + "/" + filename + "_" + option + "_" + str(t) + file_extension
            IO().save_mesh(meshes, save_to)
            verts = final_verts

        t += 1
