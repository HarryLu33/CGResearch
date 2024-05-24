import torch
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes
from get_neighbours_utils import get_all_neighbours
from hc_utils import hc_algorithm
from laplacian_utils import laplacian_coordinates


def laplacian_optimising(filename_ori, filename_target, us_HC: bool):
    laplacian_operation(filename_ori, filename_target, "optimising", us_HC)


def laplacian_smoothing(filename_ori, filename_target, us_HC: bool):
    laplacian_operation(filename_ori, filename_target, "smoothing", us_HC)


def laplacian_operation(filename_ori, filename_target, option, us_HC: bool):
    # check cuda
    print(f"CUDA version: {torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    file_extension = ".obj"
    verts, faces, aux = load_obj(filename_ori + file_extension)
    meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
    num_verts = verts.shape[0] if isinstance(verts, torch.Tensor) else len(verts)

    # Get the Laplacian matrix for the whole mesh
    L_u = meshes.laplacian_packed().to_dense()
    I_m = torch.eye(num_verts)
    A = torch.cat((L_u, I_m), dim=0)

    f = None
    all_neighbours_indexes = None
    if option == "optimising":
        all_neighbours_indexes = get_all_neighbours(verts, meshes.edges_packed())
        f = laplacian_coordinates(verts, all_neighbours_indexes, False)
    elif option == "smoothing":
        zero_matrix = torch.zeros(num_verts, 3)
        f = zero_matrix

    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    V_d = torch.stack((x, y, z), dim=1)

    b = torch.cat((f, V_d), dim=0)

    x_solution = torch.inverse(A.t() @ A) @ A.t() @ b

    new_verts = x_solution

    if us_HC:
        if all_neighbours_indexes is None:
            all_neighbours_indexes = get_all_neighbours(verts, meshes.edges_packed())
        final_verts_hc = hc_algorithm(verts, new_verts, all_neighbours_indexes, 0.8)
        meshes = Meshes(verts=[final_verts_hc], faces=[faces.verts_idx])
        IO().save_mesh(meshes, filename_target + file_extension)
    else:
        final_verts = new_verts
        meshes = Meshes(verts=[final_verts], faces=[faces.verts_idx])
        IO().save_mesh(meshes, filename_target + file_extension)
