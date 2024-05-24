import torch
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes

from get_neighbours_utils import get_all_neighbours
from hc_utils import hc_algorithm
from laplacian_utils import laplacian_coordinates

# check cuda
print(f"CUDA version: {torch.version.cuda}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

filename = "Models/Teddy.obj"
verts, faces, aux = load_obj(filename)
meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
num_verts = verts.shape[0] if isinstance(verts, torch.Tensor) else len(verts)

# Get the Laplacian matrix for the whole mesh
L_u = meshes.laplacian_packed().to_dense()
# L_u = laplacian(verts, meshes.edges_packed())
I_m = torch.eye(num_verts)
A = torch.cat((L_u, I_m), dim=0)

all_neighbours_indexes = get_all_neighbours(verts, meshes.edges_packed())
laplacian = laplacian_coordinates(verts, all_neighbours_indexes, False)

x = verts[:, 0]
y = verts[:, 1]
z = verts[:, 2]
V_d = torch.stack((x, y, z), dim=1)

b = torch.cat((laplacian, V_d), dim=0)

x_solution = torch.inverse(A.t() @ A) @ A.t() @ b

new_verts = x_solution

# final_verts = hc_algorithm(verts, new_verts, all_neighbours_indexes, 0.8)
final_verts = new_verts

meshes = Meshes(verts=[final_verts], faces=[faces.verts_idx])
IO().save_mesh(meshes, "Models/optimized_model.obj")
