import torch
from pytorch3d.io import load_obj, IO
from pytorch3d.structures import Meshes

filename = "Models/dolphin.obj"
verts, faces, aux = load_obj(filename)
meshes = Meshes(verts=[verts], faces=[faces.verts_idx])

# Add noise to the vertex positions

# Standard deviation of the Gaussian noise
noise_std = 0.002
noise = torch.randn_like(verts) * noise_std
noisy_verts = verts + noise

# Create noisy mesh
noisy_meshes = Meshes(verts=[noisy_verts], faces=[faces.verts_idx])

IO().save_mesh(noisy_meshes, "Models/noisy_dolphin.obj")
