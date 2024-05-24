from laplacian_mesh_operation import laplacian_smoothing, laplacian_optimization

turns = 5

laplacian_smoothing("noisy_mesh", False, turns)
laplacian_smoothing("noisy_mesh", True, turns)
laplacian_smoothing("Teddy", False, turns)
laplacian_smoothing("Teddy", True, turns)
laplacian_optimization("Teddy", False, turns)
laplacian_optimization("Teddy", True, turns)
