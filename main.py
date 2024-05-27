from laplacian_mesh_operation import laplacian_smoothing, laplacian_optimization

turns = 4
weight = "constant"
B = 0.6

laplacian_smoothing("noisy_dolphin", weight, False, B, turns)
laplacian_smoothing("noisy_dolphin", weight, True, B, turns)
laplacian_smoothing("Teddy", weight, False, B, turns)
laplacian_smoothing("Teddy", weight, True, B, turns)
laplacian_optimization("Teddy", weight, False, B, turns)
laplacian_optimization("Teddy", weight, True, B, turns)
