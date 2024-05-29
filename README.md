# Volume Preserved Laplacian Mesh Optimization
This Python project performs Laplacian mesh optimization and smoothing on 3D models with volume preservation features. It uses the Pytorch and PyTorch3D libraries for mesh processing and linear algebra operations. The project provides two main functions in laplacian_mesh_operation.py: laplacian_optimization and laplacian_smoothing which are supported by the the following utility modules: get_neighbours_utils.py, laplacian_utils.py, and radius_ratio.py 
## Installation 
To use this script, you need to have Python and the required libraries installed. The detailed installation guide for pytorch3d can be found at https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
## File Description
### main.py
The main file demonstrates the experiment using the laplacian_smoothing and laplacian_optimization functions from the laplacian_mesh_operation module. The script applies Laplacian smoothing and optimization to two 3D models, "noisy_dolphin" and "Teddy". For each model, the operations are performed with and without the HC algorithm, using a cotangent weight and iterating for four turns. The B parameter for the HC algorithm is set to 0.6.
####Parameters:
- turns (int): The number of iterations for the operation.
- weight (str): The weight parameter for the Laplacian operation, specified as "cotangent".
- B (float): The parameter for the HC algorithm.
####Summary of Operations
Laplacian Smoothing on "noisy_dolphin":

- Without HC algorithm
- With HC algorithm
- 
Laplacian Smoothing on "Teddy":

- Without HC algorithm
- With HC algorithm
- 
Laplacian Optimization on "Teddy":

- Without HC algorithm
- With HC algorithm
By running this script, you can observe the effects of the Laplacian smoothing and optimization processes on the models, including the impact of using the HC algorithm to refine the vertex positions. The result obj files are stored in the smoothing and optimization folders correspondingly.
### laplacian_mesh_operation.py
This file provides two interface functions :
- laplacian_optimization(filename, weight, us_HC, B, turns)
- laplacian_smoothing(filename, weight, us_HC, B, turns)

They have the following parameters:

- filename (str): The name of the OBJ file (without extension) located in the "Models" folder.
- weight (float): The weight parameter for the Laplacian operation.
- us_HC (bool): A flag to use the HC algorithm.
- B (float): The parameter for the HC algorithm scalar.
- turns (int): The number of iterations for the (optimization/smoothing)operation.
  
The laplacian_operation function is the core of this script. It performs the following steps:
- Load Meshes: Reads the mesh from the specified OBJ file.
- Compute Neighbours: Calculates the neighboring vertices if the operation is "optimization".
- Iterative Processing: For each iteration:
  * Compute the Laplacian matrix.
  * Construct the system of equations and solve for new vertex positions.
  * Optionally apply the HC algorithm for further refinement.
The script saves the processed mesh to the "Models/optimization" or "Models/smoothing" folder with an appropriate filename indicating the operation type, parameters, and iteration count. It also logs the radius ratios of the resulting mesh, which includes the maximum, median, mean, and minimum radius ratios.
