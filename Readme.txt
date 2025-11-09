============================================================================

Title  -   Efficient Probabilistic Visualization of Local Divergence of 2D Vector Fields with Independent Gaussian Uncertainty
Authors -  T. A. J. Ouermi, E. Li, K. Moreland, D. Pugmire, C. R. Johnson, T. M. Athawale
Date    -  Nov 02, 2025
 
2025 IEEE Uncertainty Visualization Workshop, Vienna, Austria.
 
Preprint: https://arxiv.org/pdf/2510.01190

This code is distributed under the OSI-approved BSD 3-clause License. See LICENSE.txt for details.
 
============================================================================

- Build the viskores library freshly (with serial/OpenMP/GPU backend you need) using the viskores source code (tested with viskores 1.0).

- Create a build directory within the source code directory and "cd" to it.

- ccmake ..

- For the viskores dir, point to viskoresBuild/lib/cmake/viskores-1.0

- Generate the makefile and create an executable.

- Run "./vf_divergence_uncertainty". The code will run on the Red Sea simulation dataset in the paper. The Red Sea dataset is courtesy of IEEE SciVis constest 2020. https://kaust-vislab.github.io/SciVis2020/

- Visualize the generated result in ParaView with parameter/colormap settings in the paper.

- Additional note about the OpenMP usage on MAC:

The default Mac clang++/g++compilers do not support OpenMP!! 

So we have to use the llvm clang++ compiler for that purpose.

So update CMAKE_CXX_COMPILER to /opt/homebrew/opt/llvm/bin/clang++ or 
wherever the llvm clang++ is installed