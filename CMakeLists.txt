#add the directory that contains the Viskores config file to the cmake
#path so that our code can find Viskores
#normally when running CMake, you need to set the Viskores_DIR to
#find the ViskoresConfig.cmake file. 

cmake_minimum_required(VERSION 3.12...3.15 FATAL_ERROR)
project(Viskores_tut)

#Find the Viskores package
find_package(Viskores REQUIRED QUIET)

if(Viskores_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

add_executable(vf_divergence_uncertainty vf_divergence_uncertainty.cxx)
target_link_libraries(vf_divergence_uncertainty PRIVATE viskores::filter_core viskores::filter_vector_analysis viskores::io)
if(Viskores_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(vf_divergence_uncertainty PRIVATE OpenMP::OpenMP_CXX)
endif()

