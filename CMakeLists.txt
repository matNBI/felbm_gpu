###############################################################################
# felbm_gpu — CUDA multiphase LBM solver (felbm_gpu binary only).
#
# Reuses the felbm_local HOST headers for config/geometry/init/operators and
# compiles the CUDA engine. Build:
#
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=80        # 80=A100, 86=RTX30, 89=RTX40, 90=H100
#   cmake --build . -j
#
# Single precision:  cmake .. -DFELBM_SINGLE=ON
###############################################################################
cmake_minimum_required(VERSION 3.18)
project(felbm_gpu LANGUAGES CXX CUDA)

option(FELBM_SINGLE "Use single-precision (float) on the device" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)   # sensible default; override for your card
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- dependencies (same as felbm_local) -------------------------------------
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(TIFF REQUIRED)
find_package(OpenMP REQUIRED)   # felbm_local host code uses OpenMP

# ---- reused felbm_local host headers ----------------------------------------
set(FELBM_LOCAL "${CMAKE_CURRENT_SOURCE_DIR}/../felbm_local" CACHE PATH "Path to the CPU felbm_local tree")

# felbm_local library sources that carry out-of-line definitions used by the
# header code (e.g. EquilibriumDistributionMRT::m_mass_matrix_inv, string helpers).
# The other felbm_local *.cpp are empty (header-only), so only these are needed.
set(FELBM_LOCAL_SRC
  ${FELBM_LOCAL}/UTIL/src/util_string_helper.cpp
  ${FELBM_LOCAL}/LBM/src/lbm_equilbrium_distribution_mrt.cpp
)

# Two targets: the solver (felbm_gpu) and the CPU-vs-GPU validation harness.
add_executable(felbm_gpu      src/felbm_gpu_main.cu  ${FELBM_LOCAL_SRC})
add_executable(compare_cpu_gpu src/compare_cpu_gpu.cu ${FELBM_LOCAL_SRC})

# Apply the same include dirs / libs / flags to both.
foreach(tgt felbm_gpu compare_cpu_gpu)
  target_include_directories(${tgt} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${FELBM_LOCAL}/LBM/include
    ${FELBM_LOCAL}/LINALG/include
    ${FELBM_LOCAL}/UTIL/include
    ${HDF5_INCLUDE_DIRS}
  )
  target_link_libraries(${tgt} PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} TIFF::TIFF OpenMP::OpenMP_CXX)
  if(FELBM_SINGLE)
    target_compile_definitions(${tgt} PRIVATE FELBM_SINGLE)
  endif()
  # Let the host compiler see the felbm_local headers as host C++ under nvcc.
  target_compile_options(${tgt} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>   # felbm_local host code uses OpenMP pragmas
  )
endforeach()
