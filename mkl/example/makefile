#The settings of system, core and compiler with the path and version of oneapi and mkl
SYSTEM = linux
CORE = intel64
CC = gcc
ONEAPI_VERSION = 2021.2.0
ONEAPI_COMPILER_PATH = /opt/intel/oneapi/compiler/$(ONEAPI_VERSION)/$(SYSTEM)/
MKL_PATH = /opt/intel/oneapi/mkl/$(ONEAPI_VERSION)/

INCLUDE = $(MKL_PATH)include
COMPILER_FLAGS = -O3 -DMKL_ILP64 -m64 -I$(INCLUDE) -w

#MKL_LP64, MKL_THREAD, MKL_CORE and ONEAPI_OMP5 represent 4 lib should be linked directly
MKL_LIB = $(MKL_PATH)/lib/$(CORE)/
MKL_LP64 = $(MKL_LIB)libmkl_intel_ilp64.a
MKL_THREAD = $(MKL_LIB)libmkl_intel_thread.a
MKL_CORE = $(MKL_LIB)libmkl_core.a
ONEAPI_OMP5 = $(ONEAPI_COMPILER_PATH)compiler/lib/$(CORE)/libiomp5.a
LINK_FLAGS = -Wl,--start-group $(MKL_LP64) $(MKL_THREAD) $(MKL_CORE) $(ONEAPI_OMP5) -Wl,--end-group -L$(ONEAPI_COMPILER_PATH) -lpthread -ldl -lm -fopenmp

DEMO = dashsvdtest
SOURCE = dashsvd_test.c ../src/dashsvd.c ../src/matrix_vector_functions.c
$(DEMO): $(SOURCE)
	$(CC) $(SOURCE) $(COMPILER_FLAGS) $(LINK_FLAGS) -o $(DEMO)

clean:
	rm -rf $(DEMO)
