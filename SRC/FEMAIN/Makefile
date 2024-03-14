################################################################################
# FastEddy®: SRC/FEMAIN/Makefile
# ©2016 University Corporation for Atmospheric Research
#
# This file is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ################################################################################
#
# ################################################################################
# # Available build targets
# ################################################################################

TARGETS = \
       FastEddy 


################################################################################
## Compiler and flags
################################################################################

TEST_CC = mpicc
TEST_CU_CC = nvcc

DEBUG_CFLAGS = -g 

DEFINES = -DCUB_IGNORE_DEPRECATED_CPP_DIALECT -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT


TEST_CFLAGS = -Wall -m64 ${DEFINES} ${INCLUDES} ${OTHER_INCLUDES}
ARCH_CU_FLAGS = -arch=sm_70
TEST_CU_CFLAGS = ${ARCH_CU_FLAGS} -m64 -std=c++11 ${DEFINES} ${INCLUDES} ${OTHER_INCLUDES}

L_CPPFLAGS =

#OTHER_INCLUDES =
# For compilation on NCAR's Derecho supercomputing using Cray compiler use the OTHER_INCLUDES below
OTHER_INCLUDES = -I/opt/cray/pe/mpich/8.1.25/ucx/cray/10.0/include/

TEST_LDFLAGS = -L.
TEST_CU_LDFLAGS = -L.

TEST_LIBS = -lm -lmpi -lstdc++ -lcurand -lcudart -lnetcdf
TEST_CU_LIBS = -lm -lmpi -lcudart


################################################################################
# Sub-directories to INCLUDE 
# ################################################################################

INCLUDES = \
	-I. \
	-I../TIME_INTEGRATION/ \
	-I../TIME_INTEGRATION/CUDA \
	-I../HYDRO_CORE \
	-I../HYDRO_CORE/CUDA \
	-I../GRID \
	-I../GRID/CUDA \
	-I../FECUDA \
	-I../IO/ \
	-I../MEM_UTILS/ \
	-I../FEMPI/ \
	-I../PARAMETERS/

################################################################################
# Headers
# ###############################################################################

TEST_HDRS = \
	../TIME_INTEGRATION/time_integration.h \
        ../TIME_INTEGRATION/CUDA/cuda_timeInt.h \
        ../TIME_INTEGRATION/CUDA/cuda_timeIntDevice_cu.h \
	../HYDRO_CORE/hydro_core.h \
	../HYDRO_CORE/CUDA/cuda_hydroCore.h \
	../HYDRO_CORE/CUDA/cuda_hydroCoreDevice_cu.h \
	../GRID/grid.h \
	../GRID/CUDA/cuda_grid.h \
	../GRID/CUDA/cuda_gridDevice_cu.h \
        ../FECUDA/fecuda.h \
        ../FECUDA/fecuda_Device_cu.h \
	../IO/io.h \
	../MEM_UTILS/mem_utils.h \
	../FEMPI/fempi.h \
	../PARAMETERS/parameters.h \
	../PARAMETERS/hashTable.h

################################################################################
# Objects
# ###############################################################################

TEST_LIB_OBJS =

TEST_OBJS = ./FastEddy.o \
	../TIME_INTEGRATION/time_integration.o \
        ../TIME_INTEGRATION/CUDA/cuda_timeInt.o \
        ../TIME_INTEGRATION/CUDA/cuda_timeIntDevice.o \
	../HYDRO_CORE/hydro_core.o \
	../HYDRO_CORE/CUDA/cuda_hydroCore.o \
	../HYDRO_CORE/CUDA/cuda_hydroCoreDevice.o \
	../GRID/grid.o \
	../GRID/CUDA/cuda_grid.o \
	../GRID/CUDA/cuda_gridDevice.o \
        ../FECUDA/fecuda.o \
        ../FECUDA/fecuda_Device.o \
	../IO/io.o \
	../IO/ioVarsList.o \
	../MEM_UTILS/mem_utils.o \
	../FEMPI/fempi.o \
	../PARAMETERS/parameters.o \
	../PARAMETERS/hashTable.o

################################################################################
# Targets
# ###############################################################################

all: FastEddy

################################################################################
# Generic compile rules
# ################################################################################
%.o: %.cu
	rm -rf ./FastEddy_devlink.o; \
	$(TEST_CU_CC) $(TEST_CU_CFLAGS) -dc $< -o $@

.c.o:
	${COMP_FLAGS} -c $< -o $@

################################################################################
# Module=specific compile rules
# ################################################################################

../TIME_INTEGBRATION/CUDA/cuda_timeIntDevice.o: ../CUDA/cuda_timeIntDevice.cu \
        ../TIME_INTEGRATION.CUDA/cuda_RKschemes.cu
	rm -rf ./FastEddy_devlink.o; \
	$(TEST_CU_CC) $(TEST_CU_CFLAGS) -dc $< -o $@

../IO/io.o: ../IO/io.c \
	../IO/io_netcdf.c \
	../IO/io_binary.c 
	$(COMP_FLAGS) -c $< -o $@

../FECUDA/fecuda_Device.o: ../FECUDA/fecuda_Device.cu \
	../FECUDA/fecuda_Utils.cu \
	../FECUDA/fecuda_PlugIns.cu 
	rm -rf ./FastEddy_devlink.o; \
	$(TEST_CU_CC) $(TEST_CU_CFLAGS) -dc $< -o $@

../HYDRO_CORE/CUDA/cuda_hydroCoreDevice.o: ../HYDRO_CORE/CUDA/cuda_hydroCoreDevice.cu \
	../HYDRO_CORE/CUDA/cuda_BaseStateDevice.cu \
	../HYDRO_CORE/CUDA/cuda_buoyancyDevice.cu \
	../HYDRO_CORE/CUDA/cuda_coriolisDevice.cu \
	../HYDRO_CORE/CUDA/cuda_pressureDevice.cu \
	../HYDRO_CORE/CUDA/cuda_BCsDevice.cu \
	../HYDRO_CORE/CUDA/cuda_rayleighDampingDevice.cu \
	../HYDRO_CORE/CUDA/cuda_advectionDevice.cu \
	../HYDRO_CORE/CUDA/cuda_molecularDiffDevice.cu \
	../HYDRO_CORE/CUDA/cuda_surfaceLayerDevice.cu \
	../HYDRO_CORE/CUDA/cuda_sgsTurbDevice.cu \
        ../HYDRO_CORE/CUDA/cuda_sgstkeDevice.cu \
        ../HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice.cu \
        ../HYDRO_CORE/CUDA/cuda_moistureDevice.cu \
        ../HYDRO_CORE/CUDA/cuda_filtersDevice.cu
	rm -rf ./FastEddy_devlink.o; \
	$(TEST_CU_CC) $(TEST_CU_CFLAGS) -dc $< -o $@
################################################################################
# Generic Executable
# ###############################################################################

FastEddy: COMP_FLAGS = ${TEST_CC} ${TEST_CFLAGS}

TEST_DEPENDENCIES = ${TEST_OBJS} ${TEST_HDRS} FastEddy_devlink.o

FastEddy_devlink.o:
	$(TEST_CU_CC) ${ARCH_CU_FLAGS} ${TEST_OBJS} ${TEST_LDFLAGS} ${TEST_LIBS} -dlink -o FastEddy_devlink.o
FastEddy: ${TEST_DEPENDENCIES}
	${COMP_FLAGS} -o FastEddy ${TEST_OBJS} ./FastEddy_devlink.o ${TEST_LDFLAGS} ${TEST_LIBS}

################################################################################
# Clean
# ################################################################################

clean:
	rm -rf ${TARGETS} ${TEST_OBJS} ${TEST_LIB_OBJS} ./FastEddy_devlink.o

################################################################################
# Check
# ###############################################################################

check:
	@echo CC = ${CC}
	@echo CFLAGS = ${CFLAGS} 
	@echo LDFLAGS = ${LDFLAGS} 
	@echo LIBS = ${LIBS}
	@echo DEPENDENCIES = ${TEST_DEPENDENCIES}
