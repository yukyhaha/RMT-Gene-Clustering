#####################################################
#
# Project makefile.
# Provided by NVIDIA Corporation.
# Modified by Yun Zhang
# Date: April 29, 2012
#
#####################################################
INCLUDES=-I/usr/local/cula/include
LIBPATH32=-L/usr/local/cula/lib -L/usr/local/cuda/lib
# Add source files here
EXECUTABLE	:= rmt
# CUDA source files (compiled with cudacc)
CUFILES := \
	rmta_gpu.cu \
# CUDA dependency files
CU_DEPS := \
	cublas_eig.cu \
	varimax.cu \
# C/C++ source files (compiled with gcc / c++)
CCFILES := \
	main.cpp \
# Additional libraries needed by the project
USECUBLAS := 1

#####################################################
# Rules and targets

include ../../common/common.mk
LIB += -L/usr/local/cula/lib -I/usr/local/cula/include -llapack -llapack_atlas 
LIB += -lcula -lblas -latlas -lm -lcuda
