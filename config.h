////////////////////////////////////////////////////////////
//	config.h
//
//	Configuration header file for project.
//  Includes necessary header files and
//  defines constants used.
//
//	Author: Yun Zhang
//	Date Created: November 29, 2011
//	Last Modified: April 18, 2012
////////////////////////////////////////////////////////////
#ifndef CONFIG_H
#define CONFIG_H

//include LAPACK
#include "clapack.h"
#include <culapack.h>


//include necessary C++ headers
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//include necessary files for CUDA
#include <cublas.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
using namespace std;

//define constants used
//default separator
#define DEF_SEPARATOR '\t'

//cluster membership threshold
#define THRESHOLD 0.55f

//maximum iterations for Varimax
#define VARIMAX_ITERATION 100

//stopping criterion for Varimax
#define VARIMAX_EPSILON 1e-6

//maximum iterations for QR alg.
#define QR_ITERATIONS 100

//desired accuracy of QR algorithm
#define QR_EPSILON 1.0e-6

//number of threads for 2d array
#define TWOD_THREADS 16

//number of threads for 1d array
#define ONED_THREADS 384

//default number of cluster groups
#define K	30

//print available GPU memory if # of genes exceeds this number
#define LARGE_MATRIX 2048

//dataset to use if none entered on command-line
#define FILE_PATH "/home/jingram/NVIDIA_GPU_Computing_SDK/C/src/zhu_RMT/pca.txt"

#endif
//end config.h
