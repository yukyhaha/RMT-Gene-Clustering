////////////////////////////////////////////////////////////
//  rmta_gpu.cu
//
//  Provides functions for microarray analysis
//  using random matrix theory
//  on the GPU.
//
//  Author: Yun Zhang
//	Date Created: November 29, 2011
//	Last Modified: April 18, 2012
////////////////////////////////////////////////////////////
//include necessary files
#include "config.h"
#include <cublas_eig.cu>
#include <varimax.cu>

//include necessary external functions
extern int culaSstein(float *eigenvalues, float *eigenvectors, 
			float *matrix, int n,float vl);

extern int checkStatus2(culaStatus status);

int culaSstein2(float *eigenvalues, float *eigenvectors, float *matrix, int k, int n)
{

       int *ifail = new int[n];
       culaStatus status;
  
       float vl = eigenvalues[n-k]+1;
       float vu = eigenvalues[n-1]+0.1;
       float abstol=0.0;
       int m = 0;
       float* w = new float[n];
       float* z = new float[n*n];

       status = culaSsyevx('V', 'I', 'L', n, matrix, n,vl, vu, n-k+1, n, abstol, &m, w, z, n, ifail);
       int result =checkStatus2(status);
        
	int  count = 0;
	for(int i = k-1; i >= 0; i--)
	{
		for(int j = 0; j < n; j++)
		{
		    eigenvectors[count*n+j] = z[i*n+j];
		}
		count++;

	}

	//float temp;
         for(int i=0; i< m; i++){
             printf("%f ", w[i]);
             eigenvalues[i] = w[m-1-i];
          }

	for(int i = k; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			eigenvectors[i*n+j] = 0.0;
		}
	}


	delete[] ifail;
        delete[] w;
        delete[] z;
        return result;
}


////////////////////////////////////////////////////////////
//  
//  Kernel function.  Compute row averages of g_data.
//
//  @param	g_data 	input dataset
//  @param	average	contains computed row averages
//  @param	rows   	# of rows of g_data
//  @param	cols   	# of columns of g_data
////////////////////////////////////////////////////////////
__global__ void GetAverages(float* g_data, float* average, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < rows)
	{	
		float avg = 0.0;
		
		for(int j=0;j<cols;j++)
		{
			avg += g_data[(idx*cols)+j];
		}
		
		average[idx] = avg / cols;
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel function.  Compute row standard deviations 
//  of g_data.
//
//  @param	g_data 	input dataset
//  @param	average	contains computed row averages
//  @param	stdev  	contains computed standard deviations
//  @param	rows   	# of rows of g_data
//  @param	cols   	# of columns of g_data
////////////////////////////////////////////////////////////
__global__ void GetStdev(float* g_data, float* average, float* stdev, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < rows)
	{
		float std = 0.0;
		
		for(int j=0;j<cols;j++)
		{
			std += powf(g_data[idx*cols+j] - average[idx],2);
		}
		
		stdev[idx] = (float) sqrtf(fabs(std/(cols-1)));
		
		if(stdev[idx] == 0) stdev[idx] = 1;
	}
}

////////////////////////////////////////////////////////////
//  
//  Compute Pearson correlation matrix of g_data.
//
//  @param	g_data 	input dataset
//  @param	average	contains computed row averages
//  @param	stdev  	contains computed standard deviations
//  @param	pearson	contains Pearson correlation matrix
//  @param	rows   	# of rows of g_data
//  @param	cols   	# of columns of g_data
////////////////////////////////////////////////////////////
__global__ void ComputeCorrelation(float* g_data, float* average, float* stdev, float* pearson, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if(idx < rows && idy < rows && idx <= idy)
	{
		float pear = 0.0;
		
		for(int j = 0; j < cols; j++)
		{
			pear += (g_data[idx*cols+j] - average[idx])*(g_data[idy*cols+j]-average[idy]);
		}
		
		pear = pear / ((cols-1) * stdev[idx] * stdev[idy]);
		
		pearson[(idx*rows)+idy] = (float) pear;
		pearson[(idy*rows)+idx] = pear;
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel function.  Transforms eigenvectors to component
//  loadings.
//
//  @param	eigenvectors	eigenvectors to be transformed
//  @param	eigenvalues 	corresponding eigenvalues
//  @param	rows        	# of rows of eigenvectors
//  @param	cols        	# of columns of eigenvectors
////////////////////////////////////////////////////////////
__global__ void Transform(float *eigenvectors, float *eigenvalues, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx < rows && idy < rows)
	{
		eigenvectors[idx*rows+idy] = eigenvectors[idx*rows+idy] * sqrtf(fabs(eigenvalues[idy]));
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel function.  Transposes matrix in into matrix out.
//  Modified from CUDA SDK Code samples.
//
//  @param	in  	input matrix
//  @param	out 	output transposed matrix in
//  @param	rows	# of rows of in and out
//  @param	cols	# of columns of in and out
////////////////////////////////////////////////////////////
__global__ void Transpose(float *in, float *out, int rows, int cols)
{
	__shared__ float block[TWOD_THREADS][TWOD_THREADS];
	
	// read the matrix tile into shared memory
	int xIndex = blockIdx.x * TWOD_THREADS + threadIdx.x;
	int yIndex = blockIdx.y * TWOD_THREADS + threadIdx.y;
	if((xIndex < cols) && (yIndex < rows))
	{
		int index_in = yIndex * cols + xIndex;
		block[threadIdx.y][threadIdx.x] = in[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TWOD_THREADS + threadIdx.x;
	yIndex = blockIdx.x * TWOD_THREADS + threadIdx.y;
	if((xIndex < rows) && (yIndex < cols))
	{
		int index_out = yIndex * rows + xIndex;
		out[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel function. Get diagonal elements of matrix M.
//
//  @param	M    	input matrix
//  @param	diags	output, diagonal elements
//  @param	n    	# of rows and columns of M
////////////////////////////////////////////////////////////
__global__ void GetDiagonals(float *M, float *diags, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < n)
	{
		diags[idx] = M[idx*n+idx];
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel function. Get superdiagonals of matrix M
//
//  @param	M    	input matrix
//  @param	sd	output, superdiagonal elements
//  @param	n    	# of rows and columns of M
////////////////////////////////////////////////////////////
__global__ void GetSuperDiagonals(float *M, float *sd, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < n && idx > 0)
	{
		sd[idx-1] = M[idx*n+(idx-1)];
	}
}

////////////////////////////////////////////////////////////
//  
//  Kernel wrapper function.  Computes Pearson correlation
//  matrix of data.
//
//  @param	data	input data
//  @param	pearson	outputted Pearson matrix
//  @param	rows   	# of rows of data, # of rows and 
//        	       	columns of pearson
//  @param	cols   	# of columns of data
////////////////////////////////////////////////////////////
void gpu_pearson(float *data, float *pearson, int rows, int cols)
{
	//setup execution parameters
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	float *average;
	float *stdev;
	
	//allocate memory
	cudaMalloc((void**)&average, rows*sizeof(float));
	cudaMemset(average, 0, rows*sizeof(float));

	cudaMalloc((void**)&stdev, rows*sizeof(float));
	cudaMemset(stdev, 0, rows*sizeof(float));
	
	cudaMemset(pearson, 0, rows*rows*sizeof(float));
	
	//compute number of blocks and number of threads
	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	dimGrid.x = (rows+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = dimGrid.x;
	

	//compute Pearson correlations
	GetAverages<<<dimGrid,dimBlock>>>(data, average, rows, cols);
	cudaThreadSynchronize();
	GetStdev<<<dimGrid,dimBlock>>>(data, average, stdev, rows, cols);	
	cudaThreadSynchronize();
	ComputeCorrelation<<<dimGrid,dimBlock>>>(data,average,stdev,
						pearson, rows, cols);
	
	//release memory
	cudaFree(average);
	cudaFree(stdev);
	average = NULL;
	stdev = NULL;
}

////////////////////////////////////////////////////////////
//  
//  Kernel wrapper function.  Transforms eigenvectors to 
//  component loadings.
//
//  @param	eigenvectors	eigenvectors to be transformed
//  @param	eigenvalues 	corresponding eigenvalues
//  @param	rows        	# of rows of eigenvectors
//  @param	cols        	# of columns of eigenvectors
////////////////////////////////////////////////////////////
void gpu_transform_eigenvectors(float *eigenvectors, float *eigenvalues, int rows, int cols)
{
	// setup execution parameters
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	
	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	dimGrid.x = (rows+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = dimGrid.x;
	
	Transform<<< dimGrid, dimBlock >>>(eigenvectors, eigenvalues, rows, cols);
}

////////////////////////////////////////////////////////////
//  
//  Kernel wrapper function.  Transposes matrix in 
//  into matrix out.
//
//  @param	in  	input matrix
//  @param	out 	output transposed matrix in
//  @param	rows	# of rows of in and out
//  @param	cols	# of columns of in and out
////////////////////////////////////////////////////////////
void gpu_transpose(float *in, float *out, int rows, int cols)
{
	// setup execution parameters
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	
	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	dimGrid.x = (rows+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = dimGrid.x;
	
	Transpose<<< dimGrid, dimBlock >>>(in, out, rows, cols);
}

////////////////////////////////////////////////////////////
//  
//  Kernel wrapper function. 
//  Get diagonal elements of matrix M.
//
//  @param	M    	input matrix
//  @param	diags	output, diagonal elements
//  @param	n    	# of rows and columns of M
////////////////////////////////////////////////////////////
void gpu_get_diagonals(float *M, float *diagonals, int n)
{
	// setup execution parameters
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	
	dimBlock.x = ONED_THREADS;
	dimGrid.x = n / dimBlock.x + (n % dimBlock.x == 0 ? 0 : 1);
	
	GetDiagonals<<< dimGrid, dimBlock >>>(M, diagonals, n);
}

////////////////////////////////////////////////////////////
//  
//  Kernel wrapper function. 
//  Get superdiagonal elements of matrix M.
//
//  @param	M       	input matrix
//  @param	supdiags	output, superdiag. elements
//  @param	n       	# of rows and columns of M
////////////////////////////////////////////////////////////
void gpu_get_superdiagonals(float *M, float *supdiags, int n)
{
	// setup execution parameters
	dim3 grid(1,1,1);
	dim3 block(1,1,1);
	
	block.x = ONED_THREADS;
	grid.x = n / block.x + (n % block.x == 0 ? 0 : 1);
	
	GetSuperDiagonals<<< grid, block >>>(M, supdiags, n);
}

////////////////////////////////////////////////////////////
//  
//  Prints current free memory on the GPU.
// 
////////////////////////////////////////////////////////////
void print_mem()
{
 	uint free, total;
	cuMemGetInfo(&free, &total);
	printf("\t\t Current Free GPU Memory: %i MB\n", (free>>20));  
}

////////////////////////////////////////////////////////////
//  
//  Entry point for GPU RMT algorithm.
//
//  @param	data       	loaded dataset
//  @param	random     	loaded random matrix
//  @param	pearson_out	data Pearson matrix (output)
//  @param	pear_in    	true if data is Pearson matrix
//  @param	curr_step  	current step in algorithm
//  @param	cols       	# of columns of data
//  @param	rows         	# of rows of data
////////////////////////////////////////////////////////////
extern "C" int runRMT(float *data, float *random, float *pearson_out, float *rotations, bool pear_in, int curr_step, int cols, int rows)
{
	//set device
	cudaSetDevice(cutGetMaxGflopsDeviceId());
	cublasStatus stat;
	culaStatus status;
        status = culaInitialize();
        checkStatus2(status);
	//declare variables
	float *g_data;
	float *g_workm;
        float *g_workv;

	float *g_transform;
	float *g_transpose;
	
	float *cd_eigen = new float[rows];
	float *cr_eigen = new float[rows];
	float *c_diags = new float[rows];
	float *c_superdiags = new float[rows-1];
	
	int meaningful;
	const int data_mem_size = sizeof(float) * rows * cols;
	const int pearson_mem_size = sizeof(float) * rows * rows;
	const int rows_mem_size = sizeof(float) * rows;
	int return_value;
	
	//allocate device memory
	cudaMalloc((void**)&g_data, data_mem_size);
	cudaMemset(g_data, 0, data_mem_size);
	
	cudaMalloc((void**)&g_workm, pearson_mem_size);
	cudaMemset(g_workm, 0, pearson_mem_size);

        cudaMalloc((void**)&g_workv, rows_mem_size);
	cudaMemset(g_workm, 0, rows_mem_size);

	cudaMalloc((void**)&g_transform, pearson_mem_size);
	cudaMemset(g_transform, 0, pearson_mem_size);

	cudaMalloc((void**)&g_transpose, pearson_mem_size);
	cudaMemset(g_transpose, 0, pearson_mem_size);
	
	if(rows >= LARGE_MATRIX) print_mem();

	if(!pear_in)
	{
		// copy host memory to device
		cudaMemcpy(g_data, random, data_mem_size, cudaMemcpyHostToDevice);
	
		printf("(On GPU) Step %i: Calculating Pearson correlation coefficients for random matrix...", curr_step);
		fflush(stdout);	

		gpu_pearson(g_data, g_workm, rows, cols);

		printf("Done.\n");
		fflush(stdout);
		curr_step++;
	}
	else
	{
		cudaMemcpy(g_workm, random, pearson_mem_size, cudaMemcpyHostToDevice);
	}
	
	if(rows >= LARGE_MATRIX) print_mem();
	
	//printf("(On GPU)\t Reducing random matrix to tridiagonal form...");
	//fflush(stdout);
	stat = cublasInit();
	
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	
	printf("(On CPU) Step %i: Computing eigenvalues for Random matrice...", curr_step);
	fflush(stdout);

        float *matrix = new float[pearson_mem_size]; 
	/*
	 * compute eigenvalues for the random matrix
	 */
    
        gpu_transpose(g_workm, g_transpose, rows, rows);
        cudaMemcpy(matrix, g_transpose, pearson_mem_size, cudaMemcpyDeviceToHost);
        culaSsyev('N', 'L', rows, matrix, rows, cr_eigen);
	return_value = checkStatus2(status);
        gpu_transpose(g_transpose, g_workm, rows, rows);

	printf("Done.\n");
	fflush(stdout);
	
	if(return_value == EXIT_FAILURE)
	{
		printf("\nOut of memory!!! Try running again...\n");
		fflush(stdout);
		
		cudaFree(g_workm);
		g_workm = NULL;
		cudaFree(g_transform);
		g_transform = NULL;
		cudaFree(g_data);
		g_data = NULL;
			
		delete[] cd_eigen;
		delete[] cr_eigen;
		delete[] c_diags;
		delete[] c_superdiags;
		cublasShutdown();
		cudaThreadExit();
		return EXIT_FAILURE;
	}
	
	if(!pear_in)
	{
		cudaMemset(g_data, 0, pearson_mem_size);
		cudaMemcpy(g_data, data, data_mem_size, cudaMemcpyHostToDevice);
		
		printf("(On GPU)\t Calculating Pearson correlation coefficients for data matrix...", curr_step);
		fflush(stdout);
	
		gpu_pearson(g_data, g_workm, rows, cols);

		//free unneeded memory
		cudaFree(g_data);
		g_data = NULL;
	
		cudaMemcpy(pearson_out, g_workm, pearson_mem_size, cudaMemcpyDeviceToHost);
		
		printf("Done.\n");
		fflush(stdout);
	}
	else
	{
		//free unneeded memory
		cudaFree(g_data);
		g_data = NULL;

		memcpy(pearson_out, data, data_mem_size);

		cudaMemcpy(g_workm, data, data_mem_size, cudaMemcpyHostToDevice);

	}
	
	if(rows >= LARGE_MATRIX) print_mem();

        printf("Done.\n");
	fflush(stdout);


	/*
	 * compute eigenvalues for the data matrix
	 */
	 
        printf("(On CPU) Step %i: Computing eigenvalues for data matrice...", curr_step);
	fflush(stdout);
 
        float *pw = new float[rows];
	/*
	 * compute eigenvalues for the data matrix
	 */
    
        gpu_transpose(g_workm, g_transpose, rows, rows);
        cudaMemcpy(matrix, g_transpose, pearson_mem_size, cudaMemcpyDeviceToHost);
        culaSsyev('N', 'L', rows, matrix, rows, cd_eigen);
	checkStatus2(status);
        gpu_transpose(g_transpose, g_workm, rows, rows);

	printf("Done.\n");
	fflush(stdout);
	
	if(return_value == EXIT_FAILURE)
	{
		printf("\nOut of memory!!! Try running again...\n");
		fflush(stdout);
		
		cudaFree(g_workm);
		g_workm = NULL;
		cudaFree(g_transform);
		g_transform = NULL;
			
		delete[] cd_eigen;
		delete[] cr_eigen;
		delete[] c_diags;
		delete[] c_superdiags;
		cublasShutdown();
		cudaThreadExit();
		return EXIT_FAILURE;
	}


	printf("\t\t Data Eigenvalues\tRandom Eigenvalues (Last 5)\n");
	for(int i = rows-4; i < rows; i++) printf("\t\t %f\t\t%f\n", cd_eigen[i], cr_eigen[i]);
        
	if(pear_in||cd_eigen[rows-2]<cr_eigen[rows-1])
	{
		if(rows < K)
			meaningful = rows;
		else
			meaningful = K;
               printf("(On CPU)\t Computing tridiagonal eigenvectors for the K largest eigenvalues (K = %i)...", meaningful);
	       fflush(stdout);
                culaSstein2(cd_eigen, rotations, matrix, meaningful,rows);

               
	}else{
               printf("(On CPU)\t Computing tridiagonal eigenvectors for the eigenvalues larger than the largest element in random %f ", cr_eigen[4]);
	       fflush(stdout);
               meaningful = culaSstein(cd_eigen, rotations, matrix, rows, cr_eigen[rows-1]);
       }
	
 	
        checkStatus2(status);
	cudaMemcpy(g_transpose, rotations, pearson_mem_size, cudaMemcpyHostToDevice);
	
	printf("Done.\n");
	fflush(stdout);

	printf("(On GPU)\t Backtransforming tridiagonal eigenvectors to those of the original matrix...");
	fflush(stdout);
	
	return_value = cublasSsormtr(g_transpose, g_transform, meaningful, rows);
	gpu_transpose(g_transpose, g_workm, rows, rows);
	
	printf("Done.\n");
	fflush(stdout);
	
	
	//transform eigenvectors to loadings
	printf("(On GPU) Step %i: Transforming eigenvectors to component loadings...", curr_step);
	fflush(stdout);
	
	cudaFree(g_transform);
	g_transform = NULL;
	
	cudaMemcpy(g_workv, cd_eigen, meaningful*sizeof(float), cudaMemcpyHostToDevice);
	gpu_transform_eigenvectors(g_workm, g_workv, rows, rows);
	
	printf("Done.\n");
	fflush(stdout);
	curr_step++;

	cudaFree(g_workv);
	g_workv = NULL;
	
	if(rows >= LARGE_MATRIX) print_mem();
	
	//rotate loadings using varimax
	printf("(On GPU) Step %i: Orthogonal rotation using Varimax...", curr_step);
	fflush(stdout);
	
	gpu_transpose(g_workm, g_transpose, rows, rows);
	gpu_varimax(g_transpose, rows, meaningful);
	gpu_transpose(g_transpose, g_workm, rows, rows);
	
	printf("Done.\n\n");
	fflush(stdout);
	
	//copy results from device to host
	cudaMemcpy(rotations, g_workm, pearson_mem_size, cudaMemcpyDeviceToHost);
	
	// cleanup memory
	cudaFree(g_workm);
	g_workm = NULL;
	cudaFree(g_transform);
	g_transform = NULL;
	
	delete[] cd_eigen;
	delete[] cr_eigen;
	delete[] c_diags;
	delete[] c_superdiags;
	cublasShutdown();

        checkStatus2(status);
        culaShutdown(); 
	cudaThreadExit();
	
	return meaningful;
}




//end rmta_gpu.cu
