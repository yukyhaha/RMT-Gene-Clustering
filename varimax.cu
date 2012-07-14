////////////////////////////////////////////////////////////
//	varimax.cu
//
//	Provides functions for the Varimax rotation method
//	on the GPU.  Note: Not fully parallelized yet.
//
//	Author: Yun Zhang
//	Date Created: November 29, 2011
//	Last Modified: April 18, 2012
////////////////////////////////////////////////////////////
#include "config.h"

////////////////////////////////////////////////////////////
//	
// Kernel function. Compute current variance of loadings.
//
// @param	load		array of component loadings 
//       	        	to be rotated
// @param	crit		holds computed criteria
// @param	rows		# of rows of loadings
// @param	cols		# of columns of loadings
////////////////////////////////////////////////////////////
__global__ void CompCriteria(float *load, float *crit, 
			     int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
		
	if(idx == 0)
	{
		float c;
		float r = (float) rows;

		c = 0.0;
		for (int i = 0; i < cols; i++)
		{
			float s2 = 0.0;
			for(int t = 0; t < rows; t++)
			{
				int ind = i*rows+t;
				float sq=load[ind]*load[ind];

				s2 += sq;
				c += sq*sq;
			}
			c -= s2*s2/r;
		}
		crit[0] = c;
	}
}

////////////////////////////////////////////////////////////
//	
// Kernel function. Compute rotation factors for column k
// and j.
//
// @param	load		array of component loadings 
//       	        	to be rotated
// @param	j   		first column to rotate
// @param	k   		second column to rotate
// @param	angl		holds computed angle
// @param	rows		# of rows of loadings
////////////////////////////////////////////////////////////
__global__ void RFactors(float *load,int j,int k,float *angl,int rows)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx == 0)
	{
		float a;
		float b;
		float c;
		float d;
		float r;
		float denominator;
		float numerator;
		
		a = 0.0;
		b = 0.0;
		c = 0.0;
		d = 0.0;
		r = (float) rows;
		
		//compute factors
		for (int i = 0; i < rows; i++)
		{
			float c2 = load[j*rows+i]*load[j*rows+i];
			c2 -= load[k*rows+i]*load[k*rows+i];
			float s2 = 2.0*load[j*rows+i]*load[k*rows+i];

			a += c2;
			b += s2;
			c += c2*c2 - s2*s2;
			d += c2*s2;
		}
		denominator = (r*c - (a*a - b*b));
		numerator = 2.0*(r*d - a*b);

		//compute angle
		angl[0] = 0.25f * atan2f(numerator,denominator);
	}
}

////////////////////////////////////////////////////////////
//	
// Kernel function.  Rotate column j and k of load using angle.
//
// @param	load		array of component loadings 
//       	        	to be rotated
// @param	j   		first column to rotate
// @param	k   		second column to rotate
// @param	angl		rotation angle
// @param	rows		# of rows of loadings
////////////////////////////////////////////////////////////
__global__ void Rotate(float *load, int j, int k, 
			float *angle, int rows)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int jindex = j*rows+idx;
	int kindex = k*rows+idx;
	
	if(idx < rows)
	{
		float c = cosf(angle[0]);
		float s = sinf(angle[0]);
		float t = c*load[jindex]+s*load[kindex];
		load[kindex] = -s*load[jindex]+c*load[kindex];
		load[jindex] = t;
	}
}

////////////////////////////////////////////////////////////
//	
// Compute current variance of loadings.
//
// @param	load		array of component loadings 
//       	    		to be rotated
// @param	rows		# of rows of loadings
// @param	cols		# of columns of loadings
////////////////////////////////////////////////////////////
float gpu_comp_criteria(float *load, int rows, int cols)
{
	dim3 grid(1,1,1);
	dim3 block(1,1,1);     
	int num_blocks;
	float *g_crit;
	float *sums;
	
	block.x = ONED_THREADS;
	grid.x = (cols+block.x-1) / block.x; 
	num_blocks = grid.x;
	sums = new float[num_blocks];
	cudaMalloc((void**)&g_crit, num_blocks*sizeof(float));
	cudaMemset(g_crit, 0, num_blocks*sizeof(float));
	
	CompCriteria<<<grid, block>>>(load,g_crit,rows,cols);
	cublasGetVector(1, sizeof(float), g_crit, 1, sums, 1);
	
	cudaFree(g_crit);
	return sums[0];
}

////////////////////////////////////////////////////////////
//	
// Compute rotation angle and rotate columns j and k
// in array load using kernels on the GPU.
//
// @param	load		array of component loadings 
//       	    		to be rotated
// @param	j   		first column to rotate
// @param	k   		second column to rotate
// @param	rows		# of rows of loadings
// @param	cols		# of columns of loadings
////////////////////////////////////////////////////////////
void rotate(float *load,int j,int k,int rows,int cols)
{
	float *g_angl;
	dim3 grid(1,1,1);
	dim3 block(1,1,1);

	cudaMalloc((void**)&g_angl, sizeof(float));
	
	block.x = ONED_THREADS;
	grid.x = (rows+block.x-1) / block.x; 
	
	RFactors<<<grid, block>>>(load,j,k,g_angl,rows);
	Rotate<<<grid, block>>>(load,j,k,g_angl,rows);
	
	cudaFree(g_angl);
}

////////////////////////////////////////////////////////////
//	
// Varimax rotation on the GPU.
//
// @param	loadings	array of component loadings 
//       	        	to be rotated
// @param	m		# of rows of loadings
// @param	n		# of columns of loadings
////////////////////////////////////////////////////////////
int gpu_varimax(float *loadings, int m, int n)
{
	int iter;
	bool done; 
	float trot;
 	float crit;
		
	//initialize variables
	iter = 0;
	crit = gpu_comp_criteria(loadings, m, n);
	done = false;
	
	//begin varimax
	while(!done && iter < VARIMAX_ITERATION)
	{
		float old_crit = crit;

		//rotate each column
		for (int j = 0; j < n-1; j++)
		{
			for (int k = j + 1;k < n ;k++)
			{
				rotate(loadings,j,k,m,n);
			}
		}
		iter++;
		
		//check stopping criterion
		crit = gpu_comp_criteria(loadings, m, n);
		trot = (crit>0.0f) ? (crit-old_crit)/crit : 0.0f;

		if(trot > VARIMAX_EPSILON)
		{
			done = true;
		}
	}
	
	return EXIT_SUCCESS;
}
//end varimax.cu
