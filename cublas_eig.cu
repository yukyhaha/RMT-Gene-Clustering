////////////////////////////////////////////////////////////
//  cublas_eig.cu
//
//  Provides functions for tridiagonal reduction, blocked
//  and unblocked QR decomposition, the QR algorithm for
//  computing eigenvalues/eigenvectors, and back-
//  transformation of Householder reflectors using CUBLAS.
//  Also provides some basic matrix construction functions.
//  Attempted to follow CUBLAS naming conventions as closely
//  as possible without using names for possible future
//  CUBLAS functions to avoid confusion.
//	
// 	Author: Yun Zhang
//	Date Created: November 29, 2011
//	Last Modified: April 18, 2012
//////////////////////////////////////////////////////////// 
#include "config.h"

////////////////////////////////////////////////////////////
//
// Kernel function.  Initializes M to an identity matrix.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows of M
// @param	cols	# of cols of M
//////////////////////////////////////////////////////////// 
__global__ void Identity(float *M, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx < cols && idy < rows)
	{
		if(idx == idy)
			M[idx*cols+idx] = (float) 1.0;
		else
			M[idx*cols+idy] = (float) 0.0;
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Initializes M to a zeroes matrix.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows of M
// @param	cols	# of cols of M
//////////////////////////////////////////////////////////// 
__global__ void Zeroes(float *M, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx < cols && idy < rows)
	{
		M[idx*cols+idy] = (float) 0.0;
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Initializes M to a ones matrix.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows of M
// @param	cols	# of cols of M
//////////////////////////////////////////////////////////// 
__global__ void Ones(float *M, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(idx < cols && idy < rows)
	{
		M[idx*cols+idy] = (float) 1.0;
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Given column x, computes tridiagonal
// reduction vector v.
//
// @param	x   	column to reduce
// @param	v   	reduction vector
// @param	norm	norm of x
// @param	col 	current column
// @param	n   	number of elements in x and v
////////////////////////////////////////////////////////////
__global__ void GetReduction(float *x, float *v, float norm, int col, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < n)
	{
		float s;
		float r;
		float u1;
	
		//sign
		if(x[col] < 0.0) s = 1;
		else if(x[col] == 0.0) s = 0;
		else s = -1;

		u1 = x[col] - s * norm;
		
		r = sqrtf(fabs(2 * norm * u1));

		//create vector
		if(idx > col && r != 0)
			v[idx] = x[idx] / r;
		else if(idx == col && r != 0)
			v[idx] = u1 / r;
		else
			v[idx] = 0.0;
	}
    	
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Given column x, computes Householder
// reflector vector v.
//
// @param	x   	column to generate vector from
// @param	v   	Householder vector
// @param	norm	norm of x
// @param	b   	beta
// @param	n   	number of elements in x and v
////////////////////////////////////////////////////////////
__global__ void GetHouseholder(float *x, float *v, float norm, float *b, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float s;
	float u1;

	if(idx < n)
	{
		if(x[0] < 0.0) s = 1;
		else if(x[0] == 0.0) s = 0;
		else s = -1;

		u1 = x[0] - s * norm;
		
		if(u1 != 0) v[idx] = x[idx] / u1;
	
	 	if(idx == 0)
		{
			v[idx] = 1;
			b[0] = (float) (-s * u1 / norm);
		}
	}
    	
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Computes a Wilkinson shift for 
// matrix M.
//
// @param	M    	matrix to shift
// @param	shift	holds computed shift
// @param	n    	# of rows and columns in M
////////////////////////////////////////////////////////////
__global__ void WilkinsonShift(float *M, float *shift, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx == 0)
	{
		float am;
		float ams;
		float bms;
		float delta;
		float mu;
		float sign;
		float denominator;
		
		//init
		denominator = 0.0;
		delta = 0.0;
		mu = 0.0;
		am = M[(n-1)*n+(n-1)];
		bms = M[(n-2)*n+(n-1)];
		ams = M[(n-2)*n+(n-2)];
		
		delta = (ams - am) / 2;
		
		if(delta <= 0.0) sign = 1;
		else sign = -1;
		
		//compute shift
		denominator = fabs(delta) + sqrtf(fabs(powf(delta,2) - powf(bms,2)));
		
		if(denominator != 0)
			mu = (am - (sign * powf(bms, 2))) / denominator;
		else
			mu = 0.0;
		
		shift[0] = mu;
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Applies shift s to matrix M.
//
// @param	M	matrix to shift
// @param	s	shift to apply
// @param	n	# of rows and columns in M
////////////////////////////////////////////////////////////
__global__ void ShiftMatrix(float *M, float *s, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < n)
	{
		M[idx*n+idx] = M[idx*n+idx] - s[0];
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Determines if a subdiagonal element
// is small enough to ignore
//
// @param	M	matrix to deflate
// @param	n	# of rows and columns in M
////////////////////////////////////////////////////////////
__global__ void DeflateMatrix(float *M, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx > 0 && idx < n)
	{
		float left;
		float right;
		
		left = fabs(M[(idx-1)*n+idx]) * fabs(M[idx*n+(idx-1)]);
		right = fabs(M[idx*n+idx]) * fabs(M[idx*n+idx] - M[(idx-1)*n+(idx-1)]);
		
		if(left <= (QR_EPSILON * right))
		{
			M[(idx-1)*n+idx] = 0.0;
		}
	}
}

////////////////////////////////////////////////////////////
//
// Kernel function.  Check if matrix M has met the 
// QR algorithm convergence criteria.
//
// @param	M     	matrix to check
// @param	bcount	holds count of converged subdiagonal
//       	      	elements
// @param	n     	# of rows and columns in M
////////////////////////////////////////////////////////////
__global__ void ConvergenceCheck(float *M, int *bcount, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx == 0)
	{
		int count = 0;
		
		for(int i = 1; i < n - 1; i++)
			if(fabs(M[i*n+i]) > fabs(M[(i+1)*n+(i+1)]) && 
				fabs(M[(i-1)*n+i]) == 0)
					count++;	

		bcount[0] = count;
	}
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Initialize matrix M to an
// identity matrix.  M should be stored as a single
// dimension array.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows in M
// @param	cols	# of columns in M
////////////////////////////////////////////////////////////
void eye(float *M, int rows, int cols)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);

	//set number of threads
	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	//set number of thread blocks
	dimGrid.x = (cols+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = (rows+dimBlock.y-1) / dimBlock.y;
	
	//run kernel
	Identity<<<dimGrid, dimBlock>>>(M, rows, cols);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Initialize matrix M to a
// zeroes matrix.  M should be stored as a single
// dimension array.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows in M
// @param	cols	# of columns in M
////////////////////////////////////////////////////////////
void zeroes(float *M, int rows, int cols)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);

	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	dimGrid.x = (cols+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = (rows+dimBlock.y-1) / dimBlock.y;

	Zeroes<<<dimGrid, dimBlock>>>(M, rows, cols);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Initialize matrix M to an
// ones matrix.  M should be stored as a single
// dimension array.
//
// @param	M   	matrix to initialize
// @param	rows	# of rows in M
// @param	cols	# of columns in M
////////////////////////////////////////////////////////////
void ones(float *M, int rows, int cols)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);

	dimBlock.x = TWOD_THREADS;
	dimBlock.y = TWOD_THREADS;
	
	dimGrid.x = (cols+dimBlock.x-1) / dimBlock.x;
	dimGrid.y = (rows+dimBlock.y-1) / dimBlock.y;
	
	Ones<<<dimGrid, dimBlock>>>(M, rows, cols);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Given column x, computes tridiagonal
// reduction vector v.
//
// @param	x          	column to reduce
// @param	v          	reduction vector
// @param	curr_column	current column
// @param	n          	number of elements in x and v
////////////////////////////////////////////////////////////
void reduce(float *x, float *v, int curr_column, int n)
{
	//define variables
	float norm;
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	
	dimBlock.x = ONED_THREADS;
	dimGrid.x = (n+dimBlock.x-1) / dimBlock.x; 
	
	//generate Householder vector and beta
	norm = cublasSnrm2(n-curr_column, x+curr_column, 1);
	GetReduction<<<dimGrid, dimBlock>>>(x, v, norm, curr_column, n);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Given column x, computes 
// Householder reflector v.
//
// @param	x   	column to generate vector from
// @param	v   	Householder vector
// @param	beta	holds computed beta
// @param	n   	number of elements in x and v
////////////////////////////////////////////////////////////
void householder(float *x, float *v, float *beta, int n)
{
	//define variables
	float norm;
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	float *b;
	
	dimBlock.x = ONED_THREADS;
	dimGrid.x = (n+dimBlock.x-1) / dimBlock.x; 
	cublasAlloc(1, sizeof(*x),(void**)&b);
	
	//generate Householder vector and beta
	norm = cublasSnrm2(n, x, 1);
	GetHouseholder<<<dimGrid, dimBlock>>>(x, v, norm, b, n);
	cublasGetVector(1, sizeof(float), b, 1, beta, 1);

	cublasFree(b);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Compute Wilkinson shift
// for matrix M.
//
// @param	M	matrix to check
// @param	n	# of rows and columns in M
////////////////////////////////////////////////////////////
float compute_shift(float *M, int n)
{
	//define variables
	float *wshift;
	float mu;
	
	cublasAlloc(1, sizeof(*M),(void**)&wshift);

	WilkinsonShift<<<1, 1>>>(M, wshift, n);
	cublasGetVector(1, sizeof(*M), wshift, 1, &mu, 1);
	
	cublasFree(wshift);
	
	return mu;
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Apply Wilkinson shift
// to matrix M.
//
// @param	M    	matrix to check
// @param	shift	shift to apply
// @param	n    	# of rows and columns in M
////////////////////////////////////////////////////////////
void apply_shift(float *M, float shift, int n)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	float *wshift;
	
	dimBlock.x = ONED_THREADS;
	dimGrid.x = (n+dimBlock.x-1) / dimBlock.x;

	cublasAlloc(1, sizeof(*M),(void**)&wshift);
	cublasSetVector(1, sizeof(float), &shift, 1, wshift, 1);
	ShiftMatrix<<<dimGrid, dimBlock>>>(M, wshift, n);
	
	cublasFree(wshift);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Check if subdiagonal elements
// of matrix M are insignificant.  If so, set element
// to 0.
//
// @param	M	matrix to deflate
// @param	n	# of rows and columns in M
////////////////////////////////////////////////////////////
void deflate(float *M, int n)
{
	//define variables
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(1,1,1);
	
	dimBlock.x = ONED_THREADS;
	dimGrid.x = (n+dimBlock.x-1) / dimBlock.x; 
	
	//deflate matrix
	DeflateMatrix<<<dimGrid, dimBlock>>>(M, n);
}

////////////////////////////////////////////////////////////
//
// Kernel wrapper function.  Check if matrix M has met the 
// QR algorithm convergence criteria.
//
// @param	M	matrix to check
// @param	n	# of rows and columns in M
////////////////////////////////////////////////////////////
bool check_convergence(float *M, int n)
{
	//define variables
	int *block_count;
	int b;
	
	cublasAlloc(1, sizeof(int),(void**)&block_count);

	ConvergenceCheck<<<1, 1>>>(M, block_count, n);
	cublasGetVector(1, sizeof(int), block_count, 1, &b, 1);
	
	cublasFree(block_count);
	
	if(b >= n-2) return true;
	else return false;
}

////////////////////////////////////////////////////////////
//
// Reduces symmetric matrix A to tridiagonal form using
// Householder reflections.
//
// @param	uplo	if 'A', apply reduction to whole 
//       	    	matrix
//       	    	if 'L', only apply to lower matrix
// @param	A   	n*n, matrix to reduce
// @param	Vs  	n*n, on exit, contains Householder 
//       	    	reflections
// @param	n   	# of rows and cols of Vs and A
//////////////////////////////////////////////////////////// 
int cublasSytrd(char uplo, float *A, float *Vs, int n)
{
	float c;
	float *w;
	cublasStatus stat;

	if(uplo != 'A' && uplo != 'L') return EXIT_FAILURE;
	
	stat = cublasAlloc(n, sizeof(*A), (void**)&w);
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	
	zeroes(Vs, n, n);
	for(int k = 0; k < n - 2; k++)
	{
		cublasScopy(n-k, A+(n*k)+k, 1, Vs+(n*k)+k, 1);
		
		reduce(A+(n*k)+k,Vs+(n*k)+k,1,n-k);
		
		if(uplo == 'A')
		{	
			//w = A * v; --level 2 BLAS (SGEMV)
			cublasSgemv('n', n-k, n-k, 1, A+(n*k)+k, n, Vs+(n*k)+k, 1, 0, w+k, 1);
			
			//w = w - c * v; level 1 BLAS (SAXPY)
			c = cublasSdot(n-k, Vs+(n*k)+k, 1, w+k, 1);
			cublasSaxpy(n-k, -c, Vs+(n*k)+k, 1, w+k, 1);
		
			//A = A - 2*v*w’ - 2*w*v’; level 2 BLAS (SGER)
			cublasSger(n-k, n-k, -2, Vs+(n*k)+k, 1, w+k, 1, A+(n*k)+k, n);
			cublasSger(n-k, n-k, -2, w+k, 1, Vs+(n*k)+k, 1, A+(n*k)+k, n);
		}
		else if(uplo == 'L')
		{	
			//w = A * v; --level 2 BLAS (SSYMV)
			cublasSsymv(uplo, n-k, 1, A+(n*k)+k, n, Vs+(n*k)+k, 1, 0, w+k, 1);

			//w = w - c * v; level 1 BLAS (SAXPY)
			c = cublasSdot(n-k, Vs+(n*k)+k, 1, w+k, 1);
			cublasSaxpy(n-k, -c, Vs+(n*k)+k, 1, w+k, 1);
		
			//A = A - 2*v*w’ - 2*w*v’; level 2 BLAS (SSYR2K)
			cublasSsyr2k(uplo, 'n', n-k, 1, -2, Vs+(n*k)+k, n, w+k, n, 1, A+(n*k)+k, n);
		}
	}
	
	cublasFree(w);
	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
//
// Unblocked QR decomposition.  
// A = QR where Q is an orthogonal matrix 
// and R is an upper triangular matrix
//
// @param	Q	n*n, on exit contains matrix
//       	 	orthogonal matrix Q
// @param	A	n*n, on entry, matrix to decompose
//       	 	on exit, upper triangle matrix R
// @param	n	# of rows and cols of A and Q
//////////////////////////////////////////////////////////// 
int cublasSyhqrd(float *Q, float *A, int n)
{
	float *beta = new float[1];
	
	//variables to reside in GPU memory
	float *v;
	float *work;
	cublasStatus stat;

	//allocate memory on device
	if((stat = cublasAlloc( n, sizeof(*A), (void**)&v)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n, sizeof(*A), (void**)&work)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	
	cudaMemset(v, 0, n*sizeof(float));
	cudaMemset(work, 0, n*sizeof(float));
	
	/* Initialize Q to an identity matrix */
	eye(Q, n, n);
	
	//begin QR decomposition
	for(int k = 0; k < n; k++)
	{
		/* Generate Householder vector and beta */
		householder(A+((n*k)+k), v, beta, n-k);

		/* Update matrix A */
		cublasSgemv('t', n-k, n-k, 1, A+((n*k)+k), n, v, 1, 0, work, 1);
		cublasSger(n-k, n-k, -beta[0], v, 1, work, 1, A+((n*k)+k), n);
		
		/* Update matrix Q */
		cublasSgemv('n', n, n-k, 1, Q+(n*k), n, v, 1, 0, work, 1);
		cublasSger(n, n-k, -beta[0], work, 1, v, 1, Q+(n*k), n);
	}
	
	//cleanup
	cublasFree(v);
	cublasFree(work);

	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
//
// Determine block size to use for the blocked Householder
// method cublasSybqrd.
// cublasSybqrd helper function.
//
// @param	n	# of rows and columns of matrix
//////////////////////////////////////////////////////////// 
int partition(int n)
{
	int factor;

	factor = 1;
	for(int i = 2; i < (int) (n/4); i++)
	{
		if(n % i == 0) factor = i;
	}

	return factor;
}

////////////////////////////////////////////////////////////
//
// Computes W and Y to be applied to Q and R, using
// Householder vectors (stored in V) and beta values
// computed for the current block r.
// cublasSybqrd helper function.
//
// @param	W           	matrix to apply to Q & R
// @param	Y          	matrix to apply to Q & R
// @param	V          	Householder vectors of block r
// @param	beta       	betas of block r
// @param	work_vector	temporary work vector
// @param	r          	block size
// @param	n          	# of rows and columns
////////////////////////////////////////////////////////////
void computeWY(float *W, float *Y, float *V, float *beta, float *work_vector, int r, int n)
{	
	//Y = V(1:n, 1)
	cublasScopy(n, V, 1, Y, 1);

	//W = -B(1) * V(1:n, 1)
	cublasSaxpy(n, -beta[0], V, 1, W, 1);
	
	zeroes(work_vector,n ,1);
	for(int j = 1; j < r; j++)
	{
		//W(:,j) =-B(j)*v - B(j)*WY^H v		
		cublasScopy(n, V+(n*j), 1, W+(n*j), 1);
		cublasSgemv('t', n, r, 1, Y, n, W+(n*j), 1, 0, work_vector, 1);
		cublasSgemv('n', n, r, -beta[j], W, n, work_vector, 1, -beta[j], W+(n*j), 1);
		
		//Y = [Y v]
		cublasScopy(n, V+(n*j), 1, Y+(n*j), 1);
	}
}

////////////////////////////////////////////////////////////
//
// Blocked Householder QR decomposition.  Uses mostly
// matrix-matrix products to update Q and R.
//
// @param	Q	n*n, on exit contains matrix
//       	 	orthogonal matrix Q
// @param	A	n*n, on entry, matrix to decompose
//       	 	on exit, upper triangle matrix R
// @param	n	# of rows and cols of A and Q	
////////////////////////////////////////////////////////////
int cublasSybqrd(float *Q, float *A, int n)
{
	//local CPU variables
	float *beta;
	float *b;
	int r;
	int s;
	int u;
	int sr;
	int sc;
	
	//variables to reside in GPU memory
	float *V;
	float *W;
	float *Y;
	float *work_matrix;
	float *work_vector;
	cublasStatus stat;
	
	/* Partition A into r blocks */
	r = partition(n);

	//allocate memory on device
	if((stat = cublasAlloc( n, sizeof(*A), (void**)&work_vector)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n*r, sizeof(*A), (void**)&V)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n*r, sizeof(*A), (void**)&W)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n*r, sizeof(*A), (void**)&Y)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat=cublasAlloc( n*n,sizeof(*A),(void**)&work_matrix))!=CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	b = new float[1];
	beta = new float[r];
	
	//set initial values
	zeroes(work_matrix, n, n);
	zeroes(work_vector, n, 1);
	
	/* Initialize Q to an identity matrix */
	eye(Q, n, n);
	
	//begin QR algorithm
	for(int k = 0; k < (n/r); k++)
	{
		//reset variables
		zeroes(W, n, r);
		zeroes(Y, n, r);
		zeroes(V, n, r);
		
		s = k * r;
		for(int j = 0; j < r; j++)
		{
			u = s + j;
			sr = n - u;
			sc = (s + r) - u;
			beta[j] = 0.0;
			
			/* Generate Householder vector and beta */
			householder(A+(n*u)+u, V+(n*j)+j, b, n-u);

			/* Update current block of A */
			cublasSgemv('t', sr, sc, 1, A+(n*u)+u, n, V+(n*j)+j, 1, 0, work_vector, 1);
			cublasSger(sr, sc, -b[0], V+(n*j)+j, 1, work_vector, 1, A+(n*u)+u, n);

			//beta = 2 / vv^T
			beta[j] = b[0];
		}
		
		/* Compute W and Y from V and B */
		computeWY(W, Y, V, beta, work_vector, r, n);
		
		/* Update blocks A_k+1 to A_n/r */
		for(int i = k+1; i < (n/r); i++)
		{
			int t = i * r;
			
			cublasSgemm('t', 'n', n, r, n-s, 1, W, n, A+(n*t)+s, n, 0, work_matrix, n);
			cublasSgemm('n', 'n', n-s, r, r, 1, Y, n, work_matrix, n, 1, A+(n*t)+s, n);
		}
		
		/* Update Q */
		cublasSgemm('n', 'n', n, r, n-s, 1, Q+(n*s), n, W, n, 0, work_matrix, n);
		cublasSgemm('n', 't', n, n-s, r, 1, work_matrix, n, Y, n, 1, Q+(n*s), n);
	}

	//cleanup
	cublasFree(V);
	cublasFree(W);
	cublasFree(Y);
	cublasFree(work_matrix);
	cublasFree(work_vector);

	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
//
// QR algorithm using blocked Householder QR decomposition,
// deflation, and Wilkinson shift.
//
// @param	Q	n*n, on exit, contains eigenvectors	
// @param	R	n*n, on entry, contains matrix
//       	 	to find eigenstates.  on exit,
//       	 	contains eigenvalues in the diagonal
//       	 	elements.
// @param	n	# of rows and cols of Q and R
////////////////////////////////////////////////////////////
int cublasSsybqr(float *Q, float *R, int n)
{
	float *I;
	float *H;
	float *T;
	float *work;
	float s;
	bool converged;
	int iter;
	cublasStatus stat;
	
	if((stat = cublasAlloc( n, sizeof(*R), (void**)&work)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n*n, sizeof(*R), (void**)&I)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( 1, sizeof(*R), (void**)&T)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if((stat = cublasAlloc( n*n, sizeof(*R), (void**)&H)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	
	//initialize variables
	eye(I, n, n);
	zeroes(H, n, n);
	converged = false;
	iter = 0;
	s = 0.0;
	
	//reduce to tridiagonal form
	cublasSytrd('A', R, H, n);

	//begin QR algorithm
	while(!converged)
	{	
		//shift matrix
		s = compute_shift(R, n);
		apply_shift(R, s, n);
		
		//QR decomposition
		cublasSybqrd(Q,R,n);

		//update eigenvalue matrix
		cublasSgemm('n','n',n,n,n,1,R,n,Q,n,0,R,n);
		
		//update eigenvector matrix
		cublasSgemm('n','n',n,n,n,1,I,n,Q,n,0,I,n);
		
		//shift results back
		s = -s;
		apply_shift(R, s, n);
		
		//deflate matrix
		deflate(R, n);
		
		//check for convergence
		converged = check_convergence(R, n);
		if(iter >= QR_ITERATIONS) converged = true;
		
		iter++;
	}
	
	//back transform eigenvectors
	for(int i = n - 2; i >= 0; i--)
	{
		cublasSgemv('t', n, n, 1, I, n, H+(n*i), 1, 0, work, 1);
		cublasSger(n, n, -2, H+(n*i), 1, work, 1, I, n);
	}
	cublasScopy(n*n, I, 1, Q, 1);
	
	//cleanup
	cublasFree(I);
	cublasFree(work);
	cublasFree(T);
	cublasFree(H);
	
	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
//
// Backtransforms matrix Q using reflector matrix H.
//
// @param	Q	matrix to transform	
// @param	H	reflector matrix	
// @param	k	# of columns to apply reflector to
// @param	n	# of rows and cols of Q and H
////////////////////////////////////////////////////////////
int cublasSsormtr(float *Q, float *H, int k, int n)
{
	float *work;
	cublasStatus stat;
	
	if((stat = cublasAlloc(n, sizeof(*Q), (void**)&work)) != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;
	if(stat != CUBLAS_STATUS_SUCCESS) return EXIT_FAILURE;

	zeroes(work, n, 1);
	
	//back transform eigenvectors
	for(int i = n - 2; i >= 0; i--)
	{
		cublasSgemv('t', n, k, 1, Q, n, H+(n*i), 1, 0, work, 1);
		cublasSger(n, k, -2, H+(n*i), 1, work, 1, Q, n);
	}
	
	//cleanup
	cublasFree(work);
	
	return EXIT_SUCCESS;
}
//end cublas_eig.cu