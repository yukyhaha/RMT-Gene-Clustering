////////////////////////////////////////////////////////////
//  rmta.h
//
//  Provides basic functions needed for microarray
//  analysis using PCA.  Also provides the sequential
//  algorithm.
//
//  Author: Yun Zhang
//  Date Created: November 29, 2011
//  Last Modified: April 15, 2012
////////////////////////////////////////////////////////////
//include system
#include "config.h"
#include "culapackdevice.h"

//global variables
int t_rows;
int t_cols;
string *names;

//include necessary LAPACK functions
extern "C" void ssytrd_(char *uplo, int *n, float *a, int *lda,
				float *d, float *e, float *tau,
				float *work, int *lwork, int *info);

extern "C" void sstebz_(char *range, char *order, int *n,
				float *vl, float *vu, int *il, int *iu,
				float *abstol, float *d, float *e,
				int *m, int *nsplit, float *w, int *iblock,
				int *isplit, float *work, int *iwork,
				int *info);

extern "C" void sstein_(int *n, float *d, float *e, int *m,
				float *w, int *iblock, int *isplit, float *z,
				int *ldz, float *work, int *iwork, int *ifail,
				int *info);

extern "C" void sormtr_(char *side, char *uplo, char *trans,
				int *m, int *n, float *a, int *lda, float *tau,
				float *c, int *ldc, float *work, int *lwork,
				int *info);

extern "C" float slansy_(char *norm, char *uplo, int *n,
				float *a, int *lda, float *work);

extern "C" void sscal_(int *n, float *sa, float *sx, int *incx);

extern "C" float slanst_(char *norm, int *n, float *d,
				 float *e);

extern "C" float slamch_(char *cmach);

float get_max(float *array, int rows, int cols);
void matrix_to_file(float *array, int rows, int cols, const char *filename, char sep);

int checkStatus2(culaStatus status)
{
    char buf[80];

    if(!status)
        return 5;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);

    culaShutdown();
    return EXIT_FAILURE;
}


////////////////////////////////////////////////////////////
//
//	Computes eigenvectors of the k largest supplied
//	eigenvalues for a tridiagonal matrix.  Eigenvalues
//  must be in ascending order on entry.  On exit,
//  eigenvalues and their corresponding eigenvectors
//  will be in descending order.
//
//  @param	eigenvalues 	eigenvalues of trid. matrix
//  @param	eigenvectors	on exit, contains eigenvectors
//  @param	d           	diag. elements of trid. matrix
//	@param	e           	superdiag. elements
//	@param	k           	# of eigenvectors to compute
//	@param	n           	size of d
////////////////////////////////////////////////////////////
int culaSstein(float* eigenvalues, float *eigenvectors, float *matrix, int n,float vl)
{

	int *ifail= new int[n];

       float vu = 3.40282e+38;
       float abstol=0.0;
       int m = 0;
       float* w = new float[n];
       float* z = new float[n*n];      
 
      culaStatus status = culaSsyevx('V', 'V', 'L', n, matrix, n,vl, vu, n-m+1, n, abstol, &m, w, z, n, ifail);
      checkStatus2(status);
    
      for(int i=0; i< m; i++){
          eigenvalues[i] = w[m-1-i];
      }

      //copy vectors and eigenvalues 
      //to output array in reverse (for descending order)

	int  count = 0;
	for(int i = m-1; i >= 0; i--)
	{
		for(int j = 0; j < n; j++)
		{
		    eigenvectors[count*n+j] = z[i*n+j];

		}
		count++;

	}
      

	for(int i = m; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			eigenvectors[i*n+j] = 0.0;
		}
	}

	//cleanup

	delete[] ifail;
        delete[] w;
        delete[] z; 
        return m;
}

////////////////////////////////////////////////////////////
//
//	Loads dataset into memory.
//
//  @param	filename 	path to dataset
//  @param	separator	character that separates columns
////////////////////////////////////////////////////////////
float *load_data(const string &filename, char separator)
{
	ifstream data_file;
	istringstream iss;
	string line, token;
	int total_rows, total_columns, row, column;
	float *data_set;
	int rows = 0, columns = 0;

	//open file
	data_file.open(filename.c_str());

	//make sure file is open
  	if (data_file.is_open())
  	{

		//get the total number of rows and columns in the data file.
		total_rows = 0;
		total_columns = 0;
		while (!data_file.eof())
		{
      		getline(data_file,line);
			iss.str(line);

			while (getline(iss, token, separator))
			{
					total_columns++;
			}
			total_rows++;
			iss.clear();

			if(data_file.peek() == data_file.eof()) break;
    	}

		//allocate memory for dataset
		rows = total_rows;
		columns = (total_columns / total_rows) - 1;

		data_set = new float[rows*columns];
		names = new string[rows];

		//go through data file again and load it
		data_file.clear();
		data_file.seekg(0);
		row = 0;
		column = 0;
		while (!data_file.eof())
		{
      		getline(data_file,line);
			iss.str(line);
			int count = 0;
			while (getline(iss, token, separator))
			{
				if(count > 0)
				{
					data_set[row*columns+column] = strtod(token.c_str(), NULL);
					column++;
				}
				else
				{
					names[row] = token.c_str();
				}
				count++;
			}
			row++;
			iss.clear();

			if(data_file.peek() == data_file.eof()) break;
			if(column != 0 and column % columns == 0) column = 0;
    	}
    	data_file.close();
    	t_rows = rows;
    	t_cols = columns;
		return data_set;
  	}
	else
		return NULL;
}

////////////////////////////////////////////////////////////
//
//  Create random data matrix where each element e
//  satisfies low <= e <= high.
//
//  @param	rows	# of rows for created matrix
//  @param	cols	# of columns for created matrix
//  @param	low 	lowest value allowed for an element
//	@param	high	highest value allowed for an element
////////////////////////////////////////////////////////////
float *create_random_matrix(int rows, int cols, float low, float high)
{
	float *random;
	float r;

	random = new float[rows*cols];

	srand(time(NULL));

	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			r = (float) 1-((float) rand() / (float) RAND_MAX)*2;

			while(r < low || r > high)
				r = (float) (((float) rand() / (float) RAND_MAX)*high -
				((float) rand() / (float) RAND_MAX)*fabs(low));

			random[(i*cols)+j] = r;
		}
	}

	return random;
}

////////////////////////////////////////////////////////////
//
//  Create symmetric random data matrix where each element
//  e satisfies low <= e <= high.
//
//  @param	rows	# of rows for created matrix
//  @param	cols	# of columns for created matrix
//  @param	low 	lowest value allowed for an element
//	@param	high	highest value allowed for an element
////////////////////////////////////////////////////////////
float *create_symrand_matrix(int rows, int cols, float low, float high)
{
	float *random;
	float r;

	random = new float[rows*cols];

	srand(time(NULL));

	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			if(i > j) continue;
			r = (float) 1-((float) rand() / (float) RAND_MAX)*2;

			while(r < low || r > high)
				r = (float) (((float) rand() / (float) RAND_MAX)*high -
				((float) rand() / (float) RAND_MAX)*fabs(low));

			random[(i*cols)+j] = r;
			random[(j*cols)+i] = r;
		}
	}

	return random;
}

////////////////////////////////////////////////////////////
//
//  Normalizes each element of dataset to:
//  dataset[element] - average[row] / stdev[row]
//
//  @param	dataset	single dimension array in row-major
//        	       	that holds dataset
//  @param	rows   	# of rows of dataset
//  @param	cols   	# of columns of dataset
////////////////////////////////////////////////////////////
void normalize(float *dataset, int rows, int cols)
{
	float *average;
	float *stdev;

	average = new float[rows*cols];
	stdev = new float[rows*cols];

	//calculate averages
	for(int i=0;i<rows;i++)
	{
		average[i] = 0;
		for(int j=0;j<cols;j++)
		{
			if(isnan(dataset[(i*cols)+j]))
				dataset[(i*cols)+j] = 0.0;

			average[i] += dataset[(i*cols)+j];
		}
		average[i] = average[i] / cols;
	}

	//calculate standard deviations
	for(int i=0;i<rows;i++)
	{
		stdev[i] = 0;
		for(int j=0;j<cols;j++)
		{
			stdev[i] += powf(dataset[i*cols+j] - average[i],2);
		}
		stdev[i] = sqrtf(fabs(stdev[i]));

		if(stdev[i] == 0.0)
			stdev[i] = 1.0;
	}

	//normalize the data
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			dataset[i*cols+j] = (dataset[i*cols+j] - average[i]) / stdev[i];
		}
	}

	delete[] average;
	delete[] stdev;
}

////////////////////////////////////////////////////////////
//
//  calculate_pearsons helper function.  Computes Pearson
//  correlation of x and y.
//
//  @param	x      	first element to calculate
//  @param	y      	second element
//  @param	dataset	loaded dataset
//  @param	average	row averages of dataset
//  @param	stdev  	row standard deviations
//	@param	cols   	# of columns of dataset
////////////////////////////////////////////////////////////
float p(int x, int y, float *dataset, float *average, float *stdev, int cols)
{
	float pc = 0;

	for(int i = 0; i < cols; i++)
	{
		pc += (dataset[x*cols+i] - average[x])*(dataset[y*cols+i]-average[y]);
	}

	pc /= ((cols-1) * stdev[x] * stdev[y]);

	return pc;
}

////////////////////////////////////////////////////////////
//
//	Calculates the Pearson correlation matrix of dataset.
//
//  @param	pearson	output, contains computed Pearson
//  @param	dataset	array to compute Pearson coefficients of
//  @param	rows   	# of rows of dataset
//  @param	cols   	# of columns of dataset
////////////////////////////////////////////////////////////
void calculate_pearsons(float *pearson, float *dataset, int rows, int cols)
{
	float *average;
	float *stdev;
	float pear;

	average = new float[rows*cols];
	stdev = new float[rows*cols];

	//calculate averages
	for(int i = 0; i < rows; i++)
	{
		average[i] = 0;
		for(int j = 0; j < cols; j++)
		{
			average[i] += dataset[(i*cols)+j];
		}
		average[i] = average[i] / cols;
	}

	//calculate standard deviations
	for(int i = 0; i < rows; i++)
	{
		stdev[i] = 0;
		for(int j = 0; j < cols; j++)
		{
			stdev[i] += powf(dataset[i*cols+j], 2) - powf(average[i],2);
		}
		stdev[i] = sqrtf(fabs(stdev[i]/(cols-1)));

		if(stdev[i] == 0) stdev[i] = 1.0;
	}

	//calculate pearson coefficients
	for(int i = 0; i < rows; i++)
	{
		for(int j = 0; j < rows;j++)
		{
			if(i > j) continue;
			pear = p(i, j, dataset, average, stdev, cols);

			pearson[(i*rows)+j] = pear;
			pearson[(j*rows)+i] = pear;
		}
	}

	delete[] average;
	delete[] stdev;
}

////////////////////////////////////////////////////////////
//
//	Uses LAPACK to reduce symmetric matrix A to tridiagonal
//	form.
//	non-diagonal or sub-diagonal elements will contain
//	transformation reflectors.
//
//  @param	A  	matrix to reduce
//  @param	tau	on exit, contains computed betas
//  @param	n  	# of rows and columns of A
////////////////////////////////////////////////////////////
void ssytrd(float *A, float *tau, int n)
{
	char uplo;
	int lwork, info;
	float *work;
	float *work0;
	float *d;
	float *e;

	uplo = 'U';

	lwork=-1;
	work0 = new float[1];
	d = new float[n];
	e = new float[n-1];

	//get scale factor
	char s = 'S';
	char p = 'P';
	float safmin = slamch_(&s);
	float eps = slamch_(&p);
	float smlnum = safmin / eps;
	float bignum = 1 / smlnum;
	float rmin = sqrtf(smlnum);
	float rmax = sqrtf(bignum);
	char norm;
	float fnorm;
	float sigma;
	float *nwork = new float[n];
	int incx = 1;
	int n2 = n*n;

	//scale if necessary
	norm = 'M';
	fnorm = slansy_(&norm, &uplo, &n, A, &n, nwork);

	if(fnorm > 0 && fnorm < rmin)
		sigma = rmin / fnorm;
	else if(fnorm > rmax)
		sigma = rmax / fnorm;
	else
		sigma = 1;

	if(sigma != 1)
		sscal_(&n2, &sigma, A, &incx);

	//workspace query to determine optimal size of work array
	ssytrd_(&uplo, &n, A, &n, d, e, tau, work0, &lwork,
			&info);

	lwork = (int) work0[0];
	work = new float[lwork];

	//reduce to tridiagonal form
	ssytrd_(&uplo, &n, A, &n, d, e, tau, work, &lwork,
			&info);

	//remove scale
	if(sigma != 1)
	{
		sigma = 1/sigma;
		sscal_(&n2, &sigma, A, &incx);
		sscal_(&n, &sigma, d, &incx);
		sscal_(&n, &sigma, e, &incx);
	}

	delete[] nwork;
	delete[] work0;
	delete[] work;
	delete[] d;
	delete[] e;
}

////////////////////////////////////////////////////////////
//
//	Uses LAPACK to compute eigenvalues of a tridiagonal
//	matrix.
//
//  @param	eigenvalues	will contain eigenvalues on exit
//  @param	d          	diagonal elements of trid. matrix
//  @param	e          	superdiagonals of trid. matrix
//	@param	n          	size of d
////////////////////////////////////////////////////////////
void sstebz(float *eigenvalues, float *d, float *e, int n)
{
	//setup variables
	int m;
	int nsplit;
	int info;
	char range;
	char order;
	float *work;
	int *iwork;
	float abs = 0;
	int i = 0;
	float f = 0.0;
	int *iblock;
	int *isplit;

	range = 'A';
	order = 'E';

	work = new float[4*n];
	iwork = new int[3*n];
	iblock = new int[n];
	isplit = new int[n];

	//get scale factor
	char s = 'S';
	char p = 'P';
	float safmin = slamch_(&s);
	float eps = slamch_(&p);
	float smlnum = safmin / eps;
	float bignum = 1 / smlnum;
	float rmin = sqrtf(smlnum);
	float rmax = sqrtf(bignum);
	char norm;
	float fnorm;
	float sigma;
	int incx = 1;

	//scale if necessary
	norm = 'M';
	fnorm = slanst_(&norm, &n, d, e);

	if(fnorm > 0 && fnorm < rmin)
		sigma = rmin / fnorm;
	else if(fnorm > rmax)
		sigma = rmax / fnorm;
	else
		sigma = 1;

	for(int i = 0; i < n; i++)
	{
		if(fabs(d[i]) < eps) sigma = 1 / eps;
	}

	if(sigma != 1)
	{
		sscal_(&n, &sigma, d, &incx);
		sscal_(&n, &sigma, e, &incx);
	}

	//compute eigenvalues
	sstebz_(&range, &order, &n, &f, &f, &i, &i, &abs,
			d, e, &m, &nsplit, eigenvalues,
			iblock, isplit, work, iwork, &info);

	//remove scaling
	if(sigma != 1)
	{
		sigma = 1/sigma;
		sscal_(&n, &sigma, eigenvalues, &incx);
		sscal_(&n, &sigma, d, &incx);
		sscal_(&n, &sigma, e, &incx);
	}

	//cleanup
	delete[] work;
	delete[] iwork;
	delete[] iblock;
	delete[] isplit;
}

////////////////////////////////////////////////////////////
//
//	Computes eigenvectors of the k largest supplied
//	eigenvalues for a tridiagonal matrix.  Eigenvalues
//  must be in ascending order on entry.  On exit,
//  eigenvalues and their corresponding eigenvectors
//  will be in descending order.
//
//  @param	eigenvalues 	eigenvalues of trid. matrix
//  @param	eigenvectors	on exit, contains eigenvectors
//  @param	d           	diag. elements of trid. matrix
//	@param	e           	superdiag. elements
//	@param	k           	# of eigenvectors to compute
//	@param	n           	size of d
////////////////////////////////////////////////////////////
void sstein(float *eigenvalues, float *eigenvectors, float *d, float *e, int k, int n)
{
	//setup variables
	int info;
	int *ifail;
	float *work;
	int *iwork;
	float *vecs;
	int *iblock;
	int *isplit;

	ifail = new int[k];
	work = new float[5*n];
	iwork = new int[n];
	vecs = new float[n*n];
	iblock = new int[n];
	isplit = new int[n];

	//set iblock
	for(int i = 0; i < n; i++)
	{
		iblock[i] = 1;
	}

	//set isplit
	isplit[0] = n;

	//initialize vectors
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			vecs[i*n+j] = 0.0;
		}
	}

	//compute eigenvectors
	sstein_(&n, d, e, &k, eigenvalues+(n-k), iblock, isplit,
			vecs, &n, work, iwork, ifail,
			&info);

	//copy vectors and eigenvalues
	//to output array in reverse (for descending order)
	int count = 0;
	for(int i = k-1; i >= 0; i--)
	{
		for(int j = 0; j < n; j++)
		{
			eigenvectors[count*n+j] = vecs[i*n+j];
		}
		count++;
	}

	//set eigenvalues in reverse order (descending order)
	float temp;
	for(int i = 0; i < floor(n/2); i++)
	{
		temp = eigenvalues[n-i-1];
		eigenvalues[n-i-1] = eigenvalues[i];
		eigenvalues[i] = temp;
	}

	//make sure unwanted vectors are zero
	for(int i = k; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			eigenvectors[i*n+j] = 0.0;
		}
	}

	//cleanup
	delete[] vecs;
	delete[] ifail;
	delete[] work;
	delete[] iwork;
	delete[] iblock;
	delete[] isplit;
}

////////////////////////////////////////////////////////////
//
//	Backtransforms tridiagonal eigenvectors to those of
//  the original matrix.
//
//  @param	eigenvectors	eigenvectors to transform
//  @param	transform   	transform. matrix from ssytrd
//  @param	tau         	reflectors from ssytrd
//	@param	n           	# of rows and columns of eigen.
////////////////////////////////////////////////////////////
void sormtr(float *eigenvectors, float *transform, float *tau, int n)
{
	char side;
	char uplo;
	char trans;
	int info;
	int lwork;
	float *work0;
	float *work;

	side = 'L';
	uplo = 'U';
	trans = 'N';

	lwork = -1;
	work0 = new float[1];

	sormtr_(&side, &uplo, &trans, &n, &n,
				transform, &n, tau, eigenvectors, &n,
				work0, &lwork, &info);

	lwork = work0[0];
	work = new float[lwork];

	sormtr_(&side, &uplo, &trans, &n, &n,
				transform, &n, tau, eigenvectors, &n,
				work, &lwork, &info);

	delete[] work0;
	delete[] work;
}

////////////////////////////////////////////////////////////
//
//	Compares two arrays and returns # of elements of a1
//  that are larger than the largest element in a2. Assumes
//  arrays are in descending order.
//
//  @param	a1	array to compare to max element of a2
//  @param	a2	max element of a2 used for comparison
//  @param	n 	size of a1 and a2
////////////////////////////////////////////////////////////
int compare(float *a1, float *a2, int n)
{
	int count = n-1;
	float a2_max = get_max(a2, n, 1);

	while(a1[count] > a2_max && count > 0) count--;

	return n-count-1;
}

////////////////////////////////////////////////////////////
//
//	Transforms eigenvectors to component loadings.
//
//  @param	eigenvectors	eigenvectors to transform
//  @param	eigenvalues 	corresponding eigenvalues
//  @param	rows        	# of rows of eigenvectors
//	@param	cols        	# of cols of eigenvectors
////////////////////////////////////////////////////////////
void transform_eigenvectors(float *eigenvectors, float *eigenvalues, int rows, int cols)
{
	for(int i=0; i<rows; i++)
	{
		for(int j=0; j<cols; j++)
		{
			eigenvectors[i*rows+j] = (float) (eigenvectors[i*rows+j]*sqrtf(fabs(eigenvalues[j])));
		}
	}
}

////////////////////////////////////////////////////////////
//
//	varimax helper function.  Computes current variance
//	used for stopping criterion.
//
//  @param	loadings	current loadings
//  @param	rows    	# of rows of loadings
//  @param	cols    	# of columns of loadings
////////////////////////////////////////////////////////////
float compute_variance(float *loadings, int rows, int cols)
{
	float c;
	float r = (float) rows;

	c = 0.0;
	for (int i = 0; i < cols; i++)
	{
		float s2 = 0.0;

		for(int t = 0; t < rows; t++)
		{
			float sq = loadings[i*rows+t]*loadings[i*rows+t];

			s2 += sq;
			c += sq*sq;
		}
		c -= s2*s2/r;
	}

	return c;
 }

////////////////////////////////////////////////////////////
//
//	Performs varimax rotation of array aload.
//
//  @param	aload	array to rotate
//  @param	rows 	# of rows of aload
//  @param	cols 	# of columns of aload
////////////////////////////////////////////////////////////
void varimax(float *aload, int rows, int cols)
{
	int iter;
 	float crit, r = (float) rows;
	float denominator, numerator, angl, s;
	bool done;
	float a, b, c, d;

	//init
	crit = compute_variance(aload, rows, cols);
	done = false;
	iter = 0;

	//begin varimax
	while(!done && iter < VARIMAX_ITERATION)
	{
		float oldCrit = crit;

		for(int j = 0; j < cols-1; j++)
		{
			for(int k = j + 1; k < cols; k++)
			{
				a = 0.0;
				b = 0.0;
				c = 0.0;
				d = 0.0;
				s = 0.0;

				for (int i = 0; i < rows; i++)
				{
					float c2 = aload[j*rows+i]*aload[j*rows+i] - aload[k*rows+i]*aload[k*rows+i];
					float s2 = 2.0*aload[j*rows+i]*aload[k*rows+i];

					a += c2;
					b += s2;
					c += c2*c2 - s2*s2;
					d += 2.0*c2*s2;
				}

				denominator = c - 0.5*(a*a - b*b)/r;
				numerator = (d - 2.0*(0.5*a*b)/r);

				//compute angle and apply rotation
				angl = 0.25 * atan2f(numerator,denominator);

				c = cosf(angl);
				s = sinf(angl);

				for (int i = 0; i < rows ;i++)
				{
					float t = c*aload[j*rows+i] + s*aload[k*rows+i];

					aload[k*rows+i] = -s*aload[j*rows+i] + c*aload[k*rows+i];
					aload[j*rows+i] = t;
				}
			}
		}
		iter++;

		//check if stopping criterion met
		crit = compute_variance(aload, rows, cols);
		s = (crit > 0.0f) ? (crit - oldCrit)/crit : 0.0f;

		if (s > VARIMAX_EPSILON)
		{
			done = true;
		}
	}
}

////////////////////////////////////////////////////////////
//
//	Writes Pearson correlation array to a Pajek
//  compatible .net file.
//
//  @param	pearson 	array to write to file
//  @param	rows    	# of rows of array
//  @param	cols    	# of columns of array
//	@param	filename	output filename
////////////////////////////////////////////////////////////
void pajek_net(float *pearson, int rows, int cols, const char *filename)
{
	ofstream file;
	file.open(filename, ios::out);

	if(!file.is_open()) return;

	file << "*Vertices " << rows << endl;

	for(int i=0; i<rows; i++)
	{
		file << (i+1) << " ";
		file << "\"" << names[i];
		file << "\" ";
		file << "ic pink bc black\n";
	}

	file << "*Edges" << endl;

	for(int i=0; i<rows;i++)
	{
		for(int j=0; j<cols; j++)
		{
			file << (i+1) << " " << (j+1);
			file << " " << pearson[i*rows+j];
			file << " " << "c blue\n";
		}
	}

	file.close();
}

////////////////////////////////////////////////////////////
//
//	Outputs Pajek compatible cluster file to be used in
//  conjunction with the .net file.  Calls method to
//  generate .net file automatically.
//
//  @param	data    	data to cluster
//	@param	pearson  	Pearson correlation matrix
//  @param	rows    	# of rows of data
//  @param	cols    	# of columns of data
//	@param	filename	output filename
//  @param	net     	filename for Pajek network file
//        	        	to be created
////////////////////////////////////////////////////////////
void cluster_genes(float *loadings, float *pearson, int rows, int cols, const char *filename, const char *net)
{
	ofstream file;
	file.open(filename, ios::out);
	if(!file.is_open()) return;

	float max_loading;
	int max_index;

	file << "*Vertices " << rows << endl;

	//create Pajek .clu file
	for(int i=0; i<rows; i++)
	{
		max_loading = 0;
		max_index = 0;
		for(int j=0; j<cols; j++)
		{
			if(fabs(loadings[i*rows+j]) > THRESHOLD)
			{
				if(fabs(loadings[i*rows+j]) > max_loading)
				{
					max_loading = fabs(loadings[i*rows+j]);
					max_index = (j+1);
				}
			}
		}
		file << max_index;
		file << endl;
	}

	//create network file
	pajek_net(pearson, rows, rows, net);

	file.close();
}

////////////////////////////////////////////////////////////
//
//	Returns minimum value in array.
//
//  @param	array	array to find minimum of
//  @param	rows 	# of rows of array
//  @param	cols 	# of columns of array
////////////////////////////////////////////////////////////
float get_min(float *array, int rows, int cols)
{
	float min = 0.0;

	for(int i=0; i<rows; i++)
	{
		for(int j=0;j<cols;j++)
		{
			if(array[(i*cols)+j] < min) min = array[(i*cols)+j];
		}
	}

	return min;
}

////////////////////////////////////////////////////////////
//
//	Returns maximum value in array.
//
//  @param	array	array to find maximum element of
//  @param	rows 	# of rows of array
//  @param	cols	# of columns of array
////////////////////////////////////////////////////////////
float get_max(float *array, int rows, int cols)
{
	float max = 0.0;

	for(int i=0; i<rows; i++)
	{
		for(int j=0;j<cols;j++)
		{
			if(array[(i*cols)+j] > max) max = array[(i*cols)+j];
		}
	}

	return max;
}

////////////////////////////////////////////////////////////
//
//	Returns input matrix in column-major format.
//
//  @param	in  	array in row-major format
//  @param	rows	# of rows of in
//  @param	cols	# of columns of in
////////////////////////////////////////////////////////////
float* ctof(float *in, int rows, int cols)
{
	float *out;

	out = new float[rows*cols];

	for(int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{
			out[i+j*cols] = in[i*cols+j];
		}
	}

	delete[] in;
  	return out;
}

////////////////////////////////////////////////////////////
//
//	Returns input matrix in row-major format
//
//  @param	in  	array in column-major format
//  @param	rows	# of rows of in
//  @param	cols	# of columns of in
////////////////////////////////////////////////////////////
float *ftoc(float *in, int rows, int cols)
{
	float *out;

	out = new float[rows*cols];

	for(int i=0; i<rows; i++)
	{
		for (int j=0; j<cols; j++)
		{
			out[i*cols+j] = in[i+j*cols];
		}
	}

	delete[] in;
  	return out;
}

////////////////////////////////////////////////////////////
//
//	Writes array to output file.
//
//  @param	array    	array to write to file
//  @param	rows    	# of rows of array
//  @param	cols    	# of columns of array
//	@param	filename	output filename
//	@param	sep     	column separator
////////////////////////////////////////////////////////////
void matrix_to_file(float *array, int rows, int cols, const char *filename, char sep)
{
	ofstream file;
	file.open(filename, ios::out);

	if(!file.is_open()) return;

	for(int i=0; i<rows;i++)
	{
		for(int j=0; j<cols; j++)
		{
			file << array[i*cols+j];
			file << sep;
		}
		if(sep != '\n') file << endl;
	}

	file.close();
}

////////////////////////////////////////////////////////////
//
//	Prints array to console.
//
//  @param	array	array to print
//  @param	rows 	# of rows of array
//  @param	cols 	# of columns of array
////////////////////////////////////////////////////////////
void matrix_to_screen(float *array, int rows, int cols)
{
	for(int i=0; i<rows;i++)
	{
		for(int j=0; j<cols; j++)
		{
			printf("%f\t", array[i*cols+j]);
		}
		printf("\n");
	}
}

////////////////////////////////////////////////////////////
//
//	Sequential version of the RMT-based approach to
//	microarray analysis.
//
//  @param	filename 	filename of dataset
//  @param	pear_in   	'N' or 'n' if inputted dataset is
//	      	         	not a Pearson matrix
//  @param	separator	column separator of dataset
//	@param	verbose  	if true, write intermediate arrays
//	      	         	to output files
////////////////////////////////////////////////////////////
int rmt_sequential(char *filename, char pear_in, char separator, bool verbose)
{
	//define variables
	time_t start_time, end_time;
	float elapsed_time;
	float *dataset;
	float *random;
	float *pearson;
	float *random_coeff;
	float *random_eig;
	float *pearson_eig;
	float *w;
	float *tau;
	float *rtau;
	float *transform;
	float *d;
	float *e;
	float *p;
	float minimum, maximum;
	bool pearson_entered;
	int step;
	int me;

	printf("\n|----------------------------------");
	printf("RMT Approach Sequential Version");
	printf("------------------------------------|\n");

	step = 0;
	time(&start_time);
	printf("Step %i: Instantiating variables and loading data...", step);
	fflush(stdout);

	if(pear_in == 'N' || pear_in == 'n')
		pearson_entered = false;
	else
		pearson_entered = true;

	if(!pearson_entered)
	{
		if((dataset = load_data(filename, separator)) == NULL) return EXIT_FAILURE;
		pearson = new float[t_rows*t_rows];
	}
	else
	{
		dataset = new float[t_rows*t_cols];
		if((pearson = load_data(filename, separator)) == NULL) return EXIT_FAILURE;
	}

	//instantiate variables
	random_coeff = new float[t_rows*t_rows];
	random_eig = new float[t_rows];
	pearson_eig = new float[t_rows];
	w = new float[t_rows];
	tau = new float[t_rows-1];
	rtau = new float[t_rows-1];
	transform = new float[t_rows*t_rows];
	e = new float[t_rows-1];
	d = new float[t_rows];
	p = new float[t_rows*t_rows];

	printf("Done.\n");
	fflush(stdout);
	step++;
	printf("\tTotal Rows: %i\tTotal Columns: %i\n", t_rows, t_cols);

	if(t_rows == 0 || t_cols == 0) return EXIT_FAILURE;

	if(!pearson_entered)
	{
		//step 1: normalize the data
		printf("Step %i: Normalizing raw data matrix...", step);
		fflush(stdout);

		normalize(dataset, t_rows, t_cols);

		printf("Done.\n");
		fflush(stdout);
		step++;
	}

	printf("\tCreating random data matrix...");
	fflush(stdout);

	if(!pearson_entered)
	{
		//create random matrix and eigenvalue matrices
		minimum = get_min(dataset, t_rows, t_cols);
		maximum = get_max(dataset, t_rows, t_cols);

		random = create_random_matrix(t_rows, t_cols, minimum, maximum);
	}
	else
	{
		minimum = get_min(pearson, t_rows, t_cols);
		maximum = get_max(pearson, t_rows, t_cols);

		random = create_random_matrix(t_rows, t_cols, minimum, maximum);
	}

	printf("Done.\n");
	fflush(stdout);

	if(!pearson_entered)
	{
		//step 2: calculate the pearson coefficient between genes
		printf("Step %i: Calculating Pearson correlation coefficients for both matrices...", step);
		fflush(stdout);

		calculate_pearsons(pearson, dataset, t_rows, t_cols);
		if(verbose) matrix_to_file(pearson, t_rows, t_cols, "seq_pearson.txt", '\t');

		calculate_pearsons(random_coeff, random, t_rows, t_cols);

		printf("Done.\n");
		fflush(stdout);
		step++;

		//free some memory
		delete[] dataset;
		delete[] random;
	}
	else
	{
		memcpy(random_coeff, random, t_rows*t_rows*sizeof(float));
	}

	memcpy(p, pearson, t_rows*t_rows*sizeof(float));

	//step 3: compute eigenvalues/eigenvectors for both matrices
	printf("Step %i: Reducing both matrices to tridiagonal form...", step);
	fflush(stdout);

	//compute for pearson data matrix
	pearson = ctof(pearson, t_rows, t_rows); //transform to column-major form
	ssytrd(pearson, tau, t_rows);				//reduce to tridiagonal
	pearson = ftoc(pearson, t_rows, t_rows);	//transform to row-major form

	//compute for pearson random matrix
	random_coeff = ctof(random_coeff, t_rows, t_rows);
	ssytrd(random_coeff, rtau, t_rows);
	random_coeff = ftoc(random_coeff, t_rows, t_rows);

	printf("Done.\n");
	fflush(stdout);

	printf("\tComputing eigenvalues for both Pearson matrices...");
	fflush(stdout);

	//compute for pearson data matrix
	pearson = ctof(pearson, t_rows, t_rows); //transform to column-major form
	//get diagonal and superdiagonal
	for(int i = 0; i < t_rows; i++)
	{
		if(i > 0) e[i-1] = pearson[i*t_rows+(i-1)];
		d[i] = pearson[i*t_rows+i];
	}
	sstebz(pearson_eig, d, e, t_rows);	//compute eigenvalues
	pearson = ftoc(pearson, t_rows, t_rows);	//transform to row-major form

	//compute for pearson random matrix
	random_coeff = ctof(random_coeff, t_rows, t_rows);
	for(int i = 0; i < t_rows; i++)
	{
			if(i > 0) e[i-1] = random_coeff[i*t_rows+(i-1)];
			d[i] = random_coeff[i*t_rows+i];
	}
	sstebz(random_eig, d, e, t_rows);
	random_coeff = ftoc(random_coeff, t_rows, t_rows);

	printf("Done.\n");
	fflush(stdout);

	printf("\tData Eigenvalues\tRandom Eigenvalues (Last 5)\n");
	for(int i = t_rows-5; i < t_rows; i++) printf("\t%f\t\t%f\n", pearson_eig[i], random_eig[i]);

	if(verbose)
	{
		matrix_to_file(pearson, t_rows, t_rows, "seq_eigenvectors.txt", '\t');
		matrix_to_file(pearson_eig, t_rows, 1, "seq_eigenvalues.txt", '\t');
		matrix_to_file(random_eig, t_rows, 1, "seq_random_eigenvalues.txt", '\t');
	}

	printf("\tComparing eigenvalues of the random and raw data Pearson matrices...");
	fflush(stdout);

	me = compare(pearson_eig, random_eig, t_rows);

	printf("Done.\n");
	fflush(stdout);
	step++;

	if(me <= 1 || pearson_entered)
	{
		if(t_rows < K)
			me = t_rows;
		else
			me = K;
	}

	printf("\tMeaningful eigenvalues: %i\n", me);

	printf("\tComputing tridiagonal eigenvectors for the K largest eigenvalues (K = %i)...", me);
	fflush(stdout);

	pearson = ctof(pearson, t_rows, t_rows);
	//preserve transformation matrix
	memcpy(transform, pearson, t_rows*t_rows*sizeof(float));
	//get diagonal and super diagonal
	for(int i = 0; i < t_rows; i++)
	{
		if(i > 0) e[i-1] = pearson[i*t_rows+(i-1)];
		d[i] = pearson[i*t_rows+i];
	}
	sstein(pearson_eig, pearson, d, e, me, t_rows);
	pearson = ctof(pearson, t_rows, t_rows);

	printf("Done.\n");
	fflush(stdout);

	printf("\tBacktransforming tridiagonal eigenvectors to those of the original matrix...");
	fflush(stdout);

	pearson = ctof(pearson, t_rows, t_rows);
	sormtr(pearson, transform, tau, t_rows);
	pearson = ctof(pearson, t_rows, t_rows);

	printf("Done.\n");
	fflush(stdout);

	//step 4: transform eigenvectors
	printf("Step %i: Transforming eigenvectors to component loadings...", step);
	fflush(stdout);

	transform_eigenvectors(pearson, pearson_eig, t_rows, me);
	if(verbose) matrix_to_file(pearson, t_rows, t_rows, "seq_transformed.txt", '\t');

	printf("Done.\n");
	fflush(stdout);
	step++;

	//step 5: Varimax rotation
	printf("Step %i: Orthogonal rotation using Varimax...", step);
	fflush(stdout);

	pearson = ctof(pearson, t_rows, t_rows);
	varimax(pearson, t_rows, me);
	pearson = ftoc(pearson, t_rows, t_rows);

	printf("Done.\n\n");
	fflush(stdout);

	if(t_rows < 10) matrix_to_screen(pearson, t_rows, t_rows);

	printf("\nWriting output to files...");
	fflush(stdout);

	matrix_to_file(pearson, t_rows, t_rows, "seq_rotations.txt", '\t');
	matrix_to_file(p, t_rows, t_rows, "seq_pearson.txt", '\t');

	printf("Done.\n");
	fflush(stdout);

	printf("Creating Pajek files...");
	fflush(stdout);

	cluster_genes(pearson, p, t_rows, me, "seq_clusters.clu", "seq_network.net");

	printf("Done.\n");
	fflush(stdout);

	time(&end_time);
	elapsed_time = difftime(end_time,start_time);
	printf("Total Elapsed Time: %.9f seconds\n", elapsed_time);
	printf("Total Elapsed Time: %.9f minutes\n\n", elapsed_time / 60);

	delete[] pearson;
	delete[] pearson_eig;
	delete[] random_eig;
	delete[] random_coeff;
	delete[] w;
	delete[] tau;
	delete[] rtau;
	delete[] transform;
	delete[] d;
	delete[] e;
	return EXIT_SUCCESS;
}
//end rmta.h
