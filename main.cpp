
//include necessary files
#include "rmta.h"

//global variables for total rows and columns of dataset
extern int t_rows;
extern int t_cols;

//include external functions
extern float *load_data(const string &filename,char separator);
extern void normalize(float *dataset,int rows,int cols);
extern float get_min(float *array,int rows,int cols);
extern float get_max(float *array,int rows,int cols);
extern void matrix_to_file(float *array,int rows,int columns,
							const char *filename,char separator);
extern void cluster_genes(float *loadings, float *pearson,
							int rows, int cols,
							const char *filename, const char *net);
extern int rmt_sequential(char *filename,char separator,
							bool verbose);
extern "C" int runRMT(float *data, float *random,
						float *pearson_out,float *rotations,
						bool pear_in,int curr_step,
						int cols, int rows);


////////////////////////////////////////////////////////////
//
// C++ wrapper function which runs the CUDA device
// code for the Random Matrix Theory-based
// microarray clustering algorithm.  Loads data
// into memory, creates random matrix and calls
// device portion of the code (method runRMT).
// Clusters results and writes output to files.
//
// @param	filename	full path to dataset file
// @param	pear_char	'Y' or 'y' for yes else no,
//       	         	states if dataset is a
//       	         	Pearson matrix
// @param	separator	character separating columns
//       	         	in data file
////////////////////////////////////////////////////////////
int rmt_gpu(char *filename, char pear_char, char separator)
{
	//define variables
	time_t start_time, end_time;
	float elapsed_time;
	float *dataset;
	float *random;
	float *rotations;
	float *pearson;
	float minimum, maximum;
	int me = 0;
	int step;
	bool pearson_entered;

	printf("\n|-------------------------------------");
	printf("RMT Approach GPU Version");
	printf("----------------------------------------|\n");

	if(pear_char == 'Y' || pear_char == 'y')
		pearson_entered = true;
	else
		pearson_entered = false;

	step = 0;
	time(&start_time);
	printf("(On CPU) Step %i: Instantiating variables", step);
	printf(" and loading data...");
	fflush(stdout);

	//step 0: load data and instantiate variables

	if((dataset = load_data(filename,separator)) == NULL) {
		printf("Unable to read file...\n");
		return EXIT_FAILURE;
	}

	rotations = new float[t_rows*t_rows];
	pearson = new float[t_rows*t_rows];

	printf("Done.\n");
	fflush(stdout);
	step++;

	printf("\t\t Total Rows: %i\tTotal Columns: %i\n",
			t_rows, t_cols);

	if(!pearson_entered)
	{
		//step 1: normalize the data and create random matrix
		printf("(On CPU) Step %i: Normalizing data matrix...",
				step);
		fflush(stdout);

		normalize(dataset,t_rows,t_cols);

		printf("Done.\n");
		fflush(stdout);
		step++;
	}

	printf("(On CPU)\t Creating random data matrix...");
	fflush(stdout);

	//create random matrix
	minimum = get_min(dataset,t_rows,t_cols);
	maximum = get_max(dataset,t_rows,t_cols);

	random = create_random_matrix(t_rows,t_cols,minimum,maximum);

	printf("Done.\n");
	fflush(stdout);

    //run the device part of the program
    me = runRMT(dataset, random, pearson, rotations,
    			pearson_entered, step, t_cols, t_rows);

	if(t_rows < 10)
		matrix_to_screen(rotations,t_rows,t_rows);

	printf("\n(On CPU) Writing output to files...");
	fflush(stdout);

	matrix_to_file(rotations,t_rows,t_rows,"gpu_rotation.txt",
					'\t');
	matrix_to_file(pearson,t_rows,t_rows,"gpu_pearson.txt",'\t');

	printf("Done.\n");
	fflush(stdout);

	//Final step: cluster data
	printf("(On CPU) Creating Pajek files...");
	fflush(stdout);

	cluster_genes(rotations, pearson, t_rows, me, "gpu_clusters.clu", "gpu_network.net");

	printf("Done.\n");
	fflush(stdout);

	printf("%f,%f,%f\n", rotations[0], rotations[1], rotations[2]);
        time(&end_time);
	elapsed_time = difftime(end_time,start_time);
	printf("Total Elapsed Time: %.9f seconds\n",
			elapsed_time);
	printf("Total Elapsed Time: %.9f minutes\n\n",
			elapsed_time / 60);

	delete[] dataset;
	delete[] random;
	delete[] rotations;

	return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////
//
// main method; determines what dataset to use
// and asks user which version they would like
// to run and what options to use.
//
// @param	argc	number of arguments supplied
//       	    	should be <= 2
// @param	argv	if argc == 2, argv will contain
//       	    	full path to the dataset file
////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	bool verbose;
	bool sequential;
	char seqchar, verbchar, pearchar, gpuchar;
	char *filename;
	char separator = '\t';
	int length;

	//check arguments
	if(argc < 2)	//no dataset entered on command-line
	{
		length = strlen(FILE_PATH)+1;
		if((filename=(char *) malloc(length*sizeof(char)))==NULL)
			return EXIT_FAILURE;

		strncpy(filename, FILE_PATH, length);

		printf("\nNo path entered on command-line. Using default:\n");
		printf("%s\n\n", filename);

		separator = DEF_SEPARATOR;
	}
	else if(argc == 2) //path to dataset entered on command-line
	{
		length = strlen(argv[1])+1;
		if((filename=(char *) malloc(length*sizeof(char)))==NULL)
			return EXIT_FAILURE;

		strncpy(filename, argv[1], length);

		printf("\nDataset entered on command-line.  Using:\n");
		printf("%s\n", filename);

		separator = DEF_SEPARATOR;
	}
	else	//incorrect number of arguments
	{
		printf("Too many arguments supplied!! Exiting...\n");
		return EXIT_FAILURE;
	}

	//ask user about type of dataset
	cout << "Is dataset a Pearson correlation matrix";
	cout << " (Type 'n' for raw data) (Y/n)? ";
	cin >> pearchar;
	getchar();

	//ask user if he/she would like to run sequential portion
	cout << "Run the sequential (CPU) code (Y/n)? ";
	cin >> seqchar;
	getchar();

	if(seqchar == 'Y' || seqchar == 'y')
	{
		//get runtime parameters for sequential version
		sequential = true;
		cout << "Run in verbose mode ";
		cout << "(May increase sequential runtime!) (Y/n)? ";
		cin >> verbchar;
		getchar();

		if(verbchar == 'Y' || verbchar == 'y')
			verbose = true;
		else
			verbose = false;

	}
	else
		sequential = false;

	//ask user if he/she would like to run GPU portion
	cout << "Run the GPU code (Y/n)? ";
	cin >> gpuchar;
	getchar();

	//run sequential portion if desired
	if(sequential)
	{
		rmt_sequential(filename, pearchar, separator, verbose);
	}

	//run GPU version if desired
	if(gpuchar == 'Y' || gpuchar == 'y')
	{
		rmt_gpu(filename, pearchar, separator);
	}

	free(filename);

	return EXIT_SUCCESS;
}
//end main.cpp
