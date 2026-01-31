#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ORANGE "\033[38;5;208m"
#define RED   "\033[31m"
#define BLUE  "\033[34m"
#define GRAY  "\033[90m"
#define RESET "\033[0m"

void swap(double** a, double** b)
{
	double *temp = *b;
	*b = *a;
	*a = temp;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank;		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int comm_size;	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);	

	double delta_t = 0.02;
	int grid_size_x = 1024 / 16;
	int grid_size_y = 1024 / 16;
	
	int num_time_steps = 3000;
	double conductivity = 0.1;

	double* T_k;
	double* T_kn;

	// Decomposition of 2D Grid among processes
	int dims[2] = {0, 0};		// 0 to let MPI decide
	int periods[2] = {0, 0};	// no wrap around
	MPI_Comm cart;
	MPI_Dims_create(comm_size, 2, dims);

	MPI_Cart_create(
		MPI_COMM_WORLD,		// Old communicator
		2,					// Number of dimensions
		dims,				// Number of process per dimension (see above)
		periods,			// Periodicity of each dimension
		1,					// Allow rank reordering
		&cart				// New communicator
	);

	// 2D Coordinate for each rank
	int coords[2];
	MPI_Cart_coords(cart, rank, 2, coords);

	// Determine block size for each rank
	int block_x = grid_size_x / dims[0];
	int block_y = grid_size_y / dims[1];

	int local_w = block_x + 2; // Row width including halo
	int local_h = block_y + 2;
	
	// Allocate blocks with halos

	//	HHHHHHH		| H	...	H_n |
	//	HabcdeH		| H a_0 ... a_n H |
	//	HfghijH		...
	//	HklmnoH		...
	//	HHHHHHH		| H ... H_n |

	T_k = calloc(local_w * local_h, sizeof(double));
	T_kn = calloc(local_w * local_h, sizeof(double));

	// Initialize array values: index [x, y] = x + y
	
	for (int y = 1; y <= block_y; y++)
	{
		for (int x = 1; x <= block_x; x++)
		{
			int global_x = coords[0] * block_x + (x - 1);
			int global_y = coords[1] * block_y + (y - 1);

			int index = y * local_w + x;
			T_k[index] = global_x + global_y;
		}
	}

	// Process neighbors
	int up, down, left, right;	// dim[0] = rows ; dim[1] = columns
	//MPI_Cart_shift(cart, 0, 1, &up, &down);
	//MPI_Cart_shift(cart, 1, 1, &left, &right);

	/*
	// Compute
	for (int k = 0; k < num_time_steps; k++)
	{
		// Sync borders
		

		for (int y = 0; y < block_y; y++)
			for (int x = 0; x < block_x; x++)
			{
				int i = block_x * y + x; // index in 1d array of x,y
				
				int i_left = x != 0 ? i - 1 : i + 1;
				//int i_right = 
			}
		swap(&T_k, &T_kn);
	}
	*/

	/*
	if (rank == 0)
	{
		for (int y = 0; y < local_h; y++)
		{
			printf("\n");
			for (int x = 0; x < local_w; x++)
			{
				double v = T_k[y * local_w + x];

				if (x == 0 || y == 0 || x == local_w - 1 || y == local_h - 1)
					printf(ORANGE " %6.2f " RESET, v);
				else
					printf(" %6.2f ", v);
			}
		}
	}
	*/

	MPI_Barrier(cart);
	
	free(T_k);
	free(T_kn);

	MPI_Comm_free(&cart);
	MPI_Finalize();
	return 0;
}
