#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ORANGE "\033[38;5;208m"
#define RED   "\033[31m"
#define BLUE  "\033[34m"
#define GRAY  "\033[90m"
#define RESET "\033[0m"

#define IDX(x,y) ((y) * local_w + (x))

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
	int grid_size_x = 1024;
	int grid_size_y = 1024; 
	
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

	// Determine block size for each rank DIMENSIONS MAY BE SWAPPED !!! DOESN'T MATTER FOR SQUARE CASE
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

	// Create vector for column halos
	MPI_Datatype column_type;
	MPI_Type_vector(
		block_y,		// Number of blocks (rows)
		1,				// 1 element per row
		local_w,		// Stride, distance between each halo
		MPI_DOUBLE,		// Old datatype
		&column_type	// New datatype
	);
	MPI_Type_commit(&column_type);

	// Process neighbors
	int up, down, left, right;	// dim[0] = rows ; dim[1] = columns
	MPI_Cart_shift(cart, 0, 1, &up, &down);
	MPI_Cart_shift(cart, 1, 1, &left, &right);

	// Compute
	double begin_parallel = MPI_Wtime();
	for (int k = 0; k < num_time_steps; k++)
	{
		//if (rank == 0)
			//printf("time step: %d \n", k);	
	
		// Sync borders, send: left/right as 0, up/down as 1
		MPI_Sendrecv(&T_k[IDX(1, 1)], 1, column_type, left, 0, &T_k[IDX(0, 1)], 1, column_type, left, 0, cart, MPI_STATUS_IGNORE); // Left
		MPI_Sendrecv(&T_k[IDX(block_x, 1)], 1, column_type, right, 0, &T_k[IDX(block_x + 1, 1)], 1, column_type, right, 0, cart, MPI_STATUS_IGNORE); // Right
		MPI_Sendrecv(&T_k[IDX(1, 1)], block_x, MPI_DOUBLE, up, 1, &T_k[IDX(1, 0)], block_x, MPI_DOUBLE, up, 1, cart, MPI_STATUS_IGNORE); // Up
		MPI_Sendrecv(&T_k[IDX(1, block_y)], block_x, MPI_DOUBLE, down, 1, &T_k[IDX(1, block_y + 1)], block_x, MPI_DOUBLE, down, 1, cart, MPI_STATUS_IGNORE); // Down
	
		// Border halos for edge processes
		if (up == MPI_PROC_NULL)
		 {
			for (int x = 1; x <= block_x; x++) 
			{
				T_k[IDX(x, 0)] = T_k[IDX(x, 2)];
			}
		}

		if (down == MPI_PROC_NULL) 
		{
			for (int x = 1; x <= block_x; x++) 
			{
				T_k[IDX(x, block_y + 1)] = T_k[IDX(x, block_y - 1)];
			}
		}

		if (left == MPI_PROC_NULL) 
		{
			for (int y = 1; y <= block_y; y++) 
			{
				T_k[IDX(0, y)] = T_k[IDX(2, y)];
			}
		}

		if (right == MPI_PROC_NULL) 
		{
			for (int y = 1; y <= block_y; y++) 
			{
				T_k[IDX(block_x + 1, y)] = T_k[IDX(block_x - 1, y)];
			}
		}
	
		// Computing index values			
		for (int y = 1; y <= block_y; y++)
			for (int x = 1; x <= block_x; x++)
			{
				int i = IDX(x, y);

				// Obtain left, right, down, and up indices for current index i
				int i_left = IDX(x - 1, y);
				int i_right = IDX(x + 1, y);
				int i_up = IDX(x, y - 1);
				int i_down = IDX(x, y + 1);

				double dTdt_i = conductivity *  (-4 * T_k[i] +  T_k[i_left] + T_k[i_right] + T_k[i_down] + T_k[i_up]);
				T_kn[i] = T_k[i] + delta_t * dTdt_i;
			}
		/* DEBUG PRINT
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

		swap(&T_k, &T_kn);
	}

	// Compute Average
	double T_average_parallel = 0;
	double local_sum = 0;	

	for (int y = 1; y <= block_y; y++)
		for (int x = 1; x <= block_x; x++)
			local_sum += T_k[IDX(x, y)];

	MPI_Reduce(&local_sum, &T_average_parallel, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
	
	if (rank == 0)
		T_average_parallel = T_average_parallel / (double)(grid_size_x * grid_size_y);

	double end_parallel = MPI_Wtime();
	double time_parallel = end_parallel - begin_parallel;
	double time_sequential;

	if (rank == 0) // Sequential
	{
		// Setup
		double* T_k_s = malloc(grid_size_x * grid_size_y * sizeof(double));
		double* T_kn_s = malloc(grid_size_x * grid_size_y * sizeof(double));
		
		for (int y = 0; y < grid_size_y; y++)
			for (int x = 0; x < grid_size_x; x++)
				T_k_s[grid_size_x * y + x] = x + y;		

		double begin_sequential = MPI_Wtime();
		
		for (int k = 0; k < num_time_steps; k++)
		{
			for (int y = 0; y < grid_size_y; y++)
				for (int x = 0; x < grid_size_x; x++)
				{
					int i = grid_size_x * y + x;
					
					int i_left = x != 0 ? i - 1 : i + 1;
					int i_right = x != grid_size_x - 1 ? i + 1 : i - 1;
					int i_down = y != 0 ? i - grid_size_x : i + grid_size_x;
					int i_up = y != grid_size_y - 1 ? i + grid_size_x : i - grid_size_x;

					double dTdt_i = conductivity * (-4 * T_k_s[i] + T_k_s[i_left] + T_k_s[i_right] + T_k_s[i_down] + T_k_s[i_up]);
					T_kn_s[i] = T_k_s[i] + delta_t * dTdt_i;
				}
			swap(&T_k_s, &T_kn_s);
		}
			
		double T_average_sequential = 0;
		for (int y = 0; y < grid_size_y; y++)
			for (int x = 0; x < grid_size_x; x++)
				T_average_sequential += T_k_s[grid_size_x * y + x];		

		T_average_sequential = T_average_sequential / (double)(grid_size_x * grid_size_y);

		double end_sequential = MPI_Wtime();
		time_sequential = end_sequential - begin_sequential;
		double speedup = time_sequential / time_parallel;		

		printf("Parallel T_Average: \t %.6f \nSequential T_Average: \t %.6f \n", T_average_parallel, T_average_sequential);
		printf("Parallel Time: %.3f s \nSequential Time: %.3f s\n", time_parallel, time_sequential);
		printf("Speedup: %.2fx\n", speedup);
	}

	MPI_Barrier(cart);
	
	free(T_k);
	free(T_kn);

	MPI_Comm_free(&cart);
	MPI_Finalize();
	return 0;
}
