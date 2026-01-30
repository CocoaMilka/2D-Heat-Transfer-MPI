#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

	// Allocate blocks with halos
	// | row_0 | ... | row_n | halo_t | ... | halo_r |
	T_k = malloc((block_x + 2)  * (block_y + 2) * sizeof(double));

	// Initialize array values: index [x, y] = x + y
	for (int y = 0; y < block_y; y++)
	{
		for (int x = 0; x < block_x; x++)
		{
			int global_x = coords[0] * block_x + x;
			int global_y = coords[1] * block_y + y;

			int index = y * block_x + x;
			T_k[index] = global_x + global_y;
		}
	}

	/* DEBUG PRINT
	if (rank == 1)
	{
		for (int i = 0; i < block_y; i++)
		{
			printf("\n");
			for (int j = 0; j < block_x; j++)
				printf(" %6.2f ", T_k[i * block_x + j]);
		}
	}
	*/


	MPI_Comm_free(&cart);
	MPI_Finalize();
	return 0;
}
