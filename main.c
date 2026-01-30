#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank;		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int comm_size;	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);	


	MPI_Finalize();
	return 0;
}
