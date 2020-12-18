#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#define m_printf if (myrank==0)printf
#define TMAX 20
#define NX 20
#define NY 30

double EX[NX][NY];
double EY[NX][NY];
double HZ[NX][NY];
int i, j, it;
int tmax = TMAX;
int nx = NX;
int ny = NY;

MPI_Request requests[2];
MPI_Status statuses[2];
int myrank, ranksize;
int start_row, last_row, nrow;

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Barrier(MPI_COMM_WORLD);
    start_row = (myrank * nx) / ranksize;
    last_row = (((myrank + 1) * nx) / ranksize) - 1;
    nrow = last_row - start_row + 1;
    //init
    for (i = start_row; i <= last_row; i++)
    {
        for (j = 0; j < ny; j++)
        {
            EX[i][j] = ((double)i * (j + 1)) / nx;
            EY[i][j] = ((double)i * (j + 2)) / ny;
            HZ[i][j] = ((double)i * (j + 3)) / nx;
        }
    }

    //kernel_fdtd_2d
    double start = MPI_Wtime();
    for (it = 1; it <= tmax; it++)
    {
        //sending row between processes
        if (myrank) {
            MPI_Irecv(HZ[start_row - 1], NY, MPI_DOUBLE, myrank - 1, 215, MPI_COMM_WORLD, requests);
        }
        if (myrank != ranksize - 1) {
            MPI_Isend(HZ[last_row], NY, MPI_DOUBLE, myrank + 1, 215, MPI_COMM_WORLD, requests + 1);
        }
        //wait
        int count = 2, shift = 0;
        if (!myrank) {
            count -= 1;
            shift = 1;
        }
        if (myrank == ranksize - 1) {
            count -= 1;
        }
        MPI_Waitall(count, requests + shift, statuses);

        //calculate EY and EX
        if (!myrank)
        {
            for (j = 0; j < ny; j++)
                EY[0][j] = (double)it-1;
        }

        for (i = start_row; i <= last_row; i++)
        {
            if (i == 0) {continue;} //dont take 0 row
            for (j = 0; j < ny; j++)
                EY[i][j] = EY[i][j] - 0.5 * (EY[i][j] - HZ[i - 1][j]); //need previous row HZ
        }

        for (i = start_row; i <= last_row; i++)
        {
            for (j = 1; j < ny; j++)
                EX[i][j] = EX[i][j] - 0.5 * (HZ[i][j] - HZ[i][j - 1]); //just this line
        }


        //sending row between processes
        if (myrank != ranksize - 1) {
            MPI_Irecv(EY[last_row + 1], NY, MPI_DOUBLE, myrank + 1, 216, MPI_COMM_WORLD, requests + 1);
        }

        if (myrank) {
            MPI_Isend(EY[start_row], NY, MPI_DOUBLE, myrank - 1, 216, MPI_COMM_WORLD, requests);
        }
        //wait
        count = 2, shift = 0;
        if (!myrank) {
            count -= 1;
            shift = 1;
        }
        if (myrank == ranksize - 1) {
            count -= 1;
        }
        MPI_Waitall(count, requests + shift, statuses);
        //calculate HZ
        for (i = start_row; i <= last_row; i++)
        {
            if (i == nx) {continue;} //dont take last row
            for (j = 0; j < ny - 1; j++)
                HZ[i][j] = HZ[i][j] - 0.7 * (EX[i][j + 1] - EX[i][j] + EY[i + 1][j] - EY[i][j]); //need next row EY
        }
    
    }
    double end = MPI_Wtime();
    m_printf("Time of task = %f\n", end - start);
    MPI_Finalize();
    return 0;
}
