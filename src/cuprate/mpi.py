"""Minimal MPI runtime bindings."""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def is_root(target_rank=0):
    return rank == target_rank
