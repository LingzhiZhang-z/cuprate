"""
MPI Configuration Module

This module provides global access to MPI communication objects.
It should be imported at the beginning of any module that needs MPI functionality.
"""

from mpi4py import MPI

# Initialize MPI communication objects
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_comm():
    """Get the global MPI communicator"""
    return comm

def get_rank():
    """Get the current process rank"""
    return rank

def get_size():
    """Get the total number of processes"""
    return size

def is_root(target_rank=0):
    """Check if current process is the root process (rank 0)"""
    return rank == target_rank

def barrier():
    """Synchronize all processes"""
    comm.Barrier()

def broadcast(data, root=0):
    """Broadcast data from root to all processes"""
    return comm.bcast(data, root=root)

def gather(data, root=0):
    """Gather data from all processes to root"""
    return comm.gather(data, root=root)

def scatter(data, root=0):
    """Scatter data from root to all processes"""
    return comm.scatter(data, root=root)

def reduce(data, op=MPI.SUM, root=0):
    """Reduce data across all processes"""
    return comm.reduce(data, op=op, root=root)

def allreduce(data, op=MPI.SUM):
    """Reduce data across all processes and broadcast result to all"""
    return comm.allreduce(data, op=op) 