"""
struct MPIData{A,B,C}
    comm::A
    world_sz::B
    rank::C
end

function MPIData(comm=default_comm())
    return MPIData(comm, MPI.Comm_size(comm), MPI.Comm_rank(comm))
end

function default_comm()
    # Initialize MPI if not already done
    MPI.REFCOUNT[] == -1 && MPI.Init()
    return MPI.COMM_WORLD
end

#parallel_execution_cache(::ParallelMPI) = MPIData()
num_workers(tc::MPIData) = tc.world_sz
"""

struct MPI_cache{A,B,C}
    comm::A
    rank::B
    nworkers::C
end

export set_mpi

function set_mpi()
    MPI.Init()
    comm=MPI.COMM_WORLD
    return MPI_cache(comm, MPI.Comm_rank(comm), max(1,MPI.Comm_size(comm)))
end

export workers_sum!

function workers_sum!(data, comm) 
    MPI.Allreduce!(data, MPI.SUM, comm)
    return data
end

function workers_sum!(target, data, comm) 
    target = MPI.Allreduce(data, MPI.SUM, comm)
    return target
end


