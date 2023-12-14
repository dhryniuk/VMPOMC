export set_mpi, workers_sum!


struct MPI_cache{A,B,C}
    comm::A
    rank::B
    nworkers::C
end

Base.display(mpi::MPI_cache) = begin
    println("\nMPI:")
    println("n_worlds\t", mpi.nworkers)
    println("root\t\t", 0)
end

function set_mpi()
    MPI.Init()
    comm=MPI.COMM_WORLD
    return MPI_cache(comm, MPI.Comm_rank(comm), max(1,MPI.Comm_size(comm)))
end

function workers_sum!(data, comm) 
    MPI.Allreduce!(data, MPI.SUM, comm)
    return data
end

function workers_sum!(target, data, comm) 
    target = MPI.Allreduce(data, MPI.SUM, comm)
    return target
end


