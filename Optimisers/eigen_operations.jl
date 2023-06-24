# This file contains routines required for the computation of purely-diagonal terms of the Lindbladian

abstract type EigenOperations end

abstract type IsingInteraction <: EigenOperations end

struct Ising <: IsingInteraction end

struct LongRangeIsing <: IsingInteraction
    α::Float64
    Kac_norm::Float64
end

function HarmonicNumber(n::Int,α::Float64)
    h=0
    for i in 1:n
        h+=i^(-α)
    end
    return h
end

function Kac_norm(params::Parameters)
    N = params.N
    α = params.α

    if mod(N,2)==0
        return (2*HarmonicNumber(1+N÷2,α) - 1 - (1+N÷2)^(-α))
    else
        return (2*HarmonicNumber(1+(N-1)÷2,α) - 1)
    end
end

function LongRangeIsing(params::Parameters)
    α = params.α
    #K = 1
    K = Kac_norm(params)
    return LongRangeIsing(α,K)
end

abstract type Dephasing <: EigenOperations end

#struct NoDephasing <: Dephasing end

struct LocalDephasing <: Dephasing end

struct CollectiveDephasing <: Dephasing end