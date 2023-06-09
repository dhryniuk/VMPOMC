# This file contains routines detailing the computation of diagonal terms of the Lindbladian

abstract type EigenOperations end

struct Ising <: EigenOperations end

struct LongRangeIsing <: EigenOperations
    α::Float64
    Kac_norm::Float64
end

function LongRangeIsing(params::Parameters)
    α = params.α
    #Kac_norm = calculate_Kac_norm(α, params)
    Kac_norm = 1
    return LongRangeIsing(α,Kac_norm)
end

