export Optimizer


function Optimizer(method::String, sampler::MetropolisSampler, A::Array{T,3}, l1::Matrix{T}, ϵ::Float64, params::Parameters, ising_op::String="Ising", dephasing_op::String="Local") where {T<:Complex{<:AbstractFloat}}
    if method=="Exact"
        return Exact(sampler, A, l1, params, ising_op, dephasing_op)
    elseif method=="SGD"
        return SGD(sampler, A, l1, params, ising_op, dephasing_op)
    elseif method=="SR"
        return SR(sampler, A, l1, ϵ, params, ising_op, dephasing_op)
    else
        error("Unrecognized method")
    end
end