export gradient

function gradient(method::String, A::Array{<:Complex{<:AbstractFloat}}, l1::Matrix{<:Complex{<:AbstractFloat}}, params::Parameters; 
    parallel::Bool=false, β::Float64=1.0, l2::Matrix{<:Complex{<:AbstractFloat}}=zeros(ComplexF64,1,1), basis=nothing, sampler::MetropolisSampler, ϵ::Float64=0.0)
    #l2::Matrix{<:Complex{<:AbstractFloat}}=nothing, basis=nothing, N_MC::Int64=nothing, ϵ::AbstractFloat=nothing)

    """ Interface function for selecting appropriate gradient descent method """

    if method=="exact"
        if basis==nothing
            error("Basis not given")
        end
        if l2==zeros(ComplexF64,1,1)
            return Exact_MPO_gradient(A, l1, basis, params)
        else
            return Exact_MPO_gradient_two_body(A, l1, l2, basis, params)
        end
    elseif method=="SGD"
        if sampler.N_MC==0
            error("Number of MC samples not specified")
        end
        if l2==zeros(ComplexF64,1,1)
            if β==1.0
                #return SGD_MPO_gradient(A, l1, N_MC, params)
                if parallel==false
                    return SGD_MPO_gradient(A, l1, sampler, params)
                else
                    return distributed_SGD_MPO_gradient(A, l1, N_MC, params)
                end
            else 
                return reweighted_SGD_MPO_gradient(β, A, l1, N_MC, params)
            end
        else
            return SGD_MPO_gradient_two_body(A, l1, l2, N_MC, params)
        end
    elseif method=="SR"
        if sampler.N_MC==0
            error("Number of MC samples not specified")
        end
        if ϵ==0.0
            error("ϵ-regulator not specified")
        end
        if l2==zeros(ComplexF64,1,1)
            if parallel==false
                return SR_MPO_gradient(A, l1, sampler, ϵ, params)
            else
                return distributed_SR_MPO_gradient(A, l1, N_MC, ϵ, params)
            end
        else
            if parallel==false
                if β==1.0
                    return SR_MPO_gradient_two_body(A, l1, l2, N_MC, ϵ, params)
                else 
                    return reweighted_SR_MPO_gradient_two_body(β, A, l1, l2, N_MC, ϵ, params)
                end
            else
                return distributed_SR_MPO_gradient_two_body(A, l1, l2, N_MC, ϵ, params)
            end
        end
    else
        error("Unrecognized method")
    end
end