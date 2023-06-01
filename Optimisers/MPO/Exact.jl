export Exact, optimize!

#temporary only:
export Update!


mutable struct ExactCache{T} <: OptimizerCache
    #Ensemble averages:
    L∂L::Array{T,3}
    ΔLL::Array{T,3}

    #Sums:
    Z::T
    mlL::T   #mean local Lindbladian

    #Gradient:
    ∇::Array{T,3}
end

function ExactCache(A,params)
    exact=ExactCache(
        zeros(eltype(A),params.χ,params.χ,4),
        zeros(eltype(A),params.χ,params.χ,4),
        convert(eltype(A),0),
        convert(eltype(A),0),
        zeros(eltype(A),params.χ,params.χ,4)
    )  
    return exact
end




mutable struct Exact{T<:Complex{<:AbstractFloat}} <: Optimizer

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::ExactCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Parameters:
    params::parameters

    #Workspace:
    workspace::workspace{T}#Union{workspace,Nothing}

end

function Exact(sampler::MetropolisSampler, l1::Matrix{<:Complex{<:AbstractFloat}}, params::parameters)
    A = rand(ComplexF64,params.χ,params.χ,4)
    #optimizer = Exact(A, sampler, nothing, l1, params, nothing)
    optimizer = Exact(A, sampler, ExactCache(A, params), l1, params, set_workspace(A, params))
    return optimizer
end

function Initialize!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}
    optimizer.optimizer_cache = ExactCache(optimizer.A, optimizer.params)
    optimizer.workspace = set_workspace(optimizer.A, optimizer.params)
end




function one_body_Lindblad_term(sample::projector, j::UInt8, l1::Matrix{<:Complex{<:AbstractFloat}}, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)
    
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
    s::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[j],sample.bra[j])]
    #mul!(cache.bra_L, s, conj.(l1))
    mul!(cache.bra_L, s, l1)

    #Iterate over all 4 one-body vectorized basis projectors:
    @inbounds for (i,state) in zip(1:4,TPSC)
        loc = cache.bra_L[i]
        if loc!=0
            #Compute estimator:
            mul!(cache.loc_1, cache.L_set[j], @view(A[:,:,i]))
            mul!(cache.loc_2, cache.loc_1, cache.R_set[(params.N+1-j)])
            local_L += loc.*tr(cache.loc_2)
            
            #Compute derivative:
            if state==(sample.ket[j],sample.bra[j])   #check if diagonal
                cache.local_∇L_diagonal_coeff += loc
            else
                micro_sample::projector = projector(sample)
                micro_sample.ket[j] = state[1]
                micro_sample.bra[j] = state[2]
                cache.micro_L_set = L_MPO_strings!(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings!(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)
            end
        end
    end
    return local_L, local_∇L
end

function Lindblad_Ising_interaction_energy(sample::projector, boundary_conditions, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters)
    l_int::eltype(A)=0
    for j::UInt8 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    if boundary_conditions=="periodic"
        l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
        l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
        l_int += l_int_ket-l_int_bra
    end
    return 1.0im*params.J*l_int
    #return -1.0im*params.J*l_int
end

function long_range_interaction(sample::projector, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters)
    l_int_ket::eltype(A) = 0.0
    l_int_bra::eltype(A) = 0.0
    l_int::eltype(A) = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^params.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return 1.0im*params.J*l_int
end




function Update!(optimizer::Exact{T}, sample::projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params=optimizer.params
    A=optimizer.A
    l1=optimizer.l1
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    #Initialize auxiliary arrays:
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3} = zeros(eltype(A),params.χ,params.χ,4)
    l_int::eltype(A) = 0
    cache.local_∇L_diagonal_coeff = 0

    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.R_set = R_MPO_strings!(cache.R_set, sample,A,params,cache)

    ρ_sample::ComplexF64 = tr(cache.L_set[params.N+1])
    p_sample::ComplexF64 = ρ_sample*conj(ρ_sample)
    data.Z += p_sample

    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        #1-local part:
        lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
        local_L += lL
        local_∇L += l∇L
    end

    local_L /=ρ_sample
    local_∇L/=ρ_sample

    #Add in diagonal part of the local derivative:
    local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

    #Add in interaction terms:
    l_int = long_range_interaction(sample, A, params)
    local_L +=l_int
    local_∇L+=l_int*cache.Δ

    #Update L∂L* ensemble average:
    data.L∂L+=p_sample*local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL+=p_sample*cache.Δ

    #Mean local Lindbladian:
    data.mlL += real(p_sample*local_L*conj(local_L))
end

#Computes gradient tensor from ensemble averages
function ∇!(data::ExactCache)
    data.∇ = (data.L∂L-data.ΔLL)/data.Z
end

function Finalize!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}
    data=optimizer.optimizer_cache
    data.mlL/=data.Z
    data.ΔLL.=conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL.*=data.mlL
    ∇!(data)
end

"""
export Exact_MPO_gradient

function Exact_MPO_gradient(optimizer::Exact{T}, basis) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)

    for k in 1:optimizer.params.dim
        for l in 1:optimizer.params.dim
            sample = projector(basis[k],basis[l])
            Update!(optimizer, sample) 
        end
    end

    Finalize!(optimizer)

    return optimizer.optimizer_cache.∇, optimizer.optimizer_cache.mlL
end
"""

function compute_gradient!(optimizer::Exact{T}, basis) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)
    for k in 1:optimizer.params.dim
        for l in 1:optimizer.params.dim
            sample = projector(basis[k],basis[l])
            Update!(optimizer, sample) 
        end
    end
    Finalize!(optimizer)
end

function optimize!(optimizer::Exact{T}, basis, δ) where {T<:Complex{<:AbstractFloat}}

    compute_gradient!(optimizer, basis)

    ∇ = optimizer.optimizer_cache.∇
    ∇./=maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end