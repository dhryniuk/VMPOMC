export Exact, Optimize!

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



mutable struct Exactl1{T<:Complex{<:AbstractFloat}} <: Exact{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::ExactCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #Eigen operations:
    eigen_ops::EigenOperations

    #Parameters:
    params::parameters

    #Workspace:
    workspace::workspace{T}#Union{workspace,Nothing}

end

#Constructor:
function Exact(sampler::MetropolisSampler, l1::Matrix{<:Complex{<:AbstractFloat}}, params::parameters, eigen_op::String="Ising")
    A = rand(ComplexF64,params.χ,params.χ,4)
    if eigen_op=="Ising"
        optimizer = Exactl1(A, sampler, ExactCache(A, params), l1, Ising(), params, set_workspace(A, params))
    elseif eigen_op=="LongRangeIsing" || eigen_op=="LRIsing" || eigen_op=="Long Range Ising"
        @assert params.α>0
        optimizer = Exactl1(A, sampler, ExactCache(A, params), l1, LongRangeIsing(params), params, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
    end
    return optimizer
end

mutable struct Exactl2{T<:Complex{<:AbstractFloat}} <: Exact{T}

    #MPO:
    A::Array{T,3}

    #Sampler:
    sampler::MetropolisSampler

    #Optimizer:
    optimizer_cache::ExactCache{T}#Union{ExactCache{T},Nothing}

    #1-local Lindbladian:
    l1::Matrix{T}

    #2-local Lindbladian:
    l2::Matrix{T}

    #Eigen operations:
    eigen_ops::EigenOperations

    #Parameters:
    params::parameters

    #Workspace:
    workspace::workspace{T}#Union{workspace,Nothing}

end

#Constructor:
function Exact(sampler::MetropolisSampler, l1::Matrix{<:Complex{<:AbstractFloat}}, l2::Matrix{<:Complex{<:AbstractFloat}}, params::parameters, eigen_op::String="Ising")
    A = rand(ComplexF64,params.χ,params.χ,4)
    if eigen_op=="Ising"
        optimizer = Exactl2(A, sampler, ExactCache(A, params), l1, l2, Ising(), params, set_workspace(A, params))
    elseif eigen_op=="LongRangeIsing" || eigen_op=="LRIsing" || eigen_op=="Long Range Ising"
        @assert params.α>0
        optimizer = Exactl2(A, sampler, ExactCache(A, params), l1, l2, LongRangeIsing(params), params, set_workspace(A, params))
    else
        error("Unrecognized eigen-operation")
    end    
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


function Ising_interaction_energy(eigen_ops::Ising, sample::projector, optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.A
    params = optimizer.params

    l_int::eltype(A)=0
    for j::UInt8 in 1:params.N-1
        l_int_ket = (2*sample.ket[j]-1)*(2*sample.ket[j+1]-1)
        l_int_bra = (2*sample.bra[j]-1)*(2*sample.bra[j+1]-1)
        l_int += l_int_ket-l_int_bra
    end
    l_int_ket = (2*sample.ket[params.N]-1)*(2*sample.ket[1]-1)
    l_int_bra = (2*sample.bra[params.N]-1)*(2*sample.bra[1]-1)
    l_int += l_int_ket-l_int_bra
    return 1.0im*params.J*l_int
    #return -1.0im*params.J*l_int
end

function Ising_interaction_energy(eigen_ops::LongRangeIsing, sample::projector, optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}} 

    A = optimizer.A
    params = optimizer.params

    l_int_ket::eltype(A) = 0.0
    l_int_bra::eltype(A) = 0.0
    l_int::eltype(A) = 0.0
    for i::Int16 in 1:params.N-1
        for j::Int16 in i+1:params.N
            l_int_ket = (2*sample.ket[i]-1)*(2*sample.ket[j]-1)
            l_int_bra = (2*sample.bra[i]-1)*(2*sample.bra[j]-1)
            dist = min(abs(i-j), abs(params.N+i-j))^eigen_ops.α
            l_int += (l_int_ket-l_int_bra)/dist
        end
    end
    return 1.0im*params.J*l_int/eigen_ops.Kac_norm
end

function SweepLindblad!(sample::projector, ρ_sample::T, optimizer::Exactl1{T}, local_L::T, local_∇L::Array{T,3}) where {T<:Complex{<:AbstractFloat}} 

    params=optimizer.params
    A=optimizer.A
    l1=optimizer.l1
    cache = optimizer.workspace

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
        local_L += lL
        local_∇L += l∇L
    end

    local_L /=ρ_sample
    local_∇L/=ρ_sample

    return local_L, local_∇L
end

function two_body_Lindblad_term(sample::projector, k::UInt8, l2::Matrix, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)
    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)

    s1::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[k],sample.bra[k])]
    s2::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[k+1],sample.bra[k+1])]
    s = kron(s1,s2)
    bra_L::Matrix{eltype(A)} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt8,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt8,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::eltype(A) = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( (cache.L_set[k]*A[:,:,i])*A[:,:,j]*cache.R_set[(params.N-k)])
                micro_sample::projector = projector(sample)
                micro_sample.ket[k] = state_i[1]
                micro_sample.bra[k] = state_i[2]
                micro_sample.ket[k+1] = state_j[1]
                micro_sample.bra[k+1] = state_j[2]
                
                cache.micro_L_set = L_MPO_strings!(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings!(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)
            end
        end
    end
    return local_L, local_∇L
end

function boundary_two_body_Lindblad_term(sample::projector, l2::Matrix, A::Array{<:Complex{<:AbstractFloat},3}, params::parameters, cache::workspace)

    #Need to find middle matrix product, by inverting the first tensor A:
    M = inv(A[:,:,dINDEX[(sample.ket[1],sample.bra[1])]])*cache.L_set[params.N]

    local_L::eltype(A) = 0
    local_∇L::Array{eltype(A),3}=zeros(eltype(A),params.χ,params.χ,4)

    s1::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[params.N],sample.bra[params.N])]
    s2::Matrix{eltype(A)} = dVEC_transpose[(sample.ket[1],sample.bra[1])]
    s = kron(s1,s2)
    bra_L::Matrix{eltype(A)} = s*conj(l2)
    #@inbounds for i::UInt16 in 1:4, j::UInt16 in 1:4
    for (i::UInt8,state_i::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})
        for (j::UInt8,state_j::Tuple{Bool,Bool}) in zip(1:4,TPSC::Vector{Tuple{Bool,Bool}})

            loc::eltype(A) = bra_L[j+4*(i-1)]
            if loc!=0
                local_L += loc*tr( M*A[:,:,i]*A[:,:,j] )
                #local_L += loc*MPO(params,sample,A)
                micro_sample::projector = projector(sample)
                micro_sample.ket[1] = state_i[1]
                micro_sample.bra[1] = state_i[2]
                micro_sample.ket[params.N] = state_j[1]
                micro_sample.bra[params.N] = state_j[2]
                
                cache.micro_L_set = L_MPO_strings!(cache.micro_L_set, micro_sample, A, params, cache)
                cache.micro_R_set = R_MPO_strings!(cache.micro_R_set, micro_sample, A, params, cache)
                local_∇L.+= loc.*∂MPO(micro_sample, cache.micro_L_set, cache.micro_R_set, params, cache)
            end
        end
    end
    return local_L, local_∇L
end

function SweepLindblad!(sample::projector, ρ_sample::T, optimizer::Exactl2{T}, local_L::T, local_∇L::Array{T,3}) where {T<:Complex{<:AbstractFloat}} 

    params=optimizer.params
    A=optimizer.A
    l1=optimizer.l1
    l2=optimizer.l2
    cache = optimizer.workspace

    #Calculate L∂L*:
    for j::UInt8 in 1:params.N
        lL, l∇L = one_body_Lindblad_term(sample,j,l1,A,params,cache)
        local_L += lL
        local_∇L += l∇L
    end
    for j::UInt8 in 1:params.N-1
        lL, l∇L = two_body_Lindblad_term(sample,j,l2,A,params,cache)
        local_L += lL
        local_∇L += l∇L
    end
    if params.N>2
        lL, l∇L = boundary_two_body_Lindblad_term(sample,l2,A,params,cache)
        local_L += lL
        local_∇L += l∇L
    end

    local_L /=ρ_sample
    local_∇L/=ρ_sample

    return local_L, local_∇L
end

function Update!(optimizer::Exact{T}, sample::projector) where {T<:Complex{<:AbstractFloat}} #... the ensemble averages etc.

    params=optimizer.params
    A=optimizer.A
    #l1=optimizer.l1
    data=optimizer.optimizer_cache
    cache = optimizer.workspace

    #Initialize auxiliary arrays:
    local_L::T = 0
    local_∇L::Array{T,3} = zeros(T,params.χ,params.χ,4)
    l_int::T = 0
    cache.local_∇L_diagonal_coeff = 0

    cache.L_set = L_MPO_strings!(cache.L_set, sample,A,params,cache)
    cache.R_set = R_MPO_strings!(cache.R_set, sample,A,params,cache)

    ρ_sample::T = tr(cache.L_set[params.N+1])
    p_sample::T = ρ_sample*conj(ρ_sample)
    data.Z += p_sample

    cache.Δ = ∂MPO(sample, cache.L_set, cache.R_set, params, cache)./ρ_sample

    #Sweep lattice:
    local_L, local_∇L = SweepLindblad!(sample, ρ_sample, optimizer, local_L, local_∇L)

    #Add in diagonal part of the local derivative:
    local_∇L.+=cache.local_∇L_diagonal_coeff.*cache.Δ

    #Add in interaction terms:
    l_int = Ising_interaction_energy(optimizer.eigen_ops, sample, optimizer)
    local_L +=l_int
    local_∇L+=l_int*cache.Δ

    #Update L∂L* ensemble average:
    data.L∂L+=p_sample*local_L*conj(local_∇L)

    #Update ΔLL ensemble average:
    data.ΔLL+=p_sample*cache.Δ

    #Mean local Lindbladian:
    data.mlL += real(p_sample*local_L*conj(local_L))
end

function Finalize!(optimizer::Exact{T}) where {T<:Complex{<:AbstractFloat}}
    data=optimizer.optimizer_cache
    data.mlL/=data.Z
    data.ΔLL.=conj.(data.ΔLL) #remember to take the complex conjugate
    data.ΔLL.*=data.mlL
    data.∇ = (data.L∂L-data.ΔLL)/data.Z
end

function compute_gradient!(optimizer::Exact{T}, basis::Basis) where {T<:Complex{<:AbstractFloat}}

    Initialize!(optimizer)
    for k in 1:optimizer.params.dim
        for l in 1:optimizer.params.dim
            sample = projector(basis[k],basis[l])
            Update!(optimizer, sample) 
        end
    end
    Finalize!(optimizer)
end

function Optimize!(optimizer::Exact{T}, basis::Basis, δ::Float64) where {T<:Complex{<:AbstractFloat}}

    compute_gradient!(optimizer, basis)

    ∇ = optimizer.optimizer_cache.∇
    ∇./=maximum(abs.(∇))

    new_A = similar(optimizer.A)
    new_A = optimizer.A - δ*∇
    optimizer.A = new_A
    optimizer.A = normalize_MPO!(optimizer.params, optimizer.A)
end