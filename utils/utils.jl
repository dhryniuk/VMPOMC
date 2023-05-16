export make_density_matrix, adaptive_step_size

export dINDEX


#Useful dictionaries:
dREVINDEX::Dict{Int8,Tuple{Bool,Bool}} = Dict(1 => (0,0), 2 => (0,1), 3 => (1,0), 4 => (1,1))
dINDEX::Dict{Tuple{Bool,Bool},Int8} = Dict((0,0) => 1, (0,1) => 2, (1,0) => 3, (1,1) => 4)
function dINDEXf(b::Bool, k::Bool)
    return 1+2*b+k
end
dVEC =   Dict((0,0) => [1,0,0,0], (0,1) => [0,1,0,0], (1,0) => [0,0,1,0], (1,1) => [0,0,0,1])
#dVEC_transpose::Dict{Tuple{Bool,Bool},Matrix{ComplexF64}} = Dict((0,0) => [1 0 0 0], (0,1) => [0 1 0 0], (1,0) => [0 0 1 0], (1,1) => [0 0 0 1])
dVEC_transpose::Dict{Tuple{Bool,Bool},Matrix} = Dict((0,0) => [1 0 0 0], (0,1) => [0 1 0 0], (1,0) => [0 0 1 0], (1,1) => [0 0 0 1])
dUNVEC = Dict([1,0,0,0] => (0,0), [0,1,0,0] => (0,1), [0,0,1,0] => (1,0), [0,0,0,1] => (1,1))
TPSC::Vector{Tuple{Bool,Bool}} = [(0,0),(0,1),(1,0),(1,1)]
#TPSC = [(0,0),(1,0),(0,1),(1,1)]

dINDEX2 = Dict(1 => 1, 0 => 2)
TPSC2 = [false,true]
#dVEC2 = Dict(0 => [1 0], 1 => [0 1])
dVEC2 = Dict(0 => [1,0], 1 => [0,1])

#=
dREVINDEX = Dict(1 => (1,1), 2 => (1,0), 3 => (0,1), 4 => (0,0))
dINDEX = Dict((1,1) => 1, (1,0) => 2, (0,1) => 3, (0,0) => 4)
dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))
TPSC = [(1,1),(1,0),(0,1),(0,0)]
#dINDEX2 = Dict(1 => 1, 0 => 2)
=#

mutable struct parameters
    N::Int64
    dim::Int64
    χ::Int64
    Jx::Float32
    Jy::Float32
    J::Float32
    hx::Float32
    hz::Float32
    γ::Float32
    α::Int
    burn_in::Int
end

function flatten_index(i,j,s,p::parameters)
    return i+p.χ*(j-1)+p.χ^2*(s-1)
end

function draw2(n)
    a = rand(1:n)
    b = rand(1:n)
    while b==a
        b = rand(1:n)
    end
    return a, b
end
function draw3(n)
    a = rand(1:n)
    b = rand(1:n)
    c = rand(1:n)
    while b==a
        b = rand(1:n)
    end
    while c==a && c==b
        c = rand(1:n)
    end
    return a, b, c
end

function adaptive_step_size(δ, current_L, previous_L)
    if current_L>previous_L
        δ=0.95*δ
    else 
        δ=1.02*δ
    end
    return δ
end

function make_density_matrix(params, A, basis)
    ρ = zeros(ComplexF64, params.dim, params.dim)
    k=0
    for ket in basis
        k+=1
        b=0
        for bra in basis
            b+=1
            sample = density_matrix(1,ket,bra)
            #ρ_sample = MPO(sample,A)
            ρ[k,b] = MPO(params, sample, A)
        end
    end
    return ρ
end

function set_parameters(N,χ,Jx,Jy,J,hx,hz,γ,α,burn_in)
	params.N = N;
    params.dim = 2^N;
    params.χ = χ;
    params.Jx = Jx;
    params.Jy = Jy;
    params.J = J;
    params.hx = hx;
    params.hz = hz;
    params.γ = γ;
    params.α = α;
    params.burn_in = burn_in;
end

""" Cache which stores intermediate results to reduce memory allocations"""
mutable struct workspace{T<:Complex{<:AbstractFloat}}
    L_set::Vector{Matrix{T}}
    R_set::Vector{Matrix{T}}
    micro_L_set::Vector{Matrix{T}}
    micro_R_set::Vector{Matrix{T}}
    plus_S::Array{T,2}
    B::Matrix{T}
    ID::Matrix{T}
    loc_1::Matrix{T}
    loc_2::Matrix{T}
    Metro_1::Matrix{T}
    Metro_2::Matrix{T}
    C_mat::Matrix{T}
    bra_L::Matrix{T}
    Δ::Array{T,3} #tensor of derivatives
    local_∇L_diagonal_coeff::ComplexF64
end

function set_workspace(A::Array{<:Complex{<:AbstractFloat}}, params::parameters)
    cache = workspace(
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        [ Matrix{eltype(A)}(undef,params.χ,params.χ) for _ in 1:params.N+1 ],
        zeros(eltype(A), 4*params.χ^2,4*params.χ^2),
        zeros(eltype(A), params.χ,params.χ),
        Matrix{eltype(A)}(I, params.χ, params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), params.χ,params.χ),
        zeros(eltype(A), 1, 4),
        zeros(eltype(A), params.χ, params.χ, 4),
        0.0+0.0im
        )
    return cache
end
