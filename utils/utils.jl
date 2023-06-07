export make_density_matrix, adaptive_step_size

export dINDEX


#Basis type alias:
Basis = Vector{Vector{Bool}}



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
            sample = projector(ket,bra)
            #ρ_sample = MPO(sample,A)
            ρ[k,b] = MPO(params, sample, A)
        end
    end
    return ρ
end

