export make_one_body_Lindbladian, ⊗, id, sx, sy, sz, sp, sm, generate_bit_basis


#Basis type alias:
Basis = Vector{Vector{Bool}}

⊗(x,y) = kron(x,y)

id = [1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im]
sx = [0.0+0.0im 1.0+0.0im; 1.0+0.0im 0.0+0.0im]
sy = [0.0+0.0im 0.0-1im; 0.0+1im 0.0+0.0im]
sz = [1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

sp_id = sparse(id)
sp_sx = sparse(sx)
sp_sy = sparse(sy)
sp_sz = sparse(sz)
sp_sp = sparse(sp)
sp_sm = sparse(sm)

#Useful dictionaries:
dREVINDEX::Dict{Int8,Tuple{Bool,Bool}} = Dict(1 => (0,0), 2 => (0,1), 3 => (1,0), 4 => (1,1))
dINDEX::Dict{Tuple{Bool,Bool},Int8} = Dict((0,0) => 1, (0,1) => 2, (1,0) => 3, (1,1) => 4)
function dINDEXf(b::Bool, k::Bool)
    return 1+2*b+k
end
dVEC =   Dict((0,0) => [1,0,0,0], (0,1) => [0,1,0,0], (1,0) => [0,0,1,0], (1,1) => [0,0,0,1])
dVEC_transpose::Dict{Tuple{Bool,Bool},Matrix} = Dict((0,0) => [1 0 0 0], (0,1) => [0 1 0 0], (1,0) => [0 0 1 0], (1,1) => [0 0 0 1])
dUNVEC = Dict([1,0,0,0] => (0,0), [0,1,0,0] => (0,1), [0,0,1,0] => (1,0), [0,0,0,1] => (1,1))
TLS_Liouville_Space::Vector{Tuple{Bool,Bool}} = [(0,0),(0,1),(1,0),(1,1)]
dINDEX2 = Dict(1 => 1, 0 => 2)
dVEC2 = Dict(0 => [1,0], 1 => [0,1])

function flatten_index(i,j,s,p::Parameters)
    return i+p.χ*(j-1)+p.χ^2*(s-1)
end

function make_one_body_Lindbladian(H, Γ)
    L_H = -1im*(H⊗id - id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗id/2 - id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end

function make_two_body_Lindblad_Hamiltonian(A, B)
    L_H = -1im*( (A⊗id)⊗(B⊗id) - (id⊗transpose(A))⊗(id⊗transpose(B)) )
    return L_H
end

#Ising bit-basis:
function generate_bit_basis(N)
    set::Vector{Vector{Bool}} = [[true], [false]]
    @simd for i in 1:N-1
        new_set::Vector{Vector{Bool}} = []
        @simd for state in set
            state2::Vector{Bool} = copy(state)
            state = vcat(state, true)
            state2 = vcat(state2, false)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Bool}}(set)
end