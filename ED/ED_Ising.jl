#using Combinatorics
export generate_bit_basis_reversed, make_bit_Hamiltonian
export generate_bit_basis


#Ising bit-basis:
function generate_bit_basis(N)#(N::UInt8)
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

#Ising bit-basis:
function generate_bit_basis_reversed(N)#(N::UInt8)
    set::Vector{Vector{Bool}} = [[false], [true]]
    @simd for i in 1:N-1
        new_set::Vector{Vector{Bool}} = []
        @simd for state in set
            state2::Vector{Bool} = copy(state)
            state = vcat(state, false)
            state2 = vcat(state2, true)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Bool}}(set)
end

#Ising bit-Hamiltonian:
function make_bit_Hamiltonian(N, J::Real, h::Real, basis::Vector{Vector{Bool}})#(N::UInt8, J::Real, h::Real, basis::Vector{Vector{Bool}})
    dim::UInt32 = length(basis)
    H::Matrix{Float64} = zeros(dim, dim)

    #Diagonal part:
    i::UInt32 = 0
    for state in basis
        i += 1
        @simd for spin in 1:N#-1 #HERE!
            #H[i, i] -= state[spin]*state[spin+1]
            H[i, i] -= J*(2*state[spin]-1)*(2*state[mod(spin,N)+1]-1)
        end
    end

    #Off-diagonal part:
    b::UInt32 = 0
    for bra in basis
        b += 1
        k::UInt32 = 0
        for ket in basis
            k += 1
            @simd for spin in 1:N
                ket_prime::Vector{Bool} = copy(ket)
                ket_prime[spin] = (1-ket_prime[spin])
                if bra == ket_prime
                    H[b, k] = -h
                end
            end
        end
    end
    return H
end

export magnetization
function magnetization(state, basis)
    M = 0.
    for (i, bstate) in enumerate(basis)
        bstate_M = 0.
        for spin in bstate
            bstate_M += (state[i]^2 * (spin ? 1 : -1))/length(bstate)
        end
        @assert abs(bstate_M) <= 1
        M += abs(bstate_M)
    end
    return M
end



###OLD:

#Ising basis:
function generate_basis(N::UInt8)
    set = [[1], [-1]]
    for i in 1:N-1
        new_set=[]
        for state in set
            state2 = deepcopy(state)
            state = vcat(state, 1)
            state2 = vcat(state2, -1)
            push!(new_set, state)
            push!(new_set, state2)
        end
        set = new_set
    end
    return Vector{Vector{Int8}}(set)
end

#Ising Hamiltonian:
function make_Hamiltonian(N, J, h, basis)
    dim = length(basis)
    H = zeros(dim, dim)

    #Diagonal part:
    i = 0
    for state in basis
        i += 1
        for spin in 1:N#-1
            #H[i, i] -= state[spin]*state[spin+1]
            H[i, i] -= J*state[spin]*state[mod(spin,N)+1]
        end
    end

    #Off-diagonal part:
    b = 0
    for bra in basis
        b += 1
        k = 0
        for ket in basis
            k += 1
            #if abs(sum(bra)-sum(ket)) == 2
            for spin in 1:N
                ket_prime = copy(ket)
                ket_prime[spin] = -ket_prime[spin]
                if bra == ket_prime
                    H[b, k] = -h
                end
            end
            #end
        end
    end
    return H
end


#Magnetization:
function magnetization_single(N, J, h, basis, state)
    return sum(state)
end


function magnetization(N, J, h, basis, ψ_GS)
    M = 0.
    #println(dim)
    #println(length(basis))
    for (i, state) in enumerate(basis)
        state = basis[i]
        state_M = sum(state)*ψ_GS[i]^2   #or use abs2. 
        M += abs(state_M)
    end
    return M/N
end