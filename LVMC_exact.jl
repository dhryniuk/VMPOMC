using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")
using BenchmarkTools


J=1.0
γ=1.0

L = make_Liouvillian(J*sx,γ*sz)

display(L)
#error()

#trial_ρ = [1.0 1+0.0001im; 1-0.41im 0.2]
#trial_ρ = [0.1 -0.2im; 0.2im 0.9]
#trial_ρ = [0.5 -0.01im; 0.01im 0.5]
#trial_ρ = [0.5 -0.9im; 1+0.1im 0.5]
#trial_ρ = [4*J^2/(4*J^2+γ^2)+0.1 2im*J*γ/(4*J^2+γ^2)+0.1; -2im*J*γ/(4*J^2+γ^2)+0.1 1]
#trial_ρ = [4/5+0.5 2im*J*γ/(4*J^2+γ^2)+0.5; -2im*J*γ/(4*J^2+γ^2) 1]
trial_ρ=rand(ComplexF64, 2,2)
#trial_ρ+=conj(trial_ρ)
trial_ρ/=tr(trial_ρ)

init_trial_ρ = copy(trial_ρ)
display(init_trial_ρ)

basis=generate_bit_basis(1)
display(basis)



#function vectorize(sket,sbra)
#    return [sket,1-sket]⊗[sbra,1-sbra]
#end
#@btime begin
    #b=vectorize(rand(0:1),rand(0:1))
    #display(b)
#end

dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))

TPSC = [(1,1),(1,0),(0,1),(0,0)]

@btime begin
    b=dVEC[(rand(0:1),rand(0:1))]
    #display(b)
end



function bra_L(bra,L)
    return transpose(bra)*L
end

#display(transpose(vec)*L)



mutable struct density_matrix
    coeff::ComplexF64
    ket::Vector{Int8}
    bra::Vector{Int8}
end


function find_index(i)
    if i==1
        return 1
    elseif i==0
        return 2
    end
end

function calculate_Z(J,γ,trial_ρ)
    Z = 0
    for i in [1,0]
        for j in [1,0]
            local state = density_matrix(1,[i],[j])
            αρ = trial_ρ[find_index(state.ket[1]) , find_index(state.bra[1])]
            p_α = αρ*conj(αρ)
            Z += p_α
        end
    end
    return Z
end

function local_Lindbladian(sample::density_matrix,J,γ,ψ)
    local_L=0

    s = dVEC[(sample.ket[1],sample.bra[1])]
    bra_L = transpose(s)*L

    for i in 1:4
        loc = bra_L[i]
        state = TPSC[i]
        local_L += loc*ψ[find_index(state[1]), find_index(state[2])]
    end

    local_L/=ψ[find_index(sample.ket[1]) , find_index(sample.bra[1])]
    return local_L*conj(local_L)
end

function calculate_mean_local_Lindbladian(J,γ,trial_ρ)
    mean_local_Lindbladian = 0
    Z=0
    for i in [1,0]
        for j in [1,0]
            local state = density_matrix(1,[i],[j])
            αρ = trial_ρ[find_index(state.ket[1]) , find_index(state.bra[1])]
            p_α = αρ*conj(αρ)
            Z+=p_α
            L_local = local_Lindbladian(state,J,γ,trial_ρ)
            mean_local_Lindbladian+=p_α*L_local
        end
    end
    return mean_local_Lindbladian/Z
end


function calculate_L∇L(J,γ,trial_ρ,ii,jj)
    Z=0
    L∇L=0
    for i in [1,0]
        for j in [1,0]
            local_L=0
            ∇local_L=0
            local sample = density_matrix(1,[i],[j])
            αρ = trial_ρ[find_index(sample.ket[1]) , find_index(sample.bra[1])]
            p_α = αρ*conj(αρ)
            Z+=p_α

            s = dVEC[(sample.ket[1],sample.bra[1])]
            bra_L = transpose(s)*L
            
            for i in 1:4
                loc = bra_L[i]
                state = TPSC[i]
                local_L += loc*trial_ρ[find_index(state[1]), find_index(state[2])]
                if ii==state[1] && jj==state[2]
                    ∇local_L += conj(loc)
                end
            end
            L∇L+=local_L*∇local_L
        end
    end
    return L∇L/Z
end


function gradient(J,γ,trial_ρ)
    Z = calculate_Z(J,γ,trial_ρ)
    mean_local_Lindbladian = calculate_mean_local_Lindbladian(J,γ,trial_ρ)
    F = zeros(ComplexF64, 2,2)
    for i in [1,0]
        for j in [1,0]
            local state = density_matrix(1,[i],[j])
            αρ = trial_ρ[find_index(state.ket[1]) , find_index(state.bra[1])]
            p_α = αρ*conj(αρ)
            F[find_index(i),find_index(j)] = calculate_L∇L(J,γ,trial_ρ,i,j) - mean_local_Lindbladian*αρ/Z^2
        end
    end
    return F./Z
end


#F = gradient(J,γ,trial_ρ)
#display(F)
#error()

δχ = 0.01
@time begin
    for i in 1:1000
        new_trial_ρ = trial_ρ - δχ*(gradient(J,γ,trial_ρ))
        global trial_ρ = new_trial_ρ
        global trial_ρ./=tr(trial_ρ)
        println(calculate_mean_local_Lindbladian(J,γ,trial_ρ))
    end
end

println("Initial ρ:")
display(init_trial_ρ)
println("Final ρ:")
display(trial_ρ)