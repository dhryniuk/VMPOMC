using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")
using BenchmarkTools


J=1.0
γ=1.0
N=2
dim = 2^N

l1 = make_Liouvillian(J*sx,γ*sm)

display(l1)
#error()

#trial_ρ = [1.0 1+0.0001im; 1-0.41im 0.2]
#trial_ρ = [0.1 -0.2im; 0.2im 0.9]
#trial_ρ = [0.5 -0.01im; 0.01im 0.5]
#trial_ρ = [0.5 -0.9im; 1+0.1im 0.5]
#trial_ρ = [4*J^2/(4*J^2+γ^2)+0.1 2im*J*γ/(4*J^2+γ^2)+0.1; -2im*J*γ/(4*J^2+γ^2)+0.1 1]
#trial_ρ = [4/5+0.5 2im*J*γ/(4*J^2+γ^2)+0.5; -2im*J*γ/(4*J^2+γ^2) 1]
trial_ρ=rand(ComplexF64, 2*N,2*N)
#trial_ρ+=conj(trial_ρ)
trial_ρ/=tr(trial_ρ)

init_trial_ρ = copy(trial_ρ)
display(init_trial_ρ)

basis=generate_bit_basis(N)
display(basis)

dINDEX = Dict((1,1) => 1, (1,0) => 2, (0,1) => 3, (0,0) => 4)
dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))

TPSC = [(1,1),(1,0),(0,1),(0,0)]


function bra_L(bra,L)
    return transpose(bra)*L
end

mutable struct density_matrix
    coeff::ComplexF64
    ket::Vector{Int8}
    bra::Vector{Int8}
end


function MPO(sample, A)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
    end
    #display(MPO)
    return tr(MPO)::ComplexF64
end

function MPO_inserted(sample, A, j, state)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        if i==j
            MPO*=A[:,:,dINDEX[state]]
        else
            MPO*=A[:,:,dINDEX[(sample.ket[i],sample.bra[i])]]
        end
    end
    return tr(MPO)::ComplexF64
end

function MPO_Z(A)
    Z=0
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) 
            Z+=MPO(sample,A)*conj(MPO(sample,A))
        end
    end
    return Z
end

function local_Lindbladian(J,γ,A) #should be called mean local lindbladian
    L_LOCAL=0
    #println("START")
    Z = MPO_Z(A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            for j in 1:N
                local_L=0
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                #display(bra_L)

                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                end

                local_L/=ρ_sample
            
                L_LOCAL+=local_L*conj(local_L)*ρ_sample*conj(ρ_sample)
                #println(local_L*conj(local_L))
            end
        end
    end

    #2-local part:
    #TBD

    return L_LOCAL/Z#*conj(L_LOCAL)
end

χ=1
#A_init=rand(ComplexF64, χ,χ,4)
#A_init=rand(Float64, χ,χ,4)
#A=copy(A_init)

#sample = density_matrix(1,[1],[0])
#M = MPO(sample,A_init)

#display(A_init)
#display(M)


#χ=1
A=zeros(ComplexF64, χ,χ,4)
A[:, :, 1] .= 0.5
A[:, :, 2] .= 0.2im
A[:, :, 3] .= -0.2im
A[:, :, 4] .= 0.5
#val = local_Lindbladian(J,γ,A)
#display(val)

#id=[1 0; 0 1]
#r=0.1*rand(χ,χ)
#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= id+0.1*rand(χ,χ)
#A[:, :, 2] .= 0.1*rand(χ,χ)
#A[:, :, 3] .= 0.1*rand(χ,χ)
#A[:, :, 4] .= id+0.1*rand(χ,χ)

#error()


#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    #B_list=Matrix{ComplexF64}[]
    #for j::UInt8 in 1:N-1 #fix N=1 case!
    #    push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    #end
    #return B_list
    #return 1
    #return [[1 0; 0 1]]
    return [Matrix{Int}(I, χ, χ)]#[[1 0; 0 1]]
end

function derv_MPO(i, j, u, sample, A)
    sum::ComplexF64 = 0
    for m::UInt8 in 1:N
        #println(u, " | ", sample)
        #if u == state #(sample.ket[m],sample.bra[m])
        if u == (sample.ket[m],sample.bra[m])
            B = prod(B_list(m, sample, A))
            #println(i,j)
            #println(B)
            #println(B_list(m, sample, A))
            sum += B[i,j] + B[j,i]
        end
    end
    if i==j
        sum/=2
    end
    return sum#/MPO(density_matrix(1,[sample[1]],[sample[2]]),A)
end

function calculate_gradient(J,γ,A,ii,jj,u)
    L∇L=0
    ΔLL=0
    #println("START")
    Z = MPO_Z(A)
    A_conjugate = conj(A)
    mean_local_Lindbladian = local_Lindbladian(J,γ,A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)

            #L∇L*:
            for j in 1:N
                local_L=0
                local_∇L=0
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                #display(bra_L)

                for i in 1:4
                    loc = bra_L[i]
                    #display(loc)
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                    micro_sample = sample
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]
                    local_∇L+= conj(loc)*derv_MPO(ii,jj,u,micro_sample,A_conjugate) #state,u or u,state?
                    #println(conj(loc), " . ", local_∇L)
                    #println(conj(loc), " . ", derv_MPO(ii,jj,u,state,A_conjugate))
                    #2-local part:
                    #TBD
                end

                #local_L /=ρ_sample
                #local_∇L/=conj(ρ_sample)
            
                #println(local_L, " ; ", local_∇L)
                L∇L+=local_L*local_∇L
            end

            #ΔLL:
            local_Δ=0
            local_Δ+=MPO(sample,A)*derv_MPO(ii,jj,u,sample,A_conjugate)

            ΔLL+=local_Δ
        end
    end

    ΔLL*=mean_local_Lindbladian

    #display(L∇L)

    return (L∇L-ΔLL)/Z
end


function normalize_MPO(A)
    #MPO=Matrix{ComplexF64}(I, χ, χ)
    #for i::UInt8 in 1:1#N
    #    MPO*=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])
        #MPO*=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])
    #end
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N#/2
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end

function normalize2_MPO(A)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        MPO*=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])#/2
        #MPO*=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])
    end
    return tr(MPO)#::ComplexF64
end

#error()

g = calculate_gradient(J,γ,A,1,1,(0,0))
display(g)

δχ = 0.003
@time begin
    for k in 1:1000
        new_A=zeros(ComplexF64, χ,χ,4)
        for i in 1:χ
            for j in 1:χ
                for u in TPSC
                    new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - δχ*(calculate_gradient(J,γ,A,i,j,u))
                end
            end
        end
        global A = new_A
        global A./=normalize_MPO(A)
        println(local_Lindbladian(J,γ,A))
    end
end



function show_density_matrix(A)
    den_mat = zeros(ComplexF64, dim,dim)
    k=0
    for ket in basis
        k+=1
        b=0
        for bra in basis
            b+=1
            sample = density_matrix(1,ket,bra)
            ρ_sample = MPO(sample,A)
            den_mat[k,b] = ρ_sample#^2
        end
    end
    display(den_mat)
    display(eigen(den_mat))
end


show_density_matrix(A)


error()










































function find_index(i)
    if i==1
        return 1
    elseif i==0
        return 2
    end
end

function calculate_Z(J,γ,trial_ρ)
    Z = 0
    for j in 1:N
        for i in [1,0]
            for j in [1,0]
                local state = density_matrix(1,[i],[j])
                αρ = trial_ρ[find_index(state.ket[1]) , find_index(state.bra[1])]
                p_α = αρ*conj(αρ)
                Z += p_α
            end
        end
    end
    return Z
end

function local_Lindbladian(sample::density_matrix,J,γ,ψ)
    LOCAL_L=0

    #1-local part:
    for j in 1:N
        local_L=0
        s = dVEC[(sample.ket[j],sample.bra[j])]
        bra_L = transpose(s)*L

        for i in 1:4
            loc = bra_L[i]
            state = TPSC[i]
            local_L += loc*ψ[find_index(state[j]), find_index(state[j])]
        end

        local_L/=ψ[find_index(sample.ket[j]) , find_index(sample.bra[j])]
        LOCAL_L+=local_L
    end

    #2-local part:
    #TBD

    return LOCAL_L*conj(LOCAL_L)
end

function calculate_mean_local_Lindbladian(J,γ,trial_ρ)
    mean_local_Lindbladian = 0
    Z=0
    for j in 1:N
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
    end
    return mean_local_Lindbladian/Z
end

function calculate_L∇L(J,γ,trial_ρ,ii,jj)
    Z=0
    L∇L=0
    for j in 1:N
        for i in [1,0]
            for j in [1,0]
                local_L=0
                ∇local_L=0
                local sample = density_matrix(1,[i],[j])
                αρ = trial_ρ[find_index(sample.ket[j]) , find_index(sample.bra[j)]
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

error()


using SparseArrays
using Plots
using Printf
using LinearAlgebra
using DelimitedFiles
#include("IsingED.jl")


function flip(sigma::Vector{Bool}, i::Int64)
    sigma_flip = copy(sigma) #deepcopy(sigma)
    sigma_flip[i] = !sigma_flip[i]
    return sigma_flip
end







error()




function MPS_energy(A)
    e_total::Float64=0
    Z::Float64=0

    for j in 1:dim
    #for j in 1:dim
        rho = basis[j]
        p::Float64=MPO(rho,A)*conj(MPO(rho,A))
        Z+=p

        # Interaction term
        #e_i::Int8 = 0
        #for i::UInt8 in 1:N
        #    e_i += (2*sigma[i]-1)*(2*sigma[mod(i,N)+1]-1)
        #end
        #e_interaction::Float64=-J*e_i

        # Field term
        states_with_flip::Vector{Vector{Bool}} = [flip(sigma, i) for i in 1:N]
        e_f::Float64 = 0
        for state in states_with_flip
            e_f += MPS(state,Ap,Am)/MPS(sigma,Ap,Am)#*MPS(sigma,Ap,Am)
        end
        e_field::Float64=-h*e_f
        e_total+=(e_interaction+e_field)*p
    end
    return e_total/Z
end


function MPS_energy(Ap, Am, N_MC)
    E_TOTAL::Float64 = 0
    e_total_thread::Vector{Float64} = zeros(Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        #e_total=0
        sigma::Vector{Bool} = ones(N)
        states_with_flip::Vector{Vector{Bool}} = [flip(sigma, i) for i in 1:N]
        #display(states_with_flip)
        for j::UInt64 in 1:N_MC
            for s::UInt8 in 1:N_sweeps
                for i::UInt8 in 1:N
                    sigma = Metropolis_single_flip(sigma, Ap, Am, i)
                end
            end
            #sigma = Metropolis_full_flip(sigma)
            #sigma = Metropolis_full_sweep(sigma, Ap, Am, states_with_flip)
            #sigma = Metropolis_single_flip(sigma, Ap, Am)
            #p = MPS(sigma,Ap,Am)^2 # prob dist
            #display(sigma)

            # Interaction term
            e_i::Int8 = 0
            for i::UInt8 in 1:N
                e_i += (2*sigma[i]-1)*(2*sigma[mod(i,N)+1]-1)
            end
            e_interaction=-J*e_i

            states_with_flip = [flip(sigma, i) for i in 1:N]

            # Field term
            e_f::Float64 = 0
            for state in states_with_flip
                e_f += MPS(state,Ap,Am)/MPS(sigma,Ap,Am)
            end
            e_field=-h*e_f

            e_total_thread[Threads.threadid()]+=e_interaction+e_field
        end
        #E_TOTAL+=e_total
    end
    E_TOTAL=sum(e_total_thread)/Threads.nthreads()/N_MC
    return E_TOTAL#/N_MC#e_total/N_MC
end

#GRADIENT:
function B_list(m::UInt8, sigma::Vector{Bool}, Ap::Union{Matrix{Float64},Matrix{ComplexF64}}, Am::Union{Matrix{Float64},Matrix{ComplexF64}})
    #B_list=Matrix{Float64}[]
    B_list=Matrix{ComplexF64}[]
    for i::UInt8 in 1:N-1
        if sigma[mod(m+i-1,N)+1]==1
            push!(B_list,Ap)
        else
            push!(B_list,Am)
        end
    end
    return B_list
end

function derv_MPS(i::UInt8, j::UInt8, s, sigma::Vector{Bool}, Ap::Union{Matrix{Float64},Matrix{ComplexF64}}, Am::Union{Matrix{Float64},Matrix{ComplexF64}})
    sum::ComplexF64 = 0
    for m::UInt8 in 1:N
        if s == sigma[m]
            B = prod(B_list(m, sigma, Ap, Am))
            sum += B[i,j] + B[j,i]
        end
    end
    if i==j
        sum/=2
    end
    return sum/MPS(sigma,Ap,Am)
end


function E_gradient(Ap::Union{Matrix{Float64},Matrix{ComplexF64}}, Am::Union{Matrix{Float64},Matrix{ComplexF64}}, N_MC)
    e_gradp = Matrix{ComplexF64}(zeros(χ,χ))
    e_gradm = Matrix{ComplexF64}(zeros(χ,χ))

    #e_local = MPS_local_energy_MC_parallel(Ap, Am, N_MC, 1)
    e_local = MPS_energy(Ap, Am)
    #e_local = MPS_local_energy_MC(Ap, Am, N_MC)
    #e_local = MPS_local_energy_random(Ap, Am, 100)
    Z::Float64 = 0

    k::UInt64 = 1
    @simd for k in 1:dim
        e_interaction::Float64 = 0
        e_field::Float64 = 0

        sigma = basis[k]
        p::Float64 = MPS(sigma,Ap,Am)*conj(MPS(sigma,Ap,Am))
        Z+=p

        # Interaction term:
        e_i::Int8 = 0
        for m in 1:N
            e_i += (2*sigma[m]-1)*(2*sigma[mod(m,N)+1]-1)
        end
        e_interaction = -J*e_i

        # Field term:
        if real(MPS(sigma,Ap,Am))!=0
            states_with_flip::Vector{Vector{Bool}} = [flip(sigma, i) for i in 1:N]
            e_f::Float64 = 0
            for state in states_with_flip
                e_f -= MPS(state,Ap,Am)/MPS(sigma,Ap,Am)
            end
            e_field = h*e_f
        end

        Dp = Matrix{ComplexF64}(zeros(χ,χ))
        Dm = Matrix{ComplexF64}(zeros(χ,χ))
        for i::UInt8 in 1:χ
            for j::UInt8 in 1:χ
                Dp[i,j] = derv_MPS(i, j, 1, sigma, Ap, Am)
                Dm[i,j] = derv_MPS(i, j, 0, sigma, Ap, Am)
            end
        end

        e_gradp += p*Dp*(e_interaction+e_field-e_local)
        e_gradm += p*Dm*(e_interaction+e_field-e_local)
    end
    return e_gradp/Z, e_gradm/Z, e_local
end

function optimize(Ap_init::Union{Matrix{Float64},Matrix{ComplexF64}}, Am_init::Union{Matrix{Float64},Matrix{ComplexF64}}, δ0, Q, G, k0, GS_exact, diff_E)
    energies_array::Vector{Float64} = [GS_exact]   #list of energies, with first entry exact
    Ap::Matrix{ComplexF64} = deepcopy(Ap_init)
    Am::Matrix{ComplexF64} = deepcopy(Am_init)
    converged::Bool=false
    #for k in 1:60
    e::Float64 = 1
    e_gradp::Matrix{ComplexF64}, e_gradm::Matrix{ComplexF64} = zeros(χ,χ), zeros(χ,χ)
    k::UInt64 = 0
    while converged==false
        Ap_old = deepcopy(Ap)
        Am_old = deepcopy(Am)

        e_old = e
        k+=1
        δ::Float64 = δ0*Q^k
        for n::UInt64 in 1:G*k
            e_gradp_old, e_gradm_old = e_gradp, e_gradm
            #e_gradp, e_gradm, e = (E_gradient(Ap, Am, 1)) #take real part for complex wfunction
            e_gradp, e_gradm, e = E_gradient(Ap, Am, k0+10*k, 2)

            #display(e_gradp)
            #display(e_gradp./maximum(e_gradp))

            #Ap = Ap - δ*e_gradp./maximum(abs.(e_gradp))# - δ/2*e_gradp_old./maximum(e_gradp_old)#./maximum(e_gradp)#.*rand(χ,χ)
            #Am = Am - δ*e_gradm./maximum(abs.(e_gradm))# - δ/2*e_gradm_old./maximum(e_gradm_old)#./maximum(e_gradm)#.*rand(χ,χ)

            Ap = Ap - δ*(sign.(e_gradp)).*rand(χ,χ) - δ/3*(sign.(e_gradp_old))
            Am = Am - δ*(sign.(e_gradm)).*rand(χ,χ) - δ/3*(sign.(e_gradm_old))

            #Ap = Ap - δ*(sign.(e_gradp)).*rand(χ,χ) - δ/2*(sign.(e_gradp_old)).*rand(χ,χ)
            #Am = Am - δ*(sign.(e_gradm)).*rand(χ,χ) - δ/2*(sign.(e_gradm_old)).*rand(χ,χ)

            #Ap = Ap - δ*(sign.(e_gradp))#.*rand(χ,χ)
            #Am = Am - δ*(sign.(e_gradm))#.*rand(χ,χ)

            #Ap, Am = fix_common_evals(Ap, Am)
            Ap/=maximum(abs.(Ap))
            Am/=maximum(abs.(Am))

            #display(δ*(sign.(e_gradp)))
            #display(e_gradp./maximum(abs.(e_gradp)))
        end
        #display(sign.(e_gradp))
        #display(e_gradp./maximum(abs.(e_gradp)))
        #display(Ap)
        #e_old = e
        e = MPS_energy(Ap,Am)#MPS_local_energy_MC(Ap,Am, 1000)
        #push!(energies_array, e)
        println("k=$k: ", (e-GS_exact)/N)

        ΔE = abs(e-e_old)
        println(ΔE)
        if ΔE < diff_E
            converged=true
        end

        #push!(energies_array, MPS_energy(Ap,Am))
        #println("k=$k: ", MPS_energy(Ap,Am)-GS_exact)

        #println(abs(MPS_energy(Ap,Am)-MPS_energy(Ap_old,Am_old)))
        #if abs(MPS_energy(Ap,Am)-MPS_energy(Ap_old,Am_old)) < ΔE
        #    converged=true
        #end
    end
    return Ap, Am, energies_array
end
