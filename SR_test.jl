using SparseArrays
using Plots
using Printf
using LinearAlgebra
using DelimitedFiles
include("IsingED.jl")


function flip(sigma, i)
    sigma_flip = deepcopy(sigma)
    sigma_flip[i] = 0-sigma_flip[i]
    return sigma_flip
end


const N=8
const dim = 2^N #Dimension of Hilbert Space
const J=1
const h=1
χ=4 #bond dimension

const basis = generate_basis(N)
#display(basis)
const H = make_Hamiltonian(N,J,h,basis)
@time begin
    E_n, ψ_n = eigen(H)
end

#display(H)
#display(ψ_n[:,1])
#display(E_n[1])


println("Number of threads = ", Threads.nthreads())



function MPS(sigma,A)
    M=Matrix{Int}(I, χ, χ)
    for i in 1:N
        if sigma[i]==1
            M*=A[:,:,1]
        else
            M*=A[:,:,2]
        end
    end
    return tr(M)
end

function MPS_Z(A)
    Z=0
    for j in 1:dim
        sigma = basis[j]
        Z+=MPS(sigma,A)^2
    end
    return Z
end

function MPS_local_energy(A)
    e_total=0
    Z=0
    for j in 1:dim
        sigma = basis[j]
        p = MPS(sigma,A)*conj(MPS(sigma,A))   #prob density
        Z+=p

        # Interaction term
        e_i = 0
        for i in 1:N
            e_i += sigma[i]*sigma[mod(i,N)+1]
        end
        e_interaction=-J*e_i

        # Field term
        states_with_flip = [flip(sigma, i) for i in 1:N]
        e_f = 0
        for state in states_with_flip
            e_f += MPS(state,A)/MPS(sigma,A)
        end
        e_field=-h*e_f
        e_total+=p*(e_interaction+e_field)
    end
    return e_total/Z
end

#GRADIENT:
function B_list(m, sigma, A)
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for i in 1:N-1
        if sigma[mod(m+i-1,N)+1]==1
            push!(B_list,A[:,:,1])
        else
            push!(B_list,A[:,:,2])
        end
    end
    return B_list
end

function derv_MPS(i, j, s, sigma, A)
    sum = 0
    for m in 1:N
        if s == sigma[m]
            B = prod(B_list(m, sigma, A))
            sum += B[i,j] + B[j,i]
        end
    end
    if i==j
        sum/=2
    end
    return sum/MPS(sigma,A)
end

function E_gradient(A)
    e_grad = zeros(ComplexF64,χ,χ,2)
    #e_grad = Matrix{Float64}(zeros(χ,χ,2))#change to complex!
    #e_gradm = Matrix{Float64}(zeros(χ,χ))

    #e_grad = [Matrix{Float64}(zeros(χ,χ)) for _ = 1:N]
    e_local = MPS_local_energy(A)
    Z=0

    for k in 1:dim
        e_interaction = 0
        e_field = 0

        sigma = basis[k]
        p = MPS(sigma,A)*conj(MPS(sigma,A))
        Z+=p

        # Interaction term:
        e_i = 0
        for m in 1:N
            e_i += sigma[m]*sigma[mod(m,N)+1]
        end
        e_interaction=-J*e_i

        # Field term:
        if real(MPS(sigma,A))!=0
            states_with_flip = [flip(sigma, i) for i in 1:N]
            e_f = 0
            for state in states_with_flip
                e_f -= MPS(state,A)/MPS(sigma,A)
            end
            e_field=h*e_f
        end

        #Dp = Matrix{ComplexF64}(zeros(χ,χ))
        #Dm = Matrix{ComplexF64}(zeros(χ,χ))
        D = zeros(ComplexF64,χ,χ,2)
        #D = Matrix{ComplexF64}(zeros(χ,χ,2))
        for s in 1:2
            for i in 1:χ
                for j in 1:χ
                    D[i,j,s] = derv_MPS(i, j, (-1)^(s-1), sigma, A)
                end
            end
        end

        e_grad += p*conj(D)*(e_interaction+e_field-e_local)
    end
    return e_grad/Z
end


function flatten_index(i,j,s)
    #return j+χ*(i-1)+χ^2*(s-1)
    return i+χ*(j-1)+χ^2*(s-1)
end

function SR_exact(A)
    S = zeros(ComplexF64,2*χ^2,2*χ^2) #replace by undef array
    G = zeros(ComplexF64,χ,χ,2)
    L = zeros(ComplexF64,χ,χ,2)
    R = zeros(ComplexF64,χ,χ,2)
    Z = MPS_Z(A)

    for k in 1:dim
        sample = basis[k] #replace by Monte Carlo
        ρ_sample = MPS(sample,A)
        p_sample = ρ_sample*conj(ρ_sample)

        for s in 1:2
            for j in 1:χ
                for i in 1:χ
                    G[i,j,s] = derv_MPS(i, j, (-1)^(s-1), sample, A)
                    L[i,j,s]+= p_sample*conj(G[i,j,s])
                    R[i,j,s]+= p_sample*G[i,j,s]
                end
            end
        end
        for s in 1:2
            for j in 1:χ
                for i in 1:χ
                    for ss in 1:2
                        for jj in 1:χ
                            for ii in 1:χ
                                S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += p_sample*conj(G[i,j,s])*G[ii,jj,ss]
                            end
                        end
                    end
                end
            end
        end
    end

    #display(S)
    
    S./=Z
    L./=Z
    R./=Z

    for s in 1:2
        for j in 1:χ
            for i in 1:χ
                for ss in 1:2
                    for jj in 1:χ
                        for ii in 1:χ
                            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= L[i,j,s]*R[ii,jj,ss]
                        end
                    end
                end
            end
        end
    end

    return S
end

function optimize(A_init, δ0, QQ, G, k0, GS_exact, ΔE)
    #energies_array = [GS_exact] 
    #energies_arrayb = [GS_exact]   #list of energies, with first entry exact
    energies_array = [(MPS_local_energy(A_init)-GS_exact)/N] 
    energies_arrayb = [(MPS_local_energy(A_init)-GS_exact)/N] 
    A = deepcopy(A_init)
    B = deepcopy(A_init)
    converged=false
    #for k in 1:60
    e=1
    e_grad = zeros(χ,χ)
    k=0
    Q=1
    while converged==false
        k+=1
        #δ = δ0*Q^k
        #δ = δ0*sqrt(Q)*0.99^k
        δ = δ0*Q^(1/3)*0.99^k
        for n in 1:1#G*k
            gradA = (E_gradient(A))
            gradB = (E_gradient(B)) #take real part for complex wfunction
            #e_gradp, e_gradm = E_gradient_MC(Ap, Am, k0+10*k, 2)

            gradA./=maximum(abs.(gradA))
            #gradB./=maximum(abs.(gradB))
            #A = A-δ*grad

            flat_gradA = reshape(gradA,2*χ^2)
            flat_gradB = reshape(gradB,2*χ^2)
            flat_A = reshape(A,2*χ^2)
            flat_B = reshape(B,2*χ^2)
            #flat_A = flat_A-δ*flat_grad
            #flat_B = flat_B-δ*2*flat_gradB
            flat_B = flat_B-δ*0.7*sign.(flat_gradB)

            S=SR_exact(A)+0.0001*Matrix{Int}(I, χ*χ*2, χ*χ*2)
            flat_A = flat_A-δ*inv(S)*flat_gradA

            #display(A)

            A = reshape(flat_A,χ,χ,2)
            B = reshape(flat_B,χ,χ,2)

            A./=maximum(abs.(A))
            B./=maximum(abs.(B))
        end

        e_old = e
        e = MPS_local_energy(A)
        eb = MPS_local_energy(B)
        push!(energies_array,(e-GS_exact)/N)
        push!(energies_arrayb,(eb-GS_exact)/N)
        Q = (e-GS_exact)/N
        println("k=$k: ", real((e-GS_exact)/N), " ; ", real((eb-GS_exact)/N))

        #println(abs(e-e_old))
        if abs(e-e_old) < ΔE
            converged=true
        end
    end
    return A, energies_array, energies_arrayb
end

#A_init = rand(ComplexF64,χ,χ,2)
A_init = rand(χ,χ,2)

println("MCMC χ=$χ:")
@time begin
    A, energies_array, energies_arrayb = optimize(A_init, 0.25, 0.99, 10, 20, E_n[1], 10^-9)
end

println("MPS Ground State:  ", MPS_local_energy(A))#MPS_energy(Ap,Am)/N)
println("True Ground State: ", E_n[1])



yticks_array = [10.0^(-i) for i in 1:10]

p=plot(3:length(energies_array), energies_array[3:end], yticks=(yticks_array), dpi=600, yaxis=:log)
plot!(3:length(energies_arrayb), energies_arrayb[3:end])
display(p)

p=plot(3:length(energies_array), energies_array[3:end], dpi=600)
plot!(3:length(energies_arrayb), energies_arrayb[3:end])
display(p)













































error()

function MPS(sigma,Ap,Am)
    A=Matrix{Int}(I, χ, χ)
    for i in 1:N
        if sigma[i]==1
            A*=Ap
        else
            A*=Am
        end
    end
    return tr(A)
end

function MPS_Z(Ap, Am)
    Z=0
    for j in 1:dim
        sigma = basis[j]
        Z+=MPS(sigma,Ap,Am)^2
    end
    return Z
end

function MPS_energy(Ap, Am)
    e_total=0
    Z=0

    for j in 1:dim
        sigma = basis[j]
        p=MPS(sigma,Ap,Am)*conj(MPS(sigma,Ap,Am))
        Z+=p

        # Interaction term
        e_i = 0
        for i in 1:N
            e_i += sigma[i]*sigma[mod(i,N)+1]
        end
        e_interaction=-J*e_i

        # Field term
        states_with_flip = [flip(sigma, i) for i in 1:N]
        e_f = 0
        for state in states_with_flip
            e_f -= MPS(state,Ap,Am)/MPS(sigma,Ap,Am)#*MPS(sigma,Ap,Am)
        end
        e_field=h*e_f
        e_total+=(e_interaction+e_field)*p
    end
    return e_total/Z
end

function Metropolis_full_sweep(sigma, Ap, Am, states_with_flip)
    sigma_new = deepcopy(sigma)
    for i in 1:N
        #sigma_p = deepcopy(sigma)
        #sigma_p[i] = -sigma[i]
        sigma_p = states_with_flip[i]
        metropolis_prob = MPS(sigma_p,Ap,Am)^2/MPS(sigma,Ap,Am)^2
        if rand() <= metropolis_prob
            sigma_new[i] = -sigma_new[i]
            #sigma[i] = -sigma[i]
        end
    end
    return sigma_new
end

function Metropolis_single_flip(sigma, Ap, Am, site_index)
    #site_index = rand(1:N)
    sigma_p = deepcopy(sigma)
    sigma_p[site_index] = -sigma[site_index]
    metropolis_prob = real((MPS(sigma_p,Ap,Am)*conj(MPS(sigma_p,Ap,Am)))/(MPS(sigma,Ap,Am)*conj(MPS(sigma,Ap,Am))))
    if rand() <= metropolis_prob
        sigma[site_index] = -sigma[site_index]
    end
    return sigma
end

function MPS_local_energy_MC_parallel(Ap, Am, N_MC, N_sweeps)
    E_TOTAL = 0
    e_total_thread=zeros(Threads.nthreads())
    Threads.@threads for i in 1:Threads.nthreads()
        #e_total=0
        sigma=ones(N)
        states_with_flip = [flip(sigma, i) for i in 1:N]
        for j in 1:N_MC
            for s in 1:N_sweeps
                for i in 1:N
                    sigma = Metropolis_single_flip(sigma, Ap, Am, i)
                end
            end
            #sigma = Metropolis_full_flip(sigma)
            #sigma = Metropolis_full_sweep(sigma, Ap, Am, states_with_flip)
            #sigma = Metropolis_single_flip(sigma, Ap, Am)
            #p = MPS(sigma,Ap,Am)^2 # prob dist
            #display(sigma)

            # Interaction term
            e_i = 0
            for i in 1:N
                e_i += sigma[i]*sigma[mod(i,N)+1]
            end
            e_interaction=-J*e_i

            states_with_flip = [flip(sigma, i) for i in 1:N]

            # Field term
            e_f = 0
            for state in states_with_flip
                e_f += real(MPS(state,Ap,Am)/MPS(sigma,Ap,Am))
            end
            e_field=-h*e_f

            e_total_thread[Threads.threadid()]+=e_interaction+e_field
        end
        #E_TOTAL+=e_total
    end
    E_TOTAL=sum(e_total_thread)/Threads.nthreads()/N_MC
    return E_TOTAL#/N_MC#e_total/N_MC
end


function MPS_local_energy(Ap, Am)
    e_total=0
    Z=0
    for j in 1:dim
        sigma = basis[j]
        p = MPS(sigma,Ap,Am)*conj(MPS(sigma,Ap,Am))   #prob density
        Z+=p

        # Interaction term
        e_i = 0
        for i in 1:N
            e_i += sigma[i]*sigma[mod(i,N)+1]
        end
        e_interaction=-J*e_i

        # Field term
        states_with_flip = [flip(sigma, i) for i in 1:N]
        e_f = 0
        for state in states_with_flip
            e_f += MPS(state,Ap,Am)/MPS(sigma,Ap,Am)
        end
        e_field=-h*e_f
        e_total+=p*(e_interaction+e_field)
    end
    return e_total/Z
end

#GRADIENT:
function B_list(m, sigma, Ap, Am)
    #B_list=Matrix{Float64}[]
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for i in 1:N-1
        if sigma[mod(m+i-1,N)+1]==1
            push!(B_list,Ap)
        else
            push!(B_list,Am)
        end
    end
    return B_list
end

function derv_MPS(i, j, s, sigma, Ap, Am)
    sum = 0
    for m in 1:N
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

function E_gradient_SR(Ap, Am, N_MC, N_sweeps)
    e_gradp = Matrix{Float64}(zeros(χ,χ))
    e_gradm = Matrix{Float64}(zeros(χ,χ))

    e_local = MPS_local_energy_MC_parallel(Ap, Am, N_MC, N_sweeps)
    #e_local = MPS_local_energy(Ap, Am)

    sigma = ones(N)
    states_with_flip = [flip(sigma, i) for i in 1:N]

    for k in 1:N_MC
        e_interaction = 0
        e_field = 0

        for s in 1:N_sweeps
            for i in 1:N
                sigma = Metropolis_single_flip(sigma, Ap, Am, i)
            end
        end

        # Interaction term:
        e_i = 0
        for m in 1:N
            e_i += sigma[m]*sigma[mod(m,N)+1]
        end
        e_interaction=-J*e_i

        states_with_flip = [flip(sigma, i) for i in 1:N]

        # Field term:
        if MPS(sigma,Ap,Am)!=0
            e_f = 0
            for state in states_with_flip
                e_f -= real(MPS(state,Ap,Am)/MPS(sigma,Ap,Am))
            end
            e_field+=h*e_f
        end

        cov_matrix = Matrix{ComplexF64}(zeros(2*χ^2,2*χ^2))

        Dp = Vector{ComplexF64}(zeros(χ^2))
        Dm = Vector{ComplexF64}(zeros(χ^2))
        D = Vector{ComplexF64}(zeros(2*χ^2))
        for i in 1:χ
            for j in 1:χ
                Dp[i+χ*(j-1)] = derv_MPS(i, j, 1, sigma, Ap, Am)
                Dm[i+χ*(j-1)] = derv_MPS(i, j,-1, sigma, Ap, Am)
            end
        end
        D = vcat(Dp,Dm)

        e_gradp += Dp*(e_interaction+e_field-e_local)
        e_gradm += Dm*(e_interaction+e_field-e_local)

        for α in 1:2*χ^2
            for β in 1:2*χ^2
                cov_matrix[α,β] = D[α]*D[β] - D[α]*D[β]
            end
        end
    end

    return e_gradp/N_MC, e_gradm/N_MC
end

function E_gradient_MC(Ap, Am, N_MC, N_sweeps)
    e_gradp = Matrix{Float64}(zeros(χ,χ))
    e_gradm = Matrix{Float64}(zeros(χ,χ))

    e_local = MPS_local_energy_MC_parallel(Ap, Am, N_MC, N_sweeps)
    #e_local = MPS_local_energy(Ap, Am)

    sigma = ones(N)
    states_with_flip = [flip(sigma, i) for i in 1:N]

    for k in 1:N_MC
        e_interaction = 0
        e_field = 0

        #sigma = Metropolis_full_sweep(sigma, states_with_flip)
        #sigma = Metropolis_full_flip(sigma)
        #sigma = Metropolis_single_flip(sigma, Ap, Am)
        for s in 1:N_sweeps
            for i in 1:N
                sigma = Metropolis_single_flip(sigma, Ap, Am, i)
            end
        end

        # Interaction term:
        e_i = 0
        for m in 1:N
            e_i += sigma[m]*sigma[mod(m,N)+1]
        end
        e_interaction=-J*e_i

        states_with_flip = [flip(sigma, i) for i in 1:N]

        # Field term:
        if MPS(sigma,Ap,Am)!=0
            e_f = 0
            for state in states_with_flip
                e_f -= real(MPS(state,Ap,Am)/MPS(sigma,Ap,Am))
            end
            e_field+=h*e_f
        end

        #Dp = Matrix{Float64}(zeros(χ,χ))
        #Dm = Matrix{Float64}(zeros(χ,χ))
        Dp = Matrix{ComplexF64}(zeros(χ,χ))
        Dm = Matrix{ComplexF64}(zeros(χ,χ))
        for i in 1:χ
            for j in 1:χ
                Dp[i,j] = derv_MPS(i, j, 1, sigma, Ap, Am)
                Dm[i,j] = derv_MPS(i, j,-1, sigma, Ap, Am)
            end
        end
        e_gradp += Dp*(e_interaction+e_field-e_local)
        e_gradm += Dm*(e_interaction+e_field-e_local)
    end

    return e_gradp/N_MC, e_gradm/N_MC
end

function E_gradient(Ap, Am, N_MC)
    e_gradp = Matrix{Float64}(zeros(χ,χ))
    e_gradm = Matrix{Float64}(zeros(χ,χ))

    #e_grad = [Matrix{Float64}(zeros(χ,χ)) for _ = 1:N]
    e_local = MPS_local_energy(Ap, Am)
    #e_local = MPS_local_energy_MC_parallel(Ap, Am, N_MC, 1)
    #e_local = MPS_local_energy_random(Ap, Am, 100)
    Z=0

    for k in 1:dim
        e_interaction = 0
        e_field = 0

        sigma = basis[k]
        p = MPS(sigma,Ap,Am)*conj(MPS(sigma,Ap,Am))
        Z+=p

        # Interaction term:
        e_i = 0
        for m in 1:N
            e_i += sigma[m]*sigma[mod(m,N)+1]
        end
        e_interaction=-J*e_i

        # Field term:
        if real(MPS(sigma,Ap,Am))!=0
            states_with_flip = [flip(sigma, i) for i in 1:N]
            e_f = 0
            for state in states_with_flip
                e_f -= MPS(state,Ap,Am)/MPS(sigma,Ap,Am)
            end
            e_field=h*e_f
        end

        Dp = Matrix{ComplexF64}(zeros(χ,χ))
        Dm = Matrix{ComplexF64}(zeros(χ,χ))
        for i in 1:χ
            for j in 1:χ
                Dp[i,j] = derv_MPS(i, j, 1, sigma, Ap, Am)
                Dm[i,j] = derv_MPS(i, j,-1, sigma, Ap, Am)
            end
        end

        e_gradp += p*Dp*(e_interaction+e_field-e_local)
        e_gradm += p*Dm*(e_interaction+e_field-e_local)
    end
    return e_gradp/Z, e_gradm/Z
end


function fix_common_evals(Ap, Am)
    Ap_diag, Ap_Q = eigen(Ap)
    Am_diag, Am_Q = eigen(Am)
    A_av = Diagonal(Ap_diag+Am_diag)/2
    Ap_new = Ap_Q*A_av*inv(Ap_Q)
    Am_new = Am_Q*A_av*inv(Am_Q)
    #if conj(Ap_new) == Ap_new
    #    return Ap_new, Am_new
    #else
    #    return Ap, Am
    #end
    return Ap_new, Am_new
end

function calculate_MPS_magnetization(Ap, Am, basis)
    M=0
    for (i, state) in enumerate(basis)
        state = basis[i]
        state_M = sum(state)*MPS(state, Ap, Am)^2   #or use abs2. 
        M += abs(state_M)
    end
    return M/(N*MPS_Z(Ap, Am))
end

function double_χ(Ap, Am)
    global χ*=2
    Bp = kron(Ap,[1 1; 1 1])
    Bm = kron(Am,[1 1; 1 1])
    return Bp, Bm
end


function optimize(Ap_init, Am_init, δ0, Q, G, k0, GS_exact, ΔE)
    energies_array = [GS_exact]   #list of energies, with first entry exact
    Ap = deepcopy(Ap_init)
    Am = deepcopy(Am_init)
    converged=false
    #for k in 1:60
    e=1
    e_gradp, e_gradm = zeros(χ,χ), zeros(χ,χ)
    k=0
    while converged==false
        Ap_old = deepcopy(Ap)
        Am_old = deepcopy(Am)

        k+=1
        δ = δ0*Q^k
        for n in 1:G*k
            e_gradp_old, e_gradm_old = e_gradp, e_gradm
            e_gradp, e_gradm = (E_gradient(Ap, Am, 1)) #take real part for complex wfunction
            #e_gradp, e_gradm = E_gradient_MC(Ap, Am, k0+10*k, 2)


            #display(e_gradp)
            #display(e_gradp./maximum(e_gradp))

            #Ap = Ap - δ*e_gradp./maximum(abs.(e_gradp))# - δ/2*e_gradp_old./maximum(e_gradp_old)#./maximum(e_gradp)#.*rand(χ,χ)
            #Am = Am - δ*e_gradm./maximum(abs.(e_gradm))# - δ/2*e_gradm_old./maximum(e_gradm_old)#./maximum(e_gradm)#.*rand(χ,χ)

            Ap = Ap - δ*(sign.(e_gradp)) - δ/3*(sign.(e_gradp_old))
            Am = Am - δ*(sign.(e_gradm)) - δ/3*(sign.(e_gradm_old))

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
        e_old = e
        e = MPS_local_energy(Ap,Am)#MPS_local_energy_MC(Ap,Am, 1000)
        #push!(energies_array, e)
        println("k=$k: ", (e-GS_exact)/N)

        #println(abs(e-e_old))
        if abs(e-e_old) < ΔE
            converged=true
        end
    end
    return Ap, Am, energies_array
end

function flatten_index(i,j,s)
    #return j+χ*(i-1)+χ^2*(s-1)
    return i+χ*(j-1)+χ^2*(s-1)
end

function SR_exact(Ap,Am) #exactly
    S = zeros(ComplexF64,2*χ^2,2*χ^2) #replace by undef array
    G = zeros(ComplexF64,χ,χ,2)
    L = zeros(ComplexF64,χ,χ,2)
    R = zeros(ComplexF64,χ,χ,2)
    Z = MPS_Z(Ap,Am)

    for k in 1:dim
        sample = basis[k] #replace by Monte Carlo
        ρ_sample = MPS(sample,Ap,Am)
        p_sample = ρ_sample*conj(ρ_sample)

        for s in 1:2
            for j in 1:χ
                for i in 1:χ
                    G[i,j,s] = derv_MPS(i, j, (-1)^(s-1), sample, Ap, Am)
                    L[i,j,s]+= p_sample*conj(G[i,j,s])
                    R[i,j,s]+= p_sample*G[i,j,s]
                end
            end
        end
        for s in 1:2
            for j in 1:χ
                for i in 1:χ
                    for ss in 1:2
                        for jj in 1:χ
                            for ii in 1:χ
                                S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] += p_sample*conj(G[i,j,s])*G[ii,jj,ss]
                            end
                        end
                    end
                end
            end
        end
    end

    #display(S)
    
    S./=Z
    L./=Z
    R./=Z

    for s in 1:2
        for j in 1:χ
            for i in 1:χ
                for ss in 1:2
                    for jj in 1:χ
                        for ii in 1:χ
                            S[flatten_index(i,j,s),flatten_index(ii,jj,ss)] -= L[i,j,s]*R[ii,jj,ss]
                        end
                    end
                end
            end
        end
    end

    return S
end


function SRoptimize(Ap_init, Am_init, δ0, Q, G, k0, GS_exact, ΔE)
    energies_array = [GS_exact]   #list of energies, with first entry exact
    Ap = deepcopy(Ap_init)
    Am = deepcopy(Am_init)
    converged=false
    #for k in 1:60
    e=1
    e_gradp, e_gradm = zeros(χ,χ), zeros(χ,χ)
    k=0
    while converged==false
        #Ap_old = deepcopy(Ap)
        #Am_old = deepcopy(Am)

        k+=1
        δ = δ0*Q^k
        for n in 1:1#G*k
            A = vcat(reshape(Ap,χ^2),reshape(Am,χ^2))


            #e_gradp_old, e_gradm_old = e_gradp, e_gradm
            #e_gradp, e_gradm = (E_gradient(Ap, Am, 1)) #take real part for complex wfunction
            e_gradp, e_gradm = E_gradient_MC(Ap, Am, k0+10*k, 1)

            

            grad = vcat(reshape(e_gradp,χ^2),reshape(e_gradm,χ^2))

            #display(e_gradp)
            #display(e_gradp./maximum(e_gradp))

            #Ap = Ap - δ*e_gradp./maximum(abs.(e_gradp))# - δ/2*e_gradp_old./maximum(e_gradp_old)#./maximum(e_gradp)#.*rand(χ,χ)
            #Am = Am - δ*e_gradm./maximum(abs.(e_gradm))# - δ/2*e_gradm_old./maximum(e_gradm_old)#./maximum(e_gradm)#.*rand(χ,χ)

            #Ap = Ap - δ*(sign.(e_gradp)) - δ/3*(sign.(e_gradp_old))
            #Am = Am - δ*(sign.(e_gradm)) - δ/3*(sign.(e_gradm_old))

            #Ap = Ap - δ*(sign.(e_gradp)).*rand(χ,χ) - δ/2*(sign.(e_gradp_old)).*rand(χ,χ)
            #Am = Am - δ*(sign.(e_gradm)).*rand(χ,χ) - δ/2*(sign.(e_gradm_old)).*rand(χ,χ)

            #Ap = Ap - δ*(sign.(e_gradp))#.*rand(χ,χ)
            #Am = Am - δ*(sign.(e_gradm))#.*rand(χ,χ)

            #A = A-δ*(sign.(grad))
            grad./=maximum(abs.(grad))
            #A = A-δ*grad

            S=SR_exact(Ap,Am)+0.0001*Matrix{Int}(I, χ*χ*2, χ*χ*2)
            A = A-δ*inv(S)*grad

            #display(A)

            A = reshape(A,χ,χ,2)
            Ap = A[:, :, 1]
            Am = A[:, :, 2]

            #Ap, Am = fix_common_evals(Ap, Am)
            Ap/=maximum(abs.(Ap))
            Am/=maximum(abs.(Am))

            #display(δ*(sign.(e_gradp)))
            #display(e_gradp./maximum(abs.(e_gradp)))
        end
        #display(sign.(e_gradp))
        #display(e_gradp./maximum(abs.(e_gradp)))
        #display(Ap)
        e_old = e
        e = MPS_local_energy(Ap,Am)#MPS_local_energy_MC(Ap,Am, 1000)
        #push!(energies_array, e)
        println("k=$k: ", (e-GS_exact)/N)

        #println(abs(e-e_old))
        if abs(e-e_old) < ΔE
            converged=true
        end
    end
    return Ap, Am, energies_array
end


Ap_init = rand(χ,χ)
Am_init = rand(χ,χ)


S = SR_exact(Ap_init,Am_init)+0.0001*Matrix{Int}(I, χ*χ*2, χ*χ*2)
#display(S)
#display(inv(S))
#error()

println("MCMC χ=$χ:")
@time begin
    Ap, Am, energies_array = optimize(Ap_init, Am_init, 0.1, 0.95, 10, 20, E_n[1], 10^-23)
end

println("MPS Ground State:  ", MPS_energy(Ap,Am))#MPS_energy(Ap,Am)/N)
println("True Ground State: ", E_n[1])

#println("MPS magnetization:  ", calculate_MPS_magnetization(Ap, Am, basis))
#println("True magnetization: ", magnetization(N, J, h, basis, ψ_n[:,1]))