using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")
using BenchmarkTools


J=1.0
h=1.0
γ=1.0
N=2
dim = 2^N

l1 = make_Liouvillian(h*sx,γ*sm)

display(l1)

basis=generate_bit_basis(N)
display(basis)

dINDEX = Dict((1,1) => 1, (1,0) => 2, (0,1) => 3, (0,0) => 4)
dVEC =   Dict((1,1) => [1,0,0,0], (1,0) => [0,1,0,0], (0,1) => [0,0,1,0], (0,0) => [0,0,0,1])
dUNVEC = Dict([1,0,0,0] => (1,1), [0,1,0,0] => (1,0), [0,0,1,0] => (0,1), [0,0,0,1] => (0,0))

TPSC = [(1,1),(1,0),(0,1),(0,0)]


mutable struct density_matrix
    coeff::ComplexF64
    ket::Vector{Int8}
    bra::Vector{Int8}
end
Base.copy(ρ::density_matrix) = density_matrix(ρ.coeff, ρ.ket, ρ.bra)

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



function Metropolis_single_flip_ket(sample, A, site_index)
    sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy?
    sample_p.ket[site_index] = 1-sample.ket[site_index]
    metropolis_prob::Float64 = real((MPO(sample_p, A)*conj(MPO(sample_p, A)))/(MPO(sample, A)*conj(MPO(sample, A))))
    if rand() <= metropolis_prob
        sample = sample_p
    end
    return sample
end

function Metropolis_single_flip_bra(sample, A, site_index)
    sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy?
    sample_p.bra[site_index] = 1-sample.bra[site_index]
    metropolis_prob::Float64 = real((MPO(sample_p, A)*conj(MPO(sample_p, A)))/(MPO(sample, A)*conj(MPO(sample, A))))
    if rand() <= metropolis_prob
        sample = sample_p
    end
    return sample
end

function Metropolis_sweep(sample, A, site_index)
    sample_p = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra)) #deepcopy?
    sample_p.bra[site_index] = 1-sample.bra[site_index]
    metropolis_prob::Float64 = real((MPO(sample_p, A)*conj(MPO(sample_p, A)))/(MPO(sample, A)*conj(MPO(sample, A))))
    if rand() <= metropolis_prob
        sample = sample_p
    end
    return sample
end

function MC_local_Lindbladian(J,h,γ,A) #should be called mean local lindbladian
    L_LOCAL=0
    Z = MPO_Z(A)

    N_MC=5

    #1-local part:
    sample = density_matrix(1,ones(N),ones(N))
    for k::UInt64 in 1:N_MC
        for l::UInt8 in 1:N
            sample = Metropolis_single_flip_ket(sample, A, l)
            sample = Metropolis_single_flip_bra(sample, A, l)
        end
        #sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
        ρ_sample = MPO(sample,A)
        p_sample = ρ_sample*conj(ρ_sample)

        local_L=0
        l_int = 0
        for j in 1:N
            l_int_α = 0
            l_int_β = 0

            #1-local part:
            s = dVEC[(sample.ket[j],sample.bra[j])]
            bra_L = transpose(s)*l1
            for i in 1:4
                loc = bra_L[i]
                state = TPSC[i]
                local_L += loc*MPO_inserted(sample,A,j,state)
            end

            #2-local part: #PBC
            l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
            l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
            l_int += -1.0im*J*(l_int_α-l_int_β)
        end

        local_L/=ρ_sample
        local_L+=l_int#*MPO(sample,A)
        L_LOCAL+=p_sample*local_L*conj(local_L)
    end

    return L_LOCAL/N_MC#/Z
end


function local_Lindbladian(J,h,γ,A) #should be called mean local lindbladian
    L_LOCAL=0
    Z = MPO_Z(A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            local_L=0
            l_int = 0
            for j in 1:N
                l_int_α = 0
                l_int_β = 0

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                end

                #2-local part: #PBC
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            local_L/=ρ_sample
            local_L+=l_int#*MPO(sample,A)
            L_LOCAL+=p_sample*local_L*conj(local_L)
        end
    end

    return L_LOCAL/Z
end

χ=2
A_init=rand(ComplexF64, χ,χ,4)
A=copy(A_init)

#println(local_Lindbladian(J,h,γ,A))

#error()

#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= 0.7
#A[:, :, 2] .= 0.2im
#A[:, :, 3] .= -0.1im
#A[:, :, 4] .= 0.3

#id = Matrix{Int}(I, χ, χ)
#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= id+0.1*rand(χ,χ)
#A[:, :, 2] .= 0.1*rand(χ,χ)
#A[:, :, 3] .= 0.1*rand(χ,χ)
#A[:, :, 4] .= id+0.1*rand(χ,χ)


println(local_Lindbladian(J,h,γ,A))
#error()


#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
end

function derv_MPO(i, j, u, sample, A)
    sum::ComplexF64 = 0
    for m::UInt8 in 1:N
        if u == (sample.ket[m],sample.bra[m])
            B = prod(B_list(m, sample, A))
            sum += B[i,j] + B[j,i]
        end
    end
    if i==j
        sum/=2
    end
    return sum
end

function Δ_MPO(i, j, u, sample, A)
    return derv_MPO(i, j, u, sample, A)/MPO(sample, A)
end

function calculate_gradient(J,h,γ,A,ii,jj,u)
    L∇L=0
    ΔLL=0
    Z = MPO_Z(A)
    mean_local_Lindbladian = local_Lindbladian(J,h,γ,A)
    #mean_local_Lindbladian = MC_local_Lindbladian(J,h,γ,A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            local_L=0
            local_∇L=0

            l_int = 0

            #L∇L*:
            for j in 1:N
                l_int_α = 0
                l_int_β = 0

                #1-local part:
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                #r =rand(1:4)
                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                    micro_sample = density_matrix(1,deepcopy(sample.ket),deepcopy(sample.bra))
                    micro_sample.ket[j] = state[1]
                    micro_sample.bra[j] = state[2]

                    local_∇L+= loc*derv_MPO(ii,jj,u,micro_sample,A)
                    #if i!=r
                    #    local_∇L+= loc*derv_MPO(ii,jj,u,micro_sample,A)
                    #end
                end

                #2-local part:
                l_int_α = (2*sample.ket[j]-1)*(2*sample.ket[mod(j-2,N)+1]-1)
                l_int_β = (2*sample.bra[j]-1)*(2*sample.bra[mod(j-2,N)+1]-1)
                l_int += -1.0im*J*(l_int_α-l_int_β)
            end

            local_L /=ρ_sample
            local_∇L/=ρ_sample

            #Add in interaction terms:
            local_L +=l_int#*MPO(sample, A)
            local_∇L+=l_int*Δ_MPO(ii,jj,u,sample,A)

            L∇L+=p_sample*local_L*conj(local_∇L)

            #ΔLL:
            local_Δ=p_sample*conj(Δ_MPO(ii,jj,u,sample,A))
            ΔLL+=local_Δ
        end
    end
    ΔLL*=mean_local_Lindbladian
    return (L∇L-ΔLL)/Z
end


function normalize_MPO(A)
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end


A=copy(A_init)
δχ = 0.01
Q=0.99
@time begin
    for k in 1:1500
        new_A=zeros(ComplexF64, χ,χ,4)
        for i in 1:χ
            for j in 1:χ
                for u in TPSC
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δχ*Q*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δχ*Q*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - rand()*δχ*Q^k*sign.(calculate_gradient(J,h,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - δχ*Q^k*(calculate_gradient(J,h,γ,A,i,j,u))
                end
            end
        end
        global A = new_A
        global A./=normalize_MPO(A)
        global Q=local_Lindbladian(J,h,γ,A)
        #global Q=sqrt(local_Lindbladian(J,h,γ,A))
        println(local_Lindbladian(J,h,γ,A))
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

#SR stuff:


function flatten_index(i,j,u)
    #return u+4*(j-1)+4*χ*(i-1)
    #return u+4*(i-1)+4*χ*(j-1)
    return i+χ*(j-1)+χ^2*(u-1)
end

function OLDcalculate_metric_tensor(A) #exactly
    S = zeros(ComplexF64,4*χ^2,4*χ^2) #replace by undef array
    G = zeros(ComplexF64,χ,χ,4)
    L = zeros(ComplexF64,χ,χ,4)
    R = zeros(ComplexF64,χ,χ,4)
    Z = MPO_Z(A)

    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            for i in 1:χ
                for j in 1:χ
                    for u in TPSC
                        G[i,j,dINDEX[u]] = Δ_MPO(i,j,u,sample,A)
                        L[i,j,dINDEX[u]]+=p_sample*conj(G[i,j,dINDEX[u]])
                        R[i,j,dINDEX[u]]+=p_sample*G[i,j,dINDEX[u]]
                    end
                end
            end
            for i in 1:χ ### CHECK i-j ORDER
                for j in 1:χ
                    for u in TPSC
                        for ii in 1:χ
                            for jj in 1:χ
                                for uu in TPSC
                                    S[flatten_index(i,j,dINDEX[u]),flatten_index(ii,jj,dINDEX[uu])] += p_sample*conj(G[i,j,dINDEX[u]])*G[ii,jj,dINDEX[uu]]
                                end
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

    for i in 1:χ
        for j in 1:χ
            for u in TPSC
                for ii in 1:χ
                    for jj in 1:χ
                        for uu in TPSC
                            S[flatten_index(i,j,dINDEX[u]),flatten_index(ii,jj,dINDEX[uu])] -= L[i,j,dINDEX[u]]*R[ii,jj,dINDEX[uu]]
                        end
                    end
                end
            end
        end
    end

    return S
end
function calculate_metric_tensor(A) #exactly
    S = zeros(ComplexF64,4*χ^2,4*χ^2) #replace by undef array
    Gm = zeros(ComplexF64,χ,χ,4)
    Lm = zeros(ComplexF64,χ,χ,4)
    Rm = zeros(ComplexF64,χ,χ,4)
    G = zeros(ComplexF64,4*χ^2)
    L = zeros(ComplexF64,4*χ^2)
    R = zeros(ComplexF64,4*χ^2)
    Z = MPO_Z(A)

    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            p_sample = ρ_sample*conj(ρ_sample)

            for i in 1:χ
                for j in 1:χ
                    for u in TPSC
                        Gm[i,j,dINDEX[u]] = Δ_MPO(i,j,u,sample,A)
                        Lm[i,j,dINDEX[u]]+= p_sample*conj(Gm[i,j,dINDEX[u]])
                        Rm[i,j,dINDEX[u]]+= p_sample*Gm[i,j,dINDEX[u]]
                    end
                end
            end
            G+=reshape(Gm,4*χ^2)
            L+=reshape(Lm,4*χ^2)
            R+=reshape(Rm,4*χ^2)
            for i in 1:(4*χ^2)
                for j in 1:(4*χ^2)
                    S[i,j] += p_sample*conj(G[i])*G[j]
                end
            end
        end
    end

    #display(S)
    
    S./=Z
    L./=Z
    R./=Z

    for i in 1:(4*χ^2)
        for j in 1:(4*χ^2)
            S[i,j] -= L[i]*R[j]
        end
    end

    return S
end

S=OLDcalculate_metric_tensor(A)
#display(S)
#display(eigen(S))

S=calculate_metric_tensor(A)
#display(S)
#display(eigen(S))


#display(inv(S))
#display(inv(S)./maximum(abs.(inv(S))))

#G = zeros(ComplexF64,χ,χ,4)
#for i in 1:χ
#    for j in 1:χ
#        for u in TPSC
#            G[i,j,dINDEX[u]] = flatten_index(i,j,dINDEX[u])
#            println(flatten_index(i,j,dINDEX[u]))
#        end
#    end
#end
#display(reshape(G,2*2*4))


#error()

g = calculate_gradient(J,h,γ,A,1,1,(0,0))
display(g)
#error()


#Vectorized parameters gradient descent attempt:
δχ = 0.03
Q=0.95
@time begin
    for k in 1:500
        S_inv=Matrix{Int}(I, χ*χ*4, χ*χ*4)
        new_A=zeros(ComplexF64, χ,χ,4)
        new_A=reshape(new_A,χ*χ*4) #flatten array
        grads=zeros(ComplexF64, χ*χ*4)
        for i in 1:χ
            for j in 1:χ
                for u in TPSC
                    grads[i+χ*(j-1)+χ^2*(dINDEX[u]-1)] = calculate_gradient(J,h,γ,A,i,j,u)
                    #grads[i+χ*(j-1)+χ^2*(dINDEX[u]-1)] = sign.(calculate_gradient(J,h,γ,A,i,j,u))
                end
            end
        end
        grads./=maximum(abs.(grads))
        grads.*=-δχ
        grads.*=Q#^k

        #display(grads)

        S = OLDcalculate_metric_tensor(A)
        S_inv = inv(S + 0.0001*Matrix{Int}(I, χ*χ*4, χ*χ*4)) #Getting e-12 consistently with SR
        #if det(S)!=0
        #    S_inv = inv(S) + 0.001*Matrix{Int}(I, χ*χ*4, χ*χ*4)
        #    println("INVERTIBLE S!") ### SOMETHING WRONGA FOR χ>1
        #end

        #display(grads)
        global A=reshape(A,χ*χ*4) #flatten array

        #display(S_inv*grads)

        new_A=A+S_inv*grads
        #new_A=A+grads
        global A = new_A
        global A=reshape(A,χ,χ,4)
        global A./=normalize_MPO(A)
        #global Q=local_Lindbladian(J,h,γ,A)
        global Q=sqrt(local_Lindbladian(J,h,γ,A))
        println(local_Lindbladian(J,h,γ,A))
    end
end
