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
    Z = MPO_Z(A)

    #1-local part:
    for k in 1:dim
        for l in 1:dim
            sample = density_matrix(1,basis[k],basis[l]) #replace by Monte Carlo
            ρ_sample = MPO(sample,A)
            local_L=0
            for j in 1:N
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1

                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]
                    local_L += loc*MPO_inserted(sample,A,j,state)
                end
            end
            #local_L/=ρ_sample
            L_LOCAL+=local_L*conj(local_L)#*ρ_sample*conj(ρ_sample)
        end
    end

    #2-local part:
    #TBD

    return L_LOCAL/Z#*conj(L_LOCAL)
end

χ=4
A_init=rand(ComplexF64, χ,χ,4)
A=copy(A_init)


#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= 0.5
#A[:, :, 2] .= 0.15im
#A[:, :, 3] .= -0.15im
#A[:, :, 4] .= 0.5

#id = Matrix{Int}(I, χ, χ)
#A=zeros(ComplexF64, χ,χ,4)
#A[:, :, 1] .= id+0.1*rand(χ,χ)
#A[:, :, 2] .= 0.1*rand(χ,χ)
#A[:, :, 3] .= 0.1*rand(χ,χ)
#A[:, :, 4] .= id+0.1*rand(χ,χ)

#error()


#GRADIENT:
function B_list(m, sample, A) #FIX m ORDERING
    B_list=Matrix{ComplexF64}[]
    for j::UInt8 in 1:N-1 #fix N=1 case!
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
    #return [Matrix{Int}(I, χ, χ)]
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

function Δ_MPO(i, j, u, sample, A)
    return derv_MPO(i, j, u, sample, A)/MPO(sample, A)
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

            local_L=0
            local_∇L=0

            #L∇L*:
            for j in 1:N
                #local_L=0
                #local_∇L=0
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
                    #local_∇L+= conj(loc)*derv_MPO(ii,jj,u,micro_sample,A_conjugate) #state,u or u,state?
                    local_∇L+= conj(loc)*MPO(micro_sample, A)*Δ_MPO(ii,jj,u,micro_sample,A)

                    #2-local part:
                    #TBD
                end

                #local_L /=ρ_sample
                #local_∇L/=conj(ρ_sample)
            
                #L∇L+=local_L*local_∇L
            end
            L∇L+=local_L*local_∇L

            #ΔLL:
            local_Δ=0
            #local_Δ+=MPO(sample,A)*MPO(sample, A_conjugate)*derv_MPO(ii,jj,u,sample,A_conjugate)
            local_Δ+=MPO(sample,A)*conj(MPO(sample, A))*derv_MPO(ii,jj,u,sample,A_conjugate)
            #local_Δ+=derv_MPO(ii,jj,u,sample,A)

            #ΔLL+=conj(local_Δ)
            ΔLL+=local_Δ
        end
    end

    ΔLL*=mean_local_Lindbladian

    #display(L∇L)

    return (L∇L-ΔLL)/Z
end


function normalize_MPO(A)
    #MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(0,0)]])^N#/2
    MPO=(A[:,:,dINDEX[(1,1)]]+A[:,:,dINDEX[(1,0)]]+A[:,:,dINDEX[(0,1)]]+A[:,:,dINDEX[(0,0)]])^N
    return tr(MPO)^(1/N)#::ComplexF64
end


g = calculate_gradient(J,γ,A,1,1,(0,0))
display(g)

δχ = 0.05
Q=1.0
@time begin
    for k in 1:1500
        new_A=zeros(ComplexF64, χ,χ,4)
        for i in 1:χ
            for j in 1:χ
                for u in TPSC
                    new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - (1+rand())*δχ*Q*sign.(calculate_gradient(J,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - rand()*δχ*Q^k*sign.(calculate_gradient(J,γ,A,i,j,u))
                    #new_A[i,j,dINDEX[u]] = A[i,j,dINDEX[u]] - δχ*(calculate_gradient(J,γ,A,i,j,u))
                end
            end
        end
        global A = new_A
        global A./=normalize_MPO(A)
        global Q=local_Lindbladian(J,γ,A)
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