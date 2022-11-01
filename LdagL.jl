using LinearAlgebra
include("ED_Ising.jl")
include("ED_Lindblad.jl")
using BenchmarkTools


J=1.0
γ=1.0
N=1
dim = 2^N

l1 = make_Liouvillian(J*sx,γ*sm)
l1_dag = conj(transpose(l1))

display(l1)
display(l1_dag)


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

function MPO_inserted(sample, A, j, state, u, state2)
    MPO=Matrix{ComplexF64}(I, χ, χ)
    for i::UInt8 in 1:N
        if i==u
            MPO*=A[:,:,dINDEX[state2]]
        elseif i==j
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
            ρ_sample = conj(MPO(sample,A))
            local_L=0
            for j in 1:N
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1_dag
                #display(bra_L)

                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]

                    micro_sample = sample
                    micro_sample.ket[j]=state[1]
                    micro_sample.bra[j]=state[2]

                    for u in 1:N
                        s2 = dVEC[(micro_sample.ket[u],micro_sample.bra[u])]
                        bra_L2 = transpose(s2)*l1

                        for v in 1:4
                            loc2 = bra_L2[v]
                            state2 = TPSC[v]
                            local_L += loc*loc2*MPO_inserted(sample,A,j,state,u,state2)
                        end
                    end
                end

                #local_L/=ρ_sample
                
                #| Should this not be in the loop outside this one?
                #V 
                #L_LOCAL+=local_L*conj(local_L)*ρ_sample#*conj(ρ_sample)
                #println(local_L*conj(local_L))
            end
            L_LOCAL+=local_L*conj(local_L)*ρ_sample
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
    B_list=Matrix{ComplexF64}[Matrix{Int}(I, χ, χ)]
    for j::UInt8 in 1:N-1 #fix N=1 case!
        push!(B_list,A[:,:,dINDEX[(sample.ket[mod(m+j-1,N)+1],sample.bra[mod(m+j-1,N)+1])]])
    end
    return B_list
    #return 1
    #return [[1 0; 0 1]]
    #return [Matrix{Int}(I, χ, χ)]#[[1 0; 0 1]]
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

            local_L=0
            local_∇=0

            #L∇L*:
            for j in 1:N
                #local_L=0
                #local_∇L=0
                s = dVEC[(sample.ket[j],sample.bra[j])]
                bra_L = transpose(s)*l1
                #display(bra_L)

                for i in 1:4
                    loc = bra_L[i]
                    state = TPSC[i]

                    micro_sample = sample
                    micro_sample.ket[j]=state[1]
                    micro_sample.bra[j]=state[2]

                    for u in 1:N
                        s2 = dVEC[(micro_sample.ket[u],micro_sample.bra[u])]
                        bra_L2 = transpose(s2)*l1

                        for v in 1:4
                            loc2 = bra_L2[v]
                            state2 = TPSC[v]
                            local_L += loc*loc2*MPO_inserted(sample,A,j,state,u,state2)
                        end
                    end
                end


                #local_L /=ρ_sample
                #local_∇L/=conj(ρ_sample)
            
                #L∇L+=local_L*local_∇L
            end
            local_∇+=derv_MPO(ii,jj,u,sample,A)

            L∇L+=local_L*local_∇

            #ΔLL:
            local_Δ=0
            local_Δ+=conj(MPO(sample,A))*derv_MPO(ii,jj,u,sample,A)
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

δχ = 0.001
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













