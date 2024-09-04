export tensor_calculate_magnetization, tensor_calculate_correlation


function tensor_calculate_magnetization(optimizer::Optimizer{T}, op::Array{ComplexF64}) where {T<:Complex{<:AbstractFloat}}
    
    A = optimizer.A
    params = optimizer.params

    Af = reshape(A,params.χ,params.χ,2,2)
    B = zeros(T,params.χ,params.χ)
    @tensor B[a,b] := Af[a,b,c,d]*op[c,d]
    for _ in 1:params.N-1
        @tensor B[a,b] := B[a,c]*Af[c,b,e,e]
    end
    return @tensor B[a,a]
end

function tensor_calculate_correlation(params::Parameters, A::Array{ComplexF64,4}, op::Array{ComplexF64})

    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,c,d]*op[c,d]
    T = deepcopy(B)
    C = deepcopy(B)
    @tensor C[a,b] = B[a,c]*T[c,b]
    for _ in 1:params.N-2
        @tensor D[a,b] = C[a,c]*A[c,b,e,e]
        C=deepcopy(D)
    end
    return @tensor C[a,a]
end

function calculate_spin_spin_correlation(params::Parameters, A::Array{ComplexF64}, op, dist::Int)
    A = reshape(A,params.χ,params.χ,2,2)
    B = zeros(ComplexF64,params.χ,params.χ)
    D = zeros(ComplexF64,params.χ,params.χ)
    E = zeros(ComplexF64,params.χ,params.χ)
    @tensor B[a,b] = A[a,b,f,e]*op[e,f]
    @tensor D[a,b] = A[a,b,f,f]
    C = deepcopy(B)
    for _ in 1:dist-1
        @tensor E[a,b] = C[a,c]*D[c,b]
        C = deepcopy(E)
    end
    @tensor E[a,b] = C[a,c]*B[c,b]
    C = deepcopy(E)
    for _ in 1:params.N-1-dist
        @tensor E[a,b] = C[a,c]*D[c,b]
        C = deepcopy(E)
    end
    return @tensor C[a,a]
end

function tensor_purity(params::Parameters, A::Array{ComplexF64})
    A=reshape(A,params.χ,params.χ,2,2)
    B=rand(ComplexF64,params.χ,params.χ,params.χ,params.χ)
    @tensor B[a,b,u,v] = A[a,b,f,e]*A[u,v,e,f]
    C=deepcopy(B)
    D=deepcopy(B)
    for _ in 1:params.N-1
        @tensor D[a,b,u,v] = C[a,c,u,d]*B[c,b,d,v]
        C=deepcopy(D)
    end
    return @tensor C[a,a,u,u]
end
