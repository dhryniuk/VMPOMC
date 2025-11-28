export Parameters


mutable struct Parameters
    N::Int64
    dim_H::Int64
    dim_L::Int64
    χ::Int64
    Jx::Float32
    Jy::Float32
    J::Float32
    hx::Float32
    hz::Float32
    γ::Float32
    γ_d::Float32
    α::Float64
end

Base.display(params::Parameters) = begin
    println("\nParameters:")
    println("N\t\t", params.N)
    println("dim_H\t\t", params.dim_H)
    println("dim_L\t\t", params.dim_L)
    println("χ\t\t", params.χ)
    println("Jx\t\t", params.Jx)
    println("Jy\t\t", params.Jy)
    println("Jz\t\t", params.J)
    println("hx\t\t", params.hx)
    println("hz\t\t", params.hz)
    println("γ_l\t\t", params.γ)
    println("γ_d\t\t", params.γ_d)
    println("α\t\t", params.α)
end

function Parameters(; N::Int,
                    χ=0.0, Jx=0.0, Jy=0.0, J=0.0,
                    hx=0.0, hz=0.0, γ=0.0, γ_d=0.0, α=0.0)
    dim  = 2^N
    dim2 = 2^(2*N)
    # convert to Float64 to ensure type matches
    return Parameters(N, dim, dim2,
                      float(χ), float(Jx), float(Jy), float(J),
                      float(hx), float(hz), float(γ), float(γ_d), float(α))
end

