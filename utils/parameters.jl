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

#write a constructor that defaults to 0 whenever some parameter is not specified...

function Parameters(N,χ,Jx,Jy,J,hx,hz,γ,γ_d,α)
    return Parameters(N,2^N,2^(2*N),χ,Jx,Jy,J,hx,hz,γ,γ_d,α)
end
