export Projector


mutable struct Projector
    ket::Vector{Bool}
    bra::Vector{Bool}
end

Projector(p::Projector) = Projector(copy(p.ket), copy(p.bra))

idx(sample::Projector,i::UInt8) = 1+2*sample.ket[i]+sample.bra[i]