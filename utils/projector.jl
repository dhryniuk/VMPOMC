export projector


mutable struct projector
    ket::Vector{Bool}
    bra::Vector{Bool}
end

projector(p::projector) = projector(copy(p.ket), copy(p.bra))

idx(sample::projector,i::UInt8) = 1+2*sample.ket[i]+sample.bra[i]