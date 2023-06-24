export Projector


mutable struct Projector
    ket::Vector{Bool}
    bra::Vector{Bool}
end

Projector(p::Projector) = Projector(copy(p.ket), copy(p.bra))

idx(sample::Projector,i::UInt8) = 1+2*sample.ket[i]+sample.bra[i]

export print_canonical

function print_canonical(p::Projector)
    s = ""
    for i in 1:length(p.ket)
        s*=string(p.ket[i],pad=1)*string(p.bra[i],pad=1)
        s*=", "
    end
    println(s)
end