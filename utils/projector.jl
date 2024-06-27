export Projector, print_canonical


mutable struct Projector
    ket::Vector{Bool}
    bra::Vector{Bool}
end

Projector(p::Projector) = Projector(copy(p.ket), copy(p.bra))

idx(sample::Projector,i::UInt8) = 1+2*sample.ket[i]+sample.bra[i]
idx(sample::Projector,i::Int64) = 1+2*sample.ket[i]+sample.bra[i]


function print_canonical(p::Projector)
    s = ""
    for i in 1:length(p.ket)
        s*=string(p.ket[i],pad=1)*string(p.bra[i],pad=1)
        s*=", "
    end
    println(s)
end