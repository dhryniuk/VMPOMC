Id = [1 0; 0 1]
sx = [0 1; 1 0]
sy = [0 -1im; 1im 0]
sz = [1 0; 0 -1]
sp = (sx+1im*sy)/2
sm = (sx-1im*sy)/2

⊗(x,y) = kron(x,y)

function make_Liouvillian(H, Γ)
    L_H = -1im*(H⊗Id - Id⊗transpose(H))
    L_D = Γ⊗conj(Γ) - (conj(transpose(Γ))*Γ)⊗Id/2 - Id⊗(transpose(Γ)*conj(Γ))/2
    return L_H + L_D
end