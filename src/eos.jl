using LinearAlgebra
abstract type EOS end


struct Polytrope <:EOS
    gamma::Float64
end

function Pressure(u::Float64,eos::Polytrope)::Float64
    return (eos.gamma-1)*u
end
