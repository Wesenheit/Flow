abstract type EOS end


struct Polytrope <:EOS
    gamma::Float64
end

function Pressure(rho::Float64,eos::Polytrope)::Float64
    return power(rho,eos.gamma)
end