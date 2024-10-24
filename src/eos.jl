using LinearAlgebra
abstract type EOS end


struct Polytrope <:EOS
    gamma::Float64
end

function Pressure(rho::Float64,eos::Polytrope)::Float64
    return power(rho,eos.gamma)
end







#function F(x::Vector{Float64})
#    return [
#        equation_1,
#        equation_2,
#        ...,
#        equation_n
#    ]
#end

function F(x::Vector{Float64})
    return [
        x[1]^2 + x[2]^2 - 10,
        x[1] * x[2] + x[2] - 5
    ]
end



#function J(x::Vector{Float64})
#    return [
#        [∂f_1/∂x_1  ∂f_1/∂x_2; ...],
#        [∂f_2/∂x_1  ∂f_2/∂x_2; ...],
#        ...
#        [∂f_n/∂x_1  ∂f_n/∂x_2; ...]
#    ]
#end


function J(x::Vector{Float64})
    return [
        2*x[1]   2*x[2];
        x[2]     x[1] + 1
    ]
end





function newton_raphson(initial_guess::Vector{Float64}; tol=1e-10, max_iter=100)
    x = initial_guess
    for i in 1:max_iter
        fx = F(x)
        jx = J(x)

        if any(isnan.(fx)) || any(isinf.(fx)) || any(isnan.(jx)) || any(isinf.(jx))
            error("NaN or Inf. The program has stopped.")
        end

        delta = jx \ -fx

        x += delta

        if norm(delta) < tol
            return x
        end
    end
    error("The method does not converge after $max_iter iterations")
end

initial_guess = [0.5, 0.5]
solution = newton_raphson(initial_guess)
println("Solution: ", solution)
