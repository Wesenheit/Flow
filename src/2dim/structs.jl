using StaticArrays
using LinearAlgebra
using Base.Threads

# Used scheme
# U - conserved varaibles
# U1 = rho ut - mass conservation
# U2 = T^t_t - energy conservation
# U3 = T^t_x - momentum conservation x
# U4 = T^t_y - momentum conservation y
 
# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = ux four-velocity in x
# P4 = uy four-velocity in y

mutable struct ParVector2D{T <:Real,Nx,Ny}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    function ParVector2D{T,Nx,Ny}() where {T, Nx,Ny}
        arr = zeros(Nx,Ny,4)
        new(arr,Nx,Ny)
    end
end

Base.copy(s::ParVector2D) = ParVector2D(s.arr,s.size_X,s.size_Y)

function Jacobian(x::MVector{4,Float64},buffer::MMatrix{4,4,Float64},eos::Polytrope)
    gam::Float64 = sqrt(1 + x[3]^2 + x[4]^2) ### gamma factor
    w::Float64 = eos.gamma * x[2] + x[1] ### enthalpy w = p + u + rho
    buffer[1,1] = gam
    buffer[1,2] = 0
    buffer[1,3] = x[1] * x[3] / gam
    buffer[1,4] = x[1] * x[4] / gam

    buffer[2,1] = - gam^2
    buffer[2,2] = (eos.gamma - 1) - gam ^2 * eos.gamma
    buffer[2,3] = -2 * x[3] * w
    buffer[2,4] = -2 * x[4] * w

    buffer[3,1] = gam * x[3]
    buffer[3,2] = gam * x[3] * eos.gamma
    buffer[3,3] = (2*x[3]^2 + x[4]^2 + 1) / gam * w
    buffer[3,4] = x[3] * x[4] / gam * w

    buffer[4,1] = gam * x[4]
    buffer[4,2] = gam * x[4] * eos.gamma
    buffer[4,3] = x[3] * x[4] / gam * w
    buffer[4,4] = (2*x[4]^2 + x[3]^2 + 1) / gam * w
end

function function_PtoU(x::MVector{4,Float64}, buffer::MVector{4,Float64},eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam * x[1]
    buffer[2] = (eos.gamma-1) * x[2] - gam^2 * w
    buffer[3] = x[3] * gam * w
    buffer[4] = x[4] * gam * w
end

function function_PtoU(x::Vector{Float64}, buffer::MVector{4,Float64},eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam * x[1]
    buffer[2] = (eos.gamma-1) * x[2] - gam^2 * w
    buffer[3] = x[3] * gam * w
    buffer[4] = x[4] * gam * w
end


function function_PtoFx(x::Vector{Float64}, buffer::MVector{4,Float64},eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam*x[3]
    buffer[2] = - w *x[3] * gam
    buffer[3] = x[3]^2 * w + (eos.gamma - 1) * x[2]
    buffer[4] = x[3] * x[4] * w 
end

function function_PtoFy(x::Vector{Float64}, buffer::MVector{4,Float64},eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam*x[4]
    buffer[2] = - w *x[4] * gam
    buffer[3] = x[3] * x[4] * w 
    buffer[4] = x[4]^2 * w + (eos.gamma - 1) * x[2]
end

function PtoFx(P::ParVector2D,Fx::ParVector2D,eos::EOS)
    @threads :static for i in 1:P.size_X
        buffer::MVector{4,Float64} = @MVector zeros(4)
        for j in 1:P.size_Y
            function_PtoFx(P.arr[i,j,:],buffer,eos)
            Fx.arr[i,j,1] = buffer[1]
            Fx.arr[i,j,2] = buffer[2]
            Fx.arr[i,j,3] = buffer[3]
            Fx.arr[i,j,4] = buffer[4]
        end
    end
end

function PtoFy(P::ParVector2D,Fy::ParVector2D,eos::EOS)
    @threads :static for i in 1:P.size_X
        buffer::MVector{4,Float64} = @MVector zeros(4)
        for j in 1:P.size_Y
            function_PtoFy(P.arr[i,j,:],buffer,eos)
            Fy.arr[i,j,1] = buffer[1]
            Fy.arr[i,j,2] = buffer[2]
            Fy.arr[i,j,3] = buffer[3]
            Fy.arr[i,j,4] = buffer[4]
        end
    end
end

function PtoU(P::ParVector2D,U::ParVector2D,eos::EOS)
    @threads :static for i in 1:P.size_X
        buffer::MVector{4,Float64} = @MVector zeros(4)
        for j in 1:P.size_Y
            function_PtoU(P.arr[i,j,:],buffer,eos)
            U.arr[i,j,1] = buffer[1]
            U.arr[i,j,2] = buffer[2]
            U.arr[i,j,3] = buffer[3]
            U.arr[i,j,4] = buffer[4]
        end
    end
end


function UtoP(U::ParVector2D,P::ParVector2D,eos::EOS,n_iter::Int64,tol::Float64=1e-10)
    @threads :static for i in 1:P.size_X            
        buff_start::MVector{4,Float64} = MVector(0.,0.,0.,0.)
        buff_fun::MVector{4,Float64} = MVector(0.,0.,0.,0.)
        buff_jac::MMatrix{4,4,Float64} = @MMatrix zeros(4,4)
        for j in 1:P.size_Y
            #println(i," ",j)
            buff_start[1] = P.arr[i,j,1]
            buff_start[2] = P.arr[i,j,2]
            buff_start[3] = P.arr[i,j,3]
            buff_start[4] = P.arr[i,j,4]
            buff_fun = MVector(0.,0.,0.,0.)
            buff_jac = @MMatrix zeros(4,4)
            #buff_start = MVector(0.,0.,0.,0.)
            for num in 1:n_iter
                function_PtoU(buff_start,buff_fun,eos)
                Jacobian(buff_start,buff_jac,eos)
                buff_fun[1] = buff_fun[1] - U.arr[i,j,1]
                buff_fun[2] = buff_fun[2] - U.arr[i,j,2]
                buff_fun[3] = buff_fun[3] - U.arr[i,j,3]
                buff_fun[4] = buff_fun[4] - U.arr[i,j,4]

                buff_fun = buff_jac \ buff_fun
    
                if norm(buff_fun) < tol
                    break
                end
                buff_start = buff_start .- buff_fun
            end
            P.arr[i,j,1] = buff_start[1]
            P.arr[i,j,2] = buff_start[2]
            P.arr[i,j,3] = buff_start[3]
            P.arr[i,j,4] = buff_start[4]
        end
    end
end

