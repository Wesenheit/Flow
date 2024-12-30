using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using TimerOutputs
using KernelAbstractions
using CUDA

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



mutable struct ParVector2D{T <:Real}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    part
    function ParVector2D{T}(Nx,Ny) where {T}
        arr = zeros(4,Nx,Ny)
        partitions = partition(1:Ny,div(Ny,nthreads()))
        @assert length(partitions)==nthreads()
        new(arr,Nx,Ny,partitions)
    end
end

mutable struct CuParVector2D{T <:Real}
    # Parameter Vector
    arr::CuArray{T,3}
    size_X::Int64
    size_Y::Int64
    function CuParVector2D{T}(arr::ParVector2D{T}) where {T}
        new(CuArray(arr.arr),arr.size_X,arr.size_Y)
    end
end

Base.copy(s::ParVector2D) = ParVector2D(s.arr,s.size_X,s.size_Y)



function Jacobian(x::AbstractVector,buffer::AbstractVector,eos::Polytrope)
    gam::Float64 = sqrt(1 + x[3]^2 + x[4]^2) ### gamma factor
    w::Float64 = eos.gamma * x[2] + x[1] ### enthalpy w = p + u + rho
    buffer[1] = gam
    buffer[5] = 0
    buffer[9] = x[1] * x[3] / gam
    buffer[13] = x[1] * x[4] / gam

    buffer[2] = - gam^2
    buffer[6] = (eos.gamma - 1) - gam ^2 * eos.gamma
    buffer[10] = -2 * x[3] * w
    buffer[14] = -2 * x[4] * w

    buffer[3] = gam * x[3]
    buffer[7] = gam * x[3] * eos.gamma
    buffer[11] = (2 * x[3]^2 + x[4]^2 + 1) / gam * w
    buffer[15] = x[3] * x[4] / gam * w

    buffer[4] = gam * x[4]
    buffer[8] = gam * x[4] * eos.gamma
    buffer[12] = x[3] * x[4] / gam * w
    buffer[16] = (2 * x[4]^2 + x[3] ^ 2 + 1) / gam * w
end

@kernel function function_PtoU(P::AbstractArray, U::AbstractArray,gamma::Float64)
    i, j = @index(Global, NTuple)
    gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
    w = gamma * P[2,i,j] + P[1,i,j] 
    U[1,i,j] = gam * P[1,i,j]
    U[2,i,j] = (gamma-1) * P[2,i,j] - gam^2 * w
    U[3,i,j] = P[3,i,j] * gam * w
    U[4,i,j] = P[4,i,j] * gam * w
end




function function_UtoP(P::CuDeviceArray, U::CuDeviceArray,gamma::Float64,n_iter::Int64,tol::Float64=1e-10)
    @sync for (id_th,chunk) in enumerate(P.part)
        @spawn begin
            buff_start::MVector{4,Float64} = @MVector zeros(4)
            buff_out::MVector{4,Float64} = @MVector zeros(4)
            buff_fun::MVector{4,Float64} = @MVector zeros(4)
            buff_jac::MVector{16,Float64} = @MVector zeros(16)
            for j in chunk
                for i in 1:P.size_X
                    buff_start[1] = P.arr[1,i,j]
                    buff_start[2] = P.arr[2,i,j]
                    buff_start[3] = P.arr[3,i,j] 
                    buff_start[4] = P.arr[4,i,j]
                    for _ in 1:n_iter
                        function_PtoU(buff_start,buff_fun,eos)
                        Jacobian(buff_start,buff_jac,eos)
                        buff_fun[1] = buff_fun[1] - U.arr[1,i,j]
                        buff_fun[2] = buff_fun[2] - U.arr[2,i,j]
                        buff_fun[3] = buff_fun[3] - U.arr[3,i,j]
                        buff_fun[4] = buff_fun[4] - U.arr[4,i,j]
                        
                        LU_dec!(buff_jac,buff_fun,buff_out)
                        #mat = lu!(reshape(buff_jac,4,4))
                        #ldiv!(buff_out,mat,buff_fun)
        
                        if sqrt(buff_out[1]^2 + buff_out[2]^2 + buff_out[3]^2 + buff_out[4]^2) < tol
                            break
                        end
                        buff_start[1] = buff_start[1] - buff_out[1]
                        buff_start[2] = buff_start[2] - buff_out[2]
                        buff_start[3] = buff_start[3] - buff_out[3]
                        buff_start[4] = buff_start[4] - buff_out[4]
                    end
                    P.arr[1,i,j] = buff_start[1]
                    P.arr[2,i,j] = buff_start[2]
                    P.arr[3,i,j] = buff_start[3]
                    P.arr[4,i,j] = buff_start[4]
                end
            end
        end
    end
end


