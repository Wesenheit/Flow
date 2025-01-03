using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using TimerOutputs
using KernelAbstractions
using CUDA
using MPI
using HDF5
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

function local_to_global(local_coords, proc_coords, local_dims, grid_dims)
    # Unpack parameters
    i, j = local_coords
    px, py = proc_coords
    Nx, Ny = local_dims
    n_x, n_y = grid_dims

    # Calculate global coordinates
    global_i = px * Nx + i
    global_j = py * Ny + j

    return (global_i, global_j)
end


abstract type FlowArr{T} end


mutable struct ParVector2D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    function ParVector2D{T}(Nx,Ny) where {T}
        arr = zeros(4,Nx+2,Ny+2)
        new(arr,Nx+2,Ny+2)
    end
    function ParVector2D{T}(arr::FlowArr{T}) where {T}
        new(Array{T}(arr.arr),arr.size_X,arr.size_Y)
    end
end

mutable struct CuParVector2D{T <:Real} <: FlowArr{T}
    # Parameter Vector
    arr::CuArray{T}
    size_X::Int64
    size_Y::Int64
    function CuParVector2D{T}(arr::FlowArr{T}) where {T}
        new(CuArray{T}(arr.arr),arr.size_X,arr.size_Y)
    end

    function CuParVector2D{T}(Nx::Int64,Ny::Int64) where {T}
        new(CuArray{T}(zeros(4,Nx+2,Ny+2)),Nx+2,Ny+2)
    end
end

function VectorLike(X::FlowArr{T}) where T
    if typeof(X.arr) <: CuArray
        return CuParVector2D{T}(X.size_X-2,X.size_Y-2)
    else
        return ParVector2D{T}(X.size_X-2,X.size_Y-2)
    end
end

@kernel function function_PtoU(@Const(P::AbstractArray), U::AbstractArray,gamma::Float64)
    i, j = @index(Global, NTuple)
    gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
    w = gamma * P[2,i,j] + P[1,i,j] 
    U[1,i,j] = gam * P[1,i,j]
    U[2,i,j] = (gamma-1) * P[2,i,j] - gam^2 * w
    U[3,i,j] = P[3,i,j] * gam * w
    U[4,i,j] = P[4,i,j] * gam * w
end

@kernel function function_PtoFx(@Const(P::AbstractArray), Fx::AbstractArray,gamma::Float64)
    i, j = @index(Global, NTuple)
    gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
    w = gamma * P[2,i,j] + P[1,i,j] 

    Fx[1,i,j] = P[1,i,j]*P[3,i,j]
    Fx[2,i,j] = - w *P[3,i,j] * gam
    Fx[3,i,j] = P[3,i,j]^2 * w + (gamma - 1) * P[2,i,j]
    Fx[4,i,j] = P[3,i,j] * P[4,i,j] * w 
end


@kernel function function_PtoFy(@Const(P::AbstractArray), Fy::AbstractArray,gamma::Float64)
    i, j = @index(Global, NTuple)
    gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
    w = gamma * P[2,i,j] + P[1,i,j] 
    Fy[1,i,j] = P[1,i,j] * P[4,i,j]
    Fy[2,i,j] = - w *P[4,i,j] * gam
    Fy[3,i,j] = P[3,i,j] * P[4,i,j] * w 
    Fy[4,i,j] = P[4,i,j]^2 * w + (gamma - 1) * P[2,i,j]
end


@inline function LU_dec!(flat_matrix::AbstractVector{T}, target::AbstractVector{T}, x::AbstractVector{T}) where T

    @inline function index(i, j)
        return (j - 1) * 4 + i
    end

    for k in 1:4
        for i in k+1:4
            flat_matrix[index(i, k)] /= flat_matrix[index(k, k)]
            for j in k+1:4
                flat_matrix[index(i, j)] -= flat_matrix[index(i, k)] * flat_matrix[index(k, j)]
            end
        end
    end

    # Forward substitution to solve L*y = target (reusing x for y)
    for i in 1:4
        x[i] = target[i]
        for j in 1:i-1
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
    end

    # Backward substitution to solve U*x = y
    for i in 4:-1:1
        for j in i+1:4
            x[i] -= flat_matrix[index(i, j)] * x[j]
        end
        x[i] /= flat_matrix[index(i, i)]
    end
end

@kernel function function_UtoP(@Const(U::AbstractArray), P::AbstractArray,gamma::Float64,n_iter::Int64,tol::Float64=1e-10)
    buff_out = @private eltype(P) 4
    buff_fun = @private eltype(P) 4
    buff_jac = @private eltype(P) 16
    i, j = @index(Global, NTuple)

    for _ in 1:n_iter
        gam = sqrt(P[3,i,j]^2 + P[4,i,j]^2 + 1)
        w = gamma * P[2,i,j] + P[1,i,j] 
        buff_fun[1] = gam * P[1,i,j] - U[1,i,j]
        buff_fun[2] = (gamma-1) * P[2,i,j] - gam^2 * w - U[2,i,j]
        buff_fun[3] = P[3,i,j] * gam * w - U[3,i,j]
        buff_fun[4] = P[4,i,j] * gam * w - U[4,i,j]

        buff_jac[1] = gam
        buff_jac[5] = 0
        buff_jac[9] = P[1,i,j] * P[3,i,j] / gam
        buff_jac[13] = P[1,i,j] * P[4,i,j] / gam
    
        buff_jac[2] = - gam^2
        buff_jac[6] = (gamma - 1) - gam ^2 * gamma
        buff_jac[10] = -2 * P[3,i,j] * w
        buff_jac[14] = -2 * P[4,i,j] * w
    
        buff_jac[3] = gam * P[3,i,j]
        buff_jac[7] = gam * P[3,i,j] * gamma
        buff_jac[11] = (2 * P[3,i,j]^2 + P[4,i,j]^2 + 1) / gam * w
        buff_jac[15] = P[3,i,j] * P[4,i,j] / gam * w
    
        buff_jac[4] = gam * P[4,i,j]
        buff_jac[8] = gam * P[4,i,j] * gamma
        buff_jac[12] = P[3,i,j] * P[4,i,j] / gam * w
        buff_jac[16] = (2 * P[4,i,j]^2 + P[3,i,j] ^ 2 + 1) / gam * w                        
        LU_dec!(buff_jac,buff_fun,buff_out)

        if sqrt(buff_out[1]^2 + buff_out[2]^2 + buff_out[3]^2 + buff_out[4]^2) < tol
            break
        end
        P[1,i,j] = P[1,i,j] - buff_out[1]
        P[2,i,j] = P[2,i,j] - buff_out[2]
        P[3,i,j] = P[3,i,j] - buff_out[3]
        P[4,i,j] = P[4,i,j] - buff_out[4]
    end
end