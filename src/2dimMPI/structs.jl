using LinearAlgebra
using StaticArrays
using Base.Threads
using Base.Iterators
using TimerOutputs
using MPI
using HDF5
BLAS.set_num_threads(1)

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


mutable struct ParVector2D{T <:Real}
    # Parameter Vector
    arr::Array{T,3}
    size_X::Int64
    size_Y::Int64
    part
    function ParVector2D{T}(Nx,Ny,comm) where {T}
        arr = ones(4,Nx+2,Ny+2)
        arr[:,1,1] = [1,1,0,0]
        arr[:,end,1] = [1,1,0,0]
        arr[:,1,end] = [1,1,0,0]
        arr[:,end,end] = [1,1,0,0]
        partitions = partition(2:Ny+1,div(Ny+1,nthreads()))
        new(arr, Nx+2, Ny+2, partitions)
    end
end


function SyncBoundaryX(U::ParVector2D,comm)
    right = U.arr[:,end-1,:]
    left = U.arr[:,2,:]
        
    rightp = U.arr[:,end,:]
    leftp = U.arr[:,1,:]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,0,1)
    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,0,-1)

    MPI.Sendrecv!(right,rank_source_right,0,leftp,rank_dest_right,0,comm)

    MPI.Sendrecv!(left,rank_source_left,1,rightp,rank_dest_left,1,comm)

    U.arr[:,end,:] = rightp
    U.arr[:,1,:] = leftp
end

function SyncFlux_X_Left(PL::ParVector2D,comm)
    #we send the left flux to the right boundary
    mess = PL.arr[:,end-1,:]
        
    buff = PL.arr[:,1,:]

    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,0,-1)

    MPI.Sendrecv!(mess,rank_source_left,1,buff,rank_dest_left,1,comm)

    PL.arr[:,1,:] = buff
end

function SyncFlux_X_Right(PR::ParVector2D,comm)
    #we send the right flux to the left boundary
    mess = PR.arr[:,1,:]
        
    buff = PR.arr[:,end-1,:]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,0,1)

    MPI.Sendrecv!(mess,rank_source_right,0,buff,rank_dest_right,0,comm)

    PR.arr[:,end-1,:] = buff
end

function SyncFlux_Y_Down(PD::ParVector2D,comm)
    #we send the left flux to the right boundary
    mess = PD.arr[:,:,end-1]
        
    buff = PD.arr[:,:,1]

    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,1,-1)

    MPI.Sendrecv!(mess,rank_source_left,1,buff,rank_dest_left,1,comm)

    PD.arr[:,:,1] = buff
end

function SyncFlux_Y_Up(PU::ParVector2D,comm)
    #we send the right flux to the left boundary
    mess = PU.arr[:,:,1]
        
    buff = PU.arr[:,:,end-1]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,1,1)

    MPI.Sendrecv!(mess,rank_source_right,0,buff,rank_dest_right,0,comm)

    PU.arr[:,:,end-1] = buff
end



function SyncBoundaryY(U::ParVector2D,comm)
    up = U.arr[:,:,end-1]
    down = U.arr[:,:,2]
        
    upp = U.arr[:,:,end]
    downp = U.arr[:,:,1]

    rank_source_up,rank_dest_up = MPI.Cart_shift(comm,1,1)
    rank_source_down,rank_dest_down = MPI.Cart_shift(comm,1,-1)


    MPI.Sendrecv!(up,rank_source_up,0,downp,rank_dest_up,0,comm)

    MPI.Sendrecv!(down,rank_source_down,1,upp,rank_dest_down,1,comm)

    U.arr[:,:,end] = upp
    U.arr[:,:,1] = downp
    
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

function function_PtoU(x::AbstractVector, buffer::AbstractVector,eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = gam * x[1]
    buffer[2] = (eos.gamma-1) * x[2] - gam^2 * w
    buffer[3] = x[3] * gam * w
    buffer[4] = x[4] * gam * w
end

function function_PtoFx(x::AbstractVector, buffer::AbstractVector,eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = x[1]*x[3]
    buffer[2] = - w *x[3] * gam
    buffer[3] = x[3]^2 * w + (eos.gamma - 1) * x[2]
    buffer[4] = x[3] * x[4] * w 
end

function function_PtoFy(x::AbstractVector, buffer::AbstractVector,eos::Polytrope)
    gam::Float64 = sqrt(x[3]^2 + x[4]^2 + 1)
    w::Float64 = eos.gamma * x[2] + x[1] 
    buffer[1] = x[1]*x[4]
    buffer[2] = - w *x[4] * gam
    buffer[3] = x[3] * x[4] * w 
    buffer[4] = x[4]^2 * w + (eos.gamma - 1) * x[2]
end

function PtoFx(P::ParVector2D,Fx::ParVector2D,eos::EOS)
    @threads :static for j in 2:P.size_Y-1
        for i in 2:P.size_X-1
            bufferp = @view P.arr[:,i,j]
            buffer = @view Fx.arr[:,i,j]
            function_PtoFx(bufferp,buffer,eos)
        end
    end
end

function PtoFy(P::ParVector2D,Fy::ParVector2D,eos::EOS)
    @threads :static for j in 2:P.size_Y-1
        for i in 2:P.size_X-1
            bufferp = @view P.arr[:,i,j]
            buffer = @view Fy.arr[:,i,j]
            function_PtoFy(bufferp,buffer,eos)
        end
    end
end


function PtoU(P::ParVector2D,U::ParVector2D,eos::EOS)
    @threads :static for j in 1:P.size_Y
        for i in 1:P.size_X
            bufferp = @view P.arr[:,i,j]
            buffer = @view U.arr[:,i,j]
            function_PtoU(bufferp,buffer,eos)
        end
    end
end



function LU_dec!(flat_matrix::MVector{16,Float64}, target::MVector{4,Float64}, x::MVector{4,Float64})

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

function UtoP(U::ParVector2D,P::ParVector2D,eos::EOS,n_iter::Int64,tol::Float64=1e-10)
    @sync for (id_th,chunk) in enumerate(P.part)
        @spawn begin
            buff_start::MVector{4,Float64} = @MVector zeros(4)
            buff_out::MVector{4,Float64} = @MVector zeros(4)
            buff_fun::MVector{4,Float64} = @MVector zeros(4)
            buff_jac::MVector{16,Float64} = @MVector zeros(16)
            for j in chunk
                for i in 2:P.size_X-1
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
