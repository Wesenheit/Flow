using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

Type = Float32

@assert MPI.has_cuda()
MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false),reorder = true)
include("../../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope{Type}(5.0/3.0)
Nx = 2048-4
Ny = 2048-4
P = Flow2D.ParVector2D{Type}(Nx,Ny)
tot_X = MPI_X * Nx
tot_Y = MPI_Y * Ny

idx,idy=  MPI.Cart_coords(comm)




floor::Type = 1e-8
outer::Type = 1e-3
box_X::Type = 2*50.
box_Y::Type = 2*100.
dx::Type = 2*box_X / (tot_X)
dy::Type = box_Y/tot_Y


R_core::Type = 5.
R_outer::Type = 20.
rho_core::Type = 10
rho_outer::Type = 1
v_core::Type = -0.07
v_outer::Type = -0.02
U0::Type = 1e-2

for i in 1:P.size_X
    for j in 1:P.size_Y
        i_g = Flow2D.local_to_global(i,idx,P.size_X,MPI_X)
        j_g = Flow2D.local_to_global(j,idy,P.size_Y,MPI_Y)
        if i_g == 0 || j_g == 0 
            continue
        end

        X = i_g * dx - box_X
        Y = j_g * dy - box_X
        
        R = sqrt(X^2+Y^2)
        

        if R < R_core
            rho = rho_core 
        elseif R > R_core && R < R_outer
            rho = rho_outer
        else
            rho = outer
        end
        
	    P.arr[1, i, j] = rho * (1 + randn() * 1e-1)
	    P.arr[2, i, j] = U0 * (1 + randn() * 1e-1)	

        angle = atan(Y,X)
        polar_factor = abs(sin(angle))  
        if R < R_core
            v = v_core
        elseif R > R_core && R < R_outer
            v = v_outer + 0.2*v_outer * polar_factor
        else
            v = 0
        end

        gamma = 1/sqrt(1-v^2)
        vx = cos(angle)*v
        vy = sin(angle)*v
        ux = gamma * vx
        uy = gamma * vy
        P.arr[3,i,j] = ux * (1 + randn() * 1e-1)
        P.arr[4,i,j] = uy * (1 + randn() * 1e-1)
    end
end

dt::Type = min(dx,dy)*0.3
T::Type = box_Y*10
n_it::Int64 = 20.
tol::Type = 1e-6
if MPI.Comm_rank(comm) == 0
    println("dt: ",dt)
end
drops::Type = T/100.

CuP = Flow2D.CuParVector2D{Type}(P)
@time Flow2D.HARM_HLL(comm,CuP,MPI_X,MPI_Y,Nx,Ny,dt,dx,dy,T,eos,drops,floor,ARGS[1],n_it,tol)


MPI.Finalize()
