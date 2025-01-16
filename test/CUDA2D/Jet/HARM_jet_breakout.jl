using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

Type = Float64

@assert MPI.has_cuda()
MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false),reorder = true)
include("../../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope{Type}(5.0/3.0)
Nx = 8192 - 4
Ny = 8192 - 4
P = Flow2D.ParVector2D{Type}(Nx,Ny)
tot_X = MPI_X * Nx + 4
tot_Y = MPI_Y * Ny + 4

idx,idy=  MPI.Cart_coords(comm)

floor::Type = 1e-8
outer::Type = 1e-3
box_X::Type = 50.
box_Y::Type = 100.
R_max::Type = 50.
R_eng::Type = 10.
Temp::Type = 0.
Rho0::Type = 1e-1
U0::Type = 1e-3
max_rho::Type = 10^2
angle_jet::Type = 0.1
dx::Type = 2*box_X / (tot_X)
dy::Type = box_Y/tot_Y

for i in 1:P.size_X
    for j in 1:P.size_Y
        i_g = Flow2D.local_to_global(i,idx,P.size_X,MPI_X)
        j_g = Flow2D.local_to_global(j,idy,P.size_Y,MPI_Y)
        if i_g == 0 || j_g == 0 
            continue
        end
        X = i_g * dx - box_X
        Y = j_g * dy + R_eng * cos(angle_jet)*0.75
        R = sqrt(X^2+Y^2)
        if R < R_max
            rho = Rho0 * min((R_max/R+1e-4)^2,Rho0*10)
        else
            rho = outer
        end
        P.arr[1,i,j] = rho*(1 + randn()*5e-2)
        P.arr[2,i,j] = U0*(1+randn()*5e-2) #rho * Temp/(eos.gamma - 1)

        angle = atan(Y,X)
        if R < R_eng && abs(angle-pi/2) < angle_jet
            v = 0.3*R/R_eng
            gamma = 1/sqrt(1-v^2)
            vx = cos(angle)*v
            vy = sin(angle)*v
            ux = gamma * vx
            uy = gamma * vy
        else
            ux = 0
            uy = 0
        end
        P.arr[3,i,j] = ux
        P.arr[4,i,j] = uy
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
