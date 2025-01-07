using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

T = Float32

@assert MPI.has_cuda()
MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false),reorder = true)
include("../../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope{T}(5.0/3.0)
Nx = 8192-2
Ny = 8192-2
P = Flow2D.ParVector2D{T}(Nx,Ny)
tot_X = MPI_X * Nx
tot_Y = MPI_Y * Ny

idx,idy=  MPI.Cart_coords(comm)

floor::T = 1e-8
outer::T = 1e-3
box_X::T = 50.
box_Y::T = 100.
R_max::T = 50.
R_eng::T = 10.
Temp::T = 0.
Rho0::T = 1e-1
U0::T = 1e-3
max_rho::T = 10^2
angle_jet::T = 0.1
dx::T = 2*box_X / (tot_X)
dy::T = box_Y/tot_Y

for i in 1:Nx
    for j in 1:Ny
        i_g,j_g = Flow2D.local_to_global((i,j),(idx,idy),(Nx,Ny),(MPI_X,MPI_Y))
        X = i_g * dx - box_X
        Y = j_g * dy + R_eng * cos(angle_jet)*0.75
        R = sqrt(X^2+Y^2)
        if R < R_max
            rho = Rho0 * min((R_max/R+1e-4)^2,Rho0*10)
        else
            rho = outer
        end
        P.arr[1,i+1,j+1] = rho*(1 + randn()*5e-2)
        P.arr[2,i+1,j+1] = U0*(1+randn()*5e-2) #rho * Temp/(eos.gamma - 1)

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
        P.arr[3,i+1,j+1] = ux
        P.arr[4,i+1,j+1] = uy
    end
end
dt::T = min(dx,dy)*0.3
T::T = box_Y*10
n_it::Int64 = 20.
tol::T = 1e-6
if MPI.Comm_rank(comm) == 0
    println("dt: ",dt)
end
drops::T = T/100.

CuP = Flow2D.CuParVector2D{T}(P)
@time Flow2D.HARM_HLL(comm,CuP,MPI_X,MPI_Y,Nx,Ny,dt,dx,dy,T,eos,drops,floor,ARGS[1],n_it,tol)


MPI.Finalize()
