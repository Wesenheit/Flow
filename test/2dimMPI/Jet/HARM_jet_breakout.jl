using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 2
MPI_Y = 4
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false),reorder = true)
include("../../../src/2dimMPI/Flow2D.jl")
eos = Flow2D.Polytrope(5.0/3.0)
Nx = 1024
Ny = 1024
P = Flow2D.ParVector2D{Float64}(Nx,Ny,comm)
tot_X = MPI_X * Nx
tot_Y = MPI_Y * Ny

idx,idy=  MPI.Cart_coords(comm)

floor::Float64 = 1e-8
outer::Float64 = 1e-3
box_X::Float64 = 50.
box_Y::Float64 = 100.
R_max::Float64 = 50.
R_eng::Float64 = 10.
Temp::Float64 = 0.
Rho0::Float64 = 1e-1
U0::Float64 = 1e-3
max_rho::Float64 = 10^2
angle_jet::Float64 = 0.1
dx::Float64 = 2*box_X / (tot_X)
dy::Float64 = box_Y/tot_Y

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
dt::Float64 = min(dx,dy)*0.3
T::Float64 = box_Y*10
n_it::Int64 = 20.
tol::Float64 = 1e-6
if MPI.Comm_rank(comm) == 0
    println("dt: ",dt)
end
drops::Float64 = T/100.


i = Flow2D.HARM_HLL(comm,P,MPI_X,MPI_Y,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)


MPI.Finalize()
