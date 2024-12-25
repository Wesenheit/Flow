using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 2
MPI_Y = 2
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(false,false))
include("../../../src/2dimMPI/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 400
Ny = 400
P = Flow2D.ParVector2D{Float64}(Nx,Ny,comm)
tot_X = MPI_X * Nx
tot_Y = MPI_Y * Ny

idx,idy=  MPI.Cart_coords(comm)

floor::Float64 = 1e-5
box::Float64 = 100.
R_max::Float64 = 10.
R_eng::Float64 = 0.4
Temp::Float64 = 1
Rho0::Float64 = 0.1
max_rho::Float64 = 10^2
angle_jet::Float64 = 0.1

for i in 1:Nx
    for j in 1:Ny
        i_g,j_g = Flow2D.local_to_global((i,j),(idx,idy),(Nx,Ny),(MPI_X,MPI_Y))
        X = (i_g/tot_X - 1/2)*box
        Y = j_g/tot_Y*box
        R = sqrt(X^2+Y^2)
        if R < R_max
            rho = Rho0 * (R_max/R+1e-4)^2
        else
            rho = floor
        end
        P.arr[1,i+1,j+1] = rho
        P.arr[2,i+1,j+1] = rho * Temp/(eos.gamma - 1)

        angle = atan(Y,X)
        if R < R_eng && abs(angle) < angle_jet
            V = 0.3*R_eng
            gamma = 1/sqrt(1-V^2)
            vx = sin(angle)*v
            vy = cos(angle)*v
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
dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = min(dx,dy)*0.4
T::Float64 = box
n_it::Int64 = 20.
tol::Float64 = 1e-6
drops::Float64 = 1.


i = Flow2D.HARM_HLL(comm,P,MPI_X,MPI_Y,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)


MPI.Finalize()
