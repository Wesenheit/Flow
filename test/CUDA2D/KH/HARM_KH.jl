using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf
using MPI

MPI.Init()
comm = MPI.COMM_WORLD
MPI_X = 1
MPI_Y = 1
comm = MPI.Cart_create(comm,(MPI_X,MPI_Y), periodic=(true,true))
include("../../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope(5.0/3.0)
Nx = 200
Ny = 200
P = Flow2D.ParVector2D{Float64}(Nx,Ny)
tot_X = MPI_X * Nx
tot_Y = MPI_Y * Ny
uinf = 0.3
ratio = 0.01
U0 = 10.
Rho0 = 5.
Cs = Flow2D.SoundSpeed(Rho0,U0,eos)
KH_par = uinf/(Cs*sqrt(uinf^2+1))

idx,idy=  MPI.Cart_coords(comm)

for i in 1:Nx
    for j in 1:Ny
        P.arr[1,i+1,j+1] = Rho0
        P.arr[2,i+1,j+1] = U0
        P.arr[4,i+1,j+1] = 0.
        i_g,j_g = Flow2D.local_to_global((i,j),(idx,idy),(Nx,Ny),(MPI_X,MPI_Y))
        if j_g > div(tot_Y,2)+1
            P.arr[3,i+1,j+1] = uinf * tanh( (div(tot_Y,4)*3 - j_g) / (ratio * tot_Y))
        else
            P.arr[3,i+1,j+1] = uinf * tanh( -(div(tot_Y,4) - j_g) / (ratio * tot_Y))
        end
    end
end
P.arr[4,2:end-1,2:end-1] += randn(Nx,Ny) * uinf * 0.1

dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = min(dx,dy)*0.4
T::Float64 = 30.
n_it::Int64 = 20.
tol::Float64 = 1e-6
fps = 5
drops = 0.1
floor::Float64 = 1e-5
KH_par = uinf/(Cs*sqrt(uinf^2+1))
if MPI.Comm_rank(comm) == 0
    println("Courant/c: ",dt/min(dx,dy))
    println("T: ",T) 
    println("u/Cs: ",KH_par)
end
CuP = Flow2D.CuParVector2D{Float64}(P)


@time Flow2D.HARM_HLL(comm,CuP,MPI_X,MPI_Y,Nx,Ny,dt,dx,dy,T,eos,drops,floor,n_it,tol)


MPI.Finalize()
