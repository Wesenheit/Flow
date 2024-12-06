using BenchmarkTools
using CairoMakie
using ThreadPinning
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 100
Ny = 5
P = Flow2D.ParVector2D{Float64,Nx,Ny}()

for i in 1:Nx
    for j in 1:Ny

        P.arr[i,j,4] = 0.
        P.arr[i,j,3] = 0.

        #if j < div(Ny,3)*2 && j > div(Ny,3)
        if i < div(Nx,3)*2 && i > div(Nx,3)
            P.arr[i,j,1] = 10.
            P.arr[i,j,2] = 10^4/(eos.gamma - 1)
        else
            P.arr[i,j,1] = 1.
            P.arr[i,j,2] = 0.1/(eos.gamma - 1)
        end
    end
end
#P.arr[:,:,1] += rand(Nx,Ny)*0.1
dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = dx/100
println("Courant/c: ",dt/dx)
T::Float64 = 0.25
n_it::Int64 = 10.
tol::Float64 = 1e-6
drops::Float64 = T/20.


out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.MC,n_it,tol)

f = Figure()

image(f[1, 1], out[end].arr[:,:,1],
    axis = (title = "Default",))

save("tube_test.pdf",f)