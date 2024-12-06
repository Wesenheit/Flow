using BenchmarkTools
using CairoMakie
using ThreadPinning
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 400
Ny = 400
P = Flow2D.ParVector2D{Float64,Nx,Ny}()

uinf = 0.3
ratio = 0.5
for i in 1:Nx
    for j in 1:Ny
        P.arr[i,j,1] = 0.1
        P.arr[i,j,2] = 0.1
        P.arr[i,j,4] = 0.

        if j > div(Ny,2)+1
            P.arr[i,j,3] = uinf * tanh( (div(Ny,4)*3 - j) / (ratio * Ny))
        else
            P.arr[i,j,3] = uinf * tanh( -(div(Ny,4) - j) / (ratio * Ny))
        end

    end
end
#P.arr[:,:,4] += randn(Nx,Ny) * 0.001

dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = dx*1e-2
println("Courant/c: ",dt/dx)
T::Float64 = 1.
n_it::Int64 = 10.
tol::Float64 = 1e-6
drops::Float64 = T/20.
floor::Float64 = 1e-4

out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)

f = Figure()

image(f[1, 1], out[end].arr[:,:,3],
    axis = (title = "Default",))

save("KH_test.pdf",f)