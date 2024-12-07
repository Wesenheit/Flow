using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile

include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(5.0/3.0)
Nx = 400
Ny = 400
P = Flow2D.ParVector2D{Float64,Nx,Ny}()

uinf = 0.4
ratio = 0.01
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
P.arr[:,:,4] += randn(Nx,Ny) * 0.01

dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = min(dx,dy)*0.4
println("Courant/c: ",dt/min(dx,dy))
T::Float64 = 10 *sqrt(uinf^2 + 1)/uinf
println("T: ",T) 
n_it::Int64 = 10.
tol::Float64 = 1e-4
drops::Float64 = T/10.
floor::Float64 = 1e-4

pinthreads(:cores)
threadinfo(;)
#Profile.init()
out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)
#out = @profile Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)
#Profile.print(maxdepth = 2 ) 

f = Figure()

image(f[1, 1], out[end].arr[:,:,1],
    axis = (title = "Kelvin-Helmholtz",))

save("KH_test.pdf",f)
