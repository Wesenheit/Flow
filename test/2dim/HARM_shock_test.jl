using BenchmarkTools
using CairoMakie
using ThreadPinning
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(5.0/3.0)
Nx = 100
Ny = 100
P = Flow2D.ParVector2D{Float64,Nx,Ny}()

X_arr = LinRange(-1,1,Nx) |> collect
Y_arr = LinRange(-1,1,Ny) |> collect

for i in 1:Nx
    for j in 1:Ny
        X = X_arr[i]
        Y = Y_arr[j]
        P.arr[1,i,j] = 0.4
        P.arr[2,i,j] = 0.6
        if X < 0.1
            P.arr[3,i,j] = (Y-1)* (Y+1)    
        end        
    end
end
P.arr[4,:,:] += randn(Nx,Ny) * 0.001

dx::Float64 = 2/Nx
dy::Float64 = 2/Ny
dt::Float64 = dx*0.4
println("Courant/c: ",dt/dx)
T::Float64 = 1.
n_it::Int64 = 40.
tol::Float64 = 1e-8
drops::Float64 = T/20.
floor::Float64 = 1e-4

pinthreads(:cores)
threadinfo(;)


out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)

f = Figure()

image(f[1, 1], out[end].arr[:,:,3],
    axis = (title = "Test shock",))

save("shock_test.pdf",f)