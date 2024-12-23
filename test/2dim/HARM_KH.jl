using BenchmarkTools
using CairoMakie
using ThreadPinning
using Profile
using Printf

include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(5.0/3.0)
Nx = 1000
Ny = 1000
P = Flow2D.ParVector2D{Float64}(Nx,Ny)

uinf = 0.3
ratio = 0.1
U0 = 10.
Rho0 = 5.
Cs = Flow2D.SoundSpeed(Rho0,U0,eos)
println("u/Cs: ",uinf/(Cs*sqrt(uinf^2+1)))
KH_par = uinf/(Cs*sqrt(uinf^2+1))

for i in 1:Nx
    for j in 1:Ny
        P.arr[1,i,j] = Rho0
        P.arr[2,i,j] = U0
        P.arr[4,i,j] = 0.

        if j > div(Ny,2)+1
            P.arr[3,i,j] = uinf * tanh( (div(Ny,4)*3 - j) / (ratio * Ny))
        else
            P.arr[3,i,j] = uinf * tanh( -(div(Ny,4) - j) / (ratio * Ny))
        end

    end
end
P.arr[4,:,:] += randn(Nx,Ny) * uinf * 0.01

dx::Float64 = 1/Nx
dy::Float64 = 1/Ny
dt::Float64 = min(dx,dy)*0.4
println("Courant/c: ",dt/min(dx,dy))
T::Float64 = 1.#4 *sqrt(uinf^2 + 1)/uinf
println("T: ",T) 
n_it::Int64 = 10.
tol::Float64 = 1e-4
fps = 10
drops = T/10.
#drops::Float64 = T/(20*fps)
floor::Float64 = 1e-4

pinthreads(:cores)
threadinfo(;)

t1 = time()
@time out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)
elapsed_time = time() - t1;
println("Elapsed time: ", elapsed_time, " seconds");

"""
using Plots
using CairoMakie
min_val = 1000
max_val = 0
for i in 1:length(out)
    if maximum(out[i].arr[1,:,:]) > max_val
        global max_val = maximum(out[i].arr[1,:,:])
    end
    if minimum(out[i].arr[1,:,:]) < min_val
        global min_val = minimum(out[i].arr[1,:,:])
    end
end

anim = @animate for i in 1:length(out)
    data = out[i].arr[1,:, :]
    
    p = Plots.heatmap(data, xlabel="x", ylabel="y", color=:viridis, 
                      clims=(min_val, max_val), size=(600, 600))
end

gif(anim, @sprintf("KH_%.2f.gif",KH_par), fps=fps)
"""


