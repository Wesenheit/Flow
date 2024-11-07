using BenchmarkTools
using CairoMakie
include("../../src/1dim/Flow1D.jl")
eos = Flow1D.Polytrope(5.0/3.0)
N = 50
P = Flow1D.ParVector1D{Float64,N}()
P.arr1 .= 0.2
P.arr2 .= 0.2
for i in 1:div(N,2)
    P.arr1[i] = 10.
    P.arr2[i] = 13.33/(eos.gamma-1)
end
for i in div(N,2):N
    P.arr1[i] = 1.
    P.arr2[i] = 0.01/(eos.gamma-1)
end
#P.arr1[div(N,10)*4:div(N,10)*6] .=  1.0
#P.arr1[div(N,10)*4:div(N,10)*6] .=  0.5

#P.arr3[div(N,10)*4:div(N,10)*6] = P.arr3[div(N,10)*4:div(N,10)*6]  .+ 0.4
X = LinRange(-0.5,0.5,N) |> collect

dx::Float64 = X[2]-X[1]
dt::Float64 = 0.00003
println("velocity: ",dx/dt)
T::Float64 = 0.4
n_it::Int64 = 10.
tol::Float64 = 1e-6
drops::Float64 = T/10.
out = Flow1D.LaxFriedrich(P,N,dt,dx,T,eos,drops,n_it,tol)


X = LinRange(-0.5,0.5,N) |> collect
f = Figure()
ax = Axis(f[1, 1])
println(length(out))
for i in 1:length(out)
    lines!(ax, X, out[i].arr1 |> collect)
end
save("LaxFrid.pdf",f)