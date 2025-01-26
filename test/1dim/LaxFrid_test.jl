using BenchmarkTools
using CairoMakie
include("../../src/1dim/Flow1D.jl")
eos = Flow1D.Polytrope(5.0/3.0)
N = 512
P = Flow1D.ParVector1D{Float64,N}()
for i in 1:div(N,2)
    P.arr1[i] = 1.
    P.arr2[i] = 10^4/(eos.gamma-1)
end
for i in div(N,2):N
    P.arr1[i] = 0.1
    P.arr2[i] = 10.0/(eos.gamma-1)
end


X = LinRange(-0.5,0.5,N) |> collect

dx::Float64 = X[2]-X[1]
dt::Float64 = 0.4*dx
println("velocity: ",dx/dt)
T::Float64 = 0.5
n_it::Int64 = 30
tol::Float64 = 1e-6
drops::Float64 = T/3.
out = Flow1D.LaxFriedrich(P,N,dt,dx,T,eos,drops,n_it,tol)


X = LinRange(-0.5,0.5,N) |> collect
f = Figure()
ax = Axis(f[1, 1],title = L"$\rho$")
println(length(out))
for i in 1:length(out)
    lines!(ax, X, out[i].arr1 |> collect,label = "T = " * string(round((i-1)*drops,sigdigits = 2)))
end
f[1, 2] = Legend(f, ax, framevisible = false)
save("LaxFrid.pdf",f)
