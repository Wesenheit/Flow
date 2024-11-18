using BenchmarkTools
using CairoMakie
include("../../src/1dim/Flow1D.jl")
eos = Flow1D.Polytrope(4.0/3.0)
N = 10000
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
dt::Float64 = 0.000003
println("Courant/c: ",dt/dx)
T::Float64 = 0.5
n_it::Int64 = 10.
tol::Float64 = 1e-6
drops::Float64 = T/3.
out = Flow1D.HARM_HLL(P,N,dt,dx,T,eos,drops,Flow1D.MC,n_it,tol)


X = LinRange(-0.5,0.5,N) |> collect
f = Figure()
ax_rho = Axis(f[1, 1],xlabel = L"$X$", ylabel = L"$\rho$")
ax_P = Axis(f[1, 2],xlabel = L"$X$",ylabel = L"$p$")
ax_gamm =  Axis(f[2, 1],xlabel = L"$X$",ylabel = L"$\Gamma$")
ax_vel = Axis(f[2,2],xlabel = L"$X$",ylabel = L"$v_X$")
for i in 1:length(out)
    lines!(ax_rho, X, out[i].arr1 |> collect,color = "black")
    lines!(ax_P, X, (eos.gamma -1) * out[i].arr2 |> collect,color = "black")
    gamma = sqrt.( out[i].arr3 .^2 .+ 1)
    lines!(ax_gamm, X, gamma,color = "black")
    lines!(ax_vel, X, out[i].arr3 ./ gamma,color = "black")
end
save("HARM_HLL.pdf",f)
