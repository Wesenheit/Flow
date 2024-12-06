using BenchmarkTools
using CairoMakie
using ThreadPinning
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 200
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
dt::Float64 = dx*0.1
println("Courant/c: ",dt/dx)
T::Float64 = 0.2
n_it::Int64 = 10.
tol::Float64 = 1e-6
drops::Float64 = T/3.
floor::Float64 = 1e-4

out = Flow2D.HARM_HLL(P,Nx,Ny,dt,dx,dy,T,eos,drops,Flow2D.minmod,floor,n_it,tol)

X = LinRange(-0.5,0.5,Nx) |> collect
f = Figure()
ax_rho = Axis(f[1, 1],xlabel = L"$X$", ylabel = L"$\rho$")
ax_P = Axis(f[1, 2],xlabel = L"$X$",ylabel = L"$p$")
ax_gamm =  Axis(f[2, 1],xlabel = L"$X$",ylabel = L"$\Gamma$")
ax_vel = Axis(f[2,2],xlabel = L"$X$",ylabel = L"$v_X$")

for i in 1:length(out)
    lines!(ax_rho, X, out[i].arr[:,div(Ny,2),1] |> collect,color = "black")
    lines!(ax_P, X, (eos.gamma -1) * out[i].arr[:,div(Ny,2),2] |> collect,color = "black")
    gamma = sqrt.(  out[i].arr[:,div(Ny,3),3].^2 .+ 1)
    lines!(ax_gamm, X, gamma,color = "black")
    lines!(ax_vel, X,  out[i].arr[:,div(Ny,2),3] ./ gamma,color = "black")
end

save("tube_test.pdf",f)