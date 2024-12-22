using BenchmarkTools
using LinearAlgebra
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 2000
Ny = 2000
P = Flow2D.ParVector2D{Float64,Nx,Ny}()
U = Flow2D.ParVector2D{Float64,Nx,Ny}()
P.arr[1,:,:] = P.arr[1,:,:,] .+ 10
P.arr[2,:,:] = P.arr[2,:,:] .+ 20
P.arr[3,:,:] = P.arr[3,:,:] .+ 11.0  
P.arr[4,:,:] = P.arr[4,:,:] .+ 5.0
P.arr[:,:,:] += randn(4,Nx,Ny)/10
buff= Flow2D.BufferStruct{Float64}()

@btime Flow2D.PtoU(P,U,eos,buff)
Pprim = deepcopy(P)
P.arr = P.arr .+ randn(4,Nx,Ny)

@btime @inbounds Flow2D.UtoP(U,P,eos,buff,2,1e-5)

println(mean(abs.(P.arr[:,:,1] .- Pprim.arr[:,:,1])))
println(mean(abs.(P.arr[:,:,2] .- Pprim.arr[:,:,2])))
println(mean(abs.(P.arr[:,:,3] .- Pprim.arr[:,:,3])))
println(mean(abs.(P.arr[:,:,4] .- Pprim.arr[:,:,4])))
