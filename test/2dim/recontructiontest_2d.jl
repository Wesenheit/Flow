using BenchmarkTools
using LinearAlgebra
using ThreadPinning
#pinthreads(:cores)

typ = Float64
include("../../src/2dim/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 4096
Ny = 4096
P = Flow2D.ParVector2D{typ}(Nx,Ny)
U = Flow2D.ParVector2D{typ}(Nx,Ny)
P.arr[1,:,:] = P.arr[1,:,:,] .+ 10
P.arr[2,:,:] = P.arr[2,:,:] .+ 20
P.arr[3,:,:] = P.arr[3,:,:] .+ 11.0  
P.arr[4,:,:] = P.arr[4,:,:] .+ 5.0

@btime @inbounds Flow2D.PtoU(P,U,eos)
Pprim = deepcopy(P)
P.arr = P.arr + randn(typ,4,Nx,Ny)/10

@btime @inbounds Flow2D.UtoP(U,P,eos,20,1e-5)

println(mean(abs.(P.arr[:,:,1] .- Pprim.arr[:,:,1])))
println(mean(abs.(P.arr[:,:,2] .- Pprim.arr[:,:,2])))
println(mean(abs.(P.arr[:,:,3] .- Pprim.arr[:,:,3])))
println(mean(abs.(P.arr[:,:,4] .- Pprim.arr[:,:,4])))
