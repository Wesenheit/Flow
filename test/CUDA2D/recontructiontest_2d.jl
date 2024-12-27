using BenchmarkTools
using LinearAlgebra
using CUDA
include("../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 5000
Ny = 5000
P = Flow2D.ParVector2D{Float64}(Nx,Ny)
U = Flow2D.ParVector2D{Float64}(Nx,Ny)
P.arr[1,:,:] = P.arr[1,:,:,] .+ 10
P.arr[2,:,:] = P.arr[2,:,:] .+ 20
P.arr[3,:,:] = P.arr[3,:,:] .+ 11.0  
P.arr[4,:,:] = P.arr[4,:,:] .+ 5.0

nthreads = (16, 16)
numblocks = (cld(Nx, nthreads[1]), cld(Ny, nthreads[2]))


CuP = Flow2D.CuParVector2D{Float64}(P)
CuU = Flow2D.CuParVector2D{Float64}(U)
@btime begin
    CUDA.@sync @cuda threads=nthreads blocks=numblocks Flow2D.function_PtoU(CuP.arr,CuU.arr,Nx,Ny,eos.gamma)
end
