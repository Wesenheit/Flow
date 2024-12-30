using BenchmarkTools
using LinearAlgebra
using KernelAbstractions
using CUDA

include("../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 4096
Ny = 4096
P = Flow2D.ParVector2D{Float64}(Nx,Ny)
U = Flow2D.ParVector2D{Float64}(Nx,Ny)
P.arr[1,:,:] = P.arr[1,:,:,] .+ 10
P.arr[2,:,:] = P.arr[2,:,:] .+ 20
P.arr[3,:,:] = P.arr[3,:,:] .+ 11.0  
P.arr[4,:,:] = P.arr[4,:,:] .+ 5.0

CuP = Flow2D.CuParVector2D{Float64}(P)
CuU = Flow2D.CuParVector2D{Float64}(U)
cuda = KernelAbstractions.get_backend(CuP.arr)
PtoU_CUDA = Flow2D.function_PtoU(cuda)
println("CUDA")
@btime begin
    PtoU_CUDA(CuP.arr,CuU.arr,eos.gamma,ndrange = (Nx,Ny))
    KernelAbstractions.synchronize(cuda)
end

cpu = KernelAbstractions.get_backend(P.arr)
PtoU_CPU = Flow2D.function_PtoU(cpu)
println("CPU")
@btime begin
    PtoU_CPU(P.arr,U.arr,eos.gamma,ndrange = (Nx,Ny))
    KernelAbstractions.synchronize(cpu)
end
