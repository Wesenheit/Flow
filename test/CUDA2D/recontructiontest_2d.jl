using BenchmarkTools
using LinearAlgebra
using KernelAbstractions
using CUDA

include("../../src/CUDA2D/Flow2D.jl")
eos = Flow2D.Polytrope(4.0/3.0)
Nx = 4096-2
Ny = 4096-2
const Tdim = 32
P = Flow2D.ParVector2D{Float64}(Nx,Ny)
U = Flow2D.ParVector2D{Float64}(Nx,Ny)
P.arr[1,:,:] = P.arr[1,:,:,] .+ 4.
P.arr[2,:,:] = P.arr[2,:,:] .+ 4.
P.arr[3,:,:] = P.arr[3,:,:] .+ 11.0  
P.arr[4,:,:] = P.arr[4,:,:] .+ 5.0

CuP = Flow2D.CuParVector2D{Float64}(P)
CuU = Flow2D.CuParVector2D{Float64}(U)
cuda = KernelAbstractions.get_backend(CuP.arr)
PtoU_CUDA = Flow2D.function_PtoU(cuda,(Tdim,Tdim))
println("CUDA")
@btime begin
    PtoU_CUDA(CuP.arr,CuU.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
    KernelAbstractions.synchronize(cuda)
end

Pprim = deepcopy(P)
P.arr = P.arr + randn(4,Nx+2,Ny+2)/10
CuP = Flow2D.CuParVector2D{Float64}(P)

n_it = 30
eps = 1e-5

UtoP_CUDA = Flow2D.function_UtoP(cuda)
println("CUDA")
@time begin
    UtoP_CUDA(CuU.arr,CuP.arr,eos.gamma,n_it,eps,ndrange = (P.size_X,P.size_Y))
    KernelAbstractions.synchronize(cuda)
end
println("CUDA accuracy")
P_from_cuda = Flow2D.ParVector2D{Float64}(CuP)
println(mean(abs.(P_from_cuda.arr[:,:,1] .- Pprim.arr[:,:,1])))
println(mean(abs.(P_from_cuda.arr[:,:,2] .- Pprim.arr[:,:,2])))
println(mean(abs.(P_from_cuda.arr[:,:,3] .- Pprim.arr[:,:,3])))
println(mean(abs.(P_from_cuda.arr[:,:,4] .- Pprim.arr[:,:,4])))

