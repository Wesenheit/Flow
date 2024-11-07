using BenchmarkTools
include("../../src/1dim/Flow1D.jl")
eos = Flow1D.Polytrope(4.0/3.0)
N = 10
P = Flow1D.ParVector1D{Float64,N}()
U = Flow1D.ParVector1D{Float64,N}()
P.arr1 = P.arr1 .+ 0.1
P.arr2 = P.arr2 .+ 0.2
P.arr3 = P.arr3 .+ 11 .- LinRange(1.,10.,N) 
Flow1D.PtoU(P,U,eos)
Pprim = deepcopy(P)
P.arr1 = P.arr1 .+ randn(N)/10
P.arr2 = P.arr2 .+ randn(N)/10
P.arr3 = P.arr3 .+ randn(N)/10

@btime Flow1D.UtoP(U,P,eos,10,1e-5)
println(P.arr1 .- Pprim.arr1)
println(P.arr2 .- Pprim.arr2)
println(P.arr3 .- Pprim.arr3)