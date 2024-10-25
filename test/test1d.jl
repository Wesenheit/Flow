include("../src/1dim/Flow1D.jl")
eos = Flow1D.Polytrope(4.0/3.0)
N = 10
P = Flow1D.ParVector1D{Float64,N}()
U = Flow1D.ParVector1D{Float64,N}()
P.arr1 = P.arr1 .+ 0.1
P.arr2 = P.arr2 .+ 0.2
P.arr3 = P.arr3 .+ sqrt(2)
Flow1D.PtoU(P,U,eos)
P.arr1 = P.arr1 .+ randn(N)/100
P.arr2 = P.arr2 .+ randn(N)/100
P.arr3 = P.arr3 .+ randn(N)/100

Flow1D.UtoP(U,P,eos,10,1e-3)
println(P)