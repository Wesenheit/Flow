using StaticArrays
# Used scheme
# U - conserved varaibles
# U1 = rho ut - mass conservation
# U2 = T^t_t - energy conservation
# U3 = T^t_x - momentum conservation

# P - primitive variables
# P1 = rho - density
# P2 = u - energy density
# P3 = vx velocity in x

struct ParVector1D{T <:Real,N}
    # Parameter Vector
    arr1::MVector{N,T}
    arr2::MVector{N,T}
    arr3::MVector{N,T}
    size::Int64
    function ParVector1D{T,N}() where {T, N}
        arr1 = MVector{N,T}(zeros(N))
        arr2 = MVector{N,T}(zeros(N))
        arr3 = MVector{N,T}(zeros(N))
        new(arr1,arr2,arr3,N)
    end
end

function PtoU(P::ParVector1D,U::ParVector1D,eos::EOS)
    gamma::Float64
    pressure::Float64
    vx::Float64
    for i in 1:P.size
        gamma = 1/sqrt(1-P.arr3[i]^2)
        pressure = Pressure(P.arr1[i],eos)
        U.arr1[i] = P.arr1[i]*gamma
        U.arr2[i] = (P.arr2[i] + pressure + P.arr1[i])*gamma^2 + pressure
        U.arr3[i] = (P.arr2[i] + pressure + P.arr1[i])*gamma^2*P.arr3[1]
    end
end