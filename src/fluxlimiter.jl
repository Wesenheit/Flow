function MC(r::Float64)::Float64
    return max(0, min( 2*r,(1 + r)/2,2))
end

function minmod(Um1::AbstractVector,U::AbstractVector,Up1::AbstractVector,out1::AbstractVector,out2::AbstractVector,size::Int64)
    @simd for i in 1:size
        sp = Up1[i] -U[i]
        sm = U[i] - Um1[i]
        ssp = sign.(sp)
        ssm = sign.(sm)
        asp = abs(sp)
        asm = abs(sm)
        dU = 0.25 * (ssp + ssm) * min(asp,asm)
        out1[i] = U[i] + dU
        out2[i] = U[i] - dU
    end
end
