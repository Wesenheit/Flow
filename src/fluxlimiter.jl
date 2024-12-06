function MC(r::Float64)::Float64
    return max(0, min( 2*r,(1 + r)/2,2))
end

function minmod(Um1::Vector{Float64},U::Vector{Float64},Up1::Vector{Float64},out1::Vector{Float64},out2::Vector{Float64},size::Int64)
    sp = Up1-U
    sm = U - Um1
    ssp = sign.(sp)
    ssm = sign.(sm)
    asp = abs.(sp)
    asm = abs.(sm)
    dU = 0.25 .* (ssp + ssm) .* min(asp,asm)
    for i in 1:size
        out1[i] = U[i] + dU[i]
        out2[i] = U[i] - dU[i]
    end
end