function MC(r::Float64)::Float64
    return max(0, min( 2*r,(1 + r)/2,2))
end

function minmod(r::Float64)::Float64
    return max(min(1,r))
end