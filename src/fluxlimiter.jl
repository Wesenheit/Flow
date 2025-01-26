function MINMOD(q_im1,q_i,q_ip1)
    sp = q_ip1 - q_i
    sm = q_i - q_im1
    ssp = sign(sp)
    ssm = sign(sm)
    asp = abs(sp)
    asm = abs(sm)
    dU = 0.25 * (ssp + ssm) * min(asp,asm)
    return q_i - dU, q_i + dU    
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

function minmod_fun(a, b)
    if a * b > 0.0
        return abs(a) < abs(b) ? a : b
    else
        return 0.0
    end
end

function PPM(q_im2::T,q_im1::T,q_i::T,q_ip1::T,q_ip2::T,C::T = 0.5) where T

    #### Taken from the PLUTO
    
    vp::T = q_im2 * 1.0 /30. + q_im1 * (-13.0 / 60.) +  q_i * 47.0 / 60. + 9.0/ 20. * q_ip1 - 1.0/20. * q_ip2
    vm::T = q_im2 * (-1.0/20.) + q_im1 * (9.0 / 20.) +  q_i * 47.0 / 60. + (-13.0/ 60.) * q_ip1 + 1.0/30. * q_ip2
    

    dvp::T = vp - q_i
    dvm::T = vm - q_i

    dv::T = q_ip1 - q_i
    vp = q_i + minmod_fun(dvp,dv)
    
    dv = q_i - q_im1
    vm = q_i + minmod_fun(dvm,-dv)

    """
    #step 2
    dvp = (q_ip1 - q_i) 
    dvm = (q_i - q_im1) 
    dv = minmod_fun(dvp,dvm)

    vp = q_i + dv * C
    vm = q_i - dv * C
    """
    dvp = vp - q_i
    dvm = vm - q_i

    if dvp * dvm  >= 0
        dvp = 0
        dvm = 0
    else
        if abs(dvp) >= 2 * abs(dvm) 
            dvp = -2 * dvm
        end
        if abs(dvm) >= 2* abs(dvp)
            dvm = -2 * dvp
        end
    end
    return q_i - dvm,q_i + dvp    
end

