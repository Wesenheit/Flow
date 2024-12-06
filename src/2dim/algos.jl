function HARM_HLL(P::ParVector2D,Nx::Int64,Ny::Int64,dt::Float64,dx::Float64,dy::Float64,T::Float64,eos::EOS,drops::Float64,FluxLimiter::Function,kwargs...)
    U::ParVector2D = ParVector2D{Float64,Nx,Ny}()
    Ubuffer::ParVector2D = ParVector2D{Float64,Nx,Ny}()

    #CL::MVector{N+1,Float64} = @MVector zeros(N+1) #left sound speed
    #CR::MVector{N+1,Float64} = @MVector zeros(N+1) #right sound speed

    PR::ParVector2D = ParVector2D{Float64,Nx,Ny}() #Left primitive variable 
    PL::ParVector2D = ParVector2D{Float64,Nx,Ny}() #Right primitive variable
    PU::ParVector2D = ParVector2D{Float64,Nx,Ny}() #up primitive variable 
    PD::ParVector2D = ParVector2D{Float64,Nx,Ny}() #down primitive variable
    
    UL::ParVector2D = ParVector2D{Float64,Nx,Ny}()
    UR::ParVector2D = ParVector2D{Float64,Nx,Ny}()
    UU::ParVector2D = ParVector2D{Float64,Nx,Ny}()  
    UD::ParVector2D = ParVector2D{Float64,Nx,Ny}() 

    FL::ParVector2D = ParVector2D{Float64,Nx,Ny}() #Left flux
    FR::ParVector2D = ParVector2D{Float64,Nx,Ny}() #Right flux
    FU::ParVector2D = ParVector2D{Float64,Nx,Ny}()  
    FD::ParVector2D = ParVector2D{Float64,Nx,Ny}() 

    Fx::ParVector2D = ParVector2D{Float64,Nx,Ny}() # HLL flux
    Fy::ParVector2D = ParVector2D{Float64,Nx,Ny}() # HLL flux

    r1::Float64 = 0
    r2::Float64 = 0
    r3::Float64 = 0

    t::Float64 = 0
    PtoU(P,U,eos)
    for i in 1:Nx
        for j in 1:Ny
            @inbounds Ubuffer.arr[i,j,1] = U.arr[i,j,1]
            @inbounds Ubuffer.arr[i,j,2] = U.arr[i,j,2]
            @inbounds Ubuffer.arr[i,j,3] = U.arr[i,j,3]
            @inbounds Ubuffer.arr[i,j,4] = U.arr[i,j,4]
        end
    end

    out::Vector{ParVector2D} = []
    thres_to_dump::Float64 = drops
    push!(out,deepcopy(P))
    while t < T
        @threads :static for i in 1:Nx
            for j in 1:Ny
                if i == Nx
                    rx = FluxLimiter.( (P.arr[Nx,j,:] .- P.arr[Nx-1,j,:]) ./ (P.arr[1,j,:] .- P.arr[i,j,:] .+ 1e-8))
                    PL.arr[Nx,j,:] = P.arr[Nx,j,:] .+ 0.5 * (P.arr[1,j,:] .- P.arr[i,j,:]) .* rx
                    PR.arr[Nx-1,j,:] = P.arr[Nx,j,:] .- 0.5 * (P.arr[1,j,:] .- P.arr[i,j,:]) .* rx
                elseif i == 1
                    rx = FluxLimiter.( (P.arr[i,j,:] .- P.arr[end,j,:]) ./ (P.arr[i + 1,j,:] .- P.arr[i,j,:] .+ 1e-8))
                    PL.arr[i,j,:] = P.arr[i,j,:] .+ 0.5 * (P.arr[i + 1,j,:] .- P.arr[i,j,:]) .* rx
                    PR.arr[end,j,:] = P.arr[i,j,:] .- 0.5 * (P.arr[i + 1,j,:] .- P.arr[i,j,:]) .* rx
                else
                    rx = FluxLimiter.( (P.arr[i,j,:] .- P.arr[i-1,j,:]) ./ (P.arr[i + 1,j,:] .- P.arr[i,j,:] .+ 1e-8))
                    PL.arr[i,j,:] = P.arr[i,j,:] .+ 0.5 * (P.arr[i + 1,j,:] .- P.arr[i,j,:]) .* rx
                    PR.arr[i-1,j,:] = P.arr[i,j,:] .- 0.5 * (P.arr[i + 1,j,:] .- P.arr[i,j,:]) .* rx
                end
                if j == Ny
                    ry = FluxLimiter.( (P.arr[i,j,:] .- P.arr[i,j-1,:]) ./ (P.arr[i,1,:] .- P.arr[i,j,:] .+ 1e-8))
                    PD.arr[i,j,:] = P.arr[i,j,:] .+ 0.5 * (P.arr[i,1,:] .- P.arr[i,j,:]) .* ry
                    PU.arr[i,j-1,:] = P.arr[i,j,:] .- 0.5 * (P.arr[i,1,:] .- P.arr[i,j,:]) .* ry
                elseif j == 1 
                    ry = FluxLimiter.( (P.arr[i,j,:] .- P.arr[i,end,:]) ./ (P.arr[i,j + 1,:] .- P.arr[i,j,:] .+ 1e-8))
                    PD.arr[i,j,:] = P.arr[i,j,:] .+ 0.5 * (P.arr[i,j + 1,:] .- P.arr[i,j,:]) .* ry
                    PU.arr[i,end,:] = P.arr[i,j,:] .- 0.5 * (P.arr[i,j + 1,:] .- P.arr[i,j,:]) .* ry
                else
                    ry = FluxLimiter.( (P.arr[i,j,:] .- P.arr[i,j-1,:]) ./ (P.arr[i,j + 1,:] .- P.arr[i,j,:] .+ 1e-8) )
                    PD.arr[i,j,:] = P.arr[i,j,:] .+ 0.5 * (P.arr[i,j + 1,:] .- P.arr[i,j,:]) .* ry
                    PU.arr[i,j-1,:] = P.arr[i,j,:] .- 0.5 * (P.arr[i,j + 1,:] .- P.arr[i,j,:]) .* ry
                end
            end
        end

        PR.arr[:,:,1] = clamp.(PR.arr[:,:,1],1e-4,Inf)
        PL.arr[:,:,1] = clamp.(PL.arr[:,:,1],1e-4,Inf)
        PD.arr[:,:,1] = clamp.(PD.arr[:,:,1],1e-4,Inf)
        PU.arr[:,:,1] = clamp.(PU.arr[:,:,1],1e-4,Inf)

        PR.arr[:,:,2] = clamp.(PR.arr[:,:,2],1e-4,Inf)
        PL.arr[:,:,2] = clamp.(PL.arr[:,:,2],1e-4,Inf)
        PD.arr[:,:,2] = clamp.(PD.arr[:,:,2],1e-4,Inf)
        PU.arr[:,:,2] = clamp.(PU.arr[:,:,2],1e-4,Inf)

        PtoU(PR,UR,eos)
        PtoU(PL,UL,eos)
        PtoU(PD,UD,eos)
        PtoU(PU,UU,eos)

        PtoFx(PR,FR,eos)
        PtoFx(PL,FL,eos)
        PtoFy(PD,FD,eos)
        PtoFy(PU,FU,eos)

        @threads :static for i in 1:Nx
            for j in 1:Ny
                vL::Float64 = PL.arr[i,j,3] / sqrt(PL.arr[i,j,3]^2 + PL.arr[i,j,4]^2 + 1)
                vR::Float64 = PR.arr[i,j,3] / sqrt(PR.arr[i,j,3]^2 + PR.arr[i,j,4]^2 + 1)
                vD::Float64 = PD.arr[i,j,4] / sqrt(PD.arr[i,j,3]^2 + PD.arr[i,j,4]^2 + 1)
                vU::Float64 = PU.arr[i,j,4] / sqrt(PU.arr[i,j,3]^2 + PU.arr[i,j,4]^2 + 1)
                #vR::Float64 = PR.arr3[i] / sqrt(PR.arr3[i]^2 + 1)
                CL::Float64 = SoundSpeed(max(PL.arr[i,j,1],1e-4),max(PL.arr[i,j,2],1e-4),eos)
                CR::Float64 = SoundSpeed(max(PR.arr[i,j,1],1e-4),max(PR.arr[i,j,2],1e-4),eos)
                CD::Float64 = SoundSpeed(max(PD.arr[i,j,1],1e-4),max(PD.arr[i,j,2],1e-4),eos)
                CU::Float64 = SoundSpeed(max(PU.arr[i,j,1],1e-4),max(PU.arr[i,j,2],1e-4),eos)

                """
                sigma_S_L::Float64 = CL^2 / ( (PL.arr[i,j,3]^2 + PL.arr[i,j,4]^2 + 1) * (1-CL^2))
                sigma_S_R::Float64 = CR^2 / ( (PR.arr[i,j,3]^2 + PR.arr[i,j,4]^2 + 1) * (1-CR^2))
                sigma_S_D::Float64 = CD^2 / ( (PD.arr[i,j,3]^2 + PD.arr[i,j,4]^2 + 1) * (1-CD^2))
                sigma_S_U::Float64 = CU^2 / ( (PU.arr[i,j,3]^2 + PU.arr[i,j,4]^2 + 1) * (1-CU^2))
                sigma_S_L::Float64 = CL^2 / ( (PL.arr[i,j,3]^2 + 1) * (1-CL^2))
                sigma_S_R::Float64 = CR^2 / ( (PR.arr[i,j,3]^2 + 1) * (1-CR^2))
                sigma_S_D::Float64 = CD^2 / ( (PD.arr[i,j,4]^2 + 1) * (1-CD^2))
                sigma_S_U::Float64 = CU^2 / ( (PU.arr[i,j,4]^2 + 1) * (1-CU^2))
                """

                C_max_X = max( (vL + CL) / (1 + vL * CL), (vR + CR) / (1 + vR * CR)) 
                C_min_X = -min( (vL - CL) / (1 - vL * CL), (vR - CR) / (1 - vR * CR)) 
                C_max_Y = max( (vU + CU) / (1 + vU * CU), (vD + CD) / (1 + vD * CD)) 
                C_min_Y = -min( (vU - CU) / (1 - vU * CU), (vD - CD) / (1 - vD * CD)) 
                #C_max_X::Float64 = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
                #C_min_X::Float64 = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
                #C_max_Y::Float64 = max( (vU + sqrt(sigma_S_U * (1-vU^2 + sigma_S_U)) ) / (1 + sigma_S_U), (vD + sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
                #C_min_Y::Float64 = -min( (vU - sqrt(sigma_S_U * (1-vU^2 + sigma_S_U) )) / (1 + sigma_S_U), (vD - sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition

                @inbounds Fx.arr[i,j,:] = ( FR.arr[i,j,:] * C_min_X + FL.arr[i,j,:] * C_max_X - C_max_X * C_min_X * (UR.arr[i,j,:] - UL.arr[i,j,:])) / (C_max_X + C_min_X)
                @inbounds Fy.arr[i,j,:] = ( FU.arr[i,j,:] * C_min_Y + FD.arr[i,j,:] * C_max_Y - C_max_Y * C_min_Y * (UU.arr[i,j,:] - UD.arr[i,j,:])) / (C_max_Y + C_min_Y)
            end
        end
        @threads :static for i in 1:Nx
            for j in 1:Ny
                if i == 1 
                    im1 = Nx
                else
                    im1 = i-1
                end
                if j == 1
                    jm1 = Ny
                else
                    jm1 = j-1
                end
                @inbounds Ubuffer.arr[i,j,:] = U.arr[i,j,:] - dt/dx * (Fx.arr[i,j,:] - Fx.arr[im1,j,:]) - dt/dy * (Fy.arr[i,j,:] - Fy.arr[i,jm1,:])
                #@inbounds Ubuffer.arr[i,j,:] = U.arr[i,j,:] - dt/dy * (Fy.arr[i,j,:] - Fy.arr[i,jm1,:])
                #@inbounds Ubuffer.arr[i,j,:] = U.arr[i,j,:] - dt/dx * (Fx.arr[i,j,:] - Fx.arr[im1,j,:])

            end
        end

        @threads :static for i in 1:Nx
            for j in 1:Ny
                @inbounds U.arr[i,j,:] = Ubuffer.arr[i,j,:]
            end
        end
        #display(Ubuffer.arr[1:10,55:65,3])
        UtoP(Ubuffer,P,eos,kwargs...) #Conversion to primitive variables
        t += dt
        if t > thres_to_dump
            display(P.arr[:,:,1])
            #display(P.arr[:,1,3])
            #display(Fy.arr[45,:,4])

            push!(out,deepcopy(P))
            thres_to_dump += drops
            println(t)
        end
    end    
    return out
end