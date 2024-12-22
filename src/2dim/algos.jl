function Limit(P::ParVector2D,floor::Float64)
    @threads for j in 1:P.size_Y
        for i in 1:P.size_X
            P.arr[1,i,j] = max(P.arr[1,i,j],floor)
            P.arr[2,i,j] = max(P.arr[2,i,j],floor)
        end
    end
end


function CalculateLinear(P::ParVector2D,PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,FluxLimiter::Function)
    @sync for chunk in P.part
        @spawn begin
            buff1::MVector{4,Float64} = @MVector zeros(4)
            buff2::MVector{4,Float64} = @MVector zeros(4)
            buff3::MVector{4,Float64} = @MVector zeros(4)
            buff4::MVector{4,Float64} = @MVector zeros(4)
            buff5::MVector{4,Float64} = @MVector zeros(4)
            for j in chunk
                for i in 1:P.size_X
                    if i == P.size_X 
                        ip = 1
                        im = i-1
                    elseif i == 1
                        ip = i+1
                        im = P.size_X
                    else
                        ip = i+1
                        im = i-1
                    end

                    if j == P.size_Y
                        jp = 1
                        jm = j-1
                    elseif j == 1
                        jp = j+1
                        jm = P.size_Y
                    else
                        jp = j+1
                        jm = j-1
                    end

                    for idx in 1:4
                        buff3[idx] = P.arr[idx,im,j]
                        buff4[idx] = P.arr[idx,i,j]
                        buff5[idx] = P.arr[idx,ip,j]
                    end

                    FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)
                    for idx in 1:4
                        PL.arr[idx,i,j] = buff1[idx]
                        PR.arr[idx,im,j] = buff2[idx]
                    end

                    for idx in 1:4
                        buff3[idx] = P.arr[idx,i,jm]
                        buff4[idx] = P.arr[idx,i,j]
                        buff5[idx] = P.arr[idx,i,jp]
                    end

                    FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)
            
                    for idx in 1:4
                        PD.arr[idx,i,j] = buff1[idx]
                        PU.arr[idx,i,jm] = buff2[idx]
                    end
                end
            end
        end
    end
end


function CalculateHLLFluxes(PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,
                            FL::ParVector2D,FR::ParVector2D,FD::ParVector2D,FU::ParVector2D,
                            UL::ParVector2D,UR::ParVector2D,UD::ParVector2D,UU::ParVector2D,
                            Fx::ParVector2D,Fy::ParVector2D,eos::EOS)
    @sync for chunk in PL.part
        @spawn begin
            for j in chunk
                for i in 1:PL.size_X
                    vL::Float64 = PL.arr[3,i,j] / sqrt(PL.arr[3,i,j]^2 + PL.arr[4,i,j]^2 + 1)
                    vR::Float64 = PR.arr[3,i,j] / sqrt(PR.arr[3,i,j]^2 + PR.arr[4,i,j]^2 + 1)
                    vD::Float64 = PD.arr[4,i,j] / sqrt(PD.arr[3,i,j]^2 + PD.arr[4,i,j]^2 + 1)
                    vU::Float64 = PU.arr[4,i,j] / sqrt(PU.arr[3,i,j]^2 + PU.arr[4,i,j]^2 + 1)

                    CL::Float64 = SoundSpeed(PL.arr[1,i,j],PL.arr[2,i,j],eos)
                    CR::Float64 = SoundSpeed(PR.arr[1,i,j],PR.arr[2,i,j],eos)
                    CD::Float64 = SoundSpeed(PD.arr[1,i,j],PD.arr[2,i,j],eos)
                    CU::Float64 = SoundSpeed(PU.arr[1,i,j],PU.arr[2,i,j],eos)

        
                    sigma_S_L::Float64 = CL^2 / ( (PL.arr[3,i,j]^2 + PL.arr[4,i,j]^2 + 1) * (1-CL^2))
                    sigma_S_R::Float64 = CR^2 / ( (PR.arr[3,i,j]^2 + PR.arr[4,i,j]^2 + 1) * (1-CR^2))
                    sigma_S_D::Float64 = CD^2 / ( (PD.arr[3,i,j]^2 + PD.arr[4,i,j]^2 + 1) * (1-CD^2))
                    sigma_S_U::Float64 = CU^2 / ( (PU.arr[3,i,j]^2 + PU.arr[4,i,j]^2 + 1) * (1-CU^2))

                    C_max_X::Float64 = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
                    C_min_X::Float64 = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
                    C_max_Y::Float64 = max( (vU + sqrt(sigma_S_U * (1-vU^2 + sigma_S_U)) ) / (1 + sigma_S_U), (vD + sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
                    C_min_Y::Float64 = -min( (vU - sqrt(sigma_S_U * (1-vU^2 + sigma_S_U) )) / (1 + sigma_S_U), (vD - sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
            
                    if C_max_X < 0 
                        for idx in 1:4
                            Fx.arr[idx,i,j] =  FR.arr[idx,i,j]
                        end
                    elseif C_min_X < 0 
                        for idx in 1:4
                            Fx.arr[idx,i,j] =  FL.arr[idx,i,j] 
                        end
                    else
                        for idx in 1:4
                            Fx.arr[idx,i,j] = ( FR.arr[idx,i,j] * C_min_X + FL.arr[idx,i,j] * C_max_X - C_max_X * C_min_X * (UR.arr[idx,i,j] - UL.arr[idx,i,j])) / (C_max_X + C_min_X)
                        end
                    end

                    if C_max_Y < 0 
                        for idx in 1:4
                            Fy.arr[idx,i,j] =  FU.arr[idx,i,j]
                        end
                    elseif C_min_Y < 0 
                        for idx in 1:4
                            Fy.arr[idx,i,j] =  FD.arr[idx,i,j]
                        end 
                    else
                        for idx in 1:4
                            Fy.arr[idx,i,j] = ( FU.arr[idx,i,j] * C_min_Y + FD.arr[idx,i,j] * C_max_Y - C_max_Y * C_min_Y * (UU.arr[idx,i,j] - UD.arr[idx,i,j])) / (C_max_Y + C_min_Y)
                        end
                    end
                end
            end
        end
    end
end


function HARM_HLL(P::ParVector2D,Nx::Int64,Ny::Int64,dt::Float64,dx::Float64,dy::Float64,T::Float64,eos::EOS,drops::Float64,FluxLimiter::Function,floor::Float64 = 1e-7,kwargs...)
    U::ParVector2D = ParVector2D{Float64,Nx,Ny}()
    Uhalf::ParVector2D = ParVector2D{Float64,Nx,Ny}()
    Phalf::ParVector2D = ParVector2D{Float64,Nx,Ny}()

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

    Fxhalf::ParVector2D = ParVector2D{Float64,Nx,Ny}() # HLL flux
    Fyhalf::ParVector2D = ParVector2D{Float64,Nx,Ny}() # HLL flux

    t::Float64 = 0
    PtoU(P,U,eos)

    out::Vector{ParVector2D} = []
    thres_to_dump::Float64 = drops
    push!(out,deepcopy(P))


    while t < T

        @inbounds CalculateLinear(P,PL,PR,PD,PU,FluxLimiter)

        @inbounds PtoU(PR,UR,eos)
        @inbounds PtoU(PL,UL,eos)
        @inbounds PtoU(PD,UD,eos)
        @inbounds PtoU(PU,UU,eos)
        
        @inbounds Limit(PL,floor)
        @inbounds Limit(PR,floor)
        @inbounds Limit(PU,floor)
        @inbounds Limit(PD,floor)

        @inbounds PtoFx(PR,FR,eos)
        @inbounds PtoFx(PL,FL,eos)
        @inbounds PtoFy(PD,FD,eos)
        @inbounds PtoFy(PU,FU,eos)


        @inbounds CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fx,Fy,eos)


        @threads  for j in 1:Ny
            for i in 1:Nx
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
                for idx in 1:4
                    @inbounds Uhalf.arr[idx,i,j] = U.arr[idx,i,j] - 0.5*dt/dx * (Fx.arr[idx,i,j] - Fx.arr[idx,im1,j]) - 0.5 * dt/dy * (Fy.arr[idx,i,j] - Fy.arr[idx,i,jm1])
                end
            end
        end

        @threads for j in 1:Ny
            for i in 1:Nx
                for idx in 1:4
                    Phalf.arr[idx,i,j] = P.arr[idx,i,j]
                end
            end
        end
        @inbounds UtoP(Uhalf,Phalf,eos,kwargs...) #Conversion to primitive variables at the half-step

        Limit(Phalf,floor)

        @inbounds CalculateLinear(Phalf,PL,PR,PD,PU,FluxLimiter)

        @inbounds PtoU(PR,UR,eos)
        @inbounds PtoU(PL,UL,eos)
        @inbounds PtoU(PD,UD,eos)
        @inbounds PtoU(PU,UU,eos)
        

        Limit(PL,floor)
        Limit(PR,floor)
        Limit(PU,floor)
        Limit(PD,floor)

        @inbounds PtoFx(PR,FR,eos)
        @inbounds PtoFx(PL,FL,eos)
        @inbounds PtoFy(PD,FD,eos)
        @inbounds PtoFy(PU,FU,eos)

        @inbounds CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fxhalf,Fyhalf,eos)

        @threads  for j in 1:Ny
            for i in 1:Nx
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
                for idx in 1:4
                    @inbounds U.arr[idx,i,j] = U.arr[idx,i,j] - dt/dx * (Fxhalf.arr[idx,i,j] - Fxhalf.arr[idx,im1,j]) - dt/dy * (Fyhalf.arr[idx,i,j] - Fyhalf.arr[idx,i,jm1])
                end
            end
        end

        @inbounds UtoP(U,P,eos,kwargs...) #Conversion to primitive variables

        Limit(P,floor)

        t += dt

        if t > thres_to_dump
            push!(out,deepcopy(P))
            thres_to_dump += drops
            println(t)
        end
    end    
    return out
end
