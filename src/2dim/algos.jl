using TimerOutputs
function CalculateLinear(P::ParVector2D,PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,FluxLimiter::Function)
    buff_arr_1::Vector{MVector{4,Float64}} = []
    buff_arr_2::Vector{MVector{4,Float64}} = []
    buff_arr_3::Vector{MVector{4,Float64}} = []
    buff_arr_4::Vector{MVector{4,Float64}} = []
    buff_arr_5::Vector{MVector{4,Float64}} = []
    
    for idx in 1:nthreads()
        push!(buff_arr_1,@MVector zeros(4)) 
        push!(buff_arr_2,@MVector zeros(4)) 
        push!(buff_arr_3,@MVector zeros(4)) 
        push!(buff_arr_4,@MVector zeros(4)) 
        push!(buff_arr_5,@MVector zeros(4)) 
    end

    @threads :static for i in 1:P.size_X
        buff1 = buff_arr_1[threadid()]
        buff2 = buff_arr_2[threadid()]
        buff3 = buff_arr_3[threadid()]
        buff4 = buff_arr_4[threadid()]
        buff5 = buff_arr_5[threadid()]
        for j in 1:P.size_Y
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
                buff3[idx] = P.arr[im,j,idx]
                buff4[idx] = P.arr[i,j,idx]
                buff5[idx] = P.arr[ip,j,idx]
            end

            FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)
            for idx in 1:4
                PL.arr[i,j,idx] = buff1[idx]
                PR.arr[im,j,idx] = buff2[idx]
            end

            for idx in 1:4
                buff3[idx] = P.arr[i,jm,idx]
                buff4[idx] = P.arr[i,j,idx]
                buff5[idx] = P.arr[i,jp,idx]
            end

            FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)
            
            for idx in 1:4
                PD.arr[i,j,idx] = buff1[idx]
                PU.arr[i,jm,idx] = buff2[idx]
            end
        end
    end
end


function CalculateHLLFluxes(PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,
                            FL::ParVector2D,FR::ParVector2D,FD::ParVector2D,FU::ParVector2D,
                            UL::ParVector2D,UR::ParVector2D,UD::ParVector2D,UU::ParVector2D,
                            Fx::ParVector2D,Fy::ParVector2D,eos::EOS)
    @threads  for i in 1:PL.size_X
        for j in 1:PL.size_Y
            vL::Float64 = PL.arr[i,j,3] / sqrt(PL.arr[i,j,3]^2 + PL.arr[i,j,4]^2 + 1)
            vR::Float64 = PR.arr[i,j,3] / sqrt(PR.arr[i,j,3]^2 + PR.arr[i,j,4]^2 + 1)
            vD::Float64 = PD.arr[i,j,4] / sqrt(PD.arr[i,j,3]^2 + PD.arr[i,j,4]^2 + 1)
            vU::Float64 = PU.arr[i,j,4] / sqrt(PU.arr[i,j,3]^2 + PU.arr[i,j,4]^2 + 1)

            CL::Float64 = SoundSpeed(PL.arr[i,j,1],PL.arr[i,j,2],eos)
            CR::Float64 = SoundSpeed(PR.arr[i,j,1],PR.arr[i,j,2],eos)
            CD::Float64 = SoundSpeed(PD.arr[i,j,1],PD.arr[i,j,2],eos)
            CU::Float64 = SoundSpeed(PU.arr[i,j,1],PU.arr[i,j,2],eos)

        
            sigma_S_L::Float64 = CL^2 / ( (PL.arr[i,j,3]^2 + PL.arr[i,j,4]^2 + 1) * (1-CL^2))
            sigma_S_R::Float64 = CR^2 / ( (PR.arr[i,j,3]^2 + PR.arr[i,j,4]^2 + 1) * (1-CR^2))
            sigma_S_D::Float64 = CD^2 / ( (PD.arr[i,j,3]^2 + PD.arr[i,j,4]^2 + 1) * (1-CD^2))
            sigma_S_U::Float64 = CU^2 / ( (PU.arr[i,j,3]^2 + PU.arr[i,j,4]^2 + 1) * (1-CU^2))

            C_max_X::Float64 = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
            C_min_X::Float64 = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
            C_max_Y::Float64 = max( (vU + sqrt(sigma_S_U * (1-vU^2 + sigma_S_U)) ) / (1 + sigma_S_U), (vD + sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
            C_min_Y::Float64 = -min( (vU - sqrt(sigma_S_U * (1-vU^2 + sigma_S_U) )) / (1 + sigma_S_U), (vD - sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
            
            if C_max_X < 0 
                for idx in 1:4
                    Fx.arr[i,j,idx] =  FR.arr[i,j,idx]
                end
            elseif C_min_X < 0 
                for idx in 1:4
                    Fx.arr[i,j,idx] =  FL.arr[i,j,idx] 
                end
            else
                for idx in 1:4
                    Fx.arr[i,j,idx] = ( FR.arr[i,j,idx] * C_min_X + FL.arr[i,j,idx] * C_max_X - C_max_X * C_min_X * (UR.arr[i,j,idx] - UL.arr[i,j,idx])) / (C_max_X + C_min_X)
                end
            end

            if C_max_Y < 0 
                for idx in 1:4
                    Fy.arr[i,j,idx] =  FU.arr[i,j,idx]
                end
            elseif C_min_Y < 0 
                for idx in 1:4
                    Fy.arr[i,j,idx] =  FD.arr[i,j,idx]
                end
            else
                for idx in 1:4
                    Fy.arr[i,j,idx] = ( FU.arr[i,j,idx] * C_min_Y + FD.arr[i,j,idx] * C_max_Y - C_max_Y * C_min_Y * (UU.arr[i,j,idx] - UD.arr[i,j,idx])) / (C_max_Y + C_min_Y)
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

    to = TimerOutput()

    while t < T

        @timeit to "interpolation" @inbounds CalculateLinear(P,PL,PR,PD,PU,FluxLimiter)
        """
        PR.arr[:,:,1] = clamp.(PR.arr[:,:,1],floor,Inf)
        PL.arr[:,:,1] = clamp.(PL.arr[:,:,1],floor,Inf)
        PD.arr[:,:,1] = clamp.(PD.arr[:,:,1],floor,Inf)
        PU.arr[:,:,1] = clamp.(PU.arr[:,:,1],floor,Inf)

        PR.arr[:,:,2] = clamp.(PR.arr[:,:,2],floor,Inf)
        PL.arr[:,:,2] = clamp.(PL.arr[:,:,2],floor,Inf)
        PD.arr[:,:,2] = clamp.(PD.arr[:,:,2],floor,Inf)
        PU.arr[:,:,2] = clamp.(PU.arr[:,:,2],floor,Inf)
        """
        @inbounds PtoU(PR,UR,eos)
        @inbounds PtoU(PL,UL,eos)
        @inbounds PtoU(PD,UD,eos)
        @inbounds PtoU(PU,UU,eos)

        @inbounds PtoFx(PR,FR,eos)
        @inbounds PtoFx(PL,FL,eos)
        @inbounds PtoFy(PD,FD,eos)
        @inbounds PtoFy(PU,FU,eos)


        @timeit to "HLL" CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fx,Fy,eos)


        @threads  for i in 1:Nx
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
                for idx in 1:4
                    @inbounds Uhalf.arr[i,j,idx] = U.arr[i,j,idx] - 0.5*dt/dx * (Fx.arr[i,j,idx] - Fx.arr[im1,j,idx]) - 0.5 * dt/dy * (Fy.arr[i,j,idx] - Fy.arr[i,jm1,idx])
                end
            end
        end

        @threads for i in 1:Nx
            for j in 1:Ny
                for idx in 1:4
                    Phalf.arr[i,j,idx] = P.arr[i,j,idx]
                end
            end
        end
        @timeit to "UtoP" @inbounds UtoP(Uhalf,Phalf,eos,kwargs...) #Conversion to primitive variables at the half-step

        @timeit to "interpolation 2" @inbounds CalculateLinear(Phalf,PL,PR,PD,PU,FluxLimiter)
        
        """
        PR.arr[:,:,1] = clamp.(PR.arr[:,:,1],floor,Inf)
        PL.arr[:,:,1] = clamp.(PL.arr[:,:,1],floor,Inf)
        PD.arr[:,:,1] = clamp.(PD.arr[:,:,1],floor,Inf)
        PU.arr[:,:,1] = clamp.(PU.arr[:,:,1],floor,Inf)
        PR.arr[:,:,2] = clamp.(PR.arr[:,:,2],floor,Inf)
        PL.arr[:,:,2] = clamp.(PL.arr[:,:,2],floor,Inf)
        PD.arr[:,:,2] = clamp.(PD.arr[:,:,2],floor,Inf)
        PU.arr[:,:,2] = clamp.(PU.arr[:,:,2],floor,Inf)
        """
        @inbounds PtoU(PR,UR,eos)
        @inbounds PtoU(PL,UL,eos)
        @inbounds PtoU(PD,UD,eos)
        @inbounds PtoU(PU,UU,eos)
        @inbounds PtoFx(PR,FR,eos)
        @inbounds PtoFx(PL,FL,eos)
        @inbounds PtoFy(PD,FD,eos)
        @inbounds PtoFy(PU,FU,eos)

        @timeit to "HLL 2" @inbounds CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fxhalf,Fyhalf,eos)

        @threads  for i in 1:Nx
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
                for idx in 1:4
                    @inbounds U.arr[i,j,idx] = U.arr[i,j,idx] - dt/dx * (Fxhalf.arr[i,j,idx] - Fxhalf.arr[im1,j,idx]) - dt/dy * (Fyhalf.arr[i,j,idx] - Fyhalf.arr[i,jm1,idx])
                end
            end
        end

        @timeit to "UtoP 2" @inbounds UtoP(U,P,eos,kwargs...) #Conversion to primitive variables

        t += dt

        if t > thres_to_dump
            show(to)
            push!(out,deepcopy(P))
            thres_to_dump += drops
            println(t)
        end
    end    
    return out
end
