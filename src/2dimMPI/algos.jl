function Limit(P::ParVector2D,floor::Float64)
    @threads :static for j in 1:P.size_Y
        for i in 1:P.size_X
            P.arr[1,i,j] = max(P.arr[1,i,j],floor)
            P.arr[2,i,j] = max(P.arr[2,i,j],floor)
        end
    end
end


function CalculateLinear(P::ParVector2D,PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,FluxLimiter::Function)
    @threads :static for j in 2:P.size_Y-1
        for i in 2:P.size_X-1
            ip = i+1
            im = i-1
            jp = j+1
            jm = j-1

            buff3 = @view P.arr[:,im,j]
            buff4 = @view P.arr[:,i,j]
            buff5 = @view P.arr[:,ip,j]
            buff1 = @view PL.arr[:,i,j]
            buff2 = @view PR.arr[:,im,j]
            FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)

            buff3 = @view P.arr[:,i,jm]
            buff4 = @view P.arr[:,i,j]
            buff5 = @view P.arr[:,i,jp]
            buff1 = @view PD.arr[:,i,j]
            buff2 = @view PU.arr[:,i,jm]

            FluxLimiter(buff3,buff4,buff5,buff1,buff2,4)
        end
    end
end


function CalculateHLLFluxes(PL::ParVector2D,PR::ParVector2D,PD::ParVector2D,PU::ParVector2D,
                            FL::ParVector2D,FR::ParVector2D,FD::ParVector2D,FU::ParVector2D,
                            UL::ParVector2D,UR::ParVector2D,UD::ParVector2D,UU::ParVector2D,
                            Fx::ParVector2D,Fy::ParVector2D,eos::EOS)
    @threads for j in 1:PL.size_Y
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
                @simd for idx in 1:4
                    Fx.arr[idx,i,j] =  FR.arr[idx,i,j]
                end
            elseif C_min_X < 0 
                @simd for idx in 1:4
                    Fx.arr[idx,i,j] =  FL.arr[idx,i,j] 
                end
            else
                @simd for idx in 1:4
                    Fx.arr[idx,i,j] = ( FR.arr[idx,i,j] * C_min_X + FL.arr[idx,i,j] * C_max_X - C_max_X * C_min_X * (UR.arr[idx,i,j] - UL.arr[idx,i,j])) / (C_max_X + C_min_X)
                end
            end

            if C_max_Y < 0 
                @simd for idx in 1:4
                    Fy.arr[idx,i,j] =  FU.arr[idx,i,j]
                end
            elseif C_min_Y < 0 
                @simd for idx in 1:4
                    Fy.arr[idx,i,j] =  FD.arr[idx,i,j]
                end 
            else
                @simd for idx in 1:4
                    Fy.arr[idx,i,j] = ( FU.arr[idx,i,j] * C_min_Y + FD.arr[idx,i,j] * C_max_Y - C_max_Y * C_min_Y * (UU.arr[idx,i,j] - UD.arr[idx,i,j])) / (C_max_Y + C_min_Y)
                end
            end
        end
    end
end


function HARM_HLL(comm,P::ParVector2D,XMPI,YMPI,Nx::Int64,Ny::Int64,dt::Float64,dx::Float64,dy::Float64,T::Float64,eos::EOS,drops::Float64,FluxLimiter::Function,floor::Float64 = 1e-7,kwargs...)
    U::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)
    Uhalf::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)
    Phalf::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)

    PR::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #Left primitive variable 
    PL::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #Right primitive variable
    PU::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #up primitive variable 
    PD::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #down primitive variable

    P.arr[:,:,1] .= @view P.arr[:,:,2]
    P.arr[:,:,end] .= @view P.arr[:,:,end-1]
    P.arr[:,1,:] .= @view P.arr[:,2,:]
    P.arr[:,end,:] .= @view P.arr[:,end-1,:]

    PL.arr[:,1,:] .= @view P.arr[:,1,:]
    PR.arr[:,1,:] .= @view P.arr[:,2,:]
    PD.arr[:,:,1] .= @view P.arr[:,:,1] 
    PU.arr[:,:,1] .= @view P.arr[:,:,2] 
    
    PL.arr[:,end-1,:] .= @view P.arr[:,end-1,:]
    PR.arr[:,end-1,:] .= @view P.arr[:,end,:]
    PD.arr[:,:,end-1] .= @view P.arr[:,:,end-1]
    PU.arr[:,:,end-1] .= @view P.arr[:,:,end]
    #@inbounds CalculateLinear(Phalf,PL,PR,PD,PU,FluxLimiter)

    UL::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)
    UR::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)
    UU::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)  
    UD::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) 

    FL::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #Left flux
    FR::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) #Right flux
    FU::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm)  
    FD::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) 

    Fx::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) # HLL flux
    Fy::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) # HLL flux

    Fxhalf::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) # HLL flux
    Fyhalf::ParVector2D = ParVector2D{Float64}(Nx,Ny,comm) # HLL flux

    t::Float64 = 0
    SyncBoundaryX(P,comm)
    SyncBoundaryY(P,comm) 
    PtoU(P,U,eos)

    #display(U.arr[3,:,:])

    thres_to_dump::Float64 = drops
    i::Int64 = 0.
    while t < T

        @inbounds CalculateLinear(P,PL,PR,PD,PU,FluxLimiter)
        SyncFlux_X_Right(PR,comm)
        SyncFlux_X_Left(PL,comm)
        SyncFlux_Y_Down(PD,comm)
        SyncFlux_Y_Up(PU,comm)

        @inbounds begin
            @threads :static for j in 1:P.size_Y
                for i in 1:P.size_X

                    PL.arr[1,i,j] = max(PL.arr[1,i,j],floor)
                    PR.arr[2,i,j] = max(PR.arr[2,i,j],floor)
                    PD.arr[1,i,j] = max(PD.arr[1,i,j],floor)
                    PU.arr[2,i,j] = max(PU.arr[2,i,j],floor)

                    bufferp = @view PR.arr[:,i,j]
                    buffer = @view UR.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)

                    bufferp = @view PL.arr[:,i,j]
                    buffer = @view UL.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)
                    
                    bufferp = @view PD.arr[:,i,j]
                    buffer = @view UD.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)
                    
                    bufferp = @view PU.arr[:,i,j]
                    buffer = @view UU.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)


                    bufferp = @view PR.arr[:,i,j]
                    buffer = @view FR.arr[:,i,j]
                    function_PtoFx(bufferp,buffer,eos)

                    bufferp = @view PL.arr[:,i,j]
                    buffer = @view FL.arr[:,i,j]
                    function_PtoFx(bufferp,buffer,eos)

                    bufferp = @view PU.arr[:,i,j]
                    buffer = @view FU.arr[:,i,j]
                    function_PtoFy(bufferp,buffer,eos)

                    bufferp = @view PD.arr[:,i,j]
                    buffer = @view FD.arr[:,i,j]
                    function_PtoFy(bufferp,buffer,eos)
                end
            end
        end

        @inbounds CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fx,Fy,eos)


        @threads  :static for j in 2:P.size_Y-1
            for i in 2:P.size_X-1
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
                @simd for idx in 1:4
                    @inbounds Uhalf.arr[idx,i,j] = U.arr[idx,i,j] - 0.5*dt/dx * (Fx.arr[idx,i,j] - Fx.arr[idx,im1,j]) - 0.5 * dt/dy * (Fy.arr[idx,i,j] - Fy.arr[idx,i,jm1])
                end
            end
        end

        @threads  :static for j in 1:P.size_Y
            for i in 1:P.size_X
                @simd for idx in 1:4
                    Phalf.arr[idx,i,j] = P.arr[idx,i,j]
                end
            end
        end
        @inbounds UtoP(Uhalf,Phalf,eos,kwargs...) #Conversion to primitive variables at the half-step

        Limit(Phalf,floor)

        SyncBoundaryX(Phalf,comm)   
        SyncBoundaryY(Phalf,comm)   


        @inbounds CalculateLinear(Phalf,PL,PR,PD,PU,FluxLimiter)

        SyncFlux_X_Right(PR,comm)
        SyncFlux_X_Left(PL,comm)
        SyncFlux_Y_Down(PD,comm)
        SyncFlux_Y_Up(PU,comm)

        @inbounds begin
            @threads :static for j in 1:P.size_Y-1
                for i in 1:P.size_X-1

                    PL.arr[1,i,j] = max(PL.arr[1,i,j],floor)
                    PR.arr[2,i,j] = max(PR.arr[2,i,j],floor)
                    PD.arr[1,i,j] = max(PD.arr[1,i,j],floor)
                    PU.arr[2,i,j] = max(PU.arr[2,i,j],floor)

                    bufferp = @view PR.arr[:,i,j]
                    buffer = @view UR.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)

                    bufferp = @view PL.arr[:,i,j]
                    buffer = @view UL.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)
                    
                    bufferp = @view PD.arr[:,i,j]
                    buffer = @view UD.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)
                    
                    bufferp = @view PU.arr[:,i,j]
                    buffer = @view UU.arr[:,i,j]
                    function_PtoU(bufferp,buffer,eos)


                    bufferp = @view PR.arr[:,i,j]
                    buffer = @view FR.arr[:,i,j]
                    function_PtoFx(bufferp,buffer,eos)

                    bufferp = @view PL.arr[:,i,j]
                    buffer = @view FL.arr[:,i,j]
                    function_PtoFx(bufferp,buffer,eos)

                    bufferp = @view PU.arr[:,i,j]
                    buffer = @view FU.arr[:,i,j]
                    function_PtoFy(bufferp,buffer,eos)

                    bufferp = @view PD.arr[:,i,j]
                    buffer = @view FD.arr[:,i,j]
                    function_PtoFy(bufferp,buffer,eos)
                end
            end
        end

    @inbounds CalculateHLLFluxes(PL,PR,PD,PU,
                            FL,FR,FD,FU,
                            UL,UR,UD,UU,
                            Fxhalf,Fyhalf,eos)

        @threads :static for j in 2:P.size_Y-1
            for i in 2:P.size_X-1
                im1 = i-1
                jm1 = j-1
                @simd for idx in 1:4
                    @inbounds U.arr[idx,i,j] = U.arr[idx,i,j] - dt/dx * (Fxhalf.arr[idx,i,j] - Fxhalf.arr[idx,im1,j]) - dt/dy * (Fyhalf.arr[idx,i,j] - Fyhalf.arr[idx,i,jm1])
                end
            end
        end
  
        @inbounds UtoP(U,P,eos,kwargs...) #Conversion to primitive variables
        Limit(P,floor)

        SyncBoundaryX(P,comm)
        #SyncBoundaryX(U,comm)

        SyncBoundaryY(P,comm) 
        #SyncBoundaryY(U,comm) 
        t += dt

        if t > thres_to_dump
            i+=1
            size = MPI.Comm_size(comm)

            thres_to_dump += drops
            flat = vec(permutedims(P.arr[:,2:end-1,2:end-1],[1,2,3]))
            if MPI.Comm_rank(comm) == 0
                println(t)
                recvbuf = zeros(Float64,length(flat) *size)  #
            else
                recvbuf = nothing  # Non-root processes don't allocate
            end
            MPI.Gather!(flat, recvbuf, comm)


            if MPI.Comm_rank(comm) == 0
                
                global_matrix = zeros(4,XMPI*Nx,YMPI*Ny)
                for p in 0:(size-1)
                    px,py = MPI.Cart_coords(comm,p)
                    start_x = px * Nx + 1
                    start_y = py * Ny + 1
                    local_start = p * length(flat) + 1
                    local_end = local_start + length(flat) - 1
                    
                    global_matrix[:, start_x:start_x+Nx-1, start_y:start_y+Ny-1] = 
                        reshape(recvbuf[local_start:local_end], 4, Nx, Ny)
                end
                file = h5open("dump"*string(i)*".h5","w")
                write(file,"data",global_matrix)
                write(file,"T",t)
                write(file,"grid",[dx,dy])

                close(file)
            end
        end
    end    
    return i
end
