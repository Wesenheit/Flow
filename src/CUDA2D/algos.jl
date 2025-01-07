@kernel function function_Limit(P::AbstractArray{T},floor::T) where T
    i, j = @index(Global, NTuple)
    P[1,i,j] = max(P[1,i,j],floor)
    P[2,i,j] = max(P[2,i,j],floor)
end

@kernel function function_Update(U::AbstractArray{T},Ubuff::AbstractArray{T},@Const(Fx::AbstractArray{T}),@Const(Fy::AbstractArray{T}),dt::T,dx::T,dy::T) where T
    i, j = @index(Global, NTuple)
    Nx,Ny = @ndrange()
    @inbounds if i > 1 && i < Nx && j>1 && j<Ny
        @unroll for idx in 1:4
            Ubuff[idx,i,j] = U[idx,i,j] - dt/dx * (Fx[idx,i,j] - Fx[idx,i-1,j]) - dt/dy * (Fy[idx,i,j] - Fy[idx,i,j-1])
        end
    end
end

@kernel function function_CalculateLinear(P::AbstractArray{T},PL::AbstractArray{T},PR::AbstractArray{T},PD::AbstractArray{T},PU::AbstractArray{T}) where T
    i, j = @index(Global, NTuple)
    Nx,Ny = @ndrange()
    @inbounds if i > 1 && i < Nx && j>1 && j<Ny
        """
        sp = @private eltype(P) 4
        sm = @private eltype(P) 4
        ssp = @private eltype(P) 4
        ssm = @private eltype(P) 4
        asp = @private eltype(P) 4
        asm = @private eltype(P) 4
        dU = @private eltype(P) 4
        """
        sp = @MVector zeros(4)
        sm = @MVector zeros(4)
        ssp = @MVector zeros(4)
        ssm = @MVector zeros(4)
        asp = @MVector zeros(4)
        asm = @MVector zeros(4)
        dU = @MVector zeros(4)

        @unroll for idx in 1:4
            sp[idx] = P[idx,i+1,j] - P[idx,i,j]
            sm[idx] = P[idx,i,j] - P[idx,i-1,j]
            ssp[idx] = sign(sp[idx])
            ssm[idx] = sign(sm[idx])
            asp[idx] = abs(sp[idx])
            asm[idx] = abs(sm[idx])
            dU[idx] = 0.25 * (ssp[idx] + ssm[idx]) * min(asp[idx],asm[idx])
            PL[idx,i,j] = P[idx,i,j] + dU[idx]
            PR[idx,i-1,j] = P[idx,i,j] - dU[idx]
        end

        @unroll for idx in 1:4
            sp[idx] = P[idx,i,j+1] - P[idx,i,j]
            sm[idx] = P[idx,i,j] - P[idx,i,j-1]
            ssp[idx] = sign(sp[idx])
            ssm[idx] = sign(sm[idx])
            asp[idx] = abs(sp[idx])
            asm[idx] = abs(sm[idx])
            dU[idx] = 0.25 * (ssp[idx] + ssm[idx]) * min(asp[idx],asm[idx])
            PD[idx,i,j] = P[idx,i,j] + dU[idx]
            PU[idx,i,j-1] = P[idx,i,j] - dU[idx]
        end
    end
end


@kernel function function_CalculateHLLFluxes(PL::AbstractArray{T},PR::AbstractArray{T},PD::AbstractArray{T},PU::AbstractArray{T},
                            FL::AbstractArray{T},FR::AbstractArray{T},FD::AbstractArray{T},FU::AbstractArray{T},
                            UL::AbstractArray{T},UR::AbstractArray{T},UD::AbstractArray{T},UU::AbstractArray{T},
                            Fx::AbstractArray{T},Fy::AbstractArray{T},gamma::T) where T
    i, j = @index(Global, NTuple)
    @inbounds begin
        vL = PL[3,i,j] / sqrt(PL[3,i,j]^2 + PL[4,i,j]^2 + 1)
        vR = PR[3,i,j] / sqrt(PR[3,i,j]^2 + PR[4,i,j]^2 + 1)
        vD = PD[4,i,j] / sqrt(PD[3,i,j]^2 + PD[4,i,j]^2 + 1)
        vU = PU[4,i,j] / sqrt(PU[3,i,j]^2 + PU[4,i,j]^2 + 1)

        CL = SoundSpeed(PL[1,i,j],PL[2,i,j],gamma)
        CR = SoundSpeed(PR[1,i,j],PR[2,i,j],gamma)
        CD = SoundSpeed(PD[1,i,j],PD[2,i,j],gamma)
        CU = SoundSpeed(PU[1,i,j],PU[2,i,j],gamma)

        
        sigma_S_L = CL^2 / ( (PL[3,i,j]^2 + PL[4,i,j]^2 + 1) * (1-CL^2))
        sigma_S_R = CR^2 / ( (PR[3,i,j]^2 + PR[4,i,j]^2 + 1) * (1-CR^2))
        sigma_S_D = CD^2 / ( (PD[3,i,j]^2 + PD[4,i,j]^2 + 1) * (1-CD^2))
        sigma_S_U = CU^2 / ( (PU[3,i,j]^2 + PU[4,i,j]^2 + 1) * (1-CU^2))

        C_max_X = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_min_X = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_max_Y = max( (vU + sqrt(sigma_S_U * (1-vU^2 + sigma_S_U)) ) / (1 + sigma_S_U), (vD + sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
        C_min_Y = -min( (vU - sqrt(sigma_S_U * (1-vU^2 + sigma_S_U) )) / (1 + sigma_S_U), (vD - sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
            
     if C_max_X < 0 
            @unroll for idx in 1:4
                Fx[idx,i,j] =  FR[idx,i,j]
            end
        elseif C_min_X < 0 
            @unroll for idx in 1:4
                Fx[idx,i,j] =  FL[idx,i,j] 
            end
        else
            @unroll for idx in 1:4
                Fx[idx,i,j] = ( FR[idx,i,j] * C_min_X + FL[idx,i,j] * C_max_X - C_max_X * C_min_X * (UR[idx,i,j] - UL[idx,i,j])) / (C_max_X + C_min_X)
            end
        end

        if C_max_Y < 0 
            @unroll for idx in 1:4
                Fy[idx,i,j] =  FU[idx,i,j]
            end
        elseif C_min_Y < 0 
            @unroll for idx in 1:4
                Fy[idx,i,j] =  FD[idx,i,j]
            end 
        else
            @unroll for idx in 1:4
                Fy[idx,i,j] = ( FU[idx,i,j] * C_min_Y + FD[idx,i,j] * C_max_Y - C_max_Y * C_min_Y * (UU[idx,i,j] - UD[idx,i,j])) / (C_max_Y + C_min_Y)
            end
        end
    end
end


function HARM_HLL(comm,P::FlowArr,XMPI::Int64,YMPI::Int64,
                                    Nx::Int64,Ny::Int64,
                                    dt::T,dx::T,dy::T,
                                    T::T,eos::EOS,drops::T,
                                    floor::T = 1e-7,out_dir::String = ".",kwargs...) where T

    backend = KernelAbstractions.get_backend(P.arr)
    U = VectorLike(P)
    Uhalf = VectorLike(P)
    Phalf = VectorLike(P)
    PR = VectorLike(P) #Left primitive variable 
    PL = VectorLike(P) #Right primitive variable
    PU = VectorLike(P) #up primitive variable 
    PD = VectorLike(P) #down primitive variable

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

    UL = VectorLike(P)
    UR = VectorLike(P)
    UU = VectorLike(P)
    UD = VectorLike(P)

    FL = VectorLike(P) #Left flux
    FR = VectorLike(P) #Right flux
    FU = VectorLike(P)
    FD = VectorLike(P)

    Fx = VectorLike(P) # HLL flux
    Fy = VectorLike(P)# HLL flux

    buff_X_1::CuArray{T,2} = CuArray{T}(zeros(4,Ny+2))
    buff_X_2::CuArray{T,2} = CuArray{T}(zeros(4,Ny+2))
    buff_Y_1::CuArray{T,2} = CuArray{T}(zeros(4,Nx+2))
    buff_Y_2::CuArray{T,2} = CuArray{T}(zeros(4,Nx+2))
    t::T = 0
    SyncBoundaryX(P,comm,buff_X_1,buff_X_2)
    SyncBoundaryY(P,comm,buff_Y_1,buff_Y_2)
    Limit = function_Limit(backend)
    Update = function_Update(backend)
    PtoU = function_PtoU(backend)
    UtoP = function_UtoP(backend)
    PtoFx = function_PtoFx(backend)
    PtoFy = function_PtoFy(backend)
    CalculateHLLFluxes = function_CalculateHLLFluxes(backend)
    CalculateLinear = function_CalculateLinear(backend)

    PtoU(P.arr,U.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
    KernelAbstractions.synchronize(backend)
    thres_to_dump::T = drops
    i::Int64 = 0.
    while t < T

        @inbounds begin
            CalculateLinear(P.arr,PL.arr,PR.arr,PD.arr,PU.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            SyncFlux_X_Right(PR,comm,buff_X_1,buff_X_2)
            SyncFlux_X_Left(PL,comm,buff_X_1,buff_X_2)
            SyncFlux_Y_Down(PD,comm,buff_Y_1,buff_Y_2)
            SyncFlux_Y_Up(PU,comm,buff_Y_1,buff_Y_2)
        end
        @inbounds begin
            Limit(PL.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PR.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PD.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PU.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        
        @inbounds begin
            PtoFx(PL.arr,FL.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFx(PR.arr,FR.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFy(PD.arr,FD.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFy(PU.arr,FU.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PL.arr,UL.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PR.arr,UR.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PD.arr,UD.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PU.arr,UU.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            CalculateHLLFluxes(PL.arr,PR.arr,PD.arr,PU.arr,
                            FL.arr,FR.arr,FD.arr,FU.arr,
                            UL.arr,UR.arr,UD.arr,UU.arr,
                            Fx.arr,Fy.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
        
            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            Update(U.arr,Uhalf.arr,Fx.arr,Fy.arr,dt/2.,dx,dy,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end

        Phalf.arr = copy(P.arr)

        @inbounds begin
            UtoP(Uhalf.arr,Phalf.arr,eos.gamma,kwargs[1],kwargs[2],ndrange = (P.size_X,P.size_Y)) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
            Limit(Phalf.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        SyncBoundaryX(Phalf,comm,buff_X_1,buff_X_1)   
        SyncBoundaryY(Phalf,comm,buff_Y_1,buff_Y_2)   

        #####
        #Start of the second cycle
        #
        @inbounds begin
            CalculateLinear(Phalf.arr,PL.arr,PR.arr,PD.arr,PU.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        SyncFlux_X_Right(PR,comm,buff_X_1,buff_X_2)
        SyncFlux_X_Left(PL,comm,buff_X_1,buff_X_2)
        SyncFlux_Y_Down(PD,comm,buff_Y_1,buff_Y_2)
        SyncFlux_Y_Up(PU,comm,buff_Y_1,buff_Y_2)
        @inbounds begin
            Limit(PL.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PR.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PD.arr,floor,ndrange = (P.size_X,P.size_Y))
            Limit(PU.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            PtoFx(PL.arr,FL.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFx(PR.arr,FR.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFy(PD.arr,FD.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoFy(PU.arr,FU.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PL.arr,UL.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PR.arr,UR.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PD.arr,UD.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            PtoU(PU.arr,UU.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end

        @inbounds begin
            CalculateHLLFluxes(PL.arr,PR.arr,PD.arr,PU.arr,
                            FL.arr,FR.arr,FD.arr,FU.arr,
                            UL.arr,UR.arr,UD.arr,UU.arr,
                            Fx.arr,Fy.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))

            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            Update(U.arr,U.arr,Fx.arr,Fy.arr,dt,dx,dy,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        @inbounds begin
            UtoP(U.arr,P.arr,eos.gamma,kwargs[1],kwargs[2],ndrange = (P.size_X,P.size_Y)) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
        
            Limit(P.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        SyncBoundaryX(P,comm,buff_X_1,buff_X_2)
        SyncBoundaryX(P,comm,buff_X_1,buff_X_2)
        t += dt

        if t > thres_to_dump
            i+=1
            size = MPI.Comm_size(comm)

            thres_to_dump += drops
            flat = vec(permutedims(Array{T}(P.arr[:,2:end-1,2:end-1]),[1,2,3]))
            if MPI.Comm_rank(comm) == 0
                println(t)
                recvbuf = zeros(T,length(flat) *size)  #
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
                file = h5open(out_dir * "/dump"*string(i)*".h5","w")
                write(file,"data",global_matrix)
                write(file,"T",t)
                write(file,"grid",[dx,dy])

                close(file)
            end
        end
    end    
    return i
end
