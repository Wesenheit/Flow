@kernel function function_Limit(P::AbstractArray{T},floor::T) where T
    i, j = @index(Global, NTuple)
    P[1,i,j] = max(P[1,i,j],floor)
    P[2,i,j] = max(P[2,i,j],floor)
end

@kernel function function_FluxesX(@Const(P::AbstractArray{T}),gamma::T,floor::T,Fxglob::AbstractArray{T}) where T
    #i, j = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    igr, jgr = @index(Group, NTuple)
    
    Nx,Ny = @uniform @ndrange()

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    i = (igr - 1)*N + il
    j = (jgr - 1)*M + jl
    
    PL = @localmem eltype(P) (4,N, M)
    PR = @localmem eltype(P) (4,N, M)
    
    FL = @localmem eltype(P) (4,N, M)
    FR = @localmem eltype(P) (4,N, M)
    
    UL = @localmem eltype(P) (4,N, M)
    UR = @localmem eltype(P) (4,N, M)
    
    if i > 1 && i < Nx && j > 1 && j < Ny
        for idx in 1:4
            sp = P[idx,i+1,j] - P[idx,i,j]
            sm = P[idx,i,j] - P[idx,i-1,j]
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PL[idx,il,jl] = P[idx,i,j] + dU
            #PR[idx,mod1(il-1,N),jl] = P[idx,i,j] - dU
        end
    end

    if i < Nx-1 && j < Ny-1
        for idx in 1:4
            sp = P[idx,i+2,j] - P[idx,i+1,j]
            sm = P[idx,i+1,j] - P[idx,i,j]
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PR[idx,il,jl] = P[idx,i+1,j] - dU
        end
    end
    @synchronize
    
    bufferp = @view PR[:,il,jl]
    buffer = @view UR[:,il,jl]
    function_PtoU(bufferp,buffer,gamma)

    bufferp = @view PL[:,il,jl]
    buffer = @view UL[:,il,jl]
    function_PtoU(bufferp,buffer,gamma)
                    
    bufferp = @view PR[:,il,jl]
    buffer = @view FR[:,il,jl]
    function_PtoFx(bufferp,buffer,gamma)

    bufferp = @view PL[:,il,jl]
    buffer = @view FL[:,il,jl]
    function_PtoFx(bufferp,buffer,gamma)
    
    for idx in 1:2
        PL[idx,il,jl] = max(floor,PL[idx,il,jl])
        PR[idx,il,jl] = max(floor,PR[idx,il,jl])
    end
    if i > 1 && j > 1 && i < Nx && j < Ny
        
        vL = PL[3,il,jl] / sqrt(PL[3,il,jl]^2 + PL[4,il,jl]^2 + 1)
        vR = PR[3,il,jl] / sqrt(PR[3,il,jl]^2 + PR[4,il,jl]^2 + 1)
        
        CL = SoundSpeed(PL[1,il,jl],PL[2,il,jl],gamma)
        CR = SoundSpeed(PR[1,il,jl],PR[2,il,jl],gamma)

        
        sigma_S_L = CL^2 / ( (PL[3,il,jl]^2 + PL[4,il,jl]^2 + 1) * (1-CL^2))
        sigma_S_R = CR^2 / ( (PR[3,il,jl]^2 + PR[4,il,jl]^2 + 1) * (1-CR^2))

        C_max_X = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_min_X = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        if C_max_X < 0 
            for idx in 1:4
                Fxglob[idx,i,j] =  FR[idx,il,jl]
            end
        elseif C_min_X < 0 
            for idx in 1:4
                Fxglob[idx,i,j] =  FL[idx,il,jl] 
            end
        else
            for idx in 1:4
                Fxglob[idx,i,j] = ( FR[idx,il,jl] * C_min_X + FL[idx,il,jl] * C_max_X - C_max_X * C_min_X * (UR[idx,il,jl] - UL[idx,il,jl])) / (C_max_X + C_min_X)
            end
        end
    end
end

@kernel function function_FluxesY(@Const(P::AbstractArray{T}),gamma::T,floor::T,Fyglob::AbstractArray{T}) where T
    #i, j = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    igr, jgr = @index(Group, NTuple)
    
    Nx,Ny = @uniform @ndrange()

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    i = (igr - 1)*N + il
    j = (jgr - 1)*M + jl
    
    PD = @localmem eltype(P) (4,N, M)
    PU = @localmem eltype(P) (4,N, M)
    
    FD = @localmem eltype(P) (4,N, M)
    FU = @localmem eltype(P) (4,N, M)
    
    UD = @localmem eltype(P) (4,N, M)
    UU = @localmem eltype(P) (4,N, M)
    
    
    if i > 1 && i < Nx && j > 1 && j < Ny
        for idx in 1:4
            sp = P[idx,i,j+1] - P[idx,i,j]
            sm = P[idx,i,j] - P[idx,i,j-1]
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PD[idx,il,jl] = P[idx,i,j] + dU
            #PU[idx,il,mod1(jl-1,M)] = P[idx,i,j] - dU
        end
    end

    if i < Nx-1 && j < Ny-1
        for idx in 1:4
            sp = P[idx,i,j+2] - P[idx,i,j+1]
            sm = P[idx,i,j+1] - P[idx,i,j]
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PU[idx,il,jl] = P[idx,i,j+1] - dU
        end
    end
    @synchronize
                    
    bufferp = @view PD[:,il,jl]
    buffer = @view UD[:,il,jl]
    function_PtoU(bufferp,buffer,gamma)
                    
    bufferp = @view PU[:,il,jl]
    buffer = @view UU[:,il,jl]
    function_PtoU(bufferp,buffer,gamma)

    bufferp = @view PU[:,il,jl]
    buffer = @view FU[:,il,jl]
    function_PtoFy(bufferp,buffer,gamma)

    bufferp = @view PD[:,il,jl]
    buffer = @view FD[:,il,jl]
    function_PtoFy(bufferp,buffer,gamma)
    
    for idx in 1:2
        PD[idx,il,jl] = max(floor,PD[idx,il,jl])
        PU[idx,il,jl] = max(floor,PU[idx,il,jl])
    end
    if i > 1 && j > 1 && i < Nx && j < Ny
        
        vD = PD[4,il,jl] / sqrt(PD[3,il,jl]^2 + PD[4,il,jl]^2 + 1)
        vU = PU[4,il,jl] / sqrt(PU[3,il,jl]^2 + PU[4,il,jl]^2 + 1)
        
        CD = SoundSpeed(PD[1,il,jl],PD[2,il,jl],gamma)
        CU = SoundSpeed(PU[1,il,jl],PU[2,il,jl],gamma)

        
        sigma_S_D = CD^2 / ( (PD[3,il,jl]^2 + PD[4,il,jl]^2 + 1) * (1-CD^2))
        sigma_S_U = CU^2 / ( (PU[3,il,jl]^2 + PU[4,il,jl]^2 + 1) * (1-CU^2))

        C_max_Y = max( (vU + sqrt(sigma_S_U * (1-vU^2 + sigma_S_U)) ) / (1 + sigma_S_U), (vD + sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
        C_min_Y = -min( (vU - sqrt(sigma_S_U * (1-vU^2 + sigma_S_U) )) / (1 + sigma_S_U), (vD - sqrt(sigma_S_D * (1-vD^2 + sigma_S_D)) ) / (1 + sigma_S_D)) # velocity composition
        if C_max_Y <= 0 
            for idx in 1:4
                Fyglob[idx,i,j] =  FU[idx,il,jl]
            end
        elseif C_min_Y <= 0 
            for idx in 1:4
                Fyglob[idx,i,j] =  FD[idx,il,jl]
            end 
        else
            for idx in 1:4
                Fyglob[idx,i,j] = ( FU[idx,il,jl] * C_min_Y + FD[idx,il,jl] * C_max_Y - C_max_Y * C_min_Y * (UU[idx,il,jl] - UD[idx,il,jl])) / (C_max_Y + C_min_Y)
            end
        end
    end
end


@kernel function function_Update(U::AbstractArray{T},Ubuff::AbstractArray{T},dt::T,dx::T,dy::T,Fx::AbstractArray{T},Fy::AbstractArray{T}) where T
    i, j = @index(Global, NTuple)    
    Nx,Ny = @uniform @ndrange()
    
    if i >2 && j > 2 && i < Nx-1 && j < Ny-1
        for idx in 1:4
            Ubuff[idx,i,j] = U[idx,i,j] - dt/dx * (Fx[idx,i,j] - Fx[idx,i-1,j]) - dt/dy * (Fy[idx,i,j] - Fy[idx,i,j-1])
        end
    end
end


function HARM_HLL(comm,P::FlowArr,XMPI::Int64,YMPI::Int64,
                                    SizeX::Int64,SizeY::Int64,
                                    dt::T,dx::T,dy::T,
                                    Tmax::T,eos::EOS{T},drops::T,
                                    floor::T = 1e-7,out_dir::String = ".",kwargs...) where T

    backend = KernelAbstractions.get_backend(P.arr)
    U = VectorLike(P)
    Uhalf = VectorLike(P)
    Phalf = VectorLike(P)
    Fx = VectorLike(P)
    Fy = VectorLike(P)


    buff_X_1 = allocate(backend,T,4,2,P.size_Y)
    buff_X_2 = allocate(backend,T,4,2,P.size_Y)
    buff_X_3 = allocate(backend,T,4,2,P.size_Y)
    buff_X_4 = allocate(backend,T,4,2,P.size_Y)
    buff_Y_1 = allocate(backend,T,4,P.size_X,2)
    buff_Y_2 = allocate(backend,T,4,P.size_X,2)
    buff_Y_3 = allocate(backend,T,4,P.size_X,2)
    buff_Y_4 = allocate(backend,T,4,P.size_X,2)
    t::T = 0

    SendBoundaryX(P,comm,buff_X_1,buff_X_2)
    SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
    WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)

    Limit = function_Limit(backend)
    @assert mod(P.size_X,SizeX) == 0 
    @assert mod(P.size_Y,SizeY) == 0 
    wgX = div(P.size_X,SizeX)
    wgY = div(P.size_X,SizeY)

    FluxesX = function_FluxesX(backend, (SizeX,SizeY))
    FluxesY = function_FluxesY(backend, (SizeX,SizeY))
    Update = function_Update(backend)
    UtoP = function_UtoP(backend, (SizeX,SizeY))
    PtoU = kernel_PtoU(backend)
    
    PtoU(P.arr,U.arr,eos.gamma,ndrange = (P.size_X,P.size_Y))
    KernelAbstractions.synchronize(backend)
    thres_to_dump::T = drops
    i::Int64 = 0.
    if MPI.Comm_rank(comm) == 0
        t0 = time()
    end
    while t < Tmax

        @inbounds begin
            FluxesX(P.arr,eos.gamma,floor,Fx.arr,ndrange = (P.size_X,P.size_Y))
            FluxesY(P.arr,eos.gamma,floor,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
            Update(U.arr,Uhalf.arr,dt/2,dx,dy,Fx.arr,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end

        Phalf.arr = copy(P.arr)

        @inbounds begin
            UtoP(Uhalf.arr,Phalf.arr,eos.gamma,kwargs[1],kwargs[2],ndrange = (P.size_X,P.size_Y)) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
            Limit(Phalf.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        
        
        SendBoundaryX(P,comm,buff_X_1,buff_X_2)
        SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)

        #####
        #Start of the second cycle
        #
        #Calculate Flux
        @inbounds begin
            FluxesX(Phalf.arr,eos.gamma,floor,Fx.arr,ndrange = (P.size_X,P.size_Y))
            FluxesY(Phalf.arr,eos.gamma,floor,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
            Update(U.arr,U.arr,dt,dx,dy,Fx.arr,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        
        #sync flux on the boundaries
        
        @inbounds begin
            UtoP(U.arr,P.arr,eos.gamma,kwargs[1],kwargs[2],ndrange = (P.size_X,P.size_Y)) #Conversion to primitive variables at the half-step
            KernelAbstractions.synchronize(backend)
        
            Limit(P.arr,floor,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        
        SendBoundaryX(P,comm,buff_X_1,buff_X_2)
        SendBoundaryY(P,comm,buff_Y_1,buff_Y_2)
        WaitForBoundary(P,comm,buff_X_3,buff_X_4,buff_Y_3,buff_Y_4)
        
        t += dt
        if t > thres_to_dump
            i+=1
            #increase the threshold to dump
            thres_to_dump += drops

            #start the boundary transport
            size = MPI.Comm_size(comm)
            
            flat = vec(permutedims(Array{T}(P.arr[:,3:end-2,3:end-2]),[1,2,3]))
            if MPI.Comm_rank(comm) == 0
                println(t," elapsed: ",time() - t0, " s")
                t0 = time()
                recvbuf = zeros(T,length(flat) *size)  #
            else
                recvbuf = nothing  # Non-root processes don't allocate
            end
            MPI.Gather!(flat, recvbuf, comm)


            if MPI.Comm_rank(comm) == 0
                global_matrix = zeros(4,XMPI * (P.size_X - 4),YMPI * (P.size_Y - 4))
                for p in 0:(size-1)
                    px,py = MPI.Cart_coords(comm,p)
                    start_x = px * (P.size_X - 4) + 1
                    start_y = py * (P.size_Y - 4) + 1
                    local_start = p * length(flat) + 1
                    local_end = local_start + length(flat) - 1
                    
                    global_matrix[:, start_x:start_x+(P.size_X - 4)-1, start_y:start_y+(P.size_Y - 4)-1] = 
                        reshape(recvbuf[local_start:local_end], 4, P.size_X-4, P.size_Y-4)
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
