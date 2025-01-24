const x = UInt8(1)
const y = UInt8(2)


@kernel inbounds = true function function_Limit(P::AbstractArray{T},floor::T) where T
    i, j = @index(Global, NTuple)
    P[1,i,j] = max(P[1,i,j],floor)
    P[2,i,j] = max(P[2,i,j],floor)
end

@kernel inbounds = true function function_Fluxes(@Const(P::AbstractArray{T}),gamma::T,floor::T,Fglob::AbstractArray{T},dim::UInt8) where T
    i, j = @index(Global, NTuple)
    il, jl = @index(Local, NTuple)
    i = Int32(i)
    j = Int32(j)
    il = Int32(il)
    jl = Int32(jl)
    
    Nx,Ny = @uniform @ndrange()

    N = @uniform @groupsize()[1]
    M = @uniform @groupsize()[2]
    #size of the local threads

    ###paramters on the grid 
    # sometimes it is more beneficient to put some values in the shared memory, sometimes it is more beneficien to put them in registers
    
    #PL_arr = @localmem eltype(P) (4,N, M)
    #PR_arr = @localmem eltype(P) (4,N, M)

    #PL = @view PL_arr[:,il,jl]
    #PR = @view PR_arr[:,il,jl]
    PL = @MVector zeros(T,4)
    PR = @MVector zeros(T,4)
    
    FL_arr = @localmem eltype(P) (4,N, M)
    FR_arr = @localmem eltype(P) (4,N, M)
    FR = @view FR_arr[:,il,jl]
    FL = @view FL_arr[:,il,jl]
    #FR = @MVector zeros(T,4)
    #FL = @MVector zeros(T,4)
    
    #UL = @MVector zeros(T,4)
    #UR = @MVector zeros(T,4)

    UL_arr = @localmem eltype(P) (4,N, M)
    UR_arr = @localmem eltype(P) (4,N, M)
    UR = @view UR_arr[:,il,jl]
    UL = @view UL_arr[:,il,jl]

    Plocal = @localmem eltype(P) (4,N, M)

    for idx in 1:4 
        Plocal[idx,il,jl] = P[idx,i,j]
    end
    @synchronize

    if i > 1 && i < Nx && j > 1 && j < Ny
        for idx in 1:4
            if dim == x
                sp = P[idx,i + Int32(1),j] - Plocal[idx,il,jl]
                sm = Plocal[idx,il,jl] - P[idx,i - Int32(1),j]
            elseif dim == y
                sp = P[idx,i,j + Int32(1)] - Plocal[idx,il,jl]
                sm = Plocal[idx,il,jl] - P[idx,i,j - Int32(1)]
            end
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PL[idx] = Plocal[idx,il,jl] + dU
        end
    end

    if i < Nx-1 && j < Ny-1
        for idx in 1:4
            if dim == x
                sp = P[idx,i + Int32(2),j] - P[idx,i + Int32(1),j]
                sm = P[idx,i + Int32(1),j] - Plocal[idx,il,jl]
            elseif dim == y
                sp = P[idx,i,j + Int32(2)] - P[idx,i,j + Int32(1)]
                sm = P[idx,i,j + Int32(1)] - Plocal[idx,il,jl]
            end
            ssp = sign(sp)
            ssm = sign(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            if dim == x
                PR[idx] = P[idx,i + Int32(1),j] - dU
            elseif dim == y
                PR[idx] = P[idx,i,j + Int32(1)] - dU
            end
        end
    end
    
    for idx in 1:2
        PL[idx] = max(floor,PL[idx])
        PR[idx] = max(floor,PR[idx])
    end
    
    function_PtoU(PR,UR,gamma)
    function_PtoU(PL,UL,gamma)
    if dim == x
        function_PtoFx(PR,FR,gamma)
        function_PtoFx(PL,FL,gamma)
    elseif dim == y
        function_PtoFy(PR,FR,gamma)
        function_PtoFy(PL,FL,gamma)
    end

    if i > 1 && j > 1 && i < Nx && j < Ny
    
        lor = sqrt(PL[3]^2 + PL[4]^2 + 1)
        if dim == x
            vL = PL[3] / lor
            vR = PR[3] / lor
        elseif dim == y
            vL = PL[4] / lor
            vR = PR[4] / lor
        end
        CL = SoundSpeed(PL[1],PL[2],gamma)
        CR = SoundSpeed(PR[1],PR[2],gamma)

        
        sigma_S_L = CL^2 / ( lor^2 * (1-CL^2))
        sigma_S_R = CR^2 / ( lor^2 * (1-CR^2))

        C_max_X = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        C_min_X = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
        if C_max_X < 0 
            for idx in 1:4
                Fglob[idx,i,j] =  FR[idx]
            end
        elseif C_min_X < 0 
            for idx in 1:4
                Fglob[idx,i,j] =  FL[idx] 
            end
        else
            for idx in 1:4
                Fglob[idx,i,j] = ( FR[idx] * C_min_X + FL[idx] * C_max_X - C_max_X * C_min_X * (UR[idx] - UL[idx])) / (C_max_X + C_min_X)
            end
        end
    end
end

@kernel inbounds = true function function_Update(U::AbstractArray{T},Ubuff::AbstractArray{T},dt::T,dx::T,dy::T,Fx::AbstractArray{T},Fy::AbstractArray{T}) where T
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

    Fluxes = function_Fluxes(backend, (SizeX,SizeY))
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

        begin
            Fluxes(P.arr,eos.gamma,floor,Fx.arr,x,ndrange = (P.size_X,P.size_Y))
            Fluxes(P.arr,eos.gamma,floor,Fy.arr,y,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
            Update(U.arr,Uhalf.arr,dt/2,dx,dy,Fx.arr,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end

        Phalf.arr = copy(P.arr)

        begin
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
        begin
            Fluxes(Phalf.arr,eos.gamma,floor,Fx.arr,x,ndrange = (P.size_X,P.size_Y))
            Fluxes(Phalf.arr,eos.gamma,floor,Fy.arr,y,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
            Update(U.arr,U.arr,dt,dx,dy,Fx.arr,Fy.arr,ndrange = (P.size_X,P.size_Y))
            KernelAbstractions.synchronize(backend)
        end
        
        #sync flux on the boundaries
        
        begin
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
            
            flat = vec(permutedims(Array{T}( @view P.arr[:,3:end-2,3:end-2]),[1,2,3]))
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
