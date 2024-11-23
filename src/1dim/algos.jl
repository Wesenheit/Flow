using Base.Threads

function LaxFriedrich(P::ParVector1D,N::Int64,dt::Float64,dx::Float64,T::Float64,eos::EOS,drops::Float64,kwargs...)
    U1::ParVector1D = ParVector1D{Float64,N}()
    Ubuffer::ParVector1D = ParVector1D{Float64,N}()
    F::ParVector1D = ParVector1D{Float64,N}()
    t::Float64 = 0
    PtoU(P,U1,eos)
    for i in 1:N

        Ubuffer.arr1[i] = U1.arr1[i]
        Ubuffer.arr2[i] = U1.arr2[i]
        Ubuffer.arr3[i] = U1.arr3[i]
    end

    out::Vector{ParVector1D} = []
    thres_to_dump::Float64 = drops
    push!(out,deepcopy(P))
    while t < T
        PtoF(P,F,eos) #calculating fluxes

        #Ubuffer.arr1[1] =  0.5 * (U1.arr1[N] + U1.arr1[2]) - 0.5 * dt/dx * (F.arr1[2] - F.arr1[N]) 
        #Ubuffer.arr2[1] =  0.5 * (U1.arr2[N] + U1.arr2[2]) - 0.5 * dt/dx * (F.arr2[2] - F.arr2[N]) 
        #Ubuffer.arr3[1] =  0.5 * (U1.arr3[N] + U1.arr3[2]) - 0.5 * dt/dx * (F.arr3[2] - F.arr3[N]) 
        
        for i in 2:N-1  #Lax Friedrich code
            @inbounds Ubuffer.arr1[i] =  0.5 * (U1.arr1[i - 1] + U1.arr1[i + 1]) - 0.5 * dt/dx * (F.arr1[i + 1] - F.arr1[i - 1]) 
            @inbounds Ubuffer.arr2[i] =  0.5 * (U1.arr2[i - 1] + U1.arr2[i + 1]) - 0.5 * dt/dx * (F.arr2[i + 1] - F.arr2[i - 1]) 
            @inbounds Ubuffer.arr3[i] =  0.5 * (U1.arr3[i - 1] + U1.arr3[i + 1]) - 0.5 * dt/dx * (F.arr3[i + 1] - F.arr3[i - 1]) 
        end
        
        #Ubuffer.arr1[N] =  0.5 * (U1.arr1[N - 1] + U1.arr1[1]) - 0.5 * dt/dx * (F.arr1[1] - F.arr1[N - 1]) 
        #Ubuffer.arr2[N] =  0.5 * (U1.arr2[N - 1] + U1.arr2[1]) - 0.5 * dt/dx * (F.arr2[1] - F.arr2[N - 1]) 
        #Ubuffer.arr3[N] =  0.5 * (U1.arr3[N - 1] + U1.arr3[1]) - 0.5 * dt/dx * (F.arr3[1] - F.arr3[N - 1])
        UtoP(Ubuffer,P,eos,kwargs...) #Conversion to primitive variables
        for i in 1:N
            @inbounds U1.arr1[i] = Ubuffer.arr1[i]
            @inbounds U1.arr2[i] = Ubuffer.arr2[i]
            @inbounds U1.arr3[i] = Ubuffer.arr3[i]
        end
        t += dt
        if t > thres_to_dump
            push!(out,deepcopy(P))
            #println(P.arr3)
            thres_to_dump += drops
            println(t)
        end
        #update(pbar))
    end    
    return out
end


function HARM_HLL(P::ParVector1D,N::Int64,dt::Float64,dx::Float64,T::Float64,eos::EOS,drops::Float64,FluxLimiter::Function,kwargs...)
    U::ParVector1D = ParVector1D{Float64,N}()
    Ubuffer::ParVector1D = ParVector1D{Float64,N}()

    #CL::MVector{N+1,Float64} = @MVector zeros(N+1) #left sound speed
    #CR::MVector{N+1,Float64} = @MVector zeros(N+1) #right sound speed

    PR::ParVector1D = ParVector1D{Float64,N+1}() #Left primitive variable 
    PL::ParVector1D = ParVector1D{Float64,N+1}() #Right primitive variable
    UL::ParVector1D = ParVector1D{Float64,N+1}()
    UR::ParVector1D = ParVector1D{Float64,N+1}()
    FL::ParVector1D = ParVector1D{Float64,N+1}() #Left flux
    FR::ParVector1D = ParVector1D{Float64,N+1}() #Right flux
    F::ParVector1D = ParVector1D{Float64,N+1}() # HLL flux
    r1::Float64 = 0
    r2::Float64 = 0
    r3::Float64 = 0

    t::Float64 = 0
    PtoU(P,U,eos)
    for i in 1:N # copy of U
        @inbounds Ubuffer.arr1[i] = U.arr1[i]
        @inbounds Ubuffer.arr2[i] = U.arr2[i]
        @inbounds Ubuffer.arr3[i] = U.arr3[i]
    end

    out::Vector{ParVector1D} = []
    thres_to_dump::Float64 = drops
    push!(out,deepcopy(P))
    while t < T
        
        @threads :static for i in 2:N-1 # interpolating left and right
            r1 = FluxLimiter( (P.arr1[i] - P.arr1[i-1]) / (P.arr1[i + 1] - P.arr1[i] + 1e-4))
            r2 = FluxLimiter( (P.arr2[i] - P.arr2[i-1]) / (P.arr2[i + 1] - P.arr2[i] + 1e-4))
            r3 = FluxLimiter( (P.arr3[i] - P.arr3[i-1]) / (P.arr3[i + 1] - P.arr3[i] + 1e-4))
            @inbounds PL.arr1[i] = P.arr1[i] + 0.5 * (P.arr1[i + 1] - P.arr1[i]) * r1
            @inbounds PL.arr2[i] = P.arr2[i] + 0.5 * (P.arr2[i + 1] - P.arr2[i]) * r2
            @inbounds PL.arr3[i] = P.arr3[i] + 0.5 * (P.arr3[i + 1] - P.arr3[i]) * r3
            @inbounds PR.arr1[i-1] = P.arr1[i] - 0.5 * (P.arr1[i + 1] - P.arr1[i]) * r1
            @inbounds PR.arr2[i-1] = P.arr2[i] - 0.5 * (P.arr2[i + 1] - P.arr2[i]) * r2
            @inbounds PR.arr3[i-1] = P.arr3[i] - 0.5 * (P.arr3[i + 1] - P.arr3[i]) * r3
        end
        r1 = FluxLimiter( 0.)
        r2 = FluxLimiter( 0.)
        r3 = FluxLimiter( 0.)
        PL.arr1[1] = P.arr1[1] + 0.5 * (P.arr1[2] - P.arr1[1]) * r1
        PL.arr2[1] = P.arr2[1] + 0.5 * (P.arr2[2] - P.arr2[1]) * r2
        PL.arr3[1] = P.arr3[1] + 0.5 * (P.arr3[2] - P.arr3[1]) * r3

        r1 = FluxLimiter(1e4)
        r2 = FluxLimiter(1e4)
        r3 = FluxLimiter(1e4)
        PR.arr1[end] = P.arr1[end] #- 0.5 * (P.arr1[end] - P.arr1[end - 1]) * r1
        PR.arr2[end] = P.arr2[end] #- 0.5 * (P.arr2[end] - P.arr2[end - 1]) * r2
        PR.arr3[end] = P.arr3[end] #- 0.5 * (P.arr3[end] - P.arr3[end - 1]) * r3
        PtoU(PR,UR,eos)
        PtoU(PL,UL,eos)
        PtoF(PR,FR,eos)
        PtoF(PL,FL,eos)

        @threads :static for i in 1:N
            vL::Float64 = PL.arr3[i] / sqrt(PL.arr3[i]^2 + 1)
            vR::Float64 = PR.arr3[i] / sqrt(PR.arr3[i]^2 + 1)
            CL::Float64 = SoundSpeed(max(PL.arr1[i],1e-4),max(PL.arr2[i],1e-4),eos)
            CR::Float64 = SoundSpeed(max(PR.arr1[i],1e-4),max(PR.arr2[i],1e-4),eos)
            C_max::Float64 = max( vL + CL / (1 + vL * CL), vR + CR / (1 + vR * CR)) # velocity composition
            C_min::Float64 = -min( vL - CL / (1 - vL * CL), vR - CR / (1 - vR * CR)) # velocity composition
            @inbounds F.arr1[i] = ( FR.arr1[i] * C_min + FL.arr1[i] * C_max - C_max * C_min * (UR.arr1[i] - UL.arr1[i])) / (C_max + C_min)
            @inbounds F.arr2[i] = ( FR.arr2[i] * C_min + FL.arr2[i] * C_max - C_max * C_min * (UR.arr2[i] - UL.arr2[i])) / (C_max + C_min)
            @inbounds F.arr3[i] = ( FR.arr3[i] * C_min + FL.arr3[i] * C_max - C_max * C_min * (UR.arr3[i] - UL.arr3[i])) / (C_max + C_min)
        end

        @threads :static for i in 2:N-2
            @inbounds Ubuffer.arr1[i] = U.arr1[i] - dt/dx * (F.arr1[i] - F.arr1[i-1])
            @inbounds Ubuffer.arr2[i] = U.arr2[i] - dt/dx * (F.arr2[i] - F.arr2[i-1])
            @inbounds Ubuffer.arr3[i] = U.arr3[i] - dt/dx * (F.arr3[i] - F.arr3[i-1])

        end

        @threads :static for i in 1:N
            @inbounds U.arr1[i] = Ubuffer.arr1[i]
            @inbounds U.arr2[i] = Ubuffer.arr2[i]
            @inbounds U.arr3[i] = Ubuffer.arr3[i]
        end
        UtoP(Ubuffer,P,eos,kwargs...) #Conversion to primitive variables
        t += dt
        if t > thres_to_dump
            push!(out,deepcopy(P))
            #println(P.arr3)
            thres_to_dump += drops
            println(t)
        end
        #update(pbar))
    end    
    return out
end