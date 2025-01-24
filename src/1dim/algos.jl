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
            sp = P.arr1[i+1] -P.arr1[i]
            sm = P.arr1[i] - P.arr1[i-1]
            ssp = sign.(sp)
            ssm = sign.(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PL.arr1[i] = P.arr1[i] + dU
            PR.arr1[i-1] = P.arr1[i] - dU
            
            sp = P.arr2[i+1] -P.arr2[i]
            sm = P.arr2[i] - P.arr2[i-1]
            ssp = sign.(sp)
            ssm = sign.(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PL.arr2[i] = P.arr2[i] + dU
            PR.arr2[i-1] = P.arr2[i] - dU

            sp = P.arr3[i+1] -P.arr3[i]
            sm = P.arr3[i] - P.arr3[i-1]
            ssp = sign.(sp)
            ssm = sign.(sm)
            asp = abs(sp)
            asm = abs(sm)
            dU = 0.25 * (ssp + ssm) * min(asp,asm)
            PL.arr3[i] = P.arr3[i] + dU
            PR.arr3[i-1] = P.arr3[i] - dU
        end
        r1 = FluxLimiter( 0.)
        r2 = FluxLimiter( 0.)
        r3 = FluxLimiter( 0.)
        PL.arr1[1] = P.arr1[1] + 0.5 * (P.arr1[2] - P.arr1[1]) * r1
        PL.arr2[1] = P.arr2[1] + 0.5 * (P.arr2[2] - P.arr2[1]) * r2
        PL.arr3[1] = P.arr3[1] + 0.5 * (P.arr3[2] - P.arr3[1]) * r3

        PR.arr1[end] = P.arr1[end] #- 0.5 * (P.arr1[end] - P.arr1[end - 1]) * r1
        PR.arr2[end] = P.arr2[end] #- 0.5 * (P.arr2[end] - P.arr2[end - 1]) * r2
        PR.arr3[end] = P.arr3[end] #- 0.5 * (P.arr3[end] - P.arr3[end - 1]) * r3
        PtoU(PR,UR,eos)
        PtoU(PL,UL,eos)
        PtoF(PR,FR,eos)
        PtoF(PL,FL,eos)

        for i in 1:N
            vL::Float64 = PL.arr3[i] / sqrt(PL.arr3[i]^2 + 1)
            vR::Float64 = PR.arr3[i] / sqrt(PR.arr3[i]^2 + 1)
            CL::Float64 = SoundSpeed(max(PL.arr1[i],1e-4),max(PL.arr2[i],1e-4),eos)
            CR::Float64 = SoundSpeed(max(PR.arr1[i],1e-4),max(PR.arr2[i],1e-4),eos)
                    
            sigma_S_L::Float64 = CL^2 / ( (PL.arr3[i]^2 + 1) * (1-CL^2))
            sigma_S_R::Float64 = CR^2 / ( (PR.arr3[i]^2 + 1) * (1-CR^2))

            C_max::Float64 = max( (vL + sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR + sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
            C_min::Float64 = -min( (vL - sqrt(sigma_S_L * (1-vL^2 + sigma_S_L)) ) / (1 + sigma_S_L), (vR - sqrt(sigma_S_R * (1-vR^2 + sigma_S_R)) ) / (1 + sigma_S_R)) # velocity composition
              
            if C_max < 0 
                F.arr1[i] =  FR.arr1[i] 
                F.arr2[i] =  FR.arr2[i] 
                F.arr3[i] =  FR.arr3[i] 
            elseif C_min < 0 
                F.arr1[i] =  FL.arr1[i] 
                F.arr2[i] =  FL.arr2[i] 
                F.arr3[i] =  FL.arr3[i] 
            else
                @inbounds F.arr1[i] = ( FR.arr1[i] * C_min + FL.arr1[i] * C_max - C_max * C_min * (UR.arr1[i] - UL.arr1[i])) / (C_max + C_min)
                @inbounds F.arr2[i] = ( FR.arr2[i] * C_min + FL.arr2[i] * C_max - C_max * C_min * (UR.arr2[i] - UL.arr2[i])) / (C_max + C_min)
                @inbounds F.arr3[i] = ( FR.arr3[i] * C_min + FL.arr3[i] * C_max - C_max * C_min * (UR.arr3[i] - UL.arr3[i])) / (C_max + C_min)
            end
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
