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
        UtoP(Ubuffer,P,eos,kwargs...) #Coversion to primitive variables
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


function HARM_HLL(P::ParVector1D,N::Int64,dt::Float64,dx::Float64,T::Float64,eos::EOS,drops::Float64,kwargs...)
    U1::ParVector1D = ParVector1D{Float64,N}()
    Ubuffer::ParVector1D = ParVector1D{Float64,N}()

    C_L::MVector{N+1,Float64} = @MVector zeros(N+1) #left sound speed
    C_R::MVector{N+1,Float64} = @MVector zeros(N+1) #right sound speed

    PR::ParVector1D = ParVector1D{Float64,N+1}() #Left primitive variable 
    PL::ParVector1D = ParVector1D{Float64,N+1}() #Right primitive variable
    FL::ParVector1D = ParVector1D{Float64,N+1}() #Left flux
    FR::ParVector1D = ParVector1D{Float64,N+1}() #Right flux
    F::ParVector1D = ParVector1D{Float64,N+1}() # HLL flux

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
        
        for i in 2:N-1  #Lax Friedrich code
            Ubuffer.arr1[i] =  0.5 * (U1.arr1[i - 1] + U1.arr1[i + 1]) - 0.5 * dt/dx * (F.arr1[i + 1] - F.arr1[i - 1]) 
            Ubuffer.arr2[i] =  0.5 * (U1.arr2[i - 1] + U1.arr2[i + 1]) - 0.5 * dt/dx * (F.arr2[i + 1] - F.arr2[i - 1]) 
            Ubuffer.arr3[i] =  0.5 * (U1.arr3[i - 1] + U1.arr3[i + 1]) - 0.5 * dt/dx * (F.arr3[i + 1] - F.arr3[i - 1]) 
        end
        
        UtoP(Ubuffer,P,eos,kwargs...) #Coversion to primitive variables
        for i in 1:N
            U1.arr1[i] = Ubuffer.arr1[i]
            U1.arr2[i] = Ubuffer.arr2[i]
            U1.arr3[i] = Ubuffer.arr3[i]
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