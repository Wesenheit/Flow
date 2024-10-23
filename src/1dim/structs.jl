    using StaticArrays

    struct U1D{T <:Real,N}
        # Conserved variables for the solver
        # density * gamma
        # Energy density
        # momentum density
        rho_ut::MVector{N,T}
        T_tt::MVector{N,T}
        T_tx::MVector{N,T}
        size::Int64
        function U1D{T,N}() where {T, N}
            arr1 = MVector{N,T}(zeros(N))
            arr2 = MVector{N,T}(zeros(N))
            arr3 = MVector{N,T}(zeros(N))
            new(arr1,arr2,arr3,N)
        end
    end



    struct P1D{T <:Real,N}
        # Physical properties of the system
        #
        #
        rho::MVector{N,T}
        u::MVector{N,T}
        vx::MVector{N,T}
        size::Int64
        function P1D{T,N}() where {T, N}
            arr1 = MVector{N,T}(zeros(N))
            arr2 = MVector{N,T}(zeros(N))
            arr3 = MVector{N,T}(zeros(N))
            new(arr1,arr2,arr3)
        end
    end

    function PtoU(P::P1D,U::U1D)
        for i in 1:P.size
            gamma = 1/sqrt(1-P.vx^2)
            
        end
    end