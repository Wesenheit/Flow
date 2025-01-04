function SyncBoundaryX(U::FlowArr,comm,buff_X_1::AbstractArray,buff_X_2::AbstractArray)
    buff_X_1 .= @view U.arr[:,end-1,:]    
    buff_X_2 .= @view U.arr[:,1,:]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,0,1)
    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,0,-1)
    MPI.Sendrecv!(buff_X_1,rank_source_right,0,buff_X_2,rank_dest_right,0,comm)
    U.arr[:,1,:] .= buff_X_2

    buff_X_1 .= @view U.arr[:,2,:]
    buff_X_2 .= @view U.arr[:,end,:]

    MPI.Sendrecv!(buff_X_1,rank_source_left,1,buff_X_2,rank_dest_left,1,comm)
    U.arr[:,end,:] .= buff_X_2
end

function SyncBoundaryY(U::FlowArr,comm,buff_Y_1::AbstractArray,buff_Y_2::AbstractArray)
    buff_Y_1 .= @view U.arr[:,:,end-1]
    buff_Y_2 .= @view U.arr[:,:,1]

    rank_source_up,rank_dest_up = MPI.Cart_shift(comm,1,1)
    rank_source_down,rank_dest_down = MPI.Cart_shift(comm,1,-1)


    MPI.Sendrecv!(buff_Y_1,rank_source_up,0,buff_Y_2,rank_dest_up,0,comm)
    U.arr[:,:,1] .= buff_Y_2

    buff_Y_1 .= @view U.arr[:,:,2]
    buff_Y_2 .= @view U.arr[:,:,end]


    MPI.Sendrecv!(buff_Y_1,rank_source_down,1,buff_Y_2,rank_dest_down,1,comm)

    U.arr[:,:,end] = buff_Y_2 
end


function SyncFlux_X_Left(PL::FlowArr,comm,buff_X_1::AbstractArray,buff_X_2::AbstractArray)
    #we send the left flux to the right boundary
    buff_X_1 .= @view PL.arr[:,end-1,:]
        
    buff_X_2 .= @view PL.arr[:,1,:]

    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,0,-1)

    MPI.Sendrecv!(buff_X_1,rank_source_left,1,buff_X_2,rank_dest_left,1,comm)

    PL.arr[:,1,:] .= buff_X_2
end

function SyncFlux_X_Right(PR::FlowArr,comm,buff_X_1::AbstractArray,buff_X_2::AbstractArray)
    #we send the right flux to the left boundary
    buff_X_1 .= @view PR.arr[:,1,:]
        
    buff_X_2 .= @view PR.arr[:,end-1,:]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,0,1)

    MPI.Sendrecv!(buff_X_1,rank_source_right,0,buff_X_2,rank_dest_right,0,comm)

    PR.arr[:,end-1,:] .= buff_X_2
end

function SyncFlux_Y_Down(PD::FlowArr,comm,buff_Y_1::AbstractArray,buff_Y_2::AbstractArray)
    #we send the left flux to the right boundary
    buff_Y_1 .= @view PD.arr[:,:,end-1]
        
    buff_Y_2 .= @view PD.arr[:,:,1]

    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,1,-1)

    MPI.Sendrecv!(buff_Y_1,rank_source_left,1,buff_Y_2,rank_dest_left,1,comm)

    PD.arr[:,:,1] .= buff_Y_2
end

function SyncFlux_Y_Up(PU::FlowArr,comm,buff_Y_1::AbstractArray,buff_Y_2::AbstractArray)
    #we send the right flux to the left boundary
    buff_Y_1 .= @view PU.arr[:,:,1]
        
    buff_Y_2 .= @view PU.arr[:,:,end-1]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,1,1)

    MPI.Sendrecv!(buff_Y_1,rank_source_right,0,buff_Y_2,rank_dest_right,0,comm)

    PU.arr[:,:,end-1] .= buff_Y_2
end
