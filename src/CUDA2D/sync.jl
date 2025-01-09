function SyncBoundaryX(U::FlowArr,comm,buff_X_1::AbstractArray,buff_X_2::AbstractArray)
    buff_X_1 .= @view U.arr[:,end-3:end-2,:]    
    buff_X_2 .= @view U.arr[:,1:2,:]

    rank_source_right,rank_dest_right = MPI.Cart_shift(comm,0,1)
    rank_source_left,rank_dest_left = MPI.Cart_shift(comm,0,-1)
    MPI.Sendrecv!(buff_X_1,rank_source_right,0,buff_X_2,rank_dest_right,0,comm)
    U.arr[:,1:2,:] .= buff_X_2

    buff_X_1 .= @view U.arr[:,3:4,:]
    buff_X_2 .= @view U.arr[:,end-1:end,:]

    MPI.Sendrecv!(buff_X_1,rank_source_left,1,buff_X_2,rank_dest_left,1,comm)
    U.arr[:,end-1:end,:] .= buff_X_2
end

function SyncBoundaryY(U::FlowArr,comm,buff_Y_1::AbstractArray,buff_Y_2::AbstractArray)
    buff_Y_1 .= @view U.arr[:,:,end-3:end-2]
    buff_Y_2 .= @view U.arr[:,:,1:2]

    rank_source_up,rank_dest_up = MPI.Cart_shift(comm,1,1)
    rank_source_down,rank_dest_down = MPI.Cart_shift(comm,1,-1)


    MPI.Sendrecv!(buff_Y_1,rank_source_up,0,buff_Y_2,rank_dest_up,0,comm)
    U.arr[:,:,1:2] .= buff_Y_2

    buff_Y_1 .= @view U.arr[:,:,3:4]
    buff_Y_2 .= @view U.arr[:,:,end-1:end]


    MPI.Sendrecv!(buff_Y_1,rank_source_down,1,buff_Y_2,rank_dest_down,1,comm)

    U.arr[:,:,end-1:end] = buff_Y_2 
end

