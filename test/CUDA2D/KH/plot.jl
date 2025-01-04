using CairoMakie 
using HDF5
num = 100
idx = 3

min_val = 1000
max_val = -1000
for i in 1:num
    data = h5open("dump"*string(i)*".h5","r")
    if maximum(data["data"][idx,:,:]) > max_val
        global max_val = maximum(data["data"][idx,:,:])
    end
    if minimum(data["data"][idx,:,:]) < min_val
        global min_val = minimum(data["data"][idx,:,:])
    end
    close(data)
end

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1], title = "Kelvin-Helmholtz instability", xlabel = "X", ylabel = "Y")

data = h5open("dump1.h5","r")
hm = heatmap!(ax, data["data"][idx,:,:], colorrange = (min_val, max_val), colormap = :viridis)
println(min_val," ",max_val)
close(data)

record(fig, "KH_MPI_CUDA.mp4", 1:num; framerate = 10) do i
    println(i)
    data = h5open("dump"*string(i)*".h5","r")
    hm[1] = data["data"][idx,:,:]
    close(data)
end
