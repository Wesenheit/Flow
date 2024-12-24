using CairoMakie 
using HDF5
num=1800
idx = 3

min_val = 1000
max_val = -1000
"""
for i in 1:num
    data = h5open("dump"*string(i)*".h5")
    if maximum(data["data"][idx,:,:]) > max_val
        global max_val = maximum(data["data"][1,:,:])
    end
    if minimum(data["data"][idx,:,:]) < min_val
        global min_val = minimum(data["data"][1,:,:])
    end
    close(data)
end
"""
min_val = -1
max_val = 1

fig = Figure(size = (1920, 1080))
ax = Axis(fig[1, 1], title = "Kelvin-Helmholtz instability", xlabel = "X", ylabel = "Y")

data = h5open("dump1.h5","r")
hm = heatmap!(ax, data["data"][idx,:,:], colorrange = (-1.0, 1.0), colormap = :viridis)

close(data)

record(fig, "KH_MPI.mp4", 1:num; framerate = 10) do i
    println(i)
    data = h5open("dump"*string(i)*".h5","r")
    hm[1] = data["data"][idx,:,:]
    close(data)
end
