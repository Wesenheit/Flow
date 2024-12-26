using CairoMakie 
using HDF5
num = 100
idx = 1

min_val = 1000
max_val = -1000
for i in 1:num
    data = h5open("dump"*string(i)*".h5","r")
    if maximum(data["data"][idx,:,:]) > max_val
        global max_val = maximum(data["data"][1,:,:])
    end
    if minimum(data["data"][idx,:,:]) < min_val
        global min_val = minimum(data["data"][1,:,:])
    end
    close(data)
end
println(min_val, " ",max_val)
fig = Figure(size = (1920, 1080))
ax1 = Axis(fig[1, 1], title = "Density", xlabel = "X", ylabel = "Y")
ax2 = Axis(fig[1, 2], title = "Vy", xlabel = "X", ylabel = "Y")

vel_max = 0.3

data = h5open("dump1.h5","r")
dx = data["grid"][1]
dy = data["grid"][2]
_,X_tot,Y_tot = size(data["data"])
X = (0,X_tot * dx)
Y = (0,Y_tot * dy)
gamma = sqrt.(data["data"][3,:,:] .^ 2 +data["data"][4,:,:] .^2 .+ 1.)
hm1 = image!(ax1,X,Y,data["data"][idx,:,:], colorrange = (min_val, max_val), colormap = :viridis)
hm2 = image!(ax2,X,Y,data["data"][4,:,:] ./ gamma, colorrange = (-vel_max, vel_max), colormap = Reverse(:RdYlBu))

close(data)

record(fig, "Jet_MPI.mp4", 1:num; framerate = 3) do i
    println(i)
    data = h5open("dump"*string(i)*".h5","r")
    hm1[3] = data["data"][idx,:,:]
    gamma = sqrt.(data["data"][3,:,:] .^ 2 + data["data"][4,:,:] .^2 .+ 1.)
    hm2[3] = data["data"][4,:,:] ./ gamma
    close(data)
end
