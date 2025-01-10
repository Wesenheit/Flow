using CairoMakie
using HDF5

num = 100
idx = 1

min_val = 10000
max_val = -10000

for i in 1:num
    println("first ", i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5", "r")
    println("Frame ", i, " density stats: min = ", minimum(data["data"][idx, :, :]), ", max = ", maximum(data["data"][idx, :, :]))

    if maximum(data["data"][idx, :, :]) > max_val
        global max_val = maximum(data["data"][idx, :, :])
    end
    if minimum(data["data"][idx, :, :]) < min_val
        global min_val = minimum(data["data"][idx, :, :])
    end
    close(data)
end

println("Raw data range: ", min_val, " ", max_val)


log_min_val = log10(min_val + 1e-8)
log_max_val = log10(max_val + 1e-8)
println("Log-transformed data range: ", log_min_val, " ", log_max_val)

fig = Figure(size = (1920, 1080))
ax1 = Axis(fig[1, 1], title = "Density", xlabel = "X", ylabel = "Y")
ax2 = Axis(fig[1, 2], title = "V_tot", xlabel = "X", ylabel = "Y")

vel_max = 0.1

data = h5open(ARGS[1]*"/dump1.h5", "r")
dx = data["grid"][1]
dy = data["grid"][2]
_, X_tot, Y_tot = size(data["data"])
X = (0, X_tot * dx)
Y = (0, Y_tot * dy)
gamma = sqrt.(data["data"][3, :, :] .^ 2 + data["data"][4, :, :] .^ 2 .+ 1.)


log_density = log10.(data["data"][idx, :, :] .+ 1e-8)  

hm1 = image!(ax1, X, Y, log_density, colorrange = (log_min_val, log_max_val), colormap = :viridis)
hm2 = image!(ax2, X, Y, ((data["data"][4, :, :] ./ gamma).^ 2 + (data["data"][3, :, :] ./ gamma).^ 2).^(1/2), colorrange = (0, vel_max), colormap = :viridis)

close(data)

record(fig, "Collapse_MPI_CUDA.mp4", 1:num; framerate = 3) do i
    println("second ", i)
    data = h5open(ARGS[1]*"/dump"*string(i)*".h5", "r")

    log_density = log10.(data["data"][idx, :, :] .+ 1e-8)
    hm1[3] = log_density
    gamma = sqrt.(data["data"][3, :, :] .^ 2 + data["data"][4, :, :] .^ 2 .+ 1.)
    hm2[3] = ((data["data"][4, :, :] ./ gamma).^ 2 + (data["data"][3, :, :] ./ gamma).^ 2).^(1/2)

    close(data)
end

