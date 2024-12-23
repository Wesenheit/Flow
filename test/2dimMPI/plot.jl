using Plots 
using HDF5
num=99
idx = 3

min_val = 1000
max_val = -1000
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
min_val = -1
max_val = 1
anim = @animate for i in 1:num
    data = h5open("dump"*string(i)*".h5")

    p = Plots.heatmap(data["data"][idx,:,:], xlabel="x", ylabel="y", color=:viridis, 
                      clims=(min_val, max_val))
    close(data)
end

gif(anim,"test.gif", fps=10)
