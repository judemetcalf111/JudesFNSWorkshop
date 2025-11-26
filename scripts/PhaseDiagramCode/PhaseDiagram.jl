using DrWatson
@quickactivate("JudesFNSWorkshop")
using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Base.Threads
using CairoMakie
using FileIO
using ProgressBars
using Infiltrator
using Plots
# include("HeatMapFunction.jl")

k = 0.4
g = 0.1

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/phasedata/k=$(k)/g=$(g)"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata/k=$(k)/g=$(g)"

# Make sure output folder exists
mkpath(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
xmax = 7

kval = k

αs = collect(1.02:0.02:2.00)
# b0 = log10(.01)
# b1 = log10(100)
βs = collect(0.01:0.01:1.00)

αsize = length(αs)
βsize = length(βs)
twoDsize = αsize * βsize

# Create a 3x4x5 array of Float64, uninitialized
Results_Array = Array{Float64}(undef, αsize, βsize, 4)
rs = Array{Float64}(undef, αsize, βsize)
for (index, alph) in  enumerate(αs)
    Results_Array[index,:,2] = βs
end

for (index, bet) in enumerate(βs)
    Results_Array[:,index,1] = αs
end

center(k,t) = (xmax ./ 2) .* exp.( im .* k .* t)

centerangle(k,t) = exp.( im .* k .* t)
centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))

for file in ProgressBar(csv_files)

    filename = splitext(basename(file))[1]

    # Extract k parameter using regex
    av = match(r"a=([\d.]+)", filename)
    bv = match(r"b=([\d.]+)", filename)

    if av !== nothing
        aval = parse(Float64, av.captures[1])
    else
        println("No match found")
    end

    if bv !== nothing
        bval = parse(Float64, bv.captures[1])
    else
        println("No match found")
    end


    local aindex = findfirst(x -> x == aval, Results_Array[:,1,1])
    local bindex = findfirst(x -> x == bval, Results_Array[1,:,2])

    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    timestamp = collect(df.timestamp)
    x = collect(df.value1)
    y = collect(df.value2)
    lengthofdata = size(timestamp)

    xshift = zeros(lengthofdata)
    yshift = zeros(lengthofdata)
    xpos = zeros(lengthofdata)
    ypos = zeros(lengthofdata)

    for (index,xvalue) in enumerate(x)
        xshift[index] = xvalue - real(center(kval,timestamp[index]))
    end
    for (index,yvalue) in enumerate(y)
        yshift[index] = yvalue - imag(center(kval,timestamp[index]))
    end

    for (index,timevalue) in enumerate(timestamp)
        ypos[index] = (xshift[index] * real(centerangle(kval,timevalue))) + (yshift[index] * imag(centerangle(kval,timevalue)) )
        xpos[index] = (xshift[index] * real(centerangle90(kval,timevalue))) + (yshift[index] * imag(centerangle90(kval,timevalue)) )
    end


    meanx = mean(xpos)
    meany = mean(ypos)
    # println(bindex)
    # println(aindex)
    Results_Array[aindex, bindex,3:4] = [meanx,meany]
    rs[aindex,bindex] = meany #sqrt(mean(abs2.([meanx,meany])))
end

plot = Plots.heatmap(αs,βs,-Results_Array[:,:,3]',
    title = "Heatmap Test Lag Behind",
    xlabel = "α Tail Index",
    ylabel = "β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(output_dir * "/Lag_Behind.png", plot)

plot = Plots.heatmap(αs,βs,Results_Array[:,:,4]',
    title = "Heatmap Test Lag Aside",
    xlabel = "α Tail Index",
    ylabel = "β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(output_dir * "/Lag_Aside.png", plot)

r2 = sqrt.(Results_Array[:,:,3].^2 + Results_Array[:,:,4].^2)'

plot = Plots.heatmap(αs,βs,r2,
    title = "Heatmap Test Lag Total",
    xlabel = "α Tail Index",
    ylabel = "β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(output_dir * "/Lag_Total.png", plot)

# plot = Plots.plot(βs,r2[:,1],
#     title = "Total Attentive Lag for α = $(αs[1])",
#     xlabel = "β Momentum Factor"
#     ylabel = "Total Lag"
# )

# αindices = round.(Int, [1, αsize/4, αsize/2, 3αsize/4, αsize])
# αplots   = [fill(αs[i], βsize) for i in αindices]
# αr2      = r2[:, αindices]

# plot3d(
#     [βs for _ in 1:5],
#     αplots,
#     eachcol(αr2),
#     markersize = 5,
#     linewidth = 2,
#     palette = :plasma,   # or :plasma, :inferno, :coolwarm, etc.
#     legend = false,
#     cbar = true,
#     xlabel = "β Momentum Factor",
#     ylabel = "α Tail Index",
#     zlabel = "Attentive Lag"
# )

# βindices = round.(Int, [1, βsize/4, βsize/2, 3βsize/4, βsize])
# βplots   = [fill(βs[i], αsize) for i in βindices]
# βr2      = r2[βindices, :]

# plot3d(
#     [αs for _ in 1:5],      # x-axis (α varies)
#     βplots,                 # y-axis (β is fixed per line)
#     eachrow(βr2),           # z-axis (curves across α)
#     markersize = 5,
#     linewidth = 2,
#     palette = :plasma,      # or :inferno, :viridis, etc.
#     legend = false,
#     cbar = true,
#     xlabel = "α Tail Index",
#     ylabel = "β Momentum Factor",
#     zlabel = "Attentive Lag"
# )

# αindices = round.(Int, [1, αsize/4, αsize/2, 3αsize/4, αsize])
# αplots   = [fill(αs[i], βsize) for i in αindices]
# αr2      = r2[:, αindices]

# plot3d(
#     [βs for _ in 1:5],
#     αplots,
#     eachcol(αr2),
#     markersize = 5,
#     linewidth = 2,
#     palette = :plasma,   # or :plasma, :inferno, :coolwarm, etc.
#     legend = false,
#     cbar = true,
#     xlabel = "β Momentum Factor",
#     ylabel = "α Tail Index",
#     zlabel = "Attentive Lag"
# )

# βindices = round.(Int, [1, βsize/4, βsize/2, 3βsize/4, βsize])
# βplots   = [fill(βs[i], αsize) for i in βindices]
# βr2      = r2[βindices, :]

# plot3d(
#     [αs for _ in 1:5],      # x-axis (α varies)
#     βplots,                 # y-axis (β is fixed per line)
#     eachrow(βr2),           # z-axis (curves across α)
#     markersize = 5,
#     linewidth = 2,
#     palette = :plasma,      # or :inferno, :viridis, etc.
#     legend = false,
#     cbar = true,
#     xlabel = "α Tail Index",
#     ylabel = "β Momentum Factor",
#     zlabel = "Attentive Lag"
# )


CSV.write(output_dir * "/AverageBehind.csv", Tables.table(-Results_Array[:,:,3]))
CSV.write(output_dir * "/AverageAside.csv", Tables.table(Results_Array[:,:,4]))
CSV.write(output_dir * "/Averagealpha.csv", Tables.table(Results_Array[:,:,1]))
CSV.write(output_dir * "/Averagebeta.csv", Tables.table(Results_Array[:,:,2]))
CSV.write(output_dir * "/Averager2.csv", Tables.table(r2))
