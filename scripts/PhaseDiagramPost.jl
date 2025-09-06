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
include("HeatMapFunction.jl")

# Set input and output folders
input_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata/k=0.2/g=0.2"


# Make sure output folder exists
mkpath(input_dir)

csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

αs = Array(CSV.read(input_dir * "/" * "Averagealpha.csv", DataFrame))[:,1]
βs = Array(CSV.read(input_dir * "/" * "Averagebeta.csv", DataFrame))[1,:]
Aside = Array(CSV.read(input_dir * "/" * "AverageAside.csv", DataFrame))
Behind = Array(CSV.read(input_dir * "/" * "AverageBehind.csv", DataFrame))

αsize = length(αs)
βsize = length(βs)

plot = Plots.heatmap(αs,βs,Behind',
    title = "Heatmap Test Lag Behind",
    xlabel = "X-axis",
    ylabel = "Y-axis",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(input_dir * "/Lag_Behind.png", plot)
plot

plot = Plots.heatmap(αs,βs,Aside',
    title = "Heatmap Test Lag Aside",
    xlabel = "X-axis",
    ylabel = "Y-axis",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(input_dir * "/Lag_Aside.png", plot)

r2 = sqrt.(Behind.^2 + Aside.^2)'

plot = Plots.heatmap(αs,βs,r2,
    title = "Heatmap Test Lag Aside",
    xlabel = "X-axis",
    ylabel = "Y-axis",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(input_dir * "/Lag_Total.png", plot)

αindices = round.(Int, [1, αsize/4, αsize/2, 3αsize/4, αsize])
αplots   = [fill(αs[i], βsize) for i in αindices]
αr2      = r2[:, αindices]

plot3d(
    [βs for _ in 1:5],
    αplots,
    eachcol(αr2),
    markersize = 5,
    linewidth = 2,
    palette = :plasma,   # or :plasma, :inferno, :coolwarm, etc.
    legend = false,
    cbar = true,
    xlabel = "β Momentum Factor",
    ylabel = "α Tail Index",
    zlabel = "Attentive Lag"
)

βindices = round.(Int, [1, βsize/4, βsize/2, 3βsize/4, βsize])
βplots   = [fill(βs[i], αsize) for i in βindices]
βr2      = r2[βindices, :]

plot3d(
    [αs for _ in 1:5],      # x-axis (α varies)
    βplots,                 # y-axis (β is fixed per line)
    eachrow(βr2),           # z-axis (curves across α)
    markersize = 5,
    linewidth = 2,
    palette = :plasma,      # or :inferno, :viridis, etc.
    legend = false,
    cbar = true,
    xlabel = "α Tail Index",
    ylabel = "β Momentum Factor",
    zlabel = "Attentive Lag"
)
