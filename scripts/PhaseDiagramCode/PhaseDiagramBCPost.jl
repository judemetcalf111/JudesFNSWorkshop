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
using ColorSchemes

# Set input and output folders
input_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata/BCircular"

# Make sure output folder exists
mkpath(input_dir)

bloop = true
while bloop
    print("Beta Value?: ")
    local b_cand = readline()
    b_cand = parse(Float64, b_cand)
    if isdir(input_dir * "/b=$(b_cand)")
        println("Directories for β = $(b_cand) exist!")
        global b_value = b_cand
        global bloop = false
    else
        println("No such directories for β = $(b_cand)")
    end
end

γloop = true
while γloop
    print("Gamma Value?: ")
    local γ_cand = readline() 
    γ_cand = parse(Float64, γ_cand)
    if isdir(input_dir * "/b=$(b_value)/g=$(γ_cand)")
        println("Directories for β = $(b_value) and γ = $(γ_cand) exist! \n
        Let's run...")
        global γ_value = γ_cand
        global γloop = false
    else
        println("No such directories for β = $(b_value) and γ = $(γ_cand)")
    end
end

input_dir = input_dir * "/b=$(b_value)/g=$(γ_value)"
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

αs = Array(CSV.read(input_dir * "/" * "Averagealpha.csv", DataFrame))[:,1]
ks = Array(CSV.read(input_dir * "/" * "Averagek.csv", DataFrame))[1,:]
rs = Array(CSV.read(input_dir * "/" * "Averager.csv", DataFrame))

αsize = length(αs)
ksize = length(ks)

plotr = Plots.heatmap(αs,ks,rs',
    title = "Heatmap Test Lag Behind: β = $(b_value)",
    xlabel = "α Tail Index",
    ylabel ="k Speed Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Value"
)

save(input_dir * "/Mean(r)Phase.png", plotr)

display(plotr)


# five indicative samples of α
αvalues = [2.,1.8,1.5,1.3,1.2]
αindices = round.(Int, [findfirst(==(val), αs) for val in αvalues]) 
colours = get(ColorSchemes.thermal, range(0, .8, length=length(αvalues)))

TotalSweepPlot = Plots.plot()

for (alphin, alphval) in enumerate(αvalues)

    Plots.plot!(ks, rs[αindices[alphin],:],
        label = "α = $(alphval)",
        xlabel = "β Momentum Factor",
        ylabel = "Lag Total",
        title = "Predictive Lag Total vs 'k' Speed for γ = $(γ_value)",
        linecolor = colours[alphin], 
        linewidth = 3,
        legend = :topright
    )
end

save(input_dir * "/TotalSweepPlot.png", TotalSweepPlot)

display(TotalSweepPlot)



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
#     xlabel = "α Tail Index",
#     ylabel ="β Momentum Factor",
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
