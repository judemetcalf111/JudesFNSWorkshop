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
input_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata"

# Make sure output folder exists
mkpath(input_dir)

kloop = true
while kloop
    print("Frequency k Value?: ")
    local k_cand = readline()
    k_cand = parse(Float64, k_cand)
    if isdir(input_dir * "/k=$(k_cand)")
        println("Directories for k = $(k_cand) exist!")
        global k_value = k_cand
        global kloop = false
    else
        println("No such directories for k = $(k_cand)")
    end
end

γloop = true
while γloop
    print("Gamma Value?: ")
    local γ_cand = readline()
    γ_cand = parse(Float64, γ_cand)
    if isdir(input_dir * "/k=$(k_value)/g=$(γ_cand)")
        println("Directories for k = $(k_value) and γ = $(γ_cand) exist! \n
        Let's run...")
        global γ_value = γ_cand
        global γloop = false
    else
        println("No such directories for k = $(k_value) and γ = $(γ_cand)")
    end
end

input_dir = input_dir * "/k=$(k_value)/g=$(γ_value)"
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

αs = Array(CSV.read(input_dir * "/" * "Averagealpha.csv", DataFrame))[:,1]
βs = Array(CSV.read(input_dir * "/" * "Averagebeta.csv", DataFrame))[1,:]
Aside = Array(CSV.read(input_dir * "/" * "AverageAside.csv", DataFrame))
Behind = Array(CSV.read(input_dir * "/" * "AverageBehind.csv", DataFrame))

αsize = length(αs)
βsize = length(βs)

plotBehind = Plots.heatmap(αs,βs,Behind',
    title = "Behind Lag: k = $(k_value)",
    xlabel = "α Tail Index",
    ylabel ="β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    # colorbar_title = "Lag",
    guidefontsize = 18,
    tickfontsize = 16,
    titlefontsize = 20,
    dpi = 300
)

save(input_dir * "/Lag_Behind.pdf", plotBehind)
plotBehind

plotAside = Plots.heatmap(αs,βs,abs.(Aside'),
    title = "Aside Lag: k = $(k_value)",
    xlabel = "α Tail Index",
    ylabel ="β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    # colorbar_title = "Lag",
    guidefontsize = 18,
    tickfontsize = 16,
    titlefontsize = 20,
    dpi = 300
)

save(input_dir * "/Lag_Aside.pdf", plotAside)

r2 = sqrt.(Behind.^2 + Aside.^2)'

plotTotal = Plots.heatmap(αs,βs,r2,
    title = "Total Lag: k = $(k_value)",
    xlabel = "α Tail Index",
    ylabel ="β Momentum Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    # colorbar_title = "Lag",
    guidefontsize = 18,
    tickfontsize = 16, 
    titlefontsize = 20,
    dpi = 300
)

savefig(plotTotal,input_dir * "/Lag_Total.pdf")

# five indicative samples of α
αvalues = [2,1.98,1.5,1.3,1.2]
αindices = round.(Int, [findfirst(==(val), αs) for val in αvalues]) 
colours = get(ColorSchemes.thermal, range(0, .8, length=length(αvalues)))

BehindSweepPlot = Plots.plot(dpi = 300)

for (alphin, alphval) in enumerate(αvalues)
    BehindSweepPlot = Plots.plot!(βs, Behind'[:, αindices[alphin]],
        label = "α = $(alphval)",
        xlabel = "β Momentum Factor",
        ylabel = "Lag Behind",
        title = "Lag Behind vs β for k = $(k_value) and γ = $(γ_value)",
        linecolor = colours[alphin], 
        linewidth = 3,
        legend = :topright,
        legendfontsize = 16,
        guidefontsize = 16,
        tickfontsize = 14 
    )
end

save(input_dir * "/BehindSweepPlot.pdf", BehindSweepPlot)

display(BehindSweepPlot)

AsideSweepPlot = Plots.plot(dpi = 300)
for (alphin, alphval) in enumerate(αvalues)

    AsideSweepPlot = Plots.plot!(βs, abs.(Aside'[:, αindices[alphin]]),
        label = "α = $(alphval)",
        xlabel = "β Momentum Factor",
        ylabel = "Lag Aside",
        title = "Lag Aside vs β for k = $(k_value) and γ = $(γ_value)",
        linecolor = colours[alphin], 
        linewidth = 3,
        legend = :topright,
        legendfontsize = 16,
        guidefontsize = 16,
        tickfontsize = 14 
    )
end

save(input_dir * "/AsideSweepPlot.pdf", AsideSweepPlot)

display(AsideSweepPlot)

TotalSweepPlot = Plots.plot(dpi = 300)

for (alphin, alphval) in enumerate(αvalues)

    TotalSweepPlot = Plots.plot!(βs, r2[:, αindices[alphin]],
        label = "α = $(alphval)",
        xlabel = "β Momentum Factor",
        ylabel = "Lag Total",
        title = "Lag Total vs β for k = $(k_value) and γ = $(γ_value)",
        linecolor = colours[alphin], 
        linewidth = 3,
        legend = :topright,
        legendfontsize = 16,
        guidefontsize = 16,
        tickfontsize = 14
    )
end

save(input_dir * "/TotalSweepPlot.pdf", TotalSweepPlot)

display(TotalSweepPlot)
