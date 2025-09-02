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

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
xmax = 7

center(k,t) = (xmax ./ 2) .* exp.( im .* k .* t)

centerangle(k,t) = exp.( im .* k .* t)
centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))


for file in csv_files
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

    filename = splitext(basename(file))[1]
    # Extract k parameter using regex
    m = match(r"k=([\d.]+)", filename)
    av = match(r"a=([\d.]+)", filename)
    bv = match(r"b=([\d.]+)", filename)

    if m !== nothing
        kval = parse(Float64, m.captures[1])
    else
        println("No match found")
    end

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

    include("HeatMapFunction.jl")

    xbins,ybins,heatdatashift = heatmapconvert(xshift, yshift; xminplot=-xmax, xmaxplot=xmax, yminplot=-xmax, ymaxplot=xmax, bins=200)

    # Plot heatmap
    figshift = Figure()
    ax = Axis(figshift[1,1], xlabel="x", ylabel="y", title = "Walker Position Heatmap with Respect to Moving Well")
    hm = CairoMakie.heatmap!(ax, ybins, xbins, log1p.(heatdatashift)', colormap=:viridis)
    Colorbar(figshift[1,2], hm)
    figshift

    filepath = "/a = $(aval)/b = $(bval)/k = $(kval)/" 

    mkpath(output_dir * filepath)

    # Build output path with same base filename but PNG extension
    outname = joinpath(output_dir * filepath, splitext(basename(file))[1] * "-MovingFrame" * ".png")

    # Save figure
    save(outname, figshift)


    xbins,ybins,heatdatapos = heatmapconvert(xpos, ypos; xminplot=-xmax, xmaxplot=xmax, yminplot=-xmax, ymaxplot=xmax, bins=200)

    # Plot heatmap
    figpos = Figure()
    ax = Axis(figpos[1,1], xlabel="Radial Position 'r'", ylabel="Angular Position Tangent to Radius 's'", title = "Walker Position Heatmap in the Coordinates of Moving Well")
    hm = CairoMakie.heatmap!(ax, ybins, xbins, log1p.(heatdatapos)', colormap=:viridis)
    Colorbar(figpos[1,2], hm)

    # Build output path with same base filename but PNG extension
    outname = joinpath(output_dir * filepath, splitext(basename(file))[1] * "-CoordinateFrame" * ".png")

    # Save figure
    save(outname, figpos)

    println("Saved heatmap for $filename")
end
