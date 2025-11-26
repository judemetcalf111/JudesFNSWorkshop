using DrWatson
@quickactivate("JudesFNSWorkshop")
using CSV
using Random
using DataFrames
using GLM
using Statistics
using StatsBase
using Plots
using Base.Threads
using CairoMakie
using FileIO

include(srcdir("JudesFNSWorkshop.jl"))

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> startswith(f,input_dir * "/tfnsBrownian") && endswith(f, ".csv"), readdir(input_dir; join=true))

σ = 1
δt = 0.001

for file in csv_files
    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    timestamp = collect(df.timestamp)
    x = collect(df.value1)
    y = collect(df.value2)
    lengthofdata = length(timestamp)

    xshift = zeros(lengthofdata)
    yshift = zeros(lengthofdata)
    xpos = zeros(lengthofdata)
    ypos = zeros(lengthofdata)

    filename = splitext(basename(file))[1]
    # Extract k parameter using regex
    xmaxcand = match(r"xmax=([\d.]+)", filename)
    m = match(r"k=([\d.]+)", filename)
    av = match(r"a=([\d.]+)", filename)
    bv = match(r"b=([\d.]+)", filename)

    xmaxcand = match(r"xmax=([\d.]+)", filename)
    if xmaxcand == nothing
        xmax = 7
    else
        xmax = parse(Float64, xmaxcand.captures[1])
    end

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
    wellseed = 49
    Random.seed!(wellseed)
    wellsrandomgen = rand(Normal(0, σ * √δt),Int(lengthofdata+1))
    wellsrandom = cumsum(wellsrandomgen)

    center(k,t) = (xmax ./ 2) .* exp.( im * k * wellsrandom[round(Int,(t/δt + 1))] )

    for (index,xvalue) in enumerate(x)
        xshift[index] = xvalue - real(center(kval,timestamp[index]))
    end
    for (index,yvalue) in enumerate(y)
        yshift[index] = yvalue - imag(center(kval,timestamp[index]))
    end

    xbins,ybins,heatdatashift = JudesFNSWorkshop.heatmapconvert(xshift, yshift; xminplot=-xmax, xmaxplot=xmax, yminplot=-xmax, ymaxplot=xmax, bins=200)

    # Plot heatmap
    figshift = Figure()
    ax = Axis(figshift[1,1], xlabel="x", ylabel="y", title = "Walker Position Heatmap with Respect to Moving Well")
    hm = CairoMakie.heatmap!(ax, ybins, xbins, log1p.(heatdatashift)', colormap=:viridis)
    Colorbar(figshift[1,2], hm)
    figshift

    filepath = "/Brownian_a=$(aval)/b=$(bval)/k=$(kval)/" 

    mkpath(output_dir * filepath)

    # Build output path with same base filename but PNG extension
    outname = joinpath(output_dir * filepath, splitext(basename(file))[1] * ".png")

    # Save figure
    save(outname, figshift)

    println("Saved heatmap for $filename")

    local r = sqrt.(xshift.^2 + yshift.^2)

    local binnumber = 1000
    local min_val = minimum(r)
    local max_val =  mean(r) * 2.
    local binsrange = range(min_val, max_val, length=binnumber)
    local rhist = fit(Histogram, r[r .< max_val], binsrange)

    local counts = rhist.weights
    local edges = rhist.edges[1]

    local bin_midpoints = (edges[1:end-1] .+ edges[2:end]) ./ 2
    local bin_width = step(binsrange)

    local scaling_factor_2D = bin_midpoints
    local density_unnormalized = counts ./ (scaling_factor_2D .+ 1e-8)

    # Normalize ρ(r) such that the total probability ∫ρ(r)dV = 1.
    # We approximate the integral: Σ [ρ(r) * (4πr²) * dr]
    local integral_approx = sum(density_unnormalized .* (2π * bin_midpoints) .* bin_width)
    local prob_density = density_unnormalized ./ integral_approx


    # Xtragram = Plots.histogram(
    #     rhist,
    #     normalize = :pdf,
    #     bins = binsrange,
    #     # xscale = :log10,
    #     xlabel = "Distance from Centre",
    #     ylabel = "Frequency",
    #     xlim = [0,max_val],
    #     c = "purple",
    #     title = "Radial Square Distance Distribution: α = $(aval), β = $(bval), k = $(kval)",
    #     minorgrid = true,
    #     legend = false)

    trendx =  bin_midpoints[ bin_midpoints .< (mean(r) * 2)]
    trendy = prob_density[ bin_midpoints .< (mean(r) * 2) ]

    df = DataFrame(X = trendx, Y = log.(trendy))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    fitline = exp.((bin_midpoints .* Coeffs[2]) .+ Coeffs[1])

    XtragramNEW = Plots.plot(bin_midpoints, prob_density,
        label="Rescaled Density ρ(r)",
        st=:steppre, # Use a stepped line to resemble a histogram
        xlabel = "Distance from Centre",
        ylabel = "Frequency",
        # xscale = :log10,
        yscale = :log10,
        xlim = [min_val,max_val],
        title = "Radial Density: α = $(aval), β = $(bval), k = $(kval)",
        minorgrid = true,
        lw=2.5,
        c=:blue)

    Plots.vline!([mean(r)], c = "red", label = "mean")
    Plots.plot!(bin_midpoints,fitline, c = :purple, label = "exp($(round(Coeffs[2], digits=3))×r)")

    println("$(aval) and $(mean(r)) and $(round(Coeffs[2], digits=3))")
    display(XtragramNEW)

    outname = joinpath(output_dir * filepath, "LoggyRDist-" * splitext(basename(file))[1] * ".pdf")

    save(outname, XtragramNEW)

    XtragramLINEAR = Plots.plot(bin_midpoints, prob_density,
        label="Rescaled Density ρ(r)",
        st=:steppre, # Use a stepped line to resemble a histogram
        xlabel = "Distance from Centre",
        ylabel = "Frequency",
        # xscale = :log10,
        # yscale = :log10,
        xlim = [min_val,max_val],
        title = "Radial Density: α = $(aval), β = $(bval), k = $(kval)",
        minorgrid = true,
        lw=2.5,
        c=:blue)

    Plots.vline!([mean(r)], c = "red", label = "mean")
    Plots.plot!(bin_midpoints,fitline, c = :purple, label = "exp($(round(Coeffs[2], digits=3))×r)")

    println("$(aval) and $(mean(r)) and $(round(Coeffs[2], digits=3))")
    display(XtragramLINEAR)

    outname = joinpath(output_dir * filepath, "LinearRDist-" * splitext(basename(file))[1] * ".pdf")

    save(outname, XtragramLINEAR)

end
