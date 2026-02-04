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
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/longanalysis"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
xmax = 7
variance = 1
center(k,t) = (xmax ./ 2) .* exp.( im .* k .* t)

centerangle(k,t) = exp.( im .* k .* t)
centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))

αloop = true
while αloop
    print("Alpha Value?: ")
    local α_cand = readline()
    α_cand = parse(Float64, α_cand)
    if isdir(input_dir * "/a=$(α_cand)")
        println("Directories for α = $(α_cand) exist!")
        global α_value = α_cand
        global αloop = false
    else
        println("No such directories for α = $(α_cand)")
    end
end

βloop = true
while βloop 
    print("Beta Value?: ")
    local β_cand = readline()
    β_cand = parse(Float64, β_cand)
    if isdir(input_dir * "/a=$(α_value)/b=$(β_cand)")
        println("Directories for α = $(α_value) and β = $(β_cand) exist!")
        global β_value = β_cand
        global βloop = false
    else
        println("No such directories for α = $(α_value) and β = $(β_cand)")
    end
end

kloop = true
while kloop
    print("Frequency 'k' Value?: ")
    local k_cand = readline()
    k_cand = parse(Float64, k_cand)
    if isdir(input_dir * "/a=$(α_value)/b=$(β_value)/k=$(k_cand)")
        println("Directories for α = $(α_value), β = $(β_value), and k = $(k_cand) exist! \n
        Let's run...")
        global k_value = k_cand
        global kloop = false
    else
        println("No such directories for α = $(α_value), β = $(β_value), and k = $(k_cand)")
    end
end

Deltas = Vector{Vector{Float64}}()

csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir * "/a=$(α_value)/b=$(β_value)/k=$(k_value)"; join=true))


for file in csv_files
    # Read CSV into DataFrame
    local df = CSV.read(file, DataFrame)

    timestamp = collect(df.timestamp)
    local x = collect(df.value1)
    local y = collect(df.value2)
    lengthofdata = length(timestamp)

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
        global kval = parse(Float64, m.captures[1])
    else
        println("No match found")
    end

    if av !== nothing
        global aval = parse(Float64, av.captures[1])
    else
        println("No match found")
    end

    if bv !== nothing
        global bval = parse(Float64, bv.captures[1])
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

    ##
    ## Loop to find the Δt array, to do statistics on the runs and find the distribution of
    ## the saccades of the walker away from the attentive well
    ##

    deltanow = Vector{Float64}()
    local x0 = mean(xpos)
    local y0 = mean(ypos)

    for i in 1:lengthofdata
        # Calculate squared distance once per loop
        r_squared = (xpos[i]-x0)^2 + (ypos[i]-y0)^2

        if r_squared > variance
            # --- We are IN a run ---
            # Append the squared distance to the current run list.
            push!(deltanow, r_squared)
        else
            # --- We are OUTSIDE a run (or a run just ended) ---
            # If deltanow has values, it means a run just finished.
            if !isempty(deltanow)
                # Add the completed run to our main list.
                push!(Deltas, deltanow)
                # Reset deltanow for the next run.
                deltanow = Vector{Float64}()
            end
        end
    end

    # --- 3. Final Check: Handle the edge case ---
    # If the loop finishes while we are still in a run, we need to save the last deltanow.
    if !isempty(deltanow)
        push!(Deltas, deltanow)
    end

    # --- 4. Display the result ---
    println("Variance (Radius) = $variance, so Variance = $(variance)\n")
    println("Detected Runs (Deltas):")
    # Use a loop to print for clarity
    for (i, run) in enumerate(Deltas)
        println("  Run #$i: $(length(run))")
    end
end

Lengths = [length(run) for run in Deltas]
LengthsTrunc = [size for size in Lengths if 50 < size < 5000]

# 2. Plot the histogram.
using StatsBase, Plots


binnumber = 200


min_val = floor(log10(minimum(LengthsTrunc)))
max_val = ceil(log10(maximum(LengthsTrunc)))
binslog10 = 10 .^ (range(min_val, max_val, length=binnumber))

h = fit(Histogram, LengthsTrunc, binslog10)

# Bin edges
edges = h.edges[1]

bin_widths = diff(h.edges[1])
pdf_vals = h.weights ./ (sum(h.weights) .* bin_widths)


# Bin centers (geometric mean for log bins is better than arithmetic mean)
centers = sqrt.(edges[1:end-1] .* edges[2:end])

# Bin counts
counts = h.weights

# Filter out zero-count bins
mask = pdf_vals .> 0
global df = DataFrame(x=log10.(centers[mask]), y=log10.(pdf_vals[mask]))
global x = df.x
global y = df.y
model = lm(@formula(y ~ x), df)

intercept, slope = coef(model)  # intercept and slope

xplot = 10 .^ df.x
yplot = 10 .^ ((slope .* df.x) .+ intercept)

xann = 1000.0
yann = 2 * (10 ^ ((slope * xann) + intercept))
eqn_string = "~ Δt^($(round(slope, digits=3)))"

ΔTHist = Plots.histogram(
    LengthsTrunc,
    normalize = :pdf, # Or :none for raw counts
    bins = binslog10,       # You can adjust the number of bins
    xlim = extrema(binslog10),
    xscale = :log10,
    yscale = :log10,
    xlabel = "Δt (Number of Steps)",
    ylabel = "Frequency",
    title = "Jump Durations: α = $(aval), k = $(kval), Var = $(variance)",
    minorgrid = true,
    legend = true,
    label = "Data"
)

Plots.plot!(xplot,yplot,
            xscale = :log10,
            yscale = :log10, 
            c = "blue", linewidth = 2, label = eqn_string)

save(input_dir * "/a=$(α_value)/b=$(β_value)/k=$(k_value)/" * "DeltaTPlot.pdf",ΔTHist)
# annotate!(xann, yann, xscale = :log10, yscale = :log10, Plots.text(eqn_string, :firebrick, 3, :left))
