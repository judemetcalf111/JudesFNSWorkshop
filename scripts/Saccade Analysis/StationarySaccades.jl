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
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/SaccadeData"
xmax = 7

# center(k,t) = (xmax ./ 2) .* exp.( im .* k .* t)

# centerangle(k,t) = exp.( im .* k .* t)
# centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))

stdloop = true
while stdloop
    print("Standard Deviation of Attentive Well (e.g., 1, 0.5, 0.1)?: ")
    local stdcand = readline()
    stdcand = parse(Float64, stdcand)
    variance = stdcand^2
    if stdcand > 0
        println("Using Standard Deviation = $(stdcand), so Variance = $(variance)\n")
        global stddev = stdcand
        global variance = variance
        global stdloop = false
    else
        println("Standard Deviation must be positive!")
    end
end

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

gloop = true
while gloop
    print("Noise 'γ' Value?: ")
    local g_cand = readline()
    g_cand = parse(Float64, g_cand)
    if isdir(input_dir * "/a=$(α_value)/b=$(β_value)/g=$(g_cand)")
        println("Directories for α = $(α_value), β = $(β_value), and γ = $(g_cand) exist! \n
        Let's run...")
        global γ_value = g_cand
        global gloop = false
    else
        println("No such directories for α = $(α_value), β = $(β_value), and γ = $(g_cand)")
    end
end


Deltas = Vector{Vector{Float64}}()
Distances = Vector{Float64}()

csv_files = filter(f -> contains(f, "stationary") && endswith(f, ".csv"), readdir(input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)"; join=true))

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

    # for (index,xvalue) in enumerate(x)
    #     xshift[index] = xvalue - real(center(kval,timestamp[index]))
    # end
    # for (index,yvalue) in enumerate(y)
    #     yshift[index] = yvalue - imag(center(kval,timestamp[index]))
    # end

    # for (index,timevalue) in enumerate(timestamp)
    #     ypos[index] = (xshift[index] * real(centerangle(kval,timevalue))) + (yshift[index] * imag(centerangle(kval,timevalue)) )
    #     xpos[index] = (xshift[index] * real(centerangle90(kval,timevalue))) + (yshift[index] * imag(centerangle90(kval,timevalue)) )
    # end

    xpos = x
    ypos = y

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
                # Add the completed run to our main `Deltas` list, for Δt statistics
                push!(Deltas, deltanow)
                # We record the maximum distance of this run to `Distances` list, for amplitude statistics
                push!(Distances,sqrt(maximum(deltanow)))
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

using StatsBase, Plots

Lengths = [length(run) for run in Deltas]
Distances = Distances

binnumber = 300 
maxLen_val = ceil(log10(maximum(Lengths))) 
maxDist_val = ceil(log10(maximum(Distances))) 
minDist_val = floor(log10(minimum(Distances))) 
binsLenlog10 = 10 .^ (range(0, maxLen_val, length=binnumber)) 
binsDistlog10 = 10 .^ (range(0, maxDist_val, length=binnumber))


# --- Histogram of jump durations (Δt) ---
ΔTHist = Plots.histogram(
    Lengths,
    normalize = :pdf,
    bins = binsLenlog10,
    xlim = extrema(binsLenlog10),
    xscale = :log10,
    yscale = :log10,
    xlabel = "Δt (Number of Steps)",
    ylabel = "Probability Density",
    title = "Jump Durations: α=$(α_value), β=$(β_value), γ=$(γ_value), Var=$(variance)",
    legend = false,
    # c = :purple,
    minorgrid = true
)

# --- Histogram of jump maxima (Distances) ---
DistancesHist = histogram(
    Distances,
    bins = binsDistlog10,
    normalize = :pdf,
    xlim = extrema(binsDistlog10),
    xscale = :log10,
    yscale = :log10,
    xlabel = "Maximum Distance in Saccade",
    ylabel = "Probability Density",
    title = "Jump Maxima: α=$(α_value), β=$(β_value), γ=$(γ_value), Var=$(variance)",
    legend = false,
    # c = :red,
    minorgrid = true
)

# --- Save results ---
savefig(ΔTHist, input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" *
        "STD_$(stddev)-DeltaT_Histogram.pdf")
savefig(DistancesHist, input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" *
        "STD_$(stddev)-Distances_Histogram.pdf")

println("Saved histogram plots for Δt and Distances.")
