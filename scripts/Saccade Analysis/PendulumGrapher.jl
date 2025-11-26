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
using Tables
include(srcdir("PendulumParams.jl"))

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/SaccadeData"
xmax = 40
kval = 0.08

center(k,t) = -im .* (xmax ./ 2) .* exp.( im .* (π./3) .* cos.(k .* t))
puresin(k,t) = (xmax ./ 2) .* sin.((π./3) .* cos.(k .* t))

# centerangle(k,t) = exp.( im .* k .* t)
# centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))

params = select_parameters("/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/SaccadeData")

stddev = 2.
variance = stddev^2
(α_value, β_value, γ_value) = params


Deltas = Vector{Vector{Float64}}()
Distances = Vector{Float64}()

csv_files = filter(f -> contains(f, "pendulum") && endswith(f, ".csv") && !contains(f,"XVALUES"), readdir(input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)"; join=true))

for file in csv_files
    # Read CSV into DataFrame
    file = csv_files[1]
    global df = CSV.read(file, DataFrame)

    timestamp = collect(df.timestamp)
    global x = collect(df.value1)
    global y = collect(df.value2)
    lengthofdata = length(timestamp)

    xshift = zeros(lengthofdata)
    yshift = zeros(lengthofdata)
    xpos = zeros(lengthofdata)
    ypos = zeros(lengthofdata)

    Filename = splitext(basename(file))[1]

    for (index,xvalue) in enumerate(x)
        xshift[index] = xvalue - real(center(kval,timestamp[index]))
    end
    for (index,yvalue) in enumerate(y)
        yshift[index] = yvalue - imag(center(kval,timestamp[index]))
    end

    xpos = x
    ypos = y

    CSV.write(input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" * Filename * "XVALUES.csv", Tables.table(xpos))
    twlvpers = Int(floor(lengthofdata * .048))
    println(twlvpers)
    times = twlvpers:(twlvpers+700000)
    timesplot = times ./ 1e6

    twlvpers_Plot = Plots.plot(timesplot, xpos[times], 
        xlabel = "Time (steps × 10⁶)", 
        ylabel = "X Position (Degrees)", 
        title = "Pendulum Tracking: α=$(α_value), β=$(β_value), γ=$(γ_value)",
        label = "FNS Sampler", 
        legend = false, 
        # aspect_ratio = :equal,
        xlim = (timesplot[1],timesplot[end]),
        ylim = (-20,20),
        lw = 2.5,
        c = :darkorange,
        minorgrid = true,
        fontsize = 14
    )
    Plots.plot!(timesplot, puresin(kval,times.*1e-3),
        xlabel = "Time (steps × 10⁶)", 
        ylabel = "X Position (Degrees)", 
        legend = false, 
        linestyle = :dash,
        label = "Pendulum",
        # aspect_ratio = :equal,
        xlim = (timesplot[1],timesplot[end]),
        ylim = (-20,20),
        lw = 1.5,
        c = :black,
        minorgrid = true,
        guidefontsize = 16,
        tickfontsize = 14
    )
    savefig(twlvpers_Plot, input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" * Filename * "-Typical-XGraph.pdf")

    ##
    ## Loop to find the Δt array, to do statistics on the runs and find the distribution of
    ## the saccades of the walker away from the attentive well
    ##

    deltanow = Vector{Float64}()
    r_squared = xshift .^ 2 .+ yshift .^ 2

    for i in 1:lengthofdata
        # Calculate squared distance once per loop

        if r_squared[i] > variance
            # --- We are IN a run ---
            # Append the squared distance to the current run list.
            push!(deltanow, r_squared[i])
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

binnumber = 200 
maxLen_val = ceil(log10(maximum(Lengths))) 
maxDist_val = ceil(log10(maximum(Distances))) 
minDist_val = floor(log10(minimum(Distances))) 
binsLenlog10 = 10 .^ (range(0, maxLen_val, length=binnumber)) 
binsDistlog10 = 10 .^ (range(log10(2), maxDist_val, length=binnumber))


h = StatsBase.fit(
    Histogram, 
    Lengths, 
    binsLenlog10
)
h = StatsBase.normalize(h,mode=:pdf)

# Get x (bin centers) and y (bin heights)
x_vals = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
y_vals = h.weights

valid_indices = y_vals .> 0
log_x = log10.(x_vals[valid_indices])
log_y = log10.(y_vals[valid_indices])

X = [ones(length(log_x)) log_x]

coeffs = X \ log_y

intercept = coeffs[1]
slope = coeffs[2]

println("Fit Results (log10 space):")
println("Slope = $slope")
println("Intercept = $intercept")

x_fit = 10 .^ range(minimum(log_x), maximum(log_x), length=100)
y_fit = (10^intercept) .* (x_fit .^ slope)

ΔTHist = Plots.histogram(
    Lengths,
    normalize = :pdf,
    bins = binsLenlog10,
    xlim = extrema(binsLenlog10),
    xscale = :log10,
    yscale = :log10,
    xlabel = "Δt (Number of Steps)",
    ylabel = "Probability Density",
    label = "Saccade Data",
    title = "Jump Durations: α=$(α_value), β=$(β_value), γ=$(γ_value)",
    legend = false,
    minorgrid = true,

    fillcolor = :royalblue,  # Fill color of the bars (you had :purple)
    fillalpha = 0.7,        # Transparency of the fill (0.0 to 1.0)
    linecolor = :black,     # Color of the bar borders
    linewidth = 0.5,        # Width of the bar borders
    gridstyle = :dot,       # Grid line style (:solid, :dash, :dot)
    gridalpha = 0.5        # Transparency of grid lines
)

s = round(slope,digits=2)
Plots.plot!(
    x_fit,
    y_fit,
    label = "Power-Law Fit ΔT^$(s)",
    legend = :topright,
    color = :darkorange,
    linewidth = 3,
    linestyle = :dash,
    legendfontsize = 16,
    guidefontsize = 16,
    tickfontsize = 14,
    margin = 5Plots.mm
)

h = StatsBase.fit(
    Histogram, 
    Distances, 
    binsDistlog10
)

h = StatsBase.normalize(h,mode=:pdf)

# Get x (bin centers) and y (bin heights)
x_vals = (h.edges[1][1:end-1] .+ h.edges[1][2:end]) ./ 2
y_vals = h.weights

valid_indices = y_vals .> 0
log_x = log10.(x_vals[valid_indices])
log_y = log10.(y_vals[valid_indices])

X = [ones(length(log_x)) log_x]

coeffs = X \ log_y

intercept = coeffs[1]
slope = coeffs[2]

println("Fit Results (log10 space):")
println("Slope = $slope")
println("Intercept = $intercept")

x_fit = 10 .^ range(0, maximum(log_x), length=100)
y_fit = (10^intercept) .* (x_fit .^ slope)

# --- Histogram of jump maxima (Distances) ---
DistancesHist = histogram(
    Distances,
    bins = binsDistlog10,
    normalize = :pdf,
    xlim = extrema(binsDistlog10),
    ylim = (1e-4,1e1),
    xscale = :log10,
    yscale = :log10,
    xlabel = "Maximum Distance of Saccade",
    ylabel = "Probability Density",
    title = "Jump Maxima: α=$(α_value), β=$(β_value), γ=$(γ_value)",
    label = "Saccade Data",
    legend = false,
    # c = :red,
    minorgrid = true,

    fillcolor = :royalblue,  # Fill color of the bars (you had :purple)
    fillalpha = 0.7,        # Transparency of the fill (0.0 to 1.0)
    linecolor = :black,     # Color of the bar borders
    linewidth = 0.5,        # Width of the bar borders
    gridstyle = :dot,       # Grid line style (:solid, :dash, :dot)
    gridalpha = 0.5        # Transparency of grid lines
)

s = round(slope,digits=2)
Plots.plot!(
    x_fit,
    y_fit,
    label = "Power-Law Fit Δx^$(s)",
    xlim = extrema(binsDistlog10),
    ylim = (1e-4,1e1),
    legend = :topright,
    color = :darkorange,
    linewidth = 3,
    linestyle = :dash,
    legendfontsize = 16,
    guidefontsize = 16,
    tickfontsize = 14,
    margin = 5Plots.mm
)

# --- Save results ---
savefig(ΔTHist, input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" *
        "STD_$(stddev)-pendulum-DeltaT_Histogram.pdf")
savefig(DistancesHist, input_dir * "/a=$(α_value)/b=$(β_value)/g=$(γ_value)/" *
        "STD_$(stddev)-pendulum-Distances_Histogram.pdf")

println("Saved histogram plots for Δt and Distances.")
