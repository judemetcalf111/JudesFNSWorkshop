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

k = 0.4

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/phasedata/k=$(k)/g=0.1"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata/k=$(k)/g=0.1"

# Make sure output folder exists
mkpath(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))
xmax = 7

kval = k

αs = collect(1.05:0.01:2.00)
βs = collect(0.01:0.01:0.4)

αsize = length(αs)
βsize = length(βs)
twoDsize = αsize * βsize

# Create a 3x4x5 array of Float64, uninitialized
Jumps_Array = Array{Any}(undef, αsize, βsize, 3)

for (index, alph) in  enumerate(αs)
    Jumps_Array[index,:,2] = βs
end

for (index, bet) in enumerate(βs)
    Jumps_Array[:,index,1] = αs
end

center(k,t) = (xmax ./ 2) .* exp.( im .* k .* t)

centerangle(k,t) = exp.( im .* k .* t)
centerangle90(k,t) = exp.( im .* ( k .* t .+ (π/2)))

for file in csv_files

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

    local aindex = findfirst(x -> x == aval, Jumps_Array[:,1,1])
    local bindex = findfirst(x -> x == bval, Jumps_Array[1,:,2])

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

    variance = 10

    Deltas = Vector{Vector{Float64}}()
    deltanow = Vector{Float64}()
    x0 = mean(xpos)
    y0 = mean(ypos)

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


# CSV.write(output_dir * "/AverageBehind.csv", Tables.table(-Jumps_Array[:,:,3]))
# CSV.write(output_dir * "/AverageAside.csv", Tables.table(Jumps_Array[:,:,4]))
# CSV.write(output_dir * "/Averagealpha.csv", Tables.table(Jumps_Array[:,:,1]))
# CSV.write(output_dir * "/Averagebeta.csv", Tables.table(Jumps_Array[:,:,2]))
# CSV.write(output_dir * "/Averager2.csv", Tables.table(r2))
