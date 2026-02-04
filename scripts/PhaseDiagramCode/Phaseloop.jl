import Pkg
using DrWatson
path = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop" # Replace with your own path
quickactivate(path)

using Revise,
        ProgressBars,
        CairoMakie,
        Foresight,
        DifferentialEquations,
        FractionalNeuralSampling,
        Distributions,
        LinearAlgebra,
        StableDistributions,
        SpecialFunctions,
        Random,
        DiffEqNoiseProcess,
        PlutoUI
        
include(srcdir("JudesFNSWorkshop.jl"))

import FractionalNeuralSampling: Density, divide_dims
import SpecialFunctions: gamma
import RecursiveArrayTools: ArrayPartition

set_theme!(foresight(:physics))

xmax = 7
x0 = [3.0, 0.0] 
p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
k = 0.4

center(t) = (xmax ./ 2) .* exp.( im * k * t)

wells(t) = [MvNormal([real(center(t)), imag(center(t))], I(2))]

G(t) = MixtureModel(wells(t),[1]) |> Density

timespan = 100.
δt = 0.001
seed = 27
γ_value = 0.1
# αs = collect(1.05:0.01:2.00)
# βs = collect(0.0005:0.0005:0.06)

αs = collect(1.02:0.02:2.00)
# b0 = log10(.01)
# b1 = log10(100)
βs = collect(0.01:0.01:1.00)
interfile = "/data/exp_raw/phasedata/k=$(k)/g=$(γ_value)"

mkpath(path * interfile)

# --- Best Practice: Place all 'using' statements at the top ---
# I've moved these here from inside your loop.
# This avoids re-loading them on every iteration.
using CSV
using DataFrames
using ProgressBars # Assuming this is where ProgressBar comes from
# using DifferentialEquations, JudesFNSWorkshop, etc. (add other packages here)

# --- Assume these are defined elsewhere in your script ---
# path = "your/base/path/"
# interfile = "data_folder" 
# αs = [list of alpha values]
# βs = [list of beta values]
# timespan = (0.0, 100.0)
# γ_value = ...
# G = ...
# seed = ...
# δt = ...
# -----------------------------------------------------------

for α_value in ProgressBar(αs)
    for β_value in ProgressBar(βs)
        
        # 1. Define the target filename first
        # Note: Using joinpath is safer for constructing paths
        filename = "phaseloop_a=$(α_value)_b=$(β_value).csv"
        # Assuming 'path' and 'interfile' are strings.
        # If 'interfile' is a directory, joinpath is ideal.
        full_filepath = path * interfile * filename
        
        # 2. Check if the file already exists before doing any work
        if !isfile(full_filepath)
            # 3. If it does NOT exist, run the computation
            println("Generating: $full_filepath") # Added for visibility
            
            L = JudesFNSWorkshop.aFractionalNeuralSampler(;
                                        u0 = [0.0, 0.0, 0.0, 0.0],
                                        tspan = timespan,
                                        α = α_value, # Tail index
                                        β = β_value, # Momentum strength
                                        γ = γ_value, # Noise strength
                                        𝜋 = G, # The target distribution
                                        seed = seed)
            
            sol = solve(L, EM(); dt = δt) 
            
            # This line was in your original code but not used, so I've left it commented
            # x, y = eachrow(sol[1:2, :])
            
            walkerdata = DataFrame(sol)
            
            # Ensure the directory exists before writing
            # mkpath(dirname(full_filepath)) # Uncomment if directories might not exist
            
            CSV.write(path * interfile * filename * ".csv", walkerdata)
        
        else
            # 4. If it DOES exist, skip to the next iteration
            # You can uncomment this line for verbose logging
            # println("Skipping (already exists): $full_filepath")
        end
    end
end
