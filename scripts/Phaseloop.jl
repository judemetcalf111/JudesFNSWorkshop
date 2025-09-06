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
        JudesFNSWorkshop

import FractionalNeuralSampling: Density, divide_dims
import SpecialFunctions: gamma
import RecursiveArrayTools: ArrayPartition

set_theme!(foresight(:physics))

xmax = 7
x0 = [3.0, 0.0] 
p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
k = 0.2

center(t) = (xmax ./ 2) .* exp.( im * k * t)

wells(t) = [MvNormal([real(center(t)), imag(center(t))], I(2))]

G(t) = MixtureModel(wells(t),[1]) |> Density

timespan = 100.
δt = 0.001
seed = 27
γ_value = 0.2 
αs = collect(1.05:0.01:2.00)
βs = collect(0.001:0.001:0.1)
interfile = "/data/exp_raw/phasedata/k=$(k)/g=$(γ_value)"

mkpath(path * interfile)

for α_value in ProgressBar(αs)
    for β_value in ProgressBar(βs)
        
        L = JudesFNSWorkshop.aFractionalNeuralSampler(;
                                    u0 = [0.0, 0.0, 0.0, 0.0],
                                    tspan = timespan,
                                    α = α_value, # Tail index
                                    β = β_value, # Momentum strength
                                    γ = γ_value, # Noise strength
                                    𝜋 = G, # The target distribution
                                    seed = seed)
        
        filename = "/phaseloop_a=$(α_value)_b=$(β_value)"
        using CSV
        using DataFrames
        sol = solve(L, EM(); dt = δt) 
        x, y = eachrow(sol[1:2, :])
        
        walkerdata = DataFrame(sol)
        CSV.write(path * interfile * filename * ".csv", walkerdata)
    end
end

