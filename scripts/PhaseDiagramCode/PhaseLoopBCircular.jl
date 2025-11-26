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

xmax = 20
x0 = [3.0, 0.0] 
p0 = [0.0, 0.0]     # Be careful with types; use 0.0 not 0
timespan = 1000.0
δt = 0.001

σ = 1               # Well Brownian Movement standard deviation, set to 1, `k` determining speed

wellseed = 49
seed = 27
β_value = 0.01
γ_value = 0.1

αs = collect(1.05:0.05:2.00)
ks = collect(0.05:0.05:3.00)

interfile = "/data/exp_raw/phasedata/BCircular/b=$(β_value)/g=$(γ_value)"
wellseed = 49
Random.seed!(wellseed)

wellsrandomgen = rand(Normal(0, σ * √δt),Int(timespan/δt+1))

wellsrandom = cumsum(wellsrandomgen)

center(k,t) = (xmax ./ 2) .* exp.( im * k * wellsrandom[round(Int,t/δt)] )
wells(t) = [MvNormal([real(center(k,t+δt)), imag(center(k,t+δt))], I(2))]
G(t) = MixtureModel(wells(k,t),[1]) |> Density

mkpath(path * interfile)

for α_value in ProgressBar(αs)
    for k_value in ProgressBar(ks)
        
    center(t) = (xmax ./ 2) .* exp.( im * k_value * wellsrandom[round(Int,t/δt)] )
    wells(t) = [MvNormal([real(center(t+δt)), imag(center(t+δt))], I(2))]
    G(t) = MixtureModel(wells(t),[1]) |> Density

        L = JudesFNSWorkshop.aFractionalNeuralSampler(;
                                    u0 = [0.0, 0.0, 0.0, 0.0],
                                    tspan = timespan,
                                    α = α_value, # Tail index
                                    β = β_value, # Momentum strength
                                    γ = γ_value, # Noise strength
                                    𝜋 = G, # The target distribution
                                    seed = seed)

        filename = "/phaseloop_a=$(α_value)_k=$(k_value)"
        using CSV
        using DataFrames
        sol = solve(L, EM(); dt = δt) 
        x, y = eachrow(sol[1:2, :])
        
        walkerdata = DataFrame(sol)
        CSV.write(path * interfile * filename * ".csv", walkerdata)
    end
end
