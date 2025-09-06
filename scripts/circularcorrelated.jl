using DrWatson
using CairoMakie
using Foresight
using DifferentialEquations
using FractionalNeuralSampling
using Distributions
using LinearAlgebra
using StableDistributions
using SpecialFunctions
using Random
using DiffEqNoiseProcess
using TimeseriesMakie

import FractionalNeuralSampling: Density, divide_dims
import SpecialFunctions: gamma
import RecursiveArrayTools: ArrayPartition

@quickactivate "JudesFNSWorkshop" # Searches up the file tree from the current script
using JudesFNSWorkshop

function afns_f!(du, u, p, t)
    (α, β, γ), 𝜋 = p
    x, v = divide_dims(u, length(u) ÷ 2)

    # Here we have replaced 𝜋 -> 𝜋(t)
    b = gradlogdensity(𝜋(t))(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)

    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ .* b .+ β .* v
    dv .= β .* b
end

function afns_g!(du, u, p, t) # Same as original equations
    (α, β, γ), 𝜋 = p
    dx, dv = divide_dims(du, length(du) ÷ 2)
    dx .= γ^(1 / α) # ? × dL in the integrator.
    dv .= 0.0
end

function aFractionalNeuralSampler(;
                                  tspan, α, β, γ, u0, 𝜋,
                                  boundaries = nothing,
                                  noise_rate_prototype = zero(u0),
                                  noise = nothing,
                                  kwargs...)
    if isnothing(noise)
        noise = NoiseProcesses.LevyProcess!(α; ND = 2, W0 = zero(u0))
    end
    Sampler(afns_f!, afns_g!;
            callback = boundaries,
            u0,
            noise_rate_prototype,
            noise,
            tspan,
            p = ((α, β, γ), 𝜋),
            kwargs...)
end

begin # Generate a distribution to sample
    xmax = 7
    x0 = [3.0, 0.0]
    p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
    k = 0.02

    center(t) = (xmax ./ 2) .* exp.(im * k * t)

    wells(t) = [MvNormal([real(center(t)), imag(center(t))], I(2))]

    G(t) = MixtureModel(wells(t), [1]) |> Density
end

begin
    H = 0.7       # Hurst parameter
    timespan = 1000.0
    δt = 0.001
    α_value = 1.1
    Random.seed!(22) # ! Set the seed HERE, before running FractionalLM

    η = JudesFNSWorkshop.FractionalLM(H, α_value;
                                      dt = δt, tspan = timespan, ND = 2)
    η = hcat(η, zero(η)) |> eachrow
    η = NoiseGrid(range(0, stop = timespan, length = size(η, 1)), η)

    β_value = 0.2
    γ_value = 0.1
    L = aFractionalNeuralSampler(;
                                 u0 = [0.0, 0.0, 0.0, 0.0],
                                 tspan = timespan,
                                 α = α_value, # Tail index
                                 β = β_value, # Momentum strength
                                 γ = γ_value, # Noise strength
                                 𝜋 = G, # The target distribution
                                 noise = η,
                                 seed = 26) # ! The seed has no effect here anymore! Since the randomness is fully contained in the FractionalLM function.
end

begin # * Solve
    sol = solve(L, EM(); dt = δt)
    x, y = eachrow(sol[1:2, :])
end

begin
    trail(x, y)
end
