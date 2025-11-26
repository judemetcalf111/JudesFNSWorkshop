### A Pluto.jl notebook ###
# v0.20.15

using Markdown
using InteractiveUtils

# ╔═╡ b618183f-9fd3-4e77-9623-11c85c774f02
begin
	import Pkg
	using DrWatson
	path = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop"
	quickactivate(path)
end

# ╔═╡ d24fd425-57bd-4964-8f5b-91a3566bb453
begin
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
    using PlutoUI
	using Infiltrator
	using ProgressBars

	include(srcdir("JudesFNSWorkshop.jl"))

    import FractionalNeuralSampling: Density, divide_dims
    import SpecialFunctions: gamma
    import RecursiveArrayTools: ArrayPartition

    set_theme!(foresight(:physics))
end

# ╔═╡ 50ce1e9e-e5e1-4ab4-82d9-18ef2526c63f
md"""
# 2D FNS

This notebook simulates of the fractional neural sampling diffusion model for a target distribution, chosen here to be `Ng=3` Gaussians arranged on a ring.
The tail index `α` controls the strength of Levy jumps, the momentum parameter `β` controls the strength of local oscillations, and `γ` controls the noise strength.

To run this notebook locally, install Julia (https://github.com/JuliaLang/juliaup) and follow the instructions at the top right ("Edit or run this notebook").
"""

# ╔═╡ 2aec5c65-5e24-4327-8038-ceb70ff29d8d
md"""
## Fractional Neural Sampler

$$dx_t = -\gamma c_\alpha \nabla V(x_t)  dt  + \beta  p_t dt + \gamma^{1/\alpha}dL^\alpha_t$$

$$dp_t = -\beta c_\alpha \nabla V(x_t) dt$$



where:
- ``V(x) = -\ln[\pi(x)]`` is the potential associated with a target distribution ``\pi(x)``;
- ``dL^\alpha_t`` is the increment of a Levy process with tail index ``\alpha``;
- ``\gamma`` is the noise strength;
- ``\beta`` is the momentum parameter; and
- ``c_\alpha = \Gamma(\alpha - 1)/\Gamma(\alpha / 2)^2`` is the correction factor for the local approximation to the fractional spatial derivative.
"""


# ╔═╡ 8b1a7d48-426a-4c76-b63a-61693a457281
md"""
## Time-varying Potential

To introduce a time-varying potential, we need to turn ``\pi`` into a function of time. I've made the potentials both vary in weight and migrate across the landscape.
"""

# ╔═╡ 3a24c88c-fca5-4643-85c0-2190c4a13b5d
function afns_f!(du, u, p, t)
	    (α, β, γ), 𝜋 = p
	    x, v = divide_dims(u, length(u) ÷ 2)
		# Here we have replaced 𝜋 -> 𝜋(t)
	    b = gradlogdensity(𝜋(t))(x) * gamma(α - 1) / (gamma(α / 2) .^ 2)

	    dx, dv = divide_dims(du, length(du) ÷ 2)
	    dx .= γ .* b .+ β .* v
	    dv .= β .* b
end

# ╔═╡ 131da237-ca04-4793-b954-12e3c56c47d9
function afns_g!(du, u, p, t) # Same as original equations
	    (α, β, γ), 𝜋 = p
	    dx, dv = divide_dims(du, length(du) ÷ 2)
	    dx .= γ^(1 / α) # ? × dL in the integrator.
	    dv .= 0.0
	end

# ╔═╡ b159f245-f421-4323-8958-c0df43f5b994
function aFractionalNeuralSampler(;
	                                 tspan, α, β, γ, u0, 𝜋,
	                                 boundaries = nothing,
	                                 noise_rate_prototype = similar(u0),
	                                 noise = nothing,
	                                 kwargs...)
		if isnothing(noise)
			noise = NoiseProcesses.LevyProcess!(α; ND = 2, W0 = zero(u0))
		end

	    Sampler(afns_f!, afns_g!; callback = boundaries, kwargs..., u0,
	            noise_rate_prototype, noise,
	            tspan, p = ((α, β, γ), 𝜋))
end

# ╔═╡ 6db1a2f3-eac7-4a22-876a-cbbbb642ff48
begin # Generate a distribution to sample
    xmax = 20
    x0 = [3.0, 0.0] 
    p0 = [0.0, 0.0]     # Be careful with types; use 0.0 not 0
	k = 0.2
    timespan = 5000.0
    δt = 0.001

	swidth = 5 # Weak: 5, Far: 5, Normal: 10
	shelf_height = (5^2) / 2 # Weak: (5^2) / 2, Far: (10^2) / 2, Normal: (5^2) / 2

    wellseed = 50
    Random.seed!(wellseed)
    wellsrandomgen = rand(Normal(0, √δt),Int(timespan/δt+1))
    wellsrandom = cumsum(wellsrandomgen)
    center(t) = (xmax ./ 2) .* exp.( im * k * wellsrandom[round(Int,t/δt)])

	weights(s,p) = ( 1 + ( ( 1 / s ) * exp( p*(1 - ( 1 / s^2 ) )) ) )^(-1)

	wells(t) = [ MvNormal([real(center(t+δt)), imag(center(t+δt))], I(2)) ,
					MvNormal( [real(center(t+δt)), imag(center(t+δt))], ( swidth^2 ) * I(2)) 
					]

    G(t) = MixtureModel(wells(t), [1-weights(swidth,shelf_height),weights(swidth,shelf_height)]) |> Density

end

# ╔═╡ 6db08281-8842-4eba-bf94-808454fa05c6
begin
	using TimeseriesMakie
 	fig = Figure()
	
	xs = range(-xmax, xmax, length = 200)
	Xs = collect.(Iterators.product(xs, xs))
	X = Observable(G(0).(Xs)) # For recording
	xy = Observable([Point2f([NaN, NaN])]) 
	xy_last = Observable(Point2f([NaN, NaN]))
	color = (:black, 0.5)

	ax = Axis(fig[1, 1]; aspect=DataAspect())
	# trails!(ax, xy; n_points=100, colormap=:turbo, color=1:100) # You could go fancy by also adding `colormap=:turbo, color=1:100`
	TimeseriesMakie.heatmap!(ax, xs, xs, X, colormap=seethrough(:turbo), colorrange=(0, 0.25))
	TimeseriesMakie.scatter!(ax, xy_last; markersize=10, color=:red)
	hidedecorations!(ax)
	hidespines!(ax)
end

# ╔═╡ b60add8e-22c4-42c3-a666-7753b0dac569
begin
	# global α_value = 1.5
	global β_value = 0.01
	global γ_value = 0.1
	fullpath = path * "/data/exp_raw/"
	mkpath(fullpath)

	seeds = [ 27,
				5,
				6
	]
	
	seedvalue = 5

	αs = 1.1:0.1:2.0
	# 			208,
	# 			600,
	# 			32,
	# 			19
	# ]

	# for seedvalue in ProgressBar(seeds)
	for α_value in ProgressBar(αs)

		L = aFractionalNeuralSampler(;
									u0 = ArrayPartition(x0, p0),
									tspan = timespan,
									α = α_value, # Tail index
									β = β_value, # Momentum strength
									γ = γ_value, # Noise strength
									𝜋 = G,       # The target distribution
									seed = seedvalue)
		
		using Dates
		hourminute = Dates.format(now(), "HH:MM")
		filename = "tfnsBrownian_a=$(α_value)_b=$(β_value)_g=$(γ_value)_k=$(k)_xmax=$(xmax)-UnCorrLevy-WeakPlateau" * hourminute
		using CSV
	    using DataFrames
		sol = solve(L, EM(); dt = δt) # Takes about 5 seconds
		x, y = eachrow(sol[1:2, :])
		walkerdata = DataFrame(sol)

		CSV.write(fullpath * filename * ".csv", walkerdata)
		
		xy[] = [Point2f([NaN, NaN])]
		file = record(fig, path * "/data/sims/" * filename * ".mp4", range(1, length(sol), step=1000);
		        framerate = 48) do i
			t = sol.t[i]
		    X[] = G(t).(Xs)
			xy[] = push!(xy[], Point2f(sol[1:2, i]))
			length(xy[]) > 100 && popfirst!(xy[])
			xy_last[] = last(xy[])



		end
	end
	# end
end
