### A Pluto.jl notebook ###
# v0.20.15

using Markdown
using InteractiveUtils

# ╔═╡ b618183f-9fd3-4e77-9623-11c85c774f02
begin
	import Pkg
	using DrWatson
	path = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop" # Replace with your own path
	quickactivate(path)
end

# ╔═╡ d24fd425-57bd-4964-8f5b-91a3566bb453
begin
    using Revise,
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
end



# ╔═╡ 6db1a2f3-eac7-4a22-876a-cbbbb642ff48
begin # Generate a distribution to sample
    xmax = 7
    x0 = [3.0, 0.0] 
    p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
	k = 0.02
	
    center(t) = (xmax ./ 2) .* exp.( im * k * t)
	
    wells(t) = [MvNormal([real(center(t)), imag(center(t))], I(2))]

    G(t) = MixtureModel(wells(t),[1]) |> Density
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
	# TimeseriesMakie.trails!(ax, xy; n_points=100, colormap=:turbo, color=1:100) # You could go fancy by also adding `colormap=:turbo, color=1:100`
	TimeseriesMakie.heatmap!(ax, xs, xs, X, colormap=seethrough(:turbo), colorrange=(0, 0.25))
	TimeseriesMakie.scatter!(ax, xy_last; markersize=10, color=:red)
	hidedecorations!(ax)
	hidespines!(ax)

	println("Plotting setup...")
end

# ╔═╡ b60add8e-22c4-42c3-a666-7753b0dac569
begin

	H = 0.5       # Hurst parameter
	timespan = 500.
	δt = 0.001

	seeds = [27]#,42,132,156,109,5,3201,4325,2835,3746]
	
	for seedvalue in seeds
		α_value = 1.5
		β_value = 0.2
		γ_value = 0.1 

		Random.seed!(seedvalue) # ! Set the seed HERE, before running FractionalLM

		η = JudesFNSWorkshop.FractionalLM(H, α_value;
										dt = δt, tspan = timespan, ND = 2)
		η = hcat(η, zero(η))  |> eachrow
		η = NoiseGrid(range(0, stop = timespan, length = size(η, 1)), η)

		L = JudesFNSWorkshop.aFractionalNeuralSampler(;
									u0 = [0.0, 0.0, 0.0, 0.0],	# Initial Noise Vector
									tspan = timespan,     		# Timesteps
									α = α_value, 		  		# Tail index
									β = β_value, 		  		# Momentum strength
									γ = γ_value, 		  		# Noise strength
									𝜋 = G, 	  		   		 # The target distribution
									noise = η,
									seed = seedvalue)     		# This seed has no effect, the seed is meaningfully used in the FractionalLM function 
		
		using Dates
		hourminute = Dates.format(now(), "HH:MM")
		filename = "tfns_a=$(α_value)_b=$(β_value)_g=$(γ_value)_k=$(k)_H=$(H)-CorrLevy-" * hourminute
		using CSV
	    using DataFrames
		println("Ready to solve...")
		sol = solve(L, EM(); dt = δt) 
		x, y = eachrow(sol[1:2, :])
		println("Solved!")
		
		walkerdata = DataFrame(sol)
		CSV.write(path * "/data/exp_raw/" * filename * ".csv", walkerdata)
		
		
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
end
