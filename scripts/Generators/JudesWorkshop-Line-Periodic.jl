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

# ╔═╡ afbaee5e-a6f8-49f9-9756-96421ad76ff1
begin
    Pkg.add(["Distributions",
             "LinearAlgebra",
			 "SpecialFunctions",
			 "RecursiveArrayTools",
			 "PlutoUI"])
end

# ╔═╡ a5ec26ad-0811-4d2e-8656-69abed763f48
begin
	Pkg.add(url = "https://github.com/brendanjohnharris/FractionalNeuralSampling.jl",
	rev = "fully_fractional")
end

# ╔═╡ 70d40689-cbdc-42c7-888f-e8c14c99d23c
begin
	Pkg.add(url = "https://github.com/brendanjohnharris/TimeseriesPlots.jl")
end

# ╔═╡ d24fd425-57bd-4964-8f5b-91a3566bb453
begin
    using CairoMakie
    using Foresight
    using DifferentialEquations
    using FractionalNeuralSampling
    using Distributions
    using LinearAlgebra
    using PlutoUI

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
			# κ = -0.2
			# Γ = [1.0 κ; κ 1.0]
			noise = NoiseProcesses.LevyProcess!(α; ND = 2, W0 = zero(u0))
		end
	    Sampler(afns_f!, afns_g!; callback = boundaries, kwargs..., u0,
	            noise_rate_prototype, noise,
	            tspan, p = ((α, β, γ), 𝜋))
	end

# ╔═╡ 6db1a2f3-eac7-4a22-876a-cbbbb642ff48
begin # Generate a distribution to sample
    xmax = 7
    x0 = [3.0, 0.0] 
    p0 = [0.0, 0.0] # Be careful with types; use 0.0 not 0
    numb = 5 # Number of wells on screen at once
    Lg = 100
    Ng = 2*Lg + 1
    k = 0.05

    trainofIntegers = range(-Lg, stop = Lg, length = Ng)
    ϕs = range(0, stop = 2π, length = 2*Ng)[1:Ng]

    centers(t) = (xmax * 2 / numb) .* trainofIntegers .- (k * t) 

    𝑓 = 0.005 # Hz

    # vector of amplitudes for all phases
    a(t, ϕ) = @. sin(𝑓*t + ϕ*Ng/numb)^2
    a_norm(t) = let w = a(t, ϕs); w ./ sum(w) end

    wells(t) = [MvNormal([c, 0.0], I(2)) for c in centers(t)]

    G(t) = MixtureModel(wells(t), a_norm(t)) |> Density
end

# ╔═╡ 6db08281-8842-4eba-bf94-808454fa05c6
begin
	using TimeseriesPlots
 	fig = Figure()
	
	xs = range(-xmax, xmax, length = 200)
	Xs = collect.(Iterators.product(xs, xs))
	X = Observable(G(0).(Xs)) # For recording
	xy = Observable([Point2f([NaN, NaN])]) 
	xy_last = Observable(Point2f([NaN, NaN]))
	color = (:black, 0.5)

	ax = Axis(fig[1, 1]; aspect=DataAspect())
	# trails!(ax, xy; n_points=100, colormap=:turbo, color=1:100) # You could go fancy by also adding `colormap=:turbo, color=1:100`
	heatmap!(ax, xs, xs, X, colormap=seethrough(:turbo), colorrange=(0, 0.08*3/Lg))
	scatter!(ax, xy_last; markersize=10, color=:red)
	hidedecorations!(ax)
	hidespines!(ax)

end

# ╔═╡ a0ea8f09-296b-4782-8d4c-dd5ca738e2af
begin # Run time-varying simulation
	  # Commented out are the user inputs for parameters whenrunning through the REPL 

	
	# print("Tail Index? ")
	# α_value = readline()
	α_value = 1.1 #parse(Float64, α_value)
	# print("Momentum Strength? ")
	# β_value = readline()
	β_value = 0.2 #parse(Float64, β_value)
	# print("Noise Strength? ")
	# γ_value = readline()
	γ_value = 0.1 #parse(Float64, γ_value)
	L = aFractionalNeuralSampler(;
								u0 = ArrayPartition(x0, p0),
								tspan = 5000.0,
								α = α_value, # Tail index
								β = β_value, # Momentum strength
								γ = γ_value, # Noise strength
								𝜋 = G, # The target distribution
								seed = 41)
end

# ╔═╡ cd9f5ac8-cdbf-45b7-af67-1ba33c7df82d
begin
	sol = solve(L, EM(); dt = 0.001) # Takes about 5 seconds
	x, y = eachrow(sol[1:2, :])
end

# ╔═╡ 63215cf4-a968-4e98-a2fc-9aedb7df2db0
begin # Functions created to plot the sampler looping around the arena
	
	function periodic_distance(a, b, xmax)
	    dx = abs(a - b)
	    return min(dx, 2xmax - dx)
	end
	
	function wrap_periodic(point::Point2f, xmax)
	    return Point2f(
	        mod(point[1] + xmax, 2xmax) - xmax,
	        mod(point[2] + xmax, 2xmax) - xmax
	    )
	end
	
	function update_trail!(xy, new_point, xmax)
	    wrapped = wrap_periodic(new_point, xmax)
	    if !isempty(xy[])
	        lastp = xy[][end]
	        if periodic_distance(lastp[1], wrapped[1], xmax) > xmax/2 ||
	           periodic_distance(lastp[2], wrapped[2], xmax) > xmax/2
	            push!(xy[], Point2f(NaN, NaN))  # Break trail
	        end
	    end
	    push!(xy[], wrapped)
	    length(xy[]) > 100 && popfirst!(xy[])
	end
end

# ╔═╡ b60add8e-22c4-42c3-a666-7753b0dac569
begin
	xy[] = [Point2f([NaN, NaN])]
	filename = "tfns_a=$(α_value)_b=$(β_value)_g=$(γ_value)_Linie.mp4"
	file = record(fig, filename, range(1, length(sol), step=1000);
	        framerate = 48) do i
		t = sol.t[i]
	    X[] = G(t).(Xs)
		xy[] = push!(xy[], wrap_periodic(Point2f(sol[1:2, i]), xmax))
		length(xy[]) > 100 && popfirst!(xy[])
		update_trail!(xy, Point2f(sol[1:2, i]), xmax)
		xy_last[] = last(filter(!isnan, xy[]))
	end
end

# ╔═╡ 0d7ca9f8-ac50-40cc-bce7-a258aab6a7f8
PlutoUI.LocalResource(filename)

# ╔═╡ Cell order:
# ╠═b618183f-9fd3-4e77-9623-11c85c774f02
# ╠═afbaee5e-a6f8-49f9-9756-96421ad76ff1
# ╠═a5ec26ad-0811-4d2e-8656-69abed763f48
# ╠═70d40689-cbdc-42c7-888f-e8c14c99d23c
# ╠═d24fd425-57bd-4964-8f5b-91a3566bb453
# ╠═50ce1e9e-e5e1-4ab4-82d9-18ef2526c63f
# ╠═2aec5c65-5e24-4327-8038-ceb70ff29d8d
# ╠═8b1a7d48-426a-4c76-b63a-61693a457281
# ╠═3a24c88c-fca5-4643-85c0-2190c4a13b5d
# ╠═131da237-ca04-4793-b954-12e3c56c47d9
# ╠═b159f245-f421-4323-8958-c0df43f5b994
# ╠═6db1a2f3-eac7-4a22-876a-cbbbb642ff48
# ╠═6db08281-8842-4eba-bf94-808454fa05c6
# ╠═a0ea8f09-296b-4782-8d4c-dd5ca738e2af
# ╠═cd9f5ac8-cdbf-45b7-af67-1ba33c7df82d
# ╠═63215cf4-a968-4e98-a2fc-9aedb7df2db0
# ╠═b60add8e-22c4-42c3-a666-7753b0dac569
# ╠═0d7ca9f8-ac50-40cc-bce7-a258aab6a7f8
