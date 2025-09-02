using DrWatson
@quickactivate("JudesFNSWorkshop")
using CairoMakie
using Distributions
using StableDistributions
using SpecialFunctions
using Random
using DiffEqNoiseProcess
import Pkg
Pkg.add(url = "https://github.com/uow-statistics/Brownian.jl.git")
# THIS IS INTENDED TO BE A CLEANER (AND MORE CORRECT) VERSION OF noise.jl

function clampComponents(x, maxSize)
    if (maxSize < 0)
        error("Maximum must be non-negative")
    end
    for i in 1:size(x,1)
        R = norm(x[i,:])
        if (R > maxSize)
            x[i,:] = x[i,:] .* (maxSize/R)
        end
    end
    return x
end

# WHITE NOISE
function whiteNoise(dt=1,tspan=1000; ND=1, max=nothing, seed=nothing)
    N = Int(floor(tspan/dt))+1

    if (seed !== nothing)
        Random.seed!(seed)
    end
    dist = Normal()
    ξ = rand(dist, N)

    if (ND == 1)
        x = ξ
    elseif (ND == 2)
        θ = (2π) .* rand(N)
        x = hcat(ξ .* cos.(θ), ξ .* sin.(θ))
    else
        error("ND must be 1 or 2")
    end

    if (max !== nothing)
        x = clampComponents(x, max)
    end

    # For standard Brownian Motion, B(0) = 0
    x[1,:] .= 0

    return x
end

# BROWNIAN MOTION
function brownianMotion(dt=1,tspan=1000; σ=1, ND=1, max=nothing)
    dW = whiteNoise(dt,tspan,ND=ND,max=max)
    N = Int(floor(tspan/dt))+1
    B = zeros(N,ND)
    # B[1] = 0
    B .= cumsum(σ * sqrt(dt) .* dW, dims=1)
    return B
end

# DOUBLE BROWNIAN MOTION (INTEGRAL OF BM)
function doubleBrownianMotion(dt=1,tspan=1000; σ=1, ND=1, max=nothing)
    dB = brownianMotion(dt,tspan,ND=ND,max=max)
    N = Int(floor(tspan/dt))+1
    B = zeros(N,ND)
    # B[1] = 0
    B .= cumsum(σ * sqrt(dt) .* dB, dims=1)
    return B
end

# Geometric Brownian Motion
function GBM(; μ=1, σ=1, dt=1, tspan=1000, n = 1)
    N = Int(floor(tspan/dt))+1
    X = zeros(n, N)
    drift = μ .* collect(0:dt:tspan)
    for i in range(1,n)
        B = brownianMotion(dt, tspan, σ=σ)
        X[i,:] = exp.(drift .+ B)
    end
    return X
end

# LEVY NOISE
# Generates a vector noise sampled from a Levy Alpha Stable Distribution with parameters:
# α ∈ [1,2]: Tail index
# β ∈ [-1,1]: Skewness
# γ > 0: Scale parameter
# δ: Location parameter
# ND: No. of Dimensions (1 or 2 only)
function levyNoise(α=1.5, dt=1,tspan=1000; β=0, γ=1, δ=0, ND=1, max=nothing, seed=nothing)
    N = Int(floor(tspan/dt))+1
    dist = Stable(α, β, γ, δ)
    if (seed !== nothing)
        Random.seed!(seed)
    end
    ξ = rand(dist, N)

    if (ND == 1)
        x = ξ
    elseif (ND == 2)
        θ = (2π) .* rand(N)
        x = hcat(ξ .* cos.(θ), ξ .* sin.(θ))
    else
        error("ND must be 1 or 2")
    end

    if (max !== nothing)
        x = clampComponents(x, max)
    end

    # For standard Levy Motion, B(0) = 0
    x[1,:] .= 0

    return x
end

# LEVY MOTION
function levyMotion(α=1.5, dt=1, tspan=1000; σ=1, ND=1, max=nothing)
    dL = levyNoise(α, dt,tspan,ND=ND,max=max)
    N = Int(floor(tspan/dt))+1
    L = zeros(N,ND)
    # L[1] = 0
    L .= cumsum(σ * (dt)^(1/α) .* dL, dims=1)
    return L
end


function param_check(N, dt, tspan)
    if N !== nothing && dt !== nothing && tspan === nothing
        tspan = N * dt
    elseif dt !== nothing && tspan !== nothing && N === nothing
        N = floor(Int, tspan/dt)
    elseif N !== nothing && tspan !== nothing && dt === nothing
        dt = tspan / N
    elseif N === nothing || dt === nothing || tspan === nothing
        error("param_check: must provide at least two of (N, dt, tspan).")
    end

    return (N, dt, tspan)
end

# Fractional Brownian Motion
function FractionalBM(H; dt=nothing,tspan=nothing,N=nothing,μ=1.0,maxStep=nothing,ND=1,seed=nothing, method=:matrix)
    # Must have at least 2 of 3 (N, dt, tspan)
    N, dt, tspan = param_check(N, dt, tspan)

    if (μ < 0 || μ > 1)
        error("μ must be in the range [0,1]")
    end

    x = zeros(N,ND)

    # Generate white noise
    dW = whiteNoise(dt,tspan,ND=ND,seed=seed)

    # Determine the lower time bound for integration at each step
    # i_mins = [max(1, round(Int, i - 1 - μ * N)) for i in 1:N]

    # Precompute scaling factor
    G = dt^(H-0.5)/(gamma(H+0.5) * (H+0.5))

    # MATRIX METHOD: O(N^2) SPACE
    if (method == :matrix)
        v_mat = zeros(N,N)

        for j = 1:N
            for i = 1:j
                v_mat[i,j] = (j - i + 1)^(H+0.5) - ((j - i + 1)-1)^(H+0.5)
            end
        end

        if (ND == 1)
            x = G .* (dW[2:end,1]' * v_mat)'
        elseif (ND == 2)
            x[:,1] = G .* (dW[:,1]' * v_mat)' 
            x[:,2] = G .* (dW[:,2]' * v_mat)' 
        end 

    end

   
    # VECTOR SLICE METHOD: O(N) SPACE
    if (method == :slice)
        # This constructs the last column of the weight matrix
        #  All other columns are slices of this column
        v = [(N .- i + 1)^(H+0.5) .- (N .- i).^(H+0.5) for i in 1:N]
        for i in 1:N
            if (ND == 1)
                x[i,1] = G .* dot(dW[1:i,1], v[end-i+1:end])
            elseif (ND == 2)
                x[i,1] = G .* dot(dW[1:i,1], v[end-i+1:end])
                x[i,2] = G .* dot(dW[1:i,2], v[end-i+1:end])
            end
        end
    end

    # idk if the factors are necessary
    x = sqrt(2*dt) .* x
    return x
end




# Fractional Levy Motion
#   H: Hurst parameter defining power of integration kernel
#   α: Tail index for the Levy distribution (1 ≤ α ≤ 2)
#   σ: Noise Strength
#   μ: Memory fraction (0 ≤ μ ≤ 1)
function FractionalLM(H::Float64, α::Float64; dt=nothing, tspan=nothing, N=nothing, μ=1.0, maxStep=nothing, ND=1, method=:slice, uncorNoise=false, seed=nothing, k = 1.0)
    if (μ < 0 || μ > 1)
        error("μ must be in the range [0,1]")
    end

    # Must have at least 2 of 3 (N, dt, tspan)
    N, dt, tspan = param_check(N, dt, tspan)

    x = zeros(N,ND)

    # Construct a list of N random steps from Levy-α distribution
    dL = levyNoise(α,dt,tspan,ND=ND, max=maxStep, seed=seed)

    # Determine the lower time bound for integration at each step
    # i_mins = [max(1, round(Int, i - 1 - μ * N)) for i in 1:N]

    # Compute kernel exponent
    p = (2*H-1)/α

    # Precompute scaling factor
    G = (dt^(p+1-1/α))/(gamma(p+1.0) * (p+1.0))
    # G = (dt^(p+1-1/α))/(p+1.0)
    # G = (dt^(p+1))


    # VECTOR SLICE METHOD: O(N) SPACE
    if (method == :slice)
        # This constructs the last column of the weight matrix
        #  All other columns are slices of this column
        v = [(N .- i + 1)^(p+1.0) .- (N .- i).^(p+1.0) for i in 1:N]
        for i in 1:N
            if (ND == 1)
                x[i,1] = G .* dot(dL[1:i,1], v[end-i+1:end])
            elseif (ND == 2)
                x[i,1] = G .* dot(dL[1:i,1], v[end-i+1:end])
                x[i,2] = G .* dot(dL[1:i,2], v[end-i+1:end])
            end
        end
    end

    # MATRIX METHOD: O(N^2) SPACE
    if (method == :matrix)
        v_mat = zeros(N,N)

        for j = 1:N
            for i = 1:j
                v_mat[i,j] = (j - i + 1)^(p+1.0) - ((j - i + 1)-1)^(p+1.0)
            end
        end

        if (ND == 1)
            x = G .* (dL[2:end,1]' * v_mat)'
        elseif (ND == 2)
            x[:,1] = G .* (dL[2:end,1]' * v_mat)' 
            x[:,2] = G .* (dL[2:end,2]' * v_mat)' 
        end 

    end
    
    # TODO: NORMALISATION FACTOR BASED ON FBM
    # x = x / dt

    if (uncorNoise)
        return x, dL
    end
    return x
end

# Ornstein-Uhlenbeck Process
function OUP(θ=1, σ=1,dt=1,N=1000)
    dist = Normal()
    x = zeros(N+1)
    i = 1
    while i < N+1   
        x[i+1] = (1-θ*dt)* x[i] + σ*sqrt(dt)*rand(dist) 
        i += 1
    end
    f = CairoMakie.Figure()
    ax = Axis(f[1,1], title="Ornstein-Uhlenbeck Process", xlabel="Time", ylabel="Displacement")
    lines!(ax,  x)
    display(f)
    return x
end

#  OU Process with Correlated Levy Noise
function OULP(α=1, β=0.5, θ=1, σ=1, γ=1, dt=1,N=1000)
    dist = Stable(α, β)
    x = zeros(N+1)
    ξ = zeros(N+1)
    i = 1
    while i < N+1   
        x[i+1] = (1-θ*dt)*x[i] + γ*ξ[i]
        ξ[i+1] = (1-γ*dt)*ξ[i] + σ*sqrt(dt)*rand(dist) 
        i += 1
    end
    f = CairoMakie.Figure()
    ax = Axis(f[1,1], title="Ornstein-Uhlenbeck-Levy Process with α=$(α)", xlabel="Time", ylabel="Displacement")
    lines!(ax,  x)
    display(f)
    return x
end

function isotropicNormal(N)
    # Option 1: Sample x and y individually
    x1 = randn(N)
    y1 = randn(N)

    r = abs.(randn(N))
    θ = (2*π) .* rand(N)
    x2 = r .* cos.(θ)
    y2 = r .* sin.(θ)

    f1 = CairoMakie.Figure()
    ax1 = Axis(f1[1,1], title="Cartesian Sampling", xlabel="x", ylabel="y",aspect=AxisAspect(1))
    CairoMakie.scatter!(ax1, x1, y1, color=:blue, markersize=2)
    f1 |> display
    f2 = CairoMakie.Figure()
    ax2 = Axis(f2[1,1],title = "Polar Sampling", xlabel="x", ylabel="y",aspect=AxisAspect(1))
    CairoMakie.scatter!(ax2, x2, y2, color=:blue, markersize=2)
    f2 |> display
end



#-----| NOISE PROCESSES FROM THE Brownian PACKAGE ------------------------------------
#  https://github.com/uow-statistics/Brownian.jl.git

# Davies-Harte Method (FFT)
function fBM_DH(H; N, tspan)
    t = LinRange(0, tspan, N)
    p = FBM(t, H)
    return rand(p)
end

# Cholesky Decomposition (Covariance Matrix)
function fBM_CH(H; N, tspan)
    t = LinRange(0, tspan, N)
    p = FBM(t, H)
    return rand(p, method=:chol)
end

# Riemann-Liouville Method (Integral Method)
function fBM_RL(H; N=nothing, tspan=nothing, t=nothing)
    # For this method, tspan must be 1
    # t = LinRange(0, 1, N)
    p = FBM(0:1/2^10:1, H)
    # p = FBM(t, H)
    # We are going to require that the time points are evenly spaced to keep things simple
    dt = unique(diff(p.t))
    if length(dt) != 1
        error(
            "For this simulation technique, the fBm process must be sampled across a uniform temporal grid",
        )
    end

    dt = dt[1]

    w_mat = zeros(p.n - 1, p.n - 1) # Prealocate a weight matrix. Note that this will be upper triangular

    for j = 1:p.n-1
        for i = 1:j
            w_mat[i, j] = weight(p.h, (j - i + 1) * dt, dt)
        end
    end

    # Multiply a vector of white noise by the weight matrix and scale appropriately.
    X = dropdims(sqrt(2) * (randn(p.n - 1)' * w_mat), dims = (1)) .* sqrt(dt)

    insert!(X, 1, 0.0)  # Snap to zero. Not clear if this causes a discontinuity in the correlation structure.
    #It is worth considering alternative for the above line: generate path X of length p.n, then let X = X-X[1].

    return X
end

# ----------------------------------------------------------------

function wMatrix(H, N, dt)
    w_mat = zeros(N - 1, N - 1) # Prealocate a weight matrix. Note that this will be upper triangular
    for j = 1:N-1
        for i = 1:j
            w_mat[i, j] = weight(H, (j - i + 1) * dt, dt)
        end
    end
    return w_mat
end

function wMatrix2(H, N, dt)
    w_mat = zeros(N - 1, N - 1) # Prealocate a weight matrix. Note that this will be upper triangular
    for j = 1:N-1
        for i = 1:j
            w_mat[i, j] = weight2(H, (j - i + 1), dt)
        end
    end
    return w_mat
end

function weight(H::Float64, t::Float64, dt::Float64)
    nom = t^(H + 0.5) - (t - dt)^(H + 0.5)
    denom = gamma(H + 0.5) * (H + 0.5) * dt
    return nom / denom
end

function weight2(H::Float64, i::Int64, dt::Float64)
    nom = (i^(H + 0.5) - (i - 1)^(H + 0.5))*(dt^(H-0.5))
    denom = gamma(H + 0.5) * (H + 0.5)
    return nom / denom
end


#-------------------------------------------------------------------------------------
function plotNoise(noise, ND, noiseType::String; H=nothing, α=2.0, dt=nothing, t=nothing, tspan=nothing)
    if (dt === nothing)
        error("Must provide a timestep dt for PSD")
    end
    if (t === nothing)
        t = collect(0:dt:tspan)
        if (tspan === nothing)
            error("Must provide either a vector of times t or a simulation time tspan")
        end
    end
    
    f = CairoMakie.Figure(size=(800,2400))

    if (H === nothing)
        # Regular (Integer-H) noise
        titleString ="$(ND)D $(noiseType)" 
    else
        # Fractional Process
        titleString ="$(ND)D $(noiseType) with H = $(H) and α = $(α)" 
    end

    if ND == 1
        ax = Axis(f[1,1], title=titleString, xlabel="Time", ylabel="Displacement")
        lines!(ax, t, noise[:,1], color=:blue, linewidth=1)
    elseif ND == 2
        ax = Axis(f[1,1], title=titleString, xlabel="x", ylabel="y")
        lines!(ax, noise, color=:blue, linewidth=1)
    end

    ax_MSD = Axis(f[2, 1],xscale=log10, yscale=log10, title="Mean Absolute Displacement", xlabel="Time", ylabel="Mean Absolute Displacement")
    MAD(noise, ax_MSD)

    ax_PSD = Axis(f[3,1], xscale=log10, yscale=log10, title="PSD of $(noiseType) with H = $(H) and α = $(α)", xlabel="Frequency", ylabel="Power/Frequency")
    PSD(noise, dt, ax_PSD)

    ax_ACF = Axis(f[4,1], title="Autocorrelation Function of $(noiseType)", xlabel="Time", ylabel="Autocorrelation")
    AutoCor(noise, noiseType; dt=dt, tspan=tspan, ax=ax_ACF)


    display(f)
end

# Creates a vector of noise increments from the noise process x
# Sets the first increment to zero to match the size of the input and output matrices
function getIncrements(x)
    return [0;diff(x, dims=1)]
end

# Coverts a 2×n matrix of noise values into a vector of length n whose elements
#   are 4×4 matrices, then creates a NoiseGrid object.
function noiseMatToNoiseGrid(X; dt=nothing, tspan=nothing, t=nothing, )
    if (t === nothing)
        if (tspan === nothing || dt === nothing)
            error("You must provide either a vector of times t, or a simulation time tspan and a time step dt")
        end
        t = collect(0:dt:tspan)
    end
    # Add columns for the momentum coords (no noise)
    X_padded = hcat(X,zeros(size(X)))
    # -- Convert Matrix to Vector of 4x4 Matrices (Noise on momenta) --
    # Convert to vector of noise vectors

    Main.@infiltrate
    v = vec([X_padded[i,:] for i in axes(X_padded, 1)])

    v_grid = NoiseGrid(t, v)

    return v_grid
end

# Extracts the matrix of noise values from a NoiseGrid object
# Acts as an inverse function to noiseMatToNoiseGrid
# Specify mom=false to suppress momentum noise, and return just x noise
function noiseGridToNoiseMat(G::NoiseGrid; params=:all)
    # Vector of Diagonal Matrices
    W = G.W
    # Output noise matrix
    # Number of time points
    N = size(W,1)
    # Dimension of the system / diagonal matrices (position coords + momenta)
    d = size(W[1],1)
    X = zeros(N, d)
    for i in 1:N
        X[i,:]= ones(d)' * W[i]
    end
    M = Int(size(X,2)/2)+1
    if (params == :position)
        # Extract first d/2 columns (position coords)
        return X[:,1:M-1]
    elseif (params == :momentum)
        # Extract last d/2 columns (momenta)
        return X[:,M:end]
    end
    return X
end
