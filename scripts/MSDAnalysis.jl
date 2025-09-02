import Pkg
using DrWatson
@quickactivate("JudesFNSWorkshop")
using CSV
using DataFrames
using GLM
using Statistics
using Plots
using Base.Threads

# Load CSV

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro"

# Make sure output folder exists
isdir(output_dir) || mkdir(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

FigNumber = 1

for file in csv_files
    # Read CSV into DataFrame
    df = CSV.read(file, DataFrame)

    # Extract columns as arrays
    x = collect(df.value1)
    y = collect(df.value2)

    n = 200
    t = 0.001 * collect(1:n)
    xtest = x[1:end-n]
    m = length(xtest)

    # Preallocate accumulator
    newr2_sum = zeros(n)

    @threads for i in 1:(m)
        xi = x[i]
        yi = y[i]

        dx_view = @view x[i+1:i+n]
        dy_view = @view y[i+1:i+n]

        dx = dx_view .- xi
        dy = dy_view .- yi

        local_r2 = (dx.^2 .+ dy.^2).^(0.5) ./ (length(x)-n)
        @inbounds newr2_sum .+= local_r2

    end



    # Average across all iterations
    means = newr2_sum ./ (length(x)-n)


    df = DataFrame(X = log.(t), Y = log.(means))

    model = lm(@formula(Y ~ X), df)
    Coeffs = GLM.coef(model)
    plot(
        t,
        (t.^Coeffs[2]) .* (ℯ.^Coeffs[1]),
        xscale = :log10,
        yscale = :log10,
        label = "Trendline",       # label for first curve
        linewidth = 2,
        linestyle = :dash,         # dashed line
        color = :firebrick            # color for first curve
    )

    plot!(
        t,
        means,
        label = "Means",           # label for second curve
        linestyle = :solid,        # default, but explicit
        color = :Orange              # color for second curve
    )

    # move legend outside
    plot!(xlabel = "t", ylabel = "⟨r⟩", legend = :outerright)


    endpoint = 0.002

    tpos = endpoint                 # choose a spot
    ypos = (endpoint.^Coeffs[2]) .* (ℯ.^Coeffs[1]) .* 10   # y-value to match scale
    eqn_string = "⟨|r|⟩ ~ t^$(round(Coeffs[2], digits=3))"


    annotate!((tpos, ypos, text(eqn_string, :firebrick, 10, :left)))

    outname = joinpath(output_dir, splitext(basename(file))[1] * "-MD" * ".pdf")

    # Save figure

    savefig(outname)

    println("Fig number: $FigNumber" )
    global FigNumber = FigNumber + 1
end
