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
using Random

bval = 0.01
xmax = 20

# Set input and output folders
input_dir  = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_raw/phasedata/BCircular/b=$(bval)/g=0.1"
output_dir = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop/data/exp_pro/phasedata/BCircular/b=$(bval)/g=0.1"

# Make sure output folder exists
mkpath(output_dir)

# Get all .csv files in the input directory
csv_files = filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true))

αs = collect(1.05:0.05:2.00)
ks = collect(0.05:0.05:1.20)

αsize = length(αs)
ksize = length(ks)
twoDsize = αsize * ksize

# Create a 3x4x5 array of Float64, uninitialized
Results_Array = Array{Float64}(undef, αsize, ksize, 2)
rs = Array{Float64}(undef, αsize, ksize)
for (index, _) in  enumerate(αs)
    Results_Array[index,:,2] = ks
end

for (index, _) in enumerate(ks)
    Results_Array[:,index,1] = αs
end

σ = 1

timespan = 1000.
δt = 0.001
wellseed = 49

Random.seed!(wellseed)
wellsrandomgen = rand(Normal(0, σ * √δt),Int(timespan/δt+1))
wellsrandom = cumsum(wellsrandomgen)

center(k,t) = (xmax ./ 2) .* exp.( im * k * wellsrandom[round(Int,t/δt + 1)] )

for file in ProgressBar(csv_files)

        filename = splitext(basename(file))[1]

        # Extract k parameter using regex
        av = match(r"a=([\d.]+)", filename)
        kv = match(r"k=([\d.]+)", filename)

        if av !== nothing
            aval = parse(Float64, av.captures[1])
        else
            println("No match found")
        end

        if kv !== nothing
            kval = parse(Float64, kv.captures[1])
        else
            println("No match found")
        end

        global aindex = findfirst(x -> x == aval, Results_Array[:,1,1])
        global kindex = findfirst(x -> x == kval, Results_Array[1,:,2])

        # Read CSV into DataFrame
        df = CSV.read(file, DataFrame)

        timestamp = collect(df.timestamp)
        x = collect(df.value1)
        y = collect(df.value2)
        lengthofdata = size(timestamp)

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

        r = sqrt.((xshift.^2) .+ (yshift.^2))

        rstd = r[ r .< 1]

        # Results_Array[aindex, kindex,3] = sqrt(mean(abs2.([meanx,meany])))
        rs[aindex,kindex] = length(rstd) / length(r)
end

plotRSTD = Plots.heatmap(αs,ks,log(rs'),
    title = "Heatmap Time Spent in Inner Brownian Well",
    xlabel = "α Tail Index",
    ylabel = "k Speed Factor",
    # You can customize the color palette. Here we use a different one.
    c = :viridis,
    # Add a color bar label to explain what the colors represent.
    colorbar_title = "Proportion"
)

save(output_dir * "/RSTD.png", plotRSTD)

CSV.write(output_dir * "/Averagealpha.csv", Tables.table(Results_Array[:,:,1]))
CSV.write(output_dir * "/Averagek.csv", Tables.table(Results_Array[:,:,2]))
CSV.write(output_dir * "/RSTD.csv", Tables.table(rs))
