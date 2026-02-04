import Pkg
using DrWatson
path = "/Users/chardiol/Desktop/Theory of Brain/FNS-Julia/JudesFNSWorkshop" # Replace with your own path
quickactivate(path)

using Revise
k = 0.4
g = 0.1
interfile = "/data/exp_raw/phasedata/k=$(k)"

mkpath(path * interfile)

using CSV
using DataFrames
using ProgressBars

loc = path * interfile

csv_files = filter(f -> endswith(f, ".csv.csv"), readdir(loc; join=true))

for file in csv_files
    filename = splitext(basename(file))[1]
    print(filename)
    a = match(r"a=([\d.]+)", filename).captures[1]
    b = match(r"b=([\d.]+)", filename).captures[1]
    print(path * interfile * "/g=$(g)/" * "phaseloop_a=$(a)_b=$(b)")

    mv(file, path * interfile * "/g=$(g)/" * "phaseloop_a=$(a)_b=$(b)" * "csv",force = true)
end
