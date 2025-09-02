function heatmapconvert(x, y; xminplot=-5, xmaxplot=5, yminplot=-5, ymaxplot=5, bins=200)
    xbins = range(xminplot, xmaxplot; length=bins)
    ybins = range(yminplot, ymaxplot; length=bins)
    heat = zeros(Float64, size(xbins)[1], size(ybins)[1])
    for (xx, yy) in zip(x, y)
        xi = searchsortedfirst(xbins, xx)
        yi = searchsortedfirst(ybins, yy)
        if xi > 1 && xi ≤ size(xbins)[1] && yi > 1 && yi ≤ size(ybins)[1]
            heat[xi, yi] += 1
        end
    end
    return xbins, ybins, heat
end
