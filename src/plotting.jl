using LaTeXStrings, Plots
export plotir

function plotir(
    I::Function,
    Is::Vector{<:Real},
    Y::Vector{<:Real},
    city::String,
    p::Int64,
    names::Vector{<:AbstractString})
    plot(0:119, I, labels="Observed")
    plot!(0:119, Is, label="Simulated")
    display(plot!(title="$(city) Observed vs Simulated Rate of Infections" * "\n" * L"p=%$(p)"))
    Plots.svg(names[1])
    plot(0:119, Y, labels="Observed")
    plot!(0:119, γ̂ * Rs, label="Simulated")
    display(plot!(title="$(city) Observed vs Simulated Deaths" * "\n" * L"p=%$(p)"))
    Plots.svg(names[2])
end
function plotir(
    I::Function,
    Is::Vector{<:Real},
    Y::Vector{<:Real},
    city::String,
    p::Int64)
    plot(0:119, I, labels="Observed")
    plot!(0:119, Is, label="Simulated")
    display(plot!(title="$(city) Observed vs Simulated Rate of Infections" * "\n" * L"p=%$(p)"))
    plot(0:tmax-1, Y, labels="Observed")
    plot!(0:tmax-1, γ̂ * Rs, label="Simulated")
    display(plot!(title="$(city) Observed vs Simulated Deaths" * "\n" * L"p=%$(p)"))
end
