using JuMP, GLPK, Distances
export optGammap1, optGammap2, optGammapinf

"""
    optGammap2(R_sim, yf, tmax=120) 

Calculate optimal γ for p = 2. R_sim is vector of simulated recovered,
yf(t) is a function that calculates deaths at time t.
"""
@inline function optGammap2(R_sim::Vector{T}, yf::Function, tmax::Int=120)::Float64 where {T<:Float64}
    s1(t) = yf(t) * R_sim[t]
    s2(t) = abs(R_sim[t])^2
    gamma_c = sum(s1, 1:tmax) / sum(s2, 1:tmax)
    if gamma_c < 0
        return 0.0
    elseif gamma_c > 1
        return 1.0
    else
        return gamma_c
    end
end

"""
    optGammapinf(R_sim, yf, tmax=120) 

Calculate optimal γ for p = ∞. R_sim is vector of simulated recovered,
yf(t) is a function that calculates deaths at time t.
"""
@inline function optGammapinf(R_sim::Vector{T}, yf::Function, tmax::Int=120)::Float64 where {T<:Float64}
    model = Model()
    set_optimizer(model, GLPK.Optimizer)

    @variable(model, x1 >= 0)
    @variable(model, 1 >= x2 >= 0)

    for t = 1:tmax
        @constraint(model, (-x1) - R_sim[t] * x2 <= -yf(t))
        @constraint(model, (-x1) + R_sim[t] * x2 <= yf(t))
    end

    @objective(model, Min, x1)
    optimize!(model)

    return value.(x2)
end

"""
    optGammapin1(R_sim, yf, tmax=120) 

Calculate optimal γ for p = 1, R_sim is vector of simulated recovered,
yf(t) is a function that calculates deaths at time t.
"""
@inline function optGammap1(R_sim::Vector{T}, yf::Function, tmax::Int=120)::Float64 where {T<:Float64}
    r_vector = yf.(1:tmax) ./ R_sim
    filter!(!isinf, r_vector) # filter out r = x/0
    filter!(x -> x <= 1, r_vector)
    filter!(x -> x >= 0, r_vector)
    vf(r::Float64, tmax)::Float64 = Cityblock()(yf.(1:tmax), r * R_sim)
    # sum(x -> abs(yf.(0:tmax) - x*R_sim))
    md = Dict{Float64,Float64}(r_vector .=> vf.(r_vector, tmax))

    return argmin(md)
end