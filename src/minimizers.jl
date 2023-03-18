include("euler.jl")
"""
    J1(γ, Is, Rs, p, tmax=119, λ=1; I, y) 

J minimizer for p=[1,2]
"""
@inline function J1(γ, Is, Rs, p, tmax::Real=119, λ::Real=1; I::Function, y::Vector{<:Real})::Real # generic vars
    f1(t) = abs(I(t) - Is[t])^p
    f2(t) = abs(y[t] - γ * Rs[t])^p
    return sum(f1, 1:tmax) + λ * sum(f2, 1:tmax)
end

"""
    J2(γ, Is, Rs, tmax=119, λ=1; I, y) 

J minimizer for p=∞
"""
function J2(γ, Is, Rs, tmax::Real=119, λ::Real=1; I::Function, y::Vector{<:Real})::Float64
    f1(t) = abs(I(t) - Is[t])
    f2(t) = abs(y[t] - γ * Rs[t])
    return maximum(f1, 1:tmax) + λ * maximum(f2, 1:tmax)
end

"""
    Jplot(α, β, N̂, p)

Plotting J for p =[1,2].
"""
function Jplot(α, β, N̂, p; If::Function)
    I, R = euler(α, β, N̂)[2, 3]
    return J1(γ̂, I, R, p; I=If)
end

"""
    Jplotinf(α, β, N̂)

Plotting J for p =[1,2].
"""
function Jplotinf(α, β, N̂; If::Function)
    I, R = euler(α, β, N̂)[2, 3]
    return J2(γ̂, I, R, I=If)
end