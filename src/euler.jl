export euler
"""
    euler(α, β, N, tmax=120; I, h=0.01)

Solve SIR system using Euler's Method.
I is a function (indexed 1) that computes the infection rates.
"""
@inline function euler(alpha::Real, beta::Real, N::Real, tmax::Int=120; I::Function, h::Real=0.01)::NTuple{3,Vector{Float64}}
    n = 2

    S_sim = Vector{Float64}(undef, tmax)
    I_sim = Vector{Float64}(undef, tmax)
    R_sim = Vector{Float64}(undef, tmax)

    s = N # introduce Ntotal = sum (sir)
    i = I(1)
    r = 0
    N_total = s + i + r

    t = 0

    S_sim[1] = s
    I_sim[1] = i
    R_sim[1] = r
    n = 2
    c_target = 1 / h
    c_current = 2
    _n = 1
    while n <= tmax # pass as var, alternative counter 1/h
        ds = -beta * s * (i / N_total) # replace with Ntotal, same SEIR
        di = beta * s * (i / N_total) - alpha * i
        dr = alpha * i

        s += h * ds
        i += h * di
        r += h * dr
        N_total = s + i + r
        t += h
        if c_current == c_target
            S_sim[n] = s
            I_sim[n] = i
            R_sim[n] = r
            n += 1
            c_current = 0
        end
        c_current += 1
        _n += 1
    end

    return S_sim, I_sim, R_sim
end