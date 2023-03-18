
using CSV, DataFrames, LaTeXStrings, Plots, Distances, JuMP, GLPK # import necessary pkgs
df = CSV.read("./resources/data.csv", DataFrame) # entire df

v_infected = values(df[1, 13:end]); # vector of infected numbers
infected_dates = names(df[1, 13:end]); # vector of infected_dates
@assert length(v_infected) == length(infected_dates)
# Exercise 1
## 1)
yd = values(df[2, 13:end]);
v(t::Int)::Int = v_infected[t]
y(t::Int)::Int = yd[t];
findfirst([x >= 5 for x in v_infected])
y_deaths = values(df[2, 13+46-1:end]); # vector of deaths
function I(t::Int, τ::Int=7)::Real
    t0 = 46
    return v(t + t0 + τ) - v(t + t0 - τ)
end
popu = df.Population[1];
tmax = 119;
# df = CSV.read("../resources/data.csv", DataFrame) # entire df

# read the values of infections and deaths from the table
infected = values(df[1, 13:end]); # vector of infected numbers
deaths = values(df[2, 13:end]); # vector of death numbers
infected_dates = names(df[1, 13:end]); # vector of infected_dates
population = df.Population[1];
@assert length(infected) == length(deaths)
# parameters
Vmin = 5
τ0 = 7
λ = 1

t0 = findfirst([x >= Vmin for x in infected])
tmax = 119
# define V, Y, and I for the range of times we are interested in
V(t) = infected[t+t0]
Y(t) = deaths[t+t0]
I(t) = infected[t+t0+7] - infected[t+t0-7]
# function to find the optimal gamma for a given R_sim and p
@inline function optGammap2(R_sim::Vector{T}, yf::Function, tmax::Int=119)::Float64 where {T<:Float64}
    s1(t) = yf(t) * R_sim[t+1]
    s2(t) = abs(R_sim[t+1])^2
    gamma_c = sum(s1, 0:tmax) / sum(s2, 0:tmax)
    if gamma_c < 0
        return 0.0
    elseif gamma_c > 1
        return 1.0
    else
        return gamma_c
    end
end

@inline function optGammapinf(R_sim::Vector{T}, yf::Function, tmax::Int=119)::Float64 where {T<:Float64}
    model = Model()
    set_optimizer(model, GLPK.Optimizer)

    @variable(model, x1 >= 0)
    @variable(model, 1 >= x2 >= 0)

    for t = 0:tmax
        @constraint(model, (-x1) - R_sim[t+1] * x2 <= -yf(t))
        @constraint(model, (-x1) + R_sim[t+1] * x2 <= yf(t))
    end

    @objective(model, Min, x1)
    optimize!(model)

    return value.(x2)
end
@inline function optGammap1(R_sim::Vector{T}, yf::Function, tmax::Int=119)::Float64 where {T<:Float64}
    r_vector = yf.(0:tmax) ./ R_sim
    filter!(!isinf, r_vector) # filter out r = x/0
    filter!(x -> x <= 1, r_vector)
    filter!(x -> x >= 0, r_vector)
    vf(r::Float64, tmax)::Float64 = Cityblock()(yf.(0:tmax), r * R_sim)
    # sum(x -> abs(yf.(0:tmax) - x*R_sim))
    md = Dict{Float64,Float64}(r_vector .=> vf.(r_vector, tmax))

    return argmin(md)
end
# Euler scheme
@inline function euler(alpha::Real, beta::Real, N::Real, to::Int=120; h::Real=0.01)::NTuple{3,Vector{Float64}}
    n = 2

    S_sim = Vector{Float64}(undef, to)
    I_sim = Vector{Float64}(undef, to)
    R_sim = Vector{Float64}(undef, to)

    s = N # introduce Ntotal = sum (sir)
    i = I(0)
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
    while n <= to # pass as var, alternative counter 1/h
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
Y1 = first(y_deaths, 120);
@inline function J1(γ, Is, Rs, p, tmax::Real=119, λ::Real=1)::Real # generic vars
    f1(t) = abs(I(t) - Is[t+1])^p
    f2(t) = abs(y_deaths[t+1] - γ * Rs[t+1])^p
    return sum(f1, 0:tmax) + λ * sum(f2, 0:tmax)
end


Jd1 = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.05:0.01:0.2
    Threads.@threads for r0 in 1.5:0.1:1.9
        Threads.@threads for nn in 2:0.5:10
            β = r0 * α
            N = popu * nn / 100
            Ss, Is, Rs = euler(α, β, N)
            γ̂ = optGammap1(Rs, Y)
            Jd1[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd1)
J_min = Jd1[argmin(Jd1)];
display(L"\text{For}  p=1")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Jd1b = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.17:0.01:0.19
    Threads.@threads for r0 in 1.9:0.1:2.5
        Threads.@threads for nn in 3:0.5:5
            β = r0 * α
            N = popu * nn / 100
            Ss, Is, Rs = euler(α, β, N, 40)
            γ̂ = optGammap1(Rs, Y, 39)
            Jd1b[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd1b)
J_min = Jd1b[argmin(Jd1b)];
display(L"\text{For}  p=1")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = euler(α̂, β̂, N̂);
findfirst(x -> x <= 5, Is)
nm = ["prednycinfcp1", "prednycdtp1"]
plot(0:119, I, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=1"))
Plots.svg(nm[1])
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=1"))
Plots.svg(nm[2])

Jd2b = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.2:0.01:0.3
    Threads.@threads for r0 in 1.9:0.1:2.5
        Threads.@threads for nn in 4:0.5:5.5
            β = r0 * α
            N = popu * nn / 100
            Ss, Is, Rs = euler(α, β, N, 40)
            γ̂ = optGammap2(Rs, Y, 39)
            Jd2b[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 2, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd2b)
J_min = Jd2b[argmin(Jd2b)];
display(L"\text{For}  p=2")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = euler(α̂, β̂, N̂);
nm = ["prednycinfcp2", "prednycdtp2"]
plot(0:119, I, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=2"))
Plots.svg(nm[1])
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=2"))
Plots.svg(nm[2])

function J2(γ, Is, Rs, p, tmax::Real=119, λ::Real=1)::Float64
    f1(t) = abs(I(t) - Is[t+1])
    f2(t) = abs(y_deaths[t+1] - γ * Rs[t+1])
    return maximum(f1, 0:tmax) + λ * maximum(f2, 0:tmax)
end

Jd3b = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.2:0.01:0.3
    Threads.@threads for r0 in 1.9:0.1:2.5
        Threads.@threads for nn in 5:0.5:8
            β = r0 * α
            N = popu * nn / 100
            Ss, Is, Rs = euler(α, β, N, 40)
            γ̂ = optGammapinf(Rs, Y, 39)
            Jd3b[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd3b)
J_min = Jd3b[argmin(Jd3b)];
display(L"\text{For}  p=∞")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = euler(α̂, β̂, N̂);
nm = ["prednycinfcpinf", "prednycdtpinf"]
plot(0:119, I, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞"))
Plots.svg(nm[1])
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=∞"))
Plots.svg(nm[2])

# DC
dfdc = CSV.read("./resources/validation.csv", DataFrame) # entire df

# v_infected = values(df[1, 13:end]) # vector of infected numbers
# infected_dates = names(df[1, 13:end]) # vector of infected_dates
# @assert length(v_infected) == length(infected_dates)
idc = values(dfdc[1, 13:end]);
infected_dates_dc = names(dfdc[1, 13:end]); # vector of infected_dates
population_dc = dfdc.Population[1];
deathsdc = values(dfdc[2, 13:end]); # vector of death numbers
# parameters
Vmin = 5
# τ0 = 7
# λ = 1

t0 = findfirst([x >= Vmin for x in idc])
tmax = 119
y_deaths_dc = values(dfdc[2, 13+55-1:end]); # vector of deaths
# define V, Y, and I for the range of times we are interested in
Vdc(t) = idc[t+7]
Ydc(t) = deathsdc[t+55]
Idc(t) = idc[t+55+7] - idc[t+55-7]
# Euler scheme
@inline function eulerdc(alpha::Real, beta::Real, N::Real, to::Int=120; h::Real=0.01)::NTuple{3,Vector{Float64}}
    n = 2

    S_sim = Vector{Float64}(undef, to)
    I_sim = Vector{Float64}(undef, to)
    R_sim = Vector{Float64}(undef, to)

    s = N # introduce Ntotal = sum (sir)
    i = Idc(0)
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
    while n <= to # pass as var, alternative counter 1/h
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
Y2 = first(y_deaths_dc, 120);
function J1dc(γ, Is, Rs, p, tmax::Int64=119, λ::Real=1)::Real # generic vars
    f1(t) = abs(Idc(t) - Is[t+1])^p
    f2(t) = abs(deathsdc[t+1] - γ * Rs[t+1])^p
    return sum(f1, 0:tmax) + λ * sum(f2, 0:tmax)
end
Jd1dc = Dict{NTuple{5,Float64},Float64}()

Threads.nthreads()
@time begin
    Threads.@threads for α in 0.05:0.01:0.2
        Threads.@threads for r0 in 1.5:0.1:1.9
            Threads.@threads for nn in 2:0.5:10
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = eulerdc(α, β, N)
                γ̂ = optGammap1(Rs, Ydc)
                Jd1dc[(α, β, r0, N, γ̂)] = J1dc(γ̂, Is, Rs, 1)
            end
        end
    end
end

α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd1dc)
J_min = Jd1dc[argmin(Jd1dc)];
display(L"\text{For}  p=1")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")

function Jplotdc(α, β, p)
    S, I, R = eulerdc(α, β, N̂)
    return J1dc(γ̂, I, R, p)
end
display(surface(0.05:0.01:0.2, 1.5:0.01:1.9, (x, y) -> Jplotdc(x, x * y, 1), title=L"DC \quad J(\alpha, \beta, \hat{N}, \hat{\gamma})" * "\n" * L"p=1", xlabel="α", ylabel=L"R_{0}"))
Plots.svg("dcsurp1")

display(surface(0.05:0.01:0.2, 1.5:0.01:1.9, (x, y) -> Jplotdc(x, x * y, 1), title=L"DC \quad J(\alpha, \beta, \hat{N}, \hat{\gamma})" * "\n" * L"p=1", xlabel="α", ylabel=L"R_{0}"))
Plots.svg("dcsurp1")
Ss, Is, Rs = eulerdc(α̂, β̂, N̂);
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=1")
Plots.svg("dcobsp1");
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=1")
Plots.svg("dcdeathsp1");
Jd1bdc = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.07:0.01:0.25
    Threads.@threads for r0 in 1.7:0.1:2.1
        Threads.@threads for nn in 2.5:0.5:4
            β = r0 * α
            N = population_dc * nn / 100
            Ss, Is, Rs = eulerdc(α, β, N, 40)
            γ̂ = optGammap1(Rs, Ydc, 39)
            Jd1bdc[(α, β, r0, N, γ̂)] = J1dc(γ̂, Is, Rs, 1, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd1bdc)
J_min = Jd1bdc[argmin(Jd1bdc)];
display(L"\text{For}  p=1")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = eulerdc(α̂, β̂, N̂);
nm = ["preddcinfcp1", "preddcdtp1"]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=1"))
Plots.svg(nm[1])
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=1"))
Plots.svg(nm[2])

Jd2bdc = Dict{NTuple{5,Float64},Float64}()
Threads.@threads for α in 0.12:0.01:0.15
    Threads.@threads for r0 in 1.5:0.1:2.2
        Threads.@threads for nn in 3:0.5:4.5
            β = r0 * α
            N = population_dc * nn / 100
            Ss, Is, Rs = eulerdc(α, β, N, 40)
            γ̂ = optGammap2(Rs, Ydc, 39)
            Jd2bdc[(α, β, r0, N, γ̂)] = J1dc(γ̂, Is, Rs, 2, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd2bdc)
J_min = Jd2bdc[argmin(Jd2bdc)];
display(L"\text{For}  p=2")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = eulerdc(α̂, β̂, N̂);
nm = ["preddcinfcp2", "preddcdtp2"]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=2"))
Plots.svg(nm[1])
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=2"))
Plots.svg(nm[2])

function J2dc(γ, Is, Rs, tmax::Int=119, λ::Int=1)::Float64
    f1(t) = abs(Idc(t) - Is[t+1])
    f2(t) = abs(y_deaths_dc[t+1] - γ * Rs[t+1])
    return maximum(f1, 0:tmax) + λ * maximum(f2, 0:tmax)
end

Jd3bdc = Dict{NTuple{5,Float64},Float64}()
for α in 0.13:0.01:0.15
    for r0 in 1.8:0.1:2.5
        for nn in 3.5:0.5:4.5
            β = r0 * α
            N = population_dc * nn / 100
            Ss, Is, Rs = eulerdc(α, β, N, 40)
            γ̂ = optGammapinf(Rs, Ydc, 39)
            Jd3bdc[(α, β, r0, N, γ̂)] = J2dc(γ̂, Is, Rs, 39)
        end
    end
end
α̂, β̂, r0̂, N̂, γ̂ = argmin(Jd3bdc)
J_min = Jd3bdc[argmin(Jd3bdc)];
display(L"\text{For}  p=∞")
@show α̂, β̂, r0̂, N̂, γ̂;
display(L"J_{min}\approx %$(round(J_min, digits=3))")
Ss, Is, Rs = eulerdc(α̂, β̂, N̂);
nm = ["preddcinfcpinf", "preddcdtpinf"]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
display(plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞"))
Plots.svg(nm[1])
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
display(plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=∞"))
Plots.svg(nm[2])