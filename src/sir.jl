# activate env and instantiate pkgs
using Pkg
Pkg.activate("./p1")
Pkg.instantiate()
include("gamma.jl")
include("minimizers.jl")
include("euler.jl")
include("plotting.jl")
include("convenience.jl")
macro estimates(d)
    esc(:((α̂, β̂, r0̂, N̂, γ̂) = argmin($(d))))
end

using CSV, DataFrames, Plots
df = CSV.read("./resources/data.csv", DataFrame) # entire df
infected_nyc = values(df[1, 13:end]); # vector of infected numbers
deaths_nyc = values(df[2, 13:end]); # vector of death numbers
infected_dates_nyc = names(df[1, 13:end]); # vector of infected_dates
population_nyc = df.Population[1];
t0_nyc = findfirst([x >= 5 for x in infected_nyc])
@definei "nyc" infected_nyc t0_nyc
@definey "nyc" deaths_nyc t0_nyc
Y1 = first(deaths_nyc[t0_nyc:end], 120);

Jd1 = @jdict
@time begin
    Threads.@threads for α in 0.05:0.01:0.2
        Threads.@threads for r0 in 1.5:0.1:1.9
            Threads.@threads for nn in 2:0.5:10
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Inyc)
                γ̂ = optGammap1(Rs, Ynyc)
                Jd1[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1; I=Inyc, y=Y1)
            end
        end
    end
end

@estimates Jd1
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd1)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=1")
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=1")


Jd2 = @jdict
@time begin
    Threads.@threads for α in 0.05:0.01:0.2
        Threads.@threads for r0 in 1.5:0.1:1.9
            Threads.@threads for nn in 2:0.5:10
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Inyc)
                γ̂ = optGammap2(Rs, Ynyc)
                Jd2[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 2; I=Inyc, y=Y1)
            end
        end
    end
end
@estimates Jd2
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd2)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=2")
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=2")

Jd3 = @jdict
@time begin
    for α in 0.05:0.01:0.2
        for r0 in 1.5:0.1:1.9
            for nn in 2:0.5:10
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Inyc)
                γ̂ = optGammapinf(Rs, Ynyc)
                Jd3[(α, β, r0, N, γ̂)] = J2(γ̂, Is, Rs; I=Inyc, y=Y1)
            end
        end
    end
end
@estimates Jd3
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd3)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞")
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=∞")

"""DC"""

dfdc = CSV.read("./resources/validation.csv", DataFrame) # entire df
idc = values(dfdc[1, 13:end]);
infected_dates_dc = names(dfdc[1, 13:end]); # vector of infected_dates
population_dc = dfdc.Population[1];
deathsdc = values(dfdc[2, 13:end]); # vector of death numbers
# parameters
t0 = findfirst([x >= 5 for x in idc])

@definei "dc" idc t0
@definey "dc" deathsdc t0
Y2 = first(deathsdc[t0:end], 120);

Jd1dc = @jdict
@time begin
    Threads.@threads for α in 0.05:0.01:0.2
        Threads.@threads for r0 in 1.5:0.1:1.9
            Threads.@threads for nn in 2:0.5:10
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Idc)
                γ̂ = optGammap1(Rs, Ydc)
                Jd1dc[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd1dc
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd1dc)
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=1")
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=1")


Jd2dc = @jdict
@time begin
    Threads.@threads for α in 0.05:0.01:0.2
        Threads.@threads for r0 in 1.5:0.1:1.9
            Threads.@threads for nn in 2:0.5:10
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Idc)
                γ̂ = optGammap2(Rs, Ydc)
                Jd2dc[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 2; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd2dc
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd2dc)
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=2")
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=2")

Jd3dc = @jdict
@time begin
    for α in 0.05:0.01:0.2
        for r0 in 1.5:0.1:1.9
            for nn in 2:0.5:10
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N; I=Idc)
                γ̂ = optGammapinf(Rs, Ydc)
                Jd3dc[(α, β, r0, N, γ̂)] = J2(γ̂, Is, Rs; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd3dc
@show α̂, β̂, r0̂, N̂, γ̂;
Jd3dc[argmin(Jd3dc)]
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞")
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=∞")


"""Predictions 40 days"""
Jd1b = @jdict
@time begin
    Threads.@threads for α in 0.17:0.01:0.19
        Threads.@threads for r0 in 1.9:0.1:2.5
            Threads.@threads for nn in 3:0.5:5
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Inyc)
                γ̂ = optGammap1(Rs, Ynyc, 40)
                Jd1b[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1, 40; I=Inyc, y=Y1)
            end
        end
    end
end

@estimates Jd1b
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd1b)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
tn = findfirst(x -> x <= 5, Is)
infected_dates_nyc[tn]
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=1")
vline!([40], labels=L"t=40", linestyle=:dot)
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=1")
vline!([40], labels=L"t=40", linestyle=:dot)


Jd2b = @jdict
@time begin
    Threads.@threads for α in 0.2:0.01:0.3
        Threads.@threads for r0 in 1.9:0.1:2.5
            Threads.@threads for nn in 4:0.5:5.5
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Inyc)
                γ̂ = optGammap2(Rs, Ynyc, 40)
                Jd2b[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 2, 40; I=Inyc, y=Y1)
            end
        end
    end
end
@estimates Jd2b
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd2b)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
tn = findfirst(x -> x <= 5, Is)
infected_dates_nyc[tn]
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
vline!([40], labels=L"t=40", linestyle=:dot)
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=2")
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=2")
vline!([40], labels=L"t=40", linestyle=:dot)

Jd3b = @jdict
@time begin
    for α in 0.2:0.01:0.3
        for r0 in 1.9:0.1:2.5
            for nn in 5:0.5:8
                β = r0 * α
                N = population_nyc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Inyc)
                γ̂ = optGammapinf(Rs, Ynyc, 40)
                Jd3b[(α, β, r0, N, γ̂)] = J2(γ̂, Is, Rs, 40; I=Inyc, y=Y1)
            end
        end
    end
end
@estimates Jd3b
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd3b)
Is, Rs = euler(α̂, β̂, N̂; I=Inyc)[2:3];
tn = findfirst(x -> x <= 5, Is)
infected_dates_nyc[tn]
plot(0:119, Inyc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="NYC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞")
vline!([40], labels=L"t=40", linestyle=:dot)
plot(0:119, Y1, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="NYC Observed vs Simulated Deaths" * "\n" * L"p=∞")
vline!([40], labels=L"t=40", linestyle=:dot)

Jd1dcb = @jdict
@time begin
    Threads.@threads for α in 0.07:0.01:0.25
        Threads.@threads for r0 in 1.7:0.1:2.1
            Threads.@threads for nn in 2.5:0.5:4
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Idc)
                γ̂ = optGammap1(Rs, Ydc, 40)
                Jd1dcb[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 1, 40; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd1dcb
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd1dcb)
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
tn = findfirst(x -> x <= 5, Is)
infected_dates_dc[tn]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
vline!([40], labels=L"t=40", linestyle=:dot)
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=1")
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=1")
vline!([40], labels=L"t=40", linestyle=:dot)


Jd2dcb = @jdict
@time begin
    Threads.@threads for α in 0.12:0.01:0.15
        Threads.@threads for r0 in 1.5:0.1:2.2
            Threads.@threads for nn in 3:0.5:4.5
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Idc)
                γ̂ = optGammap2(Rs, Ydc, 40)
                Jd2dcb[(α, β, r0, N, γ̂)] = J1(γ̂, Is, Rs, 2, 40; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd2dcb
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd2dcb)
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
tn = findfirst(x -> x <= 5, Is)
# infected_dates_dc[tn]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=2")
vline!([40], labels=L"t=40", linestyle=:dot)
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=2")
vline!([40], labels=L"t=40", linestyle=:dot)

Jd3dcb = @jdict
@time begin
    for α in 0.13:0.01:0.15
        for r0 in 1.8:0.1:2.5
            for nn in 3.5:0.5:4.5
                β = r0 * α
                N = population_dc * nn / 100
                Ss, Is, Rs = euler(α, β, N, 40; I=Idc)
                γ̂ = optGammapinf(Rs, Ydc, 40)
                Jd3dcb[(α, β, r0, N, γ̂)] = J2(γ̂, Is, Rs, 40; I=Idc, y=Y2)
            end
        end
    end
end
@estimates Jd3dcb
@show α̂, β̂, r0̂, N̂, γ̂;
findmin(Jd3dcb)
Is, Rs = euler(α̂, β̂, N̂; I=Idc)[2:3];
tn = findfirst(x -> x <= 5, Is)
# infected_dates_dc[tn]
plot(0:119, Idc, labels="Observed")
plot!(0:119, Is, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Infection \quad Rate}")
plot!(title="DC Observed vs Simulated Rate of Infections" * "\n" * L"p=∞")
vline!([40], labels=L"t=40", linestyle=:dot)
plot(0:119, Y2, labels="Observed")
plot!(0:119, γ̂ * Rs, label="Simulated")
xlabel!(L"t")
ylabel!(L"\textnormal{Deaths}")
plot!(title="DC Observed vs Simulated Deaths" * "\n" * L"p=∞")
vline!([40], labels=L"t=40", linestyle=:dot)