export definei, definey, jdict, estimate

"""
    @definei name infections t0

Convenience macro to define I(t).
"""
macro definei(name, v, t0)
    s = Symbol("I$(name)")
    esc(:($(s)(t::Int) = $(v)[t+$(t0)+7-1] - $(v)[t+$(t0)-7-1]))
end

"""
    @definey name deaths t0

Convenience macro to define Y(t).
"""
macro definey(name, v, t0)
    s = Symbol("Y$(name)")
    esc(:($(s)(t::Int) = $(v)[t+$(t0)]))
end

macro jdict()
    (Dict{NTuple{5,Float64},Float64}())
end
