"""
    OhmicBath

Ohmic bath object to hold a particular parameter set.

**Fields**
- `η` -- strength.
- `ωc` -- cutoff frequence.
- `β` -- inverse temperature.
"""
struct OhmicBath <: AbstractBath
    η::Float64
    ωc::Float64
    β::Float64
end

"""
    Ohmic(η, fc, T)

Construct OhmicBath from parameters with physical unit: `η`--unitless interaction strength; `fc`--cutoff frequency in GHz; `T`--temperature in mK.
"""
function Ohmic(η, fc, T)
    ωc = 2 * π * fc
    β = temperature_2_beta(T)
    OhmicBath(η, ωc, β)
end

function Base.show(io::IO, ::MIME"text/plain", m::OhmicBath)
    print(
        io,
        "Ohmic bath instance:\n",
        "η (unitless): ",
        m.η,
        "\n",
        "ωc (GHz): ",
        m.ωc / pi / 2,
        "\n",
        "T (mK): ",
        beta_2_temperature(m.β),
    )
end

"""
    γ(ω, params::OhmicBath)

Calculate Ohmic spectrum density, defined as a full Fourier transform on the bath correlation function.
"""
function γ(ω, params::OhmicBath)
    if isapprox(ω, 0.0, atol = 1e-9)
        return 2 * pi * params.η / params.β
    else
        return 2 * pi * params.η * ω * exp(-abs(ω) / params.ωc) /
               (1 - exp(-params.β * ω))
    end
end

spectrum(ω, bath::OhmicBath) = γ(ω, bath)

"""
    S(w, bath::OhmicBath; atol=1e-7)

Calculate the Lamb shift of Ohmic spectrum. `atol` is the absolute tolerance for Cauchy principal value integral.
"""
S(w, bath::OhmicBath; atol = 1e-7) =
    lambshift(w, (ω) -> γ(ω, bath), atol = atol)

"""
    correlation(τ, bath)

Calculate the correlation function of `bath`.
"""
function correlation(τ, bath::OhmicBath)
    x2 = 1 / bath.β / bath.ωc
    x1 = 1.0im * τ / bath.β
    bath.η * (trigamma(-x1 + 1 + x2) + trigamma(x1 + x2)) / bath.β^2
end

"""
    polaron_correlation(τ, a, params::OhmicBath)

Calculate the polaron transformed correlation function of Ohmic bath. `a` is the effective system bath coupling strength. It is the Hamming distance of two energy levels with respect to the system bath coupling operator.
"""
function polaron_correlation(τ, a, params::OhmicBath)
    res = (1 + 1.0im * params.ωc * τ)^(-a * params.η)
    if !isapprox(τ, 0, atol = 1e-9)
        x = π * τ / params.β
        res *= (x / sinh(x))^(a * params.η)
    end
    res
end

function build_redfield(
    coupling,
    unitary,
    tf::Real,
    bath::OhmicBath;
    atol = 1e-8,
    rtol = 1e-6,
)
    cfun(s) = correlation(s * tf, bath)
    Redfield(coupling, unitary, cfun, atol = atol, rtol = rtol)
end

function build_redfield(
    coupling,
    unitary,
    tf::UnitTime,
    bath::OhmicBath;
    atol = 1e-8,
    rtol = 1e-6,
)
    cfun(t) = correlation(t, bath)
    Redfield(coupling, unitary, cfun, atol = atol, rtol = rtol)
end

function info_freq(bath::OhmicBath)
    println("ωc (GHz): ", bath.ωc / pi / 2)
    println("T (GHz): ", temperature_2_freq(beta_2_temperature(bath.β)))
end

function build_davies(coupling, bath::OhmicBath, ω_range, lambshift)
    if lambshift == true
        if isempty(ω_range)
            S_loc = (ω) -> S(ω, bath)
        else
            s_list = [S(ω, bath) for ω in ω_range]
            S_loc = construct_interpolations(ω_range, s_list)
        end
    else
        S_loc = (ω) -> 0.0
    end
    DaviesGenerator(coupling, (ω) = γ(ω, bath), S_loc)
end
