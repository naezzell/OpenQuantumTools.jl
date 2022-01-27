"""
$(SIGNATURES)

Solve the GAME equation defined by `A` for a total evolution time `tf`.

...
# Arguments
- `A::Annealing`: the Annealing object.
- `tf::Real`: the total annealing time.
- 'npts::Int': number of equally-spaced time samples
- 'tol::Float': desired error tolerance in solution
- 'n_trunc::Int': desired truncation; defaults to full spectrum
...
"""
function solve_game(
    A::Annealing,
    tf::Real;
    npts=1000,
    tol=1e-6, 
    n_trunc=nothing)
    """
    Wrapper that is exposed to the user. Calls solve_constant_lindblad_game
    if Lindblad operators don't depend on time. Otherwise, calls the more
    general solver using the jump operator.
    """
    constant_couplings_only = true
    for interaction in A.interactions
        coupling = interaction.coupling
        if !isa(coupling, ConstantCouplings)
            constant_couplings_only = false
        end
    end
    if constant_couplings_only === true
        result = solve_constant_lindblad_game(A::Annealing, tf::Float64, tol::Float64, npts::Int64, n_trunc::Int64)
    else
        result = "Non-constant Lindblad operators currently not supported."
    end

    return result
end

function solve_constant_lindblad_game(A::Annealing, tf::Float64, tol::Float64, npts::Int64, n_trunc::Int64)
    """
    Sets up and solves GAME equation with constant Lindblad operators.
    """
    # step 0: Unwrap desired quantities from annealing object
    H = A.H
    u0 = A.u0
    if !isa(u0, Matrix)
        ρi = reshape(u0 ⊗ u0, (length(u0), length(u0)))
    else
        ρi = u0
    end
    #TODO: Add support for multiple interactions
    interactions = A.interactions
    coupling = interactions[1].coupling
    L = coupling.mats[1]
    #TODO: Add support for T > 0 bath
    bath = interactions[1].bath

    # step 1: diagonlize H with desired truncation
    E, _ = H.EIGS(H, 0, n_trunc)
    E = reverse(E)

    # step 2: compute time-spectral density with relevant inputs
    Γ, J, _ = compute_ohmic_timed_spectral_density(E, bath.ωc, bath.η)

    # step 3: compute Lindblad and HLS operators from interactions
    M = compute_lindblad_operator(L, J)
    Hls = compute_lamb_shift(L, Γ)

    # step 4: solve time-independent game
    ρ_list, E_list = solve_time_ind_game(E, M, Hls, ρi, tf, npts, tol)

    return ρ_list, E_list
end

function solve_time_ind_game(E, M, Hls, ρi, tf, npts, epsm)
    """
    Actually solves the time-indepenent Lindblad operator GAME equation
    by iteratively using better Taylor approximations of ρ̇.
    """
    # set up static variables
    Heff = -1im * (diagm(E) + Hls) # (half of) unitary contribution
    acomm = -0.5 * M' * M # (half of) anticommutator
    dt = tf / npts
    # iterative over times and compute ρ(t)
    ρ = ρi
    ρ_list = Array([ρi])
    E_list = Array([real(tr(ρi * diagm(E)))])
    for j=1:npts
        m_der_ρ = ρ # m^{th} derivative of ρ (starting with 0)
        r_ρ = ρ # the running approximation of ρ
        epsme = 1
        m = 1
        while epsme > epsm
            # compute first half of m^th derivative (i.e. \mathcal{L}(ρ))
            m_der_ρ = 0.5 * M * m_der_ρ * M' + (Heff + acomm) * m_der_ρ
            m_der_ρ = m_der_ρ + m_der_ρ'
            # refine approx of ρ with m^th Taylor term
            r_ρ = r_ρ + (m_der_ρ * dt^m) / factorial(m)
            # compute 2-norm of added term
            epsme = norm(m_der_ρ * dt^m / factorial(m), 2)
            m += 1
        end
        ρ = r_ρ
        push!(ρ_list, ρ)
        energy = real(tr(ρ * diagm(E)))
        push!(E_list, energy)
    end
    return ρ_list, E_list
end

function compute_ohmic_timed_spectral_density(E, ωc, alp)
    """
    Computes Ohmic-timed spectral density in the time-independent
    Lindblad operator case where only relevant frequencies are
    Bohr frequencies.
    """
    # compute # of levels from E
    n_levels = length(E)
    # compute Bohr frequenices ω_{nm} = En - Em
    ω = E .- E'
    # extract unique frequenices
    unique_ω = sort!(unique(ω))
    # renormalize Bohr freqs with respect to bath cut-off
    Er = unique_ω / ωc
    # indices exlucding 0 Bohr frequency to prevent blowing up
    nω = length(unique_ω)
    nz_idxs = map(Int64, vec([1:(nω - 1) / 2 (nω + 3) / 2:nω]))

    # compute the principle spectral density (S)
    S = zeros(nω, 1);
    S[nz_idxs] = Er[nz_idxs] .* exp.(-Er[nz_idxs]) .* expinti.(Er[nz_idxs])
    S = -alp * ωc * (1 .- S) / 2

    # now compute spectral density (J) where it's positive
    J = zeros(nω, 1)
    pos_idxs = map(Int64, (nω + 3) / 2 : nω)
    J[pos_idxs] = 0.5*pi*alp*ωc * Er[pos_idxs] .* exp.(-Er[pos_idxs])

    # compute time-spectral density, Γ, in column vector form
    # unique_ω[ωidxs] == vec(ω) evalues to 'true'
    ωidxs = [findfirst(x->x==vec(ω)[i], unique_ω) for i in 1:length(ω)]
    Γ = Transpose(reshape(J[ωidxs] + 1im * S[ωidxs], n_levels, n_levels))
    # extact J and S from Γ to ensure compatibility
    J = real(Γ)
    S = imag(Γ)
    
    return Γ, J, S

function compute_lindblad_operator(L, J)
    """
    Computes time-independent Lindblad operators.
    """
    # matrix(ized) Lindblad operators
    M = L .* sqrt.(2 * J)

    return M
end

function compute_lamb_shift(L, Γ)
    """
    Computes GAME Lamb shift term.
    """
    Λ = L .* Γ
    LΛ = L * Λ
    Hls = (LΛ - LΛ') / 2im

    return Hls
end