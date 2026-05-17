using ITensors
using ITensorMPS
using CSV
using DataFrames
using LinearAlgebra


# psi - mps
# b - bond in which to orthogonalize
function ent_entropy(psi, b)
    psi = orthogonalize(psi, b)
    U, S, V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0
    for i=1:dim(S, 1)
        p = S[i, i]^2
        SvN -= p * log(p)
    end
    return SvN
end
let
    Gamma = 1.0 # 2/3
    N = 64
    sites = siteinds("S=1", N)

    os = OpSum()
    for j = 1:N-1
        os += 1.0, "Sx", j, "Sx", j+1
        os += 1.0, "Sy", j, "Sy", j+1
        os += 1.0, "Sz", j, "Sz", j+1

        os += Gamma, "Sx", j, "Sx", j+1, "Sx", j, "Sx", j+1
        os += Gamma, "Sy", j, "Sy", j+1, "Sy", j, "Sy", j+1
        os += Gamma, "Sz", j, "Sz", j+1, "Sz", j, "Sz", j+1
        os += Gamma, "Sx", j, "Sx", j+1, "Sy", j, "Sy", j+1
        os += Gamma, "Sx", j, "Sx", j+1, "Sz", j, "Sz", j+1
        os += Gamma, "Sy", j, "Sy", j+1, "Sx", j, "Sx", j+1
        os += Gamma, "Sy", j, "Sy", j+1, "Sz", j, "Sz", j+1
        os += Gamma, "Sz", j, "Sz", j+1, "Sx", j, "Sx", j+1
        os += Gamma, "Sz", j, "Sz", j+1, "Sy", j, "Sy", j+1
    end
    os += 2/3, "I", 1

    H_AKLT = MPO(os, sites)

    psi0 = random_mps(sites; linkdims = 10)

    nsweeps = 15
    maxdims = [1, 2, 3, 5, 10, 20, 50]
    cutoff = [1.0e-16]

    Sz0s, Sz1s = [], []
    E0s, E1s = [], []
    Svn0s, Svn1s = [], []
    for m in maxdims
        @show m
        E0, phi0 = dmrg(H_AKLT, psi0; nsweeps, maxdim=m, cutoff)
        @show E0 / N
        # first excited state
        psi1 = random_mps(sites; linkdims = 2)
        # weight : must be large
        E1, phi1 = dmrg(H_AKLT, [phi0], psi1; nsweeps, maxdim=m, cutoff, weight = 400)
        @show E1 / N

        # show they are orthogonal
        @show inner(phi1, phi0)
        @show (E1-E0)/N

        push!(E0s, E0 / N)
        push!(E1s, E1 / N)

        push!(Svn0s, ent_entropy(psi0, floor(Int, N/2)))
        push!(Svn1s, ent_entropy(psi1, floor(Int, N/2)))

        push!(Sz0s, expect(phi0, "Sz"))
        push!(Sz1s, expect(phi1, "Sz"))
    end
    df_energy = DataFrame(bond_dimension = collect(maxdims), E0 = E0s, E1 = E1s, Svn0 = Svn0s, Svn1 = Svn1s)
    fname = "scalar_E_GS_first_excited_per_bond_dimension_N=$(N)_Gamma=$(Gamma).csv"
    CSV.write(fname,  df_energy)
    df_observable = DataFrame(bond_dimension = collect(maxdims), Sz0 = real.(Sz0s), Sz1 = real.(Sz1s))
    fname = "vec_obs_GS_first_excited_per_bond_dimension_N=$(N)_Gamma=$(Gamma).csv"
    CSV.write(fname,  df_observable)
end






