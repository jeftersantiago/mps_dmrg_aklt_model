using OMEinsum
using LinearAlgebra
# tensor contractions
using OMEinsum
# comparision
using ITensors, ITensorMPS
#=
    M : an arbitrary MPS with order of legs: left-bottom-right
    returns a left-normalized MPS
=#
function left_canonical_form(M)
    M_copy = copy(M)
    #=
        Left-canonical normalization algorithm:
        Starting at leftmost site of the chain :
            - reshape tensor A[i]
            - apply an SVD
            - update tensor A[i]
            - update tensor A[i+1]
            - run for tensor i = 1 up to i = N - 1
    =#
    for i = 1:(length(M_copy))

        # reshape a d^N tensor into d^[i]*d^[N-i]
        T_aux = M_copy[i]
        T_aux = reshape(T_aux, (size(T_aux)[1] * size(T_aux)[2], size(T_aux)[3]))

        # perform an svd
        F = svd(T_aux, full = false)
        U = F.U
        S = F.S
        Vdag = F.Vt

        # update tensor reshaping A = M U
        M_copy[i] = reshape(U, (size(M_copy[i])[1], size(M_copy[i])[2], size(U)[2]))

        # update next site -> renaming SV^dagger = c^{sigma_2 sigma_3...}
        SVdag = Diagonal(S) * Vdag
        if i < length(M_copy) - 1
            # perform a tensor contraction using OMEisum:
            M_copy[i+1] = ein"ij,jkl->ikl"(SVdag, M_copy[i+1])
        end
    end
    return M_copy
end

function right_canonical_form(M)
    M_copy = copy(M)

    # from rightmost to leftmost
    for i in length(M_copy):-1:1
        T_aux = M_copy[i]
        T_aux = reshape(T_aux, (size(T_aux)[1], size(T_aux)[2] * size(T_aux)[3]))

        F = svd(T_aux, full = false)
        U = F.U
        S = F.S
        Vdag = F.Vt
        M_copy[i] = reshape(Vdag, (size(Vdag)[1], size(M_copy[i])[2], size(M_copy[i])[3]))

        US = U * Diagonal(S)

        if i > 1
            M_copy[i-1] = ein"ijk,kl->ijl"(M_copy[i-1], US)
        end
    end
    return M_copy
end
let
    N = 10
    d = 3
    D = 20

    # Generating a random MPS for OBC.
    Mrand = []
    push!(Mrand, rand(1, d, D))
    for l = 2:N-1
        push!(Mrand, rand(D, d, D))
    end
    push!(Mrand, rand(D, d, 1))

    println("Left canonical form - check")
    L_left = left_canonical_form(Mrand)
    for l in 1:N
        # Mleft -> right-top-left
        # julia wont permit simply using adjoint -> must reshape it via permutedims
        L_dag = conj(permutedims(L_left[l], (3, 2, 1)))

        LdagL  = ein"ijk,kjl->il"(L_dag,L_left[l])
        # checking orthonormality.
        println("l = $(l) : max(L[l]^†*L[l] - I) = $(maximum(abs.(LdagL - I)))")
    end
    println("Dimensions random MPS")
    for i = 1:length(Mrand)
        println("i = $(i) : $(size(Mrand[i]))")
    end
    println("Dimensions left-normalized MPS")
    for i = 1:length(L_left)
        println("i = $(i) : $(size(L_left[i]))")
    end

    println("Right canonical form - check")
    R_right = right_canonical_form(Mrand)
    for l in 1:N
        # Mleft -> right-top-left
        # julia wont permit simply using adjoint -> must reshape it via permutedims
        R_dag = conj(permutedims(R_right[l], (3, 2, 1)))

        RdagR  = ein"ijk,kjl->il"(R_right[l], R_dag)
        # checking orthonormality.
        println("l = $(l) : max(R[l]*R[l]^† - I) = $(maximum(abs.(RdagR - I)))")
    end
    println("Dimensions left-normalized MPS")
    for i = 1:length(R_right)
        println("i = $(i) : $(size(R_right[i]))")
    end

    L_left = left_canonical_form(Mrand)
    R_right = right_canonical_form(L_left)
    println("Dimensions left-right-normalized MPS")
    for i = 1:length(R_right)
        println("i = $(i) : $(size(R_right[i]))")
    end
end






