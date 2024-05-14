#==========================================================================================
                    ODE SOLVER FOR NEUTRINO OSCILLATIONS IN A SUPERNOVA 

DESCRIPTION: This script solves the neutrino oscillation equations in a supernova using the
Sundials.jl package. The vacuum, matter and collective terms are all considered. The script
uses the GL method as the integral solver. CVODE_BDF is used as the ODE solver.
AUTHOR: Tudor Ciobanu
==========================================================================================#

#==========================================================================================
                        IMPORTING PACKAGES AND LIBRARIES
==========================================================================================#
using DifferentialEquations, SpecialFunctions, FastGaussQuadrature, LinearAlgebra, Integrals, Sundials, JLD2, Printf, LaTeXStrings, BenchmarkTools
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

#==========================================================================================
                        SU(3) LIE ALGEBRA STRUCTURE CONSTANTS
                    
DESCRIPTION: Sets the structure constants for the SU(3) Lie algebra. The structure
constants are used to define the cross product of the Lie algebra elements. The
structure constants are defined for the 8 generators of the SU(3) Lie algebra. The
structure constants are totally antisymmetric and are defined as follows:
f₁₂₃ = 2, f₁₄₇ = 1, f₁₆₅ = 1, f₂₄₆ = 1, f₂₅₇ = 1, f₃₄₅ = 1, f₃₇₆ = 1, f₆₇₈ = √3, f₄₅₈ = √3
==========================================================================================#

function set_structure_constants(𝐟, indices, value)
    for (i, j, k) in indices
        𝐟[i, j, k] = value
        𝐟[j, k, i] = value
        𝐟[k, i, j] = value
        𝐟[j, i, k] = -value
        𝐟[i, k, j] = -value
        𝐟[k, j, i] = -value
    end
end

#==========================================================================================
                        SU(3) GENERALIZED CROSS PRODUCT

DESCRIPTION: Defines the generalized cross product for the SU(3) Lie algebra. The cross
product is defined for the 8 generators of the SU(3) Lie algebra. The cross product is
defined as follows:
C = ∑fᵢⱼₖ AᵢBⱼêₖ
where A, B and C are the elements of the SU(3) Lie algebra and fᵢⱼₖ are the structure
constants of the SU(3) Lie algebra.
==========================================================================================#

function cross_prod(A::Array{Float64, 1}, B::Array{Float64, 1}, 𝐟::Array{Float64, 3})
    C = zeros(Float64, 8)
    @inbounds @simd for k in 1:8
        for i in 1:8
            for j in 1:8
                C[k] += 𝐟[i, j, k]*A[i]*B[j]
            end
        end
    end
    return C
end

#==========================================================================================
                                    ω FUNCTION

DESCRIPTION: Defines the neutrino oscillation frequency as a function of the neutrino
energy. The neutrino oscillation frequency is defined as follows:
ω = |Δm²₁₃|/(2E)
where Δm²₁₃ is the mass squared difference between the first and third neutrino mass
eigenstates and E is the neutrino energy.
==========================================================================================#

function omega(E::Float64, Δm²₁₃::Float64)
    return abs(Δm²₁₃)/(2*E)
end

#==========================================================================================
                                COLLECTIVE POTENTIAL

DESCRIPTION: Defines the collective potential as a function of the radial coordinate. The
collective potential is defined as follows:
μ = 0.45e5*(4*r₀^2)/(3*r^2)*(1 - sqrt(1 - r₀^2/r^2) - r₀^2/(4*r^2)) [km⁻¹]
where r₀ links to the radius of the neutrino sphere and r is the radial coordinate.
==========================================================================================#

function μ(r::Float64, r₀::Int)
    return 0.45e5*(4*r₀^2)/(3*r^2)*(1 - sqrt(1 - r₀^2/r^2) - r₀^2/(4*r^2))
end

#==========================================================================================
                                MATTER POTENTIAL

DESCRIPTION: Defines the matter potential as a function of the radial coordinate. The
matter potential is defined as follows:
λ = 1.84e6/(r^(2.4)) [km⁻¹]
where r is the radial coordinate.
==========================================================================================#

function λ(r::Float64)
    return 1.84e6/(r^(2.4))
end

#==========================================================================================
                                SCALING FACTOR
==========================================================================================#

function N(ϵ_ν::Int)
    return (1 + ϵ_ν)^(1 + ϵ_ν)/gamma(1 + ϵ_ν)
end

#==========================================================================================
                                FLUX AVERAGES
==========================================================================================#

function Φ_ν(species::String, Ē::Dict{String, Float64}, L_ν::Float64)
    return L_ν/Ē[species]
end

#==========================================================================================
                             ENERGY DISTRIBUTION
==========================================================================================#
function f_e(E::Float64, Ē_ν::Float64, ϵ_ν::Int, L_ν::Float64)
    f = L_ν*N(ϵ_ν)/(Ē_ν ^2)*(E/Ē_ν)^ϵ_ν*exp(-E/Ē_ν*(ϵ_ν + 1))
    return f
end

#==========================================================================================
                                ω DISTRIBUTION
==========================================================================================#

function f_o(ω::Float64, Ē_ν::Float64, Φ::Float64, ϵ_ν::Int, L_ν::Float64, Δm²₁₃::Float64)
    E = abs(Δm²₁₃)/(2*ω)
    f = E/(Φ*ω)*f_e(abs(E), Ē_ν, ϵ_ν, L_ν)
    return f
end

#==========================================================================================
                        INTEGRATION BOUNDS TRANSFORMATION
==========================================================================================#

function transform_gauss_xw(x::Array{Float64,1}, w::Array{Float64,1}, a::Float64, b::Float64)
    x1 = a .+ (b-a)/2 * (x .+ 1)
    w1 = (b-a)/2 * w
    return (x=x1, w=w1)
end

#==========================================================================================
                            GAUSS-LEGENDRE INTEGRATION
==========================================================================================#

function GL_integral(f::Array{Float64,1}, a::Float64, b::Float64, order::Int)
    x, w = gausslegendre(order)
    T = transform_gauss_xw(x, w, a, b)
    return dot(T.w, f)
end

#==========================================================================================
                            SYSTEM INITIALIZATION
==========================================================================================#

function initialize_system_single(Eᵢ::Float64, Eₖ::Float64, Ebins::Int, Ē::Dict{String, Float64}, ϵ_ν::Int, Δm²₁₃::Float64, L_ν::Float64)
    𝐏 = zeros(Float64, 16, Ebins)

    #= Get the Gauss-Legendre quadrature points and weights based on the energy bins
    and transform them to the ω range [ωₖ, ωᵢ] where ωₖ = ω(Eₖ) and ωᵢ = ω(Eᵢ).    =#
    x, w = gausslegendre(Ebins)
    T = transform_gauss_xw(x, w, omega(Eₖ, Δm²₁₃), omega(Eᵢ, Δm²₁₃)) 
    ω = T.x

    #= Calculate the flux averages for the neutrino species and set up the distribution
    functions.                                                                      =#
    Φ = Φ_ν("nu_e", Ē, L_ν) .+ Φ_ν("nubar_e", Ē, L_ν) .+ 2*Φ_ν("nu_x", Ē, L_ν) .+ 2*Φ_ν("nu_y", Ē, L_ν)
    f_ω = f_o.(ω, Ē["nu_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) .+ f_o.(ω, Ē["nu_x"], Φ, ϵ_ν, L_ν, Δm²₁₃) .+ f_o.(ω, Ē["nu_y"], Φ, ϵ_ν, L_ν, Δm²₁₃)
    f_ω̄ = f_o.(-ω, Ē["nubar_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_x"], Φ, ϵ_ν, L_ν, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_y"], Φ, ϵ_ν, L_ν, Δm²₁₃)

    #= Set up the initial polarization vectors. =#
    for i in 1:Ebins
        𝐏[3,i] = (f_o.(ω[i],  Ē["nu_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) - f_o.(ω[i], Ē["nu_x"], Φ, ϵ_ν, L_ν, Δm²₁₃))/f_ω[i]
        𝐏[8,i] = (f_o.(ω[i], Ē["nu_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) + f_o.(ω[i], Ē["nu_x"], Φ, ϵ_ν, L_ν, Δm²₁₃) - 2*f_o.(ω[i], Ē["nu_y"], Φ, ϵ_ν, L_ν, Δm²₁₃))/(sqrt(3)*f_ω[i])
        𝐏[11,i] = (f_o.(-ω[i], Ē["nubar_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) - f_o.(-ω[i], Ē["nubar_x"], Φ, ϵ_ν, L_ν, Δm²₁₃))/f_ω̄[i]
        𝐏[16,i] = (f_o.(-ω[i], Ē["nubar_e"], Φ, ϵ_ν, L_ν, Δm²₁₃) + f_o.(-ω[i], Ē["nubar_x"], Φ, ϵ_ν, L_ν, Δm²₁₃) - 2*f_o.(-ω[i], Ē["nubar_y"], Φ, ϵ_ν, L_ν, Δm²₁₃))/(sqrt(3)*f_ω̄[i])
    end
    return ω, 𝐏, f_ω, f_ω̄
end

#==========================================================================================
                                    𝐃 MATRIX
==========================================================================================#

function D(𝐏::Array{Float64,2}, ω::Array{Float64,1}, f_ω::Array{Float64,1}, f_ω̄::Array{Float64,1}, order::Int)
    D = zeros(Float64, 8)
    ωᵢ = ω[1]
    ωₖ = ω[end]
    @inbounds for i in 1:8
        D[i] = GL_integral(f_ω.*𝐏[i, :], ωᵢ, ωₖ, order) + GL_integral(f_ω̄.*𝐏[i+8, :], ωᵢ, ωₖ, order)
    end
    return D
end

#==========================================================================================
                                    ODE FUNCTION
==========================================================================================#

function dPdr(du::Array{Float64,2}, u::Array{Float64,2}, p, t::Float64)
    @printf("r = %f\n", t)

    km⁻¹_to_eV = 1.97e-10 # eV

    ω = @views p[1]
    f_ω = @views p[2]
    f_ω̄ = @views p[3]
    𝐁 = @views p[4]
    𝐋 = @views p[5]
    r₀ = @views p[6]
    𝐟 = @views p[7]
    
    Ebins = length(ω)

    𝐃 = D(u, ω, f_ω, f_ω̄, Ebins)
    λ_t = λ(t)
    μ_t = μ(t, r₀)
    @inbounds for i in 1:Ebins
        ω_i = ω[i]/km⁻¹_to_eV
        du[1:8, i] = @views cross_prod((ω_i.*𝐁 .+ λ_t.*𝐋 .+ μ_t.*𝐃), u[1:8, i], 𝐟)
        du[9:16, i] = @views cross_prod((-ω_i.*𝐁 .+ λ_t.*𝐋 .+ μ_t.*𝐃), u[9:16, i], 𝐟)
    end
end

#==========================================================================================
                                        SOLVER
==========================================================================================#
function evolve(Eᵢ, Eₖ, Ebins, umin, umax)
    #= Set up the physical constants and conversion factors. =#
    J_to_eV = 6.25e18 # eV
    s⁻¹_to_eV = 6.58e-16 # eV

    #= Set up the SU(3) Lie algebra structure constants. =#
    𝐟 = zeros(Float64, 8, 8, 8)
    set_structure_constants(𝐟, [(1, 2, 3)], 2)
    set_structure_constants(𝐟, [(1, 4, 7), (1, 6, 5), (2, 4, 6), (2, 5, 7), (3, 4, 5), (3, 7, 6)], 1)
    set_structure_constants(𝐟, [(6, 7, 8), (4, 5, 8)], sqrt(3))

    #= Set up the model parameters. =#
    Δm²₁₃ = -2.458e-3 # eV²
    ϵ = -0.031 # unitless, >0 for normal hierarchy, <0 for inverted hierarchy
    θ₁₂ = 0.58 # radians
    θ₁₃ = 0.15 # radians
    h = sign(Δm²₁₃)
    δ = 0
    Ē_νₑ = 10e6 # eV
    Ē_ν̄ₑ = 15e6 # eV
    Ē_νₓ = 20e6 # eV
    Ē_νᵧ= 20e6 # eV
    ϵ_ν = 3 # unitless
    L_ν = 1.5e44*J_to_eV*s⁻¹_to_eV # eV^2
    r₀ = 10 # km
    ϵ_ν = 3 # unitless

    𝐁 = [ϵ*sin(2*θ₁₂)*cos(θ₁₃),
      0,
      sin(θ₁₃)^2 - ϵ*(cos(θ₁₂)^2 - sin(θ₁₂)^2*cos(θ₁₃)^2),
      (1 - ϵ*sin(θ₁₂)^2)*sin(2*θ₁₃)*cos(δ),
      (1 - ϵ*sin(θ₁₂)^2)*sin(2*θ₁₃)*sin(δ),
      -ϵ*sin(2*θ₁₂)*sin(θ₁₃)*cos(δ),
      -ϵ*sin(2*θ₁₂)*sin(θ₁₃)*sin(δ),
      ϵ/(2*sqrt(3))*(3*cos(θ₁₃)^2 - 1 + 3*sin(θ₁₃)^2*(2*cos(θ₁₂)^2 - 1)) + 1/sqrt(3)*(1 - 3*cos(θ₁₃)^2)]

    𝐋 = [0, 0, 1, 0, 0, 0, 0, 1/sqrt(3)]

    # Energy averages
    Ē = Dict([("nu_e", Ē_νₑ), ("nubar_e", Ē_ν̄ₑ), ("nu_x", Ē_νₓ), ("nubar_x", Ē_νₓ), ("nu_y", Ē_νᵧ), ("nubar_y", Ē_νᵧ)])
    ω, 𝐏₀, f_ω, f_ω̄ = initialize_system_single(Eᵢ, Eₖ, Ebins, Ē, ϵ_ν, Δm²₁₃, L_ν)
    prob = ODEProblem(dPdr, 𝐏₀, (umin, umax), [ω, f_ω, f_ω̄, 𝐁, 𝐋, r₀, 𝐟])
    println("Solving the ODE...")
    sol = solve(prob, CVODE_BDF(linear_solver = :GMRES), abstol = 1e-10, save_everystep = false, saveat = 0.1, maxiters = 1e10)
    return sol
end

# Setting up the allocated arrays
# rho_ee = 1/3 .+ 1/2 .*(sol[3, :, :] .+ 1/sqrt(3) .*sol[8, :, :])
# rho_xx = 1/3 .+ 1/2 .*(-sol[3, :, :] .+ 1/sqrt(3) .*sol[8, :, :])
# rho_yy = 1/3 .+ 1/2 .*(.- 2/sqrt(3) .*sol[8, :, :])
# rho_bar_ee = 1/3 .+ 1/2 .*(sol[11, :, :] .+ 1/sqrt(3) .*sol[16, :, :])
# rho_bar_xx = 1/3 .+ 1/2 .*(-sol[11, :, :] .+ 1/sqrt(3) .*sol[16, :, :])
# rho_bar_yy = 1/3 .+ 1/2 .*(.- 2/sqrt(3) .*sol[16, :, :])

# Plotting
# p = plot(sol.t, sol[3, 6, :], seriestype=:path,linestyle=:solid, lc=:blue, framestyle = :box, xticks =0:30:150, grid=false, legend = false, dpi = 300)
# plot!(sol.t, sol[8, 6, :], seriestype=:path,linestyle=:solid, lc=:red, framestyle = :box, legend = false)
# ylims!(0, 1)
# xlims!(0, 150)
# xlabel!("\$r \\, \\textrm{(km)}\$")
# ylabel!(L"\rho_{ee}")
# display(p)
# savefig("vacuum+coll.png")
# readline()

# l = @layout [a b; c d; e f]
# lc = [:orange, :green, :lightblue, :darkblue, :pink]
# p = plot(sol.t, rho_ee[1, :], seriestype=:path,linestyle=:solid, lc=:orange, framestyle = :box, layout = l, xticks =0:20:150, grid=false, legend = false, dpi = 300)
# for i in 1:length(E)
#     plot!(p[1], sol.t, rho_ee[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
#     plot!(p[2], sol.t, rho_bar_ee[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
#     plot!(p[3], sol.t, rho_xx[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
#     plot!(p[4], sol.t, rho_bar_xx[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
#     plot!(p[5], sol.t, rho_yy[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
#     plot!(p[6], sol.t, rho_bar_yy[i, :], seriestype=:path,linestyle=:solid, lc=lc[i], framestyle = :box, legend = false)
# end
# ylims!(0, 1)
# xlims!(0, 150)
# xlabel!(p[5], "\$r \\, \\textrm{(km)}\$")
# xlabel!(p[6], "\$r \\, \\textrm{(km)}\$")
# ylabel!(p[1], L"\rho_{ee}")
# ylabel!(p[3], L"\rho_{xx}")
# ylabel!(p[5], L"\rho_{yy}")
# display(p)
# savefig("MSW.png")
# readline()


