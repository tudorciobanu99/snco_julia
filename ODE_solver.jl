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
using DifferentialEquations, SpecialFunctions, FastGaussQuadrature, LinearAlgebra, Integrals, Sundials, JLD2, Printf, LaTeXStrings, BenchmarkTools, Plots, Statistics, Interpolations, CSV, DataFrames
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
                C[k] += 1/2*𝐟[i, j, k]*A[i]*B[j]
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

function xₛ(t)
    xₛ⁰ = -4.6e3 # km
    vₛ = 11.3e3 # km/s
    aₛ = 0.2e3 # km/s²
    xₛ = xₛ⁰ + vₛ*t + 1/2*aₛ*t^2
    return xₛ
end

function f(x, t)
    fln = (0.28 - 0.69*log(xₛ(t)))*(asin(1 - x/xₛ(t)))^(1.1)
    f = exp(fln)
    return f
end

function ρ(x, t)
    ρ₀ = 1e14*(x)^(-2.4) # g/cm³
    ξ = 10
    ρ = 0
    if t <= 1
        ρ = ρ₀
    elseif t > 1
        if x <= xₛ(t)
            ρ = ρ₀*ξ*f(x, t)
        elseif x > xₛ(t)
            ρ = ρ₀
        end
    end
    return ρ
end

function λ(r::Float64, t::Float64)
    km⁻¹_to_eV = 1.97e-10 # eV
    Yₑ = 1/2
    λ = 7.6e-8*Yₑ*ρ(r, t)*1e-6/km⁻¹_to_eV
    #return 1.84e6/(r^(2.4))
    return λ
end

#==========================================================================================
                                SCALING FACTOR
==========================================================================================#

function N(ϵ_ν::Float64)
    return (1 + ϵ_ν)^(1 + ϵ_ν)*1/gamma(1 + ϵ_ν)
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
function f_e(E::Float64, Ē_ν::Float64, ϵ_ν::Float64, L_ν::Float64)
    f = L_ν*N(ϵ_ν)/(Ē_ν^2)*(E/Ē_ν)^ϵ_ν*exp(-E/Ē_ν*(ϵ_ν + 1))
    return f
end


#==========================================================================================
                                ω DISTRIBUTION
==========================================================================================#

function f_o(ω::Float64, Ē_ν::Float64, Φ::Float64, ϵ_ν::Float64, L_ν::Float64, Δm²₁₃::Float64)
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
                            SIMPSON'S RULE FOR INTEGRATION
==========================================================================================#

function simps(y::Array{Float64, 1}, x::Array{Float64, 1})
    n = length(y)-1
    n % 2 == 0 || error("`y` length (number of intervals) must be odd")
    length(x)-1 == n || error("`x` and `y` length must be equal")
    h = (x[end]-x[1])/n
    s = sum(y[1:2:n] + 4*y[2:2:n] + y[3:2:n+1])
    return h/3 * s
end

#==========================================================================================
                            SYSTEM INITIALIZATION
==========================================================================================#

function initialize_system_single(Eᵢ::Float64, Eₖ::Float64, Ebins::Int, Ē::Dict{String, Float64}, ϵ_ν::Array{Float64, 1}, Δm²₁₃::Float64, L_ν::Array{Float64, 1})
    𝐏 = zeros(Float64, 16, Ebins)

    #= Get the Gauss-Legendre quadrature points and weights based on the energy bins
    and transform them to the ω range [ωₖ, ωᵢ] where ωₖ = ω(Eₖ) and ωᵢ = ω(Eᵢ).    =#
    x, w = gausslegendre(Ebins)
    T = transform_gauss_xw(x, w, omega(Eₖ, Δm²₁₃), omega(Eᵢ, Δm²₁₃)) 
    ω = T.x

    #= Calculate the flux averages for the neutrino species and set up the distribution
    functions.                                                                      =#
    L_νₑ, L_ν̄ₑ, L_νₓ = L_ν
    ϵ_νₑ, ϵ_ν̄ₑ, ϵ_νₓ = ϵ_ν
    Φ = Φ_ν("nu_e", Ē, L_νₑ) .+ Φ_ν("nubar_e", Ē, L_ν̄ₑ) .+ 2*Φ_ν("nu_x", Ē, L_νₓ) .+ 2*Φ_ν("nu_y", Ē, L_νₓ)
    f_ω = f_o.(ω, Ē["nu_e"], Φ, ϵ_νₑ, L_νₑ, Δm²₁₃) .+ f_o.(ω, Ē["nu_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) .+ f_o.(ω, Ē["nu_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃)
    f_ω̄ = f_o.(-ω, Ē["nubar_e"], Φ, ϵ_ν̄ₑ, L_ν̄ₑ, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃)

    #= Set up the initial polarization vectors. =#
    for i in 1:Ebins
        𝐏[3,i] = (f_o.(ω[i],  Ē["nu_e"], Φ, ϵ_νₑ, L_νₑ, Δm²₁₃) - f_o.(ω[i], Ē["nu_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃))/f_ω[i]
        𝐏[8,i] = (f_o.(ω[i], Ē["nu_e"], Φ, ϵ_νₑ, L_νₑ, Δm²₁₃) + f_o.(ω[i], Ē["nu_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) - 2*f_o.(ω[i], Ē["nu_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃))/(sqrt(3)*f_ω[i])
        𝐏[11,i] = (f_o.(-ω[i], Ē["nubar_e"], Φ, ϵ_ν̄ₑ, L_ν̄ₑ, Δm²₁₃) - f_o.(-ω[i], Ē["nubar_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃))/f_ω̄[i]
        𝐏[16,i] = (f_o.(-ω[i], Ē["nubar_e"], Φ, ϵ_ν̄ₑ, L_ν̄ₑ, Δm²₁₃) + f_o.(-ω[i], Ē["nubar_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) - 2*f_o.(-ω[i], Ē["nubar_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃))/(sqrt(3)*f_ω̄[i])
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
        D[i] = GL_integral(f_ω.*𝐏[i, :], ωᵢ, ωₖ, order) - GL_integral(f_ω̄.*𝐏[i+8, :], ωᵢ, ωₖ, order)
    end
    return D
end

#==========================================================================================
                                    ODE FUNCTION
==========================================================================================#

function dPdr(du::Array{Float64,2}, u::Array{Float64,2}, p, t::Float64)
    @printf("r = %f\n", t)

    km⁻¹_to_eV = 1.97e-10 # eV

    ω = p[1]
    f_ω = p[2]
    f_ω̄ = p[3]
    𝐁 = p[4]
    𝐋 = p[5]
    r₀ = p[6]
    𝐟 = p[7]
    
    Ebins = length(ω)

    𝐃 = D(u, ω, f_ω, f_ω̄, Ebins)
    λ_t = λ(t, 4.0)
    μ_t = μ(t, r₀)
    @inbounds for i in 1:Ebins
        ω_i = ω[i]/km⁻¹_to_eV
        du[1:8, i] = cross_prod((ω_i.*𝐁 .+ λ_t.*𝐋 .+ μ_t.*𝐃), u[1:8, i], 𝐟)
        du[9:16, i] = cross_prod((-ω_i.*𝐁 .+ λ_t*𝐋 .+ μ_t.*𝐃), u[9:16, i], 𝐟)
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
    Δm²₁₃ = -2.458e-3 # eV², >0 for normal hierarch, <0 for inverted hierarchy
    ϵ = -0.031 # unitless, >0 for normal hierarchy, <0 for inverted hierarchy
    θ₁₂ = 0.58 # radians
    θ₁₃ = 0.15 # radians
    h = sign(Δm²₁₃)
    δ = 0
    Ē_νₑ = 10e6 # eV
    Ē_ν̄ₑ = 15e6 # eV
    Ē_νₓ = 20e6 # eV
    Ē_νᵧ= 20e6 # eV
    ϵ_ν = [3.0, 3.0, 3.0] # unitless
    L_ν = [1.5e44, 1.5e44, 1.5e44]*J_to_eV*s⁻¹_to_eV # eV^2
    r₀ = 10 # km

    𝐁 = [ϵ*sin(2*θ₁₂)*cos(θ₁₃),
      0,
      sin(θ₁₃)^2 - ϵ*(cos(θ₁₂)^2 - sin(θ₁₂)^2*cos(θ₁₃)^2),
      (1 - ϵ*sin(θ₁₂)^2)*sin(2*θ₁₃)*cos(δ),
      (1 - ϵ*sin(θ₁₂)^2)*sin(2*θ₁₃)*sin(δ),
      -ϵ*sin(2*θ₁₂)*sin(θ₁₃)*cos(δ),
      -ϵ*sin(2*θ₁₂)*sin(θ₁₃)*sin(δ),
      ϵ/(2*sqrt(3))*(3*cos(θ₁₃)^2 - 1 + 3*sin(θ₁₃)^2*(2*cos(θ₁₂)^2 - 1)) + 1/sqrt(3)*(1 - 3*cos(θ₁₃)^2)]
    𝐁 = 𝐁./h

    𝐋 = [0, 0, 1, 0, 0, 0, 0, 1/sqrt(3)]

    # Energy averages
    Ē = Dict([("nu_e", Ē_νₑ), ("nubar_e", Ē_ν̄ₑ), ("nu_x", Ē_νₓ), ("nubar_x", Ē_νₓ), ("nu_y", Ē_νᵧ), ("nubar_y", Ē_νᵧ)])
    ω, 𝐏₀, f_ω, f_ω̄ = initialize_system_single(Eᵢ, Eₖ, Ebins, Ē, ϵ_ν, Δm²₁₃, L_ν)
    @save "omega.jld2" ω
    #indices = [1, 6, 11, 16, 20, 25, 28, 32, 37, 43, 47]
    #prob = ODEProblem(dPdr, 𝐏₀, (umin, umax), [ω, f_ω, f_ω̄, 𝐁, 𝐋, r₀, 𝐟])
    println("Solving the ODE...")
    # @time begin
    #     sol = solve(prob, CVODE_BDF(linear_solver =:LapackDense), maxiters = Int(1.0e18), abstol = 1e-10, save_everystep = false, saveat = 0.1)
    # end
    @load "sol_4.0_ih.jld2" sol
    ts = sol.t
    ρ_init = 1/3 .+ 1/2 .*(sol[3, 6, :] .+ 1/sqrt(3) .*sol[8, 6, :])
    ρ_final = 1/3 .+ 1/2 .*(sol[3, 99, :] .+ 1/sqrt(3) .*sol[8, 99, :])
    𝐏₀ = sol[:,:,end]
    prob = ODEProblem(dPdr, 𝐏₀, (umin, umax), [ω, f_ω, f_ω̄, 𝐁, 𝐋, r₀, 𝐟])
    @time begin
        soln =  solve(prob, maxiters = Int(1.0e18), abstol = 1e-10, save_everystep = false, saveat = 1)
    end
    ρₑₑ = 1/3 .+ 1/2 .*(soln[3, :, :] .+ 1/sqrt(3) .*soln[8, :, :])
    p = plot(soln.t, ρₑₑ[6, :], xaxis=:log, label = "1")
    plot!(soln.t, ρₑₑ[99, :], label = "2")
    plot!(ts, ρ_init, label = "3")
    plot!(ts, ρ_final, label = "4")
    ylims!(0, 1)
    xlims!(9000, 50000)
    display(p)
    readline()
    @save "sol_4.0_ih_add.jld2" soln
end

#==========================================================================================
                                    PLOTTING
==========================================================================================#

function plot_flavor_evolution(sol, indices, name, xmax)
    rho_ee = zeros(Float64, length(sol.t), length(indices))
    rho_bar_ee = zeros(Float64, length(sol.t), length(indices))
    rho_xx = zeros(Float64, length(sol.t), length(indices))
    rho_bar_xx = zeros(Float64, length(sol.t), length(indices))
    rho_yy = zeros(Float64, length(sol.t), length(indices))
    rho_bar_yy = zeros(Float64, length(sol.t), length(indices))

    for i in 1:length(indices)
        rho_ee[:, i] = 1/3 .+ 1/2 .*(sol[3, indices[i], :] .+ 1/sqrt(3) .*sol[8, indices[i], :])
        rho_xx[:, i] = 1/3 .+ 1/2 .*(-sol[3, indices[i], :] .+ 1/sqrt(3) .*sol[8, indices[i], :])
        rho_bar_xx[:, i] = 1/3 .+ 1/2 .*(-sol[11, indices[i], :] .+ 1/sqrt(3) .*sol[16, indices[i], :])
        rho_yy[:, i] = 1/3 .+ 1/2 .*(-2/sqrt(3)*sol[8, indices[i], :])
        rho_bar_yy[:, i] = 1/3 .+ 1/2 .*(-2/sqrt(3)*sol[16, indices[i], :])
        rho_bar_ee[:, i] = 1/3 .+ 1/2 .*(sol[11, indices[i], :] .+ 1/sqrt(3) .*sol[16, indices[i], :])
    end

    println("Plotting...")
    l = @layout [a b; c d; e f]
    lc = reverse([:pink, :darkblue, :lightblue, :green, :orange, :red, :lightgreen, :darkgreen, :blue, :magenta, :red])
    p = plot(sol.t, rho_ee[:, 1], seriestype=:path, linestyle=:solid, lc=lc[1], framestyle = :box, xaxis=:log, xticks = [10, 100, 1000], layout = l, grid=false, legend = false, dpi = 300)
    for i in 1:length(indices)
        plot!(p[1], sol.t, rho_ee[:, i], seriestype=:path,linestyle=:solid, xaxis=:log, lc=lc[i], xticks = [10, 100, 1000, 10000], framestyle = :box, legend = false)
        plot!(p[2], sol.t, rho_bar_ee[:, i], seriestype=:path,linestyle=:solid, lc=lc[i], xticks = [10, 100, 1000, 10000], xaxis=:log, framestyle = :box, legend = false)
        plot!(p[3], sol.t, rho_xx[:, i], seriestype=:path,linestyle=:solid, lc=lc[i], xticks = [10, 100, 1000, 10000], xaxis=:log, framestyle = :box, legend = false)
        plot!(p[4], sol.t, rho_bar_xx[:, i], seriestype=:path,linestyle=:solid, lc=lc[i], xticks = [10, 100, 1000, 10000], xaxis=:log, framestyle = :box, legend = false)
        plot!(p[5], sol.t, rho_yy[:, i], seriestype=:path,linestyle=:solid, lc=lc[i], xticks = [10, 100, 1000, 10000], xaxis=:log, framestyle = :box, legend = false)
        plot!(p[6], sol.t, rho_bar_yy[:, i], seriestype=:path,linestyle=:solid, lc=lc[i], xticks = [10, 100, 1000, 10000], xaxis=:log, framestyle = :box, legend = false)
    end
    ylims!(0, 1)
    xlims!(10, xmax)
    xlabel!(p[5], "\$r \\, \\textrm{(km)}\$")
    xlabel!(p[6], "\$r \\, \\textrm{(km)}\$")
    ylabel!(p[1], L"\rho_{ee}")
    ylabel!(p[3], L"\rho_{xx}")
    ylabel!(p[5], L"\rho_{yy}")
    display(p)
    savefig(name*".png")
    readline()
end

function plot_final_spectra(sol, Eᵢ, Eₖ, Ebins, name)
    rho_ee = mean(1/3 .+ 1/2 .*(sol[3, :, end-100:end] .+ 1/sqrt(3) .*sol[8, :, end-100:end]), dims = 2)
    rho_xx = mean(1/3 .+ 1/2 .*(-sol[3, :, end-100:end] .+ 1/sqrt(3) .*sol[8, :, end-100:end]), dims = 2)
    rho_yy = mean(1/3 .+ 1/2 .*(.- 2/sqrt(3) .*sol[8, :, end-100:end]), dims = 2)
    rho_bar_ee = mean(1/3 .+ 1/2 .*(sol[11, :, end-100:end] .+ 1/sqrt(3) .*sol[16, :, end-100:end]), dims = 2)
    rho_bar_xx = mean(1/3 .+ 1/2 .*(-sol[11, :, end-100:end] .+ 1/sqrt(3) .*sol[16, :, end-100:end]), dims = 2)
    rho_bar_yy = mean(1/3 .+ 1/2 .*(.- 2/sqrt(3) .*sol[16, :, end-100:end]), dims = 2)

    Δm²₁₃ = 2.458e-3 # eV²
    x, w = gausslegendre(Ebins)
    T = transform_gauss_xw(x, w, omega(Eₖ, Δm²₁₃), omega(Eᵢ, Δm²₁₃)) 
    ω = T.x
    E = abs(Δm²₁₃)./(2*ω)
    Ē_νₑ = 10e6 # eV
    Ē_ν̄ₑ = 15e6 # eV
    Ē_νₓ = 20e6 # eV
    Ē_νᵧ= 20e6 # eV
    J_to_eV = 6.25e18 # eV
    s⁻¹_to_eV = 6.58e-16 # eV
    
    ϵ_νₑ, ϵ_ν̄ₑ, ϵ_νₓ = [3.0, 3.0, 3.0]
    L_νₑ, L_ν̄ₑ, L_νₓ = [1.5e44, 1.5e44, 1.5e44]*J_to_eV*s⁻¹_to_eV

    Ē = Dict([("nu_e", Ē_νₑ), ("nubar_e", Ē_ν̄ₑ), ("nu_x", Ē_νₓ), ("nubar_x", Ē_νₓ), ("nu_y", Ē_νᵧ), ("nubar_y", Ē_νᵧ)])
    Φ = Φ_ν("nu_e", Ē, L_νₑ) .+ Φ_ν("nubar_e", Ē, L_ν̄ₑ) .+ 2*Φ_ν("nu_x", Ē, L_νₓ) .+ 2*Φ_ν("nu_y", Ē, L_νₓ)
    f_ω = f_o.(ω, Ē["nu_e"], Φ, ϵ_νₑ, L_νₑ, Δm²₁₃) .+ f_o.(ω, Ē["nu_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) .+ f_o.(ω, Ē["nu_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃)
    f_ω̄ = f_o.(-ω, Ē["nubar_e"], Φ, ϵ_ν̄ₑ, L_ν̄ₑ, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_x"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃) .+ f_o.(-ω, Ē["nubar_y"], Φ, ϵ_νₓ, L_νₓ, Δm²₁₃)

    f_obs_e = rho_ee.*f_ω.*ω.^2
    f_obs_x = rho_xx.*f_ω.*ω.^2
    f_obs_y = rho_yy.*f_ω.*ω.^2
    f_obs_bar_e = rho_bar_ee.*f_ω̄.*ω.^2
    f_obs_bar_x = rho_bar_xx.*f_ω̄.*ω.^2
    f_obs_bar_y = rho_bar_yy.*f_ω̄.*ω.^2

    #@save "final_spectra.jld2" f_obs_e f_obs_x f_obs_y f_obs_bar_e f_obs_bar_x f_obs_bar_y
    #@save "final_spectra_E.jld2" E

    l = @layout [a b]
    p = plot(E*1e-6, 0.0506666667*f_e.(E, Ē_νₑ, ϵ_νₑ, L_νₑ)*abs(Δm²₁₃)/(2*Φ)*1e11, layout = l, seriestype=:path, linestyle=:solid, lc=:red, label = "\$ \\nu_{e, i} \$", framestyle = :box, legend = true)
    plot!(p[1], E*1e-6, 0.0506666667*f_e.(E, Ē_νₓ, ϵ_νₓ, L_νₓ)*abs(Δm²₁₃)/(2*Φ)*1e11, seriestype=:path, linestyle=:solid, lc=:blue, label = "\$ \\nu_{(x,y), i} \$", framestyle = :box, grid=false, dpi = 300)
    plot!(p[1], E*1e-6, 0.0506666667*f_obs_e*1e11, seriestype=:path, linestyle=:dash, lc=:red, label = "\$ \\nu_{e} \$", framestyle = :box)
    plot!(p[1], E*1e-6, 0.0506666667*f_obs_x*1e11, seriestype=:path, linestyle=:dash, lc=:green, label = "\$ \\nu_{x} \$", framestyle = :box, grid=false, dpi = 300)
    plot!(p[1], E*1e-6, 0.0506666667*f_obs_y*1e11, seriestype=:path, linestyle=:dash, lc=:blue, label = "\$ \\nu_{y} \$", framestyle = :box)
    plot!(p[2], E*1e-6, 0.0506666667*f_e.(E, Ē_ν̄ₑ, ϵ_ν̄ₑ, L_ν̄ₑ)*abs(Δm²₁₃)/(2*Φ)*1e11, seriestype=:path, linestyle=:solid, lc=:red, label = "\$ \\bar{\\nu}_{e, i} \$", framestyle = :box)
    plot!(p[2], E*1e-6, 0.0506666667*f_e.(E, Ē_νₓ, ϵ_νₓ, L_νₓ)*abs(Δm²₁₃)/(2*Φ)*1e11, seriestype=:path, linestyle=:solid, lc=:blue, label = "\$ \\bar{\\nu}_{(x,y), i} \$", framestyle = :box, grid=false, dpi = 300)
    plot!(p[2], E*1e-6, 0.0506666667*f_obs_bar_e*1e11, seriestype=:path, linestyle=:dash, lc=:red, label = "\$ \\bar{\\nu}_{e} \$", framestyle = :box)
    plot!(p[2], E*1e-6, 0.0506666667*f_obs_bar_x*1e11, seriestype=:path, linestyle=:dash, lc=:green, label = "\$ \\bar{\\nu}_{x} \$", framestyle = :box, grid=false, dpi = 300)
    plot!(p[2], E*1e-6, 0.0506666667*f_obs_bar_y*1e11, seriestype=:path, linestyle=:dash, lc=:blue, label = "\$ \\bar{\\nu}_{y} \$", framestyle = :box)
    xlabel!(p[1], "\$E \\, \\textrm{(MeV)}\$")
    xlabel!(p[2], "\$E \\, \\textrm{(MeV)}\$")
    ylabel!(p[1], "Flux (a.u.)")
    ylabel!(p[2], "Flux (a.u.)")
    xlims!(0, 51)
    ylims!(p[1], 0, 0.16)
    ylims!(p[2], 0, 0.08)
    display(p)
    savefig(name*".png")
    readline()
end


function plot_density_profile(name, xmax, t, Eᵢ, Eₖ)
    max = 101
    r = 10 .^ range(1, stop=6, length=max)
    km⁻¹_to_eV = 1.97e-10 #eV
    r₀ = 10
    μ_r = μ.(r, r₀)
    λ_r = λ.(r, t)
    Δm²₁₃ = 2.5e-3
    ϵ = 1/30
    Δm²₁₂ = ϵ*Δm²₁₃
    ωₖ_13 = abs(Δm²₁₃)/(2*Eᵢ)/km⁻¹_to_eV
    ωᵢ_13 = abs(Δm²₁₃)/(2*Eₖ)/km⁻¹_to_eV
    ωₖ_12 = abs(Δm²₁₂)/(2*Eᵢ)/km⁻¹_to_eV
    ωᵢ_12 = abs(Δm²₁₂)/(2*Eₖ)/km⁻¹_to_eV

    p = plot(r, μ_r, xaxis=:log, yaxis=:log, label = L"\mu(r)", framestyle = :box, xticks = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100, 1000, 10000, 100000, 1000000], grid = false,  dpi = 300)
    plot!(p, r, λ_r, xaxis=:log, yaxis=:log, label = L"\lambda(r)", framestyle = :box, grid = false)
    plot!(p, r, repeat([ωᵢ_13], max), label = "H", fillrange = repeat([ωₖ_13], max), fillalpha = 0.35, c = 1, legend = :topright, xaxis=:log, yaxis=:log, framestyle = :box, grid = false)
    plot!(p, r, repeat([ωᵢ_12], max), label = "L", fillrange = repeat([ωₖ_12], max), fillalpha = 0.35, c = 4, legend = :topright, xaxis=:log, yaxis=:log, framestyle = :box, grid = false)
    xlabel!(p, "\$r \\, \\textrm{(km)}\$")
    ylabel!(p, "\$\\omega, \\mu, \\lambda \\, \\textrm{(km}^{-1}\\textrm{)}\$")
    xlims!(10, xmax)
    ylims!(0.00001, 1e5)
    display(p)
    savefig(name*".png")
    readline()
end


