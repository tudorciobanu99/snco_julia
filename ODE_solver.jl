using SpecialFunctions, FastGaussQuadrature, LinearAlgebra, Integrals, Plots, Sundials, JLD2, Printf

# Physical constants
const delta_m_sq_13 = 2.458e-3 # eV^2
const theta_12 = 0.58 # radians
const theta_13 = 0.15 # radians
const inv_km_to_eV = 0.197e-9 # eV
const J_to_eV = 6.25e18 # eV
const inv_s_to_eV = 6.58e-16 # eV

# Model parameters
const eps = 0.031 # unitless
const E_avg_nu_e = 10e6 # eV
const E_avg_nubar_e = 15e6 # eV
const E_avg_nu_x = 20e6 # eV
const E_avg_nu_y = 20e6 # eV
const r_0 = 10 # km
const eps_nu = 3 # unitless
const Lum = 1.5e44*J_to_eV*inv_s_to_eV # eV^2

# Allocating arrays
B = zeros(8)
L = zeros(8)
f_pos = zeros(50)
f_neg = zeros(50)

# GL integral and transformation for arbitrary boundaries
function transform_gauss_xw(x, w, a, b)
    x1 = a .+ (b-a)/2 * (x .+ 1)
    w1 = (b-a)/2 * w
    return (x=x1, w=w1)
end

function GL_integral(f, a, b, order)
    x, w = gausslegendre(order)
    T = transform_gauss_xw(x, w, a, b)
    x1 = T.x; w1 = T.w;
    I = dot(w1, f)
    return I
end

# Lie algebra structure constants
f_abc = zeros((8, 8, 8))
f_abc[1,2,3] = 2; f_abc[1,3,2] = -2; f_abc[2,1,3] = -2; f_abc[2,3,1] = 2; f_abc[3,1,2] = 2; f_abc[3,2,1] = -2;
f_abc[1,4,7] = 1; f_abc[1,7,4] = -1; f_abc[4,1,7] = -1; f_abc[4,7,1] = 1; f_abc[7,1,4] = 1; f_abc[7,4,1] = -1;
f_abc[1,6,5] = 1; f_abc[1,5,6] = -1; f_abc[6,1,5] = -1; f_abc[6,5,1] = 1; f_abc[5,1,6] = 1; f_abc[5,6,1] = -1;
f_abc[2,4,6] = 1; f_abc[2,6,4] = -1; f_abc[4,2,6] = -1; f_abc[4,6,2] = 1; f_abc[6,2,4] = 1; f_abc[6,4,2] = -1;
f_abc[2,5,7] = 1; f_abc[2,7,5] = -1; f_abc[5,2,7] = -1; f_abc[5,7,2] = 1; f_abc[7,2,5] = 1; f_abc[7,5,2] = -1;
f_abc[3,4,5] = 1; f_abc[3,5,4] = -1; f_abc[4,3,5] = -1; f_abc[4,5,3] = 1; f_abc[5,3,4] = 1; f_abc[5,4,3] = -1;
f_abc[3,7,6] = 1; f_abc[3,6,7] = -1; f_abc[7,3,6] = -1; f_abc[7,6,3] = 1; f_abc[6,3,7] = 1; f_abc[6,7,3] = -1;
f_abc[6,7,8] = sqrt(3); f_abc[6,8,7] = -sqrt(3); f_abc[7,6,8] = -sqrt(3); f_abc[7,8,6] = sqrt(3); f_abc[8,6,7] = sqrt(3); f_abc[8,7,6] = -sqrt(3);
f_abc[4,5,8] = sqrt(3); f_abc[4,8,5] = -sqrt(3); f_abc[5,4,8] = -sqrt(3); f_abc[5,8,4] = sqrt(3); f_abc[8,4,5] = sqrt(3); f_abc[8,5,4] = -sqrt(3);

# Lie algebra cross product
function cross_prod(A, B)
    C = zeros(8)
    for k in 1:8
        for i in 1:8
            for j in 1:8
                C[k] += f_abc[i, j, k]*A[i]*B[j]
            end
        end
    end
    return C
end

# Energy averages
E_avg = Dict([("nu_e", E_avg_nu_e), ("nubar_e", E_avg_nubar_e), ("nu_x", E_avg_nu_x), ("nubar_x", E_avg_nu_x), ("nu_y", E_avg_nu_y), ("nubar_y", E_avg_nu_y)])

# Omega domain
function omega(E)
    omega = abs(delta_m_sq_13)/(2*E)
    return omega
end

# Mass hierarchy dependent contribution
function h()
    h = sign(delta_m_sq_13)
    return h
end

# Scaling factor
function N()
    N = (1 + eps_nu)^(1 + eps_nu)/gamma(1 + eps_nu)
    return N
end

# Spectra normalization factors
function Phi_s(species)
    Phi = Lum/E_avg[species]
    return Phi
end

function Phi()
    Phi = Phi_s("nu_e") + Phi_s("nubar_e") + 2*Phi_s("nu_x") + 2*Phi_s("nu_y")
    return Phi
end

# Spectra
function f_e(E, species)
    f = Lum*N()/(E_avg[species]^2)*(E/E_avg[species])^eps_nu*exp(-E/E_avg[species]*(eps_nu + 1))
    return f
end

# Spectra in omega domain
function f_o(omega, species)
    f = abs(delta_m_sq_13)/(2*Phi()*omega^2)*f_e(abs(abs(delta_m_sq_13)/(2*omega)), species)
    return f
end

# Matter potential
function lamb(r)
    # r in km
    lamb = 1.84e6/(r^(2.4))*inv_km_to_eV
    return lamb
end

# Self-interaction potential
function mu(r)
    # r in km
    mu = 0.45e5*(4*r_0^2)/(3*r^2)*(1 - sqrt(1 - r_0^2/r^2) - r_0^2/(4*r^2))*inv_km_to_eV
    return mu
end

# ODE system matrices
function B_f()
    B = [eps*sin(2*theta_12)*cos(theta_13),
    0,
    sin(theta_13)^2 - eps*(cos(theta_12)^2 - sin(theta_12)^2*cos(theta_13)^2),
    (1 - eps*sin(theta_12)^2)*sin(2*theta_13),
    0,
    -eps*sin(2*theta_12)*sin(theta_13),
    0,
    eps/(2*sqrt(3))*(3*cos(theta_13)^2 - 1 + 3*sin(theta_13)^2*(2*cos(theta_12)^2 - 1)) + 1/sqrt(3)*(1 - 3*cos(theta_13)^2)]
    return B
end

function L_f()
    L = [0, 0, 1, 0, 0, 0, 0, 1/sqrt(3)]
    return L
end

function D_f(omegas, P, Pbar)
    D = zeros(8)
    Pbar_rev = Pbar[:, end:-1:1]
    for i in 1:8
        arg = f_pos.*P[i,:]
        arg_bar = -f_neg.*Pbar_rev[i,:]
        D[i] = GL_integral(arg, omegas[1], omegas[end], 50) + GL_integral(arg_bar, -omegas[end], -omegas[1], 50)
    end
    return D
end

# System initialization
function initialize_system(E_i, E_f, Ebins)
    x, w = gausslegendre(Ebins)
    T = transform_gauss_xw(x, w, omega(E_f), omega(E_i))
    omegas = T.x
    P = zeros(8, Ebins)
    Pbar = zeros(8, Ebins)
    for i in 1:Ebins
        f_t_pos = (f_o.(omegas[i], "nu_e") + f_o.(omegas[i], "nu_x") + f_o.(omegas[i], "nu_y"))
        f_t_neg = (f_o.(-omegas[i], "nubar_e") + f_o.(-omegas[i], "nubar_x") + f_o.(-omegas[i], "nubar_y"))
        P[3,i] = (f_o.(omegas[i], "nu_e") - f_o.(omegas[i], "nu_x"))/f_t_pos
        P[8,i] = (f_o.(omegas[i], "nu_e") + f_o.(omegas[i], "nu_x") - 2*f_o.(omegas[i], "nu_y"))/(sqrt(3)*f_t_pos)
        Pbar[3,i] = (f_o.(-omegas[i], "nubar_e") - f_o.(-omegas[i], "nubar_x"))/(f_t_neg)
        Pbar[8,i] = (f_o.(-omegas[i], "nubar_e") + f_o.(-omegas[i], "nubar_x") - 2*f_o.(-omegas[i], "nubar_y"))/(sqrt(3)*f_t_neg)
    end
    Ptot = vcat(P, Pbar)
    return omegas, P, Pbar, Ptot
end

# ODE system
function dPdr!(du, u, p, t)
    r = t*inv_km_to_eV
    @printf("r = %f\n", r)
    for i in 1:50
        du[1:8,i] = cross_prod(h()*p[i].*B + lamb(r)*L + mu(r)*D_f(p, u[1:8,:], u[9:16,:]), u[1:8,i])
        du[9:16,i] = cross_prod(h()*(-p[i]).*B + lamb(r)*L + mu(r)*D_f(p, u[1:8,:], u[9:16,:]), u[9:16,i])
    end
end

# Setting up the allocated arrays
omegas, P, Pbar, Ptot = initialize_system(1e6, 50e6, 50)
L = L_f()
B = B_f()
f_pos = f_o.(omegas, "nu_e") + f_o.(omegas, "nu_x") + f_o.(omegas, "nu_y")
f_neg = f_o.(-omegas, "nubar_e") + f_o.(-omegas, "nubar_x") + f_o.(-omegas, "nubar_y")
f_neg = f_neg[end:-1:1]

umin = r_0/inv_km_to_eV
umax = 100/inv_km_to_eV
dtmin = 0.001/inv_km_to_eV
saveat = LinRange(umin, umax, 1000)
prob = ODEProblem(dPdr!, Ptot, (umin, umax), omegas)
sol = solve(prob, CVODE_BDF(linear_solver = :GMRES), abstol = 1e-10, saveat = saveat, dtmax = dtmin)
save_object("sol.jld2", sol)

# sol = load_object("sol.jld2")
# p = plot(sol.t*inv_km_to_eV, sol[3,1,:], label = "P1")
# plot!(sol.t*inv_km_to_eV, sol[3,end,:], label = "P2")
# display(p)
# readline()