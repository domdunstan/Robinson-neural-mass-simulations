################
## Load required packages
using DifferentialEquations
using DelayDiffEq
using Plots
using StochasticDelayDiffEq


# Define parameters for the Robinson model
# See https://doi.org/10.1093/cercor/bhj072 for detials

# Ordered parameters are: Q_max,theta,sigma,gamma_e,alpha,beta,t0,v_ee,v_ei,v_es,v_se,v_sr,v_sn*phi_n,v_re,v_rs,noise std;
parameters_absence_seizure = [250.0,15.0,6.0,100.0,50.0,200.0,0.08,1.0,-1.8,3.2,4.4,-0.8,2.0,1.6,0.6,0.02];
parameters_tonic_clonic_seizure = [250.0,15.0,6.0,100.0,50.0,240.0,0.08,1.2,-1.8,1.4,1.0,-1.0,1.0,0.2,0.2,0.02];
parameters_resting_example = [250.0,15.0,6.0,100.0,50.0,200.0,0.08,1.0,-1.8,3.2,1,-0.8,2.0,1.6,0.6,0.02];


# Define length of transients
t_trans = 5.0;

# Define time span to run model
tspan = [0.0,10.0];
tspan2 = tspan;
tspan2[2] = tspan2[2]+t_trans


# Define ICs
u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

# Set history
h(p, t) = ones(8)


# Define deterministic part of model
function dde_Robinson(du, u, h, p, t)
            # Unpack parameters
    Q_max, theta, sigma, gamma_e, alpha, beta, t0,
    v_ee, v_ei, v_es, v_se, v_sr, v_sn_phi_n, v_re, v_rs = p
        
    # Sigmoid function
    sigmoid(input) = Q_max / (1 + exp((-pi * (input - theta)) / (sqrt(3) * sigma)))
        
        hist1 = h(p, t - tau)[1]
        hist5 = h(p, t - tau)[5]

        du[1] = u[2]
        du[2] = gamma_e^2 * (sigmoid(u[3]) - u[1]) - 2 * gamma_e * u[2]
        du[3] = u[4]
        du[4] = alpha * beta * (-u[3] + v_ee * u[1] + v_ei * sigmoid(u[3]) + v_es * sigmoid(hist5)) -
                (alpha + beta) * u[4]
        du[5] = u[6]
        du[6] = alpha * beta * (-u[5] + v_se * hist1 + v_sr * sigmoid(u[7]) + v_sn_phi_n) -
                (alpha + beta) * u[6]
        du[7] = u[8]
        du[8] = alpha * beta * (-u[7] + v_re * hist1 + v_rs * sigmoid(u[5])) -
                (alpha + beta) * u[8]
end

# Define stochastic part of model
function dde_Robinson_stoch(du, u, h, p, t)
    du .= 0.0
    noise_strength = p[end]
    alpha = p[5]
    beta = p[6]
    du[6] = alpha * beta * noise_strength
end




################################################################
#### Absence seizure example

# Solve model for qbsence seizure parameter set
tau = parameters_absence_seizure[7]
lags = [tau / 2]

# define problem
prob1 = SDDEProblem(dde_Robinson, dde_Robinson_stoch,u0, h, tspan2, parameters_absence_seizure; constant_lags = lags)

# define solver
alg = SRIW1()

# Solve
sol1 = solve(prob1, alg);

# Take model output and remove transients
phi_e = [-u[1] for u in sol1.u] # Invert time series
time1 = sol1.t
idx1 = argmin(abs.(time1 .- t_trans));
phi_e2 = phi_e[idx1:end];
time2 = time1[idx1:end]; time2 = time2.-5;

# Plot
plot(time2,phi_e2, xlabel = "Time (s)",ylabel = "Solution \u03C6\u2091", legend = false,title="Absence seizure example")


#############################################################
#### Tonic clonic seizure example

# Solve model for qbsence seizure parameter set
tau = parameters_tonic_clonic_seizure[7]
lags = [tau / 2]

# define problem
prob1 = SDDEProblem(dde_Robinson, dde_Robinson_stoch,u0, h, tspan2, parameters_tonic_clonic_seizure; constant_lags = lags)

# define solver
alg = SRIW1()

# Solve
sol1 = solve(prob1, alg);

# Take model output and remove transients
phi_e = [-u[1] for u in sol1.u] # Invert time series
time1 = sol1.t
idx1 = argmin(abs.(time1 .- t_trans));
phi_e2 = phi_e[idx1:end];
time2 = time1[idx1:end]; time2 = time2.-5;

# Plot
plot(time2,phi_e2, xlabel = "Time (s)",ylabel = "Solution \u03C6\u2091", legend = false,title="Tonic clonic seizure example")



#########################################################
#### Resting seizure example

# Solve model for qbsence seizure parameter set
tau = parameters_resting_example[7]
lags = [tau / 2]

# define problem
prob1 = SDDEProblem(dde_Robinson, dde_Robinson_stoch,u0, h, tspan2, parameters_resting_example; constant_lags = lags)

# define solver
alg = SRIW1()

# Solve
sol1 = solve(prob1, alg);

# Take model output and remove transients
phi_e = [-u[1] for u in sol1.u] # Invert time series
time1 = sol1.t
idx1 = argmin(abs.(time1 .- t_trans));
phi_e2 = phi_e[idx1:end];
time2 = time1[idx1:end]; time2 = time2.-5;

# Plot
plot(time2,phi_e2, xlabel = "Time (s)",ylabel = "Solution \u03C6\u2091", legend = false,title="Resting example")

