using DifferentialEquations

# Define the pendulum dynamics
function pendulum_dynamics!(du, u, p, t)
    θ, ω = u
    g, L, m, control = p

    # Equations of motion
    du[1] = ω
    du[2] = (g/L) * sin(θ) + control/(m * L^2)
end

# Parameters
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)
m = 1.0   # Mass of the pendulum (kg)
control = 0.0  # Control input torque (Nm)

# Initial conditions: θ = 0.1 rad, ω = 0 rad/s
u0 = [0.1, 0.0]
p = (g, L, m, control)

# Time span for the simulation
tspan = (0.0, 10.0)

# Problem definition
prob = ODEProblem(pendulum_dynamics!, u0, tspan, p)

# Define the fixed time step for RK4
dt = 0.01

# Solve the ODE using the RK4 method
sol = solve(prob, RK4(), dt=dt)

using Plots

# Plot the solution
plot(sol, idxs=(1, 2), xlabel="Time (s)", ylabel="Angle (rad) and Angular Velocity (rad/s)")