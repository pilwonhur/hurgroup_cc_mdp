using DifferentialEquations, Random, Statistics, Distributions

# Define the pendulum dynamics
function pendulum_dynamics!(du, u, p, t)
    θ, ω = u
    g, L, m, control = p

    # Equations of motion
    du[1] = ω
    du[2] = (g/L) * sin(θ) + control/(m * L^2)
end

function simulate_pendulum(u0, p, tspan)
    prob = ODEProblem(pendulum_dynamics!, u0, tspan, p)
    return solve(prob, Tsit5(), dt=0.001, saveat=0.001) # RK4()
end

# Function to discretize a continuous state
function discretize_state(θ, ω, θ_space, ω_space)
    θ=mod(θ+2π,2π)-2π
    ω=clamp(ω,ω_space[1],ω_space[end])

    dθ=(θ_space[end]-θ_space[1])/20
    θ_space_new=vcat(θ_space[1],θ_space[1:end-1] .+ dθ/2,θ_space[end])

    dω=(ω_space[end]-ω_space[1])/21
    # ω_space_new=vcat(ω_space[1],ω_space[1:end-1] .+ dω/2,ω_space[end])
    ω_space_new=vcat(ω_space[1:end])

    θ_idx = findall(x -> x >= θ, θ_space_new)[1]-1
    θ_idx=mod(θ_idx-1,20)+1
    ω_idx = clamp(findall(x -> x >= ω, ω_space_new)[1]-1,1,21)
    return (θ_idx, ω_idx)
end

function define_transitions!(env::Environment)
    actions=env.actions
    # Discretize the state space
    θ_min, θ_max = env.params[:θ_min], env.params[:θ_max]
    ω_min, ω_max = env.params[:ω_min], env.params[:ω_max]
    θ_space = env.θ_space
    ω_space = env.ω_space

    # Monte Carlo Simulation Parameters
    n_samples = 10000  # Number of random samples
    Δt = 0.1         # Small time step for simulation

    # Parameters
    g = env.params[:g]  # Acceleration due to gravity (m/s^2)
    L = env.params[:l]   # Length of the pendulum (m)
    m = env.params[:m]   # Mass of the pendulum (kg)

    # Initialize the transition probability matrix
    P = Dict{Tuple{Int, Int, Int}, Dict{Tuple{Int, Int}, Float64}}()
        
    # Populate the state space with random points and simulate dynamics
    for action in actions
        Random.seed!(1234)  # Set seed for reproducibility
        for _ in 1:n_samples
            # Randomly sample an initial state (θ, ω)
            θ0 = rand(Uniform(θ_min, θ_max))
            ω0 = rand(Uniform(ω_min, ω_max))

            # Discretize the initial state
            s = discretize_state(θ0, ω0, θ_space, ω_space)

            # Set up the problem
            u0 = [θ0, ω0]
            p = (g, L, m, action)
            tspan = (0.0, Δt)

            sol = simulate_pendulum(u0, p, tspan)

            # Get the resulting state after Δt
            θ_next, ω_next = sol.u[end]

            # Discretize the next state
            s_prime = discretize_state(θ_next, ω_next, θ_space, ω_space)

            # Update the transition probability
            if haskey(P, (s..., action))
                P[(s..., action)][s_prime] = get(P[(s..., action)], s_prime, 0.0) + 1.0
            else
                P[(s..., action)] = Dict(s_prime => 1.0)
            end
        end
    end

    # Normalize the probabilities
    for (key, value) in P
        total = sum(values(value))
        for k in keys(value)
            P[key][k] /= total
        end
    end
    env.transitions = P
end

# plot(θ_space,map(x->discretize_state(x, 0, θ_space, ω_space),θ_space))

# θ0 = rand(Uniform(θ_min, θ_max),1000)
# scatter(θ0,[discretize_state(θ0[i], 0, θ_space, ω_space)[1] for i in 1:1000])

# using Plots
# ω0 = rand(Uniform(-10.0, 10.0),1000)
# scatter(ω0,[discretize_state(-pi, ω0[i], env.θ_space, env.ω_space)[2] for i in 1:1000])