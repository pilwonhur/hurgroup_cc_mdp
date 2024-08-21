# Policy: stochastic not deterministic
# Transition probability: stochastic not deterministic

include("environment.jl")

# Define the Agent
mutable struct Agent
    env::Environment
    value_table::Dict{Tuple{Int, Int}, Float64}
    policy_table::Dict{Tuple{Int, Int}, Array{Float64, 1}}
    discount_factor::Float64
end

# Initialize a new Agent
function Agent(env::Environment)
    value_table = Dict{Tuple{Int, Int}, Float64}([(x, y) => 0.0 for x in 1:env.params[:nGridθ], y in 1:env.params[:nGridω]])
    policy_table = Dict{Tuple{Int, Int}, Array{Float64, 1}}([(x, y) => ones(Int(env.params[:nActions]))/env.params[:nActions] for x in 1:env.params[:nGridθ], y in 1:env.params[:nGridω]])
    discount_factor = 0.9
    return Agent(env, value_table, policy_table, discount_factor)
end

using StatsBase
function choose_action(policy::Array{Float64, 1})
    return sample(1:length(policy), Weights(policy)) # sample and Weights are in StatsBase package
end

function choose_random_action(actions::Array{Int, 1})
    return sample(actions, Weights(ones(length(actions))/length(actions))) # sample and Weights are in StatsBase package
end

# Define the value iteration algorithm
function value_evaluation!(agent::Agent)
    # Initialize the next value table
    V = Dict{Tuple{Int, Int}, Float64}()
    for y in 1:agent.env.params[:nGridω]
        for x in 1:agent.env.params[:nGridθ]
            V[(x, y)] = 0.0
        end
    end

    value=agent.value_table
    actions=agent.env.actions
    gamma=agent.discount_factor

    for y in 1:agent.env.params[:nGridω]
        for x in 1:agent.env.params[:nGridθ]
            state = (x, y)
            if state in agent.env.terminal_states
                continue
            end

            max_val=Float64[]
            for (i,action) in enumerate(actions)
                temp_val=0.0
                Psp = agent.env.transitions[(x, y, action)]
                for (next_state, prob) in Psp
                    reward = get(agent.env.rewards, next_state, 0.0)
                    temp_val += prob * (reward + gamma * value[next_state])
                end
                push!(max_val, temp_val)
            end
            V[state] = round(maximum(max_val), digits=2)
            max_val=round.(max_val,digits=2)
            max_index=findall(x->x==maximum(max_val),max_val)
            
            result = zeros(Int(agent.env.params[:nActions]))
            prob = 1 / length(max_index)
            for index in max_index
                result[index] = prob
            end
            agent.policy_table[state]=result
        end
    end
    agent.value_table=V
end

function choose_max_action(agent::Agent, state::Tuple{Int, Int})
    actions = agent.env.actions
    value = agent.value_table
    gamma = agent.discount_factor
    max_val = -Inf
    max_action = Int[]
    for (i, action) in enumerate(actions)
        temp_val = 0.0
        Psp = agent.env.transitions[(state[1], state[2], action)]
        for (next_state, prob) in Psp
            reward = get(agent.env.rewards, next_state, 0.0)
            temp_val += prob * (reward + gamma * value[next_state])
        end

        if temp_val==max_val
            push!(max_action, i)
        elseif temp_val>max_val
            max_val=temp_val
            max_action = Int[]
            push!(max_action, i)
        end
    end
    return max_action
end

# Simulate a game using the policy
function simulate_game(agent::Agent,start_state::Array{Float64, 1})
    policy=agent.policy_table
    θ_space = agent.env.θ_space
    ω_space = agent.env.ω_space

    current_state=discretize_state(start_state[1], start_state[2], θ_space, ω_space)
    current_state_real=start_state

    deltat=0.1
    N=150
    t_total=0:0.001:N*deltat
    x_total=zeros(length(t_total),2)
    a_total=zeros(length(t_total))
    u0=[current_state_real[1],current_state_real[2]]
    g = agent.env.params[:g]  # Acceleration due to gravity (m/s^2)
    L = agent.env.params[:l]   # Length of the pendulum (m)
    m = agent.env.params[:m]   # Mass of the pendulum (kg)
    for i in 1:N
        action = choose_action(policy[current_state])
        
        p = (g, L, m, agent.env.actions[action])
        tspan = ((i-1)*deltat, i*deltat)

        sol = simulate_pendulum(u0, p, tspan)
        u0=[sol.u[end][1],sol.u[end][2]]

        for j in 1:100
            x_total[(i-1)*100+j,1]=sol.u[j][1]  # θ
            x_total[(i-1)*100+j,2]=sol.u[j][2]  # ω
        end
        a_total[(i-1)*100+1:i*100] .= agent.env.actions[action]

        # println("Current state: ", current_state, " Action: ", env.actions[action], " New state: ", new_state)
        current_state = discretize_state(sol.u[end][1], sol.u[end][2], θ_space, ω_space)
    end
    return t_total,x_total,a_total
end

function manage_value_iteration!(agent::Agent, n_iter::Int=100)
    for i in 1:n_iter
        value_evaluation!(agent)
    end
end

# Define the environment
params=Dict(
    :g=>9.81,
    :l=>1.0,
    :m=>1.0,
    :nGridθ=>20,
    :nGridω=>21,
    :nActions=>5,
    :ω_min=>-10.0,
    :ω_max=>10.0,
    :θ_min=>-2*pi,
    :θ_max=>0.0
    )
terminal_states = [(1, 11)]
rewards = Dict((1, 11) => 100.0) #,(3, 2) => -1.0,(2, 3) => -1.0
actions = [-2, -1, 0, 1, 2]  # Discretized control inputs (Nm)
# actionSymbols = [:CW2, :CW1, :Zero, :CCW1, :CCW2]

# Create the environment and agent
env = Environment(params, terminal_states, rewards, actions)
# Get the transition function P(s' | s, a)
# include("transition_probability.jl")
define_transitions!(env)

agent=Agent(env)

# Perform value iteration
manage_value_iteration!(agent,100) # 50(rough), 100(good), 200(no big difference)

# Print the results
# println("Optimal Policy:")
# for y in 1:env.params[:nGridω]
#     for x in 1:agent.env.params[:nGridθ] 
#         print(agent.policy_table[(x, y)], " ")
#     end
#     println()
# end

# println("\nOptimal Value Function:")
# for y in 1:env.params[:nGridω]
#     for x in 1:agent.env.params[:nGridθ] 
#         print(agent.value_table[(x, y)], " ")
#     end
#     println()
# end

# Simulate a game using the optimal policy
t_total,x_total,a_total=simulate_game(agent,[-pi,0.0])

using Plots
p1=plot(t_total,x_total[:,1],xlabel="Time (s)",ylabel="Angle (rad)",label="Angle")
plot!(t_total,x_total[:,2],xlabel="Time (s)",ylabel="Angle Velocity (rad/s)",label="Angle Velocity (rad/s)")
plot!(t_total,a_total,xlabel="Time (s)",ylabel="Angle Velocity (rad/s)",label="Action (Nm)")
display(p1)
# phase portrait
p2=plot(x_total[:,1],x_total[:,2],xlabel="Angle (rad)",ylabel="Angle Velocity (rad/s)",label="Phase Portrait")
display(p2)


# Pendulum parameters
L = env.params[:l]  # Length of the pendulum (m)

# Calculate the x and y coordinates of the pendulum bob
x_series = -L * sin.(x_total[:,1])
y_series = L * cos.(x_total[:,1])

steps=20
anim_t=t_total[1:steps:end]
anim_x1=x_total[1:steps:end,1]
anim_x2=x_total[1:steps:end,2]
anim_a=a_total[1:steps:end]
anim_x_series=x_series[1:steps:end]
anim_y_series=y_series[1:steps:end]

layout = @layout [a; b c]
anim = Animation()
for i in 1:length(anim_t)
    fig = plot(layout = layout, legend=false) # grid(2,1)

    plot!(fig[1], # ADD !
    anim_t,[anim_x1,anim_x2,anim_a])
    vline!([anim_t[i]])

    plot!(fig[2],
    [0, anim_x_series[i]],[0, anim_y_series[i]],
    seriestype=:line, lw=2, marker=:o, markersize=8, label="", xlims=(-1, 1), ylims=(-1, 1), aspect_ratio=:equal
    )

    plot!(fig[3], # ADD !
    anim_x1,anim_x2)
    scatter!(fig[3],[anim_x1[i]],[anim_x2[i]])
    
    frame(anim)
end
gif(anim, "inverted_pendulum_val.gif", fps = 50)