# Policy: stochastic not deterministic
# Transition probability: stochastic not deterministic

# Define the GridWorld environment in Julia
struct Environment
    grid::Array{Int, 2} # 2D grid representing the environment (y: rows, x: columns)
    start_state::Tuple{Int, Int}
    terminal_states::Array{Tuple{Int, Int}, 1}
    rewards::Dict{Tuple{Int, Int}, Float64}
    transitions::Dict{Tuple{Int, Int, Int}, Tuple{Tuple{Int, Int}, Float64}} # (current_state, action) -> (new_state, prob for actions)
    actions::NTuple{4, Int}
end

# Initialize a new GridWorld environment
function Environment(grid_size::Tuple{Int, Int}, start_state::Tuple{Int, Int}, terminal_states::Array{Tuple{Int, Int}, 1}, rewards::Dict{Tuple{Int, Int}, Float64}, actions::NTuple{4, Int})
    grid = zeros(Int, grid_size[1], grid_size[2])
    transitions = Dict{Tuple{Int, Int, Int}, Tuple{Tuple{Int, Int}, Float64}}()
    return Environment(grid, start_state, terminal_states, rewards, transitions, actions)
end

# Define the transition function P(s' | s, a)
function define_transitions!(env::Environment, actions::NTuple{4, Int}) #, probs::Array{Float64, 1})
    for y in 1:size(env.grid, 1)
        for x in 1:size(env.grid, 2)
            for (i, action) in enumerate(actions)
                probs = 1.0
                new_state = (x, y)
                if action == 1 # :up
                    new_state = (x, min(y + 1, size(env.grid, 1)))
                elseif action == 2 # :down
                    new_state = (x, max(y - 1, 1))
                elseif action == 3 #:left
                    new_state = (max(x - 1, 1), y)
                elseif action == 4 #:right
                    new_state = (min(x + 1, size(env.grid, 2)), y)
                end
                env.transitions[(x, y, action)]=(new_state, probs) # (new_state, probs[i])
            end
        end
    end
end

# Define the Agent
mutable struct Agent
    env::Environment
    value_table::Dict{Tuple{Int, Int}, Float64}
    policy_table::Dict{Tuple{Int, Int}, Array{Float64, 1}}
    discount_factor::Float64
end

# Initialize a new Agent
function Agent(env::Environment)
    value_table = Dict{Tuple{Int, Int}, Float64}([(x, y) => 0.0 for y in 1:size(env.grid, 1), x in 1:size(env.grid, 2)])
    policy_table = Dict{Tuple{Int, Int}, Array{Float64, 1}}([(x, y) => [0.25,0.25,0.25,0.25] for y in 1:size(env.grid, 1), x in 1:size(env.grid, 2)])
    discount_factor = 0.9
    return Agent(env, value_table, policy_table, discount_factor)
end

using StatsBase
function choose_action(policy::Array{Float64, 1})
    return sample(1:4, Weights(policy)) # sample and Weights are in StatsBase package
end

# Simulate a game using the policy
function simulate_game(policy::Dict{Tuple{Int, Int}, Array{Float64, 1}},start_state::Tuple{Int, Int})
    current_state=start_state
    while true
        action = choose_action(policy[current_state])
        (new_state, prob) = env.transitions[(current_state[1], current_state[2], action)]
        println("Current state: ", current_state, " Action: ", actionSymbols[action], " New state: ", new_state)
        current_state = new_state
        if current_state in env.terminal_states
            break
        end
    end
end

# Define the policy iteration algorithm
function policy_evaluation!(env::Environment, agent::Agent)
    # Initialize the next value table
    V = Dict{Tuple{Int, Int}, Float64}()
    for y in 1:size(env.grid, 1)
        for x in 1:size(env.grid, 2)
            V[(x, y)] = 0.0
        end
    end
    policy=agent.policy_table
    value=agent.value_table
    actions=env.actions
    gamma=agent.discount_factor

    for y in 1:size(env.grid, 1)
        for x in 1:size(env.grid, 2)
            state = (x, y)
            if state in env.terminal_states
                continue
            end

            temp_val=0.0
            for (i,action) in enumerate(actions)
                (next_state, prob) = env.transitions[(x, y, action)]
                reward = get(env.rewards, next_state, 0.0)
                temp_val += policy[state][i] * prob * (reward + gamma * value[next_state])
            end
            V[state] = round(temp_val, digits=2)
        end
    end
    agent.value_table=V
end

function policy_improvement!(env::Environment, agent::Agent)
    policy=agent.policy_table
    value=agent.value_table
    actions=env.actions
    gamma=agent.discount_factor

    for y in 1:size(env.grid, 1)
        for x in 1:size(env.grid, 2)
            state = (x, y)
            if state in env.terminal_states
                continue
            end
            max_val=-Inf
            max_index=Int[]

            for (i,action) in enumerate(actions)
                (next_state, prob) = env.transitions[(x, y, action)]
                reward = get(env.rewards, next_state, 0.0)
                temp_val=prob * (reward + gamma * value[next_state])

                if temp_val==max_val
                    push!(max_index, i)
                elseif temp_val>max_val
                    max_val=temp_val
                    max_index = Int[]
                    push!(max_index, i)
                end
            end

            result = [0.0, 0.0, 0.0, 0.0]
            prob = 1 / length(max_index)
            for index in max_index
                result[index] = prob
            end
            agent.policy_table[state]=result
        end
    end
end

function policy_iteration!(env::Environment, agent::Agent)
    for j in 1:5
        for i in 1:5
            policy_evaluation!(env, agent)
        end
        policy_improvement!(env, agent)
    end
end

# Define the environment
grid_size = (5, 5)
start_state = (1, 1)
terminal_states = [(3, 3)]
rewards = Dict((3, 3) => 1.0,(3, 2) => -1.0,(2, 3) => -1.0)
actions = (1, 2, 3, 4) # :up, :down, :left, :right
actionSymbols = [:up, :down, :left, :right]

# Create the environment and agent
env = Environment(grid_size, start_state, terminal_states, rewards, actions)
define_transitions!(env, actions)
agent=Agent(env)

# Perform policy iteration
policy_iteration!(env, agent)

# Print the results
println("Optimal Policy:")
for y in 1:size(env.grid, 1)
    for x in 1:size(env.grid, 2)
        print(agent.policy_table[(x, y)], " ")
    end
    println()
end

println("\nOptimal Value Function:")
for y in 1:size(env.grid, 1)
    for x in 1:size(env.grid, 2)
        print(agent.value_table[(x, y)], " ")
    end
    println()
end

# Simulate a game using the optimal policy
simulate_game(agent.policy_table,env.start_state)

# (1,5), (2,5), (3,5), (4,5), (5,5)
# (1,4), (2,4), (3,4), (4,4), (5,4)
# (1,3), (2,3), (3,3), (4,3), (5,3)
# (1,2), (2,2), (3,2), (4,2), (5,2)
# (1,1), (2,1), (3,1), (4,1), (5,1)