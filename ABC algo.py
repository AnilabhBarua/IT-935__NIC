import numpy as np
import matplotlib.pyplot as plt

# --- Objective Function ---
def sphere_function(x):
    """
    The Sphere function to be minimized.
    f(x) = sum(x_i^2)
    Global minimum is 0 at x_i = 0 for all i.
    """
    return np.sum(x**2)

# --- ABC Algorithm Implementation ---

# 1. Initialization Phase
def initialize_population(pop_size, dim, lower_bound, upper_bound):
    """Creates the initial population of food sources (solutions)."""
    population = lower_bound + np.random.rand(pop_size, dim) * (upper_bound - lower_bound)
    return population

def calculate_fitness(cost):
    """Calculates fitness from the cost. Higher fitness is better."""
    # If cost is 0, add a small epsilon to avoid division by zero
    return 1 / (1 + cost) if cost >= 0 else 1 + np.abs(cost)

# 2. Main Bee Phases
def employed_bee_phase(population, lower_bound, upper_bound, limit_trials):
    """
    Employed bees search for new food sources and update if the new one is better.
    """
    new_population = np.copy(population)
    for i in range(len(population)):
        # Choose a random partner k different from i
        k_indices = list(range(len(population)))
        k_indices.pop(i)
        k = np.random.choice(k_indices)

        # Choose a random dimension to change
        j = np.random.randint(population.shape[1])

        # Generate a new candidate solution
        phi = np.random.uniform(-1, 1)
        new_solution = np.copy(population[i])
        new_solution[j] = population[i, j] + phi * (population[i, j] - population[k, j])
        
        # Clip the solution to stay within bounds
        new_solution[j] = np.clip(new_solution[j], lower_bound, upper_bound)
        
        # Greedy selection
        current_cost = sphere_function(population[i])
        new_cost = sphere_function(new_solution)
        
        if new_cost < current_cost:
            new_population[i] = new_solution
            limit_trials[i] = 0  # Reset trial counter if a better solution is found
        else:
            limit_trials[i] += 1
            
    return new_population, limit_trials

def onlooker_bee_phase(population, lower_bound, upper_bound, limit_trials):
    """
    Onlooker bees select food sources based on fitness and search for new ones.
    """
    new_population = np.copy(population)
    costs = np.array([sphere_function(sol) for sol in population])
    fitness_values = np.array([calculate_fitness(c) for c in costs])
    
    # Calculate selection probabilities
    probabilities = fitness_values / np.sum(fitness_values)
    
    # Onlookers choose sources and exploit them
    for _ in range(len(population)): # Number of onlookers equals number of employed bees
        # Roulette wheel selection
        chosen_index = np.random.choice(range(len(population)), p=probabilities)
        
        # Choose a random partner k different from the chosen_index
        k_indices = list(range(len(population)))
        k_indices.pop(chosen_index)
        k = np.random.choice(k_indices)
        
        # Choose a random dimension to change
        j = np.random.randint(population.shape[1])

        # Generate new candidate solution
        phi = np.random.uniform(-1, 1)
        new_solution = np.copy(population[chosen_index])
        new_solution[j] = population[chosen_index, j] + phi * (population[chosen_index, j] - population[k, j])
        
        # Clip the solution to stay within bounds
        new_solution[j] = np.clip(new_solution[j], lower_bound, upper_bound)

        # Greedy selection
        current_cost = sphere_function(population[chosen_index])
        new_cost = sphere_function(new_solution)
        
        if new_cost < current_cost:
            new_population[chosen_index] = new_solution
            limit_trials[chosen_index] = 0
        else:
            limit_trials[chosen_index] += 1
            
    return new_population, limit_trials

def scout_bee_phase(population, limit_trials, limit, lower_bound, upper_bound):
    """
    Scout bees abandon exhausted food sources and search for new ones randomly.
    """
    for i in range(len(population)):
        if limit_trials[i] > limit:
            # Abandon the source and create a new random one
            population[i] = lower_bound + np.random.rand(population.shape[1]) * (upper_bound - lower_bound)
            limit_trials[i] = 0 # Reset the trial counter for the new source
            
    return population, limit_trials

# --- Main ABC Execution ---
if __name__ == "__main__":
    # --- Parameters ---
    SN = 25              # Number of food sources (and employed bees)
    NP = 50              # Total population size (SN Employed + (NP-SN) Onlookers)
    D = 2                # Number of dimensions of the problem
    LOWER_BOUND = -10
    UPPER_BOUND = 10
    LIMIT = 10           # Limit for scout bee phase
    MAX_CYCLES = 100     # Maximum number of iterations

    # --- Initialization ---
    food_sources = initialize_population(SN, D, LOWER_BOUND, UPPER_BOUND)
    trials = np.zeros(SN) # Trial counters for each food source
    
    # Find the initial best solution
    costs = np.array([sphere_function(sol) for sol in food_sources])
    best_solution_index = np.argmin(costs)
    global_best_solution = food_sources[best_solution_index]
    global_best_cost = costs[best_solution_index]
    
    # Store history for plotting
    cost_history = [global_best_cost]

    print(f"Initial Best Cost: {global_best_cost:.6f}")

    # --- Main Loop ---
    for cycle in range(MAX_CYCLES):
        # 1. Employed Bee Phase
        food_sources, trials = employed_bee_phase(food_sources, LOWER_BOUND, UPPER_BOUND, trials)
        
        # 2. Onlooker Bee Phase
        food_sources, trials = onlooker_bee_phase(food_sources, LOWER_BOUND, UPPER_BOUND, trials)
        
        # 3. Scout Bee Phase
        food_sources, trials = scout_bee_phase(food_sources, trials, LIMIT, LOWER_BOUND, UPPER_BOUND)

        # 4. Memorize Best Solution
        current_costs = np.array([sphere_function(sol) for sol in food_sources])
        current_best_index = np.argmin(current_costs)
        if current_costs[current_best_index] < global_best_cost:
            global_best_cost = current_costs[current_best_index]
            global_best_solution = food_sources[current_best_index]
            
        cost_history.append(global_best_cost)
        
        if (cycle + 1) % 10 == 0:
            print(f"Cycle {cycle + 1}: Best Cost = {global_best_cost:.6f}")

    # --- Results ---
    print("\n--- Simulation Finished ---")
    print(f"Best Solution Found: {global_best_solution}")
    print(f"Minimum Cost Achieved: {global_best_cost}")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, marker='o', linestyle='-', color='b')
    plt.title('ABC Algorithm Convergence on Sphere Function')
    plt.xlabel('Cycle')
    plt.ylabel('Best Cost')
    plt.grid(True)
    plt.yscale('log') # Use log scale for better visualization of convergence
    plt.show()