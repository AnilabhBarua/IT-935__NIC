import numpy as np
import sys

def sphere_function(x):
    """
    The sphere function.
    f(x) = sum(x_i^2)
    The global minimum is 0 at x_i = 0 for all i.
    """
    return np.sum(x**2)

def cat_swarm_optimization(objective_func, dim, num_cats, max_iter, search_bounds, verbose=True):
    """
    Implements the Cat Swarm Optimization (CSO) algorithm.

    Parameters:
    - objective_func: The function to be minimized (e.g., sphere_function).
    - dim: Number of dimensions (n in the formula).
    - - num_cats: Total population of cats.
    - - max_iter: Maximum number of iterations.
    - - search_bounds: A tuple (min_val, max_val) for the search space.
    - - verbose: If True, prints progress.
    
    CSO Specific Parameters:
    - MR (Mixture Ratio): Percentage of cats in tracing mode.
    - SMP (Seeking Memory Pool): Number of copies for a seeking cat.
    - SRD (Seeking Range of the selected Dimension): Mutation range.
    - CDC (Counts of Dimensions to Change): Percentage of dimensions to mutate.
    - c1: Constant for tracing mode velocity update.
    - w_min/w_max: Inertia weight for tracing mode.
    """

    # --- CSO Parameters ---
    MR = 0.3  # 30% of cats will be in Tracing Mode
    SMP = 5   # Seeking Memory Pool (make 5 copies)
    SRD = 0.1 # Seeking Range (10% of total range)
    CDC = 0.8 # Counts of Dimensions to Change (80% of dimensions)
    
    # Tracing Mode Parameters (similar to PSO)
    c1 = 2.0
    w_max = 0.9
    w_min = 0.4
    
    # --- FIX 1: Removed the stray 'x' variable. ---
    min_val, max_val = search_bounds
    search_range = max_val - min_val
    max_velocity = search_range * 0.2 # Velocity clamp
    min_velocity = -max_velocity

    # --- 1. Initialization ---
    
    # Initialize positions for all cats
    positions = np.random.uniform(min_val, max_val, (num_cats, dim))
    
    # Initialize velocities (for tracing mode)
    velocities = np.random.uniform(min_velocity, max_velocity, (num_cats, dim))
    
    # Evaluate fitness for each cat
    fitness = np.array([objective_func(pos) for pos in positions])
    
    # Find the global best
    global_best_fitness = np.min(fitness)
    global_best_pos = positions[np.argmin(fitness)].copy()
    
    # --- 2. Main Iteration Loop ---
    for t in range(max_iter):
        
        # Linearly decrease inertia weight (for tracing mode)
        w = w_max - (w_max - w_min) * (t / max_iter)
        
        # Randomly assign cats to Seeking or Tracing mode based on MR
        num_tracing = int(num_cats * MR)
        
        # --- FIX 2: Correctly determine tracing indices and use a boolean array for seeking. ---
        # The number of tracing cats is MR * num_cats. The rest are seeking.
        is_tracing = np.zeros(num_cats, dtype=bool)
        tracing_indices = np.random.choice(num_cats, size=num_tracing, replace=False)
        is_tracing[tracing_indices] = True
        
        # --- 3. Process each cat ---
        for i in range(num_cats):
            
            if not is_tracing[i]:
                # --- 3a. Seeking Mode (Local Search) ---
                
                # Create SMP copies of the cat's position
                copies = np.tile(positions[i], (SMP, 1))
                
                # Mutate CDC dimensions for (SMP-1) copies
                for j in range(SMP): # Iterate over all SMP copies (including the original, if needed, but usually (SMP-1) are mutated)
                    if j == SMP - 1: # The last one is usually the original cat's position (no mutation)
                        # An alternative approach: keep the original, mutate (SMP-1) copies
                        # This implementation mutates all but the final copy, keeping the first copy unmutated for comparison.
                        continue
                        
                    num_dims_to_change = int(CDC * dim)
                    # Handle the case where CDC might result in 0 dimensions
                    if num_dims_to_change == 0:
                        num_dims_to_change = 1 
                        
                    dims_to_change = np.random.choice(dim, num_dims_to_change, replace=False)
                    
                    # Apply mutation: (rand * 2 - 1) gives [-1, 1], multiplied by range * SRD
                    mutation = (np.random.rand(num_dims_to_change) * 2 - 1) * (SRD * search_range)
                    copies[j, dims_to_change] += mutation
                
                # Ensure copies are within bounds
                copies = np.clip(copies, min_val, max_val)
                
                # Evaluate fitness of all copies
                copy_fitness = np.array([objective_func(c) for c in copies])
                
                # Select the best copy
                best_copy_idx = np.argmin(copy_fitness)
                
                # Update the cat's position and fitness to the best copy
                positions[i] = copies[best_copy_idx]
                fitness[i] = copy_fitness[best_copy_idx]

            elif is_tracing[i]:
                # --- 3b. Tracing Mode (Global Search) ---
                
                r1 = np.random.rand(dim)
                
                # Update velocity (PSO-like)
                # V_k(t+1) = w * V_k(t) + c1 * r1 * (X_gbest - X_k(t))
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (global_best_pos - positions[i]))
                
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], min_velocity, max_velocity)
                
                # Update position
                # X_k(t+1) = X_k(t) + V_k(t+1)
                positions[i] = positions[i] + velocities[i]
                
                # Clamp position to bounds
                positions[i] = np.clip(positions[i], min_val, max_val)
                
                # Evaluate new fitness
                fitness[i] = objective_func(positions[i])

        # --- 4. Update Global Best ---
        current_best_fitness = np.min(fitness)
        if current_best_fitness < global_best_fitness:
            global_best_fitness = current_best_fitness
            global_best_pos = positions[np.argmin(fitness)].copy()
            
        if verbose and (t + 1) % 10 == 0:
            print(f"Iteration {t+1}/{max_iter}, Best Fitness: {global_best_fitness:.6e}")
            
    return global_best_pos, global_best_fitness

# --- Main execution ---
if __name__ == "__main__":
    
    # Parameters for the problem
    N_DIMENSIONS = 10 
    N_CATS = 50
    MAX_ITERATIONS = 100
    SEARCH_BOUNDS = (-100, 100) # Search space for each x_i

    print(f"Running CSO to minimize Sphere function in {N_DIMENSIONS} dimensions.")
    print(f"Parameters: Cats={N_CATS}, Iterations={MAX_ITERATIONS}")
    print("---")
    
    # Run the algorithm
    best_solution, best_fitness = cat_swarm_optimization(
        sphere_function,
        dim=N_DIMENSIONS,
        num_cats=N_CATS,
        max_iter=MAX_ITERATIONS,
        search_bounds=SEARCH_BOUNDS
    )
    
    print("---")
    print("Optimization Finished.")
    print(f"Global Best Fitness Found: {best_fitness:.6e}")
    # The Sphere function has its minimum at [0, 0, ..., 0].
    # The best_solution components should be very close to 0.
    print(f"Global Best Position Found (first 5 dims): {best_solution[:5]}")