import random
import math
import copy
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_CITIES = 20
NUM_CATS = 30
ITERATIONS = 100
SMP = 5       # Seeking Memory Pool (how many copies to look at in seeking mode)
SRD = 0.2     # Seeking Range of Dimension (percentage of cities to swap)
MR = 0.3      # Mixture Ratio (probability of being in tracing mode)

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        return math.sqrt((xDis ** 2) + (yDis ** 2))

    def __repr__(self):
        return f"({self.x},{self.y})"

class Cat:
    def __init__(self, tour):
        self.tour = tour
        self.fitness = self.calculate_fitness()
        # Flag: False = Seeking, True = Tracing
        self.mode = False 

    def calculate_fitness(self):
        dist = 0
        for i in range(len(self.tour)):
            from_city = self.tour[i]
            to_city = self.tour[(i + 1) % len(self.tour)]
            dist += from_city.distance(to_city)
        return dist

    # --- SEEKING MODE: Local Search (Mutation) ---
    def seeking_mode(self):
        candidates = []
        # Create copies (clones) of the cat
        for _ in range(SMP):
            new_tour = list(self.tour)
            
            # Perform random swaps (mutation)
            # Standard CSO uses SRD to determine how many dimensions to change
            num_swaps = max(1, int(len(new_tour) * SRD))
            
            for _ in range(num_swaps):
                i, j = random.sample(range(len(new_tour)), 2)
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            candidates.append(Cat(new_tour))

        # Pick the best candidate found in the local search
        best_candidate = min(candidates, key=lambda c: c.fitness)
        
        # If the found neighbor is better than current, move there
        if best_candidate.fitness < self.fitness:
            self.tour = best_candidate.tour
            self.fitness = best_candidate.fitness

    # --- TRACING MODE: Global Search (Crossover) ---
    def tracing_mode(self, best_cat_tour):
        # In continuous CSO, Velocity = Velocity + r * (Best - Current).
        # In Discrete TSP, we use "Order Crossover" to move current tour toward best tour.
        
        size = len(self.tour)
        
        # 1. Select a random sub-segment from the Global Best Cat
        start, end = sorted(random.sample(range(size), 2))
        sub_segment = best_cat_tour[start:end+1]
        
        # 2. Create a new tour using that segment (fix it in place)
        new_tour = [None] * size
        new_tour[start:end+1] = sub_segment
        
        # 3. Fill the remaining spots with cities from the CURRENT cat
        # (maintaining their relative order, omitting cities already in sub_segment)
        current_ptr = 0
        for i in range(size):
            if new_tour[i] is None:
                # Find next city in current self.tour that isn't in sub_segment
                while self.tour[current_ptr] in sub_segment:
                    current_ptr += 1
                new_tour[i] = self.tour[current_ptr]
                current_ptr += 1
                
        self.tour = new_tour
        self.fitness = self.calculate_fitness()

# --- Main Algorithm ---
def solve_cso_tsp():
    # 1. Generate Random Cities
    cities = [City(int(random.random() * 200), int(random.random() * 200)) for _ in range(NUM_CITIES)]
    
    # 2. Initialize Cats (Random Permutations)
    cats = []
    for _ in range(NUM_CATS):
        random_tour = random.sample(cities, len(cities))
        cats.append(Cat(random_tour))

    # Track Global Best
    global_best_cat = min(cats, key=lambda c: c.fitness)
    print(f"Initial Best Distance: {global_best_cat.fitness:.2f}")

    history = []

    # 3. Evolution Loop
    for it in range(ITERATIONS):
        
        for cat in cats:
            # Set Mode Flag (Randomly decide if cat is Seeking or Tracing)
            if random.random() < MR:
                cat.mode = True # Tracing
            else:
                cat.mode = False # Seeking

            if cat.mode:
                # Move towards the global best
                cat.tracing_mode(global_best_cat.tour)
            else:
                # Look around locally
                cat.seeking_mode()

        # Update Global Best if a cat found a better path
        current_best = min(cats, key=lambda c: c.fitness)
        if current_best.fitness < global_best_cat.fitness:
            global_best_cat = copy.deepcopy(current_best)
        
        history.append(global_best_cat.fitness)

    print(f"Final Best Distance: {global_best_cat.fitness:.2f}")
    
    # --- Plotting ---
    x = [c.x for c in global_best_cat.tour]
    y = [c.y for c in global_best_cat.tour]
    # Connect back to start
    x.append(x[0])
    y.append(y[0])

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'co-')
    plt.title(f"Best Route (Dist: {global_best_cat.fitness:.1f})")
    
    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.title("Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    
    plt.show()

if __name__ == "__main__":
    solve_cso_tsp()