# Simulate Artificial Bee Colony (ABC) algorithm. Take sufficient test data; you may choose a problem of your choice to evaluate performance of ABC.
import numpy as np
import matplotlib.pyplot as plt

# Define Objective Function (Sphere Function)
def sphere(x):
    return np.sum(x ** 2)

# Artificial Bee Colony Algorithm
class ArtificialBeeColony:
    def __init__(self, func, n_dim=5, n_bees=30, limit=50, max_iter=200, lb=-5, ub=5):
        self.func = func
        self.n_dim = n_dim
        self.n_bees = n_bees
        self.limit = limit
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub

    def optimize(self):
        # Step 1: Initialize Food Sources
        food_sources = np.random.uniform(self.lb, self.ub, (self.n_bees, self.n_dim))
        fitness = np.array([self._fitness(self.func(x)) for x in food_sources])
        trial = np.zeros(self.n_bees)

        # Best solution tracking
        best_index = np.argmin([self.func(x) for x in food_sources])
        best_solution = food_sources[best_index].copy()
        best_value = self.func(best_solution)
        convergence_curve = []

        # Step 2: Iterations
        for it in range(self.max_iter):

            # Employed Bee Phase
            for i in range(self.n_bees):
                k = np.random.randint(0, self.n_bees)
                while k == i:
                    k = np.random.randint(0, self.n_bees)
                phi = np.random.uniform(-1, 1, self.n_dim)
                new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                new_solution = np.clip(new_solution, self.lb, self.ub)
                new_value = self.func(new_solution)
                if self._fitness(new_value) > fitness[i]:
                    food_sources[i] = new_solution
                    fitness[i] = self._fitness(new_value)
                    trial[i] = 0
                else:
                    trial[i] += 1

            # Onlooker Bee Phase
            prob = fitness / np.sum(fitness)
            for i in range(self.n_bees):
                if np.random.rand() < prob[i]:
                    k = np.random.randint(0, self.n_bees)
                    while k == i:
                        k = np.random.randint(0, self.n_bees)
                    phi = np.random.uniform(-1, 1, self.n_dim)
                    new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                    new_solution = np.clip(new_solution, self.lb, self.ub)
                    new_value = self.func(new_solution)
                    if self._fitness(new_value) > fitness[i]:
                        food_sources[i] = new_solution
                        fitness[i] = self._fitness(new_value)
                        trial[i] = 0
                    else:
                        trial[i] += 1

            # Scout Bee Phase
            for i in range(self.n_bees):
                if trial[i] > self.limit:
                    food_sources[i] = np.random.uniform(self.lb, self.ub, self.n_dim)
                    fitness[i] = self._fitness(self.func(food_sources[i]))
                    trial[i] = 0

            # Memorize best solution
            current_best_index = np.argmin([self.func(x) for x in food_sources])
            current_best_value = self.func(food_sources[current_best_index])
            if current_best_value < best_value:
                best_value = current_best_value
                best_solution = food_sources[current_best_index].copy()

            convergence_curve.append(best_value)
            print(f"Iteration {it+1}/{self.max_iter} — Best Value: {best_value:.6f}")

        return best_solution, best_value, convergence_curve

    def _fitness(self, value):
        if value >= 0:
            return 1 / (1 + value)
        else:
            return 1 + abs(value)

# ----------------------- Run Simulation ----------------------- #
if __name__ == "__main__":
    np.random.seed(42)
    abc = ArtificialBeeColony(func=sphere, n_dim=5, n_bees=40, limit=50, max_iter=200, lb=-5, ub=5)
    best_solution, best_value, curve = abc.optimize()

    print("\n===== Results =====")
    print("Best Solution:", best_solution)
    print("Best Objective Value:", best_value)

    # Plot Convergence Curve
    plt.plot(curve, 'r-', linewidth=2)
    plt.title("Convergence Curve of Artificial Bee Colony on Sphere Function")
    plt.xlabel("Iteration")
    plt.ylabel("Best Objective Value")
    plt.grid(True)
    plt.show()
