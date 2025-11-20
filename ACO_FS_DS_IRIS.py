# Apply ACO for features selection from a dataset(IRIS dataset) 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------------- ACO Feature Selection ---------------------- #
class ACOFeatureSelection:
    def __init__(self, n_ants=10, n_iterations=30, alpha=1, beta=1, rho=0.3, q=0.9):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # importance of pheromone
        self.beta = beta    # importance of heuristic info
        self.rho = rho      # evaporation rate
        self.q = q          # pheromone addition factor

    def _evaluate_subset(self, X, y, features):
        """Evaluate subset performance using cross-validation accuracy"""
        if np.sum(features) == 0:  # avoid empty subset
            return 0
        selected_features = np.where(features == 1)[0]
        X_sub = X[:, selected_features]
        model = LogisticRegression(max_iter=1000)
        score = cross_val_score(model, X_sub, y, cv=5).mean()
        return score

    def fit(self, X, y):
        n_features = X.shape[1]
        pheromone = np.ones(n_features)
        best_features = None
        best_score = -1
        best_scores = []

        # Heuristic information (we can use variance or correlation here)
        heuristic = np.std(X, axis=0)  # higher std = more informative

        for it in range(self.n_iterations):
            ant_solutions = []
            ant_scores = []

            for ant in range(self.n_ants):
                # Probability of selecting each feature
                probs = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probs = probs / np.sum(probs)

                # Construct feature subset (binary vector)
                features = np.zeros(n_features)
                for i in range(n_features):
                    if np.random.rand() < probs[i]:
                        features[i] = 1

                # Evaluate
                score = self._evaluate_subset(X, y, features)
                ant_solutions.append(features)
                ant_scores.append(score)

                # Update best
                if score > best_score:
                    best_score = score
                    best_features = features.copy()

            # Pheromone update
            pheromone = (1 - self.rho) * pheromone
            for i in range(n_features):
                for j, features in enumerate(ant_solutions):
                    if features[i] == 1:
                        pheromone[i] += self.q * ant_scores[j]

            best_scores.append(best_score)
            print(f"Iteration {it+1}/{self.n_iterations} — Best Accuracy: {best_score:.4f}")

        self.best_features = best_features
        self.best_score = best_score
        self.best_scores = best_scores

    def get_selected_features(self, feature_names=None):
        indices = np.where(self.best_features == 1)[0]
        if feature_names is not None:
            return [feature_names[i] for i in indices]
        else:
            return indices

# ---------------------- Run on Iris Dataset ---------------------- #
if __name__ == "__main__":
    data = load_iris()
    X, y = data.data, data.target
    X = StandardScaler().fit_transform(X)
    feature_names = data.feature_names

    aco = ACOFeatureSelection(n_ants=15, n_iterations=30, alpha=1, beta=1, rho=0.3, q=0.9)
    aco.fit(X, y)

    selected = aco.get_selected_features(feature_names)
    print("\nBest Feature Subset:", selected)
    print(f"Best Classification Accuracy: {aco.best_score:.4f}")

    # Plot convergence curve
    plt.plot(aco.best_scores, marker='o')
    plt.title("ACO Feature Selection Convergence (Iris Dataset)")
    plt.xlabel("Iteration")
    plt.ylabel("Best Accuracy")
    plt.grid(True)
    plt.show()
