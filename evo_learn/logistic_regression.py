import numpy as np
from typing import List, Optional, Callable
from sklearn.preprocessing import StandardScaler
from .quantum_inspired import evolution 

class LogisticRegression:
    def __init__(self) -> None:
        """Initialize the LogisticRegression instance with weights set to None.

        The weights are not initialized with a specific shape yet, as the input data shape
        is unknown until the fit method is called.
        """
        self.weights: Optional[np.ndarray] = None
        self.scaler = None  # Will be set during fit

    def fit(self, X: np.ndarray, y: np.ndarray, gradient_optimizer: bool = False, 
            max_iterations: int = 1000, lr: float = 0.1, regularization: str = 'None', C: float = 0, num_runs: int = 5) -> None:
        """Fit the logistic regression model to the training data.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target labels, shape (n_samples,).
            gradient_optimizer: If True, use gradient descent; otherwise, use evolutionary optimization.
            max_iterations: Maximum number of iterations for optimization.
            lr: Learning rate for gradient descent.

        The method scales the input features, initializes weights if necessary, and optimizes
        them using either gradient descent or an evolutionary algorithm.
        """
        from sklearn.preprocessing import StandardScaler
        # Scale the data to improve numerical stability
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.scaler = scaler  # Save the scaler for later use with test data

        # Initialize weights based on the training data shape if not already set
        # The +1 accounts for the bias term, which has no corresponding feature multiplier
        if self.weights is None:
            self.weights = np.random.randn(X.shape[1] + 1, 1)

        X_train = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y = y.reshape(-1, 1)  # Ensure X_train and y are 2D arrays for shape consistency

        # Choose optimization method: gradient descent or evolutionary algorithm
        if gradient_optimizer:
            self.__gradient_optimizer(X_train, y, max_iterations=max_iterations, lr=lr, regularization=regularization, C=C)
        else:
            self.__evolutionary_optimizer(X_train, y, max_iterations=max_iterations, regularization=regularization, C=C, num_runs=num_runs)

    def __evolutionary_optimizer(self, X: np.ndarray, y: np.ndarray, 
                                 max_iterations: int = 1000, regularization: str = 'None', C: float = 0,
                                 num_runs: int = 5) -> dict:
        """Optimize weights using an evolutionary algorithm over multiple runs.
        
        Args:
            X: Training features with bias term, shape (n_samples, n_features + 1).
            y: Training labels, shape (n_samples, 1).
            max_iterations: Maximum number of iterations for the evolutionary process.
            regularization: Type of regularization ('None', 'l1', 'l2', or 'elastic_net').
            C: Regularization coefficient.
            num_runs: Number of independent evolutionary runs to perform.
            
        Returns:
            A dictionary containing the optimization history (from the best run).
        
        This method runs the evolution process several times and selects the weights with the
        lowest loss (fitness) on the training data.
        """
        best_loss = float('inf')
        best_final_value = None
        best_history = None
        
        # Create the fitness function once; it closes over X and y
        fitness_function = self.__internal_loss_evolutionary(X, y, regularization=regularization, C=C)
        
        for run in range(num_runs):
            final_value, history, population = evolution(
                population_size=2000,
                num_elites=100,
                num_males=100,
                crossover_size=36,
                max_iteration=max_iterations,
                fitness=fitness_function,
                dimensions=X.shape[1],
                maximize=False
            )
            # For 1-dimensional cases, final_value is a float. Wrap it in a list.
            candidate = [final_value] if X.shape[1] == 1 else final_value
            candidate_loss = fitness_function(candidate)
            print(f"Run {run + 1}/{num_runs}, Candidate Loss: {candidate_loss}")
            
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_final_value = final_value
                best_history = history
        
        self.weights = np.array(best_final_value).reshape(-1, 1)
        return best_history

    def __gradient_optimizer(self, X: np.ndarray, y: np.ndarray, 
                            max_iterations: int = 1000, lr: float = 0.1, regularization: str = 'None', C: float = 0) -> List[float]:
        """Optimize weights using gradient descent.

        Args:
            X: Training features with bias term, shape (n_samples, n_features + 1).
            y: Training labels, shape (n_samples, 1).
            max_iterations: Maximum number of iterations for gradient descent.
            lr: Learning rate for weight updates.

        Returns:
            A list of loss values recorded at each iteration.

        The method iteratively updates weights by subtracting the gradient of the loss.
        """
        losses: List[float] = []
        for _ in range(max_iterations):
            predictions = self.__predict_proba_internal(X)
            losses.append(self.loss(predictions, y))
            grad = (X.T @ (predictions - y))
            if regularization == 'l1':
                grad_l1 = (C / X.shape[0]) * np.sign(self.weights)
                grad_l1[0] = 0
                grad += grad_l1
            elif regularization == 'l2':
                grad_l2 = (C / X.shape[0])*2*self.weights
                grad_l2[0] = 0
                grad += grad_l2
            elif regularization == 'elastic_net':
                grad_l1 = (C / X.shape[0]) * np.sign(self.weights)
                grad_l1[0] = 0
                grad_l2 = (C / X.shape[0])*2*self.weights
                grad_l2[0] = 0
                grad += (grad_l1 + grad_l2)
            
            self.weights -= lr * grad


        return losses

    def __internal_loss_evolutionary(self, X_training: np.ndarray, y_training: np.ndarray, regularization: str = 'None', C: float = 0) -> Callable[[List[float]], float]:
        """Create a loss function for the evolutionary algorithm.

        Args:
            X_training: Training features with bias term, shape (n_samples, n_features + 1).
            y_training: Training labels, shape (n_samples, 1).

        Returns:
            A function that takes a list of weights and returns the loss, with training data fixed.

        This closure adapts the loss to the evolutionary algorithm's expected fitness function format.
        """
        def loss_to_optimize(list_x: List[float]) -> float:
            weights = np.array(list_x).reshape(-1, 1)
            z = np.dot(X_training, weights)
            predictions = self.__sigmoid(z)
            weights = np.clip(weights, -250, 250)  # Prevent extreme values
            epsilon = 1e-10  # Small value to prevent log(0)
            predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

            cost = -np.sum(y_training * np.log(predictions) + (1 - y_training) * np.log(1 - predictions)) * 1/X_training.shape[0]
            if regularization == 'l1':
                cost += (C / X_training.shape[0]) * np.sum(np.abs(weights[1:]))
            elif regularization == 'l2':
                cost += (C / X_training.shape[0])*np.sum(np.square(weights[1:]))
            elif regularization == 'elastic_net':
                cost += ((C / X_training.shape[0]) * np.sum(np.abs(weights[1:])) + C*np.sum(np.square(weights[1:])))
                
            return cost
        return loss_to_optimize

    def __predict_proba_internal(self, x: np.ndarray) -> np.ndarray:
        """Compute probability predictions for internal use.

        Args:
            x: Input features with bias term, shape (n_samples, n_features + 1).

        Returns:
            Predicted probabilities, shape (n_samples, 1).

        This method assumes the bias term is already included in the input.
        """
        return self.__sigmoid(np.dot(x, self.weights))

    def __sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply the sigmoid activation function.

        Args:
            x: Input array of logits, any shape compatible with numpy operations.

        Returns:
            Sigmoid-transformed values, same shape as input.

        Clips logits to prevent numerical overflow before applying the sigmoid function.
        """
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Binary predictions (0 or 1), shape (n_samples,).
        """
        return (self.predict_proba(x) >= 0.5).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Compute probability predictions.

        Args:
            x: Input features, shape (n_samples, n_features).

        Returns:
            Predicted probabilities, shape (n_samples, 1).

        Scales the input using the stored scaler and adds the bias term before prediction.
        """
        x = self.scaler.transform(x)
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return self.__sigmoid(np.dot(x, self.weights))

    def loss(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Calculate the negative log-likelihood loss.

        Args:
            predictions: Predicted probabilities, shape (n_samples, 1).
            y: True labels, shape (n_samples, 1).

        Returns:
            The scalar loss value.
        """
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def get_weights(self) -> np.ndarray:
        """Retrieve a copy of the current weights.

        Returns:
            A copy of the weights array, shape (n_features + 1, 1).
        """
        return self.weights.copy()
