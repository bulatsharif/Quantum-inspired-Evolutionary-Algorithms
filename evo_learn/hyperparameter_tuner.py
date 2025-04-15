from .logistic_regression import LogisticRegression
from .quantum_inspired import evolution
import numpy as np
from sklearn.metrics import roc_auc_score

class HyperparameterTuner:
    def __init__(self):
        pass

    def tune(self, 
             X_train, y_train, 
             X_val, y_val,
             population_size=5,
             max_iteration=1,
             num_males=2,
             num_elites=1,
             crossover_size=3):
        
        def decode_hyperparameters(float_list):
            float_list = [float(val) for val in float_list]
            
            FLOAT16_MAX = 65504  # Correct maximum value for float16
            float_list = [np.clip(val, -FLOAT16_MAX, FLOAT16_MAX) for val in float_list]
            
            # Normalize to [0, 1] range
            normalized_list = [(val + FLOAT16_MAX) / (2 * FLOAT16_MAX) for val in float_list]
            
            # Extract normalized parameters
            float_optimizer, float_reg, float_C, float_max_iter, float_spec = normalized_list
            
            # Decode optimizer type
            optimizer = 'gradient' if float_optimizer <= 0.5 else 'evolutionary'
            
            # Decode regularization type
            if float_reg <= 0.25:
                regularization = 'None'
            elif float_reg <= 0.5:
                regularization = 'l1'
            elif float_reg <= 0.75:
                regularization = 'l2'
            else:
                regularization = 'elastic_net'
            
            # Clip normalized values to [0, 1] to ensure calculations stay in range
            float_spec = np.clip(float_spec, 0, 1)
            float_C = np.clip(float_C, 0, 1)
            float_max_iter = np.clip(float_max_iter, 0, 1)
            
            # Decode learning rate or number of runs
            if optimizer == 'gradient':
                # Learning rate between 0.001 and 1.0
                num_runs_or_lr = float(10 ** (-3 + 3 * float_spec))
            else:
                # Number of runs between 1 and 5
                num_runs_or_lr = int(1 + 4 * float_spec)
            
            # Decode regularization strength
            C = float(10 ** (-2 + 3 * float_C))
            
            # Decode max iterations (10 to 100)
            max_iters = int(float_max_iter * 90) + 10
            
            return optimizer, regularization, C, max_iters, num_runs_or_lr

        def fitness(float_list):
            optimizer, regularization, C, max_iterations, num_runs_or_lr = decode_hyperparameters(float_list)
            
            model = LogisticRegression()
            model.fit(
                X_train, y_train,
                gradient_optimizer=(optimizer == 'gradient'),
                max_iterations=max_iterations,
                lr=num_runs_or_lr,
                num_runs=int(num_runs_or_lr),
                C=C,
                regularization=regularization,
            )
            
            probas = model.predict_proba(X_val).flatten()
            auc = roc_auc_score(y_val, probas)
            return auc
        
        final_value, history, population = evolution(
            population_size=population_size,
            fitness=fitness,
            dimensions=5,  # One dimension per hyperparameter
            qubits_per_dim=16,  # 16 bits per dimension, totaling 80 bits
            num_males=num_males,
            num_elites=num_elites,
            max_iteration=max_iteration,
            crossover_size=crossover_size,
            maximize=True  # Maximize AUC score
        )
        
        best_iteration = np.argmax(history["queen_fitness"])
        best_float_list = history["queen_value"][best_iteration]
        best_auc = history["queen_fitness"][best_iteration]
        best_hyperparameters = decode_hyperparameters(best_float_list)

        # Re-train the model using the best hyperparameters on the full training data
        optimizer, regularization, C, max_iterations, num_runs_or_lr = best_hyperparameters
        best_model = LogisticRegression()
        best_model.fit(
            X_train, y_train,
            gradient_optimizer=(optimizer == 'gradient'),
            max_iterations=max_iterations,
            lr=num_runs_or_lr,
            num_runs=int(num_runs_or_lr),
            C=C,
            regularization=regularization,
        )

        return best_hyperparameters, best_auc, best_model
