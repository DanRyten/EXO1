from mealpy import GWO
import rnn

# TODO: Load dataset, window data, and pass it to the RNN model

# Define the mealpy problem
problem = {
    "obj_func": rnn.objective_function,
    "variables": 4,
    "variables_info": {
        "lstm_size1": [10, 150],
        "lstm_size2": [10, 150],
        "num_epochs": [10, 100],
        "learning_rate": [0.000001, 0.1]
    },
    "minmax": "max",
    "log_to": "console",
}

# Create the optimizer
optimizer = GWO.OriginalGWO(epoch=10, pop_size=20)

# Optimize RNN parameters using GWO
optimizer.solve(problem, mode="thread", n_workers=5)
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")