from mealpy import GWO
from mealpy.utils.space import IntegerVar, FloatVar
from rnn import objective_function

'''
Definition of the Mealpy problem for optimizing the RNN parameters.
Parameters:
    obj_func: The objective function to optimize (the precision of the trained RNN).
    variables: The number of variables in the problem.
    variables_info: A dictionary containing the information of each variable.
    bounds: A list containing the bounds of the variables.
    minmax: The optimization goal (minimize or maximize).
    log_to: The output file to log the optimization process.
'''
problem = {
    "obj_func": objective_function,
    "variables": 4,
    "variables_info": {
        "lstm_size1": IntegerVar(10, 150),
        "lstm_size2": IntegerVar(10, 150),
        "num_epochs": IntegerVar(10, 100),
        "learning_rate": FloatVar(0.000001, 0.1),
        "batch_size": IntegerVar(1, 1),
    },
    "bounds": [IntegerVar(10, 150), IntegerVar(10, 150), IntegerVar(10, 100), FloatVar(0.000001, 0.1), IntegerVar(128, 512)],
    "minmax": "max",
    "log_to": "console",
}

# Create the optimizer
optimizer = GWO.OriginalGWO(epoch=10, pop_size=20)

# Optimize RNN parameters using GWO
optimizer.solve(problem, mode="thread", n_workers=5)
print(f"Best solution: {optimizer.g_best.solution}, Best fitness: {optimizer.g_best.target.fitness}")