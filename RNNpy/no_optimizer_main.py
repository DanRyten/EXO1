from rnn import objective_function

parameters = [15, 10, 2, 0.00502, 1] # [lstm_size1, lstm_size2, num_epochs, learning_rate, batch_size]

print(f"Parameters of the model: {parameters}")

objective_function(parameters)