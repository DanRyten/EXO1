import torch
import torch.nn as nn
import random
import glob
import os
from timeit import default_timer as timer
from data_parser import ParsedFile
from windowed_data import WindowedData

TRAINING_FILES = 100

class RNN(nn.Module):
    '''RNN model with 1 fully connected input layer, 2 LSTM layers, and 1 fully connected output layer.'''

    def __init__(self, input_size, hidden_size, lstm_size1, lstm_size2, output_classes, device=torch.device('cpu')):
        '''
        Initialize the RNN model.
        Parameters:
            input_size: The size of the input data.
            hidden_size: The size of the hidden layer.
            lstm_size1: The size of the first LSTM layer.
            lstm_size2: The size of the second LSTM layer.
            output_classes: The number of output classes.
        '''

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_size1 = lstm_size1
        self.lstm_size2 = lstm_size2

        # Define the input layer
        self.input_layer = nn.Linear(int(input_size), int(hidden_size))
        
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(int(hidden_size), int(lstm_size1), batch_first=True)
        self.lstm2 = nn.LSTM(int(lstm_size1), int(lstm_size2), batch_first=True)

        # Initialize hidden states with zeroes
        self.h0_1 = torch.zeros(1, int(self.lstm_size1)).to(device)
        self.c0_1 = torch.zeros(1, int(self.lstm_size1)).to(device)
        self.h0_2 = torch.zeros(1, int(self.lstm_size2)).to(device)
        self.c0_2 = torch.zeros(1, int(self.lstm_size2)).to(device)

        # Define the output layer
        self.output_layer = nn.Linear(int(lstm_size2), int(output_classes))

        return
    
    def forward(self, x):
        '''
        Forward pass through the RNN model.
        Parameters:
            x: The input data.
        Returns:
            out: The output of the model.
            inference_time: The time it took to infer the output.
        '''
        inference_time = 0

        time_start = timer()

        # Forward through input layer
        out = self.input_layer(x)

        # Forward through LSTM layers
        out, _ = self.lstm1(out, (self.h0_1, self.c0_1))
        out, _ = self.lstm2(out, (self.h0_2, self.c0_2))

        out = out[-1, :]

        # Forward through output layer
        out = self.output_layer(out)

        out = torch.nn.functional.softmax(out, dim=0)

        time_end = timer()

        inference_time = time_end - time_start

        return out, inference_time

def objective_function(solution):
    '''
    Objective function used by the optimizer to find the best hyperparameters for the RNN model.
    Parameters:
        solution: list of hyperparameters chosen by the optimizer
    Returns:
        precision: The precision of the model (% of accuracy over a dataset).
    '''
    lstm_size1, lstm_size2, num_epochs, learning_rate, batch_size = solution
    
    # Set the device to GPU if available
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        device = torch.device('cuda')
    else:
        print("No GPU available. Using CPU.")
        device = torch.device('cpu')

    # Create the RNN model
    input_size = 2000  
    hidden_size = 250   
    output_classes = 4  

    creation_params = [input_size, hidden_size, lstm_size1, lstm_size2, output_classes]
    model = RNN(input_size, hidden_size, lstm_size1, lstm_size2, output_classes, device)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.functional.binary_cross_entropy
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Model created")

    # Get the inputs and labels and convert them to tensors
    windowed_train_inputs, windowed_train_labels, windowed_test_inputs, windowed_test_labels = get_inputs()
    train_inputs_tensor = torch.tensor(windowed_train_inputs, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(windowed_train_labels, dtype=torch.long).to(device)

    test_inputs_tensor = torch.tensor(windowed_test_inputs, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(windowed_test_labels, dtype=torch.long).to(device)

    # Create a PyTorch dataset
    train_dataset = torch.utils.data.TensorDataset(train_inputs_tensor, train_labels_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_inputs_tensor, test_labels_tensor)

    # Create a PyTorch dataloader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False)

    print("Data loaded.")
    
    # Train the model
    print("Training started.")
    for epoch in range(int(num_epochs)):
        print(f'Epoch: {epoch + 1}')
        for inputs, labels in train_dataloader:
            # Forward pass
            outputs, _ = model(inputs)
            labels = nn.functional.one_hot(labels, num_classes=output_classes).float().squeeze().to(device)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    print("Evaluation started.")
    precision = 0
    with torch.no_grad():
        model.eval()

        avg_inference_time = 0
        for inputs, labels in test_dataloader:
            outputs, inference_time = model(inputs)
            avg_inference_time += inference_time
            _, predicted = torch.max(outputs.data, 0)
            precision += 1 if [predicted == labels] else 0

        precision = precision / len(windowed_test_labels)
        avg_inference_time = avg_inference_time / len(windowed_test_labels)
    
    print(f'Precision: {precision * 100} ({precision * len(windowed_test_labels)} / {len(windowed_test_labels)})')
    print(f'Average inference time: {avg_inference_time * 1000} ms')

    # Save the model
    model_id = ''
    with open('models/model_id.txt', 'r+') as model_id_file:
        lines = model_id_file.readlines()
        if len(lines) > 2:
            last_line = lines[-2]
            previous_id = int(last_line.split(' ')[0])
        else:
            previous_id = 0
        model_id = previous_id + 1
        model_id_file.write(f'{model_id} {" ".join(map(str, creation_params))} {precision} {avg_inference_time * 1000}\n')
        
    torch.save(model.state_dict(), f'models/EMG_RNN_{model_id}.pth')

    return precision

def get_inputs():
    '''
    Get the inputs and labels for the RNN model.
    Returns:
        train_inputs: The training inputs.
        train_labels: The training labels.
        test_inputs: The testing inputs.
        test_labels: The testing labels.
    '''
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []

    pattern = '/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/Fixed_Data/*_*_*_TestSub*_ARM_*_*.csv'
    files = glob.glob(pattern)

    # Filter files with id under 105
    files = [file for file in files if not (int(os.path.basename(file).split('_')[-1].split('.')[0]) <= 105)]
    random.shuffle(files)

    # Load the data
    for file in files[:TRAINING_FILES]:
        data = ParsedFile(file)

        windowed_data = WindowedData(data, 250, 190)
        windowed_inputs, windowed_classes = windowed_data.getWindows()
        train_inputs.extend(windowed_inputs)
        train_labels.extend(windowed_classes)

    for file in files[TRAINING_FILES:]:
        data = ParsedFile(file)

        windowed_data = WindowedData(data, 250, 190)
        windowed_inputs, windowed_classes = windowed_data.getWindows()
        test_inputs.extend(windowed_inputs)
        test_labels.extend(windowed_classes)

    return train_inputs, train_labels, test_inputs, test_labels
        