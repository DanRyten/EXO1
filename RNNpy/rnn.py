import torch
import torch.nn as nn
import glob
import os
from data_parser import ParsedFile
from windowed_data import WindowedData

FILES_TO_LOAD = 76

class RNN(nn.Module):
    '''RNN model with 1 fully connected input layer, 2 LSTM layers, and 1 fully connected output layer.'''

    def __init__(self, input_size, hidden_size, lstm_size1, lstm_size2, output_classes):
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

        # Define the output layer
        self.output_layer = nn.Linear(int(lstm_size2), int(output_classes))

        return
    
    def forward(self, x):
        '''
        Forward pass through the RNN model.
        Parameters:
            x: The input data.
        '''

        # Initialize hidden states with zeroes
        h0_1 = torch.zeros(1, int(self.lstm_size1)).to(x.device)
        c0_1 = torch.zeros(1, int(self.lstm_size1)).to(x.device)
        h0_2 = torch.zeros(1, int(self.lstm_size2)).to(x.device)
        c0_2 = torch.zeros(1, int(self.lstm_size2)).to(x.device)

        # Forward through input layer
        out = self.input_layer(x)

        # Forward through LSTM layers
        out, _ = self.lstm1(out, (h0_1, c0_1))
        out, _ = self.lstm2(out, (h0_2, c0_2))

        out = out[-1, :]

        # Forward through output layer
        out = self.output_layer(out)

        out = torch.nn.functional.softmax(out, dim=0)

        return out

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
    hidden_size = 1000   
    output_classes = 4  

    model = RNN(input_size, hidden_size, lstm_size1, lstm_size2, output_classes)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.functional.binary_cross_entropy
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Model created")

    # Get the inputs and labels and convert them to tensors
    windowed_inputs, windowed_labels = get_inputs(FILES_TO_LOAD)
    inputs_tensor = torch.tensor(windowed_inputs, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(windowed_labels, dtype=torch.long).to(device)

    # Create a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)

    # Create a PyTorch dataloader
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False)

    print("Data loaded.")
    
    print("Model started.")

    # Train the model
    for epoch in range(int(num_epochs)):
        print(f'Epoch: {epoch + 1}')
        for inputs, labels in dataloader:
            # Forward pass
            outputs = model(inputs)
            labels = nn.functional.one_hot(labels, num_classes=output_classes).float().squeeze().to(device)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model
    precision = 0
    with torch.no_grad():
        model.eval()

        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 0)
            precision += 1 if [predicted == labels] else 0

        precision = precision / len(windowed_labels)
    
    print(f'Precision: {precision * 100} ({precision * len(windowed_labels)} / {len(windowed_labels)})')

    # Save the model
    torch.save(model.state_dict(), 'EMG_RNN.pth')

    return precision

def get_inputs(num_files):
    '''
    Get the inputs and labels for the RNN model.
    Parameters:
        num_files: The number of files to load (starts from file id 110).
    '''
    inputs = []
    labels = []

    pattern = '/home/fer/Uni/Erasmus/EXO/EXO-Data-Repository/*_*_*_TestSub*_ARM_*_*.csv'
    files = glob.glob(pattern)

    a = os.path.basename(files[0]).split('_')[-1].split('.')[0]

    # Filter files with id under 105
    files = [file for file in files if not (int(os.path.basename(file).split('_')[-1].split('.')[0]) <= 105)]

    # Load the data
    for i in range(num_files):
        #data = ParsedFile(f'C:\\Uni\\MDU\\EXO\\EXO1-feature_RNN\\RNNpy\\Data\\2024_4_6_TestSub20_ARM_L_{105 + i}.csv')
        data = ParsedFile(files[i])

        windowed_data = WindowedData(data, 250, 190)
        windowed_inputs, windowed_classes = windowed_data.getWindows()
        inputs.extend(windowed_inputs)
        labels.extend(windowed_classes)

    return inputs, labels
        