import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers, lstm_size1, lstm_size2, output_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_size1 = lstm_size1
        self.lstm_size2 = lstm_size2
        self.num_layers = num_lstm_layers

        # Define the input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Define the LSTM layers
        self.lstm1 = nn.LSTM(hidden_size, lstm_size1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_size1, lstm_size2, batch_first=True)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_size, output_classes)

        return
    
    def forward(self, x):
        # Initialize hidden states with zeroes
        h0_1 = torch.zeros(self.num_layers, x.size(0), self.lstm_size1).to(x.device)
        c0_1 = torch.zeros(self.num_layers, x.size(0), self.lstm_size1).to(x.device)

        h0_2 = torch.zeros(self.num_layers, x.size(0), self.lstm_size2).to(x.device)
        c0_2 = torch.zeros(self.num_layers, x.size(0), self.lstm_size2).to(x.device)

        # Forward through input layer
        out = self.input_layer(x)

        # Forward through LSTM layers
        out, _ = self.lstm1(out, (h0_1, c0_1))
        out, _ = self.lstm2(out, (h0_2, c0_2))

        # Take the output of the last time step
        out = out[:, -1, :]

        # Forward through output layer
        out = self.output_layer(out)

        return out

def objective_function(solution):
    lstm_size1, lstm_size2, num_epochs, learning_rate = solution
    
    # Create the RNN model
    input_size = 375  
    hidden_size = 375  
    num_lstm_layers = 2  
    output_classes = 3  
    
    model = RNN(input_size, hidden_size, num_lstm_layers, lstm_size1, lstm_size2, output_classes)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate the model
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        
        # Calculate the precision of the final model
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        precision = (predicted == labels).sum().item() / labels.size(0)
    
    return precision