import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_size1, lstm_size2, output_classes):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_size1 = lstm_size1
        self.lstm_size2 = lstm_size2

        #self.input_layer = nn.Linear(int(input_size), int(hidden_size))
        self.lstm1 = nn.LSTM(int(input_size), int(lstm_size1), batch_first=True)
        self.lstm2 = nn.LSTM(int(lstm_size1), int(lstm_size2), batch_first=True)

        self.output_layer = nn.Linear(int(lstm_size2), int(output_classes))

    def forward(self, x):
        #out = self.input_layer(x)
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.output_layer(out[:, -1, :])
        out = torch.nn.functional.softmax(out, dim=1)
        return out
    
    def validate(self, test_loader):
        self.eval()
        precision = 0
        correct_classes = [[0,0] for _ in range(4)]
        loss = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                loss += criterion(outputs, labels.squeeze()).item()
                _, predicted = torch.max(outputs.data, 1)
                precision += (predicted == labels.squeeze()).sum().item()
                for i in range(len(predicted)):
                    correct_classes[labels[i]][0] += 1
                    if predicted[i] == labels[i]:
                        correct_classes[labels[i]][1] += 1

        precision = precision / (len(test_loader) * 32)
        print(f'\tValidation precision: {precision * 100} ({precision * len(test_loader) * 32} / {len(test_loader) * 32})')
        for i in range(4):
            print(f'\tClass {i}: {correct_classes[i][1]} / {correct_classes[i][0]} ({correct_classes[i][1] / correct_classes[i][0] * 100}%)')

        return precision