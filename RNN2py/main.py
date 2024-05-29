from LSTM import LSTM
from data_parser import get_data
import torch
from torch.utils.data import TensorDataset, DataLoader

TRAIN_PERCENTAGE = 0.8

NUM_EPOCHS = 1

LSTM_SIZE1 = 15
LSTM_SIZE2 = 10

LEARNING_RATE = 0.052

INPUT_SIZE = 2
HIDDEN_SIZE = 50
OUTPUT_CLASSES = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

creation_params = [INPUT_SIZE, HIDDEN_SIZE, LSTM_SIZE1, LSTM_SIZE2, OUTPUT_CLASSES]
model = LSTM(*creation_params)
model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print('Model created')

train_input_tensor, train_output_tensor, test_input_tensor, test_output_tensor = get_data(TRAIN_PERCENTAGE)

train_dataset = TensorDataset(train_input_tensor, train_output_tensor)
test_dataset = TensorDataset(test_input_tensor, test_output_tensor)

batch_size = 32

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

print('Data loaded')

print('Training started')
precision = 0
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch + 1}')
    model.train()
    runningLoss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)

        loss = criterion(outputs, labels.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        runningLoss += loss.item()

    print(f'\tLoss: {runningLoss / (len(train_loader)*32)}')
    precision = model.validate(test_loader)
    print('\n')

# Save the model
model_id = ''
with open('models/model_id.txt', 'r+') as model_id_file:
    lines = model_id_file.readlines()
    if len(lines) >= 2:
        last_line = lines[-2]
        previous_id = int(last_line.split(' ')[0])
    else:
        previous_id = 0
    model_id = previous_id + 1
    model_id_file.write(f'{model_id} {" ".join(map(str, creation_params))} {precision}\n')
    
torch.save(model.state_dict(), f'models/EMG_LSTM_{model_id}.pth')
