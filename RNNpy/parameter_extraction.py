import torch
import csv
from rnn import RNN

MODEL_ID = 2

model_file = f'models/EMG_RNN_{MODEL_ID}.pth'

# Load the model parameters
print(f'Loading model parameters...')
with open('models/model_id.txt', 'r') as file:
    lines = file.readlines()
    line = lines[MODEL_ID - 1].strip()
    parameters = line.split()[1:]

# Create the base model and load the weights
print('Creating model...')
model = RNN(*map(int, parameters))
model.load_state_dict(torch.load(model_file))
model.eval()

model_params = {}
for name, param in model.named_parameters():
    model_params[name] = param.detach().numpy()

csv_file = f'models/EMG_RNN_{MODEL_ID}_parameters.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for name, param in model_params.items():
        writer.writerow([name])
        writer.writerow(param.flatten())

print(f'Parameters saved to {csv_file}')