import torch
from torchviz import make_dot
from rnn import get_inputs, RNN

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

# Visualize the model
print('Plotting model...')
_, _, inputs, _ = get_inputs()
inputs = torch.tensor(inputs, dtype=torch.float32)
outputs = model(inputs)

make_dot(outputs, params=dict(model.named_parameters())).render(f'EMG_RNN_{MODEL_ID}', directory='plots', format='png', cleanup=True)
