import random
import numpy as np
import torch
from torch import nn

strategy_name = 'Trust the PyTorch Neural Network'
class ModelV1(nn.Module):
  def __init__(self, input_shape: int, output_shape: int):
    super().__init__()
    self.layer = nn.Linear(in_features=input_shape, out_features=output_shape)
  def forward(self, x):
    return self.layer(x)

loaded_model = ModelV1(input_shape=3, output_shape=3)
loaded_model.load_state_dict(torch.load(f='C:/Users/imran/Downloads/Rock_Paper_Scissors_Models.pth'))

def move(my_history, their_history):
  mapping = {'r': 0, 'p': 1, 's': 2}
  inverse_mapping = {0: 'r', 1: 'p', 2: 's'}
  beat_mapping = {'r': 'p', 'p': 's', 's': 'r'}
  if len(their_history) < 3:
    action = random.choice(['r', 's', 'p'])
    return action
  elif their_history[len(their_history)-3] == 'x' or their_history[len(their_history)-2] == 'x' or their_history[len(their_history)-1] == 'x':
    action = random.choice(['r', 's', 'p'])
    return action
  else:
    last_3_moves = np.array([their_history[len(their_history)-3], their_history[len(their_history)-2], their_history[len(their_history)-1]])
    last_3_moves = np.vectorize(mapping.get)(last_3_moves)
    last_3_moves = torch.tensor(last_3_moves).type(torch.float)
    predicted_move = loaded_model(last_3_moves)
    predicted_move = torch.argmax(torch.softmax(predicted_move, dim=0), dim=0)
    predicted_move = inverse_mapping[int(predicted_move)]
    action = beat_mapping[predicted_move]
    return action  
