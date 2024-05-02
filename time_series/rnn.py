import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim

class TimeSeriesClassifierRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first=True,
            # dropout=0.2
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, state = self.lstm(x)
        logits = self.linear(output)
        probas = self.sigmoid(logits)
        return probas, state

    # Get the initial zero state
    # The state of an LSTM is split into two: 
    # One part that represents long-term and one part that represents short-term memory
    # def get_initial_state(self, batch_size, device):
    #     return (
    #        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device), 
    #        torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    #     )


def train_rnn(model, epochs, batch_size, train_loader, device, X_val=None, y_val=None, val_score=None):
    model.train()
    # initial_state = model.get_initial_state(batch_size, device)
    criterion = torch.nn.BCELoss() # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Hold the best model
    best_score = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(epochs):
        loss_total = 0
        for nr_batch, (X_batch, y_batch) in enumerate(train_loader):
      
            if X_batch.shape[0] != batch_size:
                continue
            
            # forward pass
            y_pred, _ = model(X_batch) # only work with logits, not with state
            # print(X_batch.shape, y_batch.shape, y_pred.shape, y_batch.unsqueeze(1).shape)
            # print(y_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # update weights
            optimizer.step()

            loss_total += loss.item()

        if X_val:
          # evaluate accuracy at end of each epoch
          model.eval()
          y_pred, _ = model(X_val)
          # print(y_pred.shape, y_pred.squeeze(1).shape, y_val.shape)
          # print(y_pred)
          score = val_score(y_pred.squeeze(1).detach().numpy().round(), y_val) # .mean()
          # print(score)
          score = float(score)
          if score > best_score:
              best_score = score
              best_weights = copy.deepcopy(model.state_dict())
        else:
          score = "not evaluated"
    
        print(f"epoch: {epoch}, train loss: {loss_total / nr_batch}, validation score: {score}")

    # restore model and return best accuracy
    if X_val:
      model.load_state_dict(best_weights)
