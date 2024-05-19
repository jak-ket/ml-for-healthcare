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
            
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, state = self.lstm(x)
        logits = self.linear(output)
        probas = self.sigmoid(logits)
        return probas, state


def train_rnn(model, epochs, batch_size, train_loader, device, X_val=None, y_val=None, val_score=None):
    model.train()
    
    criterion = torch.nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    best_score = - np.inf   
    best_weights = None

    for epoch in range(epochs):
        loss_total = 0
        for nr_batch, (X_batch, y_batch) in enumerate(train_loader):
      
            if X_batch.shape[0] != batch_size:
                continue
            
            y_pred, _ = model(X_batch) 
                        
            loss = criterion(y_pred, y_batch.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            loss_total += loss.item()

        if X_val != None:
          
          model.eval()
          y_pred, _ = model(X_val)
          
          
          score = val_score(y_pred.squeeze(1).detach().numpy().round(), y_val) 
          
          score = float(score)
          if score > best_score:
              best_score = score
              best_weights = copy.deepcopy(model.state_dict())
        else:
          score = "not evaluated"
    
        print(f"epoch: {epoch}, train loss: {loss_total / nr_batch}, validation score: {score}")

    
    if X_val != None:
      model.load_state_dict(best_weights)
