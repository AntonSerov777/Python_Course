from fit_model import get_data, fit_catboost_model
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import random
import numpy as np
from variables import LR, EPOCHS, MODEL_BERT


#============FUNCTIONS FOR GET EMBEDINGS WITH TRIPLET LOSS============#

class dataset(Dataset):
    
  def __init__(self, X, y):
    self.X = torch.tensor(X,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.X.shape[0]
 
  def __getitem__(self,idx):
    return self.X[idx],self.y[idx]

  def __len__(self):
    return self.length


class Net(nn.Module):
    
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,128)
    self.fc2 = nn.Linear(128,32)
    self.fc3 = nn.Linear(32,16)
    
  def forward(self,x):
    x = torch.tanh(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x


def fit_tiplet_loss_embed(X, y):
    
    # Model, Optimizer, Loss
    model = Net(input_shape=X.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=10.0, p=2)

    trainset = dataset(X, y)
    trainloader = DataLoader(trainset,batch_size=30,shuffle=True)
    
    for i in range(EPOCHS):
      for j, (X_train, y_train) in enumerate(trainloader):
    
        anchor = model(X_train) 
        positive = torch.zeros(anchor.shape, dtype=torch.float32)
        negative = torch.zeros(anchor.shape, dtype=torch.float32)
        for i in range(X_train.shape[0]):
          positive_choice = anchor[y_train == y_train[i]]
          positive[i] = positive_choice[random.randint(0, positive_choice.shape[0]-1)]
          negative_choice = anchor[y_train != y_train[i]]
          negative[i] = negative_choice[random.randint(0, negative_choice.shape[0]-1)]
            
        loss = loss_fn(anchor, positive, negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(model, "siamise.pt")
        
    return model(torch.tensor(X)).detach().cpu().numpy()


def transform_embed(embed):
    embed = torch.tensor(embed)
    model = torch.load("siamise.pt")
    return model(embed).detach().numpy()


if __name__ == '__main__':
    data, X, y = get_data('data.json', MODEL_BERT)
    embed_X = fit_tiplet_loss_embed(X, y)
    with open('embed_X.npy', 'wb') as f:
        np.save(f, embed_X)
    fit_catboost_model(embed_X, y)
    