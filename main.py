import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


url='https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df=pd.read_csv(url)


df['species']=df['species'].replace('setosa',0.0)
df['species']=df['species'].replace('versicolor',1.0)
df['species']=df['species'].replace('virginica',2.0)


X=df.drop('species',axis=1)
y=df['species']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train_tensor=torch.FloatTensor(X_train.values)
X_test_tensor=torch.FloatTensor(X_test.values)
y_train_tensor=torch.LongTensor(y_train.values)
y_test_tensor=torch.LongTensor(y_test.values)


class Model(nn.Module):

    #Input has 4 layers since we have 4 features
    def __init__(self, input_features=4, h1=10, h2=6, output_features=3):
        super(Model, self).__init__()
        self.fc1=nn.Linear(input_features,h1)
        self.fc2=nn.Linear(h1,h2)
        self.fc3=nn.Linear(h2,output_features)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=Model()

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=0.01)


epoch=100
losses=[]

for e in range(epoch):

    y_pred=model.forward(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e%10==0:
        print(f'Epoch: {e}, loss: {loss}')


with torch.no_grad():
    y_eval=model.forward(X_test_tensor)
    loss=criterion(y_eval,y_test_tensor)
    print(loss)
