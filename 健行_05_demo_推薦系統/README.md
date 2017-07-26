# Hackntu2017_demo
This repo is for the demo purpose of Hackntu Data science program.

# About
Recommendation System core algorithms made using AutoEncoder.
Pytorch is mainly used to build AutoEncoder.

1.training.base
- training data
  - first column represents ```ID```
  - Sec column represents ```product number```
  - third column represents ```classified web-browsing time category```, which is our target variables
 
2.testing.test
- test data
 
3.recom_autoencoder.py
- core algorithms and codes of movies recommendation system

4.hackntu.pptx
- ppt for presentation

5.autoencoder.pkl (optional)
* pretrained model for this demo data
* you can first define 
```python

class SAE(nn.Module):
    
    #---define autoencoder---#
    def __init__(self,  n_feature=nb_prod, n_hidden=20, n_reduce=10,): 
        super(SAE, self).__init__() #get inheritance from nn.module
        self.fc1 = nn.Linear(n_feature, n_hidden) #first connection with 20 elements hidden neurons
        self.fc2 = nn.Linear(n_hidden, n_reduce) #20 to 20 and create another 10 neurons
        self.fc3 = nn.Linear(n_reduce, n_hidden) #10 to 10 and decodeing
        self.fc3.a = nn.Linear(n_hidden, 20) #10 to 10 and decodeing
        self.fc3.b = nn.Linear(20, 20) #10 to 10 and decodeing
        self.fc4 = nn.Linear(20, n_feature) #finish decoding and make output
        self.activation = nn.Sigmoid() #this can be deleted
    
    #---feedforward function---#
    def forward(self, x):
        x=self.activation(self.fc1(x)) #and writes: x=nn.Sigmoid(self.fc1(x))...
        x=self.activation(self.fc2(x))
        x=self.activation(self.fc3(x))
        x=self.activation(self.fc3.a(x))
        x=self.activation(self.fc3.b(x))
        x=self.fc4(x)
        return x
   
#create optimizer
optimizer = torch.optim.Adadelta(sae.parameters(), lr=0.01, weight_decay=0.5)
#loss_func = torch.nn.CrossEntropyLoss()  #softmax for classification
loss_func=nn.MSELoss()

sae2 = SAE(n_feature=nb_movies, n_hidden=20, n_reduce=10)
```
* then ```torch.load autoencoder.pkl``` as follows:
 ```python
 sae2.load_state_dict(torch.load('autoencoder.pkl'))
 ```
 * and directly replace ```sae``` with your customized ```sae2``` [here](https://github.com/HowardNTUST/Hackntu2017_demo/blob/master/recom_autoencoder.py#L167)
  
