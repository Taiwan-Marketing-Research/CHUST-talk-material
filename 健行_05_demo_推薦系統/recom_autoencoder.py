

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Preparing the training set and the test set
training_set = pd.read_csv('training.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('testing.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and prod
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_prod = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and prod in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_prod = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_prod)
        ratings[id_prod - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


#def selu
#but selu doesn't fit well to this stacked autoencoder
from torch.nn import functional as F
class selu(nn.Module):
    def __init__(self):
        super(selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    def forward(self, x):
        temp1 = self.scale * F.relu(x)
        temp2 = self.scale * self.alpha * (F.elu(-1*F.relu(-1*x)))
        return temp1 + temp2
    
class alpha_drop(nn.Module):
    def __init__(self, p = 0.05, alpha=-1.7580993408473766, fixedPointMean=0, fixedPointVar=1):
        super(alpha_drop, self).__init__()
        keep_prob = 1 - p
        self.a = np.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * pow(alpha-fixedPointMean,2) + fixedPointVar)))
        self.b = fixedPointMean - self.a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        self.alpha = alpha
        self.keep_prob = 1 - p
        self.drop_prob = p
    def forward(self, x):
        if self.keep_prob == 1 or not self.training:
            # print("testing mode, direct return")
            return x
        else:
            random_tensor  = self.keep_prob + torch.rand(x.size())
            binary_tensor = Variable(torch.floor(random_tensor))
            x = x.mul(binary_tensor)
            ret = x + self.alpha * (1-binary_tensor)
            ret.mul_(self.a).add_(self.b)
            return ret


Selu = selu()
dropout_selu = alpha_drop(0.05)

# createing the archite of ae

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

sae = SAE(n_feature=nb_prod, n_hidden=20, n_reduce=10)

#create optimizer
optimizer = torch.optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)
#loss_func = torch.nn.CrossEntropyLoss()  #softmax for classification
loss_func=nn.MSELoss()

# Training the sae
nb_epoch = 1000

#plt.ion()   # something about plotting

for epoch in range(nb_epoch):
    train_loss= 0 #init loss = 0
    s = 0. #init compute rmse
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone() #true value
        
        if torch.sum(target.data > 0) >0: #target.data means we extract torch.FloatTensor from Variable which can't not be computed 
            output = sae(input)
            target.require_grad = False
            output[target ==0] = 0 #to save up some memory, we set vector of target ==0 in output=0, since those = 0 won't update weight
            loss=loss_func(output, target) #calculate MSE
            mean_corrector = nb_prod/float(torch.sum(target.data > 0) + 1e-10) #average error rate that targets are > 0
            loss.backward() #Direction: decide which the weight will be updated, decrease or increase in weights for exampel
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step() #update intensity of weights
            
    print('epoch: '+ str(epoch) + ' loss: '+str(train_loss/s)) 

#save model
torch.save(sae.state_dict(), 'autoencoder.pkl') #save all parameter
#test loss: 1.0699570947

#    if epoch % 2 == 0:
#        plt.cla()
#        prediction = torch.max(nn.functional.softmax(output), 1)[1]
#        pred_y = output.data.numpy().squeeze()
#        target_y = target.data.numpy()
#        plt.scatter(input.data.numpy()[:, ], input.data.numpy()[:, ], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#        plt.text(0.5, 0, 'rmse=%.2f' % float(train_loss/s), fontdict={'size': 20, 'color':  'red'})
#        plt.pause(0.1)
#
#plt.ioff()
#plt.show()

#load model

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

#loss_func = torch.nn.CrossEntropyLoss()  #softmax for classification
loss_func=nn.MSELoss()

sae2 = SAE(n_feature=nb_prod, n_hidden=20, n_reduce=10)
sae2.load_state_dict(torch.load('autoencoder.pkl'))

#test the sae
test_loss = 0
s=0.
out=[]
for id_user in range(nb_users):
    #the input formed by autoencoder will look at the ratings of prod,
    #based on thses ratings, it will predict the prod that users in test set haven't watched.
    #this is the use of bayesian method
        input = Variable(training_set[id_user]).unsqueeze(0) 
        target = Variable(test_set[id_user]) #test set true value
        
        if torch.sum(target.data > 0) >0: #target.data means we extract torch.FloatTensor from Variable which can't not be computed 
            output = sae2(input)
            out.append(output)
            target.require_grad = False
            output[target ==0] = 0 #to save up some memory, we set vector of target ==0 in output=0, since those = 0 won't update weight
            loss=loss_func(output, target) #calculate MSE
            mean_corrector = nb_prod/float(torch.sum(target.data > 0) + 1e-10) #average error rate that targets are > 0
            #loss.backward() #this is only for training
            test_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            #optimizer.step() #update intensity of weights
print('test loss: '+str(test_loss/s))

#see out
extr=[]
recom=[]
for extract in out:
    extr.append({
                    'all': extract.data.numpy().reshape(1682,1),
                    'top20': np.argsort(extract.data.numpy().tolist()[0])[::-1][:20]
                })
    recom.append({'top20': np.argsort(extract.data.numpy().tolist()[0])[::-1][:20]})

df=pd.DataFrame(recom)
df.to_csv('dd.csv', sep='\t', encoding='utf-8')
df.top20[0][:10]

#
##tset
##610 4.475
#input = Variable(training_set[0]).unsqueeze(0) 
#target = Variable(test_set[0]) 
#output = sae(input)
#output[target ==0] = 0
#
#
#inp=input.data.numpy().reshape(1682,1)
#out=output.data.numpy().reshape(1682,1)
#tru=target.data.numpy().reshape(1682,1)

