
import sys
sys.path.append("function")
sys.path.append("lstm_model")


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import function
from lstm_model.lstm import *

from keras.utils import to_categorical





torch.set_default_tensor_type('torch.DoubleTensor')

####################################
#Setup train and val data

train_origin_des_path=('data/origin_des.txt')
train_route_path=('data/route.txt')
val_origin_des_path=('data/origin_des.txt')
val_route_path=('data/route.txt')

x_train = function.read_origin_des(train_origin_des_path)
y_train = function.read_route(train_route_path)
x_val = function.read_origin_des(val_origin_des_path)
y_val = function.read_route(val_route_path)

print "x_train shape : " , x_train.shape  #"[number of routes][origin and destination][x,y]"
print "y_train shape : " , y_train.shape  #"[number of routes][number of steps]"
print "x_val shape : " , x_val.shape  #"[number of routes][origin and destination
print "y_val shape : " , y_val.shape  #"[number of routes][number of steps]"


####################################
#Setup the model

epoch=10
batch=1
steps_to_predict=40
feature_size=4
hidden_dim=30
batch_size=1
output_dim=1
number_of_routes=x_train.shape[0]


net = M_to_M_LSTM(steps_to_predict=steps_to_predict, feature_size=feature_size, hidden_dim=hidden_dim, batch_size=batch_size, output_dim=output_dim,number_of_routes=number_of_routes)

####################################
#Setup the training 

optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()

####################################
#Let's start training

val_losses = list()
predict_direction = list()

for epoch_indicator in range(epoch):
  
  hc = net.init_hidden()
  
  for batch_indicator in range(batch):

    x_train = torch.from_numpy(x_train[epoch_indicator][:][:])
    print x_train
    y_train = torch.from_numpy(to_categorical(y_train[epoch_indicator][:][:],num_classes=net.predict_dim))

    optimizer.zero_grad()

    net_output=net(x_train, hc)

    loss = criterion(net_output, y_train)

    loss = backward()

    optimizer.step()

    if batch_indicator % 5 == 0:
  
      val_h = net.init_hidden()
      val_c = net.init_hidden()

      #x_val = torch.from_numpy(to_categorical(num_classes=net.predict_dim))
      y_val = torch.from_numpy(to_categorical(y_val,num_classes=net.predict_dim))
   
      val_output = net(x_val, (val_h, val_c))

      val_loss = criterion(val_output, y_val)

      val_losses.append(val_loss.item())

      predict_direction=net.predeict(origin_position=origin_position, steps_to_predict=steps_to_predict)
    
      print "Epoch:", epoch_indicator, "Batch:", batch_indicator, "Train loss:", loss.item, "Validation loss: ", val_loss.item()
       



if __name__ == '__main__':
  main()
        
  




