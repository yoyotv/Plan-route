import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from keras.utils import to_categorical

class M_to_M_LSTM(nn.ModuleList):

      
      def __init__(self, steps_to_predict, feature_size, hidden_dim, batch_size, output_dim,number_of_routes):
        
        #M_to_M_LSTM.__init__(self)
        super(M_to_M_LSTM, self).__init__()

        self.steps_to_predict = steps_to_predict
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.predict_dim = 4
        self.number_of_routes = number_of_routes

        self.lstm1 = nn.LSTMCell(input_size = feature_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size = hidden_dim, hidden_size=hidden_dim)

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim)


      def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),torch.zeros(self.batch_size, self.hidden_dim))

      def init_hidden_predict(self):
        return (torch.zeros(1,self.hidden_dim),
                tprch.zeros(1,self.hidden_dim))
      
      def forward(self,x,hc):
       
        self.train()

        output = torch.empty((self.steps_to_predict, self.batch_size, self.predict_dim))
       
        hc1, hc2=hc, hc

        for t in range(self.steps_to_predict):

          hc1 = self.lstm1(x,hc1)
          h1, c1= hc1

          hc2 = self.lstm2(h1,hc2)
          h2, c2= hc2

          output=self.fc(self.dropout(h2))
          
          
        return output

      def predict(self, origin_position, steps_to_predict):

         self.eval()

         direction_route = np.empty(steps_to_predict+1)
           
         hc=self.init_hidden_predict()
      
         hc1, hc2 = hc, hc
     
         for i in range(self.steps_to_predict):

           y_direction = to_categorical(y_direction, num_classes=self.predict_dim)
         
           y_direction = torch.from_numpy(y_direction).unsqueeze(0)
   
           hc1 = self.lstm1(char, hc1)
           h1, _ = hc1

           hc2 = self.lstm2(h1, hc2)
           h2, _ = hc_2

           h2 = self.fc(h2)
           h2 = nn.functional.softmax(h2, dim = 1)

           #_, each_probability=h2.topk(self.predict_dim)

           direction, probability=h2.topk(1)

           direction_route[i] = direction

         return direction_route 


if __name__ == '__main__':
  main()
           
           



















