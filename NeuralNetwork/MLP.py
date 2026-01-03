import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def int_to_one_hot( y , num_labels):
    ary = np.zeros((y.shape[0],num_labels))

    for i in range(y.shape[0]):
        ary[i,y[i]] = 1
    return ary



class NeuralNetMLP:
     def __init__(self , num_features, num_hidden , num_classes , random_seed=123):
        super().__init__();
        self.num_features = num_features
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(loc=0.0 , scale = 0.1 , size=(num_hidden,num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(loc=0.0 , scale = 0.1, size=(num_classes,num_hidden))
        self.bias_out = np.zeros(num_classes)
     
     def forward(self, x):
      
      # x is [n_examples, n_features] where each row is an image of 28*28=784 pixels

      # this is for a single hidden Layer
      # input dim: [n_hidden, n_features]
      # dot [n_features, n_examples] .T
      # output dim: [n_examples, n_hidden] -> sctivation for every hidden neuron for all examples at once
      z_h = np.dot(x, self.weight_h.T) + self.bias_h
      a_h = sigmoid(z_h)
      # Output layer
      # input dim: [n_classes, n_hidden]
      # dot [n_hidden, n_examples] .T
      # output dim: [n_examples, n_classes] -> activation for every output neuron for all examples at once
      z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
      a_out = sigmoid(z_out)
      return a_h, a_out
        
if __name__ == "__main__":
    y = np.array([0,1,3]);
    ary = int_to_one_hot(y,num_labels=4)
    print(ary)
    print(sigmoid(ary))
    