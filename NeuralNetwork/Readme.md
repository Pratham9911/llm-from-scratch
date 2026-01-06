### Neural Networks

1 implemented one hot encoding:
  - it is used to convert number to truth values for nn which matches it's output
  - digit=3 and possible numbers = 0-4
    then truth value is [0,0,0,1,0]
  - without this output may mismatch from nn output

2 Gradient Decent :
  - it is used to train neural network , it identifies the best weights to learn the pattern
  - nn starts with random weight and during backpropogation they learn how to reduce loss
  - Gradient find the minima of the cost of network at each stage of training unless it finds local or global minima
  -summing the difference btw target and predicted value's square tells loss about that training , doing this for all training session and taking avg is cost.
  - gradient tells slope (error inc) and it's opposite (-grad) tells how to dec error

3 Back Propogation:
 - Backpropagation identifies how much each intermediate value (neuron, weight, bias) contributed to the final output error.