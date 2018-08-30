# BACKPROPAGATION ALGORITHM to solve XOR problem .
The solution to the XOR problem . Perfect place to start if you are new to neural networks .

# ARCHITECTURE :-

--> This is a three layer neural network . The input layer , hidden layer and the final output layer . The neural network involves a
set of algorithms that constitute to make the basis of machine learning followed by deeep learning . Firstly import numpy and matplotlib . 
They are a few important dependecies to make the above code run . There are two nodes in the input layer . The input layer takes in binary values 
The hidden layer contains three nodes . The output layer  has one neuron .


# VARIABLES USED :-
inputLayerSize = INPUT_NODES , outputLayerSize = OUTPUT_NODES , hiddenLayerSize = HIDDEN_NODES , W1 weigth towrds first neuron 
, W2 weigth towards second neuron  , z2 activty of the scond layer  , a2 activation of the second layer  , z3 activity of the third layer
, yHat activation of the final layer , dlta3 is the backpropagating error for the final layer ,  delta2 is the backpropagating error
that is considerd after the second layer and  J cost function .

# STEP'S :-

1 . Initializing the hyperparameters , that is the INPUT_NODES , OUTPUT_NODES and finally the HIDDEN_NODES . As well as initializing
the MAX_ITER values , to maximum allowable number of iterations that an algorithm can perform . ALPH is considering the learning rates .
And

2 . Randomly initializing the weights,between the layers of the neural network .
      .W1 = np.random.random((self.inputLayerSize, self.hiddenLayerSize))  // np. is to make an input of the 
      .W2 = np.random.random((self.hiddenLayerSize, self.outputLayerSize))
     
3 . z2 , the activity of the second layer is the product of the weights and input matrix and finally what comes out is the sumation of all
the bias terms and product of summation and weights . .a2 = self.sigmoid(self.z2)


4 . a2 , this is the activated function that is produced out .a2 = self.sigmoid(self.z2).

5 . z3 , the activity of the third layer is the product of the weights and input matrix and finally what comes out is the sumation of all
the bias terms and product of summation and weights . .a2 = self.sigmoid(self.z2) .

6 . sigmoid(self, z):
        returns  1/(1 + exp(-z)) 
        Using the sigmoid function as the activation function .
        
 7 .  sigmoidPrime(self,z):
        return exp(-z)/((1 + exp(-z)) ** 2)  
       This the derivative of the sigmoid function known as the sigmoidPrime .
       
 8 . To return the cost function 
  J = 0.5*sum((y - self.yHat) ** 2) .
  
  
  9 . Creating the costFunctionPrime 
  This can be done by calculating  dJdW1 and dJdW2 . 
   Return dJdW1, dJdW2 . 
   
   
   10 . Update the gradient descent parameters by iterating the for loop for a certain epoch . 
   
   # INPUT :-
   
   [[0,0],[1,0],[0,1],[1,1]] .
   
   # OUTPUT :-
   
   [[0.03388257],[0.97928562],[0.97928748],[0.00722927]] .
   
   # REFERANCES :-
   http://www.emergentmind.com/neural-network
   
  
  
 
