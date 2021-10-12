class Perceptron:
   def __init__(self, eta, epochs):
     self.weights = np.random.randn(3)*1e-4 # initializing the weight before training
     self.eta = eta
     self.epochs = epochs
     print(f"The initialized weight vector: {self.weights}")

   def activationFunction(self, inputs, weights):
     z = np.dot(inputs, weights) # Z = X * W
     return np.where(z>0, 1, 0)

   def fit(self, X, y):
     self.X = X
     self.y = y
     X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
     print(f"X with bias:\n {X_with_bias}")

     for epoch in range(self.epochs):

       print("--"*10)
       print(f"For Epoch: {epoch+1}/{self.epochs}")
       print("--"*10)

       y_hat = self.activationFunction(X_with_bias, self.weights) # Forward Propagation
       print(f"Output after the forward pass: {y_hat}")
       self.error = self.y - y_hat
       print(f"Error: {self.error}")
       self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # Backward Propagation
       print(f"Weight after the backward pass: {self.weights}")
       if(self.total_loss()==0):
         break # Early Stopping
       print("####"*10)


   def predict(self, X):
     X_with_bias = np.c_[X, -np.ones((len(X), 1))]
     return self.activationFunction(X_with_bias, self.weights)

   def total_loss(self):
     total_loss = np.sum(self.error)
     print(f"Total loss: {abs(total_loss)}")
     return total_loss