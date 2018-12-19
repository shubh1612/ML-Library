import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class LinearModel:
    
    
    def __init__(self, solver = 'gd', Lambda = 1, num_iter = 100000, alpha = .01, phi = 'linear', parameters = []):
 
        # Two options for solver: 'gd' or 'ols'
        self.X = None
        self.y = None
        self.Lambda = Lambda                 # regularisation constant
        self.solver = solver                # type of solver: 'ols' or 'gd'
        self.coeff = None                 # coeffiecients of the hypothesis
        self.num_iter = num_iter   # no of iterations for gradient descent
        self.alpha = alpha               # step size for gradient descent
        self.cost_history = None
        self.phi = phi
        self.para = parameters
        
    def cost_func(self, X, y, theta, Lambda):
        m = len(y)
        J = 0
        grad = np.zeros(theta.shape)
        temp = theta
        temp[0] = 0
        J = (sum(np.square(X.dot(theta) - y))/(2*m) + (Lambda/(2*m))*sum(np.square(temp)))
        grad = ((X.T.dot(X.dot(theta)-y)) + (Lambda*temp))/m
        return (J , grad)
        
    # def poly_phi(self, X_train):
    #     degree = self.para[0]
    #     n, m = X_train.shape
    #     X_train_new = np.zeros((n, m*degree))

    #     for i in range(degree):
    #         X_train_new[:, m*i : m*(i+1)] = np.power(X_train, i+1)

    #     return X_train_new

    # def input_vector(self, X_train):
    #     if self.phi == 'linear':
    #         return X_train
    #     elif self.phi == 'poly':
    #         return self.poly_phi(X_train)


    def train(self, X_train, y_train):

        # y_train is a column vector (mX1)
        # X_train is 2d array (mXn)
        # X_train = self.input_vector(X_train)
        m = X_train.shape[0]
        n = X_train.shape[1]
        ones = np.ones((1, m))
        X_aug = np.concatenate((ones, X_train.T), axis = 0).T
        
        self.X = X_aug
        self.y= y_train.reshape((m, 1))

        if self.solver=='ols':
            self.coeff = inv(self.Lambda*np.identity(n+1) + X_aug.transpose().dot(X_aug)).dot(X_aug.transpose()).dot(y_train)

        elif self.solver=='gd':
            a = self.gradient_descent(self.Lambda, self.alpha, self.num_iter)
            self.coeff = a[0]
            self.cost_history = a[1]
            
        
    def gradient_descent(self, Lambda, alpha, iterations):
        
        X = self.X
        y = self.y

        cost_history = [0] * iterations
        
        theta = np.zeros((X.shape[1], 1))
    
        for iteration in range(iterations):
            
            gradient = self.cost_func(X, y, theta, Lambda)[1]

            # Changing Values of theta using Gradient
            theta = theta - alpha * gradient
            # New Cost Value
            cost = self.cost_func(X, y, theta, Lambda)[0]
            cost_history[iteration] = cost

        plt.figure()
        plt.plot(np.arange(len(cost_history)), cost_history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Loss")
        plt.savefig("Images/linear_loss.png")
        plt.show()        
        return (theta, cost_history)

        
    
    def predict(self, X_test):

        # X_test = self.input_vector(X_test)
        m = X_test.shape[0]
        ones = np.ones((1,m))
        X_aug_test = np.concatenate((ones, X_test.T), axis = 0).T
        y_pred = X_aug_test.dot(self.coeff)
        return y_pred
    
    def _get_params(self):
        pass
    
    
if __name__ == "__main__":

    print("Test Linear Model class\n")

    X_train = 10*np.random.randn(10,1)+5
    y_train = 3*X_train +1

    parameters = [2]
    regr = LinearModel(solver='gd', phi = 'poly', parameters = parameters)

    # Train the model using the training sets
    regr.train(X_train, y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(X_train)
    
    # The coefficients
    print('Coefficients: \n', regr.coeff)

    # The mean squared error
    print("Mean squared error: %.2f"
      % mean_squared_error(y_train, diabetes_y_pred))

    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_train, diabetes_y_pred))

    # Plot outputs
    plt.scatter(X_train, y_train,  color='black')
    plt.plot(X_train, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()