print 'This program builds upon a neural network to implement backpropagation \n'

#Import libraries
import numpy as np

class deep_neural_network:
    
    def __init__(self, training_data, training_targets, alpha, hidden_size):
        """This constructor instantiate the neural network"""

        # Welcome Statements
        print 'Deep Neural Network Initiated \n'

        self.training_data = training_data
        self.training_targets = training_targets
        self.alpha = alpha
        self.hidden_size = hidden_size
        self.out_error = 0
        
        np.random.seed(1)

        # Declare Weights
        self.weights_ih = 2 * np.random.random((len(self.training_data[0]), self.hidden_size)) - 1
        self.weights_ho = 2 * np.random.random((self.hidden_size, len(self.training_targets[0]))) - 1
        
        print 'Weights for both layers activated\n .......................................'

    def forward_propagate(self, vector):
        """The Function forward progpagates a vector """

        initial_output = vector
        hidden_output = self.RELU(np.dot(initial_output, self.weights_ih))
        final_output = np.dot(hidden_output, self.weights_ho)
        
        return initial_output, hidden_output, final_output
    
    def backpropagate(self, each_datapoint):
        """The Function Backpropagates the errors only once"""

        #Forward Propagate
        inp, h_out, f_out = self.forward_propagate(self.training_data[each_datapoint : each_datapoint + 1])
        
        #Errors
        self.out_error += np.sum((f_out - self.training_targets[each_datapoint: each_datapoint + 1]) ** 2)

        #Deltas
        out_delta = (self.training_targets[each_datapoint : each_datapoint + 1] - f_out)
        hid_delta = out_delta.dot(self.weights_ho.T) * self.RELU_2_derivative(h_out)

        #Update Weights
        self.weights_ho += self.alpha * h_out.T.dot(out_delta)
        self.weights_ih += self.alpha * inp.T.dot(hid_delta)

        #Print Data for Iteration
        print 'Output = ', f_out
        print 'Error = ', self.out_error
        print ''


    def train(self, epochs):
        """Modelling stochsitc gradient descent, this function trains the model given the epochs"""

        for i in range(epochs):
            print '...........Epoch %s ............' %i
            print ''

            self.out_error = 0
            
            for each_datapoint in range(len(self.training_data)):
                self.backpropagate(each_datapoint)


    def RELU(self, x):
        """Implements Linear activation function for neurons"""

        return (x > 0) * x

    def RELU_2_derivative(self, x):
        """This is the derivative of the RELU activation function"""
        return x > 0

def main():
    # Define the training set and the corresponding targets
    streetlights = np.array( [ [1, 0, 1],
                               [0, 1, 1], 
                               [0, 0, 1],
                               [1, 1, 1] ] )

    walk_vs_stop = np.array([[1, 1, 0, 0]]).T

    training_dataset = streetlights
    training_targets = walk_vs_stop
    
    alpha = 0.2
    epoch = 20
    hidden_size = 4

    #Instatiate the neural network
    ANN = deep_neural_network(training_dataset, training_targets, alpha, hidden_size)
    
    # Train the model 20 times
    ANN.train(epoch)

if __name__ == '__main__':
    main()

