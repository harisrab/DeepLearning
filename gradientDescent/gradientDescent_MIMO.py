"""
This algorithm implements multiple input -> multiple output algorthim 
for stochastic gradient descent

"""
#Initial Parameters Adjustment
inputs = [8.5, 0.65, 1.2]

weights = [ [0.1, 0.1, -0.3], 
            [0.1, 0.2, 0.0 ], 
            [0.0, 1.3, 0.1 ] ]

alpha = 0.01

# Weights Matrix must have the following dimensions [outputs][inputs]
#Check correct dimensions before proceeding
try:
    assert(len(weights) == len(targets))
    assert(len(weights[0]) == len(targets))

except AssertionError as error:
    print 'Error Occured: Check dimensions for your weight matrix' 
    

targets = [0.1, 1.0, 0.1]

epochs = 200

def w_sum(v1, v2):
    assert(len(v1) == len(v2))
    
    output = 0

    for i in range(len(v1)):
        output = output + v1[i] * v2[i]

    return output

def matrix_mult(weights, inputs):
    assert(len(weights[0]) == len(inputs))
    
    #Initialize the generalized output array
    output = [0 for i in range(len(inputs))]
    
    #Compute the multiplication
    for i in range(len(weights)):
        output[i] = w_sum(weights[i], inputs)

    return output


def vector_mult(error_deltas, inputs):
    output = [[0 for i in range(len(error_deltas))] for j in range(len(inputs))]
    """
    Output Structure:
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0]

    """

    for j in range(len(inputs)):
        for i in range(len(error_deltas)):
            output[j][i] = inputs[j] * error_deltas[i]
    
    return output

def neural_network(epochs, inputs, weights, targets, alpha):
    """Function runs the core of neural implementation"""

    for epoch in range(epochs):

        #Forward Propagate and calculate the prediction
        prediction = matrix_mult(weights, inputs)

        #Calculate the errors and deltas
        errors = [0 for i in range(len(targets))]
        deltas = [0 for i in range(len(targets))]
    
        for i in range(3):
            deltas[i] = prediction[i] - targets[i]
            errors[i] = (prediction[i] - targets[i]) ** 2

        weight_deltas = vector_mult(deltas, inputs)
    
        for j in range(len(weights)):
            for i in range(len(weights[0])):
                weights[j][i] -= (weight_deltas[j][i] * alpha)
        
        #Print on each iteration
        print''
        print '>'
        print 'Prediction: ', prediction
        print 'Errors: ', errors
        print 'Deltas: ', deltas 

def main():
    #Instantiate the neural network
    neural_network(epochs, inputs, weights, targets, alpha)

if __name__ == '__main__':
    main()


