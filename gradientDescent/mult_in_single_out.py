"""
This program implements the gradient descent for multiple inputs and one output

"""

weights = [0.1, 0.2, -0.1]
target = 0.5
alpha = 0.001

#Inputs
toes =  [8.5 , 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2 , 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]]

def w_sum(input_vector, weight_vector):
    """Forward progrpogates based on the currently evaluated weights """
    
    #Check the equalivalnce of both the vectors otherwise halt the code progression
    assert(len(input_vector) == len(weight_vector))
    
    output = 0

    for i in range(len(input_vector)):
        output = output + (input_vector[i] * weight_vector[i])
    
    return output

def ele_multiplication(vector, number):
    output = [0 for i in range(len(vector))]

    for i in range(len(output)):
        output[i] = number * vector[i]

    return output

def neural_network(input_vector, weight_vector):
    """ Calculates the prediction of the neural network """
    prediction = w_sum(input_vector, weight_vector)
    
    #Error calculation 
    error = (prediction - target) ** 2
    delta = prediction - target
    weight_delta = ele_multiplication(input_vector, delta)

    weight_delta[1] = 0.0
    #Weight update
    for i in range(len(weight_vector)):
        weight_vector[i] -= weight_delta[i] * alpha

    print "Weights: ", weight_delta
    print "Weight Deltas: ", weight_delta
    print 'Prediction: ', prediction
    return prediction

def main():
    for epoch in range(1001):
        print "------------------- Epoch %s --------------" %epoch
        neural_network(input, weights)


if __name__ == '__main__':
    main()
