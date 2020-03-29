#Import libraries
import numpy as np

#Initilize the neural network
weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1

streetlights_states = np.array( [ [1, 0, 1],
                                  [0, 1, 1],
                                  [0, 0, 1],
                                  [1, 1, 1],
                                  [0, 1, 1],
                                  [1, 0, 1] ] )

command = np.array([0, 1, 0, 1, 1, 0])

def forwardPropagate(inputs, weights):
    output = inputs.dot(weights)
    return output

def neural_network(streetlights_states, command, weights, alpha):
    

    for epoch in range(200):
        for each_command in range(len(command)):
            
            inputs = streetlights_states[each_command]
            target = command[each_command]

            #Forward Propogate
            prediction = forwardPropagate(inputs, weights)

            #Error
            error = (prediction - target) ** 2
            delta = prediction - target
            weight_delta = inputs * delta

            #Update Weight
            weights -= (weight_delta * alpha)
            
        
        print '>'
        print 'Prediction = ', prediction
        print 'Error = ', error 
    
    
    print '....................................'
    print ''
    test_input = np.array([1, 1, 1])
    
    command = forwardPropagate(test_input, weights)
    
    if (command < 0.2):
        print "Don't Walk"
    
    else:
        print "Walk"


def main():
    neural_network(streetlights_states, command, weights, alpha)



if __name__ == '__main__':
    main()
