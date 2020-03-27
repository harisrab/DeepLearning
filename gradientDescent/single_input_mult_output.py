"""
This code implements single input to multiple output algorithm 
for a basic understanding of gradient descent implementation

(Basic version of Stochastic Gradient Descent)

"""

#Define Inputs
input = 0.65
weights = [0.3, 0.2, 0.9]
targets = [0.1, 1.0, 0.1]

epoch = 100

def elem_multiplication(vector, number):

    output = [0 for i in range(len(vector))]

    for i in range(len(vector)):
        output[i] = vector[i] * number
    
    return output

def CalculateError(vect1, vect2):
    
    output = [0 for i in range(len(vect1))]

    for i in range(len(vect1)):
        output[i] = (vect1[i] - vect2[i]) ** 2
    
    return output

def ElementWise_Sub(vect1, vect2):

    output = [0 for i in range(len(vect1))]

    for i in range(len(vect1)):
        output[i] = vect1[i] - vect2[i]

    return output



def neural_network(weights, targets, input, epoch):

    for i in range(epoch):

        pred = elem_multiplication(weights, input)
    
        #calculate error
        error = CalculateError(pred, targets)
        delta = ElementWise_Sub(pred, targets)
        weight_delta = elem_multiplication(delta, input)

        #update weights
        weights = ElementWise_Sub(weights, weight_delta)


        print("Prediction = ", pred)




def main():
    neural_network(weights, targets, input, epoch)
    


if __name__ == '__main__':
    main()
