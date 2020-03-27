def gradient_descent_implementation():
    """This neural network implements the gradient descent"""

    weight, goal_pred, input = (0.0, 0.8, 2)
    alpha = 0.1

    for iteration in range(100):
        #Calculate the prediction
        pred = input * weight
        
        #Error calculation
        error = (pred - goal_pred) ** 2
        delta = pred - goal_pred
        weight_delta = delta * input
        
        #Calculate the new weight
        weight = weight - (weight_delta * alpha)
        
        #Prints the information at each iteration
        print("[%s]---Error: " %iteration + str(error) + " | Prediction: "+ str(pred))

def main():
    #Create function instance
    gradient_descent_implementation()

main()
