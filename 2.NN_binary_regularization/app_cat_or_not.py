import numpy as np
import matplotlib.pyplot as plt
import h5py

from dnn_utils import *
from propagation_forward import *
from propagation_backward import *


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, lambd = 0, keep_prob = 1):#lr was 0.009

    """    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 cat / 0 not cat), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        
        # Cost function
        if lambd == 0:
            cost = compute_cost(AL, Y)
        else:
            cost = compute_cost_with_regularization(AL, Y, parameters, lambd)

    
        # Backward propagation.
        assert(lambd==0 or keep_prob==1)    # it is possible to use both L2 regularization and dropout, 
                                            # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = L_model_backward(AL, Y, caches)
        elif lambd != 0:
            grads = L_model_backward(AL, Y, caches, lambd)

 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


## RUN THE MODEL
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
parameters_reg = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True, lambd = 0.7)

## SAVE
np.save('parameters.npy', parameters) 
np.save('parameters_reg.npy', parameters_reg) 


###########################################################
## LOAD  
parameters = np.load('parameters.npy').item()
parameters_reg = np.load('parameters_reg.npy').item()


print("\n")
print ("-->On the train set:")
predictions_train = predict(train_x, train_y, parameters)
print ("-->On the test set:")
predictions_test = predict(test_x, test_y, parameters)
print("\n")

print("\n")
print ("-->Regularized On the train set:")
predictions_train = predict(train_x, train_y, parameters_reg)
print ("-->Regularized On the test set:")
predictions_test = predict(test_x, test_y, parameters_reg)
print("\n")

