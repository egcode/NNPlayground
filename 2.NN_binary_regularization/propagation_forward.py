import numpy as np

def initialize_parameters_deep(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network


    for l in range(1, L):   # Skipping first layer, since it's input
        
        #parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])*0.01 # HE initialization with ReLU
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def sigmoid(Z):
    
    A = 1/(1+np.exp(-Z))
    
    return A

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    return A


def linear_forward(A_prev, W, b):
    
    Z = W.dot(A_prev) + b
    
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    return Z

def linear_activation_forward(A_prev, W, b, activation):
    
    Z = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)
        
    linear_cache = (A_prev, W, b)
    activation_cache = Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# =============================================================================
def linear_activation_forward_with_dropout(A_prev, W, b, keep_prob, layer, activation):
    
    if layer != 1: # Everywhere except if A_prev == X, or last layer A_prev == AL
        # Dropout with relu layers
        D = np.random.rand(A_prev.shape[0], A_prev.shape[1])     
        D = (D < keep_prob)                            
        A_prev = np.multiply(A_prev, D)                         
        A_prev = A_prev / keep_prob 
    else:
        D = np.ones((A_prev.shape[0], A_prev.shape[1]))    
        
    Z = linear_forward(A_prev, W, b)    
    if activation == "sigmoid":
        A = sigmoid(Z)

    elif activation == "relu":
        A = relu(Z)

    linear_cache = (A_prev, W, b, D)
    activation_cache = Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    assert (A_prev.shape == D.shape)

    cache = (linear_cache, activation_cache)
                             
    return A, cache


def L_model_forward_with_dropout(X, parameters, keep_prob = 1):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward_with_dropout(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], keep_prob, l, activation = "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward_with_dropout(A, parameters['W' + str(L)], parameters['b' + str(L)], keep_prob, 1, activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
# =============================================================================

def compute_cost(AL, Y):
    """
    Implements cross entropy cost.

    """
    m = Y.shape[1]
    
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization.
    
    Arguments:
    AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    parameters -- python dictionary containing parameters of the model
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    m = Y.shape[1]    
    cross_entropy_cost = compute_cost(AL, Y)
    frobenius_norm_square = 0.
    
    L = (len(parameters) // 2) + 1
    for l in range(1, L): 
        W = parameters['W' + str(l)]
        frobenius_norm_square = frobenius_norm_square + np.sum(np.square(W))
    
    L2_regularization_cost = (1./m) * (lambd/2.) * (np.sum(frobenius_norm_square))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p


