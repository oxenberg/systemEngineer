import numpy as np

from .ex1_functions import (linear_activation_forward, apply_batchnorm,
                            L_layer_model, plot_costs, initialize_parameters, create_batches, compute_cost,
                            softmax_backward, Linear_backward, relu_backward)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MODE = {"predicate": False}
USE_BATCH_NORM = False

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    X_train, X_val, Y_train, Y_val = train_test_split(X.T, Y.T, test_size=0.2, random_state=1)
    X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T
    parameters = initialize_parameters(layers_dims)
    costs = []
    val_costs = []
    epochs = 1500
    num_labels = Y_train.shape[0]
    assert num_labels == layers_dims[-1]

    iterations = 0
    val_prev_cost = np.inf
    done = False

    # TODO: understand whether it should be epochs or num of iteration and the diff between them.
    batches_X, batches_y = create_batches(X_train, Y_train, batch_size)

    for i in tqdm(range(epochs)):
        for batch_x, batch_y in zip(batches_X, batches_y):

            iterations += 1
            ## Predict values
            y_predicted, caches = L_model_forward(batch_x, parameters, USE_BATCH_NORM)
            cost = compute_cost(y_predicted, batch_y)
            grads = L_model_backward(y_predicted, batch_y, caches)
            parameters = Update_parameters(parameters, grads, learning_rate)

            if iterations % 100 == 0:
                costs.append(cost)

                val_predicted, val_caches = L_model_forward(X_val, parameters, USE_BATCH_NORM)
                val_cost = compute_cost(val_predicted, Y_val)
                val_costs.append(val_cost)

                print(f"iteration number:{iterations}, train cost: {cost}, validation cost: {val_cost}")
                # : Stopping criterion
                if val_cost > val_prev_cost or iterations > num_iterations:
                    done = True
                    break
                val_prev_cost = val_cost

        #: Stopping criterion
        if done:
            break
    # TODO: remove val costs from the return
    train_accuracy = Predict(X_train, Y_train, parameters)
    print(f"Train Accuracy is : {train_accuracy}")
    validation_accuracy = Predict(X_val, Y_val, parameters)
    print(f"Validation Accuracy is : {validation_accuracy}")
    return parameters, costs, val_costs



def L_model_forward(X, parameters, use_batchnorm):
    '''

    make forward propagation for epoch

    :param X:the data, numpy array of shape (input size, number of examples)
    :param parameters: dict, initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm
    :return: A the last post-activation value
    :return: caches

    '''
    caches = []
    maskes = []

    A = X.copy()
    parameters_values_list = list(parameters.values())

    for layer, dropout_rate in zip(parameters_values_list[:-1], DROPOUT_RATE[:-1]):
        if MODE["predicate"]:
            dropout_mask = 1 - dropout_rate
        else:
            dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=A.shape)

        A *= dropout_mask
        A, cache = linear_activation_forward(
            A, layer["w"], layer["b"], "relu")

        if use_batchnorm:
            A = apply_batchnorm(A)

        maskes.append(dropout_mask)
        caches.append(cache)


    if MODE["predicate"]:
        dropout_mask = 1 - DROPOUT_RATE[-1]
    else:
        dropout_mask = np.random.binomial(1, 1 - DROPOUT_RATE[-1], size=A.shape)

    softmax_layer = parameters_values_list[-1]
    A, cache = linear_activation_forward(
        A, softmax_layer["w"], softmax_layer["b"], "softmax")


    maskes.append(dropout_mask)

    caches.append(cache)

    for i in range(len(caches)-2,-1,-1):
        caches[i]["mask"] = maskes[i+1]

    return A, caches


def linear_activation_backward(dA, cache, activation):
    '''
        activation is the activation function in the current layer.
        chache: will be a list of tuples - each tuple will have the activation cache as the firse element,
        and the linear cache as the second element
    '''
    linear_cache = cache['linear_cache']
    if activation == 'softmax':
        dZ = softmax_backward(dA, cache['activation_cache'])
        dA_prev, dW, db = Linear_backward(dZ, linear_cache)
    else:
        dZ = relu_backward(dA, cache['activation_cache'])
        dZ *= cache['mask']
        dA_prev, dW, db = Linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def Predict(X, Y, parameters):
    MODE["predicate"] = True
    y_predicted, caches = L_model_forward(X, parameters, USE_BATCH_NORM)
    y_predicted = np.argmax(y_predicted, axis=0)
    Y = np.argmax(Y, axis=0)
    matches = np.where(Y == y_predicted)[0]
    accuracy = len(matches) / len(Y)
    MODE["predicate"] = False
    return accuracy
def L_model_backward( AL, Y, caches):
    '''
    chaches: will be a list of dictionaries - each dictionart will have the activation cache
    and the linear cache of each layer

    '''
    grads = {}
    layers = len(caches)

    dA = AL - Y

    # The last layer:
    grads["dA" + str(layers - 1)], grads["dW" + str(layers)], grads["db" + str(layers)] = \
        linear_activation_backward(dA, caches[layers - 1], "softmax")

    # Rest of the layers:
    for layer in range(layers - 1, 0, -1):
        grads["dA" + str(layer - 1)], grads["dW" + str(layer)], grads["db" + str(layer)] = \
            linear_activation_backward(
                grads["dA" + str(layer)], caches[layer - 1], "relu")

    return grads

def Update_parameters( parameters, grads, learning_rate):
    '''{
        layer_n: {
            w: <matrix of weights from layer n-1 to n>
            row - current layer number of neuron, columns - previous
            b: <vector of bias from layer n-1 to n>
    }'''
    layers = len(parameters)

    for layer in range(1, layers + 1):
        # update w
        parameters["layer_" + str(layer)]['w'] = parameters["layer_" +
                                                            str(layer)]['w'] - learning_rate * grads[
                                                     'dW' + str(layer)]
        # update b
        parameters["layer_" + str(layer)]['b'] = parameters["layer_" +
                                                            str(layer)]['b'] - learning_rate * grads[
                                                     'db' + str(layer)]

    return parameters


def run_with_dropout(x_train, x_test, y_train, y_test,batch_size,num_iterations,learning_rate,layers_dim,dropout = [0, 0.1, 0, 0]):
    global DROPOUT_RATE
    DROPOUT_RATE = dropout
    parameters, costs, val_costs = L_layer_model(x_train, y_train, layers_dim,learning_rate, num_iterations, batch_size)
    accuracy = Predict(x_test, y_test, parameters)
    print(f"Test Accuracy is : {accuracy}")
    plot_costs(costs, val_costs, "With Dropout")