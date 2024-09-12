import random 
import copy # To copy the data / this library is used to copy the data and not the reference (only for basic test, not for the real solution)

def train_adaline(data_json, alpha = 0.5, theta = -1, precision = 0.01):
    """
    Function to implement the Adaline network train with the given parameters
    :params
        data_json: JSON with the data to train the network
        alpha: Learning rate
        theta: Bias
        precision: Precision to stop the training
    :return
        weights: Weights obtained after the training
        graph_data: JSON with the data to graph the training process
    """

    # print to know if the parameters are being received correctly
    print("La red Adaline se está entrenando...")
    print("Con los siguientes parámetros:")
    print("Alpha: ", alpha)
    print("Theta: ", theta)
    print("Precisión: ", precision)
    print("Datos de entrenamiento: ", data_json)
    
    # JSON for the graph data
    graph_data = {
        "epochs": [],
        "errors": [],
        "weights": [],
    }


    # Get the data X and Y from the json
    data_input = copy.deepcopy(data_json["entradas"])
    data_output = copy.deepcopy(data_json["salidas"])
    # Get the quantity of inputs (X)
    qty_inputs = len(data_input[0]) + 1 # +1 for the bias
    # Generate random weights for quantity of inputs and the bias
    weights = [round(random.random(),2) for i in range(qty_inputs)]
    obtained_output = [0 for i in range(len(data_output))]
    # number of patterns
    p = len(data_input)
    # add bias (theta) to the inputs in the first position of the list
    for i in range(p):
        data_input[i].insert(0, theta)


    actual_error = random.randint(2, 10)
    last_error = actual_error / 2
    obtained_error = 0
    epoch = 0

    # add the start data to the graph data
    graph_data["epochs"].append(epoch)
    graph_data["errors"].append(actual_error)
    graph_data["weights"].append(weights.copy())
    epoch += 1

    while abs(actual_error - last_error) > precision:
        graph_data["epochs"].append(epoch)
        last_error = actual_error
        for i in range(p):
            obtained_output[i] = sum([weights[j] * data_input[i][j] for j in range(qty_inputs)])

            # update weights
            for j in range(qty_inputs):
                weights[j] = weights[j] + alpha * (data_output[i] - obtained_output[i]) * data_input[i][j]
            
            
            # calculate the obtained error
            obtained_error = obtained_error + (data_output[i] - obtained_output[i])

        # calculate the actual error (LMS - Least Mean Squares)
        actual_error = obtained_error / p
        graph_data["errors"].append(actual_error)
        graph_data["weights"].append(weights.copy())
        obtained_error = 0
        epoch += 1
    
    print("La red Adaline ha sido entrenada con éxito")
    return weights, graph_data, theta

def adaline_aplication(data_json, weights, theta):
    """
    Function to apply the Adaline network with the given weights and theta
    :params
        data_json: JSON with the inputs data to apply the network
        weights: Weights obtained after the training
        theta: Bias
    :return
        obtained_output: Obtained output after the application (Results)
    """

    # print to know if the parameters are being received correctly
    print("La red Adaline se está aplicando...")
    print("Con los siguientes pesos: ", weights)
    print("Con el siguiente theta: ", theta)
    print("Datos de aplicación: ", data_json)

    # Get the data X from the json
    data_input = copy.deepcopy(data_json["entradas"])

    # Get the quantity of inputs (X)
    qty_inputs = len(data_input[0]) + 1 # +1 for the bias

    # add bias (theta) to the inputs in the first position of the list
    for i in range(len(data_input)):
        data_input[i].insert(0, theta)

    obtained_output = [0 for i in range(len(data_input))]

    for i in range(len(data_input)):
        obtained_output[i] = sum([weights[j] * data_input[i][j] for j in range(qty_inputs)])

    print("La red Adaline ha sido aplicada con éxito")
    print("Salida obtenida: ", obtained_output)

    return obtained_output