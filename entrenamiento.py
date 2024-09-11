import random as rnd

def train_adaline(data_json, alpha, theta, precision):
    print("La red Adaline se está entrenando...")
    print("Con los siguientes parámetros:")
    print("Alpha: ", alpha)
    print("Theta: ", theta)
    print("Precisión: ", precision)
    print("Datos de entrenamiento: ", data_json)
    
    data_input = data_json["entradas"]
    data_output = data_json["salidas"]
