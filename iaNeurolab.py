import numpy as np
from neurolab import error
from neurolab import net
from neurolab import train
from neurolab import trans
import utils

# iaNeurolab.py


def create_and_train_neuronal_network(data_set_directory, layers_array, min_value, max_value, train_selected,
                                      error_selected, lr, epochs, show, goal):
    number_of_activate_funtion = len(layers_array)
    sigmoid_function = trans.LogSig()
    neuronal_network = net.newff(minmax=[[min_value, max_value]] * 900,
                                 size=layers_array,
                                 transf=[sigmoid_function] * number_of_activate_funtion)

    trains = {'train_gd': train.train_gd, 'train_gdm': train.train_gdm, 'train_gda': train.train_gda,
              'train_gdx': train.train_gdx}

    errors = {'sse': error.SSE(), 'mse': error.MSE()}

    neuronal_network.trainf = trains[train_selected]
    neuronal_network.errorf = errors[error_selected]

    # Inicializamos los pesos a cero
    for layer in neuronal_network.layers:
        layer.np['w'][:] = 0

    # Inizializamos la red
    neuronal_network.init()

    # Imprime pesos
    # print(neuronal_network.layers[0].np)

    # Valores de entrada
    input_values = utils.letters_to_numpy_array(data_set_directory)

    # Valores de salida
    o = np.array([[1, 0, 0, 0, 0, 0]] * 20)
    p = np.array([[0, 1, 0, 0, 0, 0]] * 20)
    q = np.array([[0, 0, 1, 0, 0, 0]] * 20)
    r = np.array([[0, 0, 0, 1, 0, 0]] * 20)
    s = np.array([[0, 0, 0, 0, 1, 0]] * 20)
    t = np.array([[0, 0, 0, 0, 0, 1]] * 20)

    expected_output = np.concatenate((o, p, q, r, s, t))

    # Entrenamiento
    neuronal_network.train(input_values, expected_output, lr=lr, epochs=epochs, show=show, goal=goal)

    return neuronal_network
