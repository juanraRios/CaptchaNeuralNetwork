import numpy as np
import iaNeurolab as Ia
import utils

# Datos modificables para la creacion de la red
LAYERS_ARRAY = [50, 6]
MIN_VALUE = 0
MAX_VALUE = 1

# Datos modificables para el entrenamiento de la red
DATA_SET_DIRECTORY = './data_set/*.pgm'
LR = 0.001
EPOCHS = 10000
SHOW = 1000
GOAL = 0.001
TRAIN_SELECTED = 'train_gd'
ERROR_SELECTED = 'sse'

# Directorio del conjunto de prueba
TEST_SET_DIRECTORY = './test_set/*.pgm'

# Valores de entrada, estos se generan a partir de las muestras de prueba
input_values = utils.letters_to_numpy_array(TEST_SET_DIRECTORY)

# Como tenemos 5 letras de cada tipo creamos un vector que contiene 5 vectores(todos iguales con el valor de la letra
# esperada, por ejemplo la 'o' es [1, 0, 0, 0, 0, 0]) por cada letra
o = np.array([[1, 0, 0, 0, 0, 0]] * 5)
p = np.array([[0, 1, 0, 0, 0, 0]] * 5)
q = np.array([[0, 0, 1, 0, 0, 0]] * 5)
r = np.array([[0, 0, 0, 1, 0, 0]] * 5)
s = np.array([[0, 0, 0, 0, 1, 0]] * 5)
t = np.array([[0, 0, 0, 0, 0, 1]] * 5)

# Una vez tenemos todos los vectores concatenamos para obtener un vector de salida esperada
expected_output = np.concatenate((o, p, q, r, s, t))

# Creamos la red neuronal y la entrenamos mediante el modulo propio iaNeurolab en el que internamente hacemos uso
# del modulo neurolab
neuronal_network = Ia.create_and_train_neuronal_network(DATA_SET_DIRECTORY, LAYERS_ARRAY, MIN_VALUE,
                                                        MAX_VALUE, TRAIN_SELECTED, ERROR_SELECTED, LR, EPOCHS, SHOW,
                                                        GOAL)

# Dicionario de letras salida ---> letra
letter = {0: 'O', 1: 'P', 2: 'Q', 3: 'R', 4: 'S', 5: 'T'}

# Creamos los contadores success y fail y los inicializamos a 0
success = 0
fail = 0

# Bucle en el que introducciomos en la red cada una de las muestras y comprobamos si acierta
for i in range(0, 30):
    input_value = input_values[i].reshape(1, 900)
    expected_value = np.argmax(expected_output[i])
    simulacion = neuronal_network.sim(input_value)
    # print('La letra procesada es la ' + letter[np.argmax(simulacion)])
    if np.argmax(simulacion) == expected_value:
        # print('success')
        success += 1
    else:
        # print('fail')
        fail += 1

# Calculamos el ratio de acierto obtenido por la red en el conjunto de prueba
ratio = success*100/(success+fail)

# Imprimimos el ratio formateando este a dos decimales
print('\nSuccess ratio =', '{0:.2f}'.format(ratio))
