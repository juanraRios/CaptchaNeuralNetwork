import glob2
import numpy as np


# utils.py

def pgm_image_to_binary_array(image_path):
    with open(image_path, 'r') as f:
        result = []
        for line in f:
            for word in line.split():
                if word == '255':
                    result.append(1)
                elif word == '0':
                    result.append(0)
        return result[0:900]


# A partir de un directorio devuelve un numpy_array
def letters_to_numpy_array(letters_directory_path):
    file_paths = sorted(glob2.glob(letters_directory_path))
    c = 0
    numpy_array = np.zeros(shape=(120, 900), dtype=np.int)
    for file_path in file_paths:
        array = pgm_image_to_binary_array(file_path)
        numpy_array[c] = array
        c += 1
    return numpy_array
