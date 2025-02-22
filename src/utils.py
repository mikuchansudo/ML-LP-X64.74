import numpy as np

COLORS = ['GREEN', 'RED', 'VIOLET']
SIZES = ['BIG', 'SMALL']
NUMBERS = list(range(10))

SPECIAL_RULES = {
    0: {'size': 'SMALL', 'colors': ['RED', 'VIOLET']},
    5: {'size': 'BIG', 'colors': ['GREEN', 'VIOLET']}
}

def validate_combination(number, size, color):
    if number in SPECIAL_RULES:
        valid_size = SPECIAL_RULES[number]['size']
        valid_colors = SPECIAL_RULES[number]['colors']
        return size == valid_size and color in valid_colors
    return color != 'VIOLET'

def encode_input(number, size, color):
    number_enc = np.zeros(10)
    number_enc[number] = 1
    
    size_enc = np.zeros(2)
    size_enc[SIZES.index(size)] = 1
    
    color_enc = np.zeros(3)
    color_enc[COLORS.index(color)] = 1
    
    return np.concatenate([number_enc, size_enc, color_enc])
