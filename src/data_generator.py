import random
import pandas as pd
from datetime import datetime, timedelta
from .utils import COLORS, SIZES, NUMBERS, SPECIAL_RULES

class DataGenerator:
    def __init__(self):
        self.history = []

    def generate_valid_combination(self):
        while True:
            number = random.choice(NUMBERS)
            
            if number in SPECIAL_RULES:
                size = SPECIAL_RULES[number]['size']
                color = random.choice(SPECIAL_RULES[number]['colors'])
            else:
                size = random.choice(SIZES)
                color = random.choice([c for c in COLORS if c != 'VIOLET'])
            
            return {
                'number': number,
                'size': size,
                'color': color,
                'timestamp': datetime.now()
            }

    def generate_dataset(self, size=1000):
        data = []
        current_time = datetime.now()
        
        for i in range(size):
            record = self.generate_valid_combination()
            record['timestamp'] = current_time - timedelta(minutes=i)
            data.append(record)
        
        return pd.DataFrame(data)
