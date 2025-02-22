import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from .utils import encode_input, COLORS, SIZES, NUMBERS

class LotteryPredictor:
    def __init__(self):
        self.model = self._build_model()
        self.confidence_scores = {'number': 0, 'size': 0, 'color': 0}
        self.overall_accuracy = 0
        self.sequence_length = 10

    def _build_model(self):
        input_dim = 15
        
        model = models.Sequential([
            layers.LSTM(128, input_shape=(self.sequence_length, input_dim)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_data(self, data):
        X = []
        y = []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:i + self.sequence_length]
            target = data.iloc[i + self.sequence_length]
            
            encoded_sequence = [
                encode_input(row['number'], row['size'], row['color'])
                for _, row in sequence.iterrows()
            ]
            
            encoded_target = encode_input(
                target['number'], 
                target['size'], 
                target['color']
            )
            
            X.append(encoded_sequence)
            y.append(encoded_target)
        
        return np.array(X), np.array(y)

    def train(self, data):
        X, y = self.prepare_data(data)
        
        history = self.model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        val_acc = history.history['val_accuracy'][-1]
        self.confidence_scores = {
            'number': val_acc * 100,
            'size': val_acc * 100,
            'color': val_acc * 100
        }
        self.overall_accuracy = np.mean(list(self.confidence_scores.values()))

    def predict_next(self, recent_data):
        encoded_sequence = np.array([
            [encode_input(row['number'], row['size'], row['color'])]
            for row in recent_data[-self.sequence_length:]
        ])
        
        prediction = self.model.predict(encoded_sequence.reshape(1, -1, 15))[0]
        
        number = np.argmax(prediction[:10])
        size = SIZES[np.argmax(prediction[10:12])]
        color = COLORS[np.argmax(prediction[12:])]
        
        return {
            'number': int(number),
            'size': size,
            'color': color
        }
