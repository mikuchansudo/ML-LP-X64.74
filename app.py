import gradio as gr
from src.model import LotteryPredictor
from src.data_generator import DataGenerator
from src.utils import validate_combination
import numpy as np

# Initialize predictor and generate initial data
predictor = LotteryPredictor()
data_generator = DataGenerator()
initial_data = data_generator.generate_dataset()
predictor.train(initial_data)

# Keep track of recent inputs
recent_inputs = []

def predict(number, size, color):
    global recent_inputs
    
    if not validate_combination(number, size, color):
        return "Invalid combination! Please check the rules."
    
    # Add new input to recent inputs
    recent_inputs.append({
        'number': number,
        'size': size,
        'color': color
    })
    
    # Keep only the last 10 inputs
    if len(recent_inputs) > 10:
        recent_inputs = recent_inputs[-10:]
    
    # If we don't have enough data yet, use some from initial_data to pad
    if len(recent_inputs) < 10:
        padding_needed = 10 - len(recent_inputs)
        padding_data = initial_data.iloc[-padding_needed:].to_dict('records')
        prediction_data = padding_data + recent_inputs
    else:
        prediction_data = recent_inputs
    
    # Make prediction
    prediction = predictor.predict_next(prediction_data)
    
    result = f"""
    Predicted Next Combination:
    Number: {prediction['number']}
    Size: {prediction['size']}
    Color: {prediction['color']}
    
    Model Performance:
    Overall Accuracy: {predictor.overall_accuracy:.2f}%
    
    Confidence Scores:
    Number: {predictor.confidence_scores['number']:.2f}%
    Size: {predictor.confidence_scores['size']:.2f}%
    Color: {predictor.confidence_scores['color']:.2f}%
    
    Number of historical inputs used: {len(recent_inputs)}/10
    """
    
    return result

iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Number (0-9)", minimum=0, maximum=9),
        gr.Dropdown(choices=["BIG", "SMALL"], label="Size"),
        gr.Dropdown(choices=["GREEN", "RED", "VIOLET"], label="Color")
    ],
    outputs=gr.Textbox(label="Prediction Results"),
    title="Lottery Predictor",
    description="""
    Enter the current combination to predict the next outcome.
    The model uses the last 10 combinations to make predictions.
    
    Rules:
    - Numbers: 0 to 9
    - Sizes: BIG or SMALL
    - Colors: GREEN, RED, or VIOLET
    - Special combinations:
      * 0: SMALL RED+VIOLET only
      * 5: BIG GREEN+VIOLET only
    """,
    theme="darker"
)

if __name__ == "__main__":
    iface.launch()
