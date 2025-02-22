import gradio as gr
from src.model import LotteryPredictor
from src.data_generator import DataGenerator
from src.utils import validate_combination

predictor = LotteryPredictor()
data_generator = DataGenerator()

initial_data = data_generator.generate_dataset()
predictor.train(initial_data)

def predict(number, size, color):
    if not validate_combination(number, size, color):
        return "Invalid combination! Please check the rules."
    
    input_data = [{
        'number': number,
        'size': size,
        'color': color
    }]
    
    prediction = predictor.predict_next(input_data)
    
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
