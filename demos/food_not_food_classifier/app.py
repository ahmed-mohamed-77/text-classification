import torch
import gradio as gr
from typing import Dict
from transformers import pipeline
from transformers.pipelines.base import Pipeline

device = torch.device(device=("cuda" if torch.cuda.is_available() else "cpu"))
MODEL_NAME = "Ahmed5134/hf_food_not_food_text_classifier_distilbert_base_uncased"

# interface function
def food_classifier(text: str) -> Dict[str, float]:
    model_pipeline: Pipeline = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        device=device,
        top_k=None,
        batch_size=32
    )
    model_pipeline = model_pipeline(text)[0]
    pipeline_output = {item["label"] : item["score"] for item in model_pipeline}
    return pipeline_output

# interface
demo = gr.Interface(
    fn=food_classifier,
    title="food not food text classification".title(),
    description="classify the text to food or not food from the random sentence",
    examples=[
            [
                "The cloud looked so fluffy, almost like cotton candy.",
                "She chewed on her pencil while thinking about lunch."
            ]
        ],
    outputs=gr.Label(num_top_classes=2),
    inputs=gr.Textbox()
)

if __name__ == "__main__":
    # run the model interface
    demo.launch()

