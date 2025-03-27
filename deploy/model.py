from pydantic import BaseModel
from transformers import pipeline

def model_response(text: str):
    BATCH_SIZE = 32
    model_path = r"D:\Python\llm\hugging_face_\models"
    #
    # Load the model and tokenizer from the local path
    model_pipeline = pipeline(
        "text-classification",  # Define the task
        model=model_path,       # Path to your saved model
        tokenizer=model_path    # Path to your saved tokenizer
    )

    # Make a prediction
    result = model_pipeline(text, batch_size=BATCH_SIZE)[0]
    return result


class Question(BaseModel):
    question: str
    
    class Config:
        orm_mode = True


class Answer(BaseModel):
    question: str
    label: str
    score: float
    
    class Config:
        orm_mode = True