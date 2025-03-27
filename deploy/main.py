from fastapi import FastAPI, HTTPException, Depends
from model import Question, Answer, model_response
from database import ClassifierResponse, Base, engine
from sqlalchemy.orm import sessionmaker
import uvicorn
import gradio as gr
import requests
import threading

# Create tables
Base.metadata.create_all(bind=engine)

# Define session class
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Dependency to get DB session
def get_db():
    db = SessionLocal()  # Instantiate session
    try:
        yield db  # Yield session to FastAPI
    finally:
        db.close()  # Ensure session is closed

app = FastAPI()

@app.get(path="/")
def home_page():
    return {"description" : "Hello, World"}

@app.post("/food_not_food", response_model=Answer)
def model_prediction(user_response: Question=None, db = Depends(get_db)):
    
    response = user_response.question
    if response is None:
        raise HTTPException(status_code=404, detail="question or comment not found")
    
    output = model_response(text=response)
    
    new_record = ClassifierResponse(
        question=response,
        label=output["label"],
        score=output["score"]
    )
    
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    
    # Return the response model with the correct data
    return Answer(question=response, label=output["label"], score=output["score"])


def gradio_interface(question: str):
    # sent the user's input to the FASTAPI
    response = requests.post(
        url="http://127.0.0.1:8000/food_not_food",
        json={"question": question}
    )
    return response.json()["label"], response.json()["score"]

def run_gradio():
    gradio_ui = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(),
        outputs=[gr.Label(num_top_classes=1), gr.Number()],  # Since your function returns two values
        title="Food Not Food Text Classification",
        description="Classify the text to food or not food from the random sentence",
        examples=[
            "The cloud looked so fluffy, almost like cotton candy.",
            "She chewed on her pencil while thinking about lunch."
        ]
    ).launch(share=True, server_name="0.0.0.0", server_port=7860, inline=False)
    return gradio_ui

@app.on_event("startup")
def start_gradio():
    threading.Thread(target=run_gradio).start()

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)