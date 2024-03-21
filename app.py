import os
import sys
import uvicorn

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from textSummarization.pipeline.prediction import PredictionPipeline

text:str = "What is Text Sumarization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training Succcessful!!")
    except Exception as e:
        return Response(f"Error Ocuirred: {e}")

@app.post("/predict")
async def prediction(text):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return summary
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)