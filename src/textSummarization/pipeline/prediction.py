from transformers import AutoTokenizer, pipeline

from textSummarization.config.configuration import ConfigurationManager
from textSummarization.logging import logger

class PredictionPipeline:
    def __init__(self) -> None:
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty":0.8, 
                      "num_beams":8,
                      "max_length":128}
        
        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

        print("Dialog:")
        print(f"{text}")
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel summary:")
        print(f"{output}")

        return output