import os
import torch

from transformers import (Trainer, TrainingArguments,
                          DataCollatorForSeq2Seq,
                          AutoModelForSeq2SeqLM, AutoTokenizer)
from datasets import load_from_disk

from textSummarization.entity import ModelTrainerConfig
from textSummarization.logging import logger

class ModelTrainer:
    def __init__(self, config:ModelTrainerConfig) -> None:
        self.config = config
    
    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device using: {device}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_chkpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_chkpt).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        #Loading Data
        dataset = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir= self.config.root_dir,
            num_train_epochs= self.config.num_train_epochs,
            warmup_steps= self.config.warmup_steps,
            per_device_train_batch_size= self.config.per_device_train_batch_size,
            per_device_eval_batch_size= self.config.per_device_eval_batch_size,
            weight_decay= self.config.weight_decay,
            logging_steps= self.config.logging_steps,
            evaluation_strategy= self.config.evaluation_strategy,
            eval_steps= self.config.eval_steps,
            save_steps= self.config.save_steps,
            gradient_accumulation_steps= self.config.gradient_accumulation_steps
        )

        trainer = Trainer(
            model= model,
            args= trainer_args,
            tokenizer= tokenizer,
            data_collator= data_collator,
            train_dataset= dataset["train"],
            eval_dataset= dataset["validation"]
        )
        
        trainer.train()

        #Save Model
        # trainer.save_model(save_directory=os.path.join(self.config.root_dir,"pegasus_samsum_model"))
        model.save_pretrained(save_directory=os.path.join(self.config.root_dir,"pegasus_samsum_model"))

        #Save Tokenizer
        tokenizer.save_pretrained(save_directory=os.path.join(self.config.root_dir,"tokenizer"))
