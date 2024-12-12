import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import evaluate
from datasets import Dataset
import torch
from datetime import datetime

def train_model(model_name='HuggingFaceTB/SmolLM-135M', docstring_included=True, dataset_path = './datasets/135MtrainingDoc'):
    print('Training model') 
    
    # HuggingFaceTB/SmolLM-1.7B
    # HuggingFaceTB/SmolLM-360M
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # torch.cuda.set_device(1)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name_stripped = model_name.split('/')[1]
    date = datetime.now().strftime("%d_%m_%Y")
    output_dir_pedix = '_doc/' if docstring_included else '/'
    output_directory = './results/' + model_name_stripped + '_' + date + output_dir_pedix
    
    # Prepare tokenizer and model for Julia
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Load datasets
    trainingSet = Dataset.load_from_disk(dataset_path)
    # trainingSet = trainingSet.select(range(2))
    print('datasets loaded')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_directory,
        logging_dir='./logs',
        save_strategy='epoch',  # Save at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=None,  # Keep all checkpoints
        evaluation_strategy='no',  # No evaluation
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainingSet,
    )

    # Train the model
    trainer.train()
    print('FINISHED')
    
if __name__ == '__main__':
    # link = sys.argv[1]
    # filename = sys.argv[2]
    # train_model('HuggingFaceTB/SmolLM-135M',True,'./datasets/135MDocTrainSet')
    
    # print('###############')
    # print('FIRST COMPLETED')
    # print('###############')
    
    # train_model('HuggingFaceTB/SmolLM-135M',False, dataset_path = './datasets/135MTrainSet')
    
    
    # train_model('HuggingFaceTB/SmolLM-360M',True, dataset_path = './datasets/360MDocTrainSet')
    
    # train_model('HuggingFaceTB/SmolLM-360M',False, dataset_path = './datasets/360MTrainSet')
    

    
    train_model('HuggingFaceTB/SmolLM-1.7B',True, dataset_path = './datasets/1.7BDocTrainSet')
    
    print('###############')
    print('FIRST COMPLETED')
    print('###############')
    
    train_model('HuggingFaceTB/SmolLM-1.7B',False, dataset_path = './datasets/1.7BTrainSet')
    
    
