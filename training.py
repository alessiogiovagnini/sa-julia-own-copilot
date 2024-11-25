import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import numpy as np
import evaluate

# Define compute metrics function
def compute_metrics(eval_pred):
    
    # Load metric function
    accuracy_metric = evaluate.load('accuracy')
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Flatten the arrays and remove -100 from labels and corresponding predictions
    flattened_labels = labels.flatten()
    flattened_predictions = predictions.flatten()

    # Mask out padding values (-100) in the labels
    mask = flattened_labels != -100
    filtered_labels = flattened_labels[mask]
    filtered_predictions = flattened_predictions[mask]
    
    # Compute accuracy
    accuracy = accuracy_metric.compute(predictions=filtered_predictions, references=filtered_labels)
    return {'accuracy': accuracy['accuracy']}


def makeDataset(input_text, target_text, tokenizer, tokens_per_batch=512):
    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_length = tokens_per_batch  # Define a reasonable max_length for inputs and targets

    # Tokenize the input and target texts
    input_encodings = tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length)
    target_encodings = tokenizer(target_text, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length)
    
    # Ensure labels are aligned with input
    target_ids = target_encodings['input_ids']
    target_ids[target_ids == tokenizer.pad_token_id] = -100  # Ignore padding tokens in the loss computation

    return Dataset.from_dict({
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_ids
    })

def train_model(model_name = 'HuggingFaceTB/SmolLM-135M', docstring_included = True, tokens_per_batch = 512):
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    output_directory = './results/' + model_name

    trainingSetInputs = []
    trainingSetTargets = []
    
    evaluationSetInputs = []
    evalutionSetTargets = []
    
    testSetInputs = []
    testSetTargets = []
    

    # load the datasets
    if docstring_included:
        
        output_directory = output_directory + '_doc/'
        
        traindf = pd.read_csv('train_set_doc.csv')
        evaldf = pd.read_csv('eval_set_doc.csv')
        testdf = pd.read_csv('test_set_doc.csv')

        # Combine docstring and function signature as input
        traindf['input_text'] = traindf['docstring'] + "\n" + traindf['function_signature']
        traindf['target_text'] = traindf['function_body']
        
        evaldf['input_text'] = evaldf['docstring'] + "\n" + evaldf['function_signature']
        evaldf['target_text'] = evaldf['function_body']
        
        testdf['input_text'] = testdf['docstring'] + "\n" + testdf['function_signature']
        testdf['target_text'] = testdf['function_body']
        
        # # Split into training, evaluation, and test sets
        trainingSetInputs = traindf['input_text'].tolist()
        trainingSetTargets = traindf['target_text'].tolist()
        
        evaluationSetInputs = evaldf['input_text'].tolist()
        evalutionSetTargets = evaldf['target_text'].tolist()
        
        testSetInputs = testdf['input_text'].tolist()
        testSetTargets = testdf['target_text'].tolist()
        
    else:
        traindf = pd.read_csv('train_set.csv')
        evaldf = pd.read_csv('eval_set.csv')
        testdf = pd.read_csv('test_set.csv')
        
        output_directory = output_directory + '/'
        
        trainingSetInputs = traindf['function_signature'].tolist()
        trainingSetTargets = traindf['function_body'].tolist()
        
        evaluationSetInputs = evaldf['function_signature'].tolist()
        evalutionSetTargets = evaldf['function_body'].tolist()
        
        testSetInputs = testdf['function_signature'].tolist()
        testSetTargets = testdf['function_body'].tolist()
        
    print('data loaded')

    # Prepare tokenizer and model for Julia
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize the dataset
    trainingSet = makeDataset(trainingSetInputs, trainingSetTargets, tokenizer)
    evaluationSet = makeDataset(evaluationSetInputs, evalutionSetTargets, tokenizer)
    testSet = makeDataset(testSetInputs, testSetTargets, tokenizer)
    
    print('datasets created')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_directory,
        eval_strategy='epoch',  # Evaluate at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='epoch',  # Save at the end of each epoch
        save_total_limit=1,  # Keep only the most recent model
        load_best_model_at_end=True,  # Automatically load the best model at the end
        metric_for_best_model='eval_loss',  # Use evaluation loss to select the best model
        greater_is_better=False,  # Lower loss is better
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainingSet,
        eval_dataset=trainingSet,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # Early stopping with patience
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    print('Evaluating model on TEST SET')
    trainer.evaluate(testSet)

    print('FINISHED')
    
    
if __name__ == '__main__':
    # link = sys.argv[1]
    # filename = sys.argv[2]
    print('Training model') 
    train_model()
