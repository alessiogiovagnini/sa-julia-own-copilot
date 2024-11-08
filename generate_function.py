import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model and tokenizer
model_name = './results/checkpoint-3/'  # Path to your trained model directory
tokenizer_model = 'HuggingFaceTB/SmolLM-135M'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate code based on input prompt
def generate_code(input_prompt, max_length=100, num_return_sequences=1):
    # Tokenize the input prompt
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    
    # Generate function code
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,  # Helps to reduce repetition
            early_stopping=True
        )
    
    # Decode the generated code
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code

# Test the model with a sample input
sample_input = '"""Compute the circle radius"""' + "\n" + 'function compute_radius(r)'
generated_function = generate_code(sample_input)

print("Generated Function Output:")
print(generated_function)
