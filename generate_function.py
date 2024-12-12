import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model and tokenizer
print('Loading started')
# model_name = './results/checkpoint-3'  # Path to your trained model directory
model_name = './results/SmolLM-135M_4_12_24_doc/checkpoint-100002'  # Path to your trained model directory
tokenizer_model = 'HuggingFaceTB/SmolLM-135M'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Loading completed')


# Function to generate code based on input prompt
def generate_code(input_prompt, max_length=1024, num_return_sequences=1):
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
        )
    
    # Decode the generated code
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_code

# Test the model with a sample input
sample_input = """\"\"\"
sf_airy_Ai_deriv(x, mode) -> Cdouble

C signature:
`double gsl_sf_airy_Ai_deriv(const double x, gsl_mode_t mode)`

GSL documentation:

### `double gsl_sf_airy_Ai_deriv (double x, gsl_mode_t mode)`

> int gsl\\_sf\\_airy\\_Ai\\_deriv\\_e (double x, gsl\\_mode\\_t mode,
> gsl\\_sf\\_result \\* result)

> These routines compute the Airy function derivative $Ai'(x)$ with an
> accuracy specified by `mode`.
\"\"\"""" + '\n' + 'function sf_airy_Ai_deriv(x, mode)'

print('Generating function...')
generated_function = generate_code(sample_input)

print("\nGenerated Function:\n")
print(generated_function)
