import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os

load_dotenv()
# 1. Configuration
model_id = "google/medgemma-1.5-4b-it"
hf_token = os.getenv("HF_TOKEN") 

print(f"Loading {model_id}...")

# 2. Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    torch_dtype=torch.bfloat16, # Use bfloat16 for speed/memory efficiency
    device_map="auto"           # Automatically uses GPU if available
)

# 3. Create a Generation Pipeline
med_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

# 4. Test it
prompt = "TECHNICAL DIAGNOSIS: Comminuted fracture of the femur. \nSIMPLIFIED EXPLANATION:"
output = med_pipeline(prompt)

print("\n--- MEDGEMMA OUTPUT ---")
print(output[0]['generated_text'])