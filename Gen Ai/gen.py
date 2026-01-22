import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# FLAN-T5
device = torch.device("cpu")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small",
                                                    torch_dtype="auto")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

decoded_inputs = "How do you make chocolate chip cookies?"
encoded_inputs = tokenizer(decoded_inputs, return_tensors="pt").to(device)
encoded_outputs = model.generate(encoded_inputs["input_ids"], max_new_tokens=50)[0]
decoded_outputs = tokenizer.decode(encoded_outputs, skip_special_tokens=True)
print(decoded_outputs)
