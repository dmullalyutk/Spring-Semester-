from transformers import T5ForConditionalGeneration, T5Tokenizer, TextStreamer, pipeline



model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", torch_dtype="auto")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

streamer = TextStreamer(tokenizer)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    num_beams=1,
    do_sample=False,
    device = "cpu"
)

decoded_inputs = "How do you make chocolate chip cookies?"
outputs = generator(decoded_inputs)
decoded_outputs = outputs[0]['generated_text']


beam = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    num_beams=5,
    num_return_sequences=3,
    do_sample=False,
    early_stopping=True,
    device="cpu"
)

decoded_inputs = "How do you make chocolate chip cookies?"
outputs = beam(decoded_inputs)

for i, output in enumerate(outputs):
    print(f"Beam {i}: {output['generated_text']}")
