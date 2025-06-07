from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load once at module level
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

def summarize_text_t5(text: str) -> str:
    input_text = f"summarize the following text in points: {text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
