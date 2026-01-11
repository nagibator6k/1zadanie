from transformers import GPTNeoForCausalLM, GPT2Tokenizer


model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.config.pad_token_id = model.config.eos_token_id
input_text = "Сколько будет 2+2?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(
    input_ids, 
    max_length=100,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_p=0.95,
    temperature=0.7
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

