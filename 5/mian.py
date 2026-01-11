from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Загрузка модели и токенизатора
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Устанавливаем pad_token_id (если он не установлен)
model.config.pad_token_id = model.config.eos_token_id

# Вводим текст
input_text = "Сколько будет 2+2?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Генерация текста
output = model.generate(
    input_ids, 
    max_length=100,    # Ограничиваем длину генерируемого текста
    num_return_sequences=1,  # Число сгенерированных последовательностей
    no_repeat_ngram_size=2,  # Запрещаем повторение биграмм
    top_p=0.95,        # Для контроля случайности
    temperature=0.7    # Для контроля разнообразия
)

# Декодируем результат
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

