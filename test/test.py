from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "./kotoha_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = input("入力: ")
# text = "あめがふっている"
# text = "あおぞらがひろがる"

prompt = text

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
	inputs["input_ids"],
	max_length=100,
	num_beams=8,
	no_repeat_ngram_size=2,
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("出力:", result)
