from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "sonoisa/t5-base-japanese"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files="data.jsonl")

def preprocess(example):
	inputs = tokenizer(example["input"], truncation=True, padding="max_length", max_length=64)
	targets = tokenizer(example["output"], truncation=True, padding="max_length", max_length=64)
	inputs["labels"] = targets["input_ids"]
	return inputs

dataset = dataset.map(preprocess)

training_args = TrainingArguments(
	output_dir="./result",
	num_train_epochs=5,
	per_device_train_batch_size=4,
	save_steps=10,
	save_total_limit=2,
	logging_steps=10,
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=dataset["train"],
)

trainer.train()

model.save_pretrained("./kotoha_model")
tokenizer.save_pretrained("./kotoha_model")
