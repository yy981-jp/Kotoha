from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "sonoisa/t5-base-japanese"

# tokenizer（警告対策）
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset("json", data_files="data.jsonl")

def preprocess(example):
	inputs = tokenizer(
		example["input"],
		truncation=True,
		padding="max_length",
		max_length=32
	)

	targets = tokenizer(
		example["output"],
		truncation=True,
		padding="max_length",
		max_length=32
	)

	# 👇 これ超重要（paddingを無視させる）
	labels = targets["input_ids"]
	labels = [
		[(token if token != tokenizer.pad_token_id else -100) for token in seq]
		for seq in [labels]
	][0]

	inputs["labels"] = labels
	return inputs

dataset = dataset["train"].map(preprocess)

training_args = TrainingArguments(
	output_dir="./result",
	num_train_epochs=10,              # ←増やす
	per_device_train_batch_size=4,
	save_steps=50,
	save_total_limit=2,
	logging_steps=10,
	learning_rate=3e-5,              # ←ちょい安定化
)

trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./kotoha_model")
tokenizer.save_pretrained("./kotoha_model")
