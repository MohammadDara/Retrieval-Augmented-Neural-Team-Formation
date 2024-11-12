from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import Dataset
import torch
import datetime
import os

# Configurations
FOLD = 2
MAX_LEN = 256
TRAIN_EPOCHS = 1
BATCH_SIZE = 8
BASE_PATH = '/mnt/data/dara/teamformation_llm_V4/submit'
SAVE_PATH = f"{BASE_PATH}/outputs"

# Custom callback for saving checkpoints
class CustomCheckpointCallback(TrainerCallback):
    def __init__(self, save_every_n_epochs, output_dir):
        self.save_every_n_epochs = save_every_n_epochs
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        if int(state.epoch) % self.save_every_n_epochs == 0:
            checkpoint_dir = os.path.join(self.output_dir, f"fold_{FOLD}_checkpoint-epoch-{int(state.epoch)}")
            torch.save(model, checkpoint_dir)
            control.should_save = True

# Preprocess function for tokenization
def preprocess_function(examples):
    inputs = [ex['question'] for ex in examples['translation']]
    targets = [ex['answer'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LEN, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == '__main__':
    # Device setup
    print(torch.__version__)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # Load tokenizer and model
    tokenizer = torch.load(f"{BASE_PATH}/outputs/tokenizer.binary", map_location=device)
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Load training and testing datasets
    train_path = f"{BASE_PATH}/datasets/fold_{FOLD}_train.tsv"
    with open(train_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    data = [{'question': line.strip().split("\t")[0], 'answer': line.strip().split("\t")[1]} for line in lines]
    train_dataset = Dataset.from_dict({'translation': data})
    test_dataset = Dataset.from_dict({'translation': data})

    # Tokenize datasets
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        f"T5-finetuned-fold_{FOLD}",
        evaluation_strategy="no",
        do_eval=False,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=TRAIN_EPOCHS,
        predict_with_generate=True,
        fp16=True,
        save_steps=500000  # A large number to avoid auto-saving checkpoints
    )

    # Data collator and trainer setup
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Add custom callback
    trainer.add_callback(CustomCheckpointCallback(save_every_n_epochs=100, output_dir=f'{SAVE_PATH}/t5_trainer'))

    # Train the model
    trainer.train()
    torch.save(model, f'{SAVE_PATH}/fold_{FOLD}/teamformation_t5')
