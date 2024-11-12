import torch
from transformers import T5ForConditionalGeneration, AdamW, AutoTokenizer, RagTokenForGeneration, RagConfig, RagRetriever
from datasets import Dataset
import pandas as pd
import numpy as np
import faiss
from transformers.models.rag.retrieval_rag import CustomHFIndex
from torch.optim.lr_scheduler import StepLR
from transformers import DataCollatorForSeq2Seq
import sys



PATIENCE = 3
FOLD = 2
BASE_PATH = '/mnt/data/dara/teamformation_llm_V4/submit'
SAVE_PATH = f"{BASE_PATH}/outputs"
sys.path.append(f'{BASE_PATH}/src/encoder')

sum_loss = 0
best_loss = float('inf')
epochs_without_improvement = 0

# Ensure that you are using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = torch.load(f"{SAVE_PATH}/tokenizer.binary", map_location=device)

# Load the T5 generator model
pretrained_generator = f'{SAVE_PATH}/fold_{FOLD}/teamformation_t5'
t5_model = torch.load(pretrained_generator, map_location=device)
t5_model.resize_token_embeddings(len(tokenizer))


# Load your custom encoder
encoder = torch.load(f"{SAVE_PATH}/fold_{FOLD}/custom_question_encoder", map_location=device)
encoder.eval()

# Freeze the encoder to ensure it doesn't get updated during training
for param in encoder.parameters():
    param.requires_grad = False

# Load training dataset
# base_path = "./"
# train_path = base_path + f"fold_{fold}_train_eligible_v8.tsv"
train_path = "/mnt/data/dara/teamformation_llm/" + f"train-test/fold_{FOLD}_train_eligible_v8.tsv"

train_data = pd.read_csv(train_path, sep='\t', header=None, names=['input_text', 'target_text'])

# Prepare the retriever dataset with 'input_text' as title and 'target_text' as text
retriever_dataset = Dataset.from_dict({
    'title': train_data['input_text'].tolist(),
    'text': train_data['target_text'].tolist()
})

# FAISS Index creation and embedding with custom encoder
def encode_text_to_embeddings(examples):
    inputs = tokenizer(examples['title'], truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(device)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move inputs to GPU
    outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
    embeddings = outputs.last_hidden_state.detach().to(device)
    return {'embeddings': embeddings}

retriever_dataset = retriever_dataset.map(encode_text_to_embeddings, batched=True, batch_size=10)
embeddings_matrix = np.array(retriever_dataset['embeddings'])
dimension = embeddings_matrix.shape[1]
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))

# Add embeddings to FAISS index
retriever_dataset.add_faiss_index("embeddings", custom_index=gpu_index)

# Define custom retriever using FAISS index
rag_config = RagConfig(
    question_encoder=encoder.config.to_dict(),
    generator=t5_model.config.to_dict(),
)
rag_config.retrieval_vector_size = dimension
rag_config.index_name = "custom"

index = CustomHFIndex(rag_config.retrieval_vector_size, retriever_dataset)
rag_retriever = RagRetriever(config=rag_config, question_encoder_tokenizer=tokenizer, generator_tokenizer=tokenizer, index=index)

# Now we create the RAG model with a custom encoder and retriever, and the T5 generator
rag_model = RagTokenForGeneration(
    retriever=rag_retriever,
    generator=t5_model,
    question_encoder=encoder
).to(device)

for param in rag_model.question_encoder.parameters():
    param.requires_grad = False
rag_model.question_encoder.eval()

# Tokenization function for training data
def encode_texts(examples):
    inputs = tokenizer(
        examples['input_text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )
    targets = tokenizer(
        examples['target_text'],
        padding='max_length',
        truncation=True,
        max_length=256
    )
    examples['input_ids'] = inputs['input_ids']
    examples['attention_mask'] = inputs['attention_mask']
    examples['labels'] = targets['input_ids']
    return examples

# Apply the tokenization function to the dataset
train_dataset = Dataset.from_dict({
    'input_text': train_data['input_text'].tolist(),
    'target_text': train_data['target_text'].tolist()
})
# train_dataset = train_dataset.map(encode_texts, batched=True, batch_size=8)
train_dataset = train_dataset.map(
    encode_texts,
    batched=True,
    remove_columns=['input_text', 'target_text']
)

# Set up data loaders
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=rag_model,
    padding='longest',  # You can also use 'max_length' or 'do_not_pad' depending on your needs
    return_tensors='pt'
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

# Optimizer for the generator only
optimizer = AdamW(rag_model.generator.parameters(), lr=5e-4)
scheduler = StepLR(optimizer, step_size=40, gamma=0.5)


# Training loop
num_epochs = 500
t5_model.train()

# Training loop to fine-tune the generator based on retriever output
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = rag_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            min_length=10
        )

        loss = outputs.loss

        generated_sequences = rag_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=11
        )

        loss2 = []
        batch_size = labels.size(0)
        for sample_index in range(batch_size):
            dict_ans = {token.item(): 1 for token in labels[sample_index] if token > 32000}
            top_k_predictions = generated_sequences[sample_index][:10]

            seen = set()
            predicted_tokens_no_duplicates = []
            for tensor_item in top_k_predictions:
                item = tensor_item.item()
                if item not in seen:
                    predicted_tokens_no_duplicates.append(item)
                    seen.add(item)

            relevance_labels = [1 if token in dict_ans else 0 for token in predicted_tokens_no_duplicates]
            num_relevant_retrieved = np.sum(relevance_labels)
            if num_relevant_retrieved > 0:
                x=2
            num_relevant_total = len(dict_ans)  # Total number of relevant items
            if num_relevant_total > 0:
                recall_k = num_relevant_retrieved / num_relevant_total
            else:
                recall_k = 0.0
            loss2.append(1 - recall_k)
        tensor_loss2 = torch.tensor(loss2, device='cuda:0', requires_grad=True)

        loss += tensor_loss2

        print(loss)
        sum_loss+=loss.sum().item()

        loss.mean().backward()
        optimizer.step()

        epoch_loss += loss.sum().item()
    
    print(f"Loss after epoch {epoch+1}: {epoch_loss / len(train_dataloader)}")
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current Learning Rate: {current_lr}")
    print(f"sum_loss: {sum_loss}")

    if sum_loss < best_loss:
        best_loss = sum_loss
        epochs_without_improvement = 0  # Reset counter if improvement
    else:
        epochs_without_improvement += 1  # Increment if no improvement
        
    if epochs_without_improvement >= PATIENCE:
        epochs_without_improvement = 0
        torch.save(t5_model, f'{SAVE_PATH}/fold_{FOLD}/t5_{epoch}')
        print(f"saved early after {epoch+1} epochs with best loss {best_loss}")

    sum_loss=0

torch.save(t5_model, f'{SAVE_PATH}/fold_{FOLD}/t5_Full')
