
from transformers import AdamW
from pooling_t5_encoder import CustomT5EncoderWithMean
import torch
from transformers import T5ForConditionalGeneration, RagTokenForGeneration, RagConfig, RagRetriever
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import faiss
from datasets import Dataset as ds2
from transformers.models.rag.retrieval_rag import CustomHFIndex
import torch.nn as nn
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

FOLD = 2
PATIENCE = 3
EPOCHS_WITHOUT_IMPROVEMENT = 0
NUM_EPOCHS = 100
RETRIEVE_SIZE = 10
# BASE_PATH = "../.."
BASE_PATH = ''
SAVE_PATH = f"{BASE_PATH}/outputs"
TRAIN_PATH = BASE_PATH + f"/datasets/fold_{FOLD}_train.tsv"
MAX_LEN = 256
BATCH_SIZE = 8
PRETRAINED_GENERATOR = 't5-base'
related_papers = {}
sum_loss=0
best_loss = float('inf')

torch.cuda.set_device(0)
dev = torch.device("cuda:0")

tokenizer = torch.load(f"{BASE_PATH}/outputs/tokenizer.binary", map_location=dev)
t5_model = T5ForConditionalGeneration.from_pretrained(PRETRAINED_GENERATOR).to(dev)
t5_model.resize_token_embeddings(len(tokenizer))
custom_question_encoder = CustomT5EncoderWithMean(t5_model.config)
custom_question_encoder.to(dev)

optimizer = AdamW(custom_question_encoder.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, custom_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.custom_encoder = custom_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx, 0]
        target_text = self.data.iloc[idx, 1]

        # Tokenize the input and target text
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        inputs = {k: v.squeeze().to(dev) for k, v in inputs.items()}
        targets = {k: v.squeeze().to(dev) for k, v in targets.items()}
        return {
            'inputs': inputs['input_ids'].squeeze(),
            'inputs_attention_mask': inputs['attention_mask'].squeeze(),
            'target_ids': targets['input_ids'].squeeze(),
            'target_attention_mask': targets['attention_mask'].squeeze(),
            'text': input_text
        }

def encode_text_to_embeddings(text):
    inputs = tokenizer(text['title'], truncation=True, padding='max_length', max_length=max_len, return_tensors='pt').to(dev)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move inputs to GPU
    outputs = custom_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
    
    embeddings = outputs.last_hidden_state.detach().to(dev)
    return {'embeddings': embeddings}

def convert_to_retriever_format(examples):
    return {
        'title': examples['input_text'],
        'text': examples['target_text']
    }

def calculate_repeated_times(row):
    s = {}
    for author_id in row['target_text'].split():
        for paper in author_id_to_papers[author_id]:
            if paper in s:
                s[paper] += 1
            else:
                s[paper] = 1

    sorted_values = [key for key, value in sorted(s.items(), key=lambda item: item[1], reverse=True)]
    if len(sorted_values) < RETRIEVE_SIZE:
        sorted_values.extend([''] * (RETRIEVE_SIZE - len(sorted_values)))  # Pad with zeros if less than retrieve_size
    else:
        sorted_values = sorted_values[:RETRIEVE_SIZE]  # Truncate if more than retrieve_size
    related_papers[row['input_text']] = sorted_values
    return 

max_len = 256
batch_size = 8
train_data = pd.read_csv(TRAIN_PATH, sep='\t', header=None, names=['input_text', 'target_text'])
train_data = train_data.assign(id=range(1, len(train_data) + 1))

# Preprocess author-paper mappings
paper_to_author_ids = {row['input_text']: row['target_text'].split() for _, row in train_data.iterrows()}

author_id_to_papers = {}
for _, row in train_data.iterrows():
    for author_id in row['target_text'].split():
        if author_id not in author_id_to_papers:
            author_id_to_papers[author_id] = []
        author_id_to_papers[author_id].append(row['input_text'])

train_data.apply(calculate_repeated_times,axis=1)

train_dataset = TextDataset(train_data, tokenizer, max_len, custom_question_encoder)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

contrastive_loss = nn.CosineEmbeddingLoss()

custom_question_encoder.eval()

for epoch in range(NUM_EPOCHS):
    print(datetime.now())

    retriever_dataset = ds2.from_pandas(train_data)
    retriever_dataset = retriever_dataset.map(convert_to_retriever_format, remove_columns=['input_text', 'target_text'])

    custom_encoder = custom_question_encoder
    retriever_dataset = retriever_dataset.map(encode_text_to_embeddings, batched=True, batch_size=10)
    title_to_embeddings = {row['title']: torch.tensor(row['embeddings']).to(dev) for row in retriever_dataset}

    rag_config = RagConfig(
        question_encoder=custom_question_encoder.config.to_dict(),
        generator=t5_model.config.to_dict(),
    )
    embeddings_matrix = np.array(retriever_dataset['embeddings'])
    dimension = embeddings_matrix.shape[1]

    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))
    retriever_dataset.add_faiss_index("embeddings", custom_index=gpu_index)

    rag_config.retrieval_vector_size = max_len
    rag_config.index_name = "custom"
    index = CustomHFIndex(rag_config.retrieval_vector_size, retriever_dataset)
    rag_retriever = RagRetriever(config=rag_config, question_encoder_tokenizer=tokenizer, generator_tokenizer=tokenizer, index=index)
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        loss = torch.tensor([0.0], requires_grad=True).to(dev)

        embeddings = custom_encoder(input_ids=batch['inputs'], attention_mask=batch['inputs_attention_mask'], return_dict=True)
        embeddings_np = embeddings.last_hidden_state.detach().cpu().numpy()

        batch_size_current = embeddings_np.shape[0]
        for i in range(batch_size_current):
            embedding_np = embeddings_np[i].reshape(1, -1)

            retrieved_doc_embeds, doc_ids, retrieved_doc_dicts = rag_retriever.retrieve(embedding_np, n_docs=RETRIEVE_SIZE)
            retrieved_doc_dicts = retrieved_doc_dicts[0]
            retrieved_doc_embeds = retrieved_doc_embeds[0]

            predicted_doc_dict_list = [{} for _ in range(RETRIEVE_SIZE)]
            for key in retrieved_doc_dicts:
                for j in range(RETRIEVE_SIZE):
                    predicted_doc_dict_list[j][key] = retrieved_doc_dicts[key][j]
            for j in range(RETRIEVE_SIZE):
                predicted_doc_dict_list[j]['encoded_authors'] = tokenizer.encode(predicted_doc_dict_list[j]['text'])

            reference_authors_set = set(batch['target_ids'][i].tolist())
            cur_paper_title = batch['text'][i]

            neg=[]
            while(len(neg) < RETRIEVE_SIZE):
                rnd = np.random.randint(0, retriever_dataset.num_rows)
                emb_rnd = title_to_embeddings[retriever_dataset['title'][rnd]]
                authors_rnd = tokenizer.encode(retriever_dataset['text'][rnd])
                authors_rnd.remove(1)
                if len(set(authors_rnd).intersection(reference_authors_set)) == 0:
                    neg.append(emb_rnd)

            pos=[]
            for j in range(RETRIEVE_SIZE):
                ground_truth_item = related_papers[cur_paper_title][j]
                if ground_truth_item == '':
                    continue
                pos.append(title_to_embeddings[ground_truth_item].unsqueeze(0))
            anchor_embedding = embeddings.last_hidden_state[i].unsqueeze(0)

            # Calculate contrastive loss using positive samples
            total_pos_loss = torch.tensor([0.0], requires_grad=True).to(dev)
            for positive_embedding in pos:
                cur_loss = contrastive_loss(anchor_embedding, positive_embedding, torch.tensor([1.0]).to(dev))
                loss += cur_loss
                total_pos_loss += cur_loss

            # Calculate contrastive loss using negative samples
            total_neg_loss = torch.tensor([0.0], requires_grad=True).to(dev)
            for negative_embedding in neg:
                cur_loss = contrastive_loss(anchor_embedding, negative_embedding.unsqueeze(0), torch.tensor([-1.0]).to(dev))
                loss += cur_loss
                total_neg_loss += cur_loss

            print(f"pos loss: {total_pos_loss} - neg loss: {total_neg_loss}")

        loss /= batch_size * RETRIEVE_SIZE * 2
        print(loss)
        sum_loss+=loss.item()

        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {sum_loss}")
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current Learning Rate: {current_lr}")

    if sum_loss < best_loss:
        best_loss = sum_loss
        epochs_without_improvement = 0  # Reset counter if improvement
    else:
        epochs_without_improvement += 1  # Increment if no improvement
    sum_loss=0
        
    if epochs_without_improvement >= PATIENCE:
        epochs_without_improvement = 0
        torch.save(custom_question_encoder, f'{SAVE_PATH}/fold_{FOLD}/es_custom_question_encoder_{epoch}')
        print(f"saved early after {epoch+1} epochs with best loss {best_loss}.")

torch.save(custom_question_encoder, f'{SAVE_PATH}/fold_{FOLD}/custom_question_encoder')
