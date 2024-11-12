import torch
from transformers import RagTokenForGeneration, T5ForConditionalGeneration, AutoTokenizer, RagConfig, RagRetriever
from datasets import Dataset
import pandas as pd
import faiss
import numpy as np
from transformers.models.rag.retrieval_rag import CustomHFIndex
import sys
from sklearn.metrics import ndcg_score, average_precision_score, dcg_score
import csv
import sys


BASE_PATH = '/mnt/data/dara/teamformation_llm_V4/submit'
SAVE_PATH = f"{BASE_PATH}/outputs"
sys.path.append(f'{BASE_PATH}/src/encoder')

# Check if an argument was passed
if len(sys.argv) == 2:
    at_k = int(sys.argv[1])
else:
    at_k = 10

FOLD = 2
recalls = []
maps = []
ndcgs = []

# Ensure that you are using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your fine-tuned T5 generator model
pretrained_generator = f'{SAVE_PATH}/fold_{FOLD}/t5_Full'
t5_model = torch.load(f"{pretrained_generator}", map_location=device)

encoder = torch.load(f"{SAVE_PATH}/fold_{FOLD}/custom_question_encoder", map_location=device)

encoder.eval()
t5_model.eval()

tokenizer = torch.load(f"{SAVE_PATH}/tokenizer.binary", map_location=device)

# Load training and testing datasets
train_path = BASE_PATH + f"/datasets/fold_{FOLD}_train.tsv"
test_path = BASE_PATH + f"/datasets/fold_{FOLD}_test.tsv"

train_data = pd.read_csv(train_path, sep='\t', header=None, names=['input_text', 'target_text'])
test_data = pd.read_csv(test_path, sep='\t', header=None, names=['input_text', 'target_text'])

# Prepare the retriever dataset with 'input_text' as title and 'target_text' as text
retriever_dataset = Dataset.from_dict({
    'title': train_data['input_text'].tolist(),
    'text': train_data['target_text'].tolist()
})

# Generate embeddings for retriever dataset using your custom encoder
def encode_texts(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        embeddings = encoder(**inputs).last_hidden_state.cpu().numpy()
    return embeddings

def encode_text_to_embeddings(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(device)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move inputs to GPU
    outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], return_dict=True)
    embeddings = outputs.last_hidden_state.detach().to(device)
    return {'embeddings': embeddings}

rag_config = RagConfig(
    question_encoder=encoder.config.to_dict(),
    generator=t5_model.config.to_dict(),
)

retriever_dataset = retriever_dataset.map(encode_text_to_embeddings, batched=True, batch_size=10)
embeddings_matrix = np.array(retriever_dataset['embeddings'])
dimension = embeddings_matrix.shape[1]
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimension))

# retriever_dataset.add_faiss_index("embeddings", custom_index=gpu_index)
retriever_dataset.add_faiss_index("embeddings", custom_index=gpu_index)

rag_config.retrieval_vector_size = 512
rag_config.index_name = "custom"
index = CustomHFIndex(rag_config.retrieval_vector_size, retriever_dataset)
rag_retriever = RagRetriever(config=rag_config, question_encoder_tokenizer=tokenizer, generator_tokenizer=tokenizer, index=index)

def retrieve_docs(question_hidden_states, n_docs=5):
    # Ensure question_hidden_states is on CPU and in the correct shape
    if isinstance(question_hidden_states, torch.Tensor):
        question_hidden_states = question_hidden_states.cpu().numpy()
    if len(question_hidden_states.shape) == 3:
        question_hidden_states = question_hidden_states.squeeze(0)
    
    # Perform the search
    scores, indices = gpu_index.search(question_hidden_states, min(n_docs, gpu_index.ntotal))
    
    # Ensure indices are within bounds
    valid_indices = indices[indices < len(retriever_dataset)]
    
    # Retrieve documents
    docs = [retriever_dataset[int(idx)] for idx in valid_indices]
    
    return docs, scores

rag_model = RagTokenForGeneration(
    retriever=rag_retriever,
    generator=t5_model,
    question_encoder=encoder
)
rag_model.eval()
# Run inference on the test dataset
test_samples = test_data['input_text'].tolist()
encoded_inputs = tokenizer(test_samples, return_tensors="pt", padding=True, truncation=True).to(device)

def zero_extend_or_truncate(tensor, target_length):
    current_length = tensor.size(0)

    if current_length > target_length:
        # Truncate the tensor to target_length
        return tensor[:target_length]
    elif current_length < target_length:
        # Zero-extend the tensor to target_length
        padding = torch.zeros(target_length - current_length, *tensor.shape[1:], dtype=tensor.dtype)
        return torch.cat([tensor, padding], dim=0)
    else:
        return tensor
def ap_at_k(ground_truth, predictions, k, total_relevant):
    top_k_predictions = predictions[:k]
    score = 0.0
    num_relevant = 0
    for i, p in enumerate(top_k_predictions):
        if p in ground_truth:
            num_relevant += 1
        precision_at_i = num_relevant / (i + 1)
        score += precision_at_i
    return score / len(top_k_predictions)
    
def process_input(input_text, target_text, rag_model, tokenizer, n_docs=5):
    # Encode input
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Get question hidden states
    with torch.no_grad():
        question_hidden_states = rag_model.question_encoder(**encoded_input).last_hidden_state

    # Retrieve documents
    retrieved_docs, scores = retrieve_docs(
        question_hidden_states=question_hidden_states,
        n_docs=n_docs
    )
    # Generate
    output = rag_model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        min_length=10
    )

    predicted_tokens_with_duplicates = output[0][output[0] > 32000]

    seen = set()
    predicted_tokens_no_duplicates = []
    for tensor_item in predicted_tokens_with_duplicates:
        item = tensor_item.item()
        if item not in seen:
            predicted_tokens_no_duplicates.append(item)
            seen.add(item)

    target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
    dict_ans = {token: 1 for token in target_tokens if token > 32000}

    top_k_predictions = predicted_tokens_no_duplicates[:at_k]
    num_predictions = len(top_k_predictions)
    if num_predictions < at_k:
        padding_size = at_k - num_predictions
        top_k_predictions.extend([0] * padding_size)
    else:
        padding_size = 0  # No padding needed

    result_output_name = f"prediction_dblp_{FOLD}.txt"
    if at_k == 10:
        with open(result_output_name, 'a+') as file:
            writer = csv.writer(file)
            writer.writerow(["RAG-Dota2", 10, f"{FOLD}", len(top_k_predictions), len(target_tokens),
                            0] + top_k_predictions + target_tokens)

    relevance_labels = [1 if token in dict_ans else 0 for token in top_k_predictions]
    predicted_scores = np.arange(at_k, 0, -1)  # From at_k down to 1

    relevance_labels = np.array(relevance_labels)
    predicted_scores = np.array(predicted_scores)
    y_true = relevance_labels.reshape(1, -1)
    y_score = predicted_scores.reshape(1, -1)

    if at_k == 1:
        # If at_k is 1, make sure that the inputs are compatible with the expected format
        dcg = y_true[0][0]
        ideal_dcg = 1 if len(dict_ans) > 0 else 0
    else:
        dcg = dcg_score(y_true, y_score, k=at_k)
        ideal_relevance_score = np.array([[1] * min(len(dict_ans), at_k) + [0] * (at_k - min(len(dict_ans), at_k))])
        ideal_dcg = dcg_score(ideal_relevance_score, y_score, k=at_k)

    ndcg = dcg/ideal_dcg
    ndcgs.append(ndcg)

    ap = ap_at_k(dict_ans, top_k_predictions, at_k, sum(relevance_labels))
    maps.append(ap)
    
    # Compute Recall@k
    num_relevant_retrieved = np.sum(relevance_labels)
    num_relevant_total = len(dict_ans)  # Total number of relevant items
    if num_relevant_total > 0:
        recall_k = num_relevant_retrieved / num_relevant_total
    else:
        recall_k = 0.0
    recalls.append(recall_k)
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text, retrieved_docs, scores

for idx, (input_text, target_text) in enumerate(zip(test_data['input_text'], test_data['target_text'])):
    generated_text, retrieved_docs, scores = process_input(input_text, target_text, rag_model, tokenizer)

mean_recall = np.mean(recalls)
mean_map = np.mean(maps)
mean_ndcg = np.mean(ndcgs)

# Print the final metrics
print("\nFinal Metrics:")
print(f"Mean Recall@{at_k}: {mean_recall:.4f}")
print(f"Mean MAP: {mean_map:.4f}")
print(f"Mean NDCG: {mean_ndcg:.4f}")