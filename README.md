# Retrieval-Augmented-Neural-Team-Formation

This repository contains the implementation of the paper:  
**"Retrieval-Augmented Neural Team Formation"**  

This work introduces a Retrieval-Augmented Generation (RAG) model that integrates historical collaboration data with required skill sets, enabling the selection of expert teams with both the necessary competencies and a proven track record of effective teamwork.

---

## Features
- **Encoder**: Custom T5-based encoder optimized with contrastive learning to generate semantic embeddings for skill sets.
- **Retriever**: Efficient FAISS-based retriever to extract relevant historical team formations.
- **Generator**: Transformer-based generator tailored to propose expert teams with cohesive collaboration dynamics.
- **Datasets**: Includes DBLP and Dota2 datasets for team formation experiments.
- **Evaluation**: Implements metrics such as Recall, MAP, and NDCG.

---

## Repository Structure
```
rag-team-formation/
├── datasets/               # Sample datasets for experiments
├── src/                    # Source code
│   ├── encoder/           # Encoder training and modules
│   ├── generator/         # Generator training and modules
│   ├── model/             # RAG inference and utilities
│   ├── tokenizer/         # Tokenizer-related scripts
│   └── utils/              # Helper functions
├── figures/                # Model diagrams and result visualizations
├── LICENSE                 # License information
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/?/Retrieval-Augmented-Neural-Team-Formation.git
   cd Retrieval-Augmented-Neural-Team-Formation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Train Encoder**
Train the encoder with contrastive learning:
```bash
python src/encoder/train_encoder.py
```

### 2. **Pretrain Generator**
Pretrain the generator module:
```bash
python src/generator/pretrain_generator.py
```

### 3. **Train Generator in RAG Setup**
Fine-tune the generator with RAG:
```bash
python src/generator/train_generator.py
```

### 4. **Run Inference**
Generate expert teams using the trained RAG model:
```bash
python src/model/rag.py
```

---

## Citation
If you use this code in your research, please cite the paper:
```
@inproceedings{RAG_Teamformation,
  title={Retrieval-Augmented Neural Team Formation},
  author={},
  booktitle={},
  year={2025}
}
```
