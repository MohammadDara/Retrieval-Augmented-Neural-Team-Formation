from transformers import AutoTokenizer
import torch

BASE_ADDRESS = ''

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
    # Add custom tokens
    token_file_path = BASE_ADDRESS + "/datasets/experts.txt"
    with open(token_file_path, "r", encoding="utf-8") as token_file:
        for line in token_file:
            tokenizer.add_tokens(line.strip())
    
    # Save the tokenizer
    torch.save(tokenizer, f'{BASE_ADDRESS}/outputs/tokenizer.binary')
