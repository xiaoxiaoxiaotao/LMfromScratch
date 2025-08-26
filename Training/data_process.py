from cs336_basics.Module.BPETokenizer import BPETokenizer
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Encode the data with BPE tokenizer")
parser.add_argument("--input", type=str, required=True, 
                    help="Path to the input text file")
parser.add_argument("--vocab_filepath", type=str, default = "./output/tokenizer/vocab.json", 
                    help="Path to the vocab file")
parser.add_argument("--merges_filepath", type=str,default = "./output/tokenizer/merges.txt", 
                    help="Path to the merges file")
parser.add_argument("--output_dir", type=str, default = './output/dataset/TinyStories_train_tokens.npy', 
                    help="Path to the save file")


args = parser.parse_args()

tokenizer = BPETokenizer.from_files(
    vocab_filepath = args.vocab_filepath,
    merges_filepath = args.merges_filepath,
    special_tokens = ["<|endoftext|>"],
    )

with open(args.input, "r", encoding="utf-8") as file:
    token_ids = list(tokenizer.encode_iterable(file))  # 每行是一个 chunk
    data = np.array(token_ids, dtype='uint16')
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(args.output_dir, data)
