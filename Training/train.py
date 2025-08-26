from cs336_basics.utils.data_loader import get_batch
import numpy as np

parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
parser.add_argument("--train_input", type=str, default = "./output/dataset/TinyStories_tokens_train.npy", 
                    help="Path to the training data")
parser.add_argument("--valid_input", type=str, default = "./output/dataset/TinyStories_tokens_train.npy", 
                    help="Path to the training data")


if __name__ == "__main__":
    args = parser.parse_args()

    # load data
    dataset = np.load(args.train_input, mmap_mode='r')

    batch_size = 32
    num_epochs = 1
    
    for epoch in range(num_epochs):
        data = get_batch(dataset)