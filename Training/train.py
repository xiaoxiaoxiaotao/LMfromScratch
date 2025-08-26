import numpy as np
import argparse
import torch

from cs336_basics.utils.data_loader import get_batch
from cs336_basics.Module.transformer import TransformerLM
from cs336_basics.optim.Optimizer import AdamW
from cs336_basics.utils.loss import cross_entropy
from cs336_basics.utils.learning_rate_scheme import cosine_annealing_learning_rate
from cs336_basics.utils.checkpoint import save_checkpoint

parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
parser.add_argument("--train_input", type=str, default = "/home/mw/input/tstoken66386638/TinyStories_tokens_train.npy", 
                    help="Path to the training data")
parser.add_argument("--valid_input", type=str, default = "/home/mw/input/tstoken66386638/TinyStories_tokens_valid.npy", 
                    help="Path to the training data")
parser.add_argument("--checkpoint_path", type=str, default = "/home/mw/", 
                    help="Path to the training data")

@torch.no_grad()
def estimate_loss(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    contex_len: int,
    eval_iters: int,
    device: str
):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, batch_size, block_size, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1), y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    args = parser.parse_args()

    # load data
    train_data = np.load(args.train_input, mmap_mode='r')
    valid_data = np.load(args.valid_input, mmap_mode='r')

    batch_size = 32
    num_epochs = 1
    vocab_size = 10000
    context_len = 128
    d_model = 128
    num_heads = 8
    d_ff = 4 * d_model
    num_layers = 6
    rope_theta = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 200      # 验证时跑多少个 batch 取平均

    steps_per_epoch = len(train_data) // (batch_size * context_len)
    
    # initialize the model
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_len,
        num_layers=num_layers,
        theta=rope_theta,
        device=device,
    )

    # initialize the optimizer
    optimizer = AdamW(model.parameters())

    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        num_batches = 0
        for step in range(steps_per_epoch):
            global_step += 1
            num_batches += 1

            x, label = get_batch(train_data, batch_size, context_len, device)
            logits = model(x)
            loss = cross_entropy(logits, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss

            if global_step % eval_interval == 0:
                losses = estimate_loss(
                    model, train_data, valid_data, batch_size, contex_len, eval_iters, device
                )
                print(f"Step {global_step}: train_loss = {losses['train']:.4f}, val_loss = {losses['val']:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # save checkpoint
        save_checkpoint(model, optimizer, global_step, out)

    print("Training completed.")