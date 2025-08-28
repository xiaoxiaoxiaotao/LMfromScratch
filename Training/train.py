import numpy as np
import argparse
import torch

# 导入你自定义的模块
from cs336_basics.utils.data_loader import get_batch
from cs336_basics.Module.transformer import TransformerLM
from cs336_basics.optim.Optimizer import AdamW  # 你自定义的 AdamW
from cs336_basics.utils.loss import cross_entropy
from cs336_basics.utils.learning_rate_scheme import cosine_annealing_learning_rate
from cs336_basics.utils.checkpoint import save_checkpoint, load_checkpoint  # 假设已导入
from cs336_basics.utils.gradient_clipping import gradient_clipping


parser = argparse.ArgumentParser(description="Train a Transformer language model")
parser.add_argument("--train_input", type=str, default="/home/mw/input/tstoken66386638/TinyStories_tokens_train.npy",
                    help="Path to the training data (.npy)")
parser.add_argument("--valid_input", type=str, default="/home/mw/input/tstoken66386638/TinyStories_tokens_valid.npy",
                    help="Path to the validation data (.npy)")
parser.add_argument("--checkpoint_path", type=str, default="/home/mw/",
                    help="Directory to save model checkpoints")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume training from (optional)")


@torch.no_grad()
def estimate_loss(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    context_len: int,
    eval_iters: int,
    device: str
):
    """
    评估模型在训练集和验证集上的平均损失
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, context_len, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parser.parse_args()

    # Load data
    print("Loading training and validation data...")
    train_data = np.load(args.train_input, mmap_mode='r')
    valid_data = np.load(args.valid_input, mmap_mode='r')
    print(f"Train data size: {len(train_data)} tokens")
    print(f"Valid data size: {len(valid_data)} tokens")

    # Hyperparameters
    batch_size = 32
    num_epochs = 3
    vocab_size = 10000
    context_len = 128
    d_model = 128
    num_heads = 8
    d_ff = 4 * d_model
    num_layers = 6
    rope_theta = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters = 20
    eval_interval = 100

    # 计算每轮的步数
    steps_per_epoch = len(train_data) // (batch_size * context_len)
    total_steps = num_epochs * steps_per_epoch

    # 学习率调度参数
    alpha_max = 3e-4          # 峰值学习率
    alpha_min = 1e-5          # 最小学习率
    T_w = max(100, int(0.1 * total_steps))  # warm-up 步数（至少 100）
    T_c = total_steps         # 退火结束步数

    # 初始化模型
    print("Initializing model...")
    model = TransformerLM(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_len,
        num_layers=num_layers,
        theta=rope_theta,
        device=device,
    ).to(device)

    # 初始化优化器
    optimizer = AdamW(model.parameters(), lr=alpha_max)  # 初始值会被调度覆盖

    # 恢复训练（可选）
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        global_step = load_checkpoint(args.resume, model, optimizer)
        start_epoch = global_step // steps_per_epoch
        print(f"Resumed from step {global_step}, starting from epoch {start_epoch + 1}")

    model.train()

    print(f"Starting training for {num_epochs} epochs, total steps: {total_steps}")
    print(f"Warm-up: {T_w} steps, Peak LR: {alpha_max}, Min LR: {alpha_min}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        epoch_loss = 0.0
        num_batches = 0

        for step in range(steps_per_epoch):
            global_step += 1
            num_batches += 1

            # 清除梯度
            optimizer.zero_grad()

            # 获取训练 batch
            x, label = get_batch(train_data, batch_size, context_len, device)

            # 前向传播
            logits = model(x)
            loss = cross_entropy(logits, label)

            # 反向传播
            loss.backward()

            # 梯度裁剪（使用你写的函数）
            gradient_clipping(model.parameters(), M=1.0)

            # 动态学习率：余弦退火 + warm-up
            lr = cosine_annealing_learning_rate(
                t=global_step,
                alpha_max=alpha_max,
                alpha_min=alpha_min,
                T_w=T_w,
                T_c=T_c
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 更新参数
            optimizer.step()

            epoch_loss += loss.item()

            # 定期评估
            if global_step % eval_interval == 0:
                losses = estimate_loss(
                    model, train_data, valid_data, batch_size, context_len, eval_iters, device
                )
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {global_step}: "
                      f"train_loss = {losses['train']:.4f}, "
                      f"val_loss = {losses['val']:.4f}, "
                      f"lr = {current_lr:.2e}")

        # 每轮结束后打印平均 loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # 保存 checkpoint
        ckpt_path = f"{args.checkpoint_path}/model_step_{global_step}.pt"
        save_checkpoint(model, optimizer, global_step, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("\n✅ Training completed.")


if __name__ == "__main__":
    main()
