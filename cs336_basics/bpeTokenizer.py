import os
from typing import BinaryIO
import argparse
import regex as re
from collections import Counter, defaultdict
import multiprocessing
from functools import partial
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train a tokenizer")

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """使用特殊标记分割文本，保留特殊标记作为独立片段"""
    if not special_tokens:
        return [text]
    
    # 转义特殊标记并构建正则模式 (按长度降序排序)
    pattern = "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    # 分割并保留分隔符
    segments = re.split(f"({pattern})", text)
    return [seg for seg in segments if seg]  # 过滤空字符串

def pretoken_each_chunk(chunk: str, special_tokens: list[str]) -> list[str]:
    """正确处理特殊标记：分割后分别预处理"""
    segments = split_on_special_tokens(chunk, special_tokens)
    PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = []
    
    for segment in segments:
        if segment in special_tokens:
            # 特殊标记直接作为独立token
            pretokens.append(segment)
        else:
            # 普通文本使用正则预处理
            pretokens.extend(re.findall(PATTERN, segment))
    
    return pretokens

def get_stats(bytes_with_freq: list[tuple[list[bytes], int]]) -> defaultdict:
    """获取所有字节对的频率统计"""
    pairs = defaultdict(int)
    for byte_list, freq in bytes_with_freq:
        for i in range(len(byte_list) - 1):
            pair = (byte_list[i], byte_list[i+1])
            pairs[pair] += freq
    return pairs

def update_bytes_with_freq(bytes_with_freq, best_pair):
    """更新字节序列，将最佳字节对合并"""
    new_bytes_with_freq = []
    merged_token = best_pair[0] + best_pair[1]  # 生成合并后的字节串
    
    for byte_list, freq in bytes_with_freq:
        new_byte_list = []
        i = 0
        while i < len(byte_list):
            # 检查当前位置是否匹配 best_pair
            if i + 1 < len(byte_list) and byte_list[i] == best_pair[0] and byte_list[i + 1] == best_pair[1]:
                new_byte_list.append(merged_token)
                i += 2
            else:
                new_byte_list.append(byte_list[i])
                i += 1
        new_bytes_with_freq.append((new_byte_list, freq))
    return new_bytes_with_freq

def token_to_byte_sequence(token: bytes) -> list[bytes]:
    """将字节序列转换为单字节列表"""
    return [bytes([b]) for b in token]

def train_bpe(
    input_path: str,
    special_tokens: list[str],
    num_chunks: int = 1,  # 重命名以准确反映用途
    vocab_size: int = 30000,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # 初始化词汇表，包含特殊标记
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    # 添加256个基础字节
    for x in range(256):
        vocab[len(special_tokens) + x] = bytes([x])

    # 确保使用文档边界特殊标记作为分块依据
    doc_boundary_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    # 读取并分块文件
    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, num_chunks, doc_boundary_token)
        
        # 获取分块数据
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

    # 并行预处理分块
    with multiprocessing.Pool(processes=num_chunks) as pool:
        # 创建带固定 special_tokens 的 partial 函数
        func = partial(pretoken_each_chunk, special_tokens=special_tokens)
        # 并行处理分块 + 进度条
        results = list(tqdm(pool.imap(func, chunks), total=len(chunks), desc="Pre-tokenizing"))
    
    # 合并结果
    pretokens = [token for sublist in results for token in sublist]
    pretokens_count = Counter(pretokens)

    # 将预处理结果转换为字节序列和频率
    bytes_with_freq = []
    for token, freq in pretokens_count.items():
        # 特殊标记保持原样
        if token in special_tokens:
            bytes_with_freq.append(([token.encode("utf-8")], freq))
        else:
            # 普通token转换为字节序列
            bytes_with_freq.append((token_to_byte_sequence(token.encode("utf-8")), freq))

    # 开始BPE训练
    start_index = len(special_tokens) + 256
    remaining_vocab_size = vocab_size - start_index
    new_token = []
    
    # 显示训练进度
    pbar = tqdm(total=remaining_vocab_size, desc="Training BPE")
    
    for i in range(remaining_vocab_size):
        # 获取当前字节对频率
        pairs_freq = get_stats(bytes_with_freq)
        if not pairs_freq:
            break
        
        # 找到最频繁的字节对
        best_pair = max(
            pairs_freq,
            key=lambda pair: (pairs_freq[pair], pair)
        )
        
        # 将新token添加到词汇表
        vocab[start_index + i] = best_pair[0] + best_pair[1]
        new_token.append(best_pair)

        # 更新字节序列
        bytes_with_freq = update_bytes_with_freq(bytes_with_freq, best_pair)
        
        pbar.update(1)
    
    pbar.close()
    return vocab, new_token

if __name__ == "__main__":
    # 示例用法
    vocab, merges = train_bpe(
        "/root/CS336Assignments/assignment1-basics-main/tests/fixtures/tinystories_sample_5M.txt", 
        ["<|endoftext|>"],  # 特殊标记列表
        4,  # 使用4个分块进行并行处理
        500
    )
    print(merges)