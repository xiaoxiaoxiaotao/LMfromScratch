import os
from typing import BinaryIO
import argparse
import regex as re
from collections import Counter, defaultdict

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

def remove_special_tokens(text: str, special_tokens: list[str]) -> str:
    if not special_tokens:
        return text
    pattern = "|".join(map(re.escape, special_tokens))
    return re.sub(pattern, "", text)

def pretoken_each_chunk(
    chunk,
    special_tokens
):
    chunk = remove_special_tokens(chunk, special_tokens)
    PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(PATTERN, chunk)


def train_bpe(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 1,
    vocab_size: int = 30000,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = {}
    # add special tokens
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    # initialize the original 256 byte values
    for x in range(256):
        vocab[len(special_tokens) + x] = bytes([x])

    with open(input_path, "rb") as file:
        boundaries = find_chunk_boundaries(
            file, num_processes, "<|endoftext|>".encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.

        # get pretokens
        pretokens = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            file.seek(start)
            chunk = file.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            pretokens.extend(pretoken_each_chunk(chunk,special_tokens))

        #count pretokens
        pretokens_count = Counter(pretokens)

        # encode to bytes, and split by byte
        # bytes_with_freq is a list[tuple[list[bytes], int]]
        bytes_with_freq = [(token_to_byte_sequence(token.encode("utf-8")), freq) 
                            for (token, freq) in pretokens_count.items()]

        # merging
        vocab, newtoken = merging_bytes(bytes_with_freq, len(special_tokens), vocab, vocab_size)

        return vocab, newtoken


def update_bytes_with_freq(bytes_with_freq, best_pair):
    new_bytes_with_freq = []

    for byte_list, freq in bytes_with_freq:
        i = 0
        new_byte_list = []
        while i < len(byte_list):
            # 检查当前位置是否有 best_pair
            if i + 1 < len(byte_list) and (byte_list[i], byte_list[i + 1]) == best_pair:
                # 合并成一个新 token
                new_byte_list.append(b''.join(best_pair))
                i += 2
            else:
                new_byte_list.append(byte_list[i])
                i += 1
        new_bytes_with_freq.append((new_byte_list, freq))

    return new_bytes_with_freq


def get_stats(bytes_with_freq: list[tuple[list[bytes], int]]):
    pairs = defaultdict(int)
    for list_bytes, count in bytes_with_freq:
        for i in range(len(list_bytes) - 1):
            pair = (list_bytes[i], list_bytes[i+1])
            pairs[pair] += count
    return pairs

def merging_bytes(bytes_with_freq, len_special_tokens, vocab, vocab_size):
    start_index = len_special_tokens + 256
    remaining_vocab_size = vocab_size - start_index
    print(remaining_vocab_size)
    new_token = []
    
    for i in range(remaining_vocab_size):
        # get best pairs
        pairs_freq = get_stats(bytes_with_freq)
        if not pairs_freq:
            break  
        best_pair = max(
            pairs_freq,
            key=lambda pair: (pairs_freq[pair], pair)
        )
        # add new token to vocab
        vocab[start_index + i] = b''.join(best_pair)
        new_token.append(best_pair)

        # update frequence
        bytes_with_freq = update_bytes_with_freq(bytes_with_freq, best_pair)
    return vocab, new_token

def token_to_byte_sequence(token: bytes) -> list[bytes]:
    return [bytes([b]) for b in token]

if __name__ == "__main__":
    '''
    test_text = text = "Hello Hello world! 你好，世界。I'm learning BPE with 100 examples. Isn't it fun? 😊"

    pretokens = pretoken_each_chunk(test_text)

    pretokens_count = Counter(pretokens)

    bytes_with_freq = [(token_to_byte_sequence(token.encode("utf-8")), freq) for (token, freq) in pretokens_count.items()]

    pairs = get_stats(bytes_with_freq)

    best_pair = max(
        pairs,
        key=lambda pair: (pairs[pair], pair)
    )
    vocab = {}

    # merging
    vocab, newtoken = merging_bytes(bytes_with_freq, 0, vocab, 270)
    print(vocab)
    print()
    print(newtoken)
    '''
    vocab, merges = train_bpe("/root/CS336Assignments/assignment1-basics-main/tests/fixtures/tinystories_sample_5M.txt", 
                            ["<|endoftext|>"],1, 500)
    print(merges)


