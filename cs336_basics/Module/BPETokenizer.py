from typing import Dict, List, Tuple, Optional, Iterable, Iterator
import regex as re

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        """
        从给定的词汇表、合并列表和特殊标记构造分词器
        
        Args:
            vocab: 词汇表，映射ID到字节序列
            merges: 合并规则列表
            special_tokens: 特殊标记列表（可选）
        """
        # 保存原始参数
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # 构建 token_to_id 映射
        self.token_to_id = {}
        for idx, token_bytes in vocab.items():
            self.token_to_id[token_bytes] = idx

        for idx, token_str in enumerate(self.special_tokens):
            token_bytes = token_str.encode("utf-8")
            # 如果特殊标记不在词汇表中，append在后面
            if token_bytes not in self.token_to_id:
                self.token_to_id[token_bytes] = len(self.token_to_id)

        # 构建 id_to_token 映射 (用于解码)
        self.id_to_token = {idx: token_bytes for idx, token_bytes in vocab.items()}
        # 构建合并规则的快速查找字典
        # 注意：merges 列表的顺序定义了合并优先级，位置越靠前优先级越高
        self.merge_dict = {}
        for i, (byte1, byte2) in enumerate(merges):
            self.merge_dict[(byte1, byte2)] = i
        
        # 预编译正则表达式
        self.PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 构建特殊标记的正则模式
        if self.special_tokens:
            # 按长度降序排序，确保长标记先匹配
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
            self.special_token_pattern = re.compile(f"({pattern})")
        else:
            self.special_token_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[List[str]] = None) -> "BPETokenizer":
        """
        从文件构造分词器
        
        Args:
            vocab_filepath: 词汇表文件路径
            merges_filepath: 合并规则文件路径
            special_tokens: 特殊标记列表（可选）
            
        Returns:
            BPETokenizer实例
        """
        # 加载词汇表
        with open(vocab_filepath, "r") as f:
            vocab_dict = json.load(f)
        
        # 转换为 {id: token_bytes} 格式
        # 注意：vocab.json 中的键是通过 latin1 编码的字节序列
        vocab = {}
        for token_str, idx_str in vocab_dict.items():
            idx = int(idx_str)
            # 使用latin1解码确保字节值不变
            token_bytes = token_str.encode('latin1')
            vocab[idx] = token_bytes
        
        # 加载合并规则
        merges = []
        with open(merges_filepath, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                token1, token2 = line.strip().split()
                # 将字符串转换回字节 (通过latin1编码)
                byte1 = token1.encode('latin1')
                byte2 = token2.encode('latin1')
                merges.append((byte1, byte2))
        
        return cls(vocab, merges, special_tokens)

    def _split_on_special_tokens(self, text: str) -> List[str]:
        """使用特殊标记分割文本，保留特殊标记作为独立片段"""
        if not self.special_tokens or not self.special_token_pattern:
            return [text]
        
        segments = self.special_token_pattern.split(text)
        return [seg for seg in segments if seg]  # 过滤空字符串

    def _pretokenize(self, text: str) -> List[str]:
        """将文本分割为预处理token"""
        segments = self._split_on_special_tokens(text)
        pretokens = []
        
        for segment in segments:
            if segment in self.special_tokens:
                # 特殊标记直接作为独立token
                pretokens.append(segment)
            else:
                # 普通文本使用正则预处理
                tokens = self.PATTERN.findall(segment)
                pretokens.extend(tokens)
        
        return pretokens

    def _token_to_byte_sequence(self, token: str) -> List[bytes]:
        """将token字符串转换为单字节列表"""
        return [bytes([b]) for b in token.encode("utf-8")]

    def _apply_bpe(self, token: str) -> List[bytes]:
        """对单个token应用BPE算法"""        
        # 将token转换为字节序列
        byte_list = self._token_to_byte_sequence(token)
        
        # 如果是单字节，直接返回
        if len(byte_list) == 1:
            return byte_list
        
        # 应用BPE合并规则
        changed = True
        while changed and len(byte_list) > 1:
            changed = False
            best_pair = None
            best_idx = -1
            
            # 找到可合并的最高优先级对
            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i+1])
                if pair in self.merge_dict:
                    if best_pair is None or self.merge_dict[pair] < self.merge_dict[best_pair]:
                        best_pair = pair
                        best_idx = i
            
            # 执行合并
            if best_pair is not None:
                merged = best_pair[0] + best_pair[1]
                byte_list[best_idx:best_idx+2] = [merged]
                changed = True
        
        return byte_list
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token ID序列
        
        Args:
            text: 输入文本
            
        Returns:
            token ID列表
        """
        # 1. 预处理文本
        pretokens = self._pretokenize(text)
        
        # 2. 对每个预处理token应用BPE
        token_ids = []
        for token in pretokens:
        # 如果是特殊标记，直接返回其字节表示
            if token in self.special_tokens:
                # 特殊标记：直接查找其字节表示对应的ID
                token_bytes = token.encode("utf-8")
                if token_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[token_bytes])
            else:
                # 普通token应用BPE
                byte_tokens = self._apply_bpe(token)
                for byte_token in byte_tokens:
                    # 查找token ID
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])
                    else:
                        # 如果找不到，尝试拆分为单字节
                        for b in byte_token:
                            single_byte = bytes([b])
                            if single_byte in self.token_to_id:
                                token_ids.append(self.token_to_id[single_byte])
        
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        给定字符串可迭代对象，返回一个懒惰生成token ID的生成器
        
        Args:
            iterable: 字符串可迭代对象（如文件句柄）
            
        Returns:
            生成token ID的迭代器
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        将token ID序列解码为文本
        
        Args:
            ids: token ID列表
            
        Returns:
            解码后的文本
        """
        # 1. 将ID转换为token字节
        byte_tokens = []
        for token_id in ids:
            if token_id in self.id_to_token:
                byte_tokens.append(self.id_to_token[token_id])
        
        # 2. 将字节序列合并为字符串
        try:
            # 尝试直接解码
            return b"".join(byte_tokens).decode("utf-8")
        except UnicodeDecodeError:
            # 处理可能的编码问题
            return "".join([t.decode("utf-8", errors="replace") for t in byte_tokens])