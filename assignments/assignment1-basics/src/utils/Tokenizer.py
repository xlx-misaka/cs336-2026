"""作业 tokenization 测试使用的字节级 BPE 辅助实现。

该 tokenizer 故意基于基础数据结构实现。``pre_tokenize`` 使用 GPT-2 风格正则
拆分文本，同时保护特殊 token；``train_bpe`` 从字节级词频表学习 merge 规则；
``BpeTokenizer`` 在编码阶段应用这些规则。
"""

import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import BinaryIO

import regex as re

_GPT2_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO,
    split_special_tokens: bytes | list[bytes],
    desired_num_chunks: int = 8,
) -> list[int]:
    """在不拆散特殊 token 的前提下寻找文件分块字节偏移。

    参数
    --------
    file:
        当前指针位置任意的已打开二进制文件。函数会临时移动指针，
        结束后停在最后一次检查的位置。
    split_special_tokens:
        一个字节串或字节串序列。分块边界会落在这些模式首次出现的位置。
        若该序列为空，返回 ``[0, file_size]``。
    desired_num_chunks:
        目标分块数的上限。模式稀疏或重叠时可能返回更少且唯一的边界。

    返回
    --------
    list[int]
        从 ``0`` 开始、到文件大小结束的严格递增字节偏移列表。
    """
    desired_num_chunks = max(int(desired_num_chunks), 1)

    if isinstance(split_special_tokens, bytes):
        split_special_tokens = [split_special_tokens]
    tokens = list(split_special_tokens)
    assert all(isinstance(token, bytes) for token in tokens), "特殊 token 必须表示为字节串"

    # 获取文件总字节数
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if not tokens:
        # 没有特殊分隔符时，无法在保证多字节 UTF-8 序列完整的前提下做无模式
        # 分块，因此使用单个块是正确且安全的做法。
        return [0, file_size]

    chunk_size = file_size // desired_num_chunks

    # 初始的均匀分块边界估计，块从上一边界开始且不包含最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向后读 4KB

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从估计边界开始
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小块

            # 到达 EOF 时，该边界应当放在文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 寻找小块内最早出现的分隔特殊 token
            found_at = -1
            for token in tokens:
                match = mini_chunk.find(token)
                if match != -1 and (found_at == -1 or match < found_at):
                    found_at = match
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 去重，因此边界数量可能少于目标分块数
    return sorted(set(chunk_boundaries))


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> list[str]:
    """将文本拆成 GPT-2 风格 pretoken，同时保持特殊 token 完整。

    参数
    --------
    text:
        待拆分的字符串。
    special_tokens:
        可选的特殊 token 字符串列表。若这些 token 出现在 ``text`` 中，
        会作为单个 pretoken 输出。

    返回
    --------
    list[str]
        有序的 pretoken 列表。
    """
    special_token_set = set(special_tokens) if special_tokens else set()

    if special_tokens:
        special_regex = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        segments = re.split(f"({special_regex})", text)
    else:
        segments = [text]

    words: list[str] = []

    for segment in segments:
        if special_tokens and segment in special_token_set:
            words.append(segment)
        else:
            segment = _GPT2_PATTERN.findall(segment)
            for word in segment:
                words.append(word)

    return words


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从文本文件训练字节级 BPE 词表与 merge 顺序。

    参数
    --------
    input_path:
        UTF-8 文本语料路径。
    vocab_size:
        最终词表大小，包含 256 个单字节条目与所有特殊 token。
        若可用 pair 频率先耗尽，学出的词表可能略小于请求大小。
    special_tokens:
        不可再拆分的保留 token。训练期间不参与 merge，最后追加到词表。
    **kwargs:
        为兼容调用方保留，当前被忽略。

    返回
    --------
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
        ``(vocab, merges)``。``vocab`` 是 token ID 到字节串的映射；
        ``merges`` 是按创建顺序排列的字节对 merge 列表。
    """
    if special_tokens is None:
        special_tokens = []

    special_token_set = set(special_tokens)
    word_freq: Counter[tuple[bytes, ...]] = Counter()

    with open(input_path, mode="rb") as f:
        split_tokens = [token.encode("utf-8") for token in special_tokens]
        boundaries = find_chunk_boundaries(f, split_tokens)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode(encoding="utf-8", errors="ignore")
            words = pre_tokenize(chunk, special_tokens)
            for word in words:
                if word in special_token_set:
                    continue
                encoded_word = word.encode("utf-8")
                word_freq[tuple(bytes([b]) for b in encoded_word)] += 1

    # 记录每个word的频率
    words_list: list[list[bytes]] = []
    freqs_list: list[int] = []

    for word, freq in word_freq.items():
        words_list.append(list(word))
        freqs_list.append(freq)

    # pair_freq 记录相邻字节对的累计频率，indices 记录每个字节对出现的词索引
    pair_freq = defaultdict(int)
    indices = defaultdict(set)

    for idx, word in enumerate(words_list):
        freq = freqs_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freq[pair] += freq
            indices[pair].add(idx)

    # 初始化字节词表，并计算需要学习的 merge 数量
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    num_merge = vocab_size - 256 - len(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # 迭代选择频率最高的字节对并合并
    for _ in range(num_merge):
        if not pair_freq:
            break
        # 同频时取字典序更大的 pair
        best_pair = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token
        relevant_indices = indices[best_pair]

        for idx in relevant_indices:
            word = words_list[idx]
            freq = freqs_list[idx]
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == best_pair:
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        pair_freq[prev_pair] -= freq
                        if pair_freq[prev_pair] == 0:
                            del pair_freq[prev_pair]
                            del indices[prev_pair]

                    if i < len(word) - 2:
                        nxt_pair = (word[i + 1], word[i + 2])
                        pair_freq[nxt_pair] -= freq
                        if pair_freq[nxt_pair] == 0:
                            del pair_freq[nxt_pair]
                            del indices[nxt_pair]

                    word[i] = new_token
                    del word[i + 1]

                    if i > 0:
                        new_prev_pair = (word[i - 1], new_token)
                        pair_freq[new_prev_pair] += freq
                        indices[new_prev_pair].add(idx)

                    if i < len(word) - 1:
                        new_nxt_pair = (new_token, word[i + 1])
                        pair_freq[new_nxt_pair] += freq
                        indices[new_nxt_pair].add(idx)
                i += 1
            words_list[idx] = word

        if best_pair in pair_freq:
            del pair_freq[best_pair]
        if best_pair in indices:
            del indices[best_pair]

    # 将特殊 token 追加到词表末尾
    for s_tok in special_tokens:
        s_bytes = s_tok.encode("utf-8")
        vocab[len(vocab)] = s_bytes

    return vocab, merges


def serialize_bpe_data(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    out_path: str,
) -> None:
    """将词表与 merge 规则写入 JSON 文件，供后续加载。

    参数
    --------
    vocab:
        从整数 token ID 到字节串的映射。
    merges:
        需要保存的有序字节对 merge 列表。
    out_path:
        目标 JSON 路径。字节内容以十六进制字符串保存。

    返回
    --------
    None
    """
    data = {
        "vocab": {token.hex(): token_id for token_id, token in vocab.items()},
        "merges": [[a.hex(), b.hex()] for a, b in merges],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_bpe_data(input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """读取由 :func:`serialize_bpe_data` 生成的 BPE 数据文件。

    参数
    --------
    input_path:
        JSON 数据文件路径。

    返回
    --------
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
        恢复得到的词表与 merge 列表。
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    vocab = {token_id: bytes.fromhex(token) for token, token_id in data["vocab"].items()}
    merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in data["merges"]]
    return vocab, merges


class BpeTokenizer:
    """应用已学习的字节级 BPE 词表与 merge 优先级。

    属性
    --------
    vocab:
        从整数 token ID 到字节串的映射。
    id:
        从字节串到整数 token ID 的逆映射。
    merges:
        按创建顺序排列的字节对 merge 列表。
    merge_rank:
        从字节对到其创建顺序编号的映射，编号越小优先级越高。
    special_tokens:
        不参与 BPE 合并的特殊 token 字符串列表，可能为 ``None``。
    _special_token_set:
        用于 O(1) 成员判断的特殊 token 集合。
    """

    vocab: dict[int, bytes]
    id: dict[bytes, int]
    merges: list[tuple[bytes, bytes]]
    merge_rank: dict[tuple[bytes, bytes], int]
    special_tokens: list[str] | None
    _special_token_set: set[str]

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        """由词表和 merge 列表初始化查找表。

        参数
        --------
        vocab:
            从整数 token ID 到字节串的映射；假设字节串与 ID 一一对应。
        merges:
            按创建顺序排列的字节对 merge；越靠前优先级越高。
        special_tokens:
            可选特殊 token 字符串，编码时直接输出为单 token，不参与 BPE 合并。
        """
        self.vocab = vocab
        self.id = {b: i for i, b in vocab.items()}
        self.merges = merges
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        self._special_token_set = set(special_tokens) if special_tokens is not None else set()

    @classmethod
    def from_files(cls, input_path: str, special_tokens: list[str] | None = None) -> "BpeTokenizer":
        """从序列化生成的 JSON 数据文件创建 tokenizer。

        参数
        --------
        input_path:
            BPE 数据文件路径。
        special_tokens:
            可选的特殊 token 字符串列表。

        返回
        --------
        BpeTokenizer
            新建的 tokenizer 实例。
        """
        vocab, merges = load_bpe_data(input_path)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """将字符串编码为词表 token ID 列表。

        参数
        --------
        text:
            待编码文本。

        返回
        --------
        list[int]
            输入文本对应的 token ID 列表。空字符串返回空列表。
        """
        words = pre_tokenize(text, self.special_tokens)
        ids: list[int] = []

        for word in words:
            if word in self._special_token_set:
                ids.append(self.id[word.encode("utf-8")])
                continue

            word = [bytes([b]) for b in word.encode("utf-8")]

            while len(word) > 1:
                best_rank = len(self.merges)
                best_index = -1
                for i in range(len(word) - 1):
                    rank = self.merge_rank.get((word[i], word[i + 1]))
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_index = i
                        if rank == 0:
                            break
                if best_index == -1:
                    break
                word[best_index] = word[best_index] + word[best_index + 1]
                del word[best_index + 1]

            for b in word:
                ids.append(self.id[b])

        return ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 序列解码回文本。

        参数
        --------
        ids:
            待解码的 token ID 列表。

        返回
        --------
        str
            解码后的文本。无效 UTF-8 字节序列会被替换为 Unicode 替换字符。
        """
        data = b"".join(self.vocab[id] for id in ids)
        return data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """惰性编码一段文本迭代器。

        参数
        --------
        iterable:
            逐项产出字符串的迭代器，例如文本文件对象。

        返回
        --------
        collections.abc.Iterable[int]
            token ID 生成器。编码过程是流式的，不会把整份输入完整读入内存。
        """
        for text in iterable:
            yield from self.encode(text)
