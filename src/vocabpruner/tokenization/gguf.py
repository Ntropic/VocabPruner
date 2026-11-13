from __future__ import annotations

from pathlib import Path

from vocabpruner.storage.gguf import gguf_field, read_gguf


class GGUFTokenizer:
    def __init__(self, model: str | Path):
        reader = read_gguf(model)
        self.tokens = [str(token) for token in gguf_field(reader, "tokenizer.ggml.tokens", [])]
        if not self.tokens:
            raise ValueError("GGUF has no tokenizer.ggml.tokens metadata.")
        token_types = gguf_field(reader, "tokenizer.ggml.token_type", [1] * len(self.tokens))
        self.byte_token_ids = [index for index, kind in enumerate(token_types) if int(kind) == 6]
        self.all_special_ids = [index for index, kind in enumerate(token_types) if int(kind) in {2, 3, 4}]
        self.all_special_tokens = [self.tokens[index] for index in self.all_special_ids]
        self.special_tokens_map = {}
        self.model = str(gguf_field(reader, "tokenizer.ggml.model", ""))
        self.merges = list(gguf_field(reader, "tokenizer.ggml.merges", []))

    def get_vocab(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def decode(self, token_ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        tokens = [self.tokens[int(token_id)] for token_id in token_ids]
        if self.model == "gpt2":
            from tokenizers.decoders import ByteLevel

            return ByteLevel().decode(tokens)
        text = "".join(tokens).replace("▁", " ")
        if len(tokens) == 1 and tokens[0].startswith("<0x") and tokens[0].endswith(">"):
            try:
                return bytes([int(tokens[0][3:-1], 16)]).decode("latin-1")
            except ValueError:
                pass
        return text
