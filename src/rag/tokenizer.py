"""Chemical tokenization boundary for model sizing and BM25 indexing."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import re

import jieba

from .embeddings import TEIEmbeddings

DEFAULT_TOKEN_CACHE_SIZE = 1024
TOKEN_COUNT_BATCH_SIZE = 128
CHEMICAL_TERMS = (
    "MPa",
    "kPa",
    "mol/L",
    "GB/T",
    "CAS",
    "Ziegler-Natta",
)
_PUNCTUATION_OR_WHITESPACE = re.compile(r"[\W_]+", flags=re.UNICODE)
_JIEBA_TOKENIZER = jieba.Tokenizer()
_JIEBA_TOKENIZER.initialize()
for _term in CHEMICAL_TERMS:
    _JIEBA_TOKENIZER.add_word(_term)


class ChemicalTokenizer:
    """Keep TEI model token sizing separate from chemical BM25 tokenization."""

    def __init__(
        self,
        embeddings: TEIEmbeddings,
        cache_size: int = DEFAULT_TOKEN_CACHE_SIZE,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be a positive integer.")
        self._embeddings = embeddings
        self._cache_size = cache_size
        self._model_token_cache: OrderedDict[str, int] = OrderedDict()

    def model_tokens(self, text: str) -> int:
        """Return a cached model-accurate token count from TEI."""

        return self.model_token_counts([text])[0]

    def model_token_counts(self, texts: Sequence[str]) -> list[int]:
        """Resolve uncached model counts in bounded TEI batches."""

        requested = list(texts)
        if not requested:
            return []
        resolved_by_text = {
            text: self._model_token_cache[text]
            for text in dict.fromkeys(requested)
            if text in self._model_token_cache
        }
        missing = list(
            dict.fromkeys(
                text for text in requested if text not in self._model_token_cache
            )
        )
        for start in range(0, len(missing), TOKEN_COUNT_BATCH_SIZE):
            batch = missing[start : start + TOKEN_COUNT_BATCH_SIZE]
            counts = self._embeddings.count_tokens_batch(batch)
            if len(counts) != len(batch):
                raise ValueError("TEI returned an incomplete token-count batch.")
            for text, count in zip(batch, counts):
                resolved_by_text[text] = count
                self._remember_model_count(text, count)
        resolved: list[int] = []
        for text in requested:
            count = resolved_by_text[text]
            if text in self._model_token_cache:
                self._model_token_cache.move_to_end(text)
            resolved.append(count)
        return resolved

    def _remember_model_count(self, text: str, count: int) -> None:
        if count < 0:
            raise ValueError("Model token counts cannot be negative.")
        self._model_token_cache[text] = count
        self._model_token_cache.move_to_end(text)
        if len(self._model_token_cache) > self._cache_size:
            self._model_token_cache.popitem(last=False)

    def bm25_terms(self, text: str) -> list[str]:
        """Return normalized, punctuation-free jieba terms for FTS indexing."""

        terms: list[str] = []
        for token in _JIEBA_TOKENIZER.cut(text):
            normalized = _PUNCTUATION_OR_WHITESPACE.sub("", token)
            if normalized:
                terms.append(normalized)
        return terms
