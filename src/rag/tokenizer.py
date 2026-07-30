"""Chemical tokenization boundary for model sizing and BM25 indexing."""

from __future__ import annotations

from collections import OrderedDict
import re

import jieba

from .embeddings import TEIEmbeddings

DEFAULT_TOKEN_CACHE_SIZE = 1024
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

        cached = self._model_token_cache.get(text)
        if cached is not None:
            self._model_token_cache.move_to_end(text)
            return cached

        count = self._embeddings.count_tokens(text)
        self._model_token_cache[text] = count
        self._model_token_cache.move_to_end(text)
        if len(self._model_token_cache) > self._cache_size:
            self._model_token_cache.popitem(last=False)
        return count

    def bm25_terms(self, text: str) -> list[str]:
        """Return normalized, punctuation-free jieba terms for FTS indexing."""

        terms: list[str] = []
        for token in _JIEBA_TOKENIZER.cut(text):
            normalized = _PUNCTUATION_OR_WHITESPACE.sub("", token)
            if normalized:
                terms.append(normalized)
        return terms
