from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache
import torch


class DynamicCacheWithQuery(DynamicCache):
    '''
    Cache class used for In-context RAG
    
    Extends DynamicCache to additionally store query states for attention weight reconstruction.
    '''
    def __init__(self, query_indices=[]) -> None:
        super().__init__()
        self._query_indices = query_indices
        self.query_cache = []
        if not hasattr(self, '_seen_tokens'):
            self._seen_tokens = 0
    
    @property
    def key_cache(self):
        return [layer.keys for layer in self.layers]
    
    @property
    def value_cache(self):
        return [layer.values for layer in self.layers]
    
    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        updated_key, updated_value = super().update(key_states, value_states, layer_idx)
        if query_states is not None:
            if len(self.query_cache) <= layer_idx:
                self.query_cache.append(query_states)
            else:
                self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        
        return updated_key, updated_value
    
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(None, key_states, value_states, layer_idx)
        return cache