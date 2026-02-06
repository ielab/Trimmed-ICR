import math
import time

import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.mistral.modeling_mistral import repeat_kv
import torch
import gc
import random

from .custom.custom_cache import DynamicCacheWithQuery
from .custom.custom_modeling_mistral import MistralForCausalLM
from .custom.custom_modeling_llama import LlamaForCausalLM
from .custom.custom_modeling_qwen3 import Qwen3ForCausalLM

class InContextReranker():

    def __init__(self, 
                 base_llm_name,
                 prompt_template='instruct',
                 prompt_prefix='',
                 prompt_suffix='',
                 scoring_strategy='query_last',
                 use_fa2=True,
                 retrieval_type='QA',
                 sliding_window_size=20,
                 sliding_window_stride=None,
                 reverse_doc_order=False,
                 use_double_query=False,
                 max_layer=None,
                 aggregation_start_layer=None,
                enable_timing=True,
                ) -> None:
        '''
        Inputs:
            base_llm: The base LLM model to be used for document retrieval.
            tokenizer: The tokenizer for the base LLM model.
            prompt_template: The template for the prompt to be used for document retrieval. 
                Options: 
                    'instruct': default instruction template
                    'simple': no instruction used
                    'simple_instruct: only wrap the input with corresponding chat templates of the base model. e.g. [INST]...[/INST] for Mistral-instruct
        '''

        # Setup the base LLM
        self._base_llm_name = base_llm_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_llm_name)
        self.tokenizer = tokenizer
        print(f"initialized tokenizer for [{base_llm_name}]")
        
        if any([x in base_llm_name.lower() for x in ['mistralai/mistral', ]]):
            BaseLLMClass = MistralForCausalLM
        elif 'qwen' in base_llm_name.lower():
            BaseLLMClass = Qwen3ForCausalLM
        elif any([x in base_llm_name.lower() for x in ['llama']]):
            BaseLLMClass = LlamaForCausalLM
        else:
            print(f"Warning: The model family for [{base_llm_name}] is not supported by InContextRAGModel!")
            raise NotImplementedError

        prompt_template, prompt_prefix, prompt_suffix, = self._setup_llm_prompts(prompt_template, base_llm_name)
        
        if use_fa2:
            _attn_implementation = "flash_attention_2"
        else:
            _attn_implementation = "eager"

        llm = BaseLLMClass.from_pretrained(
                base_llm_name, 
                torch_dtype=torch.float16, 
                attn_implementation=_attn_implementation,
                device_map='auto'
            )
        self.llm = llm
        self.llm.config.pad_token_id = self.llm.config.eos_token_id
        
        # Setup prompts for ICR
        assert prompt_template in ['instruct', 'simple', 'simple_instruct'], "Invalid prompt template!"
        
        self.prompt_template = prompt_template
        self.prompt_prefix = prompt_prefix

        
        self.prompt_suffix = prompt_suffix
        self.scoring_strategy = scoring_strategy
        
        if retrieval_type == 'QA':
            print('[ICR is using QA prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please answer the following question based on the information in the paragraphs above.'
        elif retrieval_type == 'IE':
            print('[ICR is using IE prompt type]')
            self.retrieval_instruction = ' Here are some paragraphs:'
            self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.'
        else:
            raise NotImplementedError('Invalid retrieval type! Should be one of [QA, IE]')

        assert scoring_strategy in ['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'], "Invalid scoring strategy!"
        
        self._use_fa2 = use_fa2
        if use_fa2:
            print('Using FA2 for retrieval score computation.')
        else:
            print('Using eager attention weights for retrieval score computation.')
        self.num_layers = self.llm.config.num_hidden_layers
        

        self.start_layer = 0
        if max_layer is None:
            self.end_layer = self.num_layers - 1
        else:
            self.end_layer = max_layer
        
        if aggregation_start_layer is None:
            self.aggregation_start_layer = self.start_layer
        else:
            self.aggregation_start_layer = aggregation_start_layer
        self.aggregation_end_layer = self.end_layer

        if self.end_layer < self.num_layers - 1:
            self.llm._max_layers = self.end_layer + 1
            if hasattr(self.llm, 'model'):
                self.llm.model._max_layers = self.end_layer + 1
            print('[ICR] Forward pass: layers 0-{} (total {} layers, limited from {} layers)'.format(
                self.end_layer, self.end_layer + 1, self.num_layers))
        else:
            print('[ICR] Forward pass: layers 0-{} (total {} layers, using all layers)'.format(
                self.end_layer, self.end_layer + 1))
        
        print('[ICR] Aggregation: layers {}-{} (total {} layers)'.format(
            self.aggregation_start_layer, self.aggregation_end_layer, 
            self.aggregation_end_layer - self.aggregation_start_layer + 1))


        # The following settings are for constructing the input prompt.
        if 'qwen' in self._base_llm_name.lower():
            self.prompt_bos_length = 0
        else:
            self.prompt_bos_length = 1
            
        if any(x in self._base_llm_name.lower() for x in ['mistral-']):
            self.additional_prompt_offset = 1 # for models that adds a ' ' at the beginning when tokenizing the prompt. e.g. '\n\n' -> [<s>, ' ', '\n\n']
            self.prompt_separator = '\n\n'
        elif any([x in self._base_llm_name.lower() for x in ['llama', 'qwen']]):
            self.additional_prompt_offset = 0
            self.prompt_separator = ' \n\n'
        else:
            self.additional_prompt_offset = 0
            self.prompt_separator = '\n\n'

        
        # Setup sliding window.
        # ICR typically works worse with sliding window, especially with smaller window sizes. Try to fit all documents to be re-ranked in the context as much as possible. 
        self.reverse_doc_order = reverse_doc_order
        self.sliding_window_size = sliding_window_size
        if sliding_window_stride is None:
            self.sliding_window_stride = sliding_window_size//2
        else:
            self.sliding_window_stride = sliding_window_stride
        
        # Setup double query mode
        self.use_double_query = use_double_query
        if use_double_query:
            print('[ICR] Using double query mode: documents + query1 + query')
        
        # Setup timing
        self.enable_timing = enable_timing
        if enable_timing:
            print('[ICR] Timing enabled: will record forward pass and score_documents time for each query')
        
    def _setup_llm_prompts(self, prompt_template, base_llm_name):
        
        if prompt_template == '':
            prompt_template='instruct' if any(x in base_llm_name.lower() for x in ['instruct']) else 'simple'
        else:
            assert prompt_template in ['instruct', 'simple', 'simple_instruct']
        print('ICR is using prompt template [{}] for in-context retrieval'.format(prompt_template))
        
        if  'mistral' in base_llm_name.lower():
            prompt_prefix = '[INST]'
            prompt_suffix = '[/INST]'
        elif 'llama-3' in base_llm_name.lower():
            prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif 'qwen' in base_llm_name.lower():
            prompt_prefix = '<|im_start|>user\n'
            prompt_suffix = '<|im_end|>\n<|im_start|>assistant\n'
        else:
            raise NotImplementedError("Prompt prefix and suffix not defined for the model family of {}.".format(base_llm_name))
        
        return prompt_template, prompt_prefix, prompt_suffix

            
    def rerank(self, query, documents, return_per_doc_results=False, order='desc', calib_query_type='NA', return_per_layer_scores=False, query1=None):
        '''
        Rerank the documents based on the query using a sliding window strategy.
        Assume that input documents are sorted by their relevance to the query in the descending order.
        '''
        if self.enable_timing:
            self._current_query_forward_time = 0.0
            self._current_query_score_time = 0.0
        
        # reverse the order of input documents to perform sliding window from the rear to the front of the list
        # documents.reverse()
        N_docs = len(documents)

        if self.sliding_window_size < 0:
            self.sliding_window_size = N_docs
            
        sorted_doc_ids = list(range(N_docs))
        sorted_doc_ids.reverse()
        
        sorted_doc_scores = []
        if return_per_doc_results == 'tok':
            per_doc_results = []
        else:
            per_doc_results = None
        
        if return_per_layer_scores:
            sorted_doc_ids_per_layer = []
            sorted_doc_scores_per_layer = []
            num_layers = None
        
        _i = 0
        _j = min(self.sliding_window_size, N_docs)
        while True:
            
            ids = [sorted_doc_ids[i] for i in range(_i, _j)]
            if not self.reverse_doc_order:
                # Put the most relevant documents at the front of document list.
                ids.reverse()

            docs = [documents[i] for i in ids]
            sort_result = self.get_sorted_docs(query, docs, return_per_doc_results=return_per_doc_results, 
                                             order='asc', return_per_layer_scores=return_per_layer_scores,
                                             query1=query1)
            
            _sorted_doc_ids, _sorted_doc_scores = sort_result[0]
            _per_doc_results = sort_result[1]
            
            _sorted_doc_ids_per_layer = None
            _sorted_doc_scores_per_layer = None
            if return_per_layer_scores:
                _sorted_doc_ids_per_layer, _sorted_doc_scores_per_layer = sort_result[2]
                if num_layers is None:
                    num_layers = len(_sorted_doc_ids_per_layer)
                    sorted_doc_ids_per_layer = [[] for _ in range(num_layers)]
                    sorted_doc_scores_per_layer = [[] for _ in range(num_layers)]

            __sorted_doc_ids = [ids[i] for i in _sorted_doc_ids]
            for i in range(_i, _j):
                sorted_doc_ids[i] = __sorted_doc_ids[i-_i]

            if _j < N_docs:
                sorted_doc_scores.extend(_sorted_doc_scores[:self.sliding_window_stride])
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results[:self.sliding_window_stride])
                if return_per_layer_scores:
                    for layer_idx in range(num_layers):
                        sorted_doc_ids_per_layer[layer_idx].extend([ids[i] for i in _sorted_doc_ids_per_layer[layer_idx][:self.sliding_window_stride]])
                        sorted_doc_scores_per_layer[layer_idx].extend(_sorted_doc_scores_per_layer[layer_idx][:self.sliding_window_stride])
            else:
                sorted_doc_scores.extend(_sorted_doc_scores)
                if return_per_doc_results == 'tok':
                    per_doc_results.extend(_per_doc_results)
                if return_per_layer_scores:
                    for layer_idx in range(num_layers):
                        sorted_doc_ids_per_layer[layer_idx].extend([ids[i] for i in _sorted_doc_ids_per_layer[layer_idx]])
                        sorted_doc_scores_per_layer[layer_idx].extend(_sorted_doc_scores_per_layer[layer_idx])
                break

            _i += self.sliding_window_stride
            _j += self.sliding_window_stride
            _j = min(_j, N_docs)
            
        
        if order == 'desc':
            sorted_doc_ids.reverse()
            sorted_doc_scores.reverse()
            if return_per_doc_results == 'tok':
                per_doc_results.reverse()
            if return_per_layer_scores:
                for layer_idx in range(num_layers):
                    sorted_doc_ids_per_layer[layer_idx].reverse()
                    sorted_doc_scores_per_layer[layer_idx].reverse()

        assert len(sorted_doc_ids) == len(sorted_doc_scores), "Length mismatch between sorted doc ids ({}) and scores({})!".format(len(sorted_doc_ids), len(sorted_doc_scores))
        
        timing_info = None
        if self.enable_timing:
            timing_info = {
                'forward_pass_time': self._current_query_forward_time,
                'score_documents_time': self._current_query_score_time
            }
            print('[ICR Timing] Query completed - Forward pass: {:.4f}s, Score documents: {:.4f}s'.format(
                self._current_query_forward_time, self._current_query_score_time))
        
        return_values = [(sorted_doc_ids, sorted_doc_scores), per_doc_results]
        if return_per_layer_scores:
            return_values.append((sorted_doc_ids_per_layer, sorted_doc_scores_per_layer))
        
        if self.enable_timing:
            return_values.append(timing_info)
        
        return tuple(return_values)

    def score_documents(
            self,
            llm_input,
            doc_tok_idx_spans,
            query_start_tok_idx,
            query_end_tok_idx,
            context_start_idx=0,
            return_per_doc_results=False,
            long_prompt=False,
            return_cache=False,
            kv_cache=None,
            return_per_layer_scores=False,
        ):
        if self.enable_timing:
            score_start_time = time.time()

        tokenized_input = self.tokenizer(llm_input,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids[:, context_start_idx:]
        
        _query_indices = list(range(query_start_tok_idx-context_start_idx, query_end_tok_idx-context_start_idx+1))
        
        if kv_cache is None:
            if self._use_fa2:
                kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)
            else:
                kv_cache=DynamicCache()
        else:
            kv_cache.query_cache = []
            _query_indices = _query_indices
            kv_cache._query_indices = _query_indices

        if self.enable_timing:
            forward_start_time = time.time()
        
        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )
        
        if self.enable_timing:
            forward_end_time = time.time()
            forward_time = forward_end_time - forward_start_time
            self._current_query_forward_time += forward_time

        if self._use_fa2:
            # Extract key and query vectors from FA2. Then recompute attention scores for re-ranking.
            kv_cache = output.past_key_values

            long_prompt = False
            if len(_input_ids[0]) > 80000:
                # For sequences that are too long, compute scores on CPU to avoid OOM.
                # Adjust the limit here depending on your system configuration.
                print('Long sequence of more than 40K tokens detected. Computing attention scores on CPU.')
                long_prompt = True
            
            attention_weights = []
            doc_tok_weights = []
            
            if long_prompt:
                _device = 'cpu'
            else:
                _device = 'cuda:0'
            
            for i in range(self.start_layer, self.end_layer+1):
                key_states = kv_cache.layers[i].keys[:,:,:query_end_tok_idx+1]
                
                query_states = kv_cache.query_cache[i]
                
                attn_weights = self._get_attn_weights(
                    key_states, query_states, use_cpu=long_prompt
                ).to(_device).squeeze(0)
                attn_weights = attn_weights.mean(1)
                attention_weights.append(attn_weights.squeeze(0))
                
        else:
            # Directly extract attention weights from the attention layers of the LLM.
            attention_weights = [attn[0][:,query_start_tok_idx:query_end_tok_idx+1,:].mean(1) 
                                 for attn in output.attentions[:self.end_layer+1]]

        attention_weights = torch.stack(attention_weights, dim=0)
        # attention_weights shape: [num_layers, num_heads, seq_len]
        
        num_layers = attention_weights.shape[0]
        
        if return_per_doc_results != 'none':
            per_doc_results = [[None, None] for _ in range(len(doc_tok_idx_spans))]
        else:
            per_doc_results = None
        
        doc_scores_per_layer = None
        per_layer_tok_results = None
        
        agg_start_idx = self.aggregation_start_layer - self.start_layer
        agg_end_idx = self.aggregation_end_layer - self.start_layer + 1
        attention_weights_agg = attention_weights[agg_start_idx:agg_end_idx].sum(0).sum(0)  # [seq_len]
        
        doc_scores = []
        for i, doc_span in enumerate(doc_tok_idx_spans): 
            _tok_score = attention_weights_agg[doc_span[0]:doc_span[1]]
            doc_scores.append(_tok_score.sum())

            if return_per_doc_results != 'none':
                _doc_tok_ids = tokenized_input.input_ids[0][doc_span[0]:doc_span[1]]
                _doc_toks = self.tokenizer.convert_ids_to_tokens(_doc_tok_ids)
                per_doc_results[i][0] = _doc_toks
                per_doc_results[i][1] = _tok_score.clone().detach()
        
        if doc_scores:
            doc_scores = torch.stack(doc_scores).to(attention_weights.device)
        else:
            doc_scores = torch.tensor([], device=attention_weights.device)
        
        if self.enable_timing:
            score_end_time = time.time()
            score_time = score_end_time - score_start_time
            self._current_query_score_time += score_time
        
        if return_per_layer_scores:
            doc_scores_per_layer = []
            per_layer_tok_results = []
            for layer_idx in range(num_layers):
                layer_attn_weights = attention_weights[layer_idx]
                layer_attn_weights = layer_attn_weights.sum(0)
                
                layer_doc_scores = []
                layer_tok_results = []
                for i, doc_span in enumerate(doc_tok_idx_spans):
                    _tok_score = layer_attn_weights[doc_span[0]:doc_span[1]]
                    layer_doc_scores.append(_tok_score.sum())
                    
                    if return_per_doc_results != 'none':
                        layer_tok_results.append(_tok_score.clone().detach())
                
                if layer_doc_scores:
                    layer_doc_scores = torch.stack(layer_doc_scores).to(attention_weights.device)
                else:
                    layer_doc_scores = torch.tensor([], device=attention_weights.device)
                
                doc_scores_per_layer.append(layer_doc_scores)
                per_layer_tok_results.append(layer_tok_results)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return_values = [doc_scores, per_doc_results]
        if return_cache:
            return_values.append(kv_cache)
        if return_per_layer_scores:
            return_values.append(doc_scores_per_layer)
            if return_per_doc_results != 'none' and per_layer_tok_results is not None:
                return_values.append(per_layer_tok_results)
        
        return tuple(return_values)

    def get_sorted_docs(self, query, retrieval_doc_pool, return_per_doc_results=False, prompt_prefix='', order='desc', return_per_layer_scores=False, query1=None):

        kv_cache = None
        
        if self.scoring_strategy == 'query_last':
            # ICR without calibration.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)
            score_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, 
                                                return_per_doc_results=return_per_doc_results, 
                                                return_per_layer_scores=return_per_layer_scores)
            doc_scores = score_result[0]
            perdoc_result = score_result[1]
            doc_scores_per_layer = score_result[2] if return_per_layer_scores else None
        
        elif self.scoring_strategy == 'attention_sorting':
            # ICR without both calibration and attention aggregation.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)
            query_start_idx = query_end_idx # Only using last query token (i.e. attention sorting).
            score_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx,
                                                return_per_doc_results=return_per_doc_results,
                                                return_per_layer_scores=return_per_layer_scores)
            doc_scores = score_result[0]
            perdoc_result = score_result[1]
            doc_scores_per_layer = score_result[2] if return_per_layer_scores else None
        
        elif self.scoring_strategy == 'NA_only':
            # For analyzing the intrinsic bias captured by calibration scores.
            query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)
            score_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx,
                                                return_per_doc_results=return_per_doc_results,
                                                return_per_layer_scores=return_per_layer_scores)
            doc_scores = score_result[0]
            perdoc_result = score_result[1]
            doc_scores_per_layer = score_result[2] if return_per_layer_scores else None
        
        elif self.scoring_strategy == 'NA_calibration_no_agg':
            # ICR without attention aggregation.
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)
            query_start_idx = query_end_idx
            doc_scores_query, perdoc_result, _, kv_cache = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results, return_cache=True)
            
            
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)

            for i in range(len(kv_cache.layers)):
                kv_cache.layers[i].keys = kv_cache.layers[i].keys[:,:,:query_start_idx,:]
                kv_cache.layers[i].values = kv_cache.layers[i].values[:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            if not self._use_fa2:
                kv_cache = None
                
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            query_start_idx = query_end_idx
            doc_scores_calib, doc_tok_scores_calib_na,_ = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx, return_per_doc_results=return_per_doc_results,  kv_cache=kv_cache, context_start_idx=context_start_idx)

            doc_scores = doc_scores_query - doc_scores_calib
            
            if return_per_doc_results != 'none':
                for i in range(len(perdoc_result)):
                    perdoc_result[i][1] -= doc_tok_scores_calib_na[i][1]
        
        elif self.scoring_strategy == 'masked_NA_calibration':
            return_per_doc_results = 'tok'
            # The default ICR method
            
            # FP with calibration query
            calibration_query = 'N/A'
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(calibration_query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)

            calib_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx,
                                                return_per_doc_results=return_per_doc_results,
                                                return_cache=True,
                                                return_per_layer_scores=return_per_layer_scores)
            doc_scores_calib = calib_result[0]
            doc_tok_scores_calib_na = calib_result[1]
            kv_cache = calib_result[2]
            idx = 3
            doc_scores_calib_per_layer = calib_result[idx] if return_per_layer_scores else None
            calib_per_layer_tok_results = None
            if return_per_layer_scores:
                idx += 1
                if return_per_doc_results != 'none' and len(calib_result) > idx:
                    calib_per_layer_tok_results = calib_result[idx]
                    idx += 1
            
            # Use kv_cache from first query to speed up forward() for the calibration query.
            # query_start_idx should be the same for both queries.
            for i in range(len(kv_cache.layers)):
                kv_cache.layers[i].keys = kv_cache.layers[i].keys[:,:,:query_start_idx,:]
                kv_cache.layers[i].values = kv_cache.layers[i].values[:,:,:query_start_idx,:]
            kv_cache._seen_tokens = query_start_idx
            
            if not self._use_fa2:
                kv_cache = None
                
            if kv_cache is not None:
                context_start_idx=query_start_idx
            else:
                context_start_idx=0

            # FP with the actual query
            llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx = self._prepare_input_for_document_retrieval(query, retrieval_doc_pool, system_prompt=prompt_prefix, query_position='last', query1=query1)
        
            query_result = self.score_documents(llm_prompt, doc_tok_idx_spans, query_start_idx, query_end_idx,
                                               return_per_doc_results=return_per_doc_results,
                                               kv_cache=kv_cache,
                                               context_start_idx=context_start_idx,
                                               return_per_layer_scores=return_per_layer_scores)
            doc_scores_query = query_result[0]
            perdoc_result = query_result[1]
            idx = 2  # 跳过 doc_scores 和 perdoc_result
            doc_scores_query_per_layer = query_result[idx] if return_per_layer_scores else None
            query_per_layer_tok_results = None
            if return_per_layer_scores:
                idx += 1
                if return_per_doc_results != 'none' and len(query_result) > idx:
                    query_per_layer_tok_results = query_result[idx]
                    idx += 1

            
            _i = 0
            device = perdoc_result[0][1].device if perdoc_result and perdoc_result[0][1] is not None else self.llm.device
            doc_scores = torch.zeros(len(retrieval_doc_pool), device=device)
            
            doc_scores_per_layer = None
            
            if return_per_layer_scores:
                num_layers = len(doc_scores_query_per_layer)
                doc_scores_per_layer = [torch.zeros(len(retrieval_doc_pool), device=device) for _ in range(num_layers)]

            for doc_tok_score, doc_tok_score_na in zip(perdoc_result, doc_tok_scores_calib_na):
                doc_tok_score[1] = doc_tok_score[1].to(doc_tok_score_na[1].device)
                calibrated_scores = doc_tok_score[1] - doc_tok_score_na[1]
                
                mean_bias = calibrated_scores.mean()
                std_bias = calibrated_scores.std()
                threshold = mean_bias - 2*std_bias
                tok_mask = (calibrated_scores>threshold)
                
                doc_tok_score[1] = doc_tok_score[1] * tok_mask
                doc_tok_score_na[1] = doc_tok_score_na[1] * tok_mask
                doc_tok_score[1] = doc_tok_score[1] - doc_tok_score_na[1]
                doc_scores[_i] = doc_tok_score[1].sum()
                
                if return_per_layer_scores:
                    for layer_idx in range(num_layers):
                        if query_per_layer_tok_results is not None and calib_per_layer_tok_results is not None:
                            layer_query_tok_score = query_per_layer_tok_results[layer_idx][_i]
                            layer_calib_tok_score = calib_per_layer_tok_results[layer_idx][_i]
                            layer_query_tok_score = layer_query_tok_score.to(layer_calib_tok_score.device)
                            layer_calibrated_scores = layer_query_tok_score - layer_calib_tok_score
                            layer_mean_bias = layer_calibrated_scores.mean()
                            layer_std_bias = layer_calibrated_scores.std()
                            layer_threshold = layer_mean_bias - 2 * layer_std_bias
                            layer_tok_mask = (layer_calibrated_scores > layer_threshold)
                            
                            layer_query_tok_score = layer_query_tok_score * layer_tok_mask
                            layer_calib_tok_score = layer_calib_tok_score * layer_tok_mask
                            layer_query_tok_score = layer_query_tok_score - layer_calib_tok_score
                            
                            doc_scores_per_layer[layer_idx][_i] = layer_query_tok_score.sum()
                        else:
                            layer_query_score = doc_scores_query_per_layer[layer_idx][_i]
                            layer_calib_score = doc_scores_calib_per_layer[layer_idx][_i]
                            doc_scores_per_layer[layer_idx][_i] = layer_query_score - layer_calib_score
                
                _i+=1

        per_doc_result = None
        if order in ['desc', 'asc']:
            sorted_results = torch.sort(doc_scores, descending=(order=='desc'))
            if return_per_doc_results != 'none':
                per_doc_result = [(perdoc_result[i][0], perdoc_result[i][1]) for i in sorted_results.indices]
            
            return_values = [(sorted_results.indices.tolist(), sorted_results.values.tolist()), per_doc_result]
            
            if return_per_layer_scores and doc_scores_per_layer is not None:
                sorted_doc_ids_per_layer = []
                sorted_doc_scores_per_layer = []
                for layer_idx in range(len(doc_scores_per_layer)):
                    layer_scores = doc_scores_per_layer[layer_idx]
                    layer_sorted_results = torch.sort(layer_scores, descending=(order=='desc'))
                    sorted_doc_ids_per_layer.append(layer_sorted_results.indices.tolist())
                    sorted_doc_scores_per_layer.append(layer_sorted_results.values.tolist())
                return_values.append((sorted_doc_ids_per_layer, sorted_doc_scores_per_layer))
            
            return tuple(return_values)
        elif order=='none':
            # Only return the scores and the per-doc results for documents in the input order.
            # Used during development.
            return_values = [list(range(len(retrieval_doc_pool))), doc_scores, per_doc_result]
            if return_per_layer_scores and doc_scores_per_layer is not None:
                return_values.append(doc_scores_per_layer)
            return tuple(return_values)
        else:
            print(f"Invalid order: {order}. Please use 'desc', 'asc' or 'none")
            raise NotImplementedError

    def _prepare_input_for_document_retrieval(self, query, documents, system_prompt='', query_position='last', query1=None):
        '''
        Only tested with Mistral and Llama-3.1. Models using other tokenizers may need to modify this function.
        
        Args:
            query: The main query for attention computation
            documents: List of documents
            system_prompt: System prompt
            query_position: Position of query ('first' or 'last')
            query1: Optional first query (used when use_double_query=True). If None and use_double_query=True, uses query as query1.
        '''
        llm_prompt = ''
        document_span_intervals = []
        query1_start_idx = None
        query1_end_idx = None
        # Initialize query1 if use_double_query is enabled
        if self.use_double_query and query1 is None:
            query1 = query  # Use query as query1 if not provided
        

        if self.prompt_template == 'simple':
            system_prompt = ''
        elif self.prompt_template == 'simple_instruct':
            system_prompt = system_prompt
        elif self.prompt_template == 'instruct':
            if system_prompt != '':
                system_prompt = self.retrieval_instruction.format(len(documents), query) + self.prompt_separator + system_prompt
            else:
                system_prompt = self.retrieval_instruction.format(len(documents), query)
        
        system_prompt = self.prompt_prefix + system_prompt

        query_start_idx = None
        query_end_idx = None
        
        
        separator_length = self.tokenizer(self.prompt_separator, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset # remove the leading ['<s>', '_'] tokens
        
        llm_prompt = system_prompt
        
        
        prompt_length = self.tokenizer(llm_prompt+self.prompt_separator, return_tensors='pt').input_ids.size(1)-separator_length # add and subtract separator tokens for accurate prefix length
        
        if query_position == 'first':
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = f'Query:'

                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                query_start_idx = prompt_length - 1 # The ':' after 'Query'    
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length
                query_start_idx = prompt_length # The start of the query context
            
            if self.prompt_template == 'simple':
                query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
            elif self.prompt_template in ['instruct', 'simple_instruct']:
                query_prompt = f' {query.strip()}'
            
            llm_prompt += query_prompt
            prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
            query_end_idx = prompt_length - 1 
            
        
        if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch!')
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                raise Exception('ICR prompt length mismatch before adding docs.')

        _doc_separator_length = separator_length

        for i, doc in enumerate(documents):
            
            doc = f'[{i+1}] {doc}'
            prompt_length += _doc_separator_length
            llm_prompt += self.prompt_separator + doc
            doc_length = self.tokenizer(self.prompt_separator + doc, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - _doc_separator_length - self.additional_prompt_offset # - bos_length for the leading ['<s>'] token, -additional for the potential extra tokens, e.g. the '_' token between <s> and <0x0A> when <0x0A> is the first token for mistral models.
            
            document_span_intervals.append((prompt_length, prompt_length + doc_length))
            prompt_length += doc_length

            if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch @ doc {}!'.format(i))
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                print('-'*30)
                self.__show_tokens(llm_prompt)
                print('-'*30)
                print('doc length:', doc_length)
                self.__show_tokens(self.prompt_separator+doc)
                raise Exception('ICR prompt length mismatch after adding docs.')

        # Insert query1 if use_double_query is enabled
        if self.use_double_query and query_position == 'last':
            # Add query1 after documents
            query1_prompt = f'{query1.strip()}'
            prompt_length += separator_length
            llm_prompt += self.prompt_separator + query1_prompt
            query1_start_idx = prompt_length
            query1_length = self.tokenizer(self.prompt_separator + query1_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
            query1_end_idx = prompt_length + query1_length - 1
            prompt_length += query1_length
            
            # Add separator between query1 and query (only one separator)
            prompt_length += separator_length
            llm_prompt += self.prompt_separator
            
            if prompt_length != self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1):
                print('Prompt length mismatch after adding query1!')
                print(prompt_length, ' vs ', self.tokenizer(llm_prompt, return_tensors='pt').input_ids.size(1))
                raise Exception('ICR prompt length mismatch after adding query1.')

        if query_position == 'last':
            query_start_idx = prompt_length + separator_length
            if self.prompt_template in ['simple', 'instruct']:
                instruction_prompt = self.retrieval_instruction_late + self.prompt_separator + 'Query:'
                llm_prompt += self.prompt_separator + instruction_prompt 
                prompt_length += separator_length
                prompt_length += self.tokenizer(self.prompt_separator + instruction_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - separator_length - self.additional_prompt_offset
                query_start_idx = prompt_length
                
            else:
                llm_prompt += self.prompt_separator
                prompt_length += separator_length
                query_start_idx = prompt_length

        if self.prompt_template == 'simple':
            query_prompt = f' {query.strip()}{self.prompt_separator}Answer:'
        elif self.prompt_template in ['instruct', 'simple_instruct']:
            query_prompt = f' {query.strip()}'
            if query_position == 'last':
                query_prompt += self.prompt_suffix.format(len(documents))

        query_text_only = f' {query.strip()}'
        query_text_length = self.tokenizer(query_text_only, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
        
        llm_prompt += query_prompt
        prompt_length += self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
        if query_position == 'last':
            query_prompt_with_suffix_length = self.tokenizer(query_prompt, return_tensors='pt').input_ids.size(1) - self.prompt_bos_length - self.additional_prompt_offset
            suffix_length = query_prompt_with_suffix_length - query_text_length
            query_end_idx = query_start_idx + query_text_length - 1
        
        # if self.use_double_query and query1_start_idx is not None:
        #     query1_display = query1 if query1 is not None else query
        #     print(f'[DEBUG] Query1 token indices: [{query1_start_idx}-{query1_end_idx}]')

        # print(f'[DEBUG] Query token indices: [{query_start_idx}-{query_end_idx}]')
        
        return llm_prompt, document_span_intervals, query_start_idx, query_end_idx
    
    @classmethod
    def __show_tokens(self, string):
        # Shows tokenized string.
        # Mainly used for debugging prompt construction for document retrieval.
        tokenized_string_ids = self.tokenizer(string, return_tensors='pt').input_ids[0]
        print(self.tokenizer.convert_ids_to_tokens(tokenized_string_ids), tokenized_string_ids.size(0))
                      
    @classmethod
    def _get_attn_weights(cls, key_states, query_states, use_cpu=False):

        bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        if use_cpu:
            query_states = query_states.cpu()
            key_states = key_states.cpu()

        key_states = repeat_kv(key_states, num_key_value_groups)


        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        # Make causal mask and add it to attention weights.
        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(0)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True) # Log-sum-exp of attention weights for numerical stability in softmax.
        attn_weights = torch.exp(attn_weights - attn_lses) # softmax
        
        return attn_weights

    @classmethod
    def _get_causal_mask(cls, attn_weights):
        # Make causal mask for attention weights.
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(0))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask