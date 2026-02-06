from typing import List
from .rankers import LlmRanker, SearchResult
import openai
import re
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
from collections import Counter
import tiktoken
import random
import math
import os
import sys
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    print("Seems vllm is not installed, RankR1SetwiseLlmRanker only supports vllm inference so far.")

ICR_AVAILABLE = False
ICR_CLASS_AVAILABLE = False
DynamicCacheWithQuery = None
repeat_kv = None
InContextReranker = None
try:
    icr_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'icr', 'src')
    if os.path.exists(icr_path):
        sys.path.insert(0, icr_path)
        from custom.custom_cache import DynamicCacheWithQuery
        from transformers.models.mistral.modeling_mistral import repeat_kv
        ICR_AVAILABLE = True
        print("[Setwise] ICR components loaded successfully for attention scoring.")
        
        try:
            icr_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'icr')
            if os.path.exists(icr_base_path):
                if icr_base_path not in sys.path:
                    sys.path.insert(0, icr_base_path)
                from src.in_context_reranker import InContextReranker
                ICR_CLASS_AVAILABLE = True
                print("[Setwise] ICR InContextReranker class loaded successfully.")
            else:
                raise ImportError(f"ICR base path not found: {icr_base_path}")
        except Exception as e2:
            print(f"[Setwise] Warning: ICR InContextReranker class not available. ICRSetwiseRanker will not work. Error: {e2}")
except ImportError as e:
    print(f"[Setwise] Warning: ICR components not available. Attention scoring will not work. Error: {e}")

random.seed(929)


class SetwiseLlmRanker(LlmRanker):
    
    @staticmethod
    def get_label(index):
        return str(index + 1)
    
    @staticmethod
    def parse_label(output):
        output = output.strip().strip('[]').strip()
        try:
            num = int(output)
            if num < 1:
                return 0
            return num - 1
        except (ValueError, TypeError):
            return 0

    def __init__(self,
                 model_name_or_path,
                 tokenizer_name_or_path,
                 device,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 debug_log_dir=None):

        self.device = device
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.debug_log_dir = debug_log_dir
        self.current_query_id = None
        self.compare_counter = 0
        self.debug_log_file = None
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.device) if self.tokenizer else None

            max_docs = max(200, num_child + 1)
            numeric_strings = [str(i+1) for i in range(max_docs)]
            numeric_token_ids_batch = self.tokenizer.batch_encode_plus(
                numeric_strings,
                add_special_tokens=False,
                return_attention_mask=False
            ).input_ids
            self.numeric_token_ids = torch.tensor([
                ids[0] if len(ids) > 0 else self.tokenizer.encode("1", add_special_tokens=False)[0]
                for ids in numeric_token_ids_batch
            ], dtype=torch.long)
        elif self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.tokenizer.use_default_system_prompt = False
            if 'vicuna' and 'v1.5' in model_name_or_path:
                self.tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if scoring == 'likelihood':
                max_docs = max(200, num_child + 1)
                numeric_strings = [str(i+1) for i in range(max_docs)]
                numeric_token_ids_batch = self.tokenizer.batch_encode_plus(
                    numeric_strings,
                    add_special_tokens=False,
                    return_attention_mask=False
                ).input_ids
                fallback_token_ids = self.tokenizer.encode("1", add_special_tokens=False)
                fallback_token_id = fallback_token_ids[0] if len(fallback_token_ids) > 0 else self.tokenizer.eos_token_id
                self.numeric_token_ids = torch.tensor([
                    ids[0] if len(ids) > 0 else fallback_token_id 
                    for ids in numeric_token_ids_batch
                ], dtype=torch.long)
            else:
                self.numeric_token_ids = None
            
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir).eval()
        else:
            raise NotImplementedError(f"Model type {self.config.model_type} is not supported yet for setwise:(")

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation
        self.compare_counter += 1

        if docs is None or len(docs) == 0:
            return "1"
        
        debug_info = {} if self.debug_log_dir else None

        passages = "\n\n".join([f'Passage {i+1}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage:'

        if self.scoring == 'generation':
            if self.config.model_type == 't5':

                if self.num_permutation == 1:
                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids,
                                                   max_new_tokens=2)[0]

                    self.total_completion_tokens += output_ids.shape[0]

                    raw_output = self.tokenizer.decode(output_ids,
                                                       skip_special_tokens=True).strip()
                    if debug_info is not None:
                        debug_info['raw_output'] = raw_output
                    output = raw_output.strip().strip('[]').strip()
                    try:
                        num = int(output)
                        if num < 1 or num > len(docs):
                            output = "1"
                        else:
                            output = str(num)
                    except (ValueError, TypeError):
                        output = "1"
                    if debug_info is not None:
                        debug_info['final_output'] = output
                else:
                    id_passage = [(i, p) for i, p in enumerate(docs)]
                    labels = [self.get_label(i) for i in range(len(docs))]
                    batch_data = []
                    for _ in range(self.num_permutation):
                        batch_data.append([random.sample(id_passage, len(id_passage)),
                                           random.sample(labels, len(labels))])

                    batch_ref = []
                    input_text = []
                    for batch in batch_data:
                        ref = []
                        passages = []
                        characters = []
                        for p, c in zip(batch[0], batch[1]):
                            ref.append(p[0])
                            passages.append(p[1].text)
                            characters.append(c)
                        batch_ref.append((ref, characters))
                        passages = "\n\n".join([f'Passage {characters[i]}: "{passages[i]}"' for i in range(len(passages))])
                        input_text.append(f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                                          + passages + '\n\nOutput only the passage label of the most relevant passage:')

                    input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                    self.total_prompt_tokens += input_ids.shape[1] * input_ids.shape[0]

                    output_ids = self.llm.generate(input_ids,
                                                   decoder_input_ids=self.decoder_input_ids.repeat(input_ids.shape[0], 1),
                                                   max_new_tokens=2)
                    output = self.tokenizer.batch_decode(output_ids[:, self.decoder_input_ids.shape[1]:],
                                                         skip_special_tokens=True)

                    # vote
                    candidates = []
                    for ref, result in zip(batch_ref, output):
                        result = result.strip().strip('[]').strip()
                        docids, characters = ref
                        try:
                            num = int(result)
                            if num < 1 or num > len(characters):
                                print(f"Unexpected output: {result} (out of range)")
                                continue
                            win_doc = docids[characters.index(str(num))]
                            candidates.append(win_doc)
                        except (ValueError, TypeError, ValueError):
                            print(f"Unexpected output: {result}")
                            continue

                    if len(candidates) == 0:
                        print(f"Unexpected voting: {output}")
                        output = "Unexpected voting."
                    else:
                        candidate_counts = Counter(candidates)
                        max_count = max(candidate_counts.values())
                        most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                                  count == max_count]
                        if len(most_common_candidates) == 1:
                            output = self.get_label(most_common_candidates[0])
                        else:
                            output = self.get_label(random.choice(most_common_candidates))

            elif self.config.model_type == 'llama':
                conversation = [{"role": "user", "content": input_text}]

                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                prompt += " Passage:"

                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                
                # Create attention mask to avoid warnings
                attention_mask = torch.ones_like(input_ids)

                output_ids = self.llm.generate(input_ids,
                                               attention_mask=attention_mask,
                                               do_sample=False,
                                               temperature=0.0,
                                               top_p=None,
                                               max_new_tokens=3,
                                               pad_token_id=self.tokenizer.pad_token_id)[0]

                self.total_completion_tokens += output_ids.shape[0]

                raw_output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                                   skip_special_tokens=True)
                if debug_info is not None:
                    debug_info['raw_output'] = raw_output
                
                output = raw_output.strip()
                
                import re
                match = re.search(r'(\d+)', output)
                if match:
                    num = int(match.group(1))
                    if num >= 1 and num <= len(docs):
                        output = str(num)
                    else:
                        output = "1"
                else:
                    output = "1"
                
                if debug_info is not None:
                    debug_info['final_output'] = output

        elif self.scoring == 'likelihood':
            if self.config.model_type == 't5':
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.numeric_token_ids[:len(docs)]]
                    ranked = sorted(zip([self.get_label(i) for i in range(len(docs))], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]
                    if debug_info is not None:
                        debug_info['final_output'] = output
                        debug_info['raw_output'] = f"likelihood_mode: {output}"

            elif self.config.model_type == 'llama':
                conversation = [{"role": "user", "content": input_text}]
                prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                prompt += " Passage:"
                
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                
                attention_mask = torch.ones_like(input_ids)
                
                with torch.no_grad():
                    outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    
                    # Use pre-computed numeric_token_ids (similar to T5 implementation)
                    # This is much faster than encoding numbers every time
                    label_token_ids = self.numeric_token_ids[:len(docs)].to(self.device)
                    
                    # Compute probabilities for each label token (keep on GPU)
                    distributions = torch.softmax(logits, dim=-1)
                    scores = distributions[label_token_ids]
                    
                    # Rank by score (keep computation on GPU, only convert to CPU for final result)
                    # Use torch.topk for better performance
                    top_score, top_idx = torch.topk(scores, k=1)
                    output = self.get_label(top_idx.item())
                    if debug_info is not None:
                        debug_info['final_output'] = output
                        debug_info['raw_output'] = f"likelihood_mode: {output}"

            else:
                raise NotImplementedError(f"Likelihood scoring not implemented for model type: {self.config.model_type}")

        try:
            num = int(output.strip().strip('[]').strip())
            if num < 1 or num > len(docs):
                print(f"Unexpected output: {output} (out of range for {len(docs)} docs)")
        except (ValueError, TypeError):
            print(f"Unexpected output: {output}")
        
        if self.debug_log_file and debug_info is not None:
            raw_output = debug_info.get('raw_output', '')
            final_output = debug_info.get('final_output', output)
            self.debug_log_file.write(f"{raw_output} | {final_output}\n")
            self.debug_log_file.flush()

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            best_ind = self.parse_label(output)
            if best_ind >= len(inds):
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        self.compare_counter = 0  # 重置计数器
        
        if self.debug_log_dir and self.debug_log_file is None:
            import os
            os.makedirs(self.debug_log_dir, exist_ok=True)
            log_filepath = os.path.join(self.debug_log_dir, 'outputs.txt')
            self.debug_log_file = open(log_filepath, 'w', encoding='utf-8')
        
        if self.method == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.method == "bubblesort":
            last_start = max(0, len(ranking) - (self.num_child + 1))

            for i in range(self.k):
                start_ind = last_start
                end_ind = min(len(ranking), last_start + (self.num_child + 1))
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    if start_ind < 0:
                        start_ind = 0
                    if start_ind >= len(ranking):
                        break
                    if end_ind <= start_ind:
                        end_ind = min(len(ranking), start_ind + 1)
                    output = self.compare(query, ranking[start_ind:end_ind])
                    best_ind = self.parse_label(output)
                    if best_ind >= len(ranking[start_ind:end_ind]):
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind])-1:
                                last_start += len(ranking[start_ind:end_ind])-1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
                    
        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.method == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results
    
    def close_debug_log(self):
        if self.debug_log_file:
            self.debug_log_file.close()
            self.debug_log_file = None

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class ICRSetwiseRanker(SetwiseLlmRanker):
    
    def __init__(self,
                 icr_model,
                 num_child=3,
                 k=10,
                 method="heapsort",
                 scoring_strategy='masked_NA_calibration'):
        """
        """
        if not ICR_CLASS_AVAILABLE:
            raise ImportError(
                "ICR InContextReranker class is not available. Please ensure the 'icr' directory "
                "is available and the in_context_reranker module can be imported."
            )
        
        if icr_model is None:
            raise ValueError("icr_model cannot be None. Please provide an initialized InContextReranker instance.")
        
        self.icr_model = icr_model
        self.num_child = num_child
        self.k = k
        self.method = method
        self.scoring_strategy = scoring_strategy
        
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = None
        self.debug_log_dir = None
        self.debug_log_file = None
        self.current_query_id = None
        self.compare_counter = 0
        
        print(f"[ICRSetwiseRanker] Initialized with method={method}, num_child={num_child}, k={k}, "
              f"scoring_strategy={scoring_strategy}")
    
    def compare(self, query: str, docs: List):
        self.total_compare += 1
        
        if len(docs) == 0:
            print("[ICRSetwiseRanker] Warning: Empty document list in compare, returning '1'.")
            return self.get_label(0)
        
        docs_to_process = list(docs)
        
        if self.icr_model.reverse_doc_order:
            docs_to_process = docs_to_process[::-1]
        
        doc_texts = []
        valid_doc_indices = []
        for idx, doc in enumerate(docs_to_process):
            original_idx = len(docs) - 1 - idx if self.icr_model.reverse_doc_order else idx
            
            if doc.text is None:
                print(f"[ICRSetwiseRanker] Warning: Document {original_idx} has None text, skipping.")
                continue
            doc_text = str(doc.text).strip()
            if len(doc_text) == 0:
                print(f"[ICRSetwiseRanker] Warning: Document {original_idx} has empty text, skipping.")
                continue
            doc_texts.append(doc_text)
            valid_doc_indices.append(original_idx)
        
        if len(doc_texts) == 0:
            print("[ICRSetwiseRanker] Warning: All documents filtered, returning '1'.")
            return self.get_label(0)
        
        if len(doc_texts) != len(docs):
            print(f"[ICRSetwiseRanker] Warning: {len(docs) - len(doc_texts)} documents were filtered. "
                  f"Original: {len(docs)}, After filtering: {len(doc_texts)}")
        
        try:
            query_str = str(query).strip() if query is not None else ""
            if len(query_str) == 0:
                print("[ICRSetwiseRanker] Warning: Empty query, returning '1'.")
                return self.get_label(0)
            
            if hasattr(self.icr_model, 'enable_timing') and self.icr_model.enable_timing:
                if not hasattr(self.icr_model, '_current_query_forward_time'):
                    self.icr_model._current_query_forward_time = 0.0
                if not hasattr(self.icr_model, '_current_query_score_time'):
                    self.icr_model._current_query_score_time = 0.0
                self.icr_model._current_query_forward_time = 0.0
                self.icr_model._current_query_score_time = 0.0
            
            if hasattr(self.icr_model, 'prompt_separator'):
                prompt_sep = self.icr_model.prompt_separator
                if not isinstance(prompt_sep, str):
                    print(f"[ICRSetwiseRanker] Warning: prompt_separator is not a string! Type: {type(prompt_sep)}, Value: {repr(prompt_sep)}")
                    if isinstance(prompt_sep, list):
                        self.icr_model.prompt_separator = ' '.join(str(x) for x in prompt_sep)
                    else:
                        self.icr_model.prompt_separator = str(prompt_sep)
            
            sort_result = self.icr_model.get_sorted_docs(
                query=query_str,
                retrieval_doc_pool=doc_texts,
                return_per_doc_results=False,
                order='desc'
            )
            
            sorted_indices = sort_result[0][0]
            
            if not sorted_indices or len(sorted_indices) == 0:
                print("[ICRSetwiseRanker] Warning: ICR returned empty sorted_indices, returning '1'.")
                return self.get_label(0)
            
            winner_idx_in_doc_texts = sorted_indices[0]
            
            if winner_idx_in_doc_texts < len(valid_doc_indices):
                winner_idx = valid_doc_indices[winner_idx_in_doc_texts]
            else:
                print(f"[ICRSetwiseRanker] Warning: Index {winner_idx_in_doc_texts} out of range for valid_doc_indices (len={len(valid_doc_indices)}), returning '1'.")
                return self.get_label(0)
            
            if winner_idx < 0 or winner_idx >= len(docs):
                print(f"[ICRSetwiseRanker] Warning: Invalid winner_idx {winner_idx} for {len(docs)} docs, returning '1'.")
                print(f"[ICRSetwiseRanker] Debug: winner_idx_in_doc_texts={winner_idx_in_doc_texts}, "
                      f"doc_texts_len={len(doc_texts)}, docs_len={len(docs)}")
                return self.get_label(0)
            
            winner_label = self.get_label(winner_idx)
            
            return winner_label
            
        except Exception as e:
            print(f"[ICRSetwiseRanker] Error in compare: {e}")
            print(f"[ICRSetwiseRanker] Debug info:")
            print(f"  - query: {repr(query)[:100]}")
            print(f"  - doc_texts type: {type(doc_texts)}")
            print(f"  - doc_texts length: {len(doc_texts) if doc_texts else 0}")
            if doc_texts:
                print(f"  - first doc type: {type(doc_texts[0])}")
                print(f"  - first doc preview: {repr(doc_texts[0])[:100]}")
            import traceback
            traceback.print_exc()
            return self.get_label(0)
    
    def truncate(self, text, length):
        if hasattr(self.icr_model, 'tokenizer'):
            try:
                return self.icr_model.tokenizer.convert_tokens_to_string(
                    self.icr_model.tokenizer.tokenize(text)[:length]
                )
            except:
                pass
        words = text.split()[:length]
        return ' '.join(words)


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort', k=10):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.k = k
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.system_prompt = "You are RankGPT, an intelligent assistant specialized in selecting the most relevant passage from a pool of passages based on their relevance to the query."
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {i+1}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage.'

        response = openai.ChatCompletion.create(
            model=self.llm,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0.0,
            max_tokens=1
        )

        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

        output = response.choices[0].message.content.strip().upper()

        try:
            num = int(output.strip().strip('[]').strip())
            if num < 1 or num > len(docs):
                print(f"Unexpected output: {output} (out of range for {len(docs)} docs)")
                output = "1"
        except (ValueError, TypeError):
            print(f"Unexpected output: {output}")
            output = "1"

        return output

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])


class RankR1SetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self,
                 model_name_or_path,
                 prompt_file,
                 lora_name_or_path=None,
                 tokenizer_name_or_path=None,
                 num_child=19,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None,
                 verbose=False):

        if scoring != 'generation':
            raise NotImplementedError(f"Scoring method {scoring} is not supported for RankR1SetwiseLlmRanker. RankR1SetwiseLlmRanker only supports 'generation' scoring.")
        self.verbose = verbose

        import toml
        self.prompt = toml.load(prompt_file)

        from huggingface_hub import snapshot_download
        import os
        if lora_name_or_path is not None:
            if not os.path.exists(lora_name_or_path):
                lora_path = snapshot_download(lora_name_or_path)
            else:
                lora_path = lora_name_or_path
        else:
            lora_path = None

        self.lora_path = lora_path
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
        self.sampling_params = SamplingParams(temperature=0.0,
                                              max_tokens=2048)
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        self.llm = LLM(model=model_name_or_path,
                       tokenizer=tokenizer_name_or_path,
                       enable_lora=True if lora_name_or_path is not None else False,
                       max_lora_rank=32,
                       )

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [f'[{i+1}]' for i in range(len(docs))]
        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append([random.sample(id_passage, len(id_passage)),
                               labels])

        batch_ref = []
        input_text = []
        for batch in batch_data:
            ref = []
            passages = []
            characters = []
            for p, c in zip(batch[0], batch[1]):
                ref.append(p[0])
                passages.append(p[1].text)
                characters.append(c)
            batch_ref.append((ref, characters))
            passages = "\n".join([f'{characters[i]} {passages[i]}' for i in range(len(passages))])
            system_message = self.prompt["prompt_system"]
            user_message = self.prompt['prompt_user'].format(query=query,
                                                             docs=passages)
            input_text.append([
                {'role': "system", 'content': system_message},
                {'role': "user", 'content': user_message}
            ])

        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )

        results = []
        for output, input in zip(outputs, input_text):
            self.total_completion_tokens += len(output.outputs[0].token_ids)
            self.total_prompt_tokens += len(output.prompt_token_ids)

            completion = output.outputs[0].text

            if self.verbose:
                print('--------------------------------------')
                print(f'query: {query}')
                print(f'input_text:\n{self.tokenizer.apply_chat_template(input, tokenize=False)}')
                print(f'completion:\n{completion}')
                print('--------------------------------------')

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(f'input_text:\n{input}, completion:\n{completion}')

        candidates = []
        for ref, result in zip(batch_ref, results):
            result = result.strip()
            docids, characters = ref
            if result not in characters:
                if self.verbose:
                    print(f"Unexpected output: {result}")
                continue
            win_doc = docids[characters.index(result)]
            candidates.append(win_doc)

        if len(candidates) == 0:
            if self.verbose:
                print(f"Unexpected voting: {results}")
            output = "Unexpected voting."
        else:
            candidate_counts = Counter(candidates)
            max_count = max(candidate_counts.values())
            most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                      count == max_count]
            if len(most_common_candidates) == 1:
                output = f'[{most_common_candidates[0] + 1}]'
            else:
                output = f'[{random.choice(most_common_candidates) + 1}]'

        try:
            num = int(output.strip().strip('[]').strip())
            if num < 1 or num > len(docs):
                pass  # 超出范围，但继续执行
        except (ValueError, TypeError):
            pass  # 格式错误，但继续执行
        else:
            if self.verbose:
                print(f"Unexpected output: {output}")

        return output
