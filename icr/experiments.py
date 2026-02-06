import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
import random
from src.in_context_reranker import InContextReranker
import argparse
from pyserini.search import get_qrels

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_per_doc_results', type=str, default='none', choices=['none', 'tok', 'att_head'],)
parser.add_argument('--llm_name', type=str, required=True)
parser.add_argument('--scoring_strategy', type=str, default='masked_NA_calibration', choices=['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'])
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1,)
parser.add_argument('--oracle', action='store_true')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--use_eager_attn', action='store_true')
parser.add_argument('--retrieval_type', type=str, default='IE', choices=['QA', 'IE'])
parser.add_argument('--save_retrieval_results', action='store_true')
parser.add_argument('--no_rerank', action='store_true')
parser.add_argument('--beir_eval', action='store_true')
parser.add_argument('--shuffle_documents', action='store_true')
parser.add_argument('--reverse_doc_order', action='store_true')
parser.add_argument('--calib_query_type', type=str, default='NA', choices=['NA'])
parser.add_argument('--retriever', type=str, default='colbertv2', choices=['bm25', 'colbertv2'])
parser.add_argument('--reranker', type=str, choices=['icr', 'rankgpt'])
parser.add_argument('--rerank_sliding_window_size', type=int, default=-1)
parser.add_argument('--rerank_sliding_window_stride', type=int, default=10)
parser.add_argument('--disable_vllm', action='store_true')
parser.add_argument('--truncate_by_space', type=int, default=-1) 
parser.add_argument('--actual_topk', type=int, default=-1)
parser.add_argument('--per_layer_analysis', action='store_true', help='Enable per-layer analysis: compute NDCG for each layer separately')
parser.add_argument('--use_double_query', action='store_true', help='Enable double query mode: docs + query + query (first query as context, second query for attention)')
parser.add_argument('--max_layer', type=int, default=None, help='Maximum layer to compute (0-indexed). If None, uses all layers. Example: 16 means compute layers 0-16.')
parser.add_argument('--aggregation_start_layer', type=int, default=None, help='Start layer for aggregation (0-indexed). If None, starts from layer 0. Aggregation goes from this layer to max_layer.')
parser.add_argument('--disable_timing', action='store_true', help='Disable timing statistics (timing is enabled by default)')
args = parser.parse_args()

if args.beir_eval or args.data not in ['musique', 'hotpotqa', '2wikimultihopqa']:
    from beir.retrieval.evaluation import EvaluateRetrieval
    args.beir_eval = True
else:
    args.beir_eval = False

# BRIGHT数据集列表（统一定义，便于维护）
BRIGHT_DATASETS = ['aops', 'biology', 'earth_science', 'economics', 'leetcode',
                   'pony', 'psychology', 'robotics', 'stackoverflow',
                   'sustainable_living', 'theoremqa_questions', 'theoremqa_theorems']


def get_dataset_file_path(data_name, retriever, top_k):
    """
    根据数据集名称构建统一的文件路径格式：icr_{type}_{dataset}_{retriever}_top_{k}.json
    返回：(file_path, dataset_type)
    dataset_type: 'trec', 'beir', 'bright'
    """
    if data_name in ['trec-dl-19', 'trec-dl-20']:
        # TREC数据集：type = 'trec_dl'
        file_path = f'./retriever_outpout/icr_trec_dl_{data_name}_{retriever}_top_{top_k}.json'
        dataset_type = 'trec'
    elif data_name in BRIGHT_DATASETS:
        # BRIGHT数据集：type = 'bright'
        file_path = f'./retriever_outpout/icr_bright_{data_name}_{retriever}_top_{top_k}.json'
        dataset_type = 'bright'
    else:
        # BEIR数据集：type = 'beir'
        file_path = f'./retriever_outpout/icr_beir_{data_name}_{retriever}_top_{top_k}.json'
        dataset_type = 'beir'
    
    return file_path, dataset_type


def build_qrels_from_dataset(query_set, id_key, dataset_name=None):
    """Build qrels from is_supporting labels in dataset"""
    # BRIGHT datasets - load full ground truth
    if dataset_name in BRIGHT_DATASETS:
        from datasets import load_dataset
        print(f'Loading full qrels from BRIGHT dataset: {dataset_name}')
        bright_examples = load_dataset('xlangai/BRIGHT', 'examples', split=dataset_name)
        qrels = {}
        for ex in bright_examples:
            qid = str(ex['id'])
            qrels[qid] = {gold_id: 1 for gold_id in ex['gold_ids']}
        return qrels

    # Other datasets - original method
    qrels = {}
    for query in query_set:
        qid = query[id_key]
        qrels[qid] = {}
        for para in query['paragraphs']:
            if para.get('is_supporting', False):
                qrels[qid][para['idx']] = 1
    return qrels
    
if args.reranker == 'rankgpt':
    from src.rank_gpt_reranker import RankGPTModel
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

NO_TITLE_DATASETS = ['fiqa']

if __name__ == '__main__':

    if args.seed != -1:
        print('using random seed: ', args.seed)
        random.seed(args.seed)
    

    if args.data in ['musique', 'hotpotqa', '2wikimultihopqa']:
        query_set = json.load(open(f'./retriever_outpout/icr_multihop_{args.data}_colbertv2_top_{args.top_k}.json','r'))
        id_key = 'id'
        beir_exp = False
        if args.reranker == 'icr':
            args.retrieval_type='QA'
    else:
        # 统一文件路径格式：icr_{type}_{dataset}_{retriever}_top_{k}.json
        # 通过文件名前缀（trec/beir/bright）自动识别数据集类型
        file_path, dataset_type = get_dataset_file_path(args.data, args.retriever, args.top_k)
        query_set = json.load(open(file_path))

        id_key = 'idx'
        beir_exp = True

        if args.reranker == 'icr' and args.data in ['trec-covid', 'fiqa', 'webis-touche2020', 'dbpedia-entity', 'nq']:
            args.retrieval_type = 'QA'
        else:
            args.retrieval_type = 'IE'
        
        if args.debug:
            k = args.debug
            print('Debug mode, only processing {} queries out of {} ones.'.format(k, len(query_set)))
            query_set = query_set[:k]

    ks = [1,2,3,4,5,10]
    recalls = []

    llm_name = args.llm_name
    print('-'*50)

    if args.reverse_doc_order:
        print('Reversing the order of paragraphs for each query. i.e. most relevant paragraph is at the end.')

    if args.reranker == 'rankgpt':
        if args.save_per_doc_results != 'none':
            print('RankGPT does not support saving per-doc results. Setting save_per_doc_results to none.')
            args.save_per_doc_results = 'none'
    
    if not args.no_rerank:
        print('Doing re-ranking on the [{}] dataset with base retriever [{}]'.format(args.data, args.retriever))
        print('Doing re-ranking using {} + {}'.format(args.reranker, llm_name))
        if args.truncate_by_space > 0:
            # This option is added to follow RankGPT's setting
            print('Truncating each paragraph to {} words.'.format(args.truncate_by_space))

    if args.reranker == 'icr':
        print('Using ICR with scoring strategy: {}'.format(args.scoring_strategy))
    
    if args.actual_topk > 0:
        print('Using actual topk: ', args.actual_topk)
        assert not args.save_retrieval_results, 'Cannot save retrieval results when using actual topk.'
    
    # Set output file name
    reranker_str = f'{args.reranker}_{args.llm_name.split("/")[-1]}'
    if args.reranker == 'icr':
        reranker_str += f'_scoring_{args.scoring_strategy}'
    if args.rerank_sliding_window_size == -1:
        reranker_str += '_no_sw'
    if args.save_per_doc_results != 'none':
        assert args.reranker == 'icr', 'Only ICR supports saving per-doc results.'

        if beir_exp:
            per_doc_output_file = './output/per_doc_results/rerank_{}_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.save_per_doc_results,
                
                args.top_k
                )
        else:
            per_doc_output_file = './output/per_doc_results/rerank_{}_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.save_per_doc_results,
                args.top_k
                )
        if args.truncate_by_space > 0:
            per_doc_output_file=per_doc_output_file.replace('.json', '_trunc_{}.json'.format(args.truncate_by_space))

        if args.debug:
            per_doc_output_file=per_doc_output_file.replace('.json', '_debug.json')
            per_doc_output_file=per_doc_output_file.replace('.json', '_calib_type_{}.json'.format(args.calib_query_type))
            
        if args.reverse_doc_order:
            per_doc_output_file=per_doc_output_file.replace('.json', '_reverse_order.json')
        
        if args.max_layer is not None and args.aggregation_start_layer is not None:
            per_doc_output_file=per_doc_output_file.replace('.json', '_max_layer_{}_start_layer_{}.json'.format(args.max_layer, args.aggregation_start_layer))
        
        print('Saving per-doc results to {}.'.format(per_doc_output_file))
        all_per_doc_results = []
        
    if args.save_retrieval_results:
        if beir_exp:
            retrieval_output_file = './output/retrieval_results/rerank_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.top_k
                )
        else:
            retrieval_output_file = './output/retrieval_results/rerank_{}_{}_{}_top_{}.json'.format(
                args.retriever,
                args.data,
                reranker_str,
                args.top_k
                )
        if args.truncate_by_space > 0:
            retrieval_output_file=retrieval_output_file.replace('.json', '_trunc_{}.json'.format(args.truncate_by_space))
        if args.reverse_doc_order:
            retrieval_output_file=retrieval_output_file.replace('.json', '_reverse_order.json')
        base_retrieval_output_file = retrieval_output_file
        per_layer_output_file = retrieval_output_file
        if args.per_layer_analysis:
            per_layer_output_file = retrieval_output_file.replace('.json', '_per_layer.json')
        if args.use_double_query:
            retrieval_output_file = retrieval_output_file.replace('.json', '_double_query.json')
            base_retrieval_output_file = base_retrieval_output_file.replace('.json', '_double_query.json')
            per_layer_output_file = per_layer_output_file.replace('.json', '_double_query.json')
        
        print('Saving retrieval results to {}.'.format(retrieval_output_file))
    retrieval_results = {} # stored in BEIR's format
    
    all_timing_info = []
    
    retrieval_results_per_layer = None
    recalls_per_layer = None
    
    if args.per_layer_analysis and not args.no_rerank and args.reranker == 'icr':
        retrieval_results_per_layer = {}
        recalls_per_layer = [] if not args.beir_eval else None

    if not args.no_rerank:
        if args.reranker == 'icr':
            # Use LLM-based ICR
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
            reranker = InContextReranker(
                llm_name,
                scoring_strategy=args.scoring_strategy,
                use_fa2=not args.use_eager_attn,
                retrieval_type=args.retrieval_type,
                reverse_doc_order=args.reverse_doc_order,
                sliding_window_size=args.rerank_sliding_window_size,
                sliding_window_stride=args.rerank_sliding_window_stride,
                use_double_query=args.use_double_query,
                max_layer=args.max_layer,
                aggregation_start_layer=args.aggregation_start_layer,
                enable_timing=not args.disable_timing
            )
        elif args.reranker == 'rankgpt':
            reranker = RankGPTModel(
                llm_name,
                use_vllm=not(args.disable_vllm),
                sliding_window_size=args.rerank_sliding_window_size,
                sliding_window_stride=args.rerank_sliding_window_stride)
    else:
        print('Directly reporting {} results'.format(args.retriever))
        
    
    if args.data in NO_TITLE_DATASETS:
        print('Not adding title to paragraphs for the [{}] dataset.'.format(args.data))

    if args.reranker == 'rankgpt':
        format_correct_rates = []

    for i, query in enumerate(tqdm(query_set)):
        
        question = query['question']
        if beir_exp:
            paragraphs = [p for p in query['paragraphs'] if p['idx'] != query[id_key]] # remove same doc from pool for some datasets.
        else:
            paragraphs = query['paragraphs']
        if args.actual_topk > 0:
            paragraphs = paragraphs[:args.actual_topk]

        if args.truncate_by_space > 0:
            # Truncate each paragraph by space.
            # We follow the implementation of RankGPT and truncate the documents to 300 words for BEIR experiments.
            for p in paragraphs:
                p['paragraph_text'] = ' '.join(p['paragraph_text'].split(' ')[:args.truncate_by_space])
        
        if args.shuffle_documents:
            random.shuffle(paragraphs)
        
        
        total_gold_doc_num = min(args.top_k, query['num_gold_docs'])
        total_supporting_items = len([x for x in paragraphs if x['is_supporting']])
        
        if args.data in NO_TITLE_DATASETS:
            passages = [(p['paragraph_text']).strip() for p in paragraphs] 
            gold_docs = set([(p['paragraph_text']).strip() for p in paragraphs if p['is_supporting']])
        else:
            passages = [(p['title'] + '\n' + p['paragraph_text']).strip() for p in paragraphs] 

            gold_docs = set([(p['title'] + '\n' + p['paragraph_text']).strip() for p in paragraphs if p['is_supporting']])

        


        gold_ids = [_i for _i, p in enumerate(paragraphs) if p['is_supporting']]

        if args.debug:
            print('question: ', question)
            print('gold docs: ', gold_ids)

        if not args.no_rerank:
            try:
                if args.reranker == 'rankgpt':
                    sorted_doc_ids, format_correct_rate = reranker.rerank(question, passages)
                    format_correct_rates.append(format_correct_rate)
                    sorted_doc_scores = np.array(list(range(len(passages), 0, -1))) / len(passages)
                    per_doc_results = None
                    sorted_doc_ids_per_layer = None
                    sorted_doc_scores_per_layer = None
                elif args.reranker == 'icr':
                    # Pass question as query1 when use_double_query is enabled
                    query1 = question if args.use_double_query else None
                    rerank_result = reranker.rerank(question, passages, 
                                                   return_per_doc_results=args.save_per_doc_results, 
                                                   calib_query_type=args.calib_query_type,
                                                   return_per_layer_scores=args.per_layer_analysis,
                                                   query1=query1)
                    (sorted_doc_ids, sorted_doc_scores) = rerank_result[0]
                    per_doc_results = rerank_result[1]
                    
                    sorted_doc_ids_per_layer = None
                    sorted_doc_scores_per_layer = None
                    if args.per_layer_analysis:
                        sorted_doc_ids_per_layer, sorted_doc_scores_per_layer = rerank_result[2]
                    
                    timing_info = None
                    if hasattr(reranker, 'enable_timing') and reranker.enable_timing:
                        timing_info = rerank_result[-1]
                else:
                    print('Unknown reranker type!')
                    sorted_doc_ids = list(range(len(passages)))
                    sorted_doc_scores = np.array(list(range(len(passages), 0, -1)))/len(passages)
                    per_doc_results = None
                    sorted_doc_ids_per_layer = None
                    sorted_doc_scores_per_layer = None
                    timing_info = None

            except Exception as e:
                print(e)
                print('Error in retrieval for example No. {}, fall back to ColBERTv2 Results...'.format(i))
            
                sorted_doc_ids = list(range(len(passages)))                    
                sorted_doc_scores = np.array(list(range(len(passages), 0, -1)))/len(passages)
                per_doc_results = None
                sorted_doc_ids_per_layer = None
                sorted_doc_scores_per_layer = None
                timing_info = None
        else:
            # Report retriever performance
            sorted_doc_ids = list(range(len(passages)))
            sorted_doc_scores = np.array(list(range(len(passages), 0, -1)))/len(passages)

            per_doc_results = None
            timing_info = None
        if args.debug:
            print('sorted doc ids: ', sorted_doc_ids)
        
        if beir_exp:
            _id = query[id_key]
        else:
            _id = i
        retrieval_results[_id] = {}
        if 'timing_info' in locals() and timing_info is not None:
            retrieval_results[_id]['timing'] = timing_info
            all_timing_info.append(timing_info)
        for _i, sorted_idx in enumerate(sorted_doc_ids):
            if beir_exp:
                retrieval_results[_id][str(paragraphs[sorted_idx]['idx'])] = sorted_doc_scores[_i]
            else:
                retrieval_results[_id][sorted_idx] = sorted_doc_scores[_i]
        
        if args.per_layer_analysis and sorted_doc_ids_per_layer is not None:
            if _id not in retrieval_results_per_layer:
                retrieval_results_per_layer[_id] = {}
            num_layers = len(sorted_doc_ids_per_layer)
            for layer_idx in range(num_layers):
                layer_key = f'layer_{layer_idx}'
                if layer_key not in retrieval_results_per_layer[_id]:
                    retrieval_results_per_layer[_id][layer_key] = {}
                for _i, sorted_idx in enumerate(sorted_doc_ids_per_layer[layer_idx]):
                    if beir_exp:
                        retrieval_results_per_layer[_id][layer_key][str(paragraphs[sorted_idx]['idx'])] = sorted_doc_scores_per_layer[layer_idx][_i]
                    else:
                        retrieval_results_per_layer[_id][layer_key][sorted_idx] = sorted_doc_scores_per_layer[layer_idx][_i]

        recalls_at = []

        if args.save_per_doc_results != 'none':
            _per_doc_result = {
                'query': question,
                'docs':[]
            }
            for _i, _id in enumerate(sorted_doc_ids):

                _doc_result = {
                    'input_rank': _id,
                    'is_gold': _id in gold_ids,
                    'retrieval_score': np.round(sorted_doc_scores[_i], 5).tolist(),
                    'toks': per_doc_results[_i][0],
                    'scores': per_doc_results[_i][1].tolist()
                }
                _per_doc_result['docs'].append(_doc_result)
            all_per_doc_results.append(_per_doc_result)

        for k in ks:
            retrieved_docs = np.array(passages)[sorted_doc_ids[:k]]
            retrieved_docs = set(retrieved_docs)
            true_positives = gold_docs.intersection(retrieved_docs)
            n_tp = len(true_positives) # regular evaluation
            
            if args.oracle:
                n_tp = min(total_supporting_items, k) # oracle setting for performance upper bound

            if total_gold_doc_num == 0:
                recalls_at.append(0)
            else:
                recalls_at.append(n_tp / total_gold_doc_num)

        recalls.append(recalls_at)
        
        if args.per_layer_analysis and sorted_doc_ids_per_layer is not None and not args.beir_eval:
            recalls_at_per_layer = []
            num_layers = len(sorted_doc_ids_per_layer)
            for layer_idx in range(num_layers):
                layer_recalls_at = []
                layer_sorted_doc_ids = sorted_doc_ids_per_layer[layer_idx]
                for k in ks:
                    retrieved_docs = np.array(passages)[layer_sorted_doc_ids[:k]]
                    retrieved_docs = set(retrieved_docs)
                    true_positives = gold_docs.intersection(retrieved_docs)
                    n_tp = len(true_positives)
                    
                    if args.oracle:
                        n_tp = min(total_supporting_items, k)
                    
                    if total_gold_doc_num == 0:
                        layer_recalls_at.append(0)
                    else:
                        layer_recalls_at.append(n_tp / total_gold_doc_num)
                recalls_at_per_layer.append(layer_recalls_at)
            recalls_per_layer.append(recalls_at_per_layer)

    if not args.beir_eval:
        print(pd.DataFrame(recalls, columns=ks).agg(['mean']).T)
        
        if args.per_layer_analysis and recalls_per_layer:
            print('\n---- Per-Layer Recall (Multihop) ----')
            for layer_idx in range(len(recalls_per_layer[0])):
                layer_recalls_df = pd.DataFrame([r[layer_idx] for r in recalls_per_layer], columns=ks)
                print(f'\nLayer {layer_idx} Recall:\n', layer_recalls_df.agg(['mean']).T)
    
    if args.reranker == 'rankgpt':
        print('RankGPT Format Correct Rate: ', np.mean(format_correct_rates))


    if args.save_per_doc_results != 'none':
        json.dump(all_per_doc_results, open(per_doc_output_file, 'w'), indent=2)
        print(f'Saved results to {per_doc_output_file}')
    if args.save_retrieval_results:
        base_output_file = base_retrieval_output_file
        if args.data in NO_TITLE_DATASETS:
            base_output_file = base_output_file.replace('.json', '_no_title.json')
        if args.debug:
            base_output_file = base_output_file.replace('.json', '_debug.json')
        if args.no_rerank:
            base_output_file = base_output_file.replace('.json', '_{}.json'.format(args.retriever))
        if args.reranker == 'rankgpt':
            base_output_file = base_output_file.replace('.json', '_fcr_{}.json'.format(np.mean(format_correct_rates)))
        
        if args.max_layer is not None and args.aggregation_start_layer is not None:
            base_output_file = base_output_file.replace('.json', '_max_layer_{}_start_layer_{}.json'.format(args.max_layer, args.aggregation_start_layer))
        
        json.dump(retrieval_results, open(base_output_file, 'w'), indent=2)
        print(f'Saved retrieval results to {base_output_file}')
        
        if not args.beir_eval and args.save_retrieval_results and args.per_layer_analysis:
            output_data = {'all_layers': {'recall': {k: float(pd.DataFrame(recalls, columns=ks).agg(['mean']).T.loc[k, 'mean']) for k in ks}}}
            
            if all_timing_info:
                avg_forward_time = sum(t['forward_pass_time'] for t in all_timing_info) / len(all_timing_info)
                avg_score_time = sum(t['score_documents_time'] for t in all_timing_info) / len(all_timing_info)
                output_data['average_timing'] = {
                    'forward_pass_time': avg_forward_time,
                    'score_documents_time': avg_score_time,
                    'num_queries': len(all_timing_info)
                }
            
            if args.per_layer_analysis and retrieval_results_per_layer is not None and recalls_per_layer:
                per_layer_recall = {}
                for layer_idx in range(len(recalls_per_layer[0])):
                    layer_recalls_df = pd.DataFrame([r[layer_idx] for r in recalls_per_layer], columns=ks)
                    per_layer_recall[f'layer_{layer_idx}'] = {
                        'recall': {k: float(layer_recalls_df.agg(['mean']).T.loc[k, 'mean']) for k in ks}
                    }
                output_data['per_layer'] = per_layer_recall
                output_data['retrieval_results_per_layer'] = retrieval_results_per_layer
            
            per_layer_output = per_layer_output_file
            if args.data in NO_TITLE_DATASETS:
                per_layer_output = per_layer_output.replace('.json', '_no_title.json')
            if args.debug:
                per_layer_output = per_layer_output.replace('.json', '_debug.json')
            if args.no_rerank:
                per_layer_output = per_layer_output.replace('.json', '_{}.json'.format(args.retriever))
            
            if args.max_layer is not None and args.aggregation_start_layer is not None:
                per_layer_output = per_layer_output.replace('.json', '_max_layer_{}_start_layer_{}.json'.format(args.max_layer, args.aggregation_start_layer))
            
            json.dump(output_data, open(per_layer_output, 'w'), indent=2)
            print(f'\nSaved multihop per-layer analysis results to {per_layer_output}')
        
    if args.beir_eval:
        print('---- BEIR Evaluation ----')
        # 根据数据集类型加载 qrels
        # 统一通过文件名前缀（trec/beir/bright）来区分数据集类型
        _, dataset_type = get_dataset_file_path(args.data, args.retriever, args.top_k)
        
        if dataset_type == 'bright':
            # BRIGHT 数据集：从 HuggingFace 加载完整 qrels（这些数据集不在 pyserini 的预构建 qrels 中）
            _qrels = build_qrels_from_dataset(query_set, id_key, dataset_name=args.data)
        elif dataset_type == 'trec':
            # TREC 数据集：使用 pyserini 的预构建 qrels
            if args.data == 'trec-dl-19':
                qrel_name = 'dl19-passage'
            elif args.data == 'trec-dl-20':
                qrel_name = 'dl20-passage'
            else:
                raise ValueError(f'Unknown TREC dataset: {args.data}')
            _qrels = get_qrels(qrel_name)
        else:  # dataset_type == 'beir'
            # BEIR 数据集：使用 pyserini 的官方 qrels（包含所有相关文档，不仅仅是检索结果中的）
            qrel_name = 'beir-v1.0.0-{}-test'.format(args.data)
            _qrels = get_qrels(qrel_name)
        evaluator = EvaluateRetrieval()
        qrels = {}
        
        retrieval_results_for_eval = {}
        for qid in retrieval_results:
            assert isinstance(qid, str)
            retrieval_results_for_eval[qid] = {k: v for k, v in retrieval_results[qid].items() if k != 'timing'}
            try:
                __qrels = _qrels[qid]
            except:
                try:
                    __qrels = _qrels[int(qid)]
                except:
                    print('Error in qrels for query id: ', qid)
                    continue
            
            # make sure the qrels are in the right format
            qrels[qid] = {}
            for doc_id in __qrels:
                qrels[qid][str(doc_id)] = __qrels[doc_id]
               
            doc_keys = list(qrels[qid].keys())
            for key in doc_keys:
                if not isinstance(qrels[qid][key], int):
                    qrels[qid][key] = int(qrels[qid][key]) # make sure the relevance is integer
                if qrels[qid][key] == 0:
                    qrels[qid].pop(key)
            
        ndcg, _, recall, precision = evaluator.evaluate(qrels, retrieval_results_for_eval, ks)
        print('NDCG (All Layers):\n', json.dumps(ndcg, indent=2))
        
        ndcg_per_layer = None
        if args.per_layer_analysis and 'retrieval_results_per_layer' in locals() and retrieval_results_per_layer:
            print('\n---- Per-Layer Analysis ----') 
            first_qid = None
            for qid in retrieval_results_per_layer:
                if retrieval_results_per_layer[qid] and any(k.startswith('layer_') for k in retrieval_results_per_layer[qid].keys()):
                    first_qid = qid
                    break
            
            if first_qid is None:
                print("Warning: No per-layer results found in retrieval_results_per_layer")
            else:
                layer_keys = [k for k in retrieval_results_per_layer[first_qid].keys() if k.startswith('layer_')]
                layer_keys.sort(key=lambda x: int(x.split('_')[1]))
                num_layers = len(layer_keys)
                ndcg_per_layer = {}
                
                for layer_key in layer_keys:
                    layer_idx = int(layer_key.split('_')[1])
                    layer_retrieval_results = {}
                    
                    for qid in retrieval_results_per_layer:
                        if layer_key in retrieval_results_per_layer[qid]:
                            layer_retrieval_results[qid] = retrieval_results_per_layer[qid][layer_key]
                    
                    if layer_retrieval_results:
                        layer_ndcg, _, layer_recall, layer_precision = evaluator.evaluate(qrels, layer_retrieval_results, ks)
                        ndcg_per_layer[layer_key] = layer_ndcg
                        print(f'\nLayer {layer_idx} NDCG:\n', json.dumps(layer_ndcg, indent=2))
        
        if args.save_retrieval_results and args.per_layer_analysis:
            output_data = {'all_layers': ndcg}
            
            if all_timing_info:
                avg_forward_time = sum(t['forward_pass_time'] for t in all_timing_info) / len(all_timing_info)
                avg_score_time = sum(t['score_documents_time'] for t in all_timing_info) / len(all_timing_info)
                output_data['average_timing'] = {
                    'forward_pass_time': avg_forward_time,
                    'score_documents_time': avg_score_time,
                    'num_queries': len(all_timing_info)
                }
                print(f'\n[ICR Timing] Average - Forward pass: {avg_forward_time:.4f}s, Score documents: {avg_score_time:.4f}s (over {len(all_timing_info)} queries)')
            
            if args.per_layer_analysis and 'retrieval_results_per_layer' in locals() and ndcg_per_layer is not None:
                output_data['per_layer'] = ndcg_per_layer
                output_data['retrieval_results_per_layer'] = retrieval_results_per_layer
            
            per_layer_output = per_layer_output_file
            if args.data in NO_TITLE_DATASETS:
                per_layer_output = per_layer_output.replace('.json', '_no_title.json')
            if args.debug:
                per_layer_output = per_layer_output.replace('.json', '_debug.json')
            if args.no_rerank:
                per_layer_output = per_layer_output.replace('.json', '_{}.json'.format(args.retriever))
            if args.reranker == 'rankgpt':
                per_layer_output = per_layer_output.replace('.json', '_fcr_{}.json'.format(np.mean(format_correct_rates)))
            
            if args.max_layer is not None and args.aggregation_start_layer is not None:
                per_layer_output = per_layer_output.replace('.json', '_max_layer_{}_start_layer_{}.json'.format(args.max_layer, args.aggregation_start_layer))
            
            json.dump(output_data, open(per_layer_output, 'w'), indent=2)
            print(f'\nSaved analysis results to {per_layer_output}')

