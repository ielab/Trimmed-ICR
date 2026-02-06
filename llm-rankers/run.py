import logging
import ir_datasets
from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.rankers import SearchResult
from llmrankers.pointwise import PointwiseLlmRanker, MonoT5LlmRanker
from llmrankers.setwise import SetwiseLlmRanker, OpenAiSetwiseLlmRanker, ICRSetwiseRanker
from llmrankers.pairwise import PairwiseLlmRanker, DuoT5LlmRanker, OpenAiPairwiseLlmRanker
from llmrankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker, ICRListwiseRanker
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
random.seed(929)
logger = logging.getLogger(__name__)


def parse_args(parser, commands):
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    parser.parse_args(split_argv[0], namespace=args)
    for argv in split_argv[1:]:
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'w') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main(args):
    if args.pointwise:
        if 'monot5' in args.run.model_name_or_path:
            ranker = MonoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                     tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                     device=args.run.device,
                                     cache_dir=args.run.cache_dir,
                                     method=args.pointwise.method,
                                     batch_size=args.pointwise.batch_size)
        else:
            ranker = PointwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                        tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                        device=args.run.device,
                                        cache_dir=args.run.cache_dir,
                                        method=args.pointwise.method,
                                        batch_size=args.pointwise.batch_size)

    elif args.setwise:
        if args.run.openai_key:
            ranker = OpenAiSetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                            api_key=args.run.openai_key,
                                            num_child=args.setwise.num_child,
                                            method=args.setwise.method,
                                            k=args.setwise.k)
        elif args.run.scoring == 'attention':
            try:
                import os
                icr_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icr')
                if not os.path.exists(icr_base_path):
                    raise ImportError(f"ICR base path not found: {icr_base_path}")
                
                if icr_base_path not in sys.path:
                    sys.path.insert(0, icr_base_path)
                
                from src.in_context_reranker import InContextReranker
                
            except Exception as e:
                raise ImportError(
                    f"Failed to import InContextReranker. Please ensure the 'icr' directory is available at {icr_base_path}. Error: {e}"
                )
            
            retrieval_type = args.run.retrieval_type if hasattr(args.run, 'retrieval_type') and args.run.retrieval_type else 'IE'
            prompt_template = args.run.prompt_template if hasattr(args.run, 'prompt_template') and args.run.prompt_template else 'instruct'
            reverse_doc_order = args.run.reverse_doc_order if hasattr(args.run, 'reverse_doc_order') and args.run.reverse_doc_order else False
            scoring_strategy = args.run.scoring_strategy if hasattr(args.run, 'scoring_strategy') and args.run.scoring_strategy else 'masked_NA_calibration'
            max_layer = args.run.max_layer if hasattr(args.run, 'max_layer') and args.run.max_layer is not None else None
            aggregation_start_layer = args.run.aggregation_start_layer if hasattr(args.run, 'aggregation_start_layer') and args.run.aggregation_start_layer is not None else None
            
            print(f"[ICRSetwiseRanker] Initializing InContextReranker with:")
            print(f"  - base_llm_name: {args.run.model_name_or_path}")
            print(f"  - prompt_template: {prompt_template}")
            print(f"  - scoring_strategy: {scoring_strategy}")
            print(f"  - retrieval_type: {retrieval_type}")
            print(f"  - use_fa2: {args.run.use_fa2}")
            print(f"  - reverse_doc_order: {reverse_doc_order}")
            print(f"  - max_layer: {max_layer}")
            print(f"  - aggregation_start_layer: {aggregation_start_layer}")
            
            icr_model = InContextReranker(
                base_llm_name=args.run.model_name_or_path,
                prompt_template=prompt_template,
                prompt_prefix='',
                prompt_suffix='',
                scoring_strategy=scoring_strategy,
                use_fa2=args.run.use_fa2,
                retrieval_type=retrieval_type,
                sliding_window_size=-1,
                sliding_window_stride=10,
                reverse_doc_order=reverse_doc_order,
                use_double_query=False,
                max_layer=max_layer,
                aggregation_start_layer=aggregation_start_layer,
                enable_timing=False
            )
            
            if hasattr(icr_model, 'prompt_separator'):
                print(f"[ICRSetwiseRanker] ICR initialized successfully. prompt_separator type: {type(icr_model.prompt_separator)}, value: {repr(icr_model.prompt_separator)[:50]}")
            else:
                print(f"[ICRSetwiseRanker] Warning: ICR model does not have prompt_separator attribute!")
            
            ranker = ICRSetwiseRanker(
                icr_model=icr_model,
                num_child=args.setwise.num_child,
                k=args.setwise.k,
                method=args.setwise.method,
                scoring_strategy=scoring_strategy
            )
        else:
            ranker_kwargs = {
                'model_name_or_path': args.run.model_name_or_path,
                'tokenizer_name_or_path': args.run.tokenizer_name_or_path,
                'device': args.run.device,
                'cache_dir': args.run.cache_dir,
                'num_child': args.setwise.num_child,
                'scoring': args.run.scoring,
                'method': args.setwise.method,
                'num_permutation': args.setwise.num_permutation,
                'k': args.setwise.k,
                'debug_log_dir': args.run.debug_log_dir
            }
            ranker = SetwiseLlmRanker(**ranker_kwargs)

    elif args.pairwise:
        if args.pairwise.method != 'allpair':
            args.pairwise.batch_size = 2
            logger.info(f'Setting batch_size to 2.')

        if args.run.openai_key:
            ranker = OpenAiPairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             method=args.pairwise.method,
                                             k=args.pairwise.k)

        elif 'duot5' in args.run.model_name_or_path:
            ranker = DuoT5LlmRanker(model_name_or_path=args.run.model_name_or_path,
                                    tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                    device=args.run.device,
                                    cache_dir=args.run.cache_dir,
                                    method=args.pairwise.method,
                                    batch_size=args.pairwise.batch_size,
                                    k=args.pairwise.k)
        else:
            ranker = PairwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       method=args.pairwise.method,
                                       batch_size=args.pairwise.batch_size,
                                       k=args.pairwise.k)

    elif args.listwise:
        if args.run.openai_key:
            ranker = OpenAiListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                             api_key=args.run.openai_key,
                                             window_size=args.listwise.window_size,
                                             step_size=args.listwise.step_size,
                                             num_repeat=args.listwise.num_repeat)
        elif args.run.scoring == 'attention':
            try:
                import os
                icr_base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icr')
                if not os.path.exists(icr_base_path):
                    raise ImportError(f"ICR base path not found: {icr_base_path}")
                
                if icr_base_path not in sys.path:
                    sys.path.insert(0, icr_base_path)
                
                from src.in_context_reranker import InContextReranker
                
            except Exception as e:
                raise ImportError(
                    f"Failed to import InContextReranker. Please ensure the 'icr' directory is available at {icr_base_path}. Error: {e}"
                )
            
            retrieval_type = args.run.retrieval_type if hasattr(args.run, 'retrieval_type') and args.run.retrieval_type else 'IE'
            prompt_template = args.run.prompt_template if hasattr(args.run, 'prompt_template') and args.run.prompt_template else 'instruct'
            reverse_doc_order = args.run.reverse_doc_order if hasattr(args.run, 'reverse_doc_order') and args.run.reverse_doc_order else False
            scoring_strategy = args.run.scoring_strategy if hasattr(args.run, 'scoring_strategy') and args.run.scoring_strategy else 'masked_NA_calibration'
            max_layer = args.run.max_layer if hasattr(args.run, 'max_layer') and args.run.max_layer is not None else None
            aggregation_start_layer = args.run.aggregation_start_layer if hasattr(args.run, 'aggregation_start_layer') and args.run.aggregation_start_layer is not None else None
            
            print(f"[ICRListwiseRanker] Initializing InContextReranker with:")
            print(f"  - base_llm_name: {args.run.model_name_or_path}")
            print(f"  - prompt_template: {prompt_template}")
            print(f"  - scoring_strategy: {scoring_strategy}")
            print(f"  - retrieval_type: {retrieval_type}")
            print(f"  - use_fa2: {args.run.use_fa2}")
            print(f"  - reverse_doc_order: {reverse_doc_order}")
            print(f"  - max_layer: {max_layer}")
            print(f"  - aggregation_start_layer: {aggregation_start_layer}")
            
            icr_model = InContextReranker(
                base_llm_name=args.run.model_name_or_path,
                prompt_template=prompt_template,
                prompt_prefix='',
                prompt_suffix='',
                scoring_strategy=scoring_strategy,
                use_fa2=args.run.use_fa2,
                retrieval_type=retrieval_type,
                sliding_window_size=-1,
                sliding_window_stride=10,
                reverse_doc_order=reverse_doc_order,
                use_double_query=False,
                max_layer=max_layer,
                aggregation_start_layer=aggregation_start_layer,
                enable_timing=False
            )
            
            if hasattr(icr_model, 'prompt_separator'):
                print(f"[ICRListwiseRanker] ICR initialized successfully. prompt_separator type: {type(icr_model.prompt_separator)}, value: {repr(icr_model.prompt_separator)[:50]}")
            else:
                print(f"[ICRListwiseRanker] Warning: ICR model does not have prompt_separator attribute!")
            
            ranker = ICRListwiseRanker(
                icr_model=icr_model,
                window_size=args.listwise.window_size,
                step_size=args.listwise.step_size,
                num_repeat=args.listwise.num_repeat,
                scoring_strategy=scoring_strategy
            )
        else:
            ranker = ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                       tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                       device=args.run.device,
                                       cache_dir=args.run.cache_dir,
                                       window_size=args.listwise.window_size,
                                       step_size=args.listwise.step_size,
                                       scoring=args.run.scoring,
                                       num_repeat=args.listwise.num_repeat)
    else:
        raise ValueError('Must specify either --pointwise, --setwise, --pairwise or --listwise.')

    query_map = {}
    if args.run.ir_dataset_name is not None:
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        for query in dataset.queries_iter():
            qid = query.query_id
            text = query.text
            query_map[qid] = ranker.truncate(text, args.run.query_length)
        dataset = ir_datasets.load(args.run.ir_dataset_name)
        docstore = dataset.docs_store()
    else:
        topics = get_topics(args.run.pyserini_index+'-test')
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index+'.flat')

    logger.info(f'Loading first stage run from {args.run.run_path}.')
    first_stage_rankings = []
    with open(args.run.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= args.run.hits:
                continue
            if args.run.ir_dataset_name is not None:
                text = docstore.get(docid).text
                if 'title' in dir(docstore.get(docid)):
                    text = f'{docstore.get(docid).title} {text}'
            else:
                data = json.loads(docstore.doc(docid).raw())
                text = data['text']
                if 'title' in data:
                    text = f'{data["title"]} {text}'
            text = ranker.truncate(text, args.run.passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:args.run.hits]))

    reranked_results = []
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if args.run.limit is not None:
        first_stage_rankings = first_stage_rankings[:args.run.limit]
        logger.info(f'[DEBUG] Limited to {args.run.limit} queries for debugging.')
    
    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')
        
        if hasattr(ranker, 'current_query_id'):
            ranker.current_query_id = qid
        
        reranked_results.append((qid, query, ranker.rerank(query, ranking)))
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
    toc = time.time()

    avg_comparisons = total_comparisons/len(reranked_results)
    avg_prompt_tokens = total_prompt_tokens/len(reranked_results)
    avg_completion_tokens = total_completion_tokens/len(reranked_results)
    avg_time_per_query = (toc-tic)/len(reranked_results)
    
    print(f'Avg comparisons: {avg_comparisons}')
    print(f'Avg prompt tokens: {avg_prompt_tokens}')
    print(f'Avg completion tokens: {avg_completion_tokens}')
    print(f'Avg time per query: {avg_time_per_query}')

    import os
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.run.pyserini_index:
        dataset_name = args.run.pyserini_index.replace('beir-v1.0.0-', '').replace('.flat', '')
    elif args.run.ir_dataset_name:
        dataset_name = args.run.ir_dataset_name
    else:
        dataset_name = 'unknown'
    
    model_name = args.run.model_name_or_path.replace('/', '__').replace(':', '__')
    
    def abbreviate_scoring(scoring):
        abbrev_map = {
            'attention': 'attn',
            'generation': 'gen',
            'likelihood': 'lik'
        }
        return abbrev_map.get(scoring, scoring)
    
    def abbreviate_method(method):
        abbrev_map = {
            'heapsort': 'heap',
            'bubblesort': 'bubble'
        }
        return abbrev_map.get(method, method)
    
    def abbreviate_scoring_strategy(strategy):
        abbrev_map = {
            'masked_NA_calibration': 'mNAcal',
            'query_last': 'qlast',
            'attention_sorting': 'attn_sort',
            'NA_only': 'NAonly',
            'NA_calibration_no_agg': 'NAcal_noagg'
        }
        return abbrev_map.get(strategy, strategy)
    
    def abbreviate_prompt_template(template):
        abbrev_map = {
            'instruct': 'instr',
            'simple': 'simp',
            'simple_instruct': 'simp_instr'
        }
        return abbrev_map.get(template, template)
    
    filename_parts = [dataset_name, model_name, abbreviate_scoring(args.run.scoring)]
    
    if args.setwise:
        filename_parts.append(abbreviate_method(args.setwise.method))
        if args.run.scoring == 'attention':
            retrieval_type = args.run.retrieval_type if hasattr(args.run, 'retrieval_type') and args.run.retrieval_type else 'IE'
            prompt_template = args.run.prompt_template if hasattr(args.run, 'prompt_template') and args.run.prompt_template else 'instruct'
            reverse_doc_order = args.run.reverse_doc_order if hasattr(args.run, 'reverse_doc_order') and args.run.reverse_doc_order else False
            scoring_strategy = args.run.scoring_strategy if hasattr(args.run, 'scoring_strategy') and args.run.scoring_strategy else 'masked_NA_calibration'
            max_layer = args.run.max_layer if hasattr(args.run, 'max_layer') and args.run.max_layer is not None else None
            aggregation_start_layer = args.run.aggregation_start_layer if hasattr(args.run, 'aggregation_start_layer') and args.run.aggregation_start_layer is not None else None
            
            filename_parts.append(abbreviate_scoring_strategy(scoring_strategy))
            filename_parts.append(retrieval_type)
            filename_parts.append(abbreviate_prompt_template(prompt_template))
            if reverse_doc_order:
                filename_parts.append('rev')
            if max_layer is not None:
                filename_parts.append(f'maxL{max_layer}')
            if aggregation_start_layer is not None:
                filename_parts.append(f'aggL{aggregation_start_layer}')
    elif args.listwise:
        filename_parts.append(f'w{args.listwise.window_size}')
        filename_parts.append(f's{args.listwise.step_size}')
        filename_parts.append(f'r{args.listwise.num_repeat}')
        if args.run.scoring == 'attention':
            retrieval_type = args.run.retrieval_type if hasattr(args.run, 'retrieval_type') and args.run.retrieval_type else 'IE'
            prompt_template = args.run.prompt_template if hasattr(args.run, 'prompt_template') and args.run.prompt_template else 'instruct'
            reverse_doc_order = args.run.reverse_doc_order if hasattr(args.run, 'reverse_doc_order') and args.run.reverse_doc_order else False
            scoring_strategy = args.run.scoring_strategy if hasattr(args.run, 'scoring_strategy') and args.run.scoring_strategy else 'masked_NA_calibration'
            max_layer = args.run.max_layer if hasattr(args.run, 'max_layer') and args.run.max_layer is not None else None
            aggregation_start_layer = args.run.aggregation_start_layer if hasattr(args.run, 'aggregation_start_layer') and args.run.aggregation_start_layer is not None else None
            
            filename_parts.append(abbreviate_scoring_strategy(scoring_strategy))
            filename_parts.append(retrieval_type)
            filename_parts.append(abbreviate_prompt_template(prompt_template))
            if reverse_doc_order:
                filename_parts.append('rev')
            if max_layer is not None:
                filename_parts.append(f'maxL{max_layer}')
            if aggregation_start_layer is not None:
                filename_parts.append(f'aggL{aggregation_start_layer}')
    
    stats_filename = '_'.join(filename_parts) + '_stats.txt'
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f'Avg comparisons: {avg_comparisons}\n')
        f.write(f'Avg prompt tokens: {avg_prompt_tokens}\n')
        f.write(f'Avg completion tokens: {avg_completion_tokens}\n')
        f.write(f'Avg time per query: {avg_time_per_query}\n')
    
    print(f'Statistics saved to: {stats_path}')

    write_run_file(args.run.save_path, reranked_results, 'LLMRankers')
    
    if hasattr(ranker, 'close_debug_log'):
        ranker.close_debug_log()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--ir_dataset_name', type=str, default=None)
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood', 'attention'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    run_parser.add_argument('--use_fa2', action='store_true', default=True, help='Use Flash Attention 2 for attention scoring (default: True)')
    run_parser.add_argument('--no_fa2', dest='use_fa2', action='store_false', help='Disable Flash Attention 2 for attention scoring')
    run_parser.add_argument('--attention_start_layer', type=int, default=None, help='Start layer for attention aggregation (None means last 4 layers)')
    run_parser.add_argument('--attention_end_layer', type=int, default=None, help='End layer for attention aggregation (None means last layer)')
    run_parser.add_argument('--retrieval_type', type=str, default='IE', choices=['QA', 'IE'], help='Retrieval type for ICR-style prompt (QA or IE)')
    run_parser.add_argument('--prompt_template', type=str, default='instruct', choices=['instruct', 'simple', 'simple_instruct'], help='Prompt template for ICR-style attention scoring')
    run_parser.add_argument('--scoring_strategy', type=str, default='masked_NA_calibration', choices=['query_last', 'attention_sorting', 'NA_only', 'NA_calibration_no_agg', 'masked_NA_calibration'], help='ICR scoring strategy (for ICRSetwiseRanker)')
    run_parser.add_argument('--reverse_doc_order', action='store_true', help='Reverse document order (most relevant at the end, for ICRSetwiseRanker)')
    run_parser.add_argument('--max_layer', type=int, default=None, help='Maximum layer index for ICR forward pass (None means use all layers)')
    run_parser.add_argument('--aggregation_start_layer', type=int, default=None, help='Start layer index for ICR attention aggregation (None means start from layer 0)')
    run_parser.add_argument('--limit', type=int, default=None, help='Limit the number of queries to process (for debugging). If None, process all queries.')
    run_parser.add_argument('--debug_log_dir', type=str, default=None, help='Directory to save debug logs for each compare call. If None, no debug logs will be saved.')

    pointwise_parser = commands.add_parser('pointwise')
    pointwise_parser.add_argument('--method', type=str, default='yes_no',
                                  choices=['qlm', 'yes_no'])
    pointwise_parser.add_argument('--batch_size', type=int, default=2)

    pairwise_parser = commands.add_parser('pairwise')
    pairwise_parser.add_argument('--method', type=str, default='allpair',
                                 choices=['allpair', 'heapsort', 'bubblesort'])
    pairwise_parser.add_argument('--batch_size', type=int, default=2)
    pairwise_parser.add_argument('--k', type=int, default=10)

    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)

    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)

    if args.run.ir_dataset_name is not None and args.run.pyserini_index is not None:
        raise ValueError('Must specify either --ir_dataset_name or --pyserini_index, not both.')

    arg_dict = vars(args)
    if arg_dict['run'] is None or sum(arg_dict[arg] is not None for arg in arg_dict) != 2:
        raise ValueError('Need to set --run and can only set one of --pointwise, --pairwise, --setwise, --listwise')
    main(args)
