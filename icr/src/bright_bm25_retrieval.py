import json
import math
import os
import shutil
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher


def compute_ndcg(ranked_relevance, k=100, total_relevant=None):
    """
    Compute NDCG@k for a single query.

    Args:
        ranked_relevance: List of relevance scores (1 for gold doc, 0 otherwise) in rank order
        k: cutoff for NDCG computation

    Returns:
        NDCG@k score
    """
    ranked_relevance = ranked_relevance[:k]

    # Compute DCG
    dcg = 0.0
    for i, rel in enumerate(ranked_relevance):
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1)=0

    # Compute IDCG (ideal DCG - all relevant docs at the top)
    if total_relevant is None:
        total_relevant = sum(ranked_relevance)
    num_for_idcg = min(total_relevant, k)
    idcg = 0.0
    for i in range(int(num_for_idcg)):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def build_bright_index(domain, docs_dataset, index_dir):
    """
    Build a Lucene index for a BRIGHT domain.
    """
    domain_index_dir = os.path.join(index_dir, domain)

    # Skip if index already exists
    if os.path.exists(domain_index_dir) and os.path.isdir(domain_index_dir):
        index_files = os.listdir(domain_index_dir)
        if len(index_files) > 0:
            print(f'Index for {domain} already exists, skipping build...')
            return domain_index_dir

    os.makedirs(domain_index_dir, exist_ok=True)

    print(f'Building index for {domain} with {len(docs_dataset)} documents...')

    # Create JSON documents for indexing
    json_docs_dir = os.path.join(index_dir, f'{domain}_json_docs')
    os.makedirs(json_docs_dir, exist_ok=True)

    # Write documents in JSONL format for pyserini
    jsonl_file = os.path.join(json_docs_dir, 'docs.jsonl')
    with open(jsonl_file, 'w') as f:
        for doc in docs_dataset:
            doc_json = {
                'id': doc['id'],
                'contents': doc['content']
            }
            f.write(json.dumps(doc_json) + '\n')

    # Build the index using pyserini command line (more reliable)
    import subprocess
    cmd = [
        'python', '-m', 'pyserini.index.lucene',
        '--collection', 'JsonCollection',
        '--input', json_docs_dir,
        '--index', domain_index_dir,
        '--generator', 'DefaultLuceneDocumentGenerator',
        '--threads', '4',
        '--storePositions', '--storeDocvectors', '--storeRaw'
    ]
    subprocess.run(cmd, check=True)
    return domain_index_dir


def bm25_retrieve_bright(domain, examples_dataset, docs_dataset, index_dir, output_dir, K=100):
    """
    BM25 retrieval for a BRIGHT domain.
    """
    print(f'\n{"="*60}')
    print(f'Retrieving top-{K} using BM25 for BRIGHT domain: {domain}')
    print(f'{"="*60}')

    # Build or load index
    domain_index_dir = build_bright_index(domain, docs_dataset, index_dir)

    # Load searcher
    searcher = LuceneSearcher(domain_index_dir)

    # Build gold_ids lookup for each query
    gold_ids_map = {}
    excluded_ids_map = {}
    for ex in examples_dataset:
        gold_ids_map[ex['id']] = set(ex['gold_ids'])
        excluded_ids_map[ex['id']] = set(ex.get('excluded_ids', []))

    ICR_data = []
    total_queries = len(examples_dataset)
    filtered_queries = 0
    ndcg_scores_kept = []  # NDCG scores for kept queries
    ndcg_scores_all = []   # NDCG scores for all queries (unfiltered)

    for ex in examples_dataset:
        query_id = ex['id']
        query = ex['query']
        gold_ids = gold_ids_map[query_id]

        # Search
        excluded_ids = excluded_ids_map[query_id]
        hits = searcher.search(query, k=K)
        hits = [hit for hit in hits if hit.docid not in excluded_ids]

        _sample = {
            "idx": str(query_id),
            "question": query,
            "paragraphs": [],
        }

        hit_items = 0
        ranked_relevance = []  # Track relevance for NDCG computation
        for hit in hits:
            doc_id = hit.docid
            doc_raw = searcher.doc(doc_id).raw()
            _doc_json = json.loads(doc_raw)

            _is_support = doc_id in gold_ids
            ranked_relevance.append(1 if _is_support else 0)

            if _is_support:
                hit_items += 1

            # Extract title from doc_id (format: topic/source_chunk.txt)
            # Use the topic as pseudo-title
            title_parts = doc_id.split('/')
            pseudo_title = title_parts[0] if len(title_parts) > 0 else ''

            _sample['paragraphs'].append({
                'idx': doc_id,
                'title': pseudo_title,
                'paragraph_text': _doc_json.get('contents', _doc_json.get('content', '')),
                'is_supporting': _is_support,
            })

        _sample['num_gold_docs'] = hit_items

        # Compute NDCG for all queries (unfiltered)
        ndcg_100 = compute_ndcg(ranked_relevance, k=100, total_relevant=len(gold_ids))
        ndcg_10 = compute_ndcg(ranked_relevance, k=10, total_relevant=len(gold_ids))
        ndcg_scores_all.append((ndcg_100, ndcg_10))

        # Only keep queries with at least one gold doc in top K
        if hit_items > 0:
            ndcg_scores_kept.append((ndcg_100, ndcg_10))
            _sample['ndcg@100'] = ndcg_100
            _sample['ndcg@10'] = ndcg_10
            ICR_data.append(_sample)
        else:
            filtered_queries += 1

    # Log filtering stats
    kept_queries = total_queries - filtered_queries

    # NDCG for kept queries
    avg_ndcg_100_kept = sum(s[0] for s in ndcg_scores_kept) / len(ndcg_scores_kept) if ndcg_scores_kept else 0.0
    avg_ndcg_10_kept = sum(s[1] for s in ndcg_scores_kept) / len(ndcg_scores_kept) if ndcg_scores_kept else 0.0

    # NDCG for all queries (unfiltered)
    avg_ndcg_100_all = sum(s[0] for s in ndcg_scores_all) / len(ndcg_scores_all) if ndcg_scores_all else 0.0
    avg_ndcg_10_all = sum(s[1] for s in ndcg_scores_all) / len(ndcg_scores_all) if ndcg_scores_all else 0.0

    print(f'\n[{domain}] Filtered out {filtered_queries}/{total_queries} queries (no gold docs in top {K})')
    print(f'[{domain}] Kept {kept_queries}/{total_queries} queries ({100*kept_queries/total_queries:.1f}%)')
    print(f'[{domain}] Average NDCG@10  (kept): {avg_ndcg_10_kept:.4f}  (all): {avg_ndcg_10_all:.4f}')
    print(f'[{domain}] Average NDCG@100 (kept): {avg_ndcg_100_kept:.4f}  (all): {avg_ndcg_100_all:.4f}')

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'icr_bright_{domain}_bm25_top_{K}.json')
    with open(output_file, 'w') as f:
        json.dump(ICR_data, f, indent=2)

    print(f'[{domain}] Saved {len(ICR_data)} queries to {output_file}')

    return {
        'domain': domain,
        'total_queries': total_queries,
        'filtered_queries': filtered_queries,
        'kept_queries': kept_queries,
        'avg_ndcg_10_kept': avg_ndcg_10_kept,
        'avg_ndcg_100_kept': avg_ndcg_100_kept,
        'avg_ndcg_10_all': avg_ndcg_10_all,
        'avg_ndcg_100_all': avg_ndcg_100_all,
        'output_file': output_file
    }


def main():
    # Configuration
    K = 100
    INDEX_DIR = './bright_indexes'
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(current_file_dir, '..', 'retriever_outpout')
    OUTPUT_DIR = os.path.normpath(OUTPUT_DIR)

    print('Loading BRIGHT dataset...')
    examples = load_dataset('xlangai/BRIGHT', 'examples')
    docs = load_dataset('xlangai/BRIGHT', 'documents')

    domains = list(examples.keys())
    print(f'Found {len(domains)} domains: {domains}')

    # Process each domain
    results = []
    for domain in domains:
        result = bm25_retrieve_bright(
            domain=domain,
            examples_dataset=examples[domain],
            docs_dataset=docs[domain],
            index_dir=INDEX_DIR,
            output_dir=OUTPUT_DIR,
            K=K
        )
        results.append(result)

    # Print summary
    print('\n' + '='*120)
    print('SUMMARY')
    print('='*120)
    print(f'{"Domain":<25} {"Total":>8} {"Kept":>8} {"Rate":>8} {"NDCG@10":>10} {"NDCG@100":>10} {"NDCG@10":>12} {"NDCG@100":>12}')
    print(f'{"":<25} {"":>8} {"":>8} {"":>8} {"(kept)":>10} {"(kept)":>10} {"(all)":>12} {"(all)":>12}')
    print('-'*120)

    total_all = 0
    kept_all = 0
    weighted_ndcg_10_kept_sum = 0.0
    weighted_ndcg_100_kept_sum = 0.0
    weighted_ndcg_10_all_sum = 0.0
    weighted_ndcg_100_all_sum = 0.0
    for r in results:
        rate = 100 * r['kept_queries'] / r['total_queries']
        print(f'{r["domain"]:<25} {r["total_queries"]:>8} {r["kept_queries"]:>8} {rate:>7.1f}% {r["avg_ndcg_10_kept"]:>10.4f} {r["avg_ndcg_100_kept"]:>10.4f} {r["avg_ndcg_10_all"]:>12.4f} {r["avg_ndcg_100_all"]:>12.4f}')
        total_all += r['total_queries']
        kept_all += r['kept_queries']
        weighted_ndcg_10_kept_sum += r['avg_ndcg_10_kept'] * r['kept_queries']
        weighted_ndcg_100_kept_sum += r['avg_ndcg_100_kept'] * r['kept_queries']
        weighted_ndcg_10_all_sum += r['avg_ndcg_10_all'] * r['total_queries']
        weighted_ndcg_100_all_sum += r['avg_ndcg_100_all'] * r['total_queries']

    print('-'*120)
    overall_rate = 100 * kept_all / total_all
    overall_ndcg_10_kept = weighted_ndcg_10_kept_sum / kept_all if kept_all > 0 else 0.0
    overall_ndcg_100_kept = weighted_ndcg_100_kept_sum / kept_all if kept_all > 0 else 0.0
    overall_ndcg_10_all = weighted_ndcg_10_all_sum / total_all if total_all > 0 else 0.0
    overall_ndcg_100_all = weighted_ndcg_100_all_sum / total_all if total_all > 0 else 0.0
    print(f'{"TOTAL":<25} {total_all:>8} {kept_all:>8} {overall_rate:>7.1f}% {overall_ndcg_10_kept:>10.4f} {overall_ndcg_100_kept:>10.4f} {overall_ndcg_10_all:>12.4f} {overall_ndcg_100_all:>12.4f}')
    print(f'\nAll results saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
