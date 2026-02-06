import json
from pyserini.search import LuceneSearcher, get_topics, get_qrels


def bm25_retrieve_beir(task, K):
    print('Retrieving top-{} using BM25 for task:{}'.format(K, task))
    topics = get_topics('beir-v1.0.0-{}-test'.format(task))
    qrels = get_qrels('beir-v1.0.0-{}-test'.format(task))
    searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-{}.flat'.format(task))

    ICR_data = []
    for topic_idx in list(topics.keys()):
        topic = topics[topic_idx]
        query = topic['title']
        hits = searcher.search(query, k=K)
        hit_items = 0
        _sample = {
            "idx": str(topic_idx),
            "question": query,
            "paragraphs":[],
        }
        for hit in hits:
            doc_id = hit.docid
            _doc_json = json.loads(searcher.doc(doc_id).raw())
            _is_support = False
            qrel_key_type = type(list(qrels[topic_idx].keys())[0])
            if qrel_key_type == int:
                _doc_id = int(doc_id)
            else:
                _doc_id = doc_id

            if _doc_id in qrels[topic_idx].keys():
                if int(qrels[topic_idx][_doc_id]) > 0:
                    _is_support = True

            if _is_support:
                hit_items += 1
            _sample['paragraphs'].append({
            'idx': _doc_json['_id'],
            'title': _doc_json['title'],
            'paragraph_text': _doc_json['text'],
            'is_supporting': _is_support,
        })
            
        _sample['num_gold_docs'] = hit_items
        if hit_items > 0:
            ICR_data.append(_sample)
    output_file_name = '../retriever_outpout/icr_beir_{}_bm25_top_{}.json'.format(task, K)
    with open(output_file_name, 'w') as f:
        json.dump(ICR_data, f, indent=2)
    print('Saved retrieval results to ', output_file_name)


def bm25_retrieve_trec_dl(task, K):
    if task == 'trec-dl-19':
        index_name = 'msmarco-v1-passage'
        topic_name = 'dl19-passage'
        qrels_name = 'dl19-passage'
    elif task == 'trec-dl-20':
        index_name = 'msmarco-v1-passage'
        topic_name_candidates = ['dl20-passage', 'dl20']
        qrels_name_candidates = ['dl20-passage', 'trec-dl-20-passage', 'dl20']
    else:
        raise ValueError('Unsupported TREC-DL task: {}'.format(task))


    if task == 'trec-dl-20':
        last_err = None
        topics = None
        for name in topic_name_candidates:
            try:
                topics = get_topics(name)
                topic_name = name
                break
            except ValueError as e:
                last_err = e
        if topics is None:
            raise last_err

        qrels = None
        for name in qrels_name_candidates:
            try:
                qrels = get_qrels(name)
                qrels_name = name
                break
            except ValueError:
                continue
        if qrels is None:
            raise ValueError('No valid qrels name found for trec-dl-20 (tried: {})'.format(qrels_name_candidates))
    else:
        topics = get_topics(topic_name)
        qrels = get_qrels(qrels_name)
    searcher = LuceneSearcher.from_prebuilt_index(index_name)

    ICR_data = []
    for topic_idx in list(topics.keys()):
        topic = topics[topic_idx]
        query = topic['title'] if 'title' in topic else topic['query']
        hits = searcher.search(query, k=K)

        _sample = {
            "idx": str(topic_idx),
            "question": query,
            "paragraphs": [],
        }

        hit_items = 0
        qrel_topic_key_type = type(list(qrels.keys())[0])

        for hit in hits:
            doc_id = hit.docid
            _doc_json = json.loads(searcher.doc(doc_id).raw())

            if qrel_topic_key_type is int:
                qid_key = int(topic_idx)
            else:
                qid_key = str(topic_idx)

            _is_support = False
            if qid_key in qrels:
                qrel_doc_keys = list(qrels[qid_key].keys())
                if len(qrel_doc_keys) > 0:
                    qrel_doc_key_type = type(qrel_doc_keys[0])
                    if qrel_doc_key_type is int:
                        doc_key = int(doc_id)
                    else:
                        doc_key = str(doc_id)
                else:
                    doc_key = str(doc_id)

                if doc_key in qrels[qid_key] and int(qrels[qid_key][doc_key]) > 0:
                    _is_support = True
                    hit_items += 1

            contents = _doc_json.get('contents', '')
            pseudo_title = _doc_json.get('title', '') or contents[:80]

            _sample['paragraphs'].append({
                'idx': _doc_json.get('id', doc_id),
                'title': pseudo_title,
                'paragraph_text': contents,
                'is_supporting': _is_support,
            })

        _sample['num_gold_docs'] = hit_items
        if hit_items > 0:
            ICR_data.append(_sample)

    output_file_name = '../retriever_outpout/icr_trec_dl_{}_bm25_top_{}.json'.format(task, K)
    with open(output_file_name, 'w') as f:
        json.dump(ICR_data, f, indent=2)
    print('Saved retrieval results to ', output_file_name)


for task in ['trec-covid', 'nfcorpus', 'dbpedia-entity', 'scifact', 'scidocs', 'fiqa', 'fever', 'climate-fever', 'nq']:
    bm25_retrieve_beir(task, 100)

for task in ['trec-dl-19', 'trec-dl-20']:
    bm25_retrieve_trec_dl(task, 100)