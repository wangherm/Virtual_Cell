"""Versioned public gene knowledge. No perturbation-expression labels are queried."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
import torch

from .qwen import backbone_identity
from .utils import file_sha256, write_json


def cached_request(folder, url, payload):
    """POST retry with immutable, content-addressed request/response records."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    request = {"url": url, "payload": payload}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    path = folder / (key + '.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['request'] != request:
            raise ValueError('Knowledge request cache mismatch')
        return saved['response']
    for attempt in range(5):
        try:
            response = requests.post(url, data=payload, timeout=(15, 90))
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list):
                raise ValueError(f'Expected list from {url}: {str(result)[:160]}')
            write_json(path, {'request': request, 'response': result})
            return result
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            print(f'KNOWLEDGE RETRY {url} attempt={attempt+1}', flush=True)
            time.sleep(min(2 ** attempt, 15))


def resolve_cards(identifiers, records):
    """Ambiguous symbols are unavailable, never guessed using an API relevance score."""
    result = {}
    for query in identifiers:
        candidates = {str(r.get('_id')): r for r in records
                      if r.get('query') == query and not r.get('notfound') and r.get('taxid') == 9606}
        if len(candidates) != 1:
            result[query] = {'id': query, 'symbol': None, 'summary': '',
                             'status': 'missing' if not candidates else 'ambiguous'}
        else:
            r = next(iter(candidates.values()))
            result[query] = {'id': query, 'symbol': r.get('symbol'), 'name': r.get('name', ''),
                             'summary': r.get('summary', ''), 'entrezgene': r.get('entrezgene'),
                             'status': 'mapped', 'source': 'https://mygene.info/v3/query'}
    return result


def annotation_sets(assets):
    sources = json.loads((Path(assets)/'sources.json').read_text())
    sets = []
    for record in sources['files']:
        path = Path(assets)/(record['library']+'.gmt')
        if file_sha256(path) != record['sha256']:
            raise ValueError('Functional annotation checksum mismatch')
        for line in path.read_text(encoding='utf-8').splitlines():
            fields = line.split('\t')
            sets.append((record['library']+':'+fields[0], set(fields[2:])))
    return sets, sources


def build_knowledge(data, output, assets, model_spec, device='cuda', batch_size=8):
    """Build cards, frozen Qwen text embeddings, typed neighbor matrices and pathways."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/'knowledge.npz'
    identity = {'genes': data['genes'].tolist(), 'perturbations': data['perturbations'].tolist(),
                'builder_sha256': file_sha256(__file__),
                'backbone': backbone_identity(model_spec), 'format': 'round8-knowledge-v1',
                'annotation_sources': json.loads((Path(assets)/'sources.json').read_text()),
                'max_text_tokens': 192, 'string_version': '12.0', 'neighbor_limit': 16}
    if (output/'COMPLETE.json').exists():
        manifest = json.loads((output/'COMPLETE.json').read_text())
        if manifest['identity'] != identity:
            raise ValueError('Knowledge identity changed; use a new knowledge directory')
        for name, digest in manifest['files'].items():
            if file_sha256(output/name) != digest:
                raise ValueError('Knowledge content changed: '+name)
        return path
    identifiers = sorted(set(data['perturbations']) | set(data['genes']))
    records = []
    for start in range(0, len(identifiers), 100):
        records += cached_request(output/'requests', 'https://mygene.info/v3/query',
            {'q': ','.join(identifiers[start:start+100]), 'scopes': 'symbol,ensembl.gene',
             'species': '9606', 'fields': 'symbol,name,summary,entrezgene,taxid', 'size': '5'})
        print(f'KNOWLEDGE CARDS {min(start+100,len(identifiers))}/{len(identifiers)}', flush=True)
    cards = resolve_cards(identifiers, records)
    terms, sources = annotation_sets(assets)
    symbols = [cards[g]['symbol'] for g in data['genes']]
    targets = [cards[p]['symbol'] for p in data['perturbations']]
    known = sum(bool(cards[p].get('summary')) for p in data['perturbations'])
    if known < max(2, int(.5*len(targets))):
        raise ValueError(f'Too few target summaries ({known}/{len(targets)}); inspect cards/API, no fake fallback')
    for card in cards.values():
        card['terms'] = [name for name, members in terms if card['symbol'] in members][:12]
    write_json(output/'cards.json', cards)
    # Non-directional STRING functional associations. Scores are confidence, not effect strength.
    ppi = []
    unique_symbols = sorted({s for s in targets if s})
    for start in range(0, len(unique_symbols), 50):
        ppi += cached_request(output/'requests', 'https://version-12-0.string-db.org/api/json/interaction_partners',
            {'identifiers': '\r'.join(unique_symbols[start:start+50]), 'species': '9606',
             'required_score': '700', 'limit': '16', 'network_type': 'functional',
             'caller_identity': 'Virtual_Cell_round8'})
        print(f'KNOWLEDGE EDGES {min(start+50,len(unique_symbols))}/{len(unique_symbols)}', flush=True)
    write_json(output/'string_edges.json', ppi)
    relations = np.zeros((len(targets), 3, len(symbols)), dtype=np.float32)
    for pi, target in enumerate(targets):
        if not target:
            continue
        for name, members in terms:
            if target in members:
                channel = 0 if name.startswith('GO_') else 1
                relations[pi, channel] += np.array([s in members and s != target for s in symbols])
        for edge in ppi:
            if edge.get('preferredName_A') == target:
                for gi, symbol in enumerate(symbols):
                    if symbol and symbol == edge.get('preferredName_B') and symbol != target:
                        relations[pi, 2, gi] = max(relations[pi, 2, gi], float(edge['score']))
        for channel in range(3):
            row = relations[pi, channel]
            keep = np.argsort(-row, kind='stable')[:16]
            mask = np.ones(len(row), bool); mask[keep] = False; row[mask] = 0
            row /= max(float(row.sum()), 1e-12)
    if not (relations[:, 2] > 0).any():
        raise ValueError('No STRING links map to measured genes; inspect mapping, do not silently omit network')
    modules, module_names = [], []
    # Annotation-only, fixed order and coverage. No response-label selection.
    for name, members in sorted(terms):
        if not name.startswith('Reactome_'):
            continue
        v = np.array([s in members for s in symbols], dtype=np.float32)
        if 5 <= v.sum() <= 150:
            modules.append(v/v.sum()); module_names.append(name)
        if len(modules) == 64:
            break
    if not modules:
        raise ValueError('No measured Reactome modules')
    from transformers import AutoModel, AutoTokenizer
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_spec['model_id'], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    backbone = AutoModel.from_pretrained(model_spec['model_id'], local_files_only=True,
                                        dtype=dtype, attn_implementation='sdpa').to(device).eval()
    texts = []
    for g in identifiers:
        c = cards[g]
        texts.append(f"Human gene {g}. {c.get('name','')}. {c['summary']} Biological processes: "
                     + '; '.join(c['terms']) + '. Missing annotation is unknown, not no function.')
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                               max_length=192, return_tensors='pt').to(device)
            hidden = backbone(**tokens).last_hidden_state.float()
            # Last real token of causal text encoder summarizes preceding text.
            pooled = hidden[torch.arange(len(hidden), device=device), tokens.attention_mask.sum(1)-1]
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            encoded.append(pooled.cpu().numpy())
            if start % (batch_size*20) == 0:
                print(f'KNOWLEDGE ENCODE {start}/{len(texts)}', flush=True)
    vectors = np.concatenate(encoded)
    lookup = {g:i for i,g in enumerate(identifiers)}
    np.savez_compressed(path, genes=data['genes'], perturbations=data['perturbations'],
        semantic=vectors[[lookup[p] for p in data['perturbations']]],
        gene_semantic=vectors[[lookup[g] for g in data['genes']]], relations=relations,
        modules=np.asarray(modules), module_names=np.array(module_names, dtype='U'))
    del backbone
    if device == 'cuda':
        torch.cuda.empty_cache()
    write_json(output/'audit.json', {'target_summaries': known, 'targets': len(targets),
        'relation_types': ['shared_GO', 'shared_Reactome', 'STRING_functional_association'],
        'targets_with_relation': (relations.sum(2)>0).sum(0).tolist(),
        'modules': module_names, 'sources': sources,
        'mechanism': 'One-hop typed associations, no invented signed regulatory edges or verbal reasoning labels',
        'pretraining_overlap': 'Public text/backbone knowledge may include held-out biology; exploratory',
        'test_evaluated': False})
    names = ['knowledge.npz', 'cards.json', 'string_edges.json', 'audit.json']
    names += [p.relative_to(output).as_posix() for p in sorted((output/'requests').glob('*.json'))]
    write_json(output/'COMPLETE.json', {'identity': identity, 'files': {n:file_sha256(output/n) for n in names}})
    return path


def load_knowledge(path, data):
    with np.load(path, allow_pickle=False) as f:
        values = {k:f[k] for k in f.files}
    for k in ('genes', 'perturbations'):
        if not np.array_equal(values[k], data[k]):
            raise ValueError('Knowledge alignment mismatch: '+k)
    n, g = len(data['perturbations']), len(data['genes'])
    if values['semantic'].ndim != 2 or len(values['semantic']) != n or values['gene_semantic'].shape != (g, values['semantic'].shape[1]):
        raise ValueError('Invalid semantic matrices')
    if values['relations'].shape != (n, 3, g) or values['modules'].ndim != 2 or values['modules'].shape[1] != g:
        raise ValueError('Invalid relation/module shapes')
    for k in ('semantic', 'gene_semantic', 'relations', 'modules'):
        if not np.isfinite(values[k]).all():
            raise ValueError('Nonfinite knowledge: '+k)
    return values
