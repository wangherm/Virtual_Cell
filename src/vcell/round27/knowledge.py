"""Read-only local GO/Reactome knowledge with verified versions and typed relations."""
from collections import Counter,defaultdict
from pathlib import Path
from ..model_agent.artifacts import read
from ..round11 import freeze
from ..utils import file_sha256
from ..evolution.identity import canonical


def build(root,repo,identifiers):
    root=Path(root);assets=Path(repo)/'assets/function';sources=read(assets/'sources.json')
    aliases={str(g):canonical(str(g)) for g in sorted(identifiers)}
    wanted={g for g in aliases.values() if g};terms=defaultdict(list);neighbors=defaultdict(Counter)
    for source in sources['files']:
        path=assets/(source['library']+'.gmt')
        if file_sha256(path)!=source['sha256']:raise ValueError('Functional knowledge asset changed: '+source['library'])
        for line in path.read_text(encoding='utf-8').splitlines():
            fields=line.split('\t')
            if len(fields)<3:continue
            members=set(fields[2:]);hits=members&wanted
            for gene in hits:
                if len(terms[gene])<12:terms[gene].append({'term':fields[0],'library':source['library'],'source_url':source['url']})
                # Broad annotation groups are not informative evidence of direct interaction.
                if 2<=len(members)<=250:neighbors[gene].update(hits-{gene})
    cards={gene:{'symbol':gene,'annotation_available':bool(terms[gene]),'terms':terms[gene],
        'related_genes':[{'gene':g,'shared_small_terms':n,'relation':'shared_pathway_annotation'}
            for g,n in sorted(neighbors[gene].items(),key=lambda x:(-x[1],x[0]))[:12]],
        'interpretation':'Annotation membership and shared pathway, not causal regulatory direction or perturbation effect.'}
        for gene in sorted(wanted)}
    value={'scope':'public_annotation_no_expression_labels','sources':sources,'aliases':aliases,'cards':cards,
        'mapped_symbols':len(cards),'annotated_symbols':sum(c['annotation_available'] for c in cards.values()),
        'new_downloads':False}
    freeze(root/'gene_knowledge.json',value)
    return {'path':str(root/'gene_knowledge.json'),'sha256':file_sha256(root/'gene_knowledge.json'),
        'symbols':len(cards),'annotated_symbols':value['annotated_symbols'],'source_versions':[r['library'] for r in sources['files']]}


def lookup(root,genes,limit=6):
    root=Path(root);path=root/'gene_knowledge.json'
    if not path.exists() or not (root/'domain_catalog.json').exists():return {'state':'not_prepared','genes':[]}
    catalog=read(root/'domain_catalog.json');expected=catalog['gene_knowledge']['sha256']
    if file_sha256(path)!=expected:raise ValueError('Frozen gene knowledge changed')
    value=read(path);result=[]
    for query in list(dict.fromkeys(map(str,genes)))[:limit]:
        symbol=value['aliases'].get(query) or canonical(query)
        card=value['cards'].get(symbol)
        result.append({'query':query,**card} if card else {'query':query,'symbol':symbol,'annotation_available':False})
    return {'state':'available','provenance':{'snapshot_sha256':expected,'sources':value['sources']},'genes':result}
