"""Executable source diagnostics; no hidden confirmation/outer labels are opened."""
from pathlib import Path
import numpy as np
import pandas as pd
from ..round22.evaluator import inputs
from ..model_agent.artifacts import read
from ..utils import file_sha256
from ..round11 import freeze
from .runtime import source_records


def source_probe(pool,kind,destination):
    if kind not in ('reference_coverage','context_losses','target_losses','strong_response'):
        raise ValueError('Unknown source diagnostic')
    pool=Path(pool);dest=Path(destination)
    if dest.exists():
        old=read(dest)
        if old['probe']!=kind:raise ValueError('Frozen diagnostic changed')
        for p,sha in old['evidence_files'].items():
            if file_sha256(p)!=sha:raise ValueError('Source diagnostic evidence changed')
        return old
    results=[];hashes={};folds=read(pool/'folds.json')['search']
    for record in source_records(pool):
        if record['status']!='valid':continue
        folder=Path(record['directory']);csv=folder/'scores.csv'
        if file_sha256(csv)!=record['files']['scores.csv']:raise ValueError('Source score payload changed')
        frame=pd.read_csv(csv);hashes[str(csv)]=file_sha256(csv)
        if kind=='reference_coverage':
            values=[{'context':str(context),'rows':len(f),'legal_reference_rows':int(f.reference_available.sum()),
                'targets':int(f.target.nunique())} for context,f in frame.groupby('context')]
        elif kind in ('context_losses','target_losses'):
            cols=['context'] if kind=='context_losses' else ['context','target']
            values=frame.groupby(cols)[['mse','zero_mse','strong_mse','prediction_norm','truth_norm']].mean().reset_index().to_dict('records')
        else:
            values=[]
            for fold in folds:
                d,g,_=inputs(fold);npz=folder/(fold['name']+'.npz')
                if file_sha256(npz)!=record['files'][npz.name]:raise ValueError('Source prediction changed')
                with np.load(npz,allow_pickle=False) as f:prediction=f['prediction'].astype(np.float64)
                mask=g['strong_mask'];truth=g['truth'].astype(np.float64)
                values.append({'fold':fold['name'],'strong_entries':int(mask.sum()),
                    'strong_truth_rms':float(np.sqrt(np.square(truth[mask]).mean())) if mask.any() else None,
                    'strong_prediction_rms':float(np.sqrt(np.square(prediction[mask]).mean())) if mask.any() else None,
                    'strong_error':float(np.square(prediction[mask]-truth[mask]).mean()) if mask.any() else None,
                    'support_truth_cosine_mean':float(np.mean(np.sum(d['query_support']*truth,1)/np.maximum(
                        np.linalg.norm(d['query_support'],axis=1)*np.linalg.norm(truth,axis=1),1e-12)))})
                hashes[str(npz)]=file_sha256(npz)
                for p,key in ((fold['input'],'input_sha256'),(str(Path(fold['private'])/'grade.npz'),'grade_sha256')):hashes[p]=fold[key]
        # Convert missing numeric statistics to JSON null, without teaching a false zero.
        import json
        cleaned=json.loads(pd.DataFrame(values).to_json(orient='records'))
        results.append({'module_sha256':record['module_sha256'],'values':cleaned})
    value={'probe':kind,'scope':'source_search_only','methods':results,'evidence_files':hashes,
        'interpretation':'Measured source diagnostic. No newly trained predictor or independent validation is claimed.'}
    freeze(dest,value);return value
