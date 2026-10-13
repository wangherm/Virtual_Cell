"""Actual offline pretrained transformer, numeric tokens, LoRA and signed delta head."""
import gc
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..model_agent.artifacts import read,seal,verify,arrays
from ..model_agent.contracts import digest
from ..round11 import freeze
from ..selection import row_weights
from ..utils import seed_all,atomic_torch_save,write_json,file_sha256
from .agents import TEACHERS,validate_recipe


class DeepSeekPredictor(nn.Module):
    def __init__(self,spec,control_dim,target_dim,teacher_rank,output_rank,recipe):
        super().__init__();self.recipe=validate_recipe(recipe)
        self.register_buffer('teacher_mask',torch.tensor([t in recipe['teachers'] for t in TEACHERS]))
        self.rank=output_rank;self.teacher_rank=teacher_rank
        self.backbone=None
        if recipe['backbone']!='mlp':
            from transformers import AutoModel
            dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[spec['dtype']]
            base=AutoModel.from_pretrained(spec['model_dir'],local_files_only=True,trust_remote_code=False,
                dtype=dtype,attn_implementation='sdpa')
            if base.config.model_type not in ('qwen2','qwen3'):raise ValueError('Pinned Qwen-family local backbone required')
            base.config.use_cache=False
            for p in base.parameters():p.requires_grad_(False)
            if recipe['adaptation']=='lora':
                from peft import LoraConfig,get_peft_model,TaskType
                base=get_peft_model(base,LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=8,lora_alpha=16,
                    lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none'))
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.backbone=base;hidden=base.config.hidden_size
            self.control=nn.Linear(control_dim,hidden)
            self.target=nn.Linear(target_dim,hidden)
            self.teachers=nn.ModuleList([nn.Linear(teacher_rank,hidden) for _ in TEACHERS])
            self.final_token=nn.Parameter(torch.randn(1,1,hidden)*.02)
        else:
            hidden=recipe['width']
            self.encoder=nn.Sequential(nn.Linear(control_dim+target_dim+len(TEACHERS)*teacher_rank,hidden),
                nn.GELU(),nn.LayerNorm(hidden))
        self.output=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,recipe['width']),nn.GELU(),nn.Linear(recipe['width'],output_rank))
        self.gate=nn.Linear(hidden,len(TEACHERS)) if recipe['structure']=='gate' else None

    def forward(self,control,target,teachers,prompt_ids=None,prompt_mask=None):
        if not self.recipe['use_string']:target=target*0
        teachers=teachers*self.teacher_mask[None,:,None]
        if self.backbone is None:h=self.encoder(torch.cat([control,target,teachers.flatten(1)],1))
        else:
            words=self.backbone.get_input_embeddings()(prompt_ids)
            tokens=[self.control(control),self.target(target)]
            tokens += [project(teachers[:,j])*self.teacher_mask[j] for j,project in enumerate(self.teachers)]
            numeric=torch.stack(tokens,1).to(words.dtype)
            values=torch.cat([numeric,words,self.final_token.expand(len(control),-1,-1).to(words.dtype)],1)
            mask=torch.cat([torch.ones((len(control),len(tokens)),device=control.device,dtype=torch.long),prompt_mask,
                            torch.ones((len(control),1),device=control.device,dtype=torch.long)],1)
            positions=(mask.cumsum(-1)-1).clamp_min(0)
            h=self.backbone(inputs_embeds=values,attention_mask=mask,position_ids=positions,use_cache=False).last_hidden_state[:,-1].float()
        correction=.05*self.output(h)
        if self.recipe['structure']=='direct':return correction
        coeff=teachers[:,:,:self.rank]
        if self.gate is not None:
            weight=self.gate(h).masked_fill(~self.teacher_mask[None,:],-1e4).softmax(1)
        else:weight=self.teacher_mask.float()[None,:]/self.teacher_mask.sum().clamp_min(1)
        return correction+(weight[:,:,None]*coeff).sum(1)

    def trainable_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def restore(self,state):
        params={n:p for n,p in self.named_parameters() if p.requires_grad}
        if set(state)!=set(params):raise ValueError('Trainable state names changed')
        with torch.no_grad():
            for n,p in params.items():p.copy_(state[n].to(p))


def specificity_loss(pred,truth,contexts,targets):
    p=F.normalize(pred,dim=1,eps=1e-8);y=F.normalize(truth,dim=1,eps=1e-8)
    sim=p@y.T;positive=sim.diag()
    negative=(contexts[:,None]==contexts[None,:])&(targets[:,None]!=targets[None,:])
    valid=negative & (truth.norm(dim=1)[None,:]>1e-8) & (truth.norm(dim=1)[:,None]>1e-8)
    if not valid.any():return pred.sum()*0
    return F.relu(.1-positive[:,None]+sim)[valid].mean()


@torch.no_grad()
def infer(model,features,rows,device,batch):
    model.eval();out=[]
    for start in range(0,len(rows),batch):
        ix=rows[start:start+batch]
        out.append(model(*[features[k][ix].to(device) for k in ('control','target','teachers','prompt_ids','prompt_mask')]).float().cpu().numpy())
    result=np.concatenate(out)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite predictor inference')
    return result


def train_worker(cfg,resume=False):
    """Worker receives only fit/cal labels and control-only held features, no API key."""
    torch.set_num_threads(2);dest=Path(cfg['output_dir']);dest.mkdir(parents=True,exist_ok=True)
    cache=Path(cfg['cache']);verify(cache);recipe=validate_recipe(cfg['recipe'])
    identity={'config':cfg,'cache_sha256':file_sha256(cache/'COMPLETE.json'),
              'code':digest({p.name:file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})}
    stamp=digest(identity);freeze(dest/'manifest.json',{'fingerprint':stamp,**identity})
    if (dest/'COMPLETE.json').exists():verify(dest);return dest
    d=arrays(cache/'features.npz');labels=arrays(cache/'source_labels.npz')
    if set(labels)!={'fit','calibration','fit_delta','calibration_delta'}:raise ValueError('Worker label scope invalid')
    fit,cal=labels['fit'].astype(int),labels['calibration'].astype(int);query=d['query'].astype(int)
    if set(query)&set(np.r_[fit,cal]) or not len(fit) or not len(cal):raise ValueError('Worker partition overlap/empty')
    rank=min(recipe['rank'],len(d['basis']));basis=d['basis'][:rank]
    seed_all(cfg['seed']);device=cfg['device']
    model=DeepSeekPredictor(cfg['model'],d['control'].shape[1],d['target'].shape[1],d['teachers'].shape[2],rank,recipe).to(device)
    info={'backbone':recipe['backbone'],'adaptation':recipe['adaptation'],
          'total_parameters':sum(p.numel() for p in model.parameters()),
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'actual_response_rank':rank,'model_identity':cfg['model']['identity'],
          'pretrained_language_backbone':recipe['backbone']!='mlp'}
    freeze(dest/'model_info.json',info)
    features={k:torch.tensor(d[k]) for k in ('control','target','teachers')}
    if recipe['backbone']!='mlp':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']['model_dir'],local_files_only=True,trust_remote_code=False)
        tokenizer.padding_side='right'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        encoded=tokenizer(d['prompt_texts'].tolist(),padding=True,truncation=True,max_length=64,return_tensors='pt')
        features.update(prompt_ids=encoded['input_ids'],prompt_mask=encoded['attention_mask'])
    else:
        features.update(prompt_ids=torch.zeros((len(d['control']),1),dtype=torch.long),
                        prompt_mask=torch.ones((len(d['control']),1),dtype=torch.long))
    y=np.zeros((len(d['control']),d['basis'].shape[1]),np.float32)
    y[fit]=labels['fit_delta'];y[cal]=labels['calibration_delta']
    y=torch.tensor(y,device=device);bt=torch.tensor(basis,device=device)
    context=torch.tensor(d['context_codes'],device=device);target=torch.tensor(d['target_codes'],device=device)
    w=d['fit_weights'];cw=d['calibration_weights'];first=0;best=float('inf');history=[];best_state=None
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    if resume and (dest/'last.pt').exists():
        s=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
        if s['fingerprint']!=stamp:raise ValueError('Student checkpoint identity changed')
        model.restore(s['state']);opt.load_state_dict(s['optimizer']);first=s['step'];best=s['best'];best_state=s['best_state'];history=s['history']
    timer=perf_counter()
    for step in range(first,cfg['max_steps']):
        seed_all(cfg['seed']+19000+step)
        ix=np.random.default_rng(cfg['seed']+19000+step).choice(fit,cfg['batch_size'],p=w)
        model.train();opt.zero_grad(set_to_none=True)
        coeff=model(*[features[k][ix].to(device) for k in ('control','target','teachers','prompt_ids','prompt_mask')])
        pred=coeff@bt;truth=y[ix]
        loss=(pred-truth).square().mean() if recipe['loss']=='mse' else F.huber_loss(pred,truth)
        loss=loss+recipe['specificity_weight']*specificity_loss(pred,truth,context[ix],target[ix])
        if recipe['kd_weight']:
            mask=model.teacher_mask.cpu().numpy();teacher=d['teachers'][ix][:,mask,:rank].mean(1)
            loss=loss+recipe['kd_weight']*(coeff-torch.tensor(teacher,device=device)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student objective')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step()
        done=step+1
        if done==1 or done%10==0:print(f'ROUND19 TRAIN step={done}/{cfg["max_steps"]} loss={loss.item():.6f}',flush=True)
        if done%cfg['save_every']==0 or done==cfg['max_steps']:
            pred=infer(model,features,cal,device,cfg['batch_size'])@basis
            score=float(cw@np.square(pred-labels['calibration_delta']).mean(1))
            if not np.isfinite(score):raise FloatingPointError('Nonfinite source calibration')
            if score<best:best=score;best_state=model.trainable_state()
            history.append({'step':done,'source_calibration_mse_normalized':score,'fit_loss':loss.item()})
            atomic_torch_save({'fingerprint':stamp,'step':done,'state':model.trainable_state(),'optimizer':opt.state_dict(),
                'best':best,'best_state':best_state,'history':history},dest/'last.pt')
            write_json(dest/'progress.json',{'step':done,'max_steps':cfg['max_steps'],'elapsed_seconds':perf_counter()-timer})
    model.restore(best_state)
    prediction=infer(model,features,query,device,cfg['batch_size'])@basis*float(d['response_scale'])
    np.savez_compressed(dest/'prediction.npz',prediction=prediction.astype(np.float32),row_ids=d['row_ids'][query],genes=d['genes'])
    atomic_torch_save({'fingerprint':stamp,'state':best_state,'model':cfg['model'],'recipe':recipe,
        'basis':torch.tensor(basis),'response_scale':float(d['response_scale'])},dest/'best.pt')
    write_json(dest/'history.json',history);write_json(dest/'runtime.json',{'seconds':perf_counter()-timer,'steps':cfg['max_steps']})
    seal(dest);del model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return dest
