"""A bounded neural-head code language, not unrestricted Python execution."""
import ast
import builtins
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

LAYERS={'Linear','LayerNorm','GELU','ReLU','Dropout','Sequential'}
TORCH_CALLS={'cat','tanh','sigmoid'}
F_CALLS={'gelu','relu','tanh','normalize'}
ALLOWED_NODES=(ast.Module,ast.ClassDef,ast.FunctionDef,ast.arguments,ast.arg,ast.Assign,ast.Expr,ast.Return,
    ast.Call,ast.Name,ast.Attribute,ast.Constant,ast.Load,ast.Store,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.Div,
    ast.UnaryOp,ast.USub,ast.UAdd,ast.Tuple,ast.List,ast.keyword)


def validate_code(source):
    if not isinstance(source,str) or len(source)>12000:raise ValueError('Candidate source exceeds the bounded head language')
    tree=ast.parse(source)
    if len(list(ast.walk(tree)))>500:raise ValueError('Candidate head too complex')
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.ClassDef):raise ValueError('Exactly one CandidateHead class required')
    cls=tree.body[0]
    if cls.name!='CandidateHead' or cls.decorator_list or cls.keywords or len(cls.bases)!=1 or ast.unparse(cls.bases[0])!='nn.Module':
        raise ValueError('CandidateHead must derive directly from nn.Module')
    if len(cls.body)!=2 or {m.name for m in cls.body if isinstance(m,ast.FunctionDef)}!={'__init__','forward'}:
        raise ValueError('Only __init__ and forward methods are allowed')
    fields=set()
    for node in ast.walk(tree):
        if not isinstance(node,ALLOWED_NODES):raise ValueError('Unsupported candidate syntax: '+type(node).__name__)
        if isinstance(node,ast.Attribute) and node.attr.startswith('_') and node.attr!='__init__':raise ValueError('Private/dunder access forbidden')
        if isinstance(node,ast.Name) and node.id.startswith('_'):raise ValueError('Private name access forbidden')
        if isinstance(node,ast.Constant) and (isinstance(node.value,str) or isinstance(node.value,(int,float)) and not isinstance(node.value,bool) and abs(node.value)>128):
            raise ValueError('Unbounded/string literals forbidden')
    if sum(isinstance(n,ast.Call) for n in ast.walk(cls))>20:raise ValueError('Candidate call budget exceeded')
    if sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cat' for n in ast.walk(cls))>1:
        raise ValueError('Only one bounded concatenation permitted')
    for method in sorted(cls.body,key=lambda m:m.name!='__init__'):
        expected=['self','target_dim','background_dim','rank'] if method.name=='__init__' else ['self','target','background']
        if [a.arg for a in method.args.args]!=expected or method.args.defaults or method.args.kwonlyargs or method.args.vararg or method.args.kwarg or method.decorator_list or method.returns:
            raise ValueError('Exact bounded method signature required')
        local=set(expected)|{'nn','F','torch','super'}
        for node in ast.walk(method):
            if isinstance(node,ast.Assign):
                if len(node.targets)!=1:raise ValueError('Single assignment required')
                target=node.targets[0]
                if method.name=='__init__':
                    if not isinstance(target,ast.Attribute) or not isinstance(target.value,ast.Name) or target.value.id!='self':raise ValueError('Initialize only named self layers')
                    if not isinstance(node.value,ast.Call) or not isinstance(node.value.func,ast.Attribute) or not isinstance(node.value.func.value,ast.Name) or node.value.func.value.id!='nn':raise ValueError('Only bounded nn layers may be assigned to self')
                    fields.add(target.attr)
                else:
                    if not isinstance(target,ast.Name):raise ValueError('Forward may assign local tensors only')
                    local.add(target.id)
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id not in local:raise ValueError('Unknown candidate name: '+node.id)
            if isinstance(node,ast.Call):
                fn=node.func
                if isinstance(fn,ast.Name):
                    if fn.id!='super' or method.name!='__init__' or node.args or node.keywords:raise ValueError('Builtin/function calls forbidden')
                elif isinstance(fn,ast.Attribute):
                    owner=fn.value
                    if isinstance(owner,ast.Call) and isinstance(owner.func,ast.Name) and owner.func.id=='super' and fn.attr=='__init__' and method.name=='__init__':continue
                    if not isinstance(owner,ast.Name):raise ValueError('Chained call access forbidden')
                    if owner.id=='nn' and fn.attr in LAYERS:
                        if method.name!='__init__':raise ValueError('Create layers only in __init__')
                        if fn.attr=='Linear':
                            if len(node.args)!=2:raise ValueError('Two explicit Linear dimensions required')
                            for dim in node.args:
                                if isinstance(dim,ast.Name) and dim.id in ('target_dim','background_dim','rank'):continue
                                if isinstance(dim,ast.Constant) and isinstance(dim.value,int) and 1<=dim.value<=128:continue
                                if isinstance(dim,ast.BinOp) and isinstance(dim.op,ast.Add) and isinstance(dim.left,ast.Name) and isinstance(dim.right,ast.Name) and {dim.left.id,dim.right.id}=={'target_dim','background_dim'}:continue
                                raise ValueError('Unbounded Linear dimension expression')
                        elif fn.attr=='LayerNorm':
                            if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant) or not isinstance(node.args[0].value,int) or not 1<=node.args[0].value<=128:raise ValueError('Bounded LayerNorm dimension required')
                        elif fn.attr=='Dropout':
                            if len(node.args)!=1 or not isinstance(node.args[0],ast.Constant) or not isinstance(node.args[0].value,(int,float)) or not 0<=node.args[0].value<=.5:raise ValueError('Bounded Dropout probability required')
                    elif owner.id=='self' and fn.attr in fields and method.name=='forward':pass
                    elif owner.id=='torch' and fn.attr in TORCH_CALLS and method.name=='forward':
                        if fn.attr=='cat' and (len(node.args)!=1 or not isinstance(node.args[0],(ast.List,ast.Tuple)) or len(node.args[0].elts)!=2 or
                            len(node.keywords)!=1 or node.keywords[0].arg!='dim' or not isinstance(node.keywords[0].value,ast.Constant) or node.keywords[0].value.value!=1):
                            raise ValueError('Concatenate exactly two tensors along feature dimension')
                    elif owner.id=='F' and fn.attr in F_CALLS and method.name=='forward':
                        if fn.attr=='normalize' and (len(node.args)!=1 or any(k.arg=='dim' and ast.unparse(k.value) not in ('1','-1') for k in node.keywords)):
                            raise ValueError('Normalize features only; no cross-query normalization')
                    else:raise ValueError('Candidate call is not allowlisted')
                else:raise ValueError('Dynamic call forbidden')
    return tree


def build_head(source,target_dim,background_dim,rank):
    tree=validate_code(source)
    scope={'__name__':'bounded_candidate','__builtins__':{'__build_class__':builtins.__build_class__,'super':super},'nn':nn,'torch':torch,'F':F}
    exec(compile(tree,'<bounded_candidate>','exec'),scope)
    model=scope['CandidateHead'](target_dim,background_dim,rank)
    if not 0<sum(p.numel() for p in model.parameters())<=200000:raise ValueError('Candidate parameter budget exceeded')
    return model


def check_head(source,target_dim=33,background_dim=16,rank=16):
    model=build_head(source,target_dim,background_dim,rank)
    t=torch.randn(4,target_dim);b=torch.randn(4,background_dim);y=model(t,b)
    if y.shape!=(4,rank) or not torch.isfinite(y).all():raise ValueError('Candidate output shape/finiteness failed')
    y.square().mean().backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum()>0 for g in grads):raise ValueError('Candidate has no usable finite gradient')
    return {'parameters':sum(p.numel() for p in model.parameters()),'shape':[4,rank],'gradient_finite':True,
            'admission':'bounded AST + CPU shape/gradient checks; not a general-purpose code sandbox'}


BASE_HEAD='''class CandidateHead(nn.Module):
    def __init__(self, target_dim, background_dim, rank):
        super().__init__()
        self.target = nn.Linear(target_dim, 32)
        self.background = nn.Linear(background_dim, 32)
        self.output = nn.Linear(32, rank)
    def forward(self, target, background):
        t = torch.tanh(self.target(target))
        b = torch.tanh(self.background(background))
        return self.output(t + b + t * b)
'''
