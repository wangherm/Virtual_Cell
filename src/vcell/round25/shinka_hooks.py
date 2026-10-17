"""Small pinned-native-archive sampling hook; no parallel population implementation."""
import hashlib


def select_archive_action(database, original, action, banned, *args, **kwargs):
    parent, archive, top, needs_fix = original(database,*args,**kwargs)
    programs = database.get_all_programs()
    def sha(p): return hashlib.sha256(p.code.encode()).hexdigest()
    lookup = {sha(p):p for p in programs}
    kind = action.get('action') if action else None
    selected = action.get('parent_code_sha256') if action else None
    if selected:
        if selected not in lookup: raise ValueError('Requested parent is not in the persistent Shinka archive')
        parent=lookup[selected]
        if kind=='repair_candidate':
            if parent.correct: raise ValueError('Do not repair valid scientific candidates as runtime errors')
            return parent,[],[],True
        if not parent.correct: raise ValueError('Scientific revision requires a valid parent')
        needs_fix=False
    if sha(parent) in banned:
        legal=[p for p in programs if p.correct and sha(p) not in banned]
        if not legal: raise ValueError('No unblocked archive parent remains')
        parent=max(legal,key=lambda p:p.combined_score);needs_fix=False
    if kind=='crossover_methods':
        second=lookup.get(action.get('second_parent_sha256'))
        if second is None or not second.correct: raise ValueError('Crossover parent missing or invalid')
        archive=[second,*[p for p in archive if p.id not in (parent.id,second.id)]]
    archive=[p for p in archive if sha(p) not in banned]
    top=[p for p in top if sha(p) not in banned]
    return parent,archive,top,needs_fix


def install_action(action,banned):
    from shinka.database import ProgramDatabase
    original=ProgramDatabase.sample_with_fix_mode
    def sample(database,*args,**kwargs):
        return select_archive_action(database,original,action,set(banned),*args,**kwargs)
    ProgramDatabase.sample_with_fix_mode=sample


def check_pinned_api():
    from importlib.metadata import version
    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig, ProgramDatabase
    if version('shinka-evolve')!='0.0.7': raise ValueError('Pinned shinka-evolve==0.0.7 required')
    fields=EvolutionConfig.__dataclass_fields__
    required={'meta_rec_interval','meta_llm_models','patch_types','use_text_feedback','max_patch_attempts'}
    if not required.issubset(fields) or not hasattr(ProgramDatabase,'sample_with_fix_mode'):
        raise ValueError('Pinned archive/feedback API unavailable')
    if not {'num_islands','num_archive_inspirations','num_top_k_inspirations'}.issubset(DatabaseConfig.__dataclass_fields__):
        raise ValueError('Pinned native archive configuration unavailable')
    return {'version':version('shinka-evolve'),'native_capabilities':sorted(required),
            'strategy':'typed source-only critic; native meta disabled to avoid uncounted duplicate critics',
            'sampling':'native archive plus bounded parent/fix/inspiration selection hook'}
