"""Freedom to design heads inside the existing isolated, bounded AST language."""
from ..model_agent.research import ResearchPlanner,INSTRUCTIONS
from ..round19.agents import IterationAgent


class SearchAgent(ResearchPlanner):
    instructions=INSTRUCTIONS.replace('target is STRING PCA+coverage, background is frozen source-trained B2 PCA. No gene IDs or reference tensors.',
        'target contains STRING PCA+coverage followed by same-target reference coefficients. background contains source-trained B2 PCA, query-minus-donor B2, and reference-availability flag. These tensors come only from allowed source fit data.')+'''
Round20: design a bounded correction to an experimental same-target reference.
Controller applies tanh and a fixed L2 radius outside your code. Source-only MSE,
strong-gene errors, amplitude and specificity decide promotion. Independent weights
are trained from scratch for every candidate. Three proposal slots; failed and
duplicate decisions consume a slot. Do not change radius, folds, metrics or steps.
Keep reason under 800 characters. You may stop. No repository edits are executed.
'''
    def parse_action(self,content,observation):
        p=super().parse_action(content,observation)
        if len(p.reason)>800:raise ValueError('Round20 reason exceeds 800 characters')
        return p


class ShortReviewer(IterationAgent):
    instructions=IterationAgent.instructions.replace('<=2000 characters each','<=600 characters each')+'''
Review the most recent source-only bounded reference correction head. Give one
actionable hypothesis for the next code/parameter proposal; avoid repeating history.
'''
    def parse_action(self,content,observation):
        p=super().parse_action(content,observation)
        if max(len(p.reason),len(p.next_hypothesis))>600:raise ValueError('Round20 review exceeds 600 characters')
        return p
