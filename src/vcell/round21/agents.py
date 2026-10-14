"""Code freedom remains bounded; only current source signature evidence is visible."""
from ..round20.agents import SearchAgent,ShortReviewer


class SignatureScientist(SearchAgent):
    instructions=SearchAgent.instructions.replace('Round20:','Round21:')+'''
Your experimental references have passed source-only target-vs-mismatch reliability
screening. Correct-reference and shuffled-reference metrics use common supported
queries. Supported Rep1-to-Rep2 prediction is NOT unknown-cell or unknown-target
generalization. Avoid claiming biological reproducibility from technical repeats.
Use the current source results, not prior development-background scores.
'''


class SignatureReviewer(ShortReviewer):
    instructions=ShortReviewer.instructions+'''
Round21: distinguish supported technical-repeat prediction from cross-cell transfer.
Check common reference support, unknown-target subsets and actual TRAINED status.
Never promote an untrained reference fallback as a learned correction.
'''
