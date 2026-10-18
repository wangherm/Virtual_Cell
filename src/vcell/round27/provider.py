"""Keep paid usage authoritative while separately recording exact cached-response replays."""
import hashlib
from ..round25.provider import ResearchGateway as PreviousGateway
from ..model_agent.artifacts import read
from ..evolution.journal import append


class ResearchGateway(PreviousGateway):
    def forward(self,body):
        ticket_key=hashlib.sha256((self.scope+'|'+self.role).encode()).hexdigest()
        ticket=self.path.parent/'provider_tickets'/(ticket_key+'.json')
        cache=self.path.parent/'provider_responses'/(read(ticket)['request_key']+'.json') if ticket.exists() else None
        was_cached=bool(cache and cache.exists() and read(cache).get('state')=='complete')
        result=super().forward(body)
        if was_cached:append(self.path.parent/'provider_cache_reuse.jsonl','provider_cache_hit',role=self.role,ticket=self.scope,http_status=result[0])
        return result
