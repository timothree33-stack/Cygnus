import asyncio
import time
import hashlib
from typing import Callable, Optional, Dict, Any

class DebateOrchestrator:
    def __init__(self, katz, dogz, cygnus, memory, snapshot_fn: Callable, score_fn: Callable, broadcast: Callable):
        """katz/dogz/cygnus: agents or LLM clients with a .respond(topic, context) coroutine
        memory: MemorySystem instance
        snapshot_fn: callable(agent_id, text, timestamp) -> snapshot_id
        score_fn: callable(text_a, text_b) -> (score_a, score_b)  # unique scores 1-100
        broadcast: callable to send websocket events
        """
        self.katz = katz
        self.dogz = dogz
        self.cygnus = cygnus
        self.memory = memory
        self.snapshot = snapshot_fn
        self.score = score_fn
        self.broadcast = broadcast
        self._current_debates: Dict[str, Dict[str, Any]] = {}

    async def run_debate(self, debate_id: str, topic: str, rounds: int = 5, pause_sec: int = 60):
        """Run a debate with given number of rounds and pause between statements.
        Emits broadcast events for UI:
            - debate_started
            - round_started
            - statement (agent, text, timestamp)
            - snapshot_taken (agent, snapshot_id, summary)
            - scores_assigned (round, scores)
            - allcall_round (list of exchanges)
            - debate_finished (final_scores)
        """
        state = {
            'topic': topic,
            'rounds': rounds,
            'pause_sec': pause_sec,
            'round': 0,
            'history': [],
            'mode': 'normal'
        }
        self._current_debates[debate_id] = state
        await self._broadcast({'type': 'debate_started', 'debate_id': debate_id, 'topic': topic})

        for r in range(1, rounds+1):
            state['round'] = r
            await self._broadcast({'type': 'round_started', 'debate_id': debate_id, 'round': r})

            # Before each round, check for allcall request
            if state.get('mode') == 'allcall':
                await self._run_allcall_round(debate_id, topic, r)
                # return mode to normal and continue the current round
                state['mode'] = 'normal'

            # KatZ speaks (Pro)
            katz_text = await self._ask_agent(self.katz, topic, debate_id, r, 'katz')
            # Snapshot & timestamp
            katz_snapshot = self._maybe_snapshot('katz', katz_text, debate_id)
            await asyncio.sleep(pause_sec)

            # DogZ speaks (Antithesis)
            dogz_text = await self._ask_agent(self.dogz, topic, debate_id, r, 'dogz')
            dogz_snapshot = self._maybe_snapshot('dogz', dogz_text, debate_id)
            await asyncio.sleep(pause_sec)

            # Score based on difference (unique scores)
            score_k, score_d = self.score(katz_text, dogz_text, debate_id, r)
            state['history'].append({'round': r, 'katz': katz_text, 'dogz': dogz_text, 'scores': {'katz': score_k, 'dogz': score_d}})
            await self._broadcast({'type': 'scores_assigned', 'debate_id': debate_id, 'round': r, 'scores': {'katz': score_k, 'dogz': score_d}})

        # Final synthesis by Hiemdall (cygnus)
        final_synthesis = await self._ask_agent(self.cygnus, topic, debate_id, 'final', 'cygnus')
        await self._broadcast({'type': 'debate_finished', 'debate_id': debate_id, 'final': final_synthesis, 'history': state['history']})
        return state

    async def _run_allcall_round(self, debate_id: str, topic: str, round_num: int):
        """Run an All Call round where all agents respond simultaneously to a short prompt."""
        await self._broadcast({'type': 'allcall_round_started', 'debate_id': debate_id, 'round': round_num})
        # Concurrently gather both agents responses
        a_task = asyncio.create_task(self._ask_agent(self.katz, topic, debate_id, round_num, 'katz'))
        b_task = asyncio.create_task(self._ask_agent(self.dogz, topic, debate_id, round_num, 'dogz'))
        katz_text, dogz_text = await asyncio.gather(a_task, b_task)

        # Take snapshots
        ks = self._maybe_snapshot('katz', katz_text, debate_id)
        ds = self._maybe_snapshot('dogz', dogz_text, debate_id)

        # Score them relative to each other
        score_k, score_d = self.score(katz_text, dogz_text, debate_id, round_num)
        result = {'round': round_num, 'katz': katz_text, 'dogz': dogz_text, 'scores': {'katz': score_k, 'dogz': score_d}}
        # Append to history if debate tracked
        if debate_id in self._current_debates:
            self._current_debates[debate_id]['history'].append(result)
        await self._broadcast({'type': 'allcall_round', 'debate_id': debate_id, 'round': round_num, 'result': result})

    def trigger_allcall(self, debate_id: str):
        """External trigger to put a debate into All Call for next cycle."""
        if debate_id in self._current_debates:
            self._current_debates[debate_id]['mode'] = 'allcall'
            return True
        return False

    async def _ask_agent(self, agent, topic, debate_id, round_num, agent_name):
        """Ask an agent to respond. Agents may be objects with `respond` coroutine or simple callables."""
        start = time.time()
        try:
            if hasattr(agent, 'respond'):
                text = await agent.respond(topic=topic, debate_id=debate_id, round=round_num)
            else:
                # fallback: call as coroutine function
                text = await agent(topic)
        except Exception as e:
            text = f"(agent error: {e})"
        timestamp = time.time()
        await self._broadcast({'type': 'statement', 'debate_id': debate_id, 'agent': agent_name, 'text': text, 'ts': timestamp})
        return text

    def _maybe_snapshot(self, agent_id: str, text: str, debate_id: str = None):
        try:
            ts = int(time.time())
            # Allow snapshot_fn to store debate context if it accepts it
            try:
                sid, summary = self.snapshot(agent_id, text, ts, debate_id)
            except TypeError:
                # fallback to legacy signature
                sid, summary = self.snapshot(agent_id, text, ts)
            # broadcast asynchronously
            asyncio.create_task(self._broadcast({'type': 'snapshot_taken', 'debate_id': debate_id, 'agent': agent_id, 'snapshot_id': sid, 'summary': summary, 'ts': ts}))
            return sid
        except Exception:
            return None

    def _unique_scores(self, score_a: int, score_b: int):
        # Ensure uniqueness 1-100; if equal, perturb deterministically
        if score_a != score_b:
            return score_a, score_b
        # deterministic perturbation using small hash of timestamp
        h = int(hashlib.sha256(f"{time.time()}".encode()).hexdigest(), 16)
        if h % 2 == 0:
            score_a = max(1, min(100, score_a - 1))
        else:
            score_b = max(1, min(100, score_b - 1))
        if score_a == score_b:
            score_a = max(1, score_a-1)
        return score_a, score_b

    async def _broadcast(self, msg: dict):
        try:
            await self.broadcast(msg)
        except Exception:
            pass

