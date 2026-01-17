import asyncio
import time
import hashlib
from typing import Callable, Optional, Dict, Any

class DebateOrchestrator:
    def __init__(self, katz, dogz, cygnus, memory, snapshot_fn: Callable, score_fn: Callable, broadcast: Callable):
        """katz/dogz/cygnus: agents or LLM clients with a .respond(topic, context) coroutine
        memory: MemorySystem or SQLiteStore instance
        snapshot_fn: callable(agent_id, text, timestamp, debate_id=None) -> snapshot_id
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
        # If a SQLiteStore-like object is passed in as memory, use it for persistence
        self.store = None
        try:
            # duck-type: has create_debate method
            if memory and hasattr(memory, 'create_debate'):
                self.store = memory
        except Exception:
            self.store = None

    async def _ensure_store(self):
        if self.store:
            return self.store
        try:
            from ..db.sqlite_store import SQLiteStore
            self.store = SQLiteStore()
            return self.store
        except Exception:
            return None

    async def run_debate(self, debate_id: str, topic: str, rounds: int = 5, pause_sec: int = 60, katz_members: list | None = None, dogz_members: list | None = None):
        """Run a debate with given number of rounds and pause between statements.
        Supports per-team member rotation. `katz_members` and `dogz_members` are lists of member names.
        Emits broadcast events for UI:
            - debate_started
            - round_started
            - member_changed (team, member_name, member_index)
            - statement (agent, member, text, timestamp)
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
            'mode': 'normal',
            'katz_members': katz_members or [],
            'dogz_members': dogz_members or [],
            'active_members': {'katz': None, 'dogz': None}
        }
        self._current_debates[debate_id] = state
        await self._broadcast({'type': 'debate_started', 'debate_id': debate_id, 'topic': topic})

        for r in range(1, rounds+1):
            state['round'] = r
            # compute active member for each team and broadcast change if different
            def pick_active(members, rnd):
                if not members:
                    return None, None
                idx = (rnd - 1) % len(members)
                return members[idx], idx

            katz_member, katz_idx = pick_active(state['katz_members'], r)
            dogz_member, dogz_idx = pick_active(state['dogz_members'], r)

            # Broadcast changes if different from last seen
            if state['active_members'].get('katz') != katz_member:
                state['active_members']['katz'] = katz_member
                await self._broadcast({'type': 'member_changed', 'debate_id': debate_id, 'team': 'katz', 'member_name': katz_member, 'member_index': katz_idx, 'ts': time.time()})
            if state['active_members'].get('dogz') != dogz_member:
                state['active_members']['dogz'] = dogz_member
                await self._broadcast({'type': 'member_changed', 'debate_id': debate_id, 'team': 'dogz', 'member_name': dogz_member, 'member_index': dogz_idx, 'ts': time.time()})

            await self._broadcast({'type': 'round_started', 'debate_id': debate_id, 'round': r})

            # Ensure we have a store to persist debate/round data
            store = await self._ensure_store()
            round_id = None
            if store:
                round_id = store.create_round(debate_id, r)

            # Before each round, check for allcall request
            if state.get('mode') == 'allcall':
                await self._run_allcall_round(debate_id, topic, r)
                # return mode to normal and continue the current round
                state['mode'] = 'normal'

            # KatZ speaks (Pro)
            katz_text, katz_arg_id = await self._ask_agent(self.katz, topic, debate_id, r, 'katz', round_id, member_name=katz_member)
            # Snapshot & timestamp
            katz_snapshot = self._maybe_snapshot(f'katz:{katz_member}' if katz_member else 'katz', katz_text, debate_id, r)
            await asyncio.sleep(pause_sec)

            # DogZ speaks (Antithesis)
            dogz_text, dogz_arg_id = await self._ask_agent(self.dogz, topic, debate_id, r, 'dogz', round_id, member_name=dogz_member)
            dogz_snapshot = self._maybe_snapshot(f'dogz:{dogz_member}' if dogz_member else 'dogz', dogz_text, debate_id, r)
            await asyncio.sleep(pause_sec)

            # Score based on difference (unique scores)
            score_k, score_d = self.score(katz_text, dogz_text, debate_id, r)
            state['history'].append({'round': r, 'katz': katz_text, 'dogz': dogz_text, 'scores': {'katz': score_k, 'dogz': score_d}})
            await self._broadcast({'type': 'scores_assigned', 'debate_id': debate_id, 'round': r, 'scores': {'katz': score_k, 'dogz': score_d}})

            # Persist scores and update tallies
            try:
                if store and katz_arg_id and dogz_arg_id:
                    # insert across arguments (create argument_scores and update debate_tallies)
                    sid_k = store.add_score(katz_arg_id, score_k, confidence=1.0)
                    sid_d = store.add_score(dogz_arg_id, score_d, confidence=1.0)
                    # use namespaced agent ids so tallies are per-member
                    store.add_or_update_tally(debate_id, store.ensure_agent(f'katz:{katz_member}' if katz_member else 'katz', role='katz'), score_k)
                    store.add_or_update_tally(debate_id, store.ensure_agent(f'dogz:{dogz_member}' if dogz_member else 'dogz', role='dogz'), score_d)
            except Exception:
                pass
        # Final synthesis by Hiemdall (cygnus)
        final_synthesis = await self._ask_agent(self.cygnus, topic, debate_id, 'final', 'cygnus')
        await self._broadcast({'type': 'debate_finished', 'debate_id': debate_id, 'final': final_synthesis, 'history': state['history']})
        return state

    async def _run_allcall_round(self, debate_id: str, topic: str, round_num: int):
        """Run an All Call round where all agents respond simultaneously to a short prompt."""
        await self._broadcast({'type': 'allcall_round_started', 'debate_id': debate_id, 'round': round_num})
        # Determine active members for the round if available
        katz_member = None
        dogz_member = None
        if debate_id in self._current_debates:
            s = self._current_debates[debate_id]
            km = s.get('katz_members', [])
            dm = s.get('dogz_members', [])
            if km:
                katz_member = km[(round_num - 1) % len(km)]
            if dm:
                dogz_member = dm[(round_num - 1) % len(dm)]

        # Concurrently gather both agents responses
        a_task = asyncio.create_task(self._ask_agent(self.katz, topic, debate_id, round_num, 'katz', member_name=katz_member))
        b_task = asyncio.create_task(self._ask_agent(self.dogz, topic, debate_id, round_num, 'dogz', member_name=dogz_member))
        katz_text, dogz_text = await asyncio.gather(a_task, b_task)

        # Take snapshots
        ks = self._maybe_snapshot(f'katz:{katz_member}' if katz_member else 'katz', katz_text, debate_id, round_num)
        ds = self._maybe_snapshot(f'dogz:{dogz_member}' if dogz_member else 'dogz', dogz_text, debate_id, round_num)

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

    async def _ask_agent(self, agent, topic, debate_id, round_num, agent_name, round_id=None, member_name: str = None):
        """Ask an agent to respond. Agents may be objects with `respond` coroutine or simple callables.
        Returns tuple (text, argument_id_or_None) where argument_id is persisted when a store is available.
        `member_name` is optional and indicates which team member is speaking.
        """
        start = time.time()
        try:
            if hasattr(agent, 'respond'):
                # Pass member context to agent if available
                if member_name is not None:
                    text = await agent.respond(topic=topic, debate_id=debate_id, round=round_num, member=member_name)
                else:
                    text = await agent.respond(topic=topic, debate_id=debate_id, round=round_num)
            else:
                # fallback: call as coroutine function
                text = await agent(topic)
        except Exception as e:
            text = f"(agent error: {e})"
        timestamp = time.time()
        # broadcast agent as namespaced team:member when member present
        agent_id = f"{agent_name}:{member_name}" if member_name else agent_name
        await self._broadcast({'type': 'statement', 'debate_id': debate_id, 'agent': agent_id, 'team': agent_name, 'member': member_name, 'text': text, 'ts': timestamp})

        # Persist argument to store when available
        arg_id = None
        try:
            store = await self._ensure_store()
            if store:
                agent_key = f"{agent_name}:{member_name}" if member_name else agent_name
                agent_uuid = store.ensure_agent(agent_key, role=agent_name)
                arg_id = store.add_argument(round_id, agent_uuid, text) if round_id else None
        except Exception:
            arg_id = None

        return text, arg_id
    def _maybe_snapshot(self, agent_id: str, text: str, debate_id: str = None, round_num: int = None):
        try:
            ts = int(time.time())
            # Allow snapshot_fn to store debate context if it accepts it
            try:
                sid, summary = self.snapshot(agent_id, text, ts, debate_id)
            except TypeError:
                # fallback to legacy signature
                sid, summary = self.snapshot(agent_id, text, ts)

            # Attach snapshot to current debate history when possible
            try:
                if debate_id and debate_id in self._current_debates:
                    hist = self._current_debates[debate_id].setdefault('history', [])
                    if round_num is not None:
                        entry = next((h for h in hist if h.get('round') == round_num), None)
                        if not entry:
                            entry = {'round': round_num, 'katz': '', 'dogz': '', 'scores': {}}
                            hist.append(entry)
                        # store snapshot per-agent
                        key = f"{agent_id}_snapshot"
                        entry[key] = {'id': sid, 'summary': summary, 'ts': ts}
            except Exception:
                pass

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

