"""Simple SQLite storage layer for debates, rounds, arguments, scores, memories, and activity log."""
import sqlite3
import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

DEFAULT_DB_PATH = os.path.expanduser('~/Desktop/agent_panel.db')

class SQLiteStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _exec(self, sql: str, params: tuple = ()):  # helper
        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()
        return cur

    def _init_tables(self):
        # Create minimal set of tables (same as ~/Desktop/schema.sql) so the system can run
        self._exec("""
        CREATE TABLE IF NOT EXISTS agents (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          role TEXT,
          personality TEXT,
          created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS debates (
          id TEXT PRIMARY KEY,
          topic TEXT NOT NULL,
          started_at INTEGER DEFAULT (strftime('%s','now')),
          ended_at INTEGER,
          helix_active INTEGER DEFAULT 0,
          debate_number INTEGER DEFAULT 0
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS debate_rounds (
          id TEXT PRIMARY KEY,
          debate_id TEXT NOT NULL,
          round_number INTEGER NOT NULL,
          started_at INTEGER DEFAULT (strftime('%s','now')),
          ended_at INTEGER,
          judge_scored INTEGER DEFAULT 0,
          FOREIGN KEY(debate_id) REFERENCES debates(id) ON DELETE CASCADE
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS arguments (
          id TEXT PRIMARY KEY,
          round_id TEXT NOT NULL,
          agent_id TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at INTEGER DEFAULT (strftime('%s','now')),
          FOREIGN KEY(round_id) REFERENCES debate_rounds(id) ON DELETE CASCADE,
          FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS argument_scores (
          id TEXT PRIMARY KEY,
          argument_id TEXT NOT NULL,
          score INTEGER NOT NULL,
          confidence REAL DEFAULT 1.0,
          judged_at INTEGER DEFAULT (strftime('%s','now')),
          FOREIGN KEY(argument_id) REFERENCES arguments(id) ON DELETE CASCADE
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS debate_tallies (
          id TEXT PRIMARY KEY,
          debate_id TEXT NOT NULL,
          agent_id TEXT NOT NULL,
          total_score INTEGER NOT NULL DEFAULT 0,
          bonus INTEGER NOT NULL DEFAULT 0,
          FOREIGN KEY(debate_id) REFERENCES debates(id) ON DELETE CASCADE,
          FOREIGN KEY(agent_id) REFERENCES agents(id) ON DELETE CASCADE
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
          id TEXT PRIMARY KEY,
          debate_id TEXT,
          round_id TEXT,
          agent_id TEXT,
          embedding TEXT NOT NULL,
          caption TEXT,
          created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS memories (
          id TEXT PRIMARY KEY,
          agent_id TEXT,
          content TEXT NOT NULL,
          embedding TEXT,
          source TEXT,
          created_at INTEGER DEFAULT (strftime('%s','now'))
        );
        """)

        self._exec("""
        CREATE TABLE IF NOT EXISTS activity_log (
          id TEXT PRIMARY KEY,
          debate_id TEXT,
          round_id TEXT,
          event_type TEXT NOT NULL,
          payload TEXT,
          ts INTEGER DEFAULT (strftime('%s','now'))
        );
        """)

    # --- CRUD helpers ---
    def create_agent(self, name: str, role: str = '', personality: Optional[Dict] = None) -> str:
        aid = str(uuid.uuid4())
        self._exec("INSERT INTO agents(id,name,role,personality) VALUES (?,?,?,?)",
                   (aid, name, role, json.dumps(personality or {})))
        return aid

    def get_agent_by_name(self, name: str) -> Optional[Dict]:
        cur = self._exec("SELECT id,name,role,personality,created_at FROM agents WHERE name = ?", (name,))
        row = cur.fetchone()
        if row:
            d = dict(row)
            try:
                d['personality'] = json.loads(d.get('personality') or '{}')
            except Exception:
                d['personality'] = {}
            return d
        return None

    def ensure_agent(self, name: str, role: str = '', personality: Optional[Dict] = None) -> str:
        existing = self.get_agent_by_name(name)
        if existing:
            return existing['id']
        return self.create_agent(name, role, personality)

    def create_debate(self, topic: str, helix_active: bool, debate_number: int) -> str:
        did = str(uuid.uuid4())
        self._exec("INSERT INTO debates(id,topic,helix_active,debate_number) VALUES (?,?,?,?)",
                   (did, topic, int(bool(helix_active)), debate_number))
        return did

    def end_debate(self, debate_id: str):
        self._exec("UPDATE debates SET ended_at = strftime('%s','now') WHERE id = ?", (debate_id,))

    def create_round(self, debate_id: str, round_number: int) -> str:
        rid = str(uuid.uuid4())
        self._exec("INSERT INTO debate_rounds(id,debate_id,round_number) VALUES (?,?,?)",
                   (rid, debate_id, round_number))
        return rid

    def end_round(self, round_id: str):
        self._exec("UPDATE debate_rounds SET ended_at = strftime('%s','now') WHERE id = ?", (round_id,))

    def add_argument(self, round_id: str, agent_id: str, content: str) -> str:
        aid = str(uuid.uuid4())
        self._exec("INSERT INTO arguments(id,round_id,agent_id,content) VALUES (?,?,?,?)",
                   (aid, round_id, agent_id, content))
        return aid

    def add_score(self, argument_id: str, score: int, confidence: float = 1.0) -> str:
        sid = str(uuid.uuid4())
        self._exec("INSERT INTO argument_scores(id,argument_id,score,confidence) VALUES (?,?,?,?)",
                   (sid, argument_id, int(score), float(confidence)))
        return sid

    def add_or_update_tally(self, debate_id: str, agent_id: str, delta: int, bonus: int = 0):
        cur = self._exec("SELECT id,total_score FROM debate_tallies WHERE debate_id = ? AND agent_id = ?",
                         (debate_id, agent_id))
        row = cur.fetchone()
        if row:
            new_total = row['total_score'] + delta
            self._exec("UPDATE debate_tallies SET total_score=?, bonus=? WHERE id = ?",
                       (new_total, bonus, row['id']))
        else:
            tid = str(uuid.uuid4())
            self._exec("INSERT INTO debate_tallies(id,debate_id,agent_id,total_score,bonus) VALUES (?,?,?,?,?)",
                       (tid, debate_id, agent_id, delta, bonus))

    def log_activity(self, debate_id: Optional[str], round_id: Optional[str], event_type: str, payload: Optional[Dict] = None):
        aid = str(uuid.uuid4())
        self._exec("INSERT INTO activity_log(id,debate_id,round_id,event_type,payload) VALUES (?,?,?,?,?)",
                   (aid, debate_id, round_id, event_type, json.dumps(payload or {})))
        return aid

    def save_memory(self, agent_id: Optional[str], content: str, embedding: Optional[List[float]] = None, source: Optional[str] = None) -> str:
        mid = str(uuid.uuid4())
        self._exec("INSERT INTO memories(id,agent_id,content,embedding,source) VALUES (?,?,?,?,?)",
                   (mid, agent_id, content, json.dumps(embedding) if embedding is not None else None, source))
        # Upsert into vector store if available (enqueue to worker if present)
        try:
            if hasattr(self, '_upsert_worker') and getattr(self, '_upsert_worker') is not None:
                # enqueue
                try:
                    asyncio.get_event_loop().create_task(self._upsert_worker.enqueue({'type': 'memory', 'id': mid, 'embedding': embedding, 'content': content, 'source': source or 'memory'}))
                except RuntimeError:
                    # no running loop (e.g., in sync test) -> fallback to synchronous
                    self.vector_store.upsert_memory(mid, embedding, content, source=source or 'memory')
            elif hasattr(self, 'vector_store') and embedding is not None:
                self.vector_store.upsert_memory(mid, embedding, content, source=source or 'memory')
        except Exception as e:
            print(f"⚠️ Failed to upsert memory into vector store: {e}")
        return mid

    def save_image_embedding(self, debate_id: Optional[str], round_id: Optional[str], agent_id: Optional[str], embedding: List[float], caption: Optional[str] = None) -> str:
        iid = str(uuid.uuid4())
        self._exec("INSERT INTO image_embeddings(id,debate_id,round_id,agent_id,embedding,caption) VALUES (?,?,?,?,?,?)",
                   (iid, debate_id, round_id, agent_id, json.dumps(embedding), caption))
        # Upsert into vector store if available (enqueue to worker if present)
        try:
            if hasattr(self, '_upsert_worker') and getattr(self, '_upsert_worker') is not None:
                try:
                    asyncio.get_event_loop().create_task(self._upsert_worker.enqueue({'type': 'image', 'id': iid, 'embedding': embedding, 'caption': caption, 'source': 'image'}))
                except RuntimeError:
                    # no running loop (e.g., in sync test) -> fallback to synchronous
                    self.vector_store.upsert_image(iid, embedding, caption, source='image')
            elif hasattr(self, 'vector_store') and embedding is not None:
                self.vector_store.upsert_image(iid, embedding, caption, source='image')
        except Exception as e:
            print(f"⚠️ Failed to upsert image embedding into vector store: {e}")
        return iid

    def get_debates(self, limit: int = 10) -> List[Dict[str, Any]]:
        cur = self._exec("SELECT id,topic,started_at,ended_at,helix_active,debate_number FROM debates ORDER BY started_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def get_memories(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return memories optionally filtered by agent_id."""
        if agent_id:
            cur = self._exec("SELECT id,agent_id,content,embedding,source,created_at FROM memories WHERE agent_id = ? ORDER BY created_at DESC", (agent_id,))
        else:
            cur = self._exec("SELECT id,agent_id,content,embedding,source,created_at FROM memories ORDER BY created_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get('embedding'):
                try:
                    r['embedding'] = json.loads(r['embedding'])
                except Exception:
                    r['embedding'] = None
        return rows

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by id. Returns True if deleted, False if not found."""
        cur = self._exec("SELECT id FROM memories WHERE id = ?", (memory_id,))
        row = cur.fetchone()
        if not row:
            return False
        self._exec("DELETE FROM memories WHERE id = ?", (memory_id,))
        # Best-effort: remove from vector store as well
        try:
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                try:
                    self.vector_store.remove_memory(memory_id)
                except Exception:
                    pass
        except Exception:
            pass
        return True

    def get_image_embeddings(self, debate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return image embeddings optionally filtered by debate_id."""
        if debate_id:
            cur = self._exec("SELECT id,debate_id,round_id,agent_id,embedding,caption,created_at FROM image_embeddings WHERE debate_id = ? ORDER BY created_at DESC", (debate_id,))
        else:
            cur = self._exec("SELECT id,debate_id,round_id,agent_id,embedding,caption,created_at FROM image_embeddings ORDER BY created_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            try:
                r['embedding'] = json.loads(r['embedding'])
            except Exception:
                r['embedding'] = None
        return rows

    def get_tallies(self, debate_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return tallies with agent names; optionally filter by debate_id."""
        if debate_id:
            cur = self._exec(
                "SELECT dt.debate_id, dt.agent_id, dt.total_score, dt.bonus, a.name as agent_name FROM debate_tallies dt LEFT JOIN agents a ON dt.agent_id = a.id WHERE dt.debate_id = ? ORDER BY dt.total_score DESC",
                (debate_id,)
            )
        else:
            cur = self._exec(
                "SELECT dt.debate_id, dt.agent_id, dt.total_score, dt.bonus, a.name as agent_name FROM debate_tallies dt LEFT JOIN agents a ON dt.agent_id = a.id ORDER BY dt.total_score DESC"
            )
        return [dict(r) for r in cur.fetchall()]

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except:
            pass
