import React, { useState, useEffect } from 'react';
import { API_BASE } from '../api';

export default function ConversationBox({ agentId, personality }: { agentId: string; personality?: string }) {
  const [messages, setMessages] = useState<{ id?: string; from: string; text: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // load persisted messages
    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        // Use generic memories endpoint (compatible with current backend)
        const res = await fetch(`${API_BASE}/api/admin/memories?agent_id=${encodeURIComponent(agentId)}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!mounted) return;
        // normalize messages (filter by 'conversation' source prefix used by save-memory)
        const msgs = (data.memories || []).filter((m: any) => (m.source||'').startsWith('conversation')).map((m: any) => ({ id: m.id, from: (m.source||'').split(':')[1] || 'human', text: m.content }));
        setMessages(msgs);
      } catch (e) {
        // ignore for now
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, [agentId]);

  const send = async () => {
    if (!input) return;
    // optimistic update
    const optimistic = { from: 'human', text: input };
    setMessages((m) => [...m, optimistic]);
    const payload = { role: 'human', text: input };
    setInput('');
    try {
      // Use generic save-memory endpoint which accepts {agent_id, content, embedding, source}
      const res = await fetch(`${API_BASE}/api/admin/save-memory`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ agent_id: agentId, content: input, source: 'conversation:human' })
      });
      if (res.ok) {
        const data = await res.json();
        // replace last optimistic message with returned one (id attach)
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (!last) return prev;
          const updated = prev.slice(0, -1).concat([{ id: data.saved, from: 'human', text: last.text }]);
          return updated;
        });
      }
    } catch (e) {
      // best-effort: keep optimistic message but mark it failed (TODO)
    }
  };

  const removeMessage = async (id?: string) => {
    if (!id) return;
    try {
      // New delete endpoint: DELETE /api/admin/memory/{id}
      const res = await fetch(`${API_BASE}/api/admin/memory/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (res.ok) {
        setMessages((m) => m.filter((msg) => msg.id !== id));
      }
    } catch (e) {
      // ignore
    }
  };

  return (
    <div className="conversation-box">
      <div className="conv-header">Agent: {agentId} • Personality: <strong>{personality}</strong> {loading && <em>loading...</em>}</div>
      <div className="conv-messages">
        {messages.map((m, i) => (
          <div key={m.id || i} className={`msg ${m.from}`}>
            {m.from}: {m.text} {m.id && <button className="small" onClick={() => removeMessage(m.id)}>Prune</button>}
          </div>
        ))}
      </div>
      <div className="conv-input">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Say something..." />
        <button onClick={send}>Send</button>
      </div>
    </div>
  );
}
