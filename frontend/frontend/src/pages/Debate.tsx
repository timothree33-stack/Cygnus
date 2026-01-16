import React, { useEffect, useState, useRef } from 'react';
import { API_BASE } from '../api';

const SAMPLE_TOPICS = [
  'Should work-from-home be the default?','Is AI regulation necessary now?','Space colonization: who leads?',
  'Open-source vs proprietary: which wins innovation?','Universal basic income – good idea?'
];

export default function Debate() {
  const [topic, setTopic] = useState('');
  const [debateId, setDebateId] = useState<string | null>(null);
  const [state, setState] = useState<any>(null);
  const [foolEnabled, setFoolEnabled] = useState(false);
  const [pauseSec, setPauseSec] = useState<number>(1);
  const [jesterFreq, setJesterFreq] = useState<number>(1); // 0-3 per 3 rounds
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    // When a debate is joined, open a WebSocket for live events (fallback to polling if it fails)
    let ws: WebSocket | null = null;
    if (debateId) {
      const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const host = window.location.hostname;
      const url = `${scheme}://${host}:8001/ws/debates/${debateId}`;
      try {
        ws = new WebSocket(url);
        ws.onmessage = async (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'scores_assigned' || msg.type === 'allcall_round' || msg.type === 'exchange') {
              // Refresh full state for simplicity to ensure UI is consistent
              const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
              if (res.ok) setState(await res.json());
            } else if (msg.type === 'snapshot_taken') {
              // optionally show snapshot notifications (ignored for now)
            } else if (msg.type === 'debate_started' && !debateId) {
              // ignore
            }
          } catch (e) { /* ignore malformed messages */ }
        };
        ws.onopen = () => { console.log('WS connected to debate', debateId); };
        ws.onclose = () => { console.log('WS closed for debate', debateId); };
      } catch (e) {
        // On error, fallback to polling
        if (!pollRef.current) {
          pollRef.current = window.setInterval(async () => {
            try {
              const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
              if (res.ok) {
                const j = await res.json();
                setState(j);
              }
            } catch (e) {}
          }, 1000);
        }
      }
    }

    // ensure initial fetch
    (async () => {
      if (debateId) {
        try {
          const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
          if (res.ok) setState(await res.json());
        } catch (e) {}
      }
    })();

    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      if (ws) { ws.close(); ws = null; }
    };
  }, [debateId]);

  function genTopic() {
    setTopic(SAMPLE_TOPICS[Math.floor(Math.random()*SAMPLE_TOPICS.length)]);
  }

  async function startDebate() {
    const q = new URLSearchParams();
    if (topic) q.set('topic', topic);
    q.set('pause_sec', String(pauseSec));
    const res = await fetch(`${API_BASE}/api/debate/start?${q.toString()}`, { method: 'POST' });
    if (res.ok) {
      const j = await res.json();
      setDebateId(j.debate_id);
    } else {
      alert('Failed to start debate');
    }
  }

  async function pause() {
    if (!debateId) return;
    await fetch(`${API_BASE}/api/debate/${debateId}/pause`, { method: 'POST' });
  }
  async function resume() {
    if (!debateId) return;
    await fetch(`${API_BASE}/api/debate/${debateId}/resume`, { method: 'POST' });
  }
  async function allcall() {
    if (!debateId) return;
    await fetch(`${API_BASE}/api/debate/${debateId}/allcall`, { method: 'POST' });
  }

  // Allow joining an existing debate by id
  async function joinDebate() {
    if (!debateId) return;
    try {
      const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
      if (res.ok) setState(await res.json());
      else alert('Debate not found');
    } catch (e) { alert('Error'); }
  }

  // Render timeline: state.history expected
  const history = (state && state.history) ? state.history : [];

  return (
    <div>
      <h2>Debate Room</h2>

      <div style={{display: 'flex', gap: 12, alignItems: 'center'}}>
        <input placeholder="Enter topic or generate..." value={topic} onChange={(e)=>setTopic(e.target.value)} style={{flex: 1}} />
        <button onClick={genTopic}>Generate</button>
        <label style={{marginLeft: 8}}>
          Pause (sec): <input type="number" value={pauseSec} min={1} max={60} onChange={(e)=>setPauseSec(parseInt(e.target.value||'1'))} style={{width: 60, marginLeft: 6}} />
        </label>
        <label style={{marginLeft: 8}}>
          Jester (per 3 rounds): <input type="number" value={jesterFreq} min={0} max={3} onChange={(e)=>setJesterFreq(parseInt(e.target.value||'1'))} style={{width: 60, marginLeft: 6}} />
        </label>
        <button onClick={startDebate} style={{marginLeft: 8}}>Start</button>
      </div>

      <div style={{marginTop: 12}}>
        <label>
          <input type="checkbox" checked={foolEnabled} onChange={(e) => setFoolEnabled(e.target.checked)} /> Enable Court-Fool (Quips)
        </label>
        {foolEnabled && <p className="muted">Court-Fool enabled: occasional humor will be shown during debates (opt-in)</p>}
      </div>

      <div style={{marginTop: 12}}>
        <input placeholder="Or paste debate id to join" value={debateId||''} onChange={(e)=>setDebateId(e.target.value||null)} />
        <button onClick={joinDebate} style={{marginLeft: 8}}>Join</button>
        <button onClick={pause} style={{marginLeft: 8}}>Pause</button>
        <button onClick={resume} style={{marginLeft: 8}}>Resume</button>
        <button onClick={allcall} style={{marginLeft: 8}}>All Call</button>
      </div>

      <div style={{marginTop: 18}}>
        <h3>Topic: {state?.topic || topic || '(none)'}</h3>
        <p>Debate ID: <code>{debateId || '(not started)'}</code></p>
        <div style={{border: '1px solid #ddd', padding: 12, borderRadius: 6, maxHeight: 400, overflow: 'auto'}}>
          {history.length === 0 && <p className="muted">No rounds yet — start a debate or join one to see the timeline.</p>}
          {history.map((r: any) => (
            <div key={r.round} style={{marginBottom: 12}}>
              <strong>Round {r.round}</strong>
              <div style={{paddingLeft: 12}}>
                <div><strong>Katz:</strong> {r.katz}</div>
                <div><strong>Dogz:</strong> {r.dogz}</div>
                <div><em>Scores:</em> Katz {r.scores?.katz} — Dogz {r.scores?.dogz}</div>
                {foolEnabled && (r.round % 3 === 0) && (Math.random() < (jesterFreq/3)) && <div style={{color: '#b00', marginTop: 6}}><em>Court-Fool:</em> {generateQuip(r.round)}</div>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function generateQuip(round:number){
  const quips = [
    "Is that your argument or are you just happy to be on stage?",
    "I once argued with a toaster and it won.",
    "A bold stance — I admire the bravery of your cursor.",
  ];
  return quips[(round + quips.length) % quips.length];
}
