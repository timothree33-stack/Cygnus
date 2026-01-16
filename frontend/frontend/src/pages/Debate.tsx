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
            if (msg.type === 'statement') {
              // agent statement - append to relevant round entry
              setState((s:any)=>{
                if(!s) return s;
                const history = [...(s.history||[])];
                const r = msg.round || msg.round_num || msg.round;
                let entry = history.find((h:any)=>h.round===r);
                if(!entry){ entry = {round: r, katz: '', dogz: '', scores: {}}; history.push(entry); }
                if(msg.agent === 'katz') entry.katz = msg.text;
                if(msg.agent === 'dogz') entry.dogz = msg.text;
                return {...s, history};
              });
            } else if (msg.type === 'scores_assigned') {
              setState((s:any)=>{
                if(!s) return s;
                const history = [...(s.history||[])];
                const r = msg.round;
                const entry = history.find((h:any)=>h.round===r);
                if(entry) entry.scores = msg.scores;
                return {...s, history};
              });
            } else if (msg.type === 'allcall_round') {
              setState((s:any)=>{
                if(!s) return s;
                const history = [...(s.history||[])];
                history.push(msg.result);
                return {...s, history};
              });
            } else if (msg.type === 'snapshot_taken') {
              // append snapshot to snapshot list
              setSnapshots((xs:any[])=>[{id: msg.snapshot_id, agent: msg.agent, summary: msg.summary, ts: msg.ts, debate_id: msg.debate_id}, ...xs]);
            } else if (msg.type === 'debate_started') {
              // set basic topic if not present
              setState((s:any)=> ({...(s||{}), topic: msg.topic}));
            }
          } catch (e) { /* ignore malformed messages */ }
        };
        ws.onopen = () => { console.log('WS connected to debate', debateId); };
        ws.onclose = () => { console.log('WS closed for debate', debateId); };
      } catch (e) {
        // On error, fallback to polling once per second
        if (!pollRef.current) {
          pollRef.current = window.setInterval(async () => {
            try {
              const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
              if (res.ok) {
                const j = await res.json();
                setState(j);
                // hydrate snapshots from history if any
                const snaps = [];
                (j.history||[]).forEach((h:any)=>{
                  if(h.snapshot_id) snaps.push({id:h.snapshot_id, agent:'', summary: h.katz||h.dogz||''});
                });
                setSnapshots(snaps);
              }
            } catch (e) {}
          }, 1000);
        }
      }
    }

    // ensure initial fetch and hydrate snapshots from history
    (async () => {
      if (debateId) {
        try {
          const res = await fetch(`${API_BASE}/api/debate/${debateId}/state`);
          if (res.ok) {
            const j = await res.json();
            setState(j);
            // hydrate snapshots from returned history entries
            const snaps: any[] = [];
            (j.history || []).forEach((h:any) => {
              // look for per-agent snapshot keys we embed on the backend
              ['katz_snapshot','dogz_snapshot'].forEach(k=>{
                if(h[k]) snaps.push({id: h[k].id, agent: k.startsWith('katz') ? 'katz' : 'dogz', summary: h[k].summary, ts: h[k].ts});
              });
            });
            if(snaps.length) setSnapshots(snaps);
          }
        } catch (e) {}
      }
    })();

    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      if (ws) { ws.close(); ws = null; }
    };
  }, [debateId]);

  // Snapshot modal helpers
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [snapshotModal, setSnapshotModal] = useState<{open:boolean,content?:string}>({open:false});

  async function viewSnapshot(id:string){
    try{
      const r = await fetch(`${API_BASE}/api/admin/memory/${id}`);
      if(r.ok){
        const j = await r.json();
        setSnapshotModal({open:true, content: j.content});
      }else{
        setSnapshotModal({open:true, content: 'Snapshot not found'});
      }
    }catch(e){ setSnapshotModal({open:true, content: 'Error fetching snapshot'}); }
  }

  async function captureFromCamera(){
    if(!debateId) return alert('Join or start a debate first');
    try{
      const res = await fetch(`${API_BASE}/api/debate/${debateId}/camera-capture`, {method: 'POST'});
      if(res.ok){
        const j = await res.json();
        // Build a small snapshot placeholder and prepend
        const s = { id: j.memory_id || j.image_saved, agent: 'camera', summary: j.memory_id ? 'Camera snapshot' : 'Image captured', ts: Date.now() };
        setSnapshots(xs => [s, ...xs]);
        // open modal directly with snapshot memory if available
        if(j.memory_id) {
          viewSnapshot(j.memory_id);
        }
      }else{
        alert('Camera capture failed');
      }
    }catch(e){ alert('Camera capture error'); }
  }

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
        <div style={{display:'flex', gap:12, marginTop:12}}>
          <div style={{flex:1, border: '1px solid #ddd', padding: 12, borderRadius: 6, maxHeight: 400, overflow: 'auto'}}>
            <h4>Timeline</h4>
            {history.length === 0 && <p className="muted">No rounds yet — start a debate or join one to see the timeline.</p>}
            {history.map((r: any, i:number) => (
              <div key={i} style={{marginBottom: 12}}>
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

          <div style={{width: 320, border: '1px solid #eee', padding: 12, borderRadius: 6}}>
            <h4>Snapshots</h4>
            <div style={{marginBottom: 8}}>
              <button onClick={async ()=>{ if(!debateId) return alert('Join or start a debate first'); await captureFromCamera(); }}>📸 Capture from camera</button>
            </div>
            {snapshots.length === 0 && <p className="muted">No snapshots yet.</p>}
            <ul>
              {snapshots.map(s=> (
                <li key={s.id} style={{marginBottom:8}}>
                  <div><strong>{s.agent}</strong> — {s.summary}</div>
                  <div style={{marginTop:6}}>
                    <button onClick={()=>viewSnapshot(s.id)}>View snapshot</button>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {snapshotModal.open && (
        <div style={{position:'fixed', left:0, top:0, right:0, bottom:0, background:'rgba(0,0,0,0.4)', display:'flex', alignItems:'center', justifyContent:'center'}}>
          <div style={{background:'#fff', padding:20, borderRadius:8, maxWidth:800}}>
            <h3>Snapshot details</h3>
            <div style={{whiteSpace:'pre-wrap', maxHeight: 400, overflow:'auto'}}>{snapshotModal.content}</div>
            <div style={{textAlign:'right', marginTop:12}}><button onClick={()=>setSnapshotModal({open:false})}>Close</button></div>
          </div>
        </div>
      )}
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
