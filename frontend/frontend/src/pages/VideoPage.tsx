import React, { useState } from 'react';
import { API_BASE } from '../api';

export default function VideoPage(){
  const [debateId, setDebateId] = useState('');
  const [lastSnapshot, setLastSnapshot] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);

  async function capture(){
    if(!debateId) return alert('Enter debate id');
    setLoading(true);
    try{
      const res = await fetch(`${API_BASE}/api/debate/${encodeURIComponent(debateId)}/camera-capture`, { method: 'POST' });
      if(res.ok){
        const j = await res.json();
        setLastSnapshot(j);
      } else {
        alert('Capture failed');
      }
    }catch(e){ alert('Error capturing'); }
    setLoading(false);
  }

  return (
    <div>
      <h2>Video / Camera</h2>
      <p>Preview and capture snapshots from a debate camera (dev stub-friendly).</p>
      <div style={{marginBottom:12}}>
        <input placeholder="Enter debate id" value={debateId} onChange={(e)=>setDebateId(e.target.value)} />
        <button onClick={capture} disabled={loading} style={{marginLeft:8}}>{loading ? 'Capturing...' : 'Capture'}</button>
      </div>
      {lastSnapshot ? (
        <div>
          <h4>Last snapshot</h4>
          <pre style={{background:'#f8f8f8', padding:8}}>{JSON.stringify(lastSnapshot, null, 2)}</pre>
        </div>
      ) : (
        <p className="muted">No snapshots yet</p>
      )}
    </div>
  );
}
