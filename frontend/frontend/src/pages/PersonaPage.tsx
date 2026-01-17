import React, { useState, useEffect } from 'react';
import { API_BASE } from '../api';

export default function PersonaPage(){
  const [agentId, setAgentId] = useState('cygnus');
  const [entries, setEntries] = useState<any[]>([]);
  const [input, setInput] = useState('');

  async function load(){
    try{
      const res = await fetch(`${API_BASE}/api/admin/agents/${encodeURIComponent(agentId)}/persona`);
      if(res.ok){
        const j = await res.json();
        setEntries(j.persona || []);
      }
    }catch(e){ }
  }

  useEffect(()=>{ load(); }, [agentId]);

  async function add(){
    if(!input) return;
    try{
      const res = await fetch(`${API_BASE}/api/admin/agents/${encodeURIComponent(agentId)}/persona`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: input }) });
      if(res.ok){ const j = await res.json(); setEntries((e)=> [{ id: j.saved, content: input, source:'persona' }, ...e]); setInput(''); }
    }catch(e){}
  }

  async function del(id:string){
    try{
      const res = await fetch(`${API_BASE}/api/admin/memory/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if(res.ok){ setEntries((e)=> e.filter(x=>x.id!==id)); }
    }catch(e){}
  }

  return (
    <div>
      <h2>Persona Manager</h2>
      <p>Manage persona entries for an agent.</p>
      <div style={{marginBottom:12}}>
        <label>Agent: <input value={agentId} onChange={(e)=>setAgentId(e.target.value)} /></label>
        <button onClick={load} style={{marginLeft:8}}>Load</button>
      </div>
      <div style={{marginBottom:12}}>
        <textarea value={input} onChange={(e)=>setInput(e.target.value)} placeholder="Add persona entry..." />
        <div><button onClick={add}>Add</button></div>
      </div>
      <div>
        <h4>Entries</h4>
        {entries.length===0 && <p className="muted">No persona entries</p>}
        <ul>
          {entries.map(p=> (<li key={p.id || p.created_at}>{p.content||p.text} {p.id && <button onClick={()=>del(p.id)}>Remove</button>}</li>))}
        </ul>
      </div>
    </div>
  );
}
