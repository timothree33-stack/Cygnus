import React from 'react';

export default function MemberDetailPanel({ round, onClose }: { round: any; onClose: () => void }) {
  if (!round) return null;
  // round may have cats_members and dogs_members arrays with member objects {name, text, score}
  const cats = round.cats_members || [];
  const dogs = round.dogs_members || [];

  return (
    <div style={{position:'fixed', right:20, top:80, width:420, maxHeight: '70vh', overflow:'auto', background:'#fff', border:'1px solid #ccc', padding:12, borderRadius:8, boxShadow:'0 3px 12px rgba(0,0,0,0.2)'}}>
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <h4>Round {round.round} — Member Details</h4>
        <button onClick={onClose}>Close</button>
      </div>
      <div style={{display:'flex', gap:12}}>
        <div style={{flex:1}}>
          <h5>Cats</h5>
          {cats.length === 0 && <div className="muted">No member details</div>}
          {cats.map((m:any,i:number)=> (
            <div key={i} style={{padding:8, borderBottom:'1px solid #eee'}}>
              <div style={{fontWeight:700}}>{m.name || `Member ${i+1}`}</div>
              <div style={{fontSize:12, color:'#333'}}>{m.text}</div>
              <div style={{fontWeight:700, marginTop:4}}>Score: {m.score}</div>
            </div>
          ))}
        </div>
        <div style={{flex:1}}>
          <h5>Dogs</h5>
          {dogs.length === 0 && <div className="muted">No member details</div>}
          {dogs.map((m:any,i:number)=> (
            <div key={i} style={{padding:8, borderBottom:'1px solid #eee'}}>
              <div style={{fontWeight:700}}>{m.name || `Member ${i+1}`}</div>
              <div style={{fontSize:12, color:'#333'}}>{m.text}</div>
              <div style={{fontWeight:700, marginTop:4}}>Score: {m.score}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
