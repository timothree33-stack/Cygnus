import React from 'react';

export default function Scoreboard({ history, onOpenMember, activeMembers }: { history: any[]; onOpenMember?: (round: any) => void; activeMembers?: any }) {
  // Detect rounds
  const rounds = (history || []).map(h => h.round).filter(r => r !== undefined);
  if (!rounds.length) return <div className="scoreboard muted">No rounds to show.</div>;

  activeMembers = activeMembers || {};

  // Build per-team rows if members exist, otherwise fallback to Katz/Dogz
  const hasMembers = (history || []).some(h => Array.isArray(h.cats_members) || Array.isArray(h.dogs_members));

  if (hasMembers) {
    // Columns: Inning numbers
    return (
      <div className="scoreboard team-scoreboard">
        <div style={{overflowX:'auto'}}>
          <table className="sb-table">
            <thead>
              <tr>
                <th>Team</th>
                {rounds.map(r => <th key={r}>Inning {r}</th>)}
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td title={activeMembers.katz ? `Active: ${activeMembers.katz}` : ''} style={{background: activeMembers.katz ? '#fff7e6' : undefined}}><strong>Cats</strong> <small>(Katz)</small> {activeMembers.katz && <span style={{marginLeft:6, color:'#b36b00'}}>✨ {activeMembers.katz}</span>}</td>
                {rounds.map(r => {
                  const entry = history.find((h: any) => h.round === r) || {};
                  const val = entry.scores?.cats || entry.scores?.katz || 0;
                  const member = entry.katz_member;
                  return <td key={r} style={{textAlign:'center'}} onClick={() => onOpenMember && onOpenMember(entry)} title={member ? `Speaker: ${member}` : ''}>{val}</td>
                })}
                <td style={{fontWeight:700, textAlign:'center'}}>{rounds.reduce((acc, r) => { const e = history.find((h:any)=>h.round===r)||{}; return acc + (e.scores?.cats||e.scores?.katz||0); }, 0)}</td>
              </tr>
              <tr>
                <td title={activeMembers.dogz ? `Active: ${activeMembers.dogz}` : ''} style={{background: activeMembers.dogz ? '#f0f8ff' : undefined}}><strong>Dogs</strong> <small>(Dogz)</small> {activeMembers.dogz && <span style={{marginLeft:6, color:'#1b6ca8'}}>✨ {activeMembers.dogz}</span>}</td>
                {rounds.map(r => {
                  const entry = history.find((h: any) => h.round === r) || {};
                  const val = entry.scores?.dogs || entry.scores?.dogz || 0;
                  const member = entry.dogz_member;
                  return <td key={r} style={{textAlign:'center'}} onClick={() => onOpenMember && onOpenMember(entry)} title={member ? `Speaker: ${member}` : ''}>{val}</td>
                })}
                <td style={{fontWeight:700, textAlign:'center'}}>{rounds.reduce((acc, r) => { const e = history.find((h:any)=>h.round===r)||{}; return acc + (e.scores?.dogs||e.scores?.dogz||0); }, 0)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  // Fallback two-row scoreboard
  return (
    <div className="scoreboard">
      <table className="sb-table">
        <thead>
          <tr>
            <th>Team</th>
            {rounds.map(r => <th key={r}>Inning {r}</th>)}
            <th>Total</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Katz</strong></td>
            {rounds.map(r => {
              const entry = history.find((h: any) => h.round === r) || {};
              return <td key={r} style={{textAlign:'center'}}>{entry.scores?.katz || 0}</td>
            })}
            <td style={{fontWeight:700, textAlign:'center'}}>{rounds.reduce((acc, r) => acc + ((history.find((h:any)=>h.round===r)||{}).scores?.katz || 0), 0)}</td>
          </tr>
          <tr>
            <td><strong>Dogz</strong></td>
            {rounds.map(r => {
              const entry = history.find((h: any) => h.round === r) || {};
              return <td key={r} style={{textAlign:'center'}}>{entry.scores?.dogz || 0}</td>
            })}
            <td style={{fontWeight:700, textAlign:'center'}}>{rounds.reduce((acc, r) => acc + ((history.find((h:any)=>h.round===r)||{}).scores?.dogz || 0), 0)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
