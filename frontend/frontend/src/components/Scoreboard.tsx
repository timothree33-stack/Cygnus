import React from 'react';
import Tooltip from './Tooltip';
import '../styles/scoreboard.css';

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
                <td className={`sb-team-active`} style={{background: activeMembers.katz ? '#fff7e6' : undefined}}>
                  <Tooltip label={activeMembers.katz ? `Active: ${activeMembers.katz}` : 'Cats team'}>
                    <div style={{display:'flex', alignItems:'center', gap:8}}>
                      <strong>Cats</strong> <small style={{opacity:0.7}}>(Katz)</small>
                      {activeMembers.katz && <span className="sb-badge" aria-hidden>{'✨ ' + activeMembers.katz}</span>}
                    </div>
                  </Tooltip>
                </td>
                {rounds.map(r => {
                  const entry = history.find((h: any) => h.round === r) || {};
                  const val = entry.scores?.cats || entry.scores?.katz || 0;
                  const member = entry.katz_member;
                  <td key={r} className="sb-cell" style={{textAlign:'center'}} onClick={() => onOpenMember && onOpenMember(entry)}>
                    <Tooltip label={member ? `Speaker: ${member}` : `Round ${r}`}>{
                      <div>{val}</div>
                    }</Tooltip>
                  </td>
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
