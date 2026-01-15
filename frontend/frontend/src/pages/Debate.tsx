import React, { useState } from 'react';

export default function Debate() {
  const [foolEnabled, setFoolEnabled] = useState(false);
  return (
    <div>
      <h2>Debate Room</h2>
      <p>Debate flows and rounds will be displayed here.</p>
      <div style={{marginTop: 12}}>
        <label>
          <input type="checkbox" checked={foolEnabled} onChange={(e) => setFoolEnabled(e.target.checked)} /> Enable Court-Fool (Quips)
        </label>
        {foolEnabled && <p className="muted">Court-Fool enabled: occasional humor will be shown during debates (opt-in)</p>}
      </div>
    </div>
  );
}
