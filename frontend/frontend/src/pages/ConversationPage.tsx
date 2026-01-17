import React, { useState } from 'react';
import ConversationBox from '../components/ConversationBox';

export default function ConversationPage() {
  const [agent, setAgent] = useState('cygnus');
  const [personality, setPersonality] = useState('Cygnus');

  return (
    <div>
      <h2>Conversation</h2>
      <p>Select an agent to chat with and manage persona/context.</p>
      <div style={{marginBottom:12}}>
        <label>Agent ID: <input value={agent} onChange={(e)=>setAgent(e.target.value)} /></label>
        <label style={{marginLeft:12}}>Personality: <input value={personality} onChange={(e)=>setPersonality(e.target.value)} /></label>
      </div>
      <ConversationBox agentId={agent} personality={personality} />
    </div>
  );
}
