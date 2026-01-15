import React from 'react';
import ConversationBox from '../components/ConversationBox';

export default function Agents() {
  return (
    <div>
      <h2>Agents</h2>
      <p>List of agents and conversation boxes.</p>
      <ConversationBox agentId="agent-1" personality="Helpful" />
    </div>
  );
}
