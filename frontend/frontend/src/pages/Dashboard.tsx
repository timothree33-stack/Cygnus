import React from 'react';
import ConversationBox from '../components/ConversationBox';

export default function Dashboard() {
  return (
    <div>
      <h2>Dashboard</h2>
      <p>Monitoring widgets (memories, MCP metrics) will appear here.</p>
      <ConversationBox agentId="dashboard" personality="Observant" />
    </div>
  );
}
