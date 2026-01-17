import React from 'react';
import ConversationBox from '../components/ConversationBox';

export default function Chat() {
  return (
    <div>
      <h2>Chat with Cygnus</h2>
      <p>Simple chat window for the Cygnus agent.</p>
      <ConversationBox agentId="cygnus" personality="Cygnus" />
    </div>
  );
}
