import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { API_BASE, WS_PATH } from './api';

interface DebateEvent {
  type: string;
  agent?: string;
  content?: string;
  round?: number;
  katz_score?: number;
  dogz_score?: number;
  winner?: string;
  debate?: { topic: string; katz_position: string; dogz_position: string };
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'debate' | 'chat' | 'schedule'>('debate');
  
  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">🦅 <span>Cygnus Pyramid</span></div>
        <nav className="tabs">
          <button className={activeTab === 'debate' ? 'active' : ''} onClick={() => setActiveTab('debate')}>
            ⚔️ Arena
          </button>
          <button className={activeTab === 'chat' ? 'active' : ''} onClick={() => setActiveTab('chat')}>
            💬 Cygnus
          </button>
          <button className={activeTab === 'schedule' ? 'active' : ''} onClick={() => setActiveTab('schedule')}>
            📋 Schedule
          </button>
        </nav>
      </header>
      
      <main className="app-main">
        {activeTab === 'debate' && <DebateArena />}
        {activeTab === 'chat' && <CygnusChat />}
        {activeTab === 'schedule' && <ScheduleView />}
      </main>
    </div>
  );
}

function DebateArena() {
  const [customTopic, setCustomTopic] = useState('');
  const [events, setEvents] = useState<DebateEvent[]>([]);
  const [isDebating, setIsDebating] = useState(false);
  const [scores, setScores] = useState({ katz: 0, dogz: 0 });
  const [round, setRound] = useState(0);
  const [winner, setWinner] = useState<string | null>(null);
  const [currentTopic, setCurrentTopic] = useState('');
  const [positions, setPositions] = useState({ katz: '', dogz: '' });
  const [cygnusSays, setCygnusSays] = useState('Welcome to the arena. Enter a topic or click Random to begin.');
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let shouldStop = false;
    let backoff = 1000;

    const getWsUrl = () => {
      // If API_BASE is configured (dev/prod), derive websocket url from it; otherwise use relative path
      if (import.meta.env.VITE_API_BASE) {
        return `${API_BASE.replace(/^http/, 'ws')}/ws`;
      }
      return '/ws';
    };

    const connect = async () => {
      if (shouldStop) return;

      // Try relative path first (works with Vite proxy in dev). If it fails to open within
      // 3s, fall back to direct backend websocket url to avoid proxy issues.
      const tryUrl = getWsUrl();
      const fallbackUrl = (import.meta.env.VITE_API_BASE ? `${API_BASE.replace(/^http/, 'ws')}${WS_PATH}` : `ws://localhost:8001${WS_PATH}`);

      const tryOnce = (url: string) => new Promise<WebSocket>((resolve, reject) => {
        const s = new WebSocket(url);
        const timer = setTimeout(() => {
          s.close();
          reject(new Error('timeout'));
        }, 3000);
        s.onopen = () => { clearTimeout(timer); resolve(s); };
        s.onerror = (e) => { clearTimeout(timer); reject(e); };
      });

      try {
        ws = await tryOnce(tryUrl);
      } catch (e) {
        console.warn('Relative WS failed, falling back to direct backend WS:', e);
        try {
          ws = await tryOnce(fallbackUrl);
        } catch (e2) {
          console.warn('Direct WS fallback failed, will retry later:', e2);
          // Schedule reconnect with backoff
          if (shouldStop) return;
          setTimeout(connect, backoff);
          backoff = Math.min(30000, backoff * 2);
          return;
        }
      }

      ws.onopen = () => {
        console.log('WebSocket connected');
        backoff = 1000;
      };

      ws.onmessage = (e) => handleWS(JSON.parse(e.data));

      ws.onclose = () => {
        if (shouldStop) return;
        console.log(`WebSocket disconnected, reconnecting in ${backoff}ms`);
        setTimeout(connect, backoff);
        backoff = Math.min(30000, backoff * 2);
      };

      ws.onerror = (e) => {
        console.warn('WebSocket error', e);
        ws?.close();
      };

      wsRef.current = ws;
    };

    connect();

    return () => {
      shouldStop = true;
      ws?.close();
    };
  }, []);

  const handleWS = (data: DebateEvent) => {
    console.log('WS:', data);
    
    if (data.type === 'debate_started') {
      setIsDebating(true);
      setScores({ katz: 0, dogz: 0 });
      setRound(0);
      setWinner(null);
      setEvents([]);
      if (data.debate) {
        setCurrentTopic(data.debate.topic);
        setPositions({ katz: data.debate.katz_position, dogz: data.debate.dogz_position });
      }
      setCygnusSays('The debate begins! Agents, take your positions.');
    }
    
    if (data.type === 'round_start') {
      setRound(data.round || 0);
      setCygnusSays(`Round ${data.round}! Present your arguments.`);
    }
    
    if (data.type === 'argument') {
      setEvents(prev => [...prev, data]);
      setCygnusSays(data.agent === 'KatZ' 
        ? 'KatZ makes their case... DogZ, respond!'
        : 'DogZ counters... Interesting argument.');
    }
    
    if (data.type === 'round_score') {
      setScores({ katz: data.katz_score || 0, dogz: data.dogz_score || 0 });
    }
    
    if (data.type === 'debate_ended') {
      setIsDebating(false);
      setWinner(data.winner || null);
      setCygnusSays(data.winner === 'TIE' 
        ? 'A draw! Both made compelling cases.'
        : `${data.winner} wins! Well argued.`);
    }
  };

  const startDebate = async (useRandom = false) => {
    const topic = useRandom ? '' : customTopic.trim();
    const url = topic 
      ? `${API_BASE}/api/debate/start?topic=${encodeURIComponent(topic)}`
      : `${API_BASE}/api/debate/start?scenario_index=${Math.floor(Math.random() * 100)}`;
    try {
      const res = await fetch(url, { method: 'POST' });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Failed to start debate: ${res.status} ${text}`);
      }
      setCustomTopic('');
    } catch (e: any) {
      console.error(e);
      setCygnusSays(`Error starting debate: ${e.message}`);
    }
  };

  const katzArgs = events.filter(e => e.agent === 'KatZ');
  const dogzArgs = events.filter(e => e.agent === 'DogZ');

  return (
    <div className="arena">
      {/* Topic Input */}
      <div className="topic-input">
        <input
          value={customTopic}
          onChange={(e) => setCustomTopic(e.target.value)}
          placeholder="Enter a debate topic..."
          disabled={isDebating}
          onKeyPress={(e) => e.key === 'Enter' && startDebate()}
        />
        <button onClick={() => startDebate()} disabled={isDebating || !customTopic.trim()}>
          Start
        </button>
        <button onClick={() => startDebate(true)} disabled={isDebating}>
          🎲 Random
        </button>
      </div>

      {/* Current Topic Display */}
      {currentTopic && (
        <div className="current-topic">
          <strong>TOPIC:</strong> {currentTopic}
        </div>
      )}

      {/* Cygnus Judge - Center Top */}
      <div className="judge-box">
        <div className="judge-avatar">🦉</div>
        <div className="judge-speech">
          <div className="judge-name">CYGNUS</div>
          <div className="judge-says">{cygnusSays}</div>
        </div>
      </div>

      {/* Scoreboard */}
      <div className="scoreboard">
        <div className="score-left">
          <span className="score-num">{scores.katz}</span>
          <span className="score-name">KatZ</span>
        </div>
        <div className="score-center">
          {isDebating ? `ROUND ${round}` : winner ? `🏆 ${winner}` : 'VS'}
        </div>
        <div className="score-right">
          <span className="score-name">DogZ</span>
          <span className="score-num">{scores.dogz}</span>
        </div>
      </div>

      {/* Split Screen Battle */}
      <div className="battle-split">
        {/* KatZ Podium - Left */}
        <div className="podium katz-podium">
          <div className="podium-header">
            <span className="podium-avatar">🐱</span>
            <div className="podium-info">
              <div className="podium-name">KatZ</div>
              <div className="podium-position">{positions.katz || 'Thesis Agent'}</div>
            </div>
          </div>
          <div className="podium-content">
            {katzArgs.map((e, i) => (
              <div key={i} className="argument-card">
                <div className="arg-round">Round {e.round}</div>
                <div className="arg-text">{e.content}</div>
              </div>
            ))}
            {katzArgs.length === 0 && !isDebating && (
              <div className="waiting">Awaiting debate...</div>
            )}
            {isDebating && katzArgs.length === dogzArgs.length && (
              <div className="thinking">🐱 Thinking...</div>
            )}
          </div>
        </div>

        {/* Center Divider */}
        <div className="center-divider">
          <div className="vs-circle">⚔️</div>
          {!isDebating && !winner && (
            <button className="big-start" onClick={() => startDebate(true)}>
              START<br/>DEBATE
            </button>
          )}
          {winner && (
            <button className="big-start" onClick={() => startDebate(true)}>
              NEW<br/>DEBATE
            </button>
          )}
        </div>

        {/* DogZ Podium - Right */}
        <div className="podium dogz-podium">
          <div className="podium-header">
            <div className="podium-info right">
              <div className="podium-name">DogZ</div>
              <div className="podium-position">{positions.dogz || 'Antithesis Agent'}</div>
            </div>
            <span className="podium-avatar">🐕</span>
          </div>
          <div className="podium-content">
            {dogzArgs.map((e, i) => (
              <div key={i} className="argument-card">
                <div className="arg-round">Round {e.round}</div>
                <div className="arg-text">{e.content}</div>
              </div>
            ))}
            {dogzArgs.length === 0 && !isDebating && (
              <div className="waiting">Awaiting debate...</div>
            )}
            {isDebating && dogzArgs.length < katzArgs.length && (
              <div className="thinking">🐕 Thinking...</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CygnusChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || loading) return;
    
    const msg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: msg }]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat?message=${encodeURIComponent(msg)}`, { method: 'POST' });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.response || 'No response' }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection error' }]);
    }
    setLoading(false);
  };

  return (
    <div className="chat">
      <div className="chat-head">
        <span>🦉</span>
        <div><strong>Cygnus</strong><br/><small>The Seeker Librarian</small></div>
      </div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div>🦉</div>
            <p>I am Cygnus. Ask me anything.</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className="msg-bubble">{m.content}</div>
          </div>
        ))}
        {loading && <div className="chat-msg assistant"><div className="msg-bubble">Thinking...</div></div>}
        <div ref={endRef} />
      </div>
      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && send()}
          placeholder="Ask Cygnus..."
        />
        <button onClick={send}>Send</button>
      </div>
    </div>
  );
}

function ScheduleView() {
  const [topics, setTopics] = useState<string[]>([]);
  
  useEffect(() => {
    fetch(`${API_BASE}/api/topics`)
      .then(r => r.json())
      .then(d => setTopics(d.topics || []))
      .catch(() => {});
  }, []);

  const startTopic = async (topic: string) => {
    await fetch(`${API_BASE}/api/debate/start?topic=${encodeURIComponent(topic)}`, { method: 'POST' });
  };

  return (
    <div className="schedule">
      <h2>📋 Topic Library ({topics.length} topics)</h2>
      <div className="topic-grid">
        {topics.map((t, i) => (
          <div key={i} className="topic-card" onClick={() => startTopic(t)}>
            <span className="topic-num">#{i + 1}</span>
            <span className="topic-text">{t}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
