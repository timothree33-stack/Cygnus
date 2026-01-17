import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Agents from './pages/Agents';
import Crawler from './pages/Crawler';
import Debate from './pages/Debate';
import Chat from './pages/Chat';
import ConversationPage from './pages/ConversationPage';
import VideoPage from './pages/VideoPage';
import PersonaPage from './pages/PersonaPage';

export default function App() {
  return (
    <div className="app">
      <nav>
        <Link to="/">Home</Link> |
        <Link to="/dashboard">Dashboard</Link> |
        <Link to="/agents">Agents</Link> |
        <Link to="/crawler">Crawler</Link> |
        <Link to="/debate">Debate</Link> |
        <Link to="/chat">Chat</Link> |
        <Link to="/conversation">Conversation</Link> |
        <Link to="/video">Video</Link> |
        <Link to="/persona">Persona</Link>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<div><h1>Cygnus Pyramid</h1><p>Welcome to the executor portal.</p></div>} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/crawler" element={<Crawler />} />
          <Route path="/debate" element={<Debate />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/conversation" element={<ConversationPage />} />
          <Route path="/video" element={<VideoPage />} />
          <Route path="/persona" element={<PersonaPage />} />
        </Routes>
      </main>
    </div>
  );
}
