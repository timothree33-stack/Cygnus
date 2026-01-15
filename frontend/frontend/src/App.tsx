import React from 'react';
import { Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Agents from './pages/Agents';
import Crawler from './pages/Crawler';
import Debate from './pages/Debate';

export default function App() {
  return (
    <div className="app">
      <nav>
        <Link to="/">Home</Link> |
        <Link to="/dashboard">Dashboard</Link> |
        <Link to="/agents">Agents</Link> |
        <Link to="/crawler">Crawler</Link> |
        <Link to="/debate">Debate</Link>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<div><h1>Cygnus Pyramid</h1><p>Welcome to the executor portal.</p></div>} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/crawler" element={<Crawler />} />
          <Route path="/debate" element={<Debate />} />
        </Routes>
      </main>
    </div>
  );
}
