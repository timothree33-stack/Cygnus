import React, { useState, useRef } from 'react';

export default function Tooltip({ children, label }: { children: React.ReactNode; label: string }) {
  const [visible, setVisible] = useState(false);
  const id = `tooltip-${Math.random().toString(36).slice(2,8)}`;
  return (
    <span style={{ position: 'relative', display: 'inline-block' }} onMouseEnter={() => setVisible(true)} onMouseLeave={() => setVisible(false)} onFocus={() => setVisible(true)} onBlur={() => setVisible(false)}>
      {React.Children.only(children)}
      <div role="tooltip" id={id} aria-hidden={!visible} style={{
        position: 'absolute',
        top: 'calc(100% + 6px)',
        left: '50%',
        transform: 'translateX(-50%)',
        background: '#111',
        color: '#fff',
        padding: '6px 8px',
        fontSize: 12,
        borderRadius: 6,
        whiteSpace: 'nowrap',
        pointerEvents: 'none',
        opacity: visible ? 1 : 0,
        transition: 'opacity 160ms ease, transform 160ms ease',
        zIndex: 1000,
        boxShadow: '0 4px 10px rgba(0,0,0,0.15)'
      }}>{label}</div>
    </span>
  );
}
