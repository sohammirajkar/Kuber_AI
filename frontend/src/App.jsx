import React, { useState } from 'react'

export default function App() {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

  async function send() {
    if (!query) return
    const userMsg = { from: 'user', text: query }
    setMessages((m) => [...m, userMsg])
    setLoading(true)
    try {
      const resp = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context_k: 3 })
      })
      const data = await resp.json()
      setMessages((m) => [...m, { from: 'kuber', text: data.answer, contexts: data.contexts }])
    } catch (e) {
      setMessages((m) => [...m, { from: 'kuber', text: 'Error: ' + e.message }])
    } finally {
      setLoading(false)
      setQuery('')
    }
  }

  return (
    <div style={{ fontFamily: 'Inter, Arial, sans-serif', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* HAT UI - top bar */}
      <header style={{ background: '#0f172a', color: '#fff', padding: '12px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 36, height: 36, borderRadius: 6, background: '#111827', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>K</div>
          <div>
            <div style={{ fontSize: 18 }}>Kuber</div>
            <div style={{ fontSize: 12, opacity: 0.7 }}>Investment AI agent — demo</div>
          </div>
        </div>
        <nav style={{ display: 'flex', gap: 12 }}>
          <button style={{ background: 'transparent', color: '#9CA3AF', border: '1px solid #1F2937', padding: '8px 12px', borderRadius: 8 }}>Dashboards</button>
          <button style={{ background: 'transparent', color: '#9CA3AF', border: '1px solid #1F2937', padding: '8px 12px', borderRadius: 8 }}>Analytics</button>
          <button style={{ background: '#06B6D4', color: '#052B2A', padding: '8px 12px', borderRadius: 8 }}>Connect Data</button>
        </nav>
      </header>

      {/* Chat area */}
      <main style={{ flex: 1, padding: 20, background: '#F8FAFC' }}>
        <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {messages.map((m, i) => (
            <div key={i} style={{ alignSelf: m.from === 'user' ? 'flex-end' : 'flex-start', background: m.from === 'user' ? '#111827' : '#fff', color: m.from === 'user' ? '#fff' : '#000', padding: 12, borderRadius: 8, boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
              <div style={{ fontSize: 14 }}>{m.text}</div>
              {m.contexts && (
                <details style={{ marginTop: 8 }}>
                  <summary>contexts</summary>
                  <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(m.contexts, null, 2)}</pre>
                </details>
              )}
            </div>
          ))}

          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Ask Kuber (e.g. compute exposure for my positions)..." style={{ flex: 1, padding: 12, borderRadius: 8, border: '1px solid #E5E7EB' }} />
            <button onClick={send} disabled={loading} style={{ padding: '10px 16px', borderRadius: 8, background: '#111827', color: '#fff' }}>{loading ? '...' : 'Send'}</button>
          </div>
        </div>
      </main>

      <footer style={{ padding: 12, textAlign: 'center', fontSize: 12, color: '#6B7280' }}>Kuber demo • not for production use</footer>
    </div>
  )
}
