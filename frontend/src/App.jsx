import { useState, useEffect, useRef } from "react"
import axios from "axios"
import { SpeedInsights } from "@vercel/speed-insights/react"

const API = "http://localhost:8000"
const GRN = "#76B900"

// ── Badge component ───────────────────────────────────────────
function Badge({ decision }) {
  const cfg = {
    retrieve: { color: "#76B900", bg: "#0d1a00" },
    meta:     { color: "#a855f7", bg: "#1a0a2e" },
    clarify:  { color: "#f59e0b", bg: "#1a1200" },
  }
  const { color, bg } = cfg[decision?.toLowerCase()] || { color: "#888", bg: "#111" }
  return (
    <span style={{
      background: bg, color, border: `1px solid ${color}33`,
      borderRadius: 4, padding: "2px 8px", fontSize: 11,
      fontFamily: "JetBrains Mono, monospace", fontWeight: 600,
      letterSpacing: "0.8px", marginTop: 6, display: "inline-block"
    }}>
      {decision?.toUpperCase()}
    </span>
  )
}

// ── Upload Progress Bar ───────────────────────────────────────
function UploadProgress({ progress, filename, totalChunks }) {
  return (
    <div style={{
      background: "#1e2a0e", border: `1px solid ${GRN}44`,
      borderRadius: 10, padding: 14, marginBottom: 8
    }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    alignItems: "center", marginBottom: 8 }}>
        <span style={{ color: "#c1cab1", fontSize: 12, fontWeight: 600,
                       maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis",
                       whiteSpace: "nowrap" }}>
          {filename}
        </span>
        <span style={{ color: GRN, fontSize: 11,
                       fontFamily: "JetBrains Mono, monospace" }}>
          {Math.round(progress)}%
        </span>
      </div>

      {/* Progress bar */}
      <div style={{ height: 4, background: "#2a2a2a", borderRadius: 4, overflow: "hidden" }}>
        <div style={{
          height: "100%", borderRadius: 4, background: GRN,
          width: `${progress}%`,
          transition: "width 0.3s ease",
          boxShadow: `0 0 8px ${GRN}88`
        }} />
      </div>

      <div style={{ display: "flex", justifyContent: "space-between",
                    marginTop: 8, alignItems: "center" }}>
        <span style={{ color: "#6b7260", fontSize: 10 }}>
          {progress < 30 ? "Extracting text…" :
           progress < 60 ? "Chunking document…" :
           progress < 90 ? "Embedding & indexing…" :
           "Finalizing…"}
        </span>
        {totalChunks > 0 && (
          <span style={{ color: "#6b7260", fontSize: 10,
                         fontFamily: "JetBrains Mono, monospace" }}>
            ~{totalChunks} chunks
          </span>
        )}
      </div>
    </div>
  )
}

// ── Sidebar ───────────────────────────────────────────────────
function Sidebar({ onUpload, docInfo, onNewChat, onBackHome }) {
  const [dragging, setDragging]     = useState(false)
  const [uploading, setUploading]   = useState(false)
  const [progress, setProgress]     = useState(0)
  const [uploadFile, setUploadFile] = useState(null)
  const fileRef = useRef()

  const handleFile = async (file) => {
    if (!file || !file.name.endsWith(".pdf")) return
    setUploading(true)
    setProgress(0)
    setUploadFile(file.name)

    // Animate progress while uploading (simulated stages)
    let pct = 0
    const tick = setInterval(() => {
      pct += Math.random() * 4
      if (pct >= 88) { clearInterval(tick) }
      setProgress(Math.min(pct, 88))
    }, 120)

    const form = new FormData()
    form.append("file", file)
    try {
      const res = await axios.post(`${API}/upload`, form)
      clearInterval(tick)
      setProgress(100)
      setTimeout(() => {
        setUploading(false)
        setProgress(0)
        setUploadFile(null)
        onUpload(res.data)
      }, 600)
    } catch (e) {
      clearInterval(tick)
      setUploading(false)
      setProgress(0)
      setUploadFile(null)
      alert("Upload failed: " + e.message)
    }
  }

  return (
    <aside style={{
      width: 280, minWidth: 280, background: "#1c1b1b",
      borderRight: "1px solid #2a2a2a", height: "100vh",
      display: "flex", flexDirection: "column", padding: "18px 16px",
      overflowY: "auto", boxSizing: "border-box"
    }}>
      {/* Logo */}
      <div style={{ color: GRN, fontSize: 15, fontWeight: 700,
                    letterSpacing: "-0.3px", paddingBottom: 4 }}>
        Document Q&A Agent
      </div>
      <div style={{ color: "#6b7260", fontSize: 10, textTransform: "uppercase",
                    letterSpacing: "1.5px", paddingBottom: 12 }}>
        Nemotron · ChromaDB · LangGraph
      </div>

      <hr style={{ borderColor: "#2a2a2a", margin: "8px 0" }} />

      {/* Upload section */}
      <div style={{ color: "#6b7260", fontSize: 10, textTransform: "uppercase",
                    letterSpacing: "1.5px", marginBottom: 8 }}>
        Document
      </div>

      {/* Drop zone — hide when uploading */}
      {!uploading && (
        <div
          onClick={() => fileRef.current.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]) }}
          style={{
            background: dragging ? "#1a3300" : "#2a2a2a",
            border: `1.5px dashed ${dragging ? GRN : "#444"}`,
            borderRadius: 10, padding: 18, cursor: "pointer",
            textAlign: "center", transition: "all 0.2s", marginBottom: 8
          }}
        >
          <div style={{ color: "#e5e2e1", fontSize: 13, fontWeight: 500, marginBottom: 4 }}>
            Drag & drop PDF
          </div>
          <div style={{ color: "#6b7260", fontSize: 11 }}>or click to browse</div>
          <input ref={fileRef} type="file" accept=".pdf" style={{ display: "none" }}
                 onChange={e => handleFile(e.target.files[0])} />
        </div>
      )}

      {/* Progress bar during upload */}
      {uploading && (
        <UploadProgress
          progress={progress}
          filename={uploadFile}
          totalChunks={0}
        />
      )}

      {/* Active doc info */}
      {docInfo && !uploading && (
        <>
          <div style={{
            background: "#2a2a2a", border: "1px solid #333",
            borderRadius: 10, padding: 12, marginBottom: 8
          }}>
            <div style={{ display: "flex", justifyContent: "space-between",
                          alignItems: "center", marginBottom: 8 }}>
              <span style={{ color: "#6b7260", fontSize: 10, textTransform: "uppercase",
                             letterSpacing: "1px" }}>Active document</span>
              <span style={{ color: GRN, fontSize: 10, fontWeight: 600 }}>● Ready</span>
            </div>
            <div style={{ color: "#e5e2e1", fontSize: 12, fontWeight: 600,
                          marginBottom: 4, wordBreak: "break-all" }}>
              {docInfo.filename}
            </div>
            <div style={{ color: "#6b7260", fontSize: 11, marginBottom: 10 }}>
              {docInfo.chunk_count} chunks indexed
            </div>

            {/* Mini progress bar showing fullness */}
            <div style={{ height: 3, background: "#1a1a1a", borderRadius: 2,
                          marginBottom: 10, overflow: "hidden" }}>
              <div style={{ height: "100%", width: "100%", background: GRN,
                            borderRadius: 2, opacity: 0.6 }} />
            </div>

            <div style={{ borderTop: "1px solid #333", paddingTop: 8 }}>
              {[["Chunk size", "500"], ["Top-K", "5"]].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between",
                                      marginBottom: 4 }}>
                  <span style={{ color: "#6b7260", fontSize: 10 }}>{k}</span>
                  <span style={{ color: "#e5e2e1", fontSize: 10,
                                 fontFamily: "JetBrains Mono, monospace" }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <button onClick={onNewChat} style={{
            background: GRN, color: "#000", border: "none",
            borderRadius: 8, fontWeight: 700, fontSize: 12,
            padding: "10px 14px", width: "100%", cursor: "pointer",
            marginBottom: 8, transition: "opacity 0.2s"
          }}
          onMouseOver={e => e.target.style.opacity = "0.85"}
          onMouseOut={e => e.target.style.opacity = "1"}
          >
            ＋ New conversation
          </button>
        </>
      )}

      <hr style={{ borderColor: "#2a2a2a", margin: "8px 0" }} />

      {/* Agent modes */}
      <div style={{ color: "#6b7260", fontSize: 10, textTransform: "uppercase",
                    letterSpacing: "1.5px", marginBottom: 10 }}>
        Agent modes
      </div>
      {[
        ["retrieve", "#76B900", "#0d1a00", "Search document"],
        ["meta",     "#a855f7", "#1a0a2e", "Conversation history"],
        ["clarify",  "#f59e0b", "#1a1200", "Ask for detail"],
      ].map(([label, color, bg, desc]) => (
        <div key={label} style={{ display: "flex", alignItems: "center",
                                   gap: 8, marginBottom: 7 }}>
          <span style={{ background: bg, color, border: `1px solid ${color}33`,
                         borderRadius: 4, padding: "2px 8px", fontSize: 10,
                         fontFamily: "JetBrains Mono, monospace", fontWeight: 600,
                         flexShrink: 0 }}>
            {label.toUpperCase()}
          </span>
          <span style={{ color: "#c1cab1", fontSize: 12 }}>{desc}</span>
        </div>
      ))}

      <hr style={{ borderColor: "#2a2a2a", margin: "8px 0" }} />

      {/* Eval results */}
      <div style={{ color: "#6b7260", fontSize: 10, textTransform: "uppercase",
                    letterSpacing: "1.5px", marginBottom: 10 }}>
        Eval results
      </div>
      {[["Eval set","6/6 · 100%"],["Retrieval","5/5 · 100%"],["Fallback","3/3 · 100%"]].map(([k,v]) => (
        <div key={k} style={{ display: "flex", justifyContent: "space-between",
                               marginBottom: 6 }}>
          <span style={{ color: "#c1cab1", fontSize: 12 }}>{k}</span>
          <span style={{ color: GRN, fontSize: 12,
                         fontFamily: "JetBrains Mono, monospace", fontWeight: 600 }}>{v}</span>
        </div>
      ))}

      <div style={{ marginTop: "auto", paddingTop: 16, color: "#444", fontSize: 10 }}>
        github.com/Jenny3306/Doc-QA-Agent
      </div>
    </aside>
  )
}

// ── Evidence Panel ────────────────────────────────────────────
function EvidencePanel({ chunks, confidence, trace }) {
  const [tab, setTab] = useState("citations")
  const tabs = ["citations", "chunks", "trace"]

  return (
    <aside style={{
      width: 320, minWidth: 320, background: "#1c1b1b",
      borderLeft: "1px solid #2a2a2a", height: "100vh",
      display: "flex", flexDirection: "column", padding: 18,
      overflowY: "auto", boxSizing: "border-box"
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
        <div style={{ width: 8, height: 8, background: GRN, borderRadius: "50%" }} />
        <span style={{ color: "#e5e2e1", fontSize: 15, fontWeight: 600 }}>Evidence</span>
      </div>

      {/* Tab bar */}
      <div style={{ display: "flex", gap: 4, marginBottom: 14,
                    borderBottom: "1px solid #2a2a2a", paddingBottom: 2 }}>
        {tabs.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "4px 10px", fontSize: 13, fontWeight: 600,
            color: tab === t ? GRN : "#6b7260",
            borderBottom: tab === t ? `2px solid ${GRN}` : "2px solid transparent",
            textTransform: "capitalize", transition: "color 0.2s"
          }}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Citations tab */}
      {tab === "citations" && (
        chunks.length > 0 ? chunks.slice(0, 3).map((chunk, i) => (
          <div key={i} style={{
            background: "#2a2a2a", borderLeft: `3px solid ${GRN}`,
            borderRadius: "0 8px 8px 0", padding: 12, marginBottom: 10
          }}>
            <div style={{ display: "flex", justifyContent: "space-between",
                          marginBottom: 8 }}>
              <span style={{ background: "#0d1a00", color: GRN, borderRadius: 4,
                             padding: "1px 7px", fontSize: 10, fontWeight: 700,
                             fontFamily: "JetBrains Mono, monospace" }}>
                Chunk {i + 1}
              </span>
              <span style={{ color: "#6b7260", fontSize: 10,
                             fontFamily: "JetBrains Mono, monospace" }}>
                {Math.max(0, confidence - i * 0.05).toFixed(2)}
              </span>
            </div>
            <div style={{ color: "#c1cab1", fontSize: 13, lineHeight: 1.6 }}>
              {chunk.slice(0, 180).replace(/\n/g, " ")}...
            </div>
          </div>
        )) : (
          <div style={{ color: "#6b7260", fontSize: 13, textAlign: "center",
                        paddingTop: 30 }}>
            Ask a question to see citations
          </div>
        )
      )}

      {/* Chunks tab */}
      {tab === "chunks" && (
        chunks.length > 0 ? chunks.map((chunk, i) => (
          <details key={i} style={{ marginBottom: 8 }}>
            <summary style={{ color: "#c1cab1", fontSize: 13, cursor: "pointer",
                              padding: "6px 10px", background: "#2a2a2a",
                              borderRadius: 6, listStyle: "none" }}>
              Chunk {i + 1} — {chunk.length} chars
            </summary>
            <div style={{ color: "#888", fontSize: 12, lineHeight: 1.6,
                          fontFamily: "JetBrains Mono, monospace",
                          padding: "8px 10px", background: "#1a1a1a",
                          borderRadius: "0 0 6px 6px", whiteSpace: "pre-wrap" }}>
              {chunk}
            </div>
          </details>
        )) : (
          <div style={{ color: "#6b7260", fontSize: 13, textAlign: "center",
                        paddingTop: 30 }}>
            No chunks retrieved yet
          </div>
        )
      )}

      {/* Trace tab */}
      {tab === "trace" && (
        trace.length > 0 ? trace.map((step, i) => (
          <div key={i} style={{ display: "flex", alignItems: "flex-start",
                                 gap: 10, marginBottom: 14, paddingLeft: 4 }}>
            <span style={{ color: step.color, fontSize: 13, minWidth: 16,
                           textAlign: "center", marginTop: 1 }}>
              {step.icon}
            </span>
            <div>
              <div style={{ color: "#e5e2e1", fontSize: 13, fontWeight: 600 }}>
                {step.label}
              </div>
              <div style={{ color: step.color, fontSize: 12,
                            fontFamily: "JetBrains Mono, monospace", marginTop: 2 }}>
                {step.value}
              </div>
            </div>
          </div>
        )) : (
          <div style={{ color: "#6b7260", fontSize: 13, textAlign: "center",
                        paddingTop: 30 }}>
            Agent trace will appear here
          </div>
        )
      )}
    </aside>
  )
}

// ── Chat Panel ────────────────────────────────────────────────
function ChatPanel({ messages, onSend, docInfo, onBackHome }) {
  const [input, setInput]   = useState("")
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSend = async () => {
    if (!input.trim() || loading) return
    const q = input.trim()
    setInput("")
    await onSend(q, setLoading)
  }

  if (!docInfo) {
    return (
      <main style={{ flex: 1, display: "flex", flexDirection: "column",
                     padding: "56px 56px 40px", background: "#131313",
                     overflow: "auto" }}>
        {/* Hero */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
            <div style={{ width: 8, height: 8, background: GRN, borderRadius: "50%" }} />
            <span style={{ color: GRN, fontSize: 12, textTransform: "uppercase",
                           letterSpacing: "2px", fontFamily: "JetBrains Mono, monospace" }}>
              System ready
            </span>
          </div>
          <h1 style={{ color: "#e5e2e1", fontSize: 48, fontWeight: 700, margin: "0 0 16px",
                       letterSpacing: "-1.5px", lineHeight: 1.1 }}>
            Document Q&A Agent
          </h1>
          <p style={{ color: "#c1cab1", fontSize: 18, margin: 0,
                      maxWidth: 540, lineHeight: 1.7 }}>
            Upload any PDF. Ask questions in plain English. Get grounded, cited
            answers powered by{" "}
            <span style={{ color: GRN, fontWeight: 600 }}>NVIDIA Nemotron</span>.
          </p>
        </div>

        {/* Stat cards */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
                      gap: 16, marginBottom: 28, maxWidth: 740 }}>
          {[["Model","Nemotron 3 Nano"],["Agent","LangGraph"],["Vector DB","ChromaDB"]].map(([k,v]) => (
            <div key={k} style={{ background: "#1c1b1b", border: "1px solid #2a2a2a",
                                   borderRadius: 12, padding: 20 }}>
              <div style={{ color: "#6b7260", fontSize: 11, textTransform: "uppercase",
                             letterSpacing: "1.2px", fontWeight: 600, marginBottom: 8 }}>
                {k}
              </div>
              <div style={{ color: GRN, fontSize: 20, fontWeight: 600,
                             fontFamily: "JetBrains Mono, monospace" }}>
                {v}
              </div>
            </div>
          ))}
        </div>

        {/* Feature cards */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
                      gap: 16, maxWidth: 740 }}>
          {[
            ["Semantic search",  "Finds content by meaning, not keywords"],
            ["Smart routing",    "LangGraph decides: search, summarize, or clarify"],
            ["Honest fallback",  "Refuses to hallucinate — verified 3/3 tests"],
          ].map(([title, desc]) => (
            <div key={title} style={{
              background: "#1c1b1b", border: "1px solid #2a2a2a",
              borderTop: `3px solid ${GRN}`, borderRadius: 12, padding: 20
            }}>
              <div style={{ color: GRN, fontSize: 20, marginBottom: 10 }}>◈</div>
              <div style={{ color: "#e5e2e1", fontSize: 15, fontWeight: 600,
                             marginBottom: 6 }}>{title}</div>
              <div style={{ color: "#6b7260", fontSize: 13, lineHeight: 1.7 }}>{desc}</div>
            </div>
          ))}
        </div>
      </main>
    )
  }

  return (
    <main style={{ flex: 1, display: "flex", flexDirection: "column",
                   background: "#131313", overflow: "hidden" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12,
                    padding: "16px 28px", borderBottom: "1px solid #2a2a2a",
                    flexShrink: 0, justifyContent: "space-between" }}>

        {/* Left side */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 7, height: 7, background: GRN, borderRadius: "50%" }} />
          <span style={{ color: "#e5e2e1", fontSize: 16, fontWeight: 600 }}>
            Ask your document
          </span>
          <span style={{ color: "#6b7260", fontSize: 13 }}>·</span>
          <span style={{ color: "#6b7260", fontSize: 13,
                         fontFamily: "JetBrains Mono, monospace",
                         maxWidth: 260, overflow: "hidden", textOverflow: "ellipsis",
                         whiteSpace: "nowrap" }}>
            {docInfo.filename}
          </span>
          <span style={{ color: "#6b7260", fontSize: 13 }}>·</span>
          <span style={{ color: "#6b7260", fontSize: 13,
                         fontFamily: "JetBrains Mono, monospace" }}>
            {docInfo.chunk_count} chunks
          </span>
        </div>

        {/* Right side — Back to home */}
        <button onClick={onBackHome} style={{
          background: "transparent",
          color: "#c1cab1",
          border: "1px solid #333",
          borderRadius: 8, padding: "6px 14px",
          fontSize: 12, fontWeight: 600,
          cursor: "pointer", transition: "all 0.2s",
          display: "flex", alignItems: "center", gap: 6,
          flexShrink: 0
        }}>
          ← Back to home
        </button>

      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "28px",
                    display: "flex", flexDirection: "column", gap: 16 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            display: "flex", flexDirection: msg.role === "user" ? "row-reverse" : "row",
            gap: 12, alignItems: "flex-start"
          }}>
            {/* Avatar */}
            <div style={{
              width: 36, height: 36, borderRadius: 9, flexShrink: 0,
              background: msg.role === "user" ? "#2a2a2a" : GRN,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 13, fontWeight: 700, color: msg.role === "user" ? "#e5e2e1" : "#000"
            }}>
              {msg.role === "user" ? "You" : "AI"}
            </div>

            {/* Bubble */}
            <div style={{ maxWidth: "72%" }}>
              <div style={{
                background: msg.role === "user" ? "#1a3300" : "#1c1b1b",
                border: `1px solid ${msg.role === "user" ? "#76B90033" : "#2a2a2a"}`,
                borderRadius: msg.role === "user" ? "14px 14px 0 14px" : "14px 14px 14px 0",
                padding: "14px 18px", color: "#e5e2e1", fontSize: 15, lineHeight: 1.7
              }}>
                {msg.content}
              </div>
              {msg.decision && <Badge decision={msg.decision} />}
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
            <div style={{ width: 36, height: 36, borderRadius: 9, background: GRN,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontSize: 13, fontWeight: 700, color: "#000" }}>AI</div>
            <div style={{ background: "#1c1b1b", border: "1px solid #2a2a2a",
                          borderRadius: "14px 14px 14px 0", padding: "14px 18px" }}>
              <div style={{ display: "flex", gap: 5 }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: 7, height: 7, borderRadius: "50%", background: GRN,
                    animation: `bounce 1s ease-in-out ${i * 0.15}s infinite`
                  }} />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div style={{ padding: "18px 28px", borderTop: "1px solid #2a2a2a", flexShrink: 0 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center",
                      background: "#1c1b1b", border: "1px solid #2a2a2a",
                      borderRadius: 14, padding: "10px 10px 10px 18px" }}>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
            placeholder="Ask a question about your document..."
            style={{
              flex: 1, background: "none", border: "none", outline: "none",
              color: "#e5e2e1", fontSize: 15, fontFamily: "DM Sans, sans-serif"
            }}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()} style={{
            background: input.trim() ? GRN : "#2a2a2a",
            color: input.trim() ? "#000" : "#555",
            border: "none", borderRadius: 9, width: 40, height: 40,
            cursor: input.trim() ? "pointer" : "default",
            fontSize: 18, fontWeight: 700, transition: "all 0.2s",
            display: "flex", alignItems: "center", justifyContent: "center",
            flexShrink: 0
          }}>
            ↑
          </button>
        </div>
        <div style={{ color: "#444", fontSize: 11, textAlign: "center", marginTop: 7 }}>
          Press Enter to send
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-6px); }
        }
      `}</style>
    </main>
  )
}

// ── Main App ──────────────────────────────────────────────────
export default function App() {
  const [messages,   setMessages]  = useState([])
  const [docInfo,    setDocInfo]   = useState(null)
  const [chunks,     setChunks]    = useState([])
  const [confidence, setConf]      = useState(0)
  const [trace,      setTrace]     = useState([])

  useEffect(() => {
    axios.get(`${API}/status`).then(res => {
      if (res.data.loaded) {
        setDocInfo({ filename: "Previous document", chunk_count: res.data.chunk_count })
      }
    }).catch(() => {})
  }, [])

  const handleUpload = (data) => {
    setDocInfo(data)
    setMessages([])
    setChunks([])
    setTrace([])
  }

  const handleNewChat = () => {
    setMessages([])
    setChunks([])
    setTrace([])
  }

  const handleSend = async (question, setLoading) => {
    setMessages(prev => [...prev, { role: "user", content: question }])
    setLoading(true)
    try {
      const chatHistory = messages.map(m => ({ role: m.role, content: m.content }))
      const res = await axios.post(`${API}/chat`, { question, chat_history: chatHistory })
      const { answer, decision, chunks, confidence, trace } = res.data
      setMessages(prev => [...prev, { role: "assistant", content: answer, decision }])
      setChunks(chunks)
      setConf(confidence)
      setTrace(trace)
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Error connecting to backend. Make sure the FastAPI server is running.",
        decision: "clarify"
      }])
    }
    setLoading(false)
  }

  const handleBackHome = () => {
    setDocInfo(null)
    setMessages([])
    setChunks([])
    setTrace([])
  }

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw",
                  background: "#131313", fontFamily: "DM Sans, sans-serif",
                  overflow: "hidden" }}>
      <Sidebar onUpload={handleUpload} docInfo={docInfo} onNewChat={handleNewChat} />
      <ChatPanel messages={messages} onSend={handleSend} docInfo={docInfo}
                 onBackHome={handleBackHome} />
      <EvidencePanel chunks={chunks} confidence={confidence} trace={trace} />
      <SpeedInsights />
    </div>
  )
}