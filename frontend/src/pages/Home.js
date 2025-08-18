

import React, { useState } from "react";
import axios from "axios";
import "./Home.css";

export default function Home() {
  const [file, setFile] = useState(null);
  const [logs, setLogs] = useState(null);
  const [question, setQuestion] = useState("");
  const [chat, setChat] = useState([]);
  const [loadingQa, setLoadingQa] = useState(false);

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const exportReport = async () => {
  try {
    const res = await axios.post(
      "http://localhost:5000/export_report",
      {
        filename: file ? file.name.split(".")[0] : "logfile",
        logs: logs,
        chat: chat
      },
      { responseType: "blob" }
    );

    // download PDF
    const url = window.URL.createObjectURL(new Blob([res.data]));
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute(
      "download",
      `${file ? file.name.split(".")[0] : "logfile"}_report.pdf`
    );
    document.body.appendChild(link);
    link.click();
    link.remove();
  } catch (err) {
    alert("Failed to export report");
    console.error(err);
  }
};


  const handleUpload = async () => {
    if (!file) return alert("Upload a log file first.");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:5000/upload", formData);
      setLogs(res.data.gemini_insights || res.data);
    } catch (err) {
      alert("Upload failed.");
      console.error(err);
    }
  };

  async function ask() {
    if (!question.trim()) return;
    setLoadingQa(true);

    // add user message immediately
    setChat((prev) => [...prev, { sender: "user", text: question }]);

    try {
      const res = await axios.post("http://localhost:5000/query", { question });
      setChat((prev) => [
        ...prev,
        { sender: "bot", text: res.data.summary, full: res.data },
      ]);
    } catch (e) {
      alert("Query failed");
    } finally {
      setLoadingQa(false);
      setQuestion("");
    }
  }

  // Threat level colors
  const threatColors = {
    High: "red",
    Medium: "yellow",
    Low: "green",
    default: "gray",
  };

  // Normalize array data
  const normalizeArray = (val) => {
    if (!val) return [];
    if (Array.isArray(val)) return val;
    return [val];
  };
  
  return (
    <div className="home-app-container">
      <div className="home-two-column">
        {/* LEFT PANEL */}
        <div className="home-left-panel">
          <h1 className="home-title">📂 Log Parser</h1>
          <input
            type="file"
            onChange={handleFileChange}
            className="home-file-input"
          />
          <button onClick={handleUpload} className="home-btn home-btn-green">
            Upload
          </button>

          {logs ? (
            <div className="home-summary-card">
              {logs.summary && (
                <div className="home-card">
                  <div className="home-card-header">
                    <h2>System Diagnostics</h2>
                    {logs.threat_level && (
                      <span
                        className={`home-badge ${
                          threatColors[logs.threat_level] || "default"
                        }`}
                      >
                        Threat: {logs.threat_level}
                      </span>
                    )}
                  </div>
                  <p>{logs.summary}</p>
                </div>
              )}

              {[
                { title: "Findings", items: normalizeArray(logs.findings), color: "blue" },
                { title: "Anomalies", items: normalizeArray(logs.anomalies), color: "red" },
                { title: "Suspicious Activities", items: normalizeArray(logs.suspicious), color: "orange" },
                { title: "Insights", items: normalizeArray(logs.insights), color: "yellow" },
                { title: "Recommendations", items: normalizeArray(logs.recommendations), color: "green" },
              ]
                .filter((section) => section.items.length > 0)
                .map((section, idx) => (
                  <div key={idx} className={`home-card border-${section.color}`}>
                    <div className="home-card-header">
                      <h2>{section.title}</h2>
                    </div>
                    <ul>
                      {section.items.map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ))}
            </div>
          ) : (
            <p className="home-placeholder">Upload a log file to see summary.</p>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div className="home-right-panel">
          <h2>💬 Ask Questions</h2>

          {/* Chat area */}
          <div className="home-chat-box">
            {chat.map((msg, idx) => (
              <div
                key={idx}
                className={`home-chat-bubble ${
                  msg.sender === "user" ? "user" : "bot"
                }`}
              >
                {msg.text}
                {msg.sender === "bot" && msg.full && (
                  <div className="home-bot-details">
                    {msg.full.findings?.length > 0 && (
                      <>
                        <strong>Findings:</strong>
                        <ul>
                          {msg.full.findings.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      </>
                    )}
                    {msg.full.recommendations?.length > 0 && (
                      <>
                        <strong>Recommendations:</strong>
                        <ul>
                          {msg.full.recommendations.map((r, i) => (
                            <li key={i}>{r}</li>
                          ))}
                        </ul>
                      </>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Input row */}
          <div className="home-input-row">
            <input
              className="home-text-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about your logs..."
            />
            <button
              onClick={ask}
              disabled={loadingQa}
              className="home-btn home-btn-blue"
            >
              {loadingQa ? "..." : "Send"}
            </button>
          </div>
          <button onClick={exportReport} className="home-btn home-btn-red">
          📄 Export Report
        </button>
        </div>
      </div>
    </div>
  );
}
