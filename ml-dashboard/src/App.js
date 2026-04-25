import { useState, useEffect, useRef } from "react";

// ─── Palette & theme ───────────────────────────────────────────────────────
const C = {
  bg: "#0a0e1a",
  panel: "#111827",
  card: "#1a2235",
  border: "#1f2d45",
  accent: "#00d4ff",
  accent2: "#7c3aed",
  green: "#10b981",
  yellow: "#f59e0b",
  red: "#ef4444",
  text: "#e2e8f0",
  muted: "#64748b",
};

// ─── Mock data ─────────────────────────────────────────────────────────────
const ALGOS = {
  rf: {
    label: "Random Forest",
    icon: "🌳",
    color: "#10b981",
    desc: "Ensemble d'arbres de décision. Robuste, peu sensible aux outliers, excellente généralisation.",
    params: [
      { key: "n_estimators", label: "Nombre d'arbres", type: "range", min: 10, max: 500, default: 100, step: 10 },
      { key: "max_depth", label: "Profondeur max", type: "range", min: 1, max: 30, default: 15, step: 1 },
      { key: "min_samples_split", label: "Min samples split", type: "range", min: 2, max: 20, default: 2, step: 1 },
      { key: "criterion", label: "Critère", type: "select", options: ["gini", "entropy"], default: "gini" },
    ],
  },
  svm: {
    label: "SVM",
    icon: "⚡",
    color: "#f59e0b",
    desc: "Support Vector Machine. Optimal pour espaces de haute dimension, efficace avec peu de données.",
    params: [
      { key: "C", label: "Régularisation C", type: "range", min: 0.01, max: 10, default: 1, step: 0.01 },
      { key: "kernel", label: "Kernel", type: "select", options: ["rbf", "linear", "poly", "sigmoid"], default: "rbf" },
      { key: "gamma", label: "Gamma", type: "select", options: ["scale", "auto"], default: "scale" },
    ],
  },
  knn: {
    label: "KNN",
    icon: "🎯",
    color: "#00d4ff",
    desc: "K-Nearest Neighbors. Algorithme simple basé sur la distance. Très intuitif à interpréter.",
    params: [
      { key: "n_neighbors", label: "Nombre de voisins (k)", type: "range", min: 1, max: 30, default: 5, step: 1 },
      { key: "metric", label: "Métrique distance", type: "select", options: ["euclidean", "manhattan", "minkowski"], default: "euclidean" },
      { key: "weights", label: "Pondération", type: "select", options: ["uniform", "distance"], default: "uniform" },
    ],
  },
  lr: {
    label: "Logistic Regression",
    icon: "📈",
    color: "#7c3aed",
    desc: "Régression logistique. Rapide, interprétable, baseline solide pour classification.",
    params: [
      { key: "C", label: "Inverse régularisation", type: "range", min: 0.01, max: 10, default: 1, step: 0.01 },
      { key: "solver", label: "Solveur", type: "select", options: ["lbfgs", "liblinear", "saga"], default: "lbfgs" },
      { key: "max_iter", label: "Max iterations", type: "range", min: 100, max: 2000, default: 200, step: 100 },
    ],
  },
  nn: {
    label: "Neural Network",
    icon: "🧠",
    color: "#ec4899",
    desc: "MLP Classifier. Capture des patterns complexes. Idéal pour grands datasets.",
    params: [
      { key: "hidden_layer_sizes", label: "Couches cachées", type: "select", options: ["(100,)", "(100,50)", "(200,100,50)", "(64,32,16)"], default: "(100,)" },
      { key: "activation", label: "Activation", type: "select", options: ["relu", "tanh", "logistic"], default: "relu" },
      { key: "learning_rate_init", label: "Learning rate", type: "range", min: 0.0001, max: 0.1, default: 0.001, step: 0.0001 },
      { key: "max_iter", label: "Epochs", type: "range", min: 50, max: 500, default: 200, step: 50 },
    ],
  },
};

const DATASET_PREVIEW = [
  { id: 1, COMPACTNESS: 95, CIRCULARITY: 48, DISTANCE_CIRCULARITY: 83, RADIUS_RATIO: 178, CLASS: "van" },
  { id: 2, COMPACTNESS: 91, CIRCULARITY: 41, DISTANCE_CIRCULARITY: 84, RADIUS_RATIO: 141, CLASS: "van" },
  { id: 3, COMPACTNESS: 104, CIRCULARITY: 50, DISTANCE_CIRCULARITY: 106, RADIUS_RATIO: 209, CLASS: "saab" },
  { id: 4, COMPACTNESS: 85, CIRCULARITY: 44, DISTANCE_CIRCULARITY: 70, RADIUS_RATIO: 205, CLASS: "bus" },
  { id: 5, COMPACTNESS: 107, CIRCULARITY: 57, DISTANCE_CIRCULARITY: 106, RADIUS_RATIO: 172, CLASS: "bus" },
  { id: 6, COMPACTNESS: 97, CIRCULARITY: 43, DISTANCE_CIRCULARITY: 73, RADIUS_RATIO: 173, CLASS: "opel" },
];

const CLASS_COLORS = { van: "#00d4ff", saab: "#10b981", bus: "#f59e0b", opel: "#7c3aed" };

// ─── Fake training simulation ──────────────────────────────────────────────
function simulateTrain(algoKey, params, onProgress, onDone) {
  const steps = 20;
  let step = 0;
  const logs = [];
  const interval = setInterval(() => {
    step++;
    const pct = Math.round((step / steps) * 100);
    const base = algoKey === "nn" ? 0.78 : algoKey === "svm" ? 0.85 : algoKey === "rf" ? 0.826 : algoKey === "knn" ? 0.79 : 0.81;
    const noise = (Math.random() - 0.5) * 0.04;
    logs.push(`[${new Date().toLocaleTimeString()}] Epoch ${step}/${steps} — loss: ${(1 - base - noise * 0.5).toFixed(4)} acc: ${(base + noise * 0.5 + step / steps * 0.02).toFixed(4)}`);
    onProgress(pct, [...logs]);
    if (step >= steps) {
      clearInterval(interval);
      const acc = +(base + (Math.random() - 0.5) * 0.02).toFixed(4);
      const f1 = +(acc - 0.01 + (Math.random() - 0.5) * 0.01).toFixed(4);
      const results = {
        accuracy: acc,
        f1: f1,
        precision: +(f1 + 0.01).toFixed(4),
        recall: +(f1 - 0.005).toFixed(4),
        cv_mean: +(acc + 0.015).toFixed(4),
        cv_std: 0.0297,
        confusion: [
          [45, 2, 1, 0],
          [3, 47, 0, 2],
          [1, 0, 44, 3],
          [0, 2, 4, 41],
        ],
        roc_data: Array.from({ length: 10 }, (_, i) => ({ fpr: i / 9, tpr: Math.min(1, (i / 9) ** 0.6 + Math.random() * 0.05) })),
        feature_importance: [
          { name: "RADIUS_RATIO", value: 0.18 },
          { name: "ELONGATEDNESS", value: 0.15 },
          { name: "SCATTER_RATIO", value: 0.13 },
          { name: "COMPACTNESS", value: 0.11 },
          { name: "CIRCULARITY", value: 0.09 },
          { name: "HOLLOWS_RATIO", value: 0.08 },
          { name: "MAX_LENGTH_RECT", value: 0.07 },
          { name: "SKEWNESS_MAJOR", value: 0.06 },
        ],
      };
      onDone(results);
    }
  }, 120);
}

// ─── Sub-components ────────────────────────────────────────────────────────
function Badge({ text, color }) {
  return (
    <span style={{ background: color + "22", color, border: `1px solid ${color}44`, borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>
      {text}
    </span>
  );
}

function StatCard({ label, value, sub, color = C.accent }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 20px", flex: 1, minWidth: 120 }}>
      <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
      <div style={{ color, fontSize: 28, fontWeight: 800, fontFamily: "monospace" }}>{(value * 100).toFixed(2)}%</div>
      {sub && <div style={{ color: C.muted, fontSize: 11, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ data }) {
  const classes = ["van", "saab", "bus", "opel"];
  const max = Math.max(...data.flat());
  return (
    <div>
      <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, marginBottom: 10, textTransform: "uppercase" }}>Matrice de Confusion</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr>
              <th style={{ padding: "4px 8px", color: C.muted, fontSize: 11 }}>Pred →</th>
              {classes.map(c => <th key={c} style={{ padding: "4px 8px", color: CLASS_COLORS[c], fontSize: 11, fontWeight: 700 }}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                <td style={{ padding: "4px 8px", color: CLASS_COLORS[classes[i]], fontSize: 11, fontWeight: 700, whiteSpace: "nowrap" }}>↓ {classes[i]}</td>
                {row.map((val, j) => {
                  const intensity = val / max;
                  const isCorrect = i === j;
                  const bg = isCorrect ? `rgba(16,185,129,${0.15 + intensity * 0.6})` : val > 0 ? `rgba(239,68,68,${0.05 + intensity * 0.4})` : "transparent";
                  return (
                    <td key={j} style={{ padding: "8px 16px", textAlign: "center", background: bg, border: `1px solid ${C.border}`, borderRadius: 4, fontFamily: "monospace", fontSize: 14, fontWeight: isCorrect ? 800 : 400, color: isCorrect ? C.green : val > 0 ? C.red : C.muted }}>
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BarChart({ data, color }) {
  const max = Math.max(...data.map(d => d.value));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {data.map((d, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 130, fontSize: 11, color: C.muted, textAlign: "right", fontFamily: "monospace" }}>{d.name}</div>
          <div style={{ flex: 1, height: 18, background: C.border, borderRadius: 4, overflow: "hidden" }}>
            <div style={{ width: `${(d.value / max) * 100}%`, height: "100%", background: `linear-gradient(90deg, ${color}, ${color}88)`, borderRadius: 4, transition: "width 0.8s ease" }} />
          </div>
          <div style={{ width: 40, fontSize: 11, color, fontFamily: "monospace", fontWeight: 700 }}>{(d.value * 100).toFixed(0)}%</div>
        </div>
      ))}
    </div>
  );
}

function ProgressBar({ pct, color }) {
  return (
    <div style={{ width: "100%", height: 8, background: C.border, borderRadius: 4, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", background: `linear-gradient(90deg, ${color}, ${color}88)`, borderRadius: 4, transition: "width 0.15s ease", boxShadow: `0 0 12px ${color}88` }} />
    </div>
  );
}

function ROCCurve({ data, color }) {
  const W = 260, H = 180, PAD = 30;
  const pts = data.map(d => [PAD + d.fpr * (W - PAD * 2), H - PAD - d.tpr * (H - PAD * 2)]);
  const diag = `M${PAD},${H - PAD} L${W - PAD},${PAD}`;
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  return (
    <svg width={W} height={H} style={{ overflow: "visible" }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke={C.border} strokeWidth={1} />
      <line x1={PAD} y1={H - PAD} x2={PAD} y2={PAD} stroke={C.border} strokeWidth={1} />
      <path d={diag} stroke={C.muted} strokeWidth={1} strokeDasharray="4,4" fill="none" />
      <path d={path + ` L${W - PAD},${H - PAD} L${PAD},${H - PAD} Z`} fill={color + "22"} />
      <path d={path} stroke={color} strokeWidth={2.5} fill="none" />
      {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3} fill={color} />)}
      <text x={PAD} y={H - 5} fill={C.muted} fontSize={10}>FPR</text>
      <text x={5} y={PAD} fill={C.muted} fontSize={10} transform={`rotate(-90,5,${PAD})`}>TPR</text>
    </svg>
  );
}

// ─── AUTOML Panel ──────────────────────────────────────────────────────────
function AutoMLPanel({ onSelectBest }) {
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState([]);
  const [best, setBest] = useState(null);

  const run = () => {
    setRunning(true);
    setBest(null);
    setProgress([]);
    const algoKeys = Object.keys(ALGOS);
    let i = 0;
    const results = [];
    const runNext = () => {
      if (i >= algoKeys.length) {
        const sorted = results.sort((a, b) => b.acc - a.acc);
        setBest(sorted[0]);
        setRunning(false);
        return;
      }
      const key = algoKeys[i];
      const acc = +(0.75 + Math.random() * 0.12).toFixed(4);
      setTimeout(() => {
        results.push({ key, label: ALGOS[key].label, icon: ALGOS[key].icon, acc, color: ALGOS[key].color });
        setProgress([...results]);
        i++;
        runNext();
      }, 600 + Math.random() * 400);
    };
    runNext();
  };

  return (
    <div style={{ background: C.card, border: `1px solid ${C.accent2}44`, borderRadius: 12, padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <div style={{ color: C.text, fontWeight: 800, fontSize: 16 }}>🤖 Mode AutoML</div>
          <div style={{ color: C.muted, fontSize: 12, marginTop: 4 }}>Teste tous les algorithmes et sélectionne automatiquement le meilleur</div>
        </div>
        <button onClick={run} disabled={running} style={{ background: running ? C.border : `linear-gradient(135deg, ${C.accent2}, #9333ea)`, color: "#fff", border: "none", borderRadius: 8, padding: "10px 20px", fontWeight: 700, cursor: running ? "not-allowed" : "pointer", fontSize: 13 }}>
          {running ? "⏳ En cours..." : "▶ Lancer AutoML"}
        </button>
      </div>

      {progress.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {progress.map((r, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, background: C.panel, borderRadius: 8, padding: "10px 16px" }}>
              <span style={{ fontSize: 20 }}>{r.icon}</span>
              <span style={{ color: C.text, fontWeight: 600, width: 150 }}>{r.label}</span>
              <div style={{ flex: 1 }}><ProgressBar pct={r.acc * 100} color={r.color} /></div>
              <span style={{ fontFamily: "monospace", color: r.color, fontWeight: 800, width: 60, textAlign: "right" }}>{(r.acc * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}

      {best && (
        <div style={{ marginTop: 20, background: `linear-gradient(135deg, ${best.color}11, ${C.panel})`, border: `1px solid ${best.color}55`, borderRadius: 10, padding: 16 }}>
          <div style={{ color: C.text, fontWeight: 800, fontSize: 14 }}>🏆 Meilleur Modèle : <span style={{ color: best.color }}>{best.icon} {best.label}</span></div>
          <div style={{ color: C.muted, fontSize: 12, marginTop: 4 }}>Accuracy : <span style={{ color: best.color, fontWeight: 700 }}>{(best.acc * 100).toFixed(2)}%</span></div>
          <button onClick={() => onSelectBest(best.key)} style={{ marginTop: 12, background: best.color, color: "#fff", border: "none", borderRadius: 6, padding: "8px 16px", fontWeight: 700, cursor: "pointer", fontSize: 12 }}>
            Utiliser ce modèle →
          </button>
        </div>
      )}
    </div>
  );
}

// ─── History Panel ─────────────────────────────────────────────────────────
function HistoryPanel({ history, onRollback }) {
  if (!history.length) return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 24, color: C.muted, fontSize: 13, textAlign: "center" }}>
      Aucune expérimentation enregistrée.
    </div>
  );
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {history.map((h, i) => (
        <div key={i} style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 18px", display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 22 }}>{ALGOS[h.algo]?.icon}</span>
          <div style={{ flex: 1 }}>
            <div style={{ color: C.text, fontWeight: 700, fontSize: 13 }}>{ALGOS[h.algo]?.label} <Badge text={`v${h.version}`} color={ALGOS[h.algo]?.color} /></div>
            <div style={{ color: C.muted, fontSize: 11, marginTop: 2 }}>{h.time} — Acc: <span style={{ color: ALGOS[h.algo]?.color, fontWeight: 700 }}>{(h.accuracy * 100).toFixed(2)}%</span></div>
          </div>
          <button onClick={() => onRollback(h)} style={{ background: "transparent", color: C.accent, border: `1px solid ${C.accent}44`, borderRadius: 6, padding: "5px 12px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
            ↩ Rollback
          </button>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ───────────────────────────────────────────────────────────────
function App() {
  const [tab, setTab] = useState("train");
  const [selectedAlgo, setSelectedAlgo] = useState("rf");
  const [compareAlgos, setCompareAlgos] = useState([]);
  const [params, setParams] = useState({});
  const [trainState, setTrainState] = useState("idle"); // idle | training | done
  const [trainProgress, setTrainProgress] = useState(0);
  const [trainLogs, setTrainLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState([]);
  const [versionCounter, setVersionCounter] = useState(1);
  const [notification, setNotification] = useState(null);
  const [dataFilter, setDataFilter] = useState("");
  const logsRef = useRef(null);

  const algo = ALGOS[selectedAlgo];

  useEffect(() => {
    const defaults = {};
    algo.params.forEach(p => { defaults[p.key] = p.default; });
    setParams(defaults);
    setResults(null);
    setTrainState("idle");
  }, [selectedAlgo]);

  useEffect(() => {
    if (logsRef.current) logsRef.current.scrollTop = logsRef.current.scrollHeight;
  }, [trainLogs]);

  const notify = (msg, color = C.green) => {
    setNotification({ msg, color });
    setTimeout(() => setNotification(null), 3500);
  };

  const handleTrain = () => {
    setTrainState("training");
    setTrainProgress(0);
    setTrainLogs([]);
    simulateTrain(selectedAlgo, params,
      (pct, logs) => { setTrainProgress(pct); setTrainLogs(logs); },
      (res) => {
        setResults(res);
        setTrainState("done");
        const ver = versionCounter;
        setVersionCounter(v => v + 1);
        setHistory(h => [{
          algo: selectedAlgo, accuracy: res.accuracy, results: res,
          version: ver, time: new Date().toLocaleTimeString(), params: { ...params }
        }, ...h]);
        notify(`✅ Entraînement terminé — Accuracy: ${(res.accuracy * 100).toFixed(2)}%`);
      }
    );
  };

  const handleRollback = (h) => {
    setSelectedAlgo(h.algo);
    setResults(h.results);
    setTrainState("done");
    notify(`↩ Rollback vers ${ALGOS[h.algo].label} v${h.version}`, C.accent);
  };

  const handleAutoMLSelect = (key) => {
    setSelectedAlgo(key);
    setTab("train");
    notify(`🤖 AutoML : ${ALGOS[key].label} sélectionné`, ALGOS[key].color);
  };

  const filteredData = DATASET_PREVIEW.filter(r =>
    dataFilter === "" || Object.values(r).some(v => String(v).toLowerCase().includes(dataFilter.toLowerCase()))
  );

  const TABS = [
    { key: "train", label: "⚙️ Entraînement" },
    { key: "results", label: "📊 Résultats" },
    { key: "data", label: "🗂️ Données" },
    { key: "compare", label: "⚖️ Comparaison" },
    { key: "automl", label: "🤖 AutoML" },
    { key: "history", label: "🕓 Historique" },
  ];

  return (
    <div style={{ minHeight: "100vh", background: C.bg, color: C.text, fontFamily: "'IBM Plex Mono', monospace", fontSize: 13 }}>
      {/* Notification */}
      {notification && (
        <div style={{ position: "fixed", top: 20, right: 20, zIndex: 999, background: C.panel, border: `1px solid ${notification.color}`, borderRadius: 10, padding: "12px 20px", color: notification.color, fontWeight: 700, fontSize: 13, boxShadow: `0 0 20px ${notification.color}44`, animation: "fadeIn 0.3s ease" }}>
          {notification.msg}
        </div>
      )}

      {/* Header */}
      <div style={{ background: C.panel, borderBottom: `1px solid ${C.border}`, padding: "0 24px", display: "flex", alignItems: "center", gap: 16, height: 56 }}>
        <div style={{ fontWeight: 800, fontSize: 15, letterSpacing: 2, color: C.accent }}>ML<span style={{ color: C.text }}>STUDIO</span></div>
        <div style={{ color: C.border, fontSize: 20 }}>|</div>
        <div style={{ color: C.muted, fontSize: 11 }}>Projet 16 — Vehicle Classification</div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <Badge text="DATASET: VEHICLE" color={C.muted} />
          <Badge text="946 obs" color={C.green} />
          <Badge text="18 features" color={C.accent} />
          {trainState === "training" && <Badge text="⏳ TRAINING" color={C.yellow} />}
          {trainState === "done" && <Badge text="✅ TRAINED" color={C.green} />}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: C.panel, borderBottom: `1px solid ${C.border}`, padding: "0 24px", display: "flex", gap: 4 }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            background: "transparent", border: "none", borderBottom: tab === t.key ? `2px solid ${C.accent}` : "2px solid transparent",
            color: tab === t.key ? C.accent : C.muted, padding: "14px 16px", cursor: "pointer", fontSize: 12, fontWeight: 700, fontFamily: "inherit", letterSpacing: 0.5
          }}>
            {t.label}
          </button>
        ))}
      </div>

      <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>

        {/* ── TRAIN TAB ── */}
        {tab === "train" && (
          <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 20 }}>
            {/* Algo selector */}
            <div>
              <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 12 }}>Choisir un algorithme</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {Object.entries(ALGOS).map(([key, a]) => (
                  <div key={key} onClick={() => setSelectedAlgo(key)} style={{
                    background: selectedAlgo === key ? `${a.color}15` : C.card,
                    border: `1px solid ${selectedAlgo === key ? a.color : C.border}`,
                    borderRadius: 10, padding: "14px 16px", cursor: "pointer",
                    transition: "all 0.2s"
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 20 }}>{a.icon}</span>
                      <div>
                        <div style={{ fontWeight: 700, color: selectedAlgo === key ? a.color : C.text, fontSize: 13 }}>{a.label}</div>
                        <div style={{ color: C.muted, fontSize: 10, marginTop: 3, lineHeight: 1.4 }}>{a.desc.slice(0, 60)}...</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Config + Training */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Algo info */}
              <div style={{ background: C.card, border: `1px solid ${algo.color}33`, borderRadius: 12, padding: 20 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
                  <span style={{ fontSize: 28 }}>{algo.icon}</span>
                  <div>
                    <div style={{ fontWeight: 800, fontSize: 16, color: algo.color }}>{algo.label}</div>
                    <div style={{ color: C.muted, fontSize: 12, marginTop: 2 }}>{algo.desc}</div>
                  </div>
                </div>
              </div>

              {/* Hyperparams */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                  <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase" }}>Hyperparamètres</div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <button onClick={() => notify("🔍 GridSearch lancé…", C.yellow)} style={{ background: "transparent", color: C.yellow, border: `1px solid ${C.yellow}44`, borderRadius: 6, padding: "5px 10px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                      GridSearch
                    </button>
                    <button onClick={() => notify("🎲 RandomSearch lancé…", C.accent2)} style={{ background: "transparent", color: C.accent2, border: `1px solid ${C.accent2}44`, borderRadius: 6, padding: "5px 10px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                      RandomSearch
                    </button>
                    <button onClick={() => notify("⚡ Optuna lancé…", C.green)} style={{ background: "transparent", color: C.green, border: `1px solid ${C.green}44`, borderRadius: 6, padding: "5px 10px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                      Optuna
                    </button>
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  {algo.params.map(p => (
                    <div key={p.key}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <label style={{ color: C.text, fontSize: 12, fontWeight: 600 }}>{p.label}</label>
                        <span style={{ color: algo.color, fontFamily: "monospace", fontSize: 12, fontWeight: 700 }}>{params[p.key]}</span>
                      </div>
                      {p.type === "range" ? (
                        <input type="range" min={p.min} max={p.max} step={p.step} value={params[p.key] || p.default}
                          onChange={e => setParams(prev => ({ ...prev, [p.key]: +e.target.value }))}
                          style={{ width: "100%", accentColor: algo.color }} />
                      ) : (
                        <select value={params[p.key] || p.default} onChange={e => setParams(prev => ({ ...prev, [p.key]: e.target.value }))}
                          style={{ width: "100%", background: C.panel, color: C.text, border: `1px solid ${C.border}`, borderRadius: 6, padding: "8px 12px", fontFamily: "inherit", fontSize: 12 }}>
                          {p.options.map(o => <option key={o} value={o}>{o}</option>)}
                        </select>
                      )}
                    </div>
                  ))}
                </div>
                <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
                  <button onClick={() => notify("💾 Configuration sauvegardée !", C.green)} style={{ background: "transparent", color: C.green, border: `1px solid ${C.green}44`, borderRadius: 6, padding: "7px 14px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                    💾 Sauvegarder config
                  </button>
                </div>
              </div>

              {/* Training controls */}
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
                  <button onClick={handleTrain} disabled={trainState === "training"} style={{
                    flex: 1, background: trainState === "training" ? C.border : `linear-gradient(135deg, ${algo.color}, ${algo.color}88)`,
                    color: "#fff", border: "none", borderRadius: 8, padding: "12px", fontWeight: 800, cursor: trainState === "training" ? "not-allowed" : "pointer", fontSize: 14, fontFamily: "inherit"
                  }}>
                    {trainState === "training" ? `⏳ Entraînement... ${trainProgress}%` : "▶ Entraîner le modèle"}
                  </button>
                  <button onClick={() => notify("📤 Modèle exporté (.pkl) !", C.accent)} disabled={trainState !== "done"} style={{
                    background: "transparent", color: trainState !== "done" ? C.muted : C.accent,
                    border: `1px solid ${trainState !== "done" ? C.border : C.accent + "44"}`, borderRadius: 8, padding: "12px 16px", cursor: trainState !== "done" ? "not-allowed" : "pointer", fontSize: 11, fontWeight: 700
                  }}>
                    📤 Export .pkl
                  </button>
                </div>
                {trainState !== "idle" && (
                  <>
                    <ProgressBar pct={trainProgress} color={algo.color} />
                    <div ref={logsRef} style={{ marginTop: 12, height: 100, overflowY: "auto", background: C.bg, borderRadius: 6, padding: 10, fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
                      {trainLogs.map((l, i) => <div key={i}>{l}</div>)}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── RESULTS TAB ── */}
        {tab === "results" && (
          <div>
            {!results ? (
              <div style={{ textAlign: "center", padding: 80, color: C.muted }}>
                <div style={{ fontSize: 48 }}>📊</div>
                <div style={{ marginTop: 16, fontSize: 14 }}>Aucun résultat disponible. Lancez d'abord un entraînement.</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                {/* Metrics */}
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <StatCard label="Accuracy" value={results.accuracy} sub="Dataset de test" color={algo.color} />
                  <StatCard label="F1-Score" value={results.f1} sub="Macro moyenne" color={C.accent} />
                  <StatCard label="Précision" value={results.precision} sub="Moyenne pondérée" color={C.green} />
                  <StatCard label="Recall" value={results.recall} sub="Moyenne pondérée" color={C.yellow} />
                  <StatCard label="CV Score" value={results.cv_mean} sub={`±${(results.cv_std * 100).toFixed(2)}% (5-fold)`} color={C.accent2} />
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  {/* Confusion */}
                  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                    <ConfusionMatrix data={results.confusion} />
                    <button onClick={() => notify("📷 Matrice exportée en PNG !", C.accent)} style={{ marginTop: 14, background: "transparent", color: C.accent, border: `1px solid ${C.accent}44`, borderRadius: 6, padding: "5px 12px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                      📷 Export PNG
                    </button>
                  </div>

                  {/* ROC */}
                  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                    <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, marginBottom: 12, textTransform: "uppercase" }}>Courbe ROC</div>
                    <ROCCurve data={results.roc_data} color={algo.color} />
                    <div style={{ color: C.muted, fontSize: 11, marginTop: 8 }}>AUC ≈ <span style={{ color: algo.color, fontWeight: 700 }}>0.{Math.round(results.accuracy * 100 + 6)}</span></div>
                  </div>
                </div>

                {/* Feature importance */}
                {selectedAlgo === "rf" && (
                  <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                    <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, marginBottom: 16, textTransform: "uppercase" }}>Importance des features</div>
                    <BarChart data={results.feature_importance} color={algo.color} />
                  </div>
                )}

                {/* Export */}
                <div style={{ display: "flex", gap: 10 }}>
                  <button onClick={() => notify("📊 Résultats exportés en CSV !", C.green)} style={{ background: "transparent", color: C.green, border: `1px solid ${C.green}44`, borderRadius: 6, padding: "8px 16px", cursor: "pointer", fontSize: 12, fontWeight: 700 }}>
                    📊 Export CSV
                  </button>
                  <button onClick={() => notify("📋 Rapport PDF généré !", C.accent2)} style={{ background: "transparent", color: C.accent2, border: `1px solid ${C.accent2}44`, borderRadius: 6, padding: "8px 16px", cursor: "pointer", fontSize: 12, fontWeight: 700 }}>
                    📋 Export Rapport PDF
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── DATA TAB ── */}
        {tab === "data" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <input placeholder="🔍 Filtrer les données..." value={dataFilter} onChange={e => setDataFilter(e.target.value)}
                style={{ flex: 1, background: C.card, border: `1px solid ${C.border}`, borderRadius: 8, padding: "10px 14px", color: C.text, fontFamily: "inherit", fontSize: 12 }} />
              <button onClick={() => notify("📁 Nouveau fichier uploadé !", C.green)} style={{ background: C.green, color: "#000", border: "none", borderRadius: 8, padding: "10px 16px", fontWeight: 800, cursor: "pointer", fontSize: 12 }}>
                ⬆ Upload données
              </button>
              <button onClick={() => notify("🧹 Valeurs manquantes supprimées !", C.yellow)} style={{ background: "transparent", color: C.yellow, border: `1px solid ${C.yellow}44`, borderRadius: 8, padding: "10px 16px", cursor: "pointer", fontSize: 12, fontWeight: 700 }}>
                🧹 Nettoyer
              </button>
            </div>

            {/* Stats */}
            <div style={{ display: "flex", gap: 12 }}>
              {["van", "saab", "bus", "opel"].map(cls => (
                <div key={cls} style={{ background: C.card, border: `1px solid ${CLASS_COLORS[cls]}33`, borderRadius: 10, padding: "12px 20px", flex: 1, textAlign: "center" }}>
                  <div style={{ color: CLASS_COLORS[cls], fontWeight: 800, fontSize: 18 }}>{Math.round(946 / 4)}</div>
                  <div style={{ color: C.muted, fontSize: 11, marginTop: 4, textTransform: "uppercase", letterSpacing: 1 }}>{cls}</div>
                </div>
              ))}
            </div>

            {/* Table */}
            <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: "hidden" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr style={{ background: C.panel }}>
                    {["ID", "COMPACTNESS", "CIRCULARITY", "DIST_CIRC", "RADIUS_RATIO", "CLASS"].map(h => (
                      <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1, borderBottom: `1px solid ${C.border}` }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((row, i) => (
                    <tr key={i} style={{ borderBottom: `1px solid ${C.border}` }}>
                      <td style={{ padding: "10px 16px", color: C.muted, fontSize: 12 }}>{row.id}</td>
                      <td style={{ padding: "10px 16px", fontFamily: "monospace", fontSize: 12 }}>{row.COMPACTNESS}</td>
                      <td style={{ padding: "10px 16px", fontFamily: "monospace", fontSize: 12 }}>{row.CIRCULARITY}</td>
                      <td style={{ padding: "10px 16px", fontFamily: "monospace", fontSize: 12 }}>{row.DISTANCE_CIRCULARITY}</td>
                      <td style={{ padding: "10px 16px", fontFamily: "monospace", fontSize: 12 }}>{row.RADIUS_RATIO}</td>
                      <td style={{ padding: "10px 16px" }}><Badge text={row.CLASS} color={CLASS_COLORS[row.CLASS]} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div style={{ padding: "10px 16px", color: C.muted, fontSize: 11 }}>Affichage : {filteredData.length} / 946 observations</div>
            </div>
          </div>
        )}

        {/* ── COMPARE TAB ── */}
        {tab === "compare" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
              <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase", marginBottom: 12 }}>Sélectionner les modèles à comparer</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {Object.entries(ALGOS).map(([key, a]) => (
                  <button key={key} onClick={() => setCompareAlgos(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key])}
                    style={{ background: compareAlgos.includes(key) ? `${a.color}22` : C.card, color: compareAlgos.includes(key) ? a.color : C.muted, border: `1px solid ${compareAlgos.includes(key) ? a.color : C.border}`, borderRadius: 8, padding: "8px 14px", cursor: "pointer", fontWeight: 700, fontSize: 12, fontFamily: "inherit" }}>
                    {a.icon} {a.label}
                  </button>
                ))}
              </div>
            </div>

            {compareAlgos.length > 0 && (
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, overflow: "hidden" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: C.panel }}>
                      {["Algorithme", "Accuracy", "F1-Score", "Précision", "Recall", "Temps train"].map(h => (
                        <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1, borderBottom: `1px solid ${C.border}` }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {compareAlgos.map((key, i) => {
                      const a = ALGOS[key];
                      const acc = 0.75 + (Object.keys(ALGOS).indexOf(key) * 0.02);
                      return (
                        <tr key={key} style={{ borderBottom: `1px solid ${C.border}` }}>
                          <td style={{ padding: "12px 16px" }}><span style={{ color: a.color, fontWeight: 700 }}>{a.icon} {a.label}</span></td>
                          <td style={{ padding: "12px 16px", fontFamily: "monospace", color: a.color, fontWeight: 800 }}>{(acc * 100).toFixed(2)}%</td>
                          <td style={{ padding: "12px 16px", fontFamily: "monospace" }}>{((acc - 0.01) * 100).toFixed(2)}%</td>
                          <td style={{ padding: "12px 16px", fontFamily: "monospace" }}>{((acc + 0.01) * 100).toFixed(2)}%</td>
                          <td style={{ padding: "12px 16px", fontFamily: "monospace" }}>{((acc - 0.015) * 100).toFixed(2)}%</td>
                          <td style={{ padding: "12px 16px", color: C.muted }}>{(0.5 + i * 0.3).toFixed(1)}s</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {compareAlgos.length > 1 && (
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, marginBottom: 16, textTransform: "uppercase" }}>Comparaison Accuracy</div>
                {compareAlgos.map((key, i) => {
                  const a = ALGOS[key];
                  const acc = 0.75 + (Object.keys(ALGOS).indexOf(key) * 0.02);
                  return (
                    <div key={key} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                      <span style={{ width: 130, color: a.color, fontWeight: 700, fontSize: 12 }}>{a.icon} {a.label}</span>
                      <div style={{ flex: 1 }}><ProgressBar pct={acc * 100} color={a.color} /></div>
                      <span style={{ fontFamily: "monospace", color: a.color, fontWeight: 800, width: 60 }}>{(acc * 100).toFixed(2)}%</span>
                    </div>
                  );
                })}
              </div>
            )}

            {compareAlgos.length === 0 && (
              <div style={{ textAlign: "center", padding: 60, color: C.muted }}>
                <div style={{ fontSize: 40 }}>⚖️</div>
                <div style={{ marginTop: 12 }}>Sélectionnez au moins 2 algorithmes pour les comparer.</div>
              </div>
            )}
          </div>
        )}

        {/* ── AUTOML TAB ── */}
        {tab === "automl" && <AutoMLPanel onSelectBest={handleAutoMLSelect} />}

        {/* ── HISTORY TAB ── */}
        {tab === "history" && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ color: C.muted, fontSize: 11, fontWeight: 700, letterSpacing: 1.5, textTransform: "uppercase" }}>
                Historique des expérimentations ({history.length})
              </div>
              <button onClick={() => notify("📊 Historique exporté en CSV !", C.green)} style={{ background: "transparent", color: C.green, border: `1px solid ${C.green}44`, borderRadius: 6, padding: "6px 12px", cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
                📊 Export CSV
              </button>
            </div>
            <HistoryPanel history={history} onRollback={handleRollback} />
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700;800&display=swap');
        * { box-sizing: border-box; }
        input[type=range] { height: 4px; }
        ::-webkit-scrollbar { width: 4px; height: 4px; }
        ::-webkit-scrollbar-track { background: ${C.panel}; }
        ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 2px; }
        select option { background: ${C.panel}; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

export default App;
