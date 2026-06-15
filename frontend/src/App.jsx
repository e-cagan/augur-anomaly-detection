import { useState, useRef, useMemo, useEffect } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  ReferenceArea,
  Tooltip,
} from "recharts";

const API_URL = "http://localhost:8000/predict";

/* Extract contiguous anomaly regions [start, end] from the frame list.
   These become the glowing alarm bands behind the trace. */
function anomalyBands(frames) {
  const bands = [];
  let start = null;
  for (const f of frames) {
    if (f.is_anomaly) {
      if (start === null) start = f.frame_idx;
    } else if (start !== null) {
      bands.push([start, f.frame_idx - 1]);
      start = null;
    }
  }
  if (start !== null) bands.push([start, frames[frames.length - 1].frame_idx]);
  return bands;
}

function TipBox({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  if (p.score === null) return null;
  return (
    <div className="tip">
      <div className="row"><span className="lab">FRAME</span><span className="val">{p.frame}</span></div>
      <div className="row">
        <span className="lab">SURPRISE</span>
        <span className={"val" + (p.is_anomaly ? " alarm" : "")}>{p.score.toExponential(2)}</span>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [drag, setDrag] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const inputRef = useRef(null);
  const videoRef = useRef(null);

  // Playable URL for the uploaded file (revoke on change to avoid leaks)
  const videoUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  useEffect(() => {
    return () => { if (videoUrl) URL.revokeObjectURL(videoUrl); };
  }, [videoUrl]);

  const status = loading ? "reading" : result ? "flagged" : "standby";
  const statusLabel = loading ? "READING FEED" : result ? "ANALYSIS COMPLETE" : "FEED STANDBY";

  function pick(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setCurrentFrame(0);
  }

  async function analyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentFrame(0);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(API_URL, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server returned ${res.status}`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(
        e.message?.includes("fetch")
          ? "Cannot reach the detector at localhost:8000. Start the backend, then run again."
          : e.message
      );
    } finally {
      setLoading(false);
    }
  }

  const fps = result?.fps || 10;

  const chartData = useMemo(
    () => result?.frames.map((f) => ({ frame: f.frame_idx, score: f.score, is_anomaly: f.is_anomaly })) ?? [],
    [result]
  );

  // Progressive reveal: hide scores beyond the playhead so the trace draws in sync
  const revealedData = useMemo(
    () => chartData.map((d) => ({ ...d, score: d.frame <= currentFrame ? d.score : null })),
    [chartData, currentFrame]
  );

  const bands = useMemo(() => (result ? anomalyBands(result.frames) : []), [result]);
  const peak = useMemo(() => {
    const s = result?.frames.map((f) => f.score).filter((v) => v !== null) ?? [];
    return s.length ? Math.max(...s) : 0;
  }, [result]);

  const liveFrame = result?.frames[currentFrame];

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <div className="wordmark">AUGUR</div>
          <div className="tagline">Every frame, predicted. Every surprise, flagged.</div>
        </div>
        <div className="status" data-state={status}>
          <span className="dot" />
          {statusLabel}
        </div>
      </header>

      <section className="intake">
        <div
          className="dropzone"
          data-drag={drag}
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
        >
          <div className="eyebrow">VIDEO FEED INPUT</div>
          <div className="prompt">Drop a video, or click to select</div>
          <div className="sub">The detector learns normal motion, then flags what it cannot predict.</div>
          {file && <div className="filename">{file.name}</div>}
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden-input"
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>

        <button className="run" onClick={analyze} disabled={!file || loading}>
          {loading ? "ANALYZING…" : "RUN DETECTION"}
        </button>

        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="stats">
          <div className="stat">
            <div className="k">FRAMES</div>
            <div className="v">{result.total_frames}</div>
          </div>
          <div className="stat">
            <div className="k">FLAGGED</div>
            <div className="v alarm">{result.frames.filter((f) => f.is_anomaly).length}</div>
          </div>
          <div className="stat">
            <div className="k">PEAK SURPRISE</div>
            <div className="v">{peak.toExponential(1)}</div>
          </div>
          <div className="stat">
            <div className="k">TRIPWIRE</div>
            <div className="v amber">{result.threshold.toExponential(1)}</div>
          </div>
        </section>
      )}

      {/* Synced playback: video + live readout */}
      {result && videoUrl && (
        <section className="playback">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            className="feed-video"
            onTimeUpdate={(e) => setCurrentFrame(Math.floor(e.target.currentTime * fps))}
          />
          <div className="live-readout">
            <span className="lr-frame">FRAME {currentFrame}</span>
            {!liveFrame || liveFrame.score === null ? (
              <span className="lr-warm">WARMING UP</span>
            ) : (
              <span className={"lr-score" + (liveFrame.is_anomaly ? " alarm" : "")}>
                {liveFrame.score.toExponential(2)} {liveFrame.is_anomaly ? "· ANOMALY" : "· normal"}
              </span>
            )}
          </div>
        </section>
      )}

      <section className="trace">
        <div className="trace-head">
          <div className="trace-title">THE SURPRISE TRACE</div>
          <div className="trace-legend">
            <span className="item"><span className="swatch calm" />signal</span>
            <span className="item"><span className="swatch amber" />tripwire</span>
            <span className="item"><span className="swatch alarm" />anomaly</span>
          </div>
        </div>

        {result ? (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={revealedData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <defs>
                <linearGradient id="calmFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#56C7BE" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="#56C7BE" stopOpacity={0} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="2 4" vertical={false} />

              {/* glowing alarm bands behind the trace (full range, not revealed) */}
              {bands.map(([a, b], i) => (
                <ReferenceArea key={i} x1={a} x2={b} fill="#FF6A5A" fillOpacity={0.10} stroke="none" />
              ))}

              <XAxis dataKey="frame" type="number" domain={[0, result.total_frames]}
                     tickLine={false} interval="preserveStartEnd" minTickGap={40} allowDataOverflow />
              <YAxis tickFormatter={(v) => v.toExponential(0)} width={56} tickLine={false} />

              {/* the amber tripwire */}
              <ReferenceLine
                y={result.threshold}
                stroke="#E6A93C"
                strokeDasharray="5 4"
                strokeWidth={1.2}
                label={{ value: "TRIPWIRE", position: "insideTopRight", fill: "#E6A93C", fontSize: 10, fontFamily: "JetBrains Mono" }}
              />

              {/* the playhead — follows the video */}
              <ReferenceLine x={currentFrame} stroke="#56C7BE" strokeWidth={1.5} strokeOpacity={0.9} />

              <Tooltip content={<TipBox />} cursor={{ stroke: "#66768A", strokeDasharray: "3 3" }} />

              {/* connectNulls=false leaves a gap over warm-up AND beyond the playhead */}
              <Area
                type="monotone"
                dataKey="score"
                stroke="#56C7BE"
                strokeWidth={1.6}
                fill="url(#calmFill)"
                connectNulls={false}
                isAnimationActive={false}
                dot={false}
                activeDot={{ r: 3, fill: "#56C7BE", stroke: "none" }}
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="idle">
            <div className="label">{loading ? "READING FEED…" : "AWAITING FEED"}</div>
            <div className="scan" />
            <div className="label" style={{ color: "var(--dim-2)", fontSize: 11 }}>
              SURPRISE / FRAME
            </div>
          </div>
        )}
      </section>

      {/* Most anomalous moments — heatmap overlays */}
      {result && result.top_anomalies?.length > 0 && (
        <section className="moments">
          <div className="moments-head">
            <div className="moments-title">MOST ANOMALOUS MOMENTS</div>
            <div className="moments-sub">Where the model was most surprised — heatmap over frame</div>
          </div>
          <div className="moments-grid">
            {result.top_anomalies.map((a) => (
              <figure className="moment" key={a.frame_idx}>
                <img
                  className="moment-img"
                  src={`data:image/png;base64,${a.overlay}`}
                  alt={`Frame ${a.frame_idx}`}
                />
                <figcaption className="moment-cap">
                  <span className="moment-frame">FRAME {a.frame_idx}</span>
                  <span className="moment-score">{a.score.toExponential(2)}</span>
                </figcaption>
              </figure>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}