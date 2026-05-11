const { useEffect, useMemo, useRef, useState } = React;

const BALL_ID = '__ball__';

const C = {
  bg: '#0D1B2A',
  panel: '#16283A',
  border: '#243B52',
  grass: '#39FF14',
  gray: '#7A9BB5',
  grayLight: '#B0C7D9',
  white: '#FFFFFF',
  red: '#FF4558',
  teamA: '#f5a623',
  teamB: '#2196f3',
  ball: '#F8F4D8',
  ballStroke: '#101010',
};

const LABEL_STYLES = {
  A: { color: C.teamA, label: 'Team A' },
  B: { color: C.teamB, label: 'Team B' },
  Other: { color: C.red, label: 'Other' },
  Ball: { color: C.ball, label: 'Ball' },
};

const POSSESSION = [
  { value: 'unknown', label: 'Unknown' },
  { value: 'A', label: 'Team A' },
  { value: 'B', label: 'Team B' },
  { value: 'contested', label: 'Contested' },
  { value: 'absent', label: 'Ball absent' },
];

(function injectStyles() {
  if (document.getElementById('annotator-style')) return;
  const el = document.createElement('style');
  el.id = 'annotator-style';
  el.textContent = `
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, sans-serif; background: ${C.bg}; color: ${C.white}; }
    h1,h2,h3 { font-family: Rajdhani, sans-serif; letter-spacing: 0.03em; margin: 0; }
    button, select, input { font-family: inherit; }
    code { color: ${C.grayLight}; }
    .btn { border: 1px solid ${C.border}; background: ${C.panel}; color: ${C.white}; border-radius: 8px; padding: 10px 14px; cursor: pointer; }
    .btn:hover { border-color: ${C.gray}; }
    .btn.primary { background: ${C.grass}; color: ${C.bg}; border-color: ${C.grass}; font-weight: 700; }
    .btn.danger { border-color: ${C.red}; color: ${C.red}; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px; border-radius: 18px; border: 1px solid ${C.border}; color: ${C.grayLight}; font-size: 12px; white-space: nowrap; }
    .card { background: ${C.panel}; border: 1px solid ${C.border}; border-radius: 12px; }
    .muted { color: ${C.gray}; }
    .field { width: 100%; background: ${C.bg}; border: 1px solid ${C.border}; color: ${C.white}; border-radius: 8px; padding: 10px 12px; }
    .label-btn { border: 1px solid ${C.border}; background: ${C.bg}; color: ${C.grayLight}; border-radius: 999px; padding: 8px 12px; cursor: pointer; }
    .label-btn.active { color: ${C.bg}; font-weight: 700; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid ${C.border}; font-size: 13px; }
    canvas { display: block; max-width: 100%; background: #000; cursor: crosshair; }
  `;
  document.head.appendChild(el);
})();

function frameDefault(frame = {}) {
  return {
    points: frame.points || [],
    ball: frame.ball || null,
    possession: frame.possession || 'unknown',
  };
}

function App() {
  const [manifestList, setManifestList] = useState([]);
  const [runList, setRunList] = useState([]);
  const [manifestId, setManifestId] = useState('');
  const [manifest, setManifest] = useState(null);
  const [annotations, setAnnotations] = useState({ manifest_id: null, frames: {} });
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedTool, setSelectedTool] = useState('A');
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [status, setStatus] = useState('Loading manifests...');
  const [draggingEntity, setDraggingEntity] = useState(null);
  const [seedRunId, setSeedRunId] = useState('');
  const [seedPreset, setSeedPreset] = useState('');
  const [seedOverwrite, setSeedOverwrite] = useState(false);
  const canvasRef = useRef(null);
  const imgRef = useRef(new Image());

  useEffect(() => {
    Promise.all([
      fetch('/api/eval/manifests').then((r) => r.json()),
      fetch('/api/eval/runs').then((r) => r.json()),
    ])
      .then(([items, runs]) => {
        setManifestList(items);
        setRunList(runs);
        if (items[0]) setManifestId(items[0].manifest_id);
        if (runs[0]) {
          setSeedRunId(runs[0].run_id);
          setSeedPreset(runs[0].models?.[0]?.preset || '');
        }
        setStatus(items.length ? 'Select a manifest to start annotating.' : 'No manifests found yet.');
      })
      .catch((err) => setStatus(`Failed to load manifests: ${err}`));
  }, []);

  useEffect(() => {
    if (!manifestId) return;
    Promise.all([
      fetch(`/api/eval/manifests/${manifestId}`).then((r) => r.json()),
      fetch(`/api/eval/annotations/${manifestId}`).then((r) => r.json()),
    ])
      .then(([m, a]) => {
        setManifest(m);
        setAnnotations(a);
        setFrameIndex(0);
        setSelectedEntity(null);
        setStatus(`Loaded manifest ${manifestId}.`);
      })
      .catch((err) => setStatus(`Failed to load manifest: ${err}`));
  }, [manifestId]);

  const frames = manifest?.frames || [];
  const currentFrame = frames[frameIndex] || null;
  const frameAnnotation = useMemo(() => {
    if (!currentFrame) return frameDefault();
    return frameDefault(annotations.frames?.[currentFrame.frame_id] || {});
  }, [annotations, currentFrame]);

  const selectedRun = useMemo(
    () => runList.find((run) => run.run_id === seedRunId) || null,
    [runList, seedRunId],
  );
  const selectedRunModels = selectedRun?.models || [];
  const ball = frameAnnotation.ball;

  useEffect(() => {
    if (!currentFrame || !canvasRef.current) return;
    const img = imgRef.current;
    img.onload = () => drawFrame();
    img.src = `/eval-media/${currentFrame.image_path.replace(/\\/g, '/')}`;
  }, [currentFrame, frameAnnotation, selectedEntity]);

  function drawBall(ctx, pt, isSelected) {
    ctx.save();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, isSelected ? 12 : 10, 0, Math.PI * 2);
    ctx.fillStyle = C.ball;
    ctx.fill();
    ctx.lineWidth = isSelected ? 4 : 3;
    ctx.strokeStyle = isSelected ? C.grass : C.ballStroke;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pt.x - 5, pt.y);
    ctx.lineTo(pt.x + 5, pt.y);
    ctx.moveTo(pt.x, pt.y - 5);
    ctx.lineTo(pt.x, pt.y + 5);
    ctx.lineWidth = 2;
    ctx.strokeStyle = C.ballStroke;
    ctx.stroke();
    ctx.font = '12px Inter';
    ctx.fillStyle = C.white;
    ctx.fillText('BALL', pt.x + 14, pt.y - 12);
    ctx.restore();
  }

  function drawFrame() {
    const frame = currentFrame;
    if (!frame) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = imgRef.current;
    canvas.width = img.naturalWidth || frame.width || 1280;
    canvas.height = img.naturalHeight || frame.height || 720;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    for (const pt of frameAnnotation.points || []) {
      const style = LABEL_STYLES[pt.team] || LABEL_STYLES.Other;
      const isSelected = selectedEntity === pt.id;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, isSelected ? 10 : 8, 0, Math.PI * 2);
      ctx.fillStyle = style.color;
      ctx.fill();
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = isSelected ? C.white : C.bg;
      ctx.stroke();
      ctx.font = '12px Inter';
      ctx.fillStyle = C.white;
      ctx.fillText(pt.id, pt.x + 12, pt.y - 10);
    }

    if (ball && ball.status !== 'absent') {
      drawBall(ctx, ball, selectedEntity === BALL_ID);
    }
  }

  function saveAll() {
    if (!manifestId) return;
    setStatus('Saving annotations...');
    fetch(`/api/eval/annotations/${manifestId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(annotations),
    })
      .then((r) => r.json())
      .then((saved) => {
        setAnnotations(saved);
        setStatus(`Saved ${manifestId} at ${new Date(saved.saved_at).toLocaleTimeString()}.`);
      })
      .catch((err) => setStatus(`Save failed: ${err}`));
  }

  function seedFromRun() {
    if (!manifestId || !seedRunId || !seedPreset) return;
    setStatus(`Seeding ${manifestId} from ${seedRunId}/${seedPreset}...`);
    fetch(`/api/eval/annotations/${manifestId}/seed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        run_id: seedRunId,
        preset: seedPreset,
        overwrite: seedOverwrite,
      }),
    })
      .then((r) => r.json())
      .then((saved) => {
        setAnnotations(saved);
        const seedInfo = saved.seed_info || {};
        setStatus(`Seeded ${seedInfo.seeded_frames || 0} frames from ${seedRunId}/${seedPreset}.`);
      })
      .catch((err) => setStatus(`Seed failed: ${err}`));
  }

  function updateCurrentFrame(mutator) {
    if (!currentFrame) return;
    setAnnotations((prev) => {
      const next = structuredClone(prev);
      next.frames ||= {};
      const existing = frameDefault(next.frames[currentFrame.frame_id] || {});
      next.frames[currentFrame.frame_id] = mutator(existing);
      return next;
    });
  }

  function nextPointId(points) {
    const taken = new Set(points.map((p) => p.id));
    let idx = 1;
    while (taken.has(`P${idx}`)) idx += 1;
    return `P${idx}`;
  }

  function toCanvasCoords(evt) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * sx,
      y: (evt.clientY - rect.top) * sy,
    };
  }

  function hitTest(x, y) {
    if (ball && ball.status !== 'absent' && Math.hypot(ball.x - x, ball.y - y) <= 18) {
      return BALL_ID;
    }
    for (const pt of frameAnnotation.points || []) {
      if (Math.hypot(pt.x - x, pt.y - y) <= 14) return pt.id;
    }
    return null;
  }

  function setBallAt(x, y) {
    updateCurrentFrame((frame) => ({
      ...frame,
      ball: { x, y, status: 'visible' },
      possession: frame.possession === 'absent' ? 'unknown' : frame.possession,
    }));
    setSelectedEntity(BALL_ID);
  }

  function onCanvasDown(evt) {
    if (evt.button === 2) evt.preventDefault();
    if (!currentFrame) return;

    const { x, y } = toCanvasCoords(evt);
    const hit = hitTest(x, y);
    if (hit) {
      setSelectedEntity(hit);
      setDraggingEntity(hit);
      return;
    }

    if (selectedTool === 'Ball') {
      setBallAt(x, y);
      setDraggingEntity(BALL_ID);
      return;
    }

    let teamToAssign = selectedTool;
    if (evt.button === 0 && selectedTool === 'B') teamToAssign = 'B';
    if (evt.button === 2) teamToAssign = 'B';

    updateCurrentFrame((frame) => {
      const points = [...(frame.points || [])];
      const id = nextPointId(points);
      points.push({ id, x, y, team: teamToAssign });
      setSelectedEntity(id);
      return { ...frame, points };
    });
  }

  function onCanvasMove(evt) {
    if (!draggingEntity || !currentFrame) return;
    const { x, y } = toCanvasCoords(evt);
    if (draggingEntity === BALL_ID) {
      setBallAt(x, y);
      return;
    }
    updateCurrentFrame((frame) => ({
      ...frame,
      points: (frame.points || []).map((pt) => (
        pt.id === draggingEntity ? { ...pt, x, y } : pt
      )),
    }));
  }

  function onCanvasUp() {
    setDraggingEntity(null);
  }

  function deleteSelected() {
    if (!selectedEntity) return;
    if (selectedEntity === BALL_ID) {
      updateCurrentFrame((frame) => ({ ...frame, ball: null }));
    } else {
      updateCurrentFrame((frame) => ({
        ...frame,
        points: (frame.points || []).filter((pt) => pt.id !== selectedEntity),
      }));
    }
    setSelectedEntity(null);
  }

  function changeSelectedTeam(team) {
    if (!selectedEntity || selectedEntity === BALL_ID) return;
    updateCurrentFrame((frame) => ({
      ...frame,
      points: (frame.points || []).map((pt) => (
        pt.id === selectedEntity ? { ...pt, team } : pt
      )),
    }));
  }

  function setPossession(possession) {
    updateCurrentFrame((frame) => ({
      ...frame,
      possession,
      ball: possession === 'absent' ? { status: 'absent' } : frame.ball,
    }));
    if (possession === 'absent') setSelectedEntity(null);
  }

  function clearBall() {
    updateCurrentFrame((frame) => ({ ...frame, ball: null, possession: 'unknown' }));
    setSelectedEntity(null);
  }

  function copyPreviousFrame() {
    if (!currentFrame || frameIndex === 0) return;
    const prevFrame = frames[frameIndex - 1];
    const prevAnn = annotations.frames?.[prevFrame.frame_id];
    if (!prevAnn) return;
    setAnnotations((prev) => {
      const next = structuredClone(prev);
      next.frames ||= {};
      next.frames[currentFrame.frame_id] = structuredClone(prevAnn);
      return next;
    });
    setStatus(`Copied labels from ${prevFrame.frame_id}.`);
  }

  useEffect(() => {
    function onKey(evt) {
      if (evt.target?.tagName === 'INPUT' || evt.target?.tagName === 'SELECT') return;
      if (evt.key === 'Delete' || evt.key === 'Backspace') {
        evt.preventDefault();
        deleteSelected();
      } else if (evt.key === 'ArrowRight') {
        setFrameIndex((i) => Math.min(i + 1, frames.length - 1));
      } else if (evt.key === 'ArrowLeft') {
        setFrameIndex((i) => Math.max(i - 1, 0));
      } else if (evt.key === '1') {
        setSelectedTool('A');
        changeSelectedTeam('A');
      } else if (evt.key === '2') {
        setSelectedTool('B');
        changeSelectedTeam('B');
      } else if (evt.key === '3') {
        setSelectedTool('Other');
        changeSelectedTeam('Other');
      } else if (evt.key === '4' || evt.key.toLowerCase() === 'b') {
        setSelectedTool('Ball');
        setSelectedEntity(BALL_ID);
      } else if (evt.key.toLowerCase() === 'x') {
        setPossession('absent');
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [frames.length, selectedEntity, frameIndex, annotations]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ borderBottom: `1px solid ${C.border}`, padding: '16px 24px', background: `${C.bg}EE` }}>
        <div style={{ maxWidth: 1500, margin: '0 auto', display: 'flex', alignItems: 'center', gap: 18 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: C.grass, fontSize: 12, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: C.grass, boxShadow: `0 0 8px ${C.grass}` }} />
              GTA Evaluation
            </div>
            <h1 style={{ fontSize: 34 }}>Ground-Truth Annotator</h1>
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ minWidth: 300 }}>
            <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>Manifest</div>
            <select className="field" value={manifestId} onChange={(e) => setManifestId(e.target.value)}>
              {manifestList.map((m) => (
                <option key={m.manifest_id} value={m.manifest_id}>
                  {m.manifest_id} ({m.frame_count} frames)
                </option>
              ))}
            </select>
          </div>
          <button className="btn primary" onClick={saveAll}>Save annotations</button>
        </div>
      </header>

      <main style={{ flex: 1, maxWidth: 1500, width: '100%', margin: '0 auto', padding: 24, display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 380px', gap: 20 }}>
        <section className="card" style={{ padding: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
            <button className="btn" onClick={() => setFrameIndex((i) => Math.max(i - 1, 0))} disabled={!frames.length || frameIndex === 0}>Prev</button>
            <button className="btn" onClick={() => setFrameIndex((i) => Math.min(i + 1, frames.length - 1))} disabled={!frames.length || frameIndex >= frames.length - 1}>Next</button>
            <button className="btn" onClick={copyPreviousFrame} disabled={frameIndex === 0}>Copy previous</button>
            <button className="btn danger" onClick={deleteSelected} disabled={!selectedEntity}>Delete selected</button>
            <div style={{ flex: 1 }} />
            <span className="pill">{frameIndex + 1} / {frames.length || 0}</span>
          </div>

          {currentFrame ? (
            <>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                <span className="pill">Clip: {currentFrame.clip_id}</span>
                <span className="pill">Frame: {currentFrame.frame_index}</span>
                <span className="pill">Time: {currentFrame.timestamp_s.toFixed(2)}s</span>
                <span className="pill">Kind: {currentFrame.kind}</span>
                {currentFrame.sequence_id && <span className="pill">Sequence: {currentFrame.sequence_id}</span>}
              </div>
              <div style={{ overflow: 'auto', borderRadius: 10, border: `1px solid ${C.border}` }}>
                <canvas
                  ref={canvasRef}
                  onContextMenu={(e) => {
                    e.preventDefault();
                    return false;
                  }}
                  onMouseDown={onCanvasDown}
                  onMouseMove={onCanvasMove}
                  onMouseUp={onCanvasUp}
                  onMouseLeave={onCanvasUp}
                />
              </div>
              <div className="muted" style={{ marginTop: 10, fontSize: 13 }}>
                Keys: <code>1</code> Team A, <code>2</code> Team B, <code>3</code> Other, <code>4</code> Ball, <code>x</code> ball absent.
              </div>
            </>
          ) : (
            <div className="muted">No frame selected.</div>
          )}
        </section>

        <aside style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Label mode</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {Object.entries(LABEL_STYLES).map(([label, style]) => (
                <button
                  key={label}
                  className={`label-btn ${selectedTool === label ? 'active' : ''}`}
                  style={{ background: selectedTool === label ? style.color : C.bg }}
                  onClick={() => setSelectedTool(label)}
                >
                  {style.label}
                </button>
              ))}
            </div>
            {selectedEntity && selectedEntity !== BALL_ID && (
              <>
                <div className="muted" style={{ fontSize: 12, marginTop: 14, marginBottom: 6 }}>Selected player point</div>
                <div style={{ fontSize: 14, marginBottom: 8 }}>{selectedEntity}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {['A', 'B', 'Other'].map((team) => (
                    <button
                      key={team}
                      className="label-btn"
                      style={{ background: LABEL_STYLES[team].color, color: C.bg, fontWeight: 700 }}
                      onClick={() => changeSelectedTeam(team)}
                    >
                      Set {LABEL_STYLES[team].label}
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Ball in this frame</div>
            {ball && ball.status !== 'absent' ? (
              <div style={{ fontSize: 14, marginBottom: 12 }}>
                <span style={{ color: C.ball }}>Visible</span> at x={ball.x.toFixed(1)}, y={ball.y.toFixed(1)}
              </div>
            ) : (
              <div className="muted" style={{ fontSize: 14, marginBottom: 12 }}>
                {frameAnnotation.possession === 'absent' ? 'Marked absent.' : 'No ball point yet.'}
              </div>
            )}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
              <button className="btn" onClick={() => setSelectedTool('Ball')}>Ball mode</button>
              <button className="btn" onClick={() => setPossession('absent')}>Ball absent</button>
              <button className="btn" onClick={clearBall}>Clear ball</button>
            </div>
            <div className="muted" style={{ fontSize: 12, marginBottom: 8 }}>Possession owner</div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {POSSESSION.map((item) => (
                <button
                  key={item.value}
                  className={`label-btn ${frameAnnotation.possession === item.value ? 'active' : ''}`}
                  style={{
                    background: frameAnnotation.possession === item.value ? C.grass : C.bg,
                    color: frameAnnotation.possession === item.value ? C.bg : C.grayLight,
                  }}
                  onClick={() => setPossession(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Bootstrap from model</div>
            <div style={{ display: 'grid', gap: 10 }}>
              <div>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>Run ID</div>
                <select
                  className="field"
                  value={seedRunId}
                  onChange={(e) => {
                    const nextRunId = e.target.value;
                    setSeedRunId(nextRunId);
                    const run = runList.find((item) => item.run_id === nextRunId);
                    setSeedPreset(run?.models?.[0]?.preset || '');
                  }}
                >
                  <option value="">Select a run...</option>
                  {runList.map((run) => (
                    <option key={run.run_id} value={run.run_id}>
                      {run.run_id} ({run.manifest_id || 'no manifest'})
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>Model preset</div>
                <select className="field" value={seedPreset} onChange={(e) => setSeedPreset(e.target.value)}>
                  <option value="">Select a preset...</option>
                  {selectedRunModels.map((model) => (
                    <option key={model.preset} value={model.preset}>
                      {model.preset}
                    </option>
                  ))}
                </select>
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                <input
                  type="checkbox"
                  checked={seedOverwrite}
                  onChange={(e) => setSeedOverwrite(e.target.checked)}
                />
                Overwrite existing frame points
              </label>
              <button className="btn" onClick={seedFromRun} disabled={!manifestId || !seedRunId || !seedPreset}>
                Seed from model run
              </button>
            </div>
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Points in this frame</div>
            {frameAnnotation.points?.length ? (
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Team</th>
                    <th>X</th>
                    <th>Y</th>
                  </tr>
                </thead>
                <tbody>
                  {frameAnnotation.points.map((pt) => (
                    <tr
                      key={pt.id}
                      onClick={() => setSelectedEntity(pt.id)}
                      style={{
                        background: selectedEntity === pt.id ? `${C.grass}18` : 'transparent',
                        cursor: 'pointer',
                      }}
                    >
                      <td>{pt.id}</td>
                      <td style={{ color: (LABEL_STYLES[pt.team] || LABEL_STYLES.Other).color }}>{pt.team}</td>
                      <td>{pt.x.toFixed(1)}</td>
                      <td>{pt.y.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="muted">No points yet.</div>
            )}
          </div>

          <div className="card" style={{ padding: 16 }}>
            <div className="muted" style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Status</div>
            <div style={{ fontSize: 13, lineHeight: 1.6 }}>{status}</div>
          </div>
        </aside>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
