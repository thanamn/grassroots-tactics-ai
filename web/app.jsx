// Grassroots Tactics AI — single-page React frontend.
//
// Adapted from the Claude-design `app.jsx` prototype (dark navy + neon-green
// theme, Rajdhani / Inter fonts, hero + StatCard + TimestampPill components).
//
// Differences from the prototype:
//   - All `window.claude.complete` chat is removed.
//   - TacticalBoard screen, formation badges, heatmap toggle, training-plan
//     button, AI-coaching-notes list, and the Tweaks panel are dropped —
//     none of those map to data we actually compute.
//   - The Onboarding flow's mock `processMsgs` ticker is replaced with real
//     polling against /api/jobs/<id>.
//   - The Dashboard's "Recent Sessions" mock array is replaced with a real
//     GET /api/jobs.
//   - Stats grid pulls from metrics.summary (Team A spread, Team B spread,
//     centroid distance, events count) instead of made-up percentages.
//   - Bilingual (en / th): UI chrome from a small T dict, AI text from
//     /api/jobs/{id}/result?lang=<en|th>.

const { useEffect, useRef, useState, useCallback } = React;

// ── Theme ───────────────────────────────────────────────────────────────────
const C = {
  bg: '#0D1B2A', card: '#1A2B3C', border: '#243B52',
  green: '#39FF14', greenDim: '#2BC710', greenAlpha: 'rgba(57,255,20,0.12)',
  white: '#FFFFFF', gray: '#7A9BB5', grayLight: '#B0C7D9',
  red: '#FF4558', yellow: '#FFB800', blue: '#4DA6FF',
  teamA: '#f5a623',  // matches overlay video
  teamB: '#2196f3',
};

// Faint hex pattern under the hero — matches the prototype's `hexPattern`.
const HEX_PATTERN = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='30' height='52' viewBox='0 0 30 52'%3E%3Cpolygon points='15,1 29,8 29,22 15,29 1,22 1,8' fill='none' stroke='%23162840' stroke-width='1'/%3E%3Cpolygon points='15,27 29,34 29,48 15,55 1,48 1,34' fill='none' stroke='%23162840' stroke-width='1'/%3E%3C/svg%3E")`;

// ── Bilingual UI strings ────────────────────────────────────────────────────
// AI-generated text (headline / implication / coaching cue) is fetched
// already-translated from the backend. This dict is only for the UI chrome
// participants will read while operating the app.
const T = {
  en: {
    eyebrow: 'AI Tactical Spacing Analysis',
    hero1: 'Upload a match video.',
    hero2: 'Get instant tactical insights.',
    heroLead: 'Your AI assistant coach watches every minute, spots spacing patterns you might miss, and tells you what to work on next session.',
    drop: 'Drag & drop your video',
    dropDrag: 'Drop it here!',
    dropHint: 'MP4, MOV, AVI · up to 500 MB · max ~10 min on CPU',
    browse: 'Browse files',
    analyseCta: '+ Analyse video',
    statVideos: 'Videos analysed',
    statDuration: 'Total minutes',
    statEvents: 'Spacing events',
    statLatest: 'Latest clip',
    recent: 'Recent analyses',
    noRecent: 'No analyses yet — upload a video to get started.',
    onbUpload: 'Upload your video',
    onbUploadSub: "Match or training footage. We'll handle the rest.",
    onbTag: 'Tag your session',
    onbTagSub: 'A little context helps your AI coach be more relevant.',
    sessionType: 'Session type',
    match: '⚽ Match',
    training: '🏃 Training',
    opponent: 'Opponent',
    notes: 'Notes for AI (optional)',
    startAnalysis: 'Start AI analysis →',
    analysing: 'Analysing your video',
    cpuNote: 'Running on CPU — this can take several minutes for short clips, much longer for long ones.',
    fastMode: 'Fast mode',
    fastModeDetail: 'every {stride} frames · {model}',
    done: 'Analysis complete!',
    doneSub: 'Found {events} spacing events in {dur}s of footage.',
    viewAnalysis: 'View full analysis →',
    backDash: '← Back',
    momentsTitle: 'Key moments',
    momentsSub: 'Sudden ≥25% shape changes within 1.5 s',
    insightsTitle: 'What the AI sees',
    summaryTitle: 'Tactical summary',
    summaryObservation: 'Observation',
    summaryImplication: 'Implication',
    summaryStats: 'By the numbers',
    coachingCue: 'Coaching cue',
    chatTitle: 'Ask your AI coach',
    chatGreeting: "I've analysed your clip. Ask me about your team's spacing — when they spread out, when they compressed, what to work on next session.",
    chatPlaceholder: 'Try "what should I fix in our spacing?"',
    chatSend: 'Send',
    chatClear: 'Clear chat',
    chatThinking: 'Thinking…',
    chatError: 'Could not reach AI. Try again in a moment.',
    snapshot: 'Match snapshot',
    teamA: 'Team A spread',
    teamB: 'Team B spread',
    gap: 'Avg team gap',
    events: 'Shape events',
    unitNote: 'Areas in k px² (pixel space, not metres). Compare clips relatively.',
    chartA: 'Team A',
    chartB: 'Team B',
    stretch: 'spread out',
    compress: 'compressed',
    stretchNote: 'creating space — opponents can exploit gaps',
    compressNote: 'defending tight — harder to play through',
    noEvents: 'No sudden shape changes detected in this clip.',
    explainerMissing: 'Explanation not available — AI may be rate-limited. Try again.',
    explainerRetry: 'Retry explanation',
    explainerRetrying: 'Thinking…',
    possA: 'Team A possession',
    possB: 'Team B possession',
    passCtA: 'Team A passes',
    passCtB: 'Team B passes',
    passAccA: 'Team A pass acc.',
    passAccB: 'Team B pass acc.',
    possTitle: 'Possession & Passing',
    error: 'Something went wrong',
    delete: 'Delete',
    confirmDelete: 'Delete this analysis and all artefacts?',
    teamALabel: 'Team A',
    teamBLabel: 'Team B',
    myTeamQ: 'Which team is yours?',
    yourTeam: 'Your Team',
    opponent: 'Opponent',
    possLabel: 'possession',
    passLabel: 'passes',
    accLabel: 'pass acc.',
    moreSpread: 'More spread',
    moreCompact: 'More compact',
    balanced: 'Balanced',
    spacingNote: 'Higher area = more spread out. Compare between teams, not across clips.',
  },
  th: {
    eyebrow: 'AI วิเคราะห์การยืนตำแหน่งของทีม',
    hero1: 'อัปโหลดคลิปการแข่งขัน',
    hero2: 'รับคำแนะนำเชิงกลยุทธ์ทันที',
    heroLead: 'AI ดูทุกจังหวะของคลิป จับรูปแบบการยืนตำแหน่งที่คุณอาจมองข้าม และบอกสิ่งที่ควรซ้อมในเซสชั่นต่อไป',
    drop: 'ลากและวางคลิปที่นี่',
    dropDrag: 'วางตรงนี้ได้เลย!',
    dropHint: 'MP4, MOV, AVI · ไม่เกิน 500 MB · CPU รับได้ราว 10 นาที',
    browse: 'เลือกไฟล์',
    analyseCta: '+ วิเคราะห์คลิป',
    statVideos: 'คลิปที่วิเคราะห์',
    statDuration: 'นาทีรวม',
    statEvents: 'จังหวะรูปแบบเปลี่ยน',
    statLatest: 'คลิปล่าสุด',
    recent: 'การวิเคราะห์ล่าสุด',
    noRecent: 'ยังไม่มีการวิเคราะห์ — อัปโหลดคลิปเพื่อเริ่มต้น',
    onbUpload: 'อัปโหลดคลิป',
    onbUploadSub: 'คลิปการแข่งหรือฝึกซ้อม — ให้เราจัดการเอง',
    onbTag: 'ใส่บริบทคลิป',
    onbTagSub: 'ข้อมูลเล็กน้อยช่วยให้ AI วิเคราะห์ได้ตรงจุดยิ่งขึ้น',
    sessionType: 'ประเภทเซสชั่น',
    match: '⚽ แข่งขัน',
    training: '🏃 ฝึกซ้อม',
    opponent: 'คู่แข่ง',
    notes: 'หมายเหตุถึง AI (ไม่จำเป็น)',
    startAnalysis: 'เริ่มวิเคราะห์ →',
    analysing: 'กำลังวิเคราะห์คลิป',
    cpuNote: 'รันบน CPU — คลิปสั้นใช้เวลาหลายนาที คลิปยาวจะนานกว่านั้นมาก',
    fastMode: 'โหมดเร็ว',
    fastModeDetail: 'ทุก {stride} เฟรม · {model}',
    done: 'วิเคราะห์เสร็จแล้ว!',
    doneSub: 'พบ {events} จังหวะรูปแบบเปลี่ยน ในคลิป {dur} วินาที',
    viewAnalysis: 'ดูผลวิเคราะห์ →',
    backDash: '← กลับ',
    momentsTitle: 'ช่วงสำคัญ',
    momentsSub: 'การเปลี่ยนรูปแบบ ≥25% ภายใน 1.5 วินาที',
    insightsTitle: 'AI วิเคราะห์ว่า',
    summaryTitle: 'สรุปเชิงกลยุทธ์',
    summaryObservation: 'สิ่งที่สังเกตได้',
    summaryImplication: 'ผลที่ตามมา',
    summaryStats: 'ตัวเลขสรุป',
    coachingCue: 'คำแนะนำสำหรับโค้ช',
    chatTitle: 'ถาม AI โค้ช',
    chatGreeting: 'วิเคราะห์คลิปเสร็จแล้ว ถามได้เลยเรื่องการยืนตำแหน่งของทีม — เมื่อไหร่ทีมกระจายตัว เมื่อไหร่บีบเข้า และควรซ้อมอะไรในเซสชั่นต่อไป',
    chatPlaceholder: 'ลองพิมพ์ "ควรแก้จุดไหนบ้าง?"',
    chatSend: 'ส่ง',
    chatClear: 'ล้างแชท',
    chatThinking: 'กำลังคิด…',
    chatError: 'ติดต่อ AI ไม่ได้ ลองอีกครั้งในอีกสักครู่',
    snapshot: 'ภาพรวมการแข่ง',
    teamA: 'พื้นที่เฉลี่ยทีม A',
    teamB: 'พื้นที่เฉลี่ยทีม B',
    gap: 'ระยะเฉลี่ยระหว่างทีม',
    events: 'จังหวะที่เปลี่ยน',
    unitNote: 'พื้นที่หน่วย k px² (พิกเซล ไม่ใช่เมตร) เทียบระหว่างคลิป ไม่ใช่ค่าสัมบูรณ์',
    chartA: 'ทีม A',
    chartB: 'ทีม B',
    stretch: 'ยืนกระจายตัว',
    compress: 'บีบเข้ามา',
    stretchNote: 'เปิดช่องว่าง — คู่แข่งใช้ประโยชน์ได้',
    compressNote: 'ป้องกันแน่น — คู่แข่งเล่นผ่านยาก',
    noEvents: 'ไม่พบการเปลี่ยนรูปแบบอย่างชัดเจนในคลิปนี้',
    explainerMissing: 'ยังไม่มีคำอธิบาย — AI อาจถูกจำกัดอัตรา ลองอีกครั้งได้',
    explainerRetry: 'ลองสร้างคำอธิบายใหม่',
    explainerRetrying: 'กำลังคิด…',
    possA: 'ครองบอลทีม A',
    possB: 'ครองบอลทีม B',
    passCtA: 'ส่งบอลทีม A',
    passCtB: 'ส่งบอลทีม B',
    passAccA: 'ความแม่นทีม A',
    passAccB: 'ความแม่นทีม B',
    possTitle: 'การครองบอล & การส่งบอล',
    error: 'มีบางอย่างผิดพลาด',
    delete: 'ลบ',
    confirmDelete: 'ลบการวิเคราะห์และไฟล์ที่เกี่ยวข้องทั้งหมด?',
    teamALabel: 'ทีม A',
    teamBLabel: 'ทีม B',
    myTeamQ: 'ทีมไหนคือทีมของคุณ?',
    yourTeam: 'ทีมของฉัน',
    opponent: 'คู่แข่ง',
    possLabel: 'ครองบอล',
    passLabel: 'ส่งบอล',
    accLabel: 'ความแม่น',
    moreSpread: 'กระจายกว่า',
    moreCompact: 'บีบกว่า',
    balanced: 'สมดุล',
    spacingNote: 'พื้นที่มากกว่า = กระจายตัวมากกว่า เทียบระหว่างสองทีมในคลิปเดียวกัน',
  },
};

// ── Global styles (injected once) ───────────────────────────────────────────
(function injectStyles() {
  if (document.getElementById('gt-style')) return;
  const el = document.createElement('style');
  el.id = 'gt-style';
  el.textContent = `
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #root { height: 100%; }
    body { background: ${C.bg}; color: ${C.white}; font-family: 'Inter', sans-serif; overflow-x: hidden; }
    h1,h2,h3,h4,h5 { font-family: 'Rajdhani', sans-serif; letter-spacing: 0.02em; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: ${C.bg}; }
    ::-webkit-scrollbar-thumb { background: ${C.border}; border-radius: 4px; }
    button { cursor: pointer; border: none; outline: none; font-family: inherit; }
    input, textarea, select { outline: none; font-family: inherit; }
    a { color: ${C.green}; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
    @keyframes slideUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:none} }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes ping { 0%{transform:scale(1);opacity:1} 100%{transform:scale(2.2);opacity:0} }
    @keyframes fadeIn { from{opacity:0} to{opacity:1} }
    @keyframes drift { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
    .slide-up { animation: slideUp 0.35s ease both; }
    .fade-in { animation: fadeIn 0.3s ease both; }
    .nav-link { transition: color 0.2s; color: ${C.gray}; text-decoration: none; font-size: 14px; font-weight: 500; padding: 6px 14px; border-radius: 6px; background: none; }
    .nav-link:hover { color: ${C.white}; background: rgba(255,255,255,0.06); }
    .nav-link.active { color: ${C.green}; }
    .btn-primary { background: ${C.green}; color: ${C.bg}; font-family:'Rajdhani',sans-serif; font-weight:700; font-size:15px; letter-spacing:0.06em; padding: 12px 28px; border-radius: 8px; transition: all 0.2s; display:inline-flex;align-items:center;gap:8px; }
    .btn-primary:hover { background: #4fff24; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(57,255,20,0.35); }
    .btn-primary:disabled { background: ${C.border}; color: ${C.gray}; cursor: not-allowed; transform: none; box-shadow: none; }
    .btn-secondary { background: transparent; color: ${C.white}; border: 1px solid ${C.border}; font-family:'Inter',sans-serif; font-size:14px; font-weight:500; padding: 10px 20px; border-radius: 8px; transition: all 0.2s; display:inline-flex;align-items:center;gap:8px; }
    .btn-secondary:hover { border-color: ${C.gray}; background: rgba(255,255,255,0.05); }
    .card { background: ${C.card}; border: 1px solid ${C.border}; border-radius: 12px; }
    .input { width: 100%; background: ${C.bg}; border: 1px solid ${C.border}; border-radius: 8px; padding: 12px 14px; color: ${C.white}; font-size: 14px; font-family: 'Inter', sans-serif; transition: border-color 0.2s; }
    .input:focus { border-color: ${C.green}; }
    textarea.input { resize: vertical; min-height: 60px; }
    .analysis-grid { display: grid; grid-template-columns: minmax(0,1fr) 380px; gap: 20px; align-items: start; }
    @media (max-width: 900px) { .analysis-grid { grid-template-columns: 1fr; } }
    .hero-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 48px; align-items: center; }
    @media (max-width: 860px) { .hero-grid { grid-template-columns: 1fr; } }
  `;
  document.head.appendChild(el);
})();

// ── Shared bits ─────────────────────────────────────────────────────────────

function ShieldLogo() {
  return (
    <svg width="32" height="36" viewBox="0 0 32 36">
      <path d="M16 1 L31 7 L31 20 Q31 30 16 35 Q1 30 1 20 L1 7 Z" fill={C.green} opacity="0.15" stroke={C.green} strokeWidth="1.5" />
      <path d="M16 6 L27 11 L27 19 Q27 26 16 30 Q5 26 5 19 L5 11 Z" fill={C.green} opacity="0.3" />
      <text x="16" y="22" textAnchor="middle" fontSize="12" fontWeight="700" fill={C.green} fontFamily="Rajdhani,sans-serif">GT</text>
    </svg>
  );
}

function Nav({ screen, setScreen, lang, setLang }) {
  return (
    <header style={{ position: 'sticky', top: 0, zIndex: 50, background: 'rgba(13,27,42,0.92)', backdropFilter: 'blur(12px)', borderBottom: `1px solid ${C.border}` }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '0 24px', height: 60, display: 'flex', alignItems: 'center', gap: 16 }}>
        <button onClick={() => setScreen({ name: 'dashboard' })} style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'none', padding: 0 }}>
          <ShieldLogo />
          <span style={{ fontFamily: 'Rajdhani,sans-serif', fontWeight: 700, fontSize: 18, letterSpacing: '0.05em', color: C.white }}>
            GRASSROOTS <span style={{ color: C.green }}>TACTICS</span> AI
          </span>
        </button>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Language toggle */}
          <div style={{ display: 'flex', background: C.bg, border: `1px solid ${C.border}`, borderRadius: 20, padding: 2 }}>
            {['en', 'th'].map((l) => (
              <button key={l} onClick={() => setLang(l)} style={{
                padding: '4px 12px', borderRadius: 18, fontSize: 12, fontWeight: 600,
                background: lang === l ? C.green : 'transparent',
                color: lang === l ? C.bg : C.gray,
                transition: 'all 0.2s',
              }}>{l === 'en' ? 'EN' : 'TH'}</button>
            ))}
          </div>
          <button className="btn-primary" style={{ padding: '8px 18px', fontSize: 13 }} onClick={() => setScreen({ name: 'onboarding' })}>
            <span>+</span> {T[lang].analyseCta.replace('+ ', '')}
          </button>
        </div>
      </div>
    </header>
  );
}

function Pulse({ size = 8, color = C.green }) {
  return <span style={{ width: size, height: size, borderRadius: '50%', background: color, display: 'inline-block', animation: 'pulse 2s infinite', boxShadow: `0 0 ${size}px ${color}` }} />;
}

function StatCard({ number, label, accent = C.green, sub }) {
  return (
    <div className="card" style={{ padding: '20px 24px', flex: 1, minWidth: 160 }}>
      <div style={{ fontSize: 11, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>{label}</div>
      <div style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 36, fontWeight: 700, color: accent, lineHeight: 1 }}>{number}</div>
      {sub && <div style={{ marginTop: 8, fontSize: 12, color: C.gray }}>{sub}</div>}
    </div>
  );
}

// ── API helpers ─────────────────────────────────────────────────────────────
async function apiListJobs() {
  const r = await fetch('/api/jobs');
  return r.ok ? r.json() : [];
}
async function apiGetJob(jobId) {
  const r = await fetch(`/api/jobs/${jobId}`);
  return r.ok ? r.json() : null;
}
async function apiGetResult(jobId, lang) {
  const r = await fetch(`/api/jobs/${jobId}/result?lang=${lang}`);
  return r.ok ? r.json() : null;
}
async function apiUpload(file, ctx) {
  const fd = new FormData();
  fd.append('video', file);
  fd.append('session_type', ctx.session_type);
  if (ctx.opponent) fd.append('opponent', ctx.opponent);
  if (ctx.notes) fd.append('notes', ctx.notes);
  fd.append('language', ctx.language || 'en');
  const r = await fetch('/api/jobs', { method: 'POST', body: fd });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`Upload failed (${r.status}): ${text || r.statusText}`);
  }
  return r.json();
}
async function apiDeleteJob(jobId) {
  await fetch(`/api/jobs/${jobId}`, { method: 'DELETE' });
}
async function apiRegenerateExplanation(jobId, lang) {
  const r = await fetch(`/api/jobs/${jobId}/explain?lang=${lang}`, { method: 'POST' });
  if (!r.ok) throw new Error(`(${r.status}) ${await r.text().catch(() => r.statusText)}`);
  return r.json();
}
async function apiChat(jobId, question, history, lang) {
  const r = await fetch(`/api/jobs/${jobId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history, lang }),
  });
  if (!r.ok) throw new Error(`(${r.status}) ${await r.text().catch(() => r.statusText)}`);
  return r.json();
}

// ── Dashboard ───────────────────────────────────────────────────────────────
function Dashboard({ setScreen, lang }) {
  const t = T[lang];
  const [dragOver, setDragOver] = useState(false);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const fileRef = useRef(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setJobs(await apiListJobs());
    setLoading(false);
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const onPick = (file) => {
    if (!file) return;
    setScreen({ name: 'onboarding', preselected: file });
  };

  const totalSeconds = jobs.reduce((s, j) => s + (j.duration_s || 0), 0);
  const totalEvents = jobs.reduce((s, j) => s + (j.events_count || 0), 0);
  const latest = jobs.find((j) => j.status === 'done');

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      {/* Hero */}
      <div style={{ backgroundImage: HEX_PATTERN, borderBottom: `1px solid ${C.border}`, padding: '56px 24px' }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }} className="hero-grid">

          {/* Left: text */}
          <div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 16 }}>
              <Pulse />
              <span style={{ fontSize: 12, color: C.green, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{t.eyebrow}</span>
            </div>
            <h1 style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 'clamp(32px,3.5vw,54px)', fontWeight: 700, lineHeight: 1.1, marginBottom: 16 }}>
              {t.hero1}<br />
              <span style={{ color: C.green }}>{t.hero2}</span>
            </h1>
            <p style={{ color: C.gray, fontSize: 15, lineHeight: 1.7, marginBottom: 32, maxWidth: 480 }}>{t.heroLead}</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                { icon: '⚽', text: lang === 'th' ? 'ติดตามผู้เล่นทุกคนอัตโนมัติ' : 'Automatic player tracking' },
                { icon: '📐', text: lang === 'th' ? 'วัดระยะห่างและพื้นที่ทีมแบบเรียลไทม์' : 'Measure spacing & team shape' },
                { icon: '🤖', text: lang === 'th' ? 'AI อธิบายเป็นภาษาโค้ช' : 'AI explains in coach language' },
              ].map(({ icon, text }) => (
                <div key={text} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14, color: C.grayLight }}>
                  <span style={{ fontSize: 16 }}>{icon}</span>
                  {text}
                </div>
              ))}
            </div>
          </div>

          {/* Right: upload box */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault(); setDragOver(false);
              const f = e.dataTransfer.files?.[0]; if (f) onPick(f);
            }}
            onClick={() => fileRef.current?.click()}
            style={{
              border: `2px dashed ${dragOver ? C.green : C.border}`,
              borderRadius: 20, padding: '48px 32px', textAlign: 'center', cursor: 'pointer',
              background: dragOver ? C.greenAlpha : 'rgba(26,43,60,0.5)',
              backdropFilter: 'blur(8px)', transition: 'all 0.25s',
              boxShadow: dragOver ? '0 0 40px rgba(57,255,20,0.2)' : '0 4px 24px rgba(0,0,0,0.3)',
            }}>
            <input ref={fileRef} type="file" accept="video/*" style={{ display: 'none' }}
                   onChange={(e) => onPick(e.target.files?.[0])} />
            <div style={{ marginBottom: 16 }}>
              <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                <rect width="64" height="64" rx="16" fill={dragOver ? 'rgba(57,255,20,0.15)' : 'rgba(36,59,82,0.8)'} />
                <rect x="10" y="18" width="34" height="26" rx="4" stroke={dragOver ? C.green : C.gray} strokeWidth="2.5" fill="none"/>
                <path d="M44 26 L54 20 L54 44 L44 38" stroke={dragOver ? C.green : C.gray} strokeWidth="2.5" strokeLinejoin="round" fill="none"/>
                <circle cx="22" cy="31" r="4" fill={dragOver ? C.green : C.gray} opacity="0.5"/>
              </svg>
            </div>
            <div style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 22, fontWeight: 700, marginBottom: 8, color: dragOver ? C.green : C.white }}>
              {dragOver ? t.dropDrag : t.drop}
            </div>
            <div style={{ fontSize: 13, color: C.gray, marginBottom: 28, lineHeight: 1.5 }}>{t.dropHint}</div>
            <button className="btn-primary" style={{ fontSize: 15, padding: '12px 32px' }}
                    onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }}>
              {t.browse}
            </button>
          </div>
        </div>
      </div>

      {/* Stats + Recent */}
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '28px 24px 0' }}>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 28 }}>
          <StatCard number={jobs.filter((j) => j.status === 'done').length} label={t.statVideos} />
          <StatCard number={(totalSeconds / 60).toFixed(1)} label={t.statDuration} accent={C.blue} />
          <StatCard number={totalEvents} label={t.statEvents} accent={C.yellow} />
          <StatCard
            number={latest ? new Date(latest.created_at).toLocaleDateString() : '—'}
            label={t.statLatest}
            accent={C.white}
          />
        </div>

        {/* Recent */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, letterSpacing: '0.02em' }}>{t.recent}</h2>
          <div style={{ flex: 1, height: 1, background: C.border }} />
        </div>
        {loading ? (
          <div style={{ color: C.gray, padding: 12 }}>…</div>
        ) : jobs.length === 0 ? (
          <div className="card" style={{ padding: 18, color: C.gray, fontSize: 14 }}>{t.noRecent}</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 40 }}>
            {jobs.map((j, i) => (
              <RecentRow key={j.job_id} job={j} delay={i * 60} t={t}
                         onOpen={() => {
                           if (j.status === 'done') setScreen({ name: 'analysis', jobId: j.job_id });
                           else setScreen({ name: 'progress', jobId: j.job_id });
                         }}
                         onDelete={async () => {
                           if (!confirm(t.confirmDelete)) return;
                           await apiDeleteJob(j.job_id);
                           refresh();
                         }} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RecentRow({ job, delay, t, onOpen, onDelete }) {
  const isMatch = job.session_type === 'match';
  const statusColor =
    job.status === 'done' ? C.green :
    job.status === 'error' ? C.red :
    C.yellow;
  return (
    <div className="card slide-up" style={{ animationDelay: `${delay}ms`, display: 'flex', alignItems: 'center', gap: 16, padding: '14px 18px', cursor: 'pointer' }}
         onClick={onOpen}>
      <div style={{
        width: 60, height: 42, borderRadius: 8, flexShrink: 0, fontSize: 20,
        background: isMatch ? 'linear-gradient(135deg, #1a4a1a 0%, #0d2b1a 100%)' : 'linear-gradient(135deg, #1a2a4a 0%, #0d1b3a 100%)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {isMatch ? '⚽' : '🏃'}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {job.opponent ? `vs ${job.opponent}` : job.filename}
        </div>
        <div style={{ display: 'flex', gap: 12, fontSize: 12, color: C.gray, flexWrap: 'wrap' }}>
          <span>{new Date(job.created_at).toLocaleString()}</span>
          {job.duration_s ? <span>· {job.duration_s.toFixed(1)} s</span> : null}
          {job.events_count != null ? <span>· <span style={{ color: C.green }}>{job.events_count} events</span></span> : null}
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: statusColor, padding: '4px 10px', borderRadius: 20, background: `${statusColor}1a`, border: `1px solid ${statusColor}33` }}>
        {job.status !== 'done' && job.status !== 'error' && <Pulse size={6} color={statusColor} />}
        {job.status}
      </div>
      <button className="btn-secondary" style={{ padding: '6px 10px', fontSize: 12 }}
              onClick={(e) => { e.stopPropagation(); onDelete(); }}>
        {t.delete}
      </button>
    </div>
  );
}

// ── Onboarding (upload + tag + processing + done) ──────────────────────────
function Onboarding({ setScreen, lang, preselected }) {
  const t = T[lang];
  const [step, setStep] = useState(preselected ? 2 : 1);
  const [file, setFile] = useState(preselected || null);
  const [dragOver, setDragOver] = useState(false);
  const [ctx, setCtx] = useState({ session_type: 'match', opponent: '', notes: '' });
  const [uploadPct, setUploadPct] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [jobId, setJobId] = useState(null);

  const fileRef = useRef(null);
  const onPickFile = (f) => { if (f) { setFile(f); setStep(2); } };

  // Upload via XHR so we can show progress %.
  const startUpload = () => {
    if (!file) return;
    setUploading(true); setUploadPct(0); setError(null);
    const fd = new FormData();
    fd.append('video', file);
    fd.append('session_type', ctx.session_type);
    if (ctx.opponent) fd.append('opponent', ctx.opponent);
    if (ctx.notes) fd.append('notes', ctx.notes);
    fd.append('language', lang);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadPct(Math.round(100 * e.loaded / e.total));
    };
    xhr.onload = () => {
      setUploading(false);
      if (xhr.status >= 200 && xhr.status < 300) {
        const job = JSON.parse(xhr.responseText);
        setJobId(job.job_id);
        setStep(3);
      } else {
        setError(`${xhr.status}: ${xhr.responseText || xhr.statusText}`);
      }
    };
    xhr.onerror = () => { setUploading(false); setError('Network error'); };
    xhr.send(fd);
  };

  const stepLabels = ['Upload', 'Tag', 'Analysing', 'Done'];

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '40px 24px' }}>
      <div style={{ maxWidth: 600, margin: '0 auto' }}>
        {/* Step indicator */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 36 }}>
          {stepLabels.map((s, i) => {
            const done = step > i + 1, active = step === i + 1;
            return (
              <React.Fragment key={s}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flex: 1 }}>
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%',
                    border: `2px solid ${done ? C.green : active ? C.green : C.border}`,
                    background: done ? C.green : active ? C.greenAlpha : 'transparent',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, fontWeight: 700,
                    color: done ? C.bg : active ? C.green : C.gray,
                    transition: 'all 0.3s',
                  }}>{done ? '✓' : i + 1}</div>
                  <span style={{ fontSize: 11, color: active ? C.green : done ? C.grayLight : C.gray, fontWeight: active ? 600 : 400 }}>{s}</span>
                </div>
                {i < stepLabels.length - 1 && (
                  <div style={{ height: 2, flex: 1, background: step > i + 1 ? C.green : C.border, marginBottom: 22, transition: 'background 0.4s' }} />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {error && (
          <div className="card slide-up" style={{ padding: 16, marginBottom: 16, borderColor: C.red, color: C.red, background: `${C.red}11` }}>
            <strong>{t.error}:</strong> {error}
          </div>
        )}

        {step === 1 && (
          <div className="slide-up card" style={{ padding: 32 }}>
            <h2 style={{ fontSize: 26, fontWeight: 700, marginBottom: 6 }}>{t.onbUpload}</h2>
            <p style={{ color: C.gray, fontSize: 14, marginBottom: 28, lineHeight: 1.5 }}>{t.onbUploadSub}</p>
            <input ref={fileRef} type="file" accept="video/*" style={{ display: 'none' }}
                   onChange={(e) => onPickFile(e.target.files?.[0])} />
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault(); setDragOver(false);
                onPickFile(e.dataTransfer.files?.[0]);
              }}
              onClick={() => fileRef.current?.click()}
              style={{ border: `2px dashed ${dragOver ? C.green : C.border}`, borderRadius: 14, padding: '48px 32px', textAlign: 'center', cursor: 'pointer', background: dragOver ? C.greenAlpha : C.bg, transition: 'all 0.2s' }}>
              <div style={{ fontSize: 48, marginBottom: 16, animation: 'drift 3s ease-in-out infinite' }}>📹</div>
              <div style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 20, fontWeight: 700, marginBottom: 8 }}>
                {dragOver ? t.dropDrag : t.drop}
              </div>
              <div style={{ color: C.gray, fontSize: 13, marginBottom: 24 }}>{t.dropHint}</div>
              <button className="btn-primary" onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }}>
                {t.browse}
              </button>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="slide-up card" style={{ padding: 32 }}>
            <h2 style={{ fontSize: 26, fontWeight: 700, marginBottom: 6 }}>{t.onbTag}</h2>
            <p style={{ color: C.gray, fontSize: 14, marginBottom: 20 }}>{t.onbTagSub}</p>
            {file && <div style={{ marginBottom: 18, fontSize: 13, color: C.grayLight, fontFamily: 'monospace', padding: '8px 12px', background: C.bg, borderRadius: 8, border: `1px solid ${C.border}` }}>
              📁 {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
            </div>}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              <div>
                <label style={{ fontSize: 12, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10, display: 'block' }}>{t.sessionType}</label>
                <div style={{ display: 'flex', gap: 10 }}>
                  {[['match', t.match], ['training', t.training]].map(([v, label]) => (
                    <button key={v} onClick={() => setCtx((c) => ({ ...c, session_type: v }))}
                            style={{
                              flex: 1, padding: '12px', borderRadius: 10,
                              border: `1.5px solid ${ctx.session_type === v ? C.green : C.border}`,
                              background: ctx.session_type === v ? C.greenAlpha : 'transparent',
                              color: ctx.session_type === v ? C.green : C.gray,
                              fontFamily: 'Rajdhani,sans-serif', fontWeight: 700, fontSize: 15,
                              transition: 'all 0.2s',
                            }}>{label}</button>
                  ))}
                </div>
              </div>
              {ctx.session_type === 'match' && (
                <div>
                  <label style={{ fontSize: 12, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8, display: 'block' }}>{t.opponent}</label>
                  <input className="input" value={ctx.opponent} onChange={(e) => setCtx((c) => ({ ...c, opponent: e.target.value }))} placeholder="Hartfield FC" />
                </div>
              )}
              <div>
                <label style={{ fontSize: 12, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8, display: 'block' }}>{t.notes}</label>
                <textarea className="input" value={ctx.notes} onChange={(e) => setCtx((c) => ({ ...c, notes: e.target.value }))} rows={3}
                          placeholder={lang === 'th' ? 'เปลี่ยนรูปแบบช่วงครึ่งหลัง...' : 'We changed shape at half time...'} />
              </div>
              {uploading ? (
                <div>
                  <div style={{ fontSize: 13, color: C.grayLight, marginBottom: 8, textAlign: 'center' }}>Uploading… {uploadPct}%</div>
                  <div style={{ height: 8, background: C.border, borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${uploadPct}%`, background: `linear-gradient(90deg, ${C.green}, #00C4FF)`, borderRadius: 4, transition: 'width 0.2s' }} />
                  </div>
                </div>
              ) : (
                <button className="btn-primary" style={{ width: '100%', justifyContent: 'center', marginTop: 4, fontSize: 16 }} onClick={startUpload}>
                  {t.startAnalysis}
                </button>
              )}
            </div>
          </div>
        )}

        {step === 3 && jobId && (
          <ProgressScreen jobId={jobId} lang={lang} onDone={() => setStep(4)} />
        )}

        {step === 4 && jobId && (
          <DoneScreen jobId={jobId} lang={lang} setScreen={setScreen} />
        )}
      </div>
    </div>
  );
}

// ── Progress (used by Onboarding step 3 and as a standalone screen) ────────
function ProgressScreen({ jobId, lang, onDone }) {
  const t = T[lang];
  const [job, setJob] = useState(null);

  useEffect(() => {
    let stopped = false;
    let timer;
    const tick = async () => {
      const j = await apiGetJob(jobId);
      if (stopped) return;
      setJob(j);
      if (j && j.status === 'done') { onDone?.(j); return; }
      if (j && j.status === 'error') return;
      timer = setTimeout(tick, 2000);
    };
    tick();
    return () => { stopped = true; if (timer) clearTimeout(timer); };
  }, [jobId, onDone]);

  if (!job) return <div className="card" style={{ padding: 32, textAlign: 'center', color: C.gray }}>…</div>;
  if (job.status === 'error') {
    return (
      <div className="card slide-up" style={{ padding: 32 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, color: C.red, marginBottom: 8 }}>{t.error}</h2>
        <div style={{ fontSize: 13, color: C.grayLight, lineHeight: 1.6, fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
          {job.error || job.stage_message}
        </div>
      </div>
    );
  }

  const pct = Math.round((job.stage_index / (job.stage_total || 5)) * 100);
  return (
    <div className="card slide-up" style={{ padding: 48, textAlign: 'center' }}>
      <div style={{ position: 'relative', width: 80, height: 80, margin: '0 auto 28px' }}>
        <div style={{ width: 80, height: 80, borderRadius: '50%', border: `3px solid ${C.green}33`, animation: 'spin 2s linear infinite', position: 'absolute' }} />
        <div style={{ width: 80, height: 80, borderRadius: '50%', border: '3px solid transparent', borderTopColor: C.green, animation: 'spin 1.2s linear infinite', position: 'absolute' }} />
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 32 }}>⚽</div>
      </div>
      <h2 style={{ fontSize: 22, fontWeight: 700, color: C.green, marginBottom: 6 }}>{t.analysing}</h2>
      <p style={{ color: C.grayLight, fontSize: 14, marginBottom: 8, minHeight: 20 }}>{job.stage_message}</p>
      {job.vid_stride > 1 && (
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6,
                      padding: '4px 12px', borderRadius: 16,
                      border: `1px solid ${C.yellow}44`,
                      background: `${C.yellow}11`,
                      color: C.yellow, fontSize: 11, fontWeight: 600,
                      letterSpacing: '0.06em', textTransform: 'uppercase',
                      marginBottom: 16 }}>
          ⚡ {t.fastMode} · {t.fastModeDetail
            .replace('{stride}', job.vid_stride)
            .replace('{model}', job.tracking_model || 'default')}
        </div>
      )}
      <p style={{ color: C.gray, fontSize: 12, marginBottom: 24 }}>{t.cpuNote}</p>
      <div style={{ height: 4, background: C.border, borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', background: `linear-gradient(90deg, ${C.green}, #00D4FF)`, borderRadius: 2, width: `${pct}%`, transition: 'width 0.6s ease' }} />
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: C.gray, fontFamily: 'monospace' }}>
        Step {job.stage_index} of {job.stage_total}
      </div>
    </div>
  );
}

function DoneScreen({ jobId, lang, setScreen }) {
  const t = T[lang];
  const [job, setJob] = useState(null);
  useEffect(() => { (async () => setJob(await apiGetJob(jobId)))(); }, [jobId]);
  if (!job) return null;
  return (
    <div className="card slide-up" style={{ padding: 48, textAlign: 'center' }}>
      <div style={{ width: 72, height: 72, borderRadius: '50%', background: C.green, margin: '0 auto 20px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        <div style={{ position: 'absolute', inset: -8, borderRadius: '50%', border: `2px solid ${C.green}`, animation: 'ping 1.5s ease-out infinite' }} />
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <path d="M7 16l7 7 11-12" stroke={C.bg} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8, color: C.green }}>{t.done}</h2>
      <p style={{ color: C.gray, fontSize: 14, lineHeight: 1.6, marginBottom: 28 }}>
        {t.doneSub
          .replace('{events}', job.events_count ?? 0)
          .replace('{dur}', (job.duration_s || 0).toFixed(1))}
      </p>
      <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }}
              onClick={() => setScreen({ name: 'analysis', jobId: job.job_id })}>
        {t.viewAnalysis}
      </button>
    </div>
  );
}

// ── Analysis (the "money screen") ──────────────────────────────────────────
function Analysis({ setScreen, lang, jobId }) {
  const t = T[lang];
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState(null);
  const [myTeam, setMyTeam] = useState(null);
  const videoRef = useRef(null);

  const reload = useCallback(async () => {
    const r = await apiGetResult(jobId, lang);
    if (!r) { setError('result-missing'); return; }
    setData(r);
  }, [jobId, lang]);

  useEffect(() => { reload(); }, [reload]);

  const onRetryExplanation = async () => {
    setRetrying(true); setRetryError(null);
    try {
      await apiRegenerateExplanation(jobId, lang);
      await reload();
    } catch (e) {
      setRetryError(String(e.message || e));
    } finally {
      setRetrying(false);
    }
  };

  if (error) return <div style={{ padding: 40, color: C.red }}>{t.error}</div>;
  if (!data) return <div style={{ padding: 40, color: C.gray }}>…</div>;

  const { job, metrics, explanation } = data;

  const labelA = myTeam === 'A' ? t.yourTeam : myTeam === 'B' ? t.opponent : t.teamALabel;
  const labelB = myTeam === 'B' ? t.yourTeam : myTeam === 'A' ? t.opponent : t.teamBLabel;

  const spreadTag = (area, other) => {
    if (!area || !other) return null;
    const ratio = area / other;
    if (ratio > 1.2) return { label: t.moreSpread, color: C.yellow };
    if (ratio < 0.83) return { label: t.moreCompact, color: C.blue };
    return { label: t.balanced, color: C.gray };
  };
  const tagA = spreadTag(teamA, teamB);
  const tagB = spreadTag(teamB, teamA);

  const summary = metrics.summary || {};
  const teamA = summary.team_A?.hull_area?.mean || 0;
  const teamB = summary.team_B?.hull_area?.mean || 0;
  const gap = summary.centroid_distance?.mean || 0;
  const events = metrics.events || [];
  const stretchCount = events.filter((e) => e.type === 'stretch').length;
  const compressCount = events.filter((e) => e.type === 'compactness_spike').length;

  const ball   = metrics.ball_metrics || {};
  const poss   = ball.possession_pct || {};
  const passes = ball.pass_count || {};
  const acc    = ball.pass_accuracy || {};

  const jumpTo = (t) => {
    if (videoRef.current) {
      videoRef.current.currentTime = t;
      videoRef.current.play().catch(() => {});
    }
  };

  return (
    <div style={{ flex: 1, overflowY: 'auto' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, fontSize: 13, color: C.gray }}>
          <button onClick={() => setScreen({ name: 'dashboard' })} style={{ background: 'none', color: C.gray, fontSize: 13 }}>{t.backDash}</button>
          <span>·</span>
          <span style={{ color: C.white }}>
            {job.opponent ? `vs ${job.opponent}` : job.filename}
          </span>
          <span style={{ color: C.gray }}>· {(job.duration_s || 0).toFixed(1)}s · {(job.fps || 0).toFixed(0)} fps</span>
        </div>

        <div className="analysis-grid">
          {/* Left: video + chart + events */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
            <div className="card" style={{ overflow: 'hidden' }}>
              <video ref={videoRef} src={`/api/jobs/${jobId}/overlay`} controls
                     style={{ width: '100%', display: 'block', background: '#000', aspectRatio: '16/9' }} />
              <div style={{ display: 'flex', gap: 16, padding: '10px 14px', borderTop: `1px solid ${C.border}` }}>
                {[['A', C.teamA, labelA], ['B', C.teamB, labelB]].map(([ab, color, label]) => (
                  <div key={ab} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: C.gray }}>
                    <span style={{ width: 24, height: 3, borderRadius: 2, background: color, display: 'inline-block' }} />
                    {label}
                  </div>
                ))}
              </div>
            </div>

            {/* Ask Your AI Coach — chat grounded in this clip's metrics */}
            <ChatPanel jobId={jobId} lang={lang} t={t} />

            {/* Key moments */}
            <div className="card" style={{ padding: 16 }}>
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: C.grayLight, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{t.momentsTitle}</div>
                <div style={{ fontSize: 12, color: C.gray, marginTop: 2 }}>{t.momentsSub}</div>
              </div>
              {events.length === 0 ? (
                <div style={{ padding: '12px 14px', borderRadius: 8, background: C.bg, color: C.gray, fontSize: 13 }}>
                  {t.noEvents}
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {events.map((ev, i) => (
                    <EventRow key={i} ev={ev} t={t} labelA={labelA} labelB={labelB} onJump={() => jumpTo(ev.t)} />
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Right rail */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            <TacticalSummary
              t={t}
              explanation={explanation}
              retrying={retrying}
              retryError={retryError}
              onRetryExplanation={onRetryExplanation}
              stretchCount={stretchCount}
              compressCount={compressCount}
              eventCount={events.length}
              gap={gap}
            />

            {/* Team picker */}
            <div className="card" style={{ padding: '14px 18px' }}>
              <div style={{ fontSize: 11, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>
                {t.myTeamQ}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                {[['A', C.teamA], ['B', C.teamB]].map(([ab, color]) => {
                  const label = ab === 'A' ? t.teamALabel : t.teamBLabel;
                  const active = myTeam === ab;
                  return (
                    <button key={ab} onClick={() => setMyTeam(active ? null : ab)} style={{
                      flex: 1, padding: '10px 12px', borderRadius: 10,
                      border: `2px solid ${active ? color : C.border}`,
                      background: active ? `${color}22` : 'transparent',
                      color: active ? color : C.gray,
                      fontFamily: 'Rajdhani,sans-serif', fontWeight: 700, fontSize: 14,
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                      transition: 'all 0.2s',
                    }}>
                      <span style={{ width: 12, height: 12, borderRadius: '50%', background: color, flexShrink: 0 }} />
                      {active ? t.yourTeam : label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Snapshot stats */}
            <div className="card" style={{ padding: 20 }}>
              <div style={{ fontSize: 11, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>{t.snapshot}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <MiniStat label={labelA} value={`${(teamA / 1000).toFixed(0)} k px²`} color={C.teamA} tag={tagA?.label} tagColor={tagA?.color} />
                <MiniStat label={labelB} value={`${(teamB / 1000).toFixed(0)} k px²`} color={C.teamB} tag={tagB?.label} tagColor={tagB?.color} />
                <MiniStat label={t.gap} value={`${gap.toFixed(0)} px`} color={C.green} />
                <MiniStat label={t.events} value={events.length} color={C.yellow} />
              </div>
              <div style={{ marginTop: 12, fontSize: 11, color: C.gray, lineHeight: 1.5 }}>
                {t.spacingNote}
              </div>
            </div>

            {/* Possession & pass card — only renders when ball data is available */}
            {Object.keys(poss).length > 0 && (
              <div className="card" style={{ padding: 20 }}>
                <div style={{ fontSize: 11, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>{t.possTitle}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <MiniStat label={`${labelA} ${t.possLabel}`}  value={`${(poss.A  || 0).toFixed(0)}%`}  color={C.teamA} />
                  <MiniStat label={`${labelB} ${t.possLabel}`}  value={`${(poss.B  || 0).toFixed(0)}%`}  color={C.teamB} />
                  <MiniStat label={`${labelA} ${t.passLabel}`}  value={passes.A != null ? passes.A : '–'} color={C.teamA} />
                  <MiniStat label={`${labelB} ${t.passLabel}`}  value={passes.B != null ? passes.B : '–'} color={C.teamB} />
                  <MiniStat label={`${labelA} ${t.accLabel}`}   value={acc.A != null ? `${(acc.A * 100).toFixed(0)}%` : '–'} color={C.teamA} />
                  <MiniStat label={`${labelB} ${t.accLabel}`}   value={acc.B != null ? `${(acc.B * 100).toFixed(0)}%` : '–'} color={C.teamB} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, color, tag, tagColor }) {
  return (
    <div style={{ background: C.bg, borderRadius: 8, padding: '10px 12px' }}>
      <div style={{ fontSize: 10, color: C.gray, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 4 }}>{label}</div>
      <div style={{ fontFamily: 'Rajdhani,sans-serif', fontSize: 22, fontWeight: 700, color }}>{value}</div>
      {tag && (
        <div style={{ marginTop: 4, fontSize: 10, fontWeight: 700, letterSpacing: '0.06em',
                      color: tagColor || color, background: `${tagColor || color}18`,
                      display: 'inline-block', padding: '2px 7px', borderRadius: 10 }}>
          {tag}
        </div>
      )}
    </div>
  );
}

function EventRow({ ev, t, labelA, labelB, onJump }) {
  const isStretch = ev.type === 'stretch';
  const teamColor = ev.team === 'team_A' ? C.teamA : C.teamB;
  const verb = isStretch ? t.stretch : t.compress;
  const note = isStretch ? t.stretchNote : t.compressNote;
  const teamLabel = ev.team === 'team_A' ? (labelA || t.teamALabel) : (labelB || t.teamBLabel);
  return (
    <button onClick={onJump} style={{
      display: 'flex', alignItems: 'center', gap: 12,
      background: C.bg, border: `1px solid ${C.border}`, borderLeft: `3px solid ${teamColor}`,
      borderRadius: 10, padding: '10px 14px', textAlign: 'left', width: '100%',
      transition: 'border-color 0.15s', color: C.grayLight,
    }}>
      <span style={{ fontFamily: 'monospace', fontWeight: 600, fontSize: 13, color: teamColor, minWidth: 56 }}>
        {ev.t.toFixed(1)}s
      </span>
      <span style={{ fontSize: 16 }}>{isStretch ? '📐' : '🔒'}</span>
      <div style={{ flex: 1, fontSize: 13, lineHeight: 1.5 }}>
        <strong style={{ color: C.white }}>{teamLabel}</strong> {verb} · <span style={{ color: C.gray }}>{note}</span>
      </div>
      <span style={{ fontFamily: 'monospace', fontSize: 11, fontWeight: 700, padding: '4px 10px', borderRadius: 20, background: `${teamColor}22`, color: teamColor }}>
        {ev.delta_pct >= 0 ? '+' : ''}{ev.delta_pct.toFixed(0)}%
      </span>
    </button>
  );
}

// ── Tactical Summary ───────────────────────────────────────────────────────
// Replaces the prior single-headline "What the AI sees" card. We split the
// existing Gemini explanation (headline / implication / coaching_cue) into
// three severity-coded items so the panel reads like the Claude design's
// `Tactical Summary` card. A fourth derived "by-the-numbers" item is
// stitched together from the events list — gives the reader a non-LLM
// fact alongside the AI text, which helps with explainability claims in
// the paper.
function SummaryItem({ icon, label, body, accent }) {
  return (
    <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start',
                  padding: '12px 14px', background: `${accent}10`,
                  border: `1px solid ${accent}33`, borderRadius: 10 }}>
      <div style={{ width: 28, height: 28, borderRadius: 6,
                    background: `${accent}22`, display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                    fontSize: 14, flexShrink: 0 }}>{icon}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: accent,
                      textTransform: 'uppercase', letterSpacing: '0.08em',
                      marginBottom: 4 }}>{label}</div>
        <div style={{ fontSize: 13, lineHeight: 1.5, color: C.grayLight }}>
          {body}
        </div>
      </div>
    </div>
  );
}

function TacticalSummary({
  t, explanation, retrying, retryError, onRetryExplanation,
  stretchCount, compressCount, eventCount, gap,
}) {
  // "By the numbers" sentence — derived from the metrics, not Gemini —
  // so the card has SOMETHING to show even when the explainer is missing.
  const stats = (
    `${eventCount} ${t.events.toLowerCase()} · ` +
    `${stretchCount} ${t.stretch} / ${compressCount} ${t.compress} · ` +
    `${gap.toFixed(0)} px ${t.gap.toLowerCase()}`
  );

  return (
    <div className="card" style={{ padding: 22 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
        <Pulse size={6} />
        <span style={{ fontSize: 11, color: C.green, fontWeight: 600,
                       letterSpacing: '0.12em', textTransform: 'uppercase' }}>
          {t.summaryTitle}
        </span>
      </div>
      {explanation ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <SummaryItem icon="🎯" label={t.summaryObservation}
                       body={explanation.headline} accent={C.blue} />
          <SummaryItem icon="⚠️" label={t.summaryImplication}
                       body={explanation.implication} accent={C.yellow} />
          <SummaryItem icon="✅" label={t.coachingCue}
                       body={explanation.coaching_cue} accent={C.green} />
          <SummaryItem icon="📊" label={t.summaryStats}
                       body={stats} accent={C.grayLight} />
          <details style={{ marginTop: 4, paddingTop: 8, borderTop: `1px solid ${C.border}` }}>
            <summary style={{ fontSize: 11, color: C.gray, cursor: 'pointer', listStyle: 'none', display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ opacity: 0.5 }}>ⓘ</span>
              <span style={{ fontFamily: 'monospace' }}>model info</span>
            </summary>
            <div style={{ marginTop: 6, fontSize: 11, color: C.gray, fontFamily: 'monospace' }}>
              {explanation.model || '?'} · prompt v{explanation.prompt_version || '?'}
            </div>
          </details>
        </div>
      ) : (
        <div style={{ color: C.yellow, fontSize: 13, lineHeight: 1.6 }}>
          {t.explainerMissing}
          {retryError && (
            <div style={{ marginTop: 8, color: C.red, fontFamily: 'monospace', fontSize: 11 }}>
              {retryError}
            </div>
          )}
          <button className="btn-primary" disabled={retrying}
                  style={{ marginTop: 12, width: '100%', justifyContent: 'center', fontSize: 13 }}
                  onClick={onRetryExplanation}>
            {retrying ? t.explainerRetrying : t.explainerRetry}
          </button>
        </div>
      )}
    </div>
  );
}

// ── Ask Your AI Coach ─────────────────────────────────────────────────────
// Real chat backed by /api/jobs/{id}/chat → Gemini 2.5 Flash with the
// banned-jargon system prompt from src/coach_chat.py. History lives in
// component state only (not persisted server-side) so "Clear chat" is a
// pure client action — the server is stateless across turns.
function ChatPanel({ jobId, lang, t }) {
  const [messages, setMessages] = useState([{ role: 'assistant', text: t.chatGreeting }]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const scrollRef = useRef(null);

  // Reset the greeting whenever the user toggles language so the panel
  // doesn't feel stuck in the previous locale. We DO drop conversation
  // history on lang switch — the model would otherwise mix languages.
  useEffect(() => {
    setMessages([{ role: 'assistant', text: t.chatGreeting }]);
    setErr(null);
  }, [lang, jobId]);  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, busy]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput(''); setErr(null);
    const newHist = [...messages, { role: 'user', text: q }];
    setMessages(newHist);
    setBusy(true);
    try {
      // Strip the bootstrap greeting before sending so the model doesn't
      // think the assistant has already been talking — the greeting is
      // pure UX, not a Gemini turn.
      const sendable = newHist.slice(1, -1);  // drop greeting, drop the question (we send it explicitly)
      const { answer } = await apiChat(jobId, q, sendable, lang);
      setMessages((m) => [...m, { role: 'assistant', text: answer }]);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const clear = () => {
    setMessages([{ role: 'assistant', text: t.chatGreeting }]);
    setErr(null);
  };

  return (
    <div className="card" style={{ padding: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                    padding: '14px 16px', borderBottom: `1px solid ${C.border}` }}>
        <div style={{ width: 28, height: 28, borderRadius: 8, background: C.green,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: C.bg, fontWeight: 700, fontSize: 12, fontFamily: 'Rajdhani,sans-serif' }}>
          AI
        </div>
        <span style={{ fontFamily: 'Rajdhani,sans-serif', fontWeight: 700, fontSize: 16,
                       letterSpacing: '0.04em' }}>
          {t.chatTitle}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: C.green, fontWeight: 600,
                       display: 'flex', alignItems: 'center', gap: 6 }}>
          <Pulse size={6} /> LIVE
        </span>
        {messages.length > 1 && (
          <button onClick={clear}
                  style={{ background: 'none', color: C.gray, fontSize: 11,
                           padding: '4px 10px', borderRadius: 16,
                           border: `1px solid ${C.border}` }}>
            {t.chatClear}
          </button>
        )}
      </div>

      <div ref={scrollRef} style={{ padding: 16, maxHeight: 320, overflowY: 'auto',
                                    display: 'flex', flexDirection: 'column', gap: 12 }}>
        {messages.map((m, i) => (
          <ChatBubble key={i} role={m.role} text={m.text} />
        ))}
        {busy && (
          <div style={{ display: 'flex', gap: 10, alignItems: 'flex-end' }}>
            <Avatar role="assistant" />
            <div style={{ background: C.card, border: `1px solid ${C.border}`,
                          borderRadius: '4px 12px 12px 12px',
                          padding: '12px 16px', display: 'flex', gap: 4 }}>
              {[0, 1, 2].map((i) => (
                <span key={i} style={{ width: 7, height: 7, borderRadius: '50%',
                                       background: C.gray,
                                       animation: `pulse 1.2s ${i * 0.2}s infinite`,
                                       display: 'inline-block' }} />
              ))}
            </div>
          </div>
        )}
        {err && (
          <div style={{ color: C.red, fontSize: 12, fontFamily: 'monospace',
                        padding: '8px 12px', background: `${C.red}11`,
                        borderRadius: 8, border: `1px solid ${C.red}33` }}>
            {t.chatError} · {err}
          </div>
        )}
      </div>

      <div style={{ padding: '12px 16px', borderTop: `1px solid ${C.border}`,
                    display: 'flex', gap: 8 }}>
        <input className="input" value={input} onChange={(e) => setInput(e.target.value)}
               onKeyDown={onKey} placeholder={t.chatPlaceholder}
               disabled={busy}
               style={{ flex: 1 }} />
        <button className="btn-primary" disabled={busy || !input.trim()}
                style={{ padding: '10px 18px', fontSize: 13 }}
                onClick={send}>
          {busy ? t.chatThinking : t.chatSend}
        </button>
      </div>
    </div>
  );
}

function Avatar({ role }) {
  const isAi = role === 'assistant';
  return (
    <div style={{ width: 28, height: 28, borderRadius: '50%',
                  background: isAi ? C.green : C.blue,
                  color: isAi ? C.bg : '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 11, fontWeight: 700, flexShrink: 0,
                  fontFamily: 'Rajdhani,sans-serif' }}>
      {isAi ? 'AI' : 'You'}
    </div>
  );
}

function ChatBubble({ role, text }) {
  const isUser = role === 'user';
  return (
    <div className="fade-in" style={{ display: 'flex', gap: 10,
                                       flexDirection: isUser ? 'row-reverse' : 'row',
                                       alignItems: 'flex-end' }}>
      <Avatar role={role} />
      <div style={{ maxWidth: '78%',
                    background: isUser ? `${C.blue}22` : C.card,
                    border: `1px solid ${isUser ? `${C.blue}44` : C.border}`,
                    borderRadius: isUser ? '12px 4px 12px 12px' : '4px 12px 12px 12px',
                    padding: '10px 14px', fontSize: 13, lineHeight: 1.5,
                    color: C.grayLight, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
        {text}
      </div>
    </div>
  );
}

// ── App ─────────────────────────────────────────────────────────────────────
function App() {
  const [screen, setScreen] = useState({ name: 'dashboard' });
  const [lang, setLang] = useState(() => (navigator.language || '').toLowerCase().startsWith('th') ? 'th' : 'en');

  let body;
  if (screen.name === 'onboarding') {
    body = <Onboarding setScreen={setScreen} lang={lang} preselected={screen.preselected} />;
  } else if (screen.name === 'progress') {
    body = (
      <div style={{ flex: 1, padding: '40px 24px', overflowY: 'auto' }}>
        <div style={{ maxWidth: 600, margin: '0 auto' }}>
          <button onClick={() => setScreen({ name: 'dashboard' })} className="btn-secondary" style={{ marginBottom: 20, fontSize: 13 }}>{T[lang].backDash}</button>
          <ProgressScreen jobId={screen.jobId} lang={lang} onDone={() => setScreen({ name: 'analysis', jobId: screen.jobId })} />
        </div>
      </div>
    );
  } else if (screen.name === 'analysis') {
    body = <Analysis setScreen={setScreen} lang={lang} jobId={screen.jobId} />;
  } else {
    body = <Dashboard setScreen={setScreen} lang={lang} />;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Nav screen={screen.name} setScreen={setScreen} lang={lang} setLang={setLang} />
      {body}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
