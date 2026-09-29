import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { Camera, Ruler, ArrowRight, CheckCircle, AlertTriangle, XCircle, Settings, Upload, RotateCcw } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000/api';
const SERVER_ROOT = 'http://localhost:8000';

// How many clicks each measurement needs, past the image being loaded:
// 4 A4 sheet corners, always manual (see measure.py's module docstring --
// SAM's mask boundary on a small sheet isn't precise enough, and that
// imprecision would scale every downstream number). Then:
//   item / floor: 1 more click, SAM segments it automatically.
//   opening:      4 more manual taps (left_bottom, right_bottom, top, floor_below_top)
//                 -- an opening is empty space, nothing for SAM to click "on".
function extraClicksNeeded(kind) {
  return kind === 'opening' ? 4 : 1;
}

function instructionFor(kind, clickCount) {
  if (clickCount < 4) {
    return `Click A4 sheet corner ${clickCount + 1} of 4 (any order, going around the sheet)`;
  }
  const extra = clickCount - 4;
  if (kind === 'opening') {
    const labels = ['left-bottom of the opening', 'right-bottom of the opening', 'top of the opening', 'the floor directly below that top point'];
    return extra < 4 ? `Click ${labels[extra]}` : 'Done -- ready to measure';
  }
  const noun = kind === 'floor' ? 'the floor patch' : 'the item';
  return extra < 1 ? `Click once on ${noun}` : 'Done -- ready to measure';
}

/** Canvas-based tap picker: click the A4 sheet's 4 corners, then either 1
 * more click (item/floor -- shows a live SAM mask preview) or 4 more manual
 * taps (opening). Click coordinates are reported in the ORIGINAL image's
 * own pixel space, matching every backend endpoint's contract. */
function TapPicker({ imagePath, kind, onReady }) {
  const canvasRef = useRef(null);
  const baseImgRef = useRef(null);
  const previewImgRef = useRef(null);
  const [clicks, setClicks] = useState([]);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [mousePos, setMousePos] = useState(null);

  const totalNeeded = 4 + extraClicksNeeded(kind);

  useEffect(() => {
    setClicks([]);
    previewImgRef.current = null;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `${SERVER_ROOT}/${imagePath}`;
    img.onload = () => {
      baseImgRef.current = img;
      const canvas = canvasRef.current;
      if (canvas && baseImgRef.current) {
        const displayW = Math.min(img.width, 700);
        canvas.width = displayW;
        canvas.height = (img.height / img.width) * displayW;
        redraw([]);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imagePath]);

  useEffect(() => {
    redraw(clicks);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clicks]);

  function redraw(currentClicks = clicks) {
    const canvas = canvasRef.current;
    if (!canvas || !baseImgRef.current) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(baseImgRef.current, 0, 0, canvas.width, canvas.height);
    if (previewImgRef.current) {
      ctx.globalAlpha = 1.0;
      ctx.drawImage(previewImgRef.current, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }
    const scale = canvas.width / baseImgRef.current.width;
    currentClicks.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x * scale, y * scale, 7, 0, 2 * Math.PI);
      ctx.fillStyle = i < 4 ? '#4ade80' : '#60a5fa';
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(String(i + 1), x * scale + 10, y * scale - 10);
    });
    if (mousePos) {
      const [mx, my] = mousePos;
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx - 20, my);
      ctx.lineTo(mx + 20, my);
      ctx.moveTo(mx, my - 20);
      ctx.lineTo(mx, my + 20);
      ctx.stroke();
    }
  }

  function handleMouseMove(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    setMousePos([e.clientX - rect.left, e.clientY - rect.top]);
    redraw(clicks);
  }

  function handleMouseLeave() {
    setMousePos(null);
    redraw(clicks);
  }

  async function handleClick(e) {
    if (clicks.length >= totalNeeded) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scale = baseImgRef.current.width / canvas.width;
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    const x = Math.round(canvasX * scale);
    const y = Math.round(canvasY * scale);
    const next = [...clicks, [x, y]];
    setClicks(next);
    redraw(next);

    const isLastNonOpeningClick = kind !== 'opening' && next.length === totalNeeded;
    if (isLastNonOpeningClick) {
      setPreviewLoading(true);
      try {
        const res = await axios.post(`${API_URL}/segment_preview`, { image_path: imagePath, click: [x, y] });
        if (res.data.ok) {
          const previewImg = new Image();
          previewImg.onload = () => {
            previewImgRef.current = previewImg;
            redraw(next);
          };
          previewImg.src = `data:image/png;base64,${res.data.preview_png_base64}`;
        }
      } catch (err) {
        console.error(err);
      }
      setPreviewLoading(false);
    }

    if (next.length === totalNeeded) {
      const sheet_px = next.slice(0, 4);
      const extra = next.slice(4);
      onReady({ sheet_px, extra: kind === 'opening' ? extra : extra[0] });
    }
  }

  function reset() {
    previewImgRef.current = null;
    setClicks([]);
    onReady(null);
    redraw([]);
  }

  return (
    <div>
      <p className="detail-text" style={{ marginBottom: '0.5rem' }}>
        {instructionFor(kind, clicks.length)} {previewLoading && '(segmenting...)'}
      </p>
      <p className="detail-text" style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
        Clicked: {clicks.length}/{totalNeeded}
        {mousePos && ` — hover: (${Math.round(mousePos[0])}, ${Math.round(mousePos[1])})`}
      </p>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{ width: '100%', borderRadius: '8px', cursor: 'crosshair', border: '1px solid var(--surface-border)', display: 'block' }}
      />
      {clicks.length > 0 && (
        <button className="btn-secondary" onClick={reset} style={{ marginTop: '0.5rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <RotateCcw size={14} /> Reset points
        </button>
      )}
    </div>
  );
}

function PhotoPicker({ label, imagePath, onChange, demoPhotos }) {
  const fileInputRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    const res = await axios.post(`${API_URL}/upload`, form);
    if (res.data.ok) onChange(res.data.image_path);
  }

  return (
    <div className="form-group">
      <label className="label">{label}</label>
      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
        <select className="input-field" value={imagePath || ''} onChange={(e) => onChange(e.target.value || null)}>
          <option value="">-- choose a demo photo --</option>
          {demoPhotos.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
        <button className="btn-secondary" onClick={() => fileInputRef.current.click()} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Upload size={16} /> Upload new photo
        </button>
        <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleUpload} />
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('measure');
  const [demoPhotos, setDemoPhotos] = useState([]);

  const [itemImage, setItemImage] = useState(null);
  const [itemPicks, setItemPicks] = useState(null);
  const [itemData, setItemData] = useState(null);
  const [loadingItem, setLoadingItem] = useState(false);

  const [spaceImage, setSpaceImage] = useState(null);
  const [spaceKind, setSpaceKind] = useState('opening');
  const [spacePicks, setSpacePicks] = useState(null);
  const [spaceData, setSpaceData] = useState(null);
  const [loadingSpace, setLoadingSpace] = useState(false);

  const [fitMode, setFitMode] = useState('passage');
  const [fitResult, setFitResult] = useState(null);
  const [loadingFit, setLoadingFit] = useState(false);

  useEffect(() => {
    axios.get(`${API_URL}/demo_photos`).then((res) => setDemoPhotos(res.data.paths)).catch(() => {});
  }, []);

  const measureItem = async () => {
    if (!itemPicks) return alert('Click the sheet corners and the item first!');
    setLoadingItem(true);
    try {
      const res = await axios.post(`${API_URL}/measure_item`, {
        image_path: itemImage,
        sheet_px: itemPicks.sheet_px,
        item_click: itemPicks.extra,
      });
      if (!res.data.ok) alert(`Measurement failed: ${res.data.error}`);
      setItemData(res.data);
    } catch (e) {
      console.error(e);
      alert('Error measuring item');
    }
    setLoadingItem(false);
  };

  const measureSpace = async () => {
    if (!spacePicks) return alert('Click the sheet corners and the space points first!');
    setLoadingSpace(true);
    try {
      const res = await axios.post(`${API_URL}/measure_space`, {
        image_path: spaceImage,
        sheet_px: spacePicks.sheet_px,
        kind: spaceKind,
        space_input: spacePicks.extra,
      });
      if (!res.data.ok) alert(`Measurement failed: ${res.data.error}`);
      setSpaceData(res.data);
    } catch (e) {
      console.error(e);
      alert('Error measuring space');
    }
    setLoadingSpace(false);
  };

  const checkFit = async () => {
    if (!itemData?.ok || !spaceData?.ok) {
      alert('Please measure a valid item and space first!');
      return;
    }
    setLoadingFit(true);
    try {
      const res = await axios.post(`${API_URL}/fit`, { item: itemData, space: spaceData, mode: fitMode });
      setFitResult(res.data);
    } catch (e) {
      console.error(e);
      alert('Error checking fit');
    }
    setLoadingFit(false);
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>Snap-to-Fit</h1>
        <p className="subtitle">Will it fit? Click a sheet, click the item, find out.</p>
      </header>

      <div className="tabs">
        <div className={`tab ${activeTab === 'measure' ? 'active' : ''}`} onClick={() => setActiveTab('measure')}>1. Measure</div>
        <div className={`tab ${activeTab === 'fit' ? 'active' : ''}`} onClick={() => setActiveTab('fit')}>2. Check Fit</div>
      </div>

      {activeTab === 'measure' && (
        <div className="card glass-panel">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}>
            <Camera /> Item Capture
          </h2>
          <PhotoPicker label="Item photo" imagePath={itemImage} demoPhotos={demoPhotos}
            onChange={(p) => { setItemImage(p); setItemPicks(null); setItemData(null); }} />
          {itemImage && (
            <TapPicker imagePath={itemImage} kind="item" onReady={setItemPicks} />
          )}
          <button className="btn-primary" onClick={measureItem} disabled={loadingItem || !itemPicks} style={{ marginTop: '1rem' }}>
            {loadingItem ? 'Measuring...' : 'Measure Item'} <Ruler size={18} />
          </button>

          {itemData?.ok && (
            <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
              <p>L: {itemData.L_cm} cm</p>
              <p>W: {itemData.W_cm} cm</p>
              <p>H: {itemData.H_cm} cm</p>
              <p className="detail-text">sigma: {(itemData.sigma_pct * 100).toFixed(1)}%</p>
              {itemData.debug_img && <img className="debug-image" src={`${SERVER_ROOT}/${itemData.debug_img}`} alt="item debug overlay" />}
            </div>
          )}

          <hr style={{ borderColor: 'var(--surface-border)', margin: '2rem 0' }} />

          <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1rem' }}>
            <Camera /> Space Capture
          </h2>
          <div className="form-group">
            <label className="label">Space Kind</label>
            <select className="input-field" value={spaceKind} onChange={(e) => { setSpaceKind(e.target.value); setSpacePicks(null); setSpaceData(null); }}>
              <option value="opening">Opening / Doorway (4 manual taps)</option>
              <option value="floor">Floor / Corner (1 click)</option>
            </select>
          </div>
          <PhotoPicker label="Space photo" imagePath={spaceImage} demoPhotos={demoPhotos}
            onChange={(p) => { setSpaceImage(p); setSpacePicks(null); setSpaceData(null); }} />
          {spaceImage && (
            <TapPicker imagePath={spaceImage} kind={spaceKind} onReady={setSpacePicks} />
          )}
          <button className="btn-primary" onClick={measureSpace} disabled={loadingSpace || !spacePicks} style={{ marginTop: '1rem' }}>
            {loadingSpace ? 'Measuring...' : 'Measure Space'} <Ruler size={18} />
          </button>

          {spaceData?.ok && (
            <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '8px' }}>
              <p>Width: {spaceData.width_cm} cm</p>
              {spaceData.depth_cm != null && <p>Depth: {spaceData.depth_cm} cm</p>}
              {spaceData.height_cm != null && <p>Height: {spaceData.height_cm} cm</p>}
              <p className="detail-text">sigma: {(spaceData.sigma_pct * 100).toFixed(1)}%</p>
              {spaceData.debug_img && <img className="debug-image" src={`${SERVER_ROOT}/${spaceData.debug_img}`} alt="space debug overlay" />}
            </div>
          )}
        </div>
      )}

      {activeTab === 'fit' && (
        <div className="card glass-panel">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.5rem' }}>
            <Settings /> Fit Assessment
          </h2>

          <div className="form-group">
            <label className="label">Assessment Mode</label>
            <select className="input-field" value={fitMode} onChange={(e) => setFitMode(e.target.value)}>
              <option value="passage">Passage (through opening)</option>
              <option value="placement">Placement (on floor)</option>
            </select>
          </div>

          <button className="btn-primary" onClick={checkFit} disabled={loadingFit} style={{ width: '100%' }}>
            {loadingFit ? 'Calculating...' : 'Run Fit Analysis'} <ArrowRight size={18} />
          </button>

          {fitResult && (
            <div className="result-card fadeIn">
              {fitResult.verdict === 'fits' && <CheckCircle color="var(--success)" size={48} />}
              {fitResult.verdict === 'tight' && <AlertTriangle color="var(--warning)" size={48} />}
              {fitResult.verdict === "doesn't fit" && <XCircle color="var(--danger)" size={48} />}

              <div className={`verdict ${fitResult.verdict === "doesn't fit" ? 'no-fit' : fitResult.verdict}`}>
                {fitResult.verdict}
              </div>

              <div className="margin-text">
                Margin: <strong>{fitResult.margin_cm.toFixed(1)} cm</strong>
              </div>

              <div className="detail-text">{fitResult.detail}</div>

              <div className={`confidence-badge tier-${fitResult.tier}`}>
                {fitResult.tier} Confidence ({(fitResult.prob_fit * 100).toFixed(0)}%)
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
