import { useRef, useEffect, useState, useCallback } from 'react';
import { PoseLandmarker, FaceLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { parse as otParse, Path as OTPath } from 'opentype.js';
import { GIFEncoder, quantize, applyPalette } from 'gifenc';
import Matter from 'matter-js';
import './FingerDrawApp.css';

const { Engine, Bodies, Body, Composite, Constraint } = Matter;

// Body landmark pairs for collision segments + per-segment thickness multiplier
const BODY_SEGMENTS = [
  { pair: [11, 12], thickness: 1.0 },  // shoulders
  { pair: [11, 13], thickness: 0.55 }, // left upper arm
  { pair: [13, 15], thickness: 0.45 }, // left forearm
  { pair: [12, 14], thickness: 0.55 }, // right upper arm
  { pair: [14, 16], thickness: 0.45 }, // right forearm
  { pair: [11, 23], thickness: 0.9 },  // left torso
  { pair: [12, 24], thickness: 0.9 },  // right torso
  { pair: [23, 24], thickness: 0.85 }, // hips
  { pair: [23, 25], thickness: 0.65 }, // left thigh
  { pair: [25, 27], thickness: 0.5 },  // left shin
  { pair: [24, 26], thickness: 0.65 }, // right thigh
  { pair: [26, 28], thickness: 0.5 },  // right shin
];

const CIRCLES_PER_SEGMENT = 16;
const BASE_CIRCLE_RADIUS = 22;
const HEAD_CIRCLES_COUNT = 9;
const REF_SHOULDER_WIDTH = 200;
const MAX_LINES = 50;

// Color palette
const COLOR_PALETTE = ['#C8FF00', '#FF3366', '#00CCFF', '#FF6600', '#FFFFFF'];

// Gaze tracking (iris landmarks from 478-point face mesh)
const LEFT_IRIS_CENTER = 468;
const RIGHT_IRIS_CENTER = 473;
const LEFT_EYE_OUTER = 33;
const RIGHT_EYE_OUTER = 263;
const GAZE_AMPLIFY = 1.8;
const GAZE_SMOOTH = 0.15;

// Trace mode (HandLandmarker)
const HAND_THUMB_TIP = 4;
const HAND_INDEX_TIP = 8;

// Mesh mode
const MESH_COLS = 12;
const MESH_ROWS = 8;
const MESH_STIFFNESS = 0.12;
const MESH_DAMPING = 0.06;
const MESH_PIN_STIFFNESS = 0.3;
const MESH_PAD = 0.5; // padding as fraction of fontSize
const MESH_COLLISION = { category: 0x0004, mask: 0 };
const WIND_FORCE = 0.0003;

// Capture / export
const ASPECT_DIMS = {
  '9:16': { w: 1080, h: 1920 },
  '4:5': { w: 1080, h: 1350 },
};
const VIDEO_FILTER = 'grayscale(100%) contrast(1.3) brightness(0.85)';
const VIDEO_SLIM = 0.94;

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function pickRecordMime() {
  const candidates = [
    'video/mp4;codecs=h264', 'video/mp4',
    'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm',
  ];
  for (const m of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(m)) return m;
  }
  return 'video/webm';
}

// Barycentric coordinates for point in triangle; returns null if outside
function bary(px, py, ax, ay, bx, by, cx, cy) {
  const d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  if (Math.abs(d) < 1e-10) return null;
  const w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / d;
  const w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / d;
  const w2 = 1 - w0 - w1;
  if (w0 >= -0.02 && w1 >= -0.02 && w2 >= -0.02) return [w0, w1, w2];
  return null;
}

// Pre-compute barycentric mapping for all path command points
function buildPathMap(commands, offX, offY, restVerts, tris) {
  return commands.map(cmd => {
    const raw = [];
    if (cmd.type === 'M' || cmd.type === 'L') raw.push({ x: cmd.x + offX, y: cmd.y + offY });
    else if (cmd.type === 'Q') {
      raw.push({ x: cmd.x1 + offX, y: cmd.y1 + offY });
      raw.push({ x: cmd.x + offX, y: cmd.y + offY });
    } else if (cmd.type === 'C') {
      raw.push({ x: cmd.x1 + offX, y: cmd.y1 + offY });
      raw.push({ x: cmd.x2 + offX, y: cmd.y2 + offY });
      raw.push({ x: cmd.x + offX, y: cmd.y + offY });
    }
    const mapped = raw.map(pt => {
      for (let i = 0; i < tris.length; i++) {
        const [a, b, c] = tris[i];
        const w = bary(pt.x, pt.y, restVerts[a].x, restVerts[a].y,
          restVerts[b].x, restVerts[b].y, restVerts[c].x, restVerts[c].y);
        if (w) return { tri: i, w };
      }
      return { tri: -1, orig: pt }; // outside mesh fallback
    });
    return { type: cmd.type, pts: mapped };
  });
}

// Transform a mapped point through deformed mesh
function deformPt(mp, tris, bodies) {
  if (mp.tri < 0) return mp.orig;
  const [a, b, c] = tris[mp.tri];
  return {
    x: mp.w[0] * bodies[a].position.x + mp.w[1] * bodies[b].position.x + mp.w[2] * bodies[c].position.x,
    y: mp.w[0] * bodies[a].position.y + mp.w[1] * bodies[b].position.y + mp.w[2] * bodies[c].position.y,
  };
}
const TRACE_MIN_DIST = 4; // min px between trail points
const TRACE_PINCH_CLOSE = 40; // px — thumb+index touching
const TRACE_PINCH_OPEN = 70;  // px — thumb+index separated

// Build default text from current Tokyo date/time
function buildDefaultText() {
  const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const now = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Tokyo' }));
  const h = now.getHours();
  const h12 = h % 12 || 12;
  const min = String(now.getMinutes()).padStart(2, '0');
  const ampm = h >= 12 ? 'PM' : 'AM';
  return `${days[now.getDay()]}\n${months[now.getMonth()]} ${now.getDate()}\n${now.getFullYear()}\n${h12}:${min} ${ampm}\nTOKYO`;
}

const DEFAULT_TEXT = buildDefaultText();

let nextLineId = 0;

export default function FingerDrawApp() {
  const videoRef = useRef(null);
  const charCanvasRef = useRef(null);
  const cameraWrapRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const faceLandmarkerRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const charColorRef = useRef(0);
  const gazePositionRef = useRef(null);
  const gazeLineRef = useRef(null);
  const gazeDropIndexRef = useRef(0);
  const faceCanvasRef = useRef(null);
  const traceRef = useRef({ state: 'idle', points: [], cumDist: [0] }); // idle → armed → tracing
  const traceDebugRef = useRef(null); // debug landmark positions
  const otFontRef = useRef(null); // opentype.js font
  const meshRef = useRef(null);   // mesh mode data
  const grainCanvasRef = useRef(null); // grain texture for compositing
  const recordCanvasRef = useRef(null);
  const recordCtxRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordingRef = useRef(false);
  const windRef = useRef(false);
  const aspectRef = useRef('9:16');
  const drawCompositeRef = useRef(null);
  const initMeshRef = useRef(null);
  const gifStateRef = useRef(null);
  const circleScaleRef = useRef(1);
  const engineRef = useRef(null);
  const segmentCirclesRef = useRef([]);
  const headCirclesRef = useRef([]);
  const lineBodiesRef = useRef([]);
  const wallsRef = useRef([]);
  const updateWallsRef = useRef(null);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const containerSizeRef = useRef({ w: 0, h: 0 });
  const grainRef = useRef(null);
  const prevLandmarksRef = useRef(null);
  const measureCtxRef = useRef(null);
  const dropIndexRef = useRef(0);

  const fileInputRef = useRef(null);
  const videoFileUrlRef = useRef(null);

  const [modelLoading, setModelLoading] = useState(true);
  const [inputText, setInputText] = useState(DEFAULT_TEXT);
  const [fontSize, setFontSize] = useState(48);
  const [tracking, setTracking] = useState(0);
  const [facingMode, setFacingMode] = useState('user');
  const [sourceMode, setSourceMode] = useState('camera'); // 'camera' | 'video'
  const [displayMode, setDisplayMode] = useState('fall'); // 'fall' | 'gaze'
  const [sweep, setSweep] = useState(true);
  const [charColor, setCharColor] = useState(0);
  const [aspect, setAspect] = useState('9:16');
  const [recording, setRecording] = useState(false);
  const [windOn, setWindOn] = useState(false);
  const [gifRecording, setGifRecording] = useState(false);
  const inputTextRef = useRef(DEFAULT_TEXT);
  const fontSizeRef = useRef(48);
  const trackingRef = useRef(0);
  const facingModeRef = useRef('user');
  const sourceModeRef = useRef('camera');
  const displayModeRef = useRef('fall');
  const sweepRef = useRef(true);

  useEffect(() => {
    inputTextRef.current = inputText;
    dropIndexRef.current = 0;
  }, [inputText]);

  useEffect(() => { fontSizeRef.current = fontSize; }, [fontSize]);
  useEffect(() => { trackingRef.current = tracking; }, [tracking]);
  useEffect(() => { facingModeRef.current = facingMode; }, [facingMode]);
  useEffect(() => { sourceModeRef.current = sourceMode; }, [sourceMode]);
  useEffect(() => { displayModeRef.current = displayMode; }, [displayMode]);
  useEffect(() => {
    sweepRef.current = sweep;
    updateWallsRef.current?.();
  }, [sweep]);
  useEffect(() => { charColorRef.current = charColor; }, [charColor]);
  useEffect(() => {
    aspectRef.current = aspect;
    // Rebuild mesh to fit new frame after layout settles
    if (displayModeRef.current === 'mesh') {
      const id = setTimeout(() => initMeshRef.current?.(), 120);
      return () => clearTimeout(id);
    }
  }, [aspect]);
  useEffect(() => { windRef.current = windOn; }, [windOn]);

  // Load font with opentype.js
  useEffect(() => {
    fetch('/fonts/OTRGrotesk-Regular.otf')
      .then(r => r.arrayBuffer())
      .then(buf => { otFontRef.current = otParse(buf); })
      .catch(e => console.error('opentype parse error:', e));
  }, []);

  // Create offscreen canvases for text measurement + face detection zoom-out
  useEffect(() => {
    const c = document.createElement('canvas');
    measureCtxRef.current = c.getContext('2d');
    const fc = document.createElement('canvas');
    fc.width = 320; fc.height = 240;
    faceCanvasRef.current = fc;
  }, []);

  // Ensure OTR Grotesk is loaded for canvas rendering
  useEffect(() => { document.fonts.load("72px 'OTR Grotesk'"); }, []);

  // Generate grain noise texture
  useEffect(() => {
    const size = 150;
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    const img = ctx.createImageData(size, size);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = Math.random() * 255;
      d[i] = v; d[i + 1] = v; d[i + 2] = v;
      d[i + 3] = 30;
    }
    ctx.putImageData(img, 0, 0);
    grainCanvasRef.current = c;
    if (grainRef.current) {
      grainRef.current.style.backgroundImage = `url(${c.toDataURL()})`;
    }
  }, []);

  // Measure line width with tracking
  const measureLineWidth = useCallback((text, fs, trk) => {
    const ctx = measureCtxRef.current;
    if (!ctx) return fs * 0.6 * text.length;
    ctx.font = `${fs}px 'OTR Grotesk', sans-serif`;
    let width = 0;
    for (let i = 0; i < text.length; i++) {
      width += ctx.measureText(text[i]).width;
      if (i < text.length - 1) width += trk;
    }
    return width;
  }, []);

  // Convert normalized pose landmark coords to container CSS coords
  const landmarkToCss = useCallback((lmX, lmY) => {
    const container = cameraWrapRef.current;
    const video = videoRef.current;
    if (!container || !video || !video.videoWidth) return { x: 0, y: 0 };
    const cRect = container.getBoundingClientRect();
    const cw = cRect.width, ch = cRect.height;
    const vidW = video.videoWidth, vidH = video.videoHeight;
    const cAspect = cw / ch, vAspect = vidW / vidH;
    let dw, dh, ox, oy;
    if (cAspect > vAspect) {
      dw = cw; dh = cw / vAspect; ox = 0; oy = (ch - dh) / 2;
    } else {
      dh = ch; dw = ch * vAspect; ox = (cw - dw) / 2; oy = 0;
    }
    const shouldMirror = sourceModeRef.current === 'camera' && facingModeRef.current === 'user';
    const adjustedX = shouldMirror ? 1 - lmX : lmX;
    return { x: adjustedX * dw + ox, y: lmY * dh + oy };
  }, []);

  // Init PoseLandmarker + FaceLandmarker + HandLandmarker in parallel
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        if (cancelled) return;
        const [pl, fl, hl] = await Promise.all([
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
              delegate: 'GPU',
            },
            numPoses: 1,
            runningMode: 'VIDEO',
          }),
          FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
              delegate: 'GPU',
            },
            numFaces: 1,
            runningMode: 'VIDEO',
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath:
                'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
              delegate: 'GPU',
            },
            numHands: 2,
            runningMode: 'VIDEO',
          }),
        ]);
        if (cancelled) { pl.close(); fl.close(); hl.close(); return; }
        poseLandmarkerRef.current = pl;
        faceLandmarkerRef.current = fl;
        handLandmarkerRef.current = hl;
        setModelLoading(false);
      } catch (e) {
        console.error('Model init error:', e);
      }
    })();
    return () => {
      cancelled = true;
      poseLandmarkerRef.current?.close();
      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
    };
  }, []);

  // Start camera (restarts when facingMode changes, only in camera mode)
  useEffect(() => {
    if (sourceMode !== 'camera') return;
    let cancelled = false;
    (async () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
        if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.loop = false;
        }
      } catch (e) {
        console.error('Camera error:', e);
      }
    })();
    return () => { cancelled = true; streamRef.current?.getTracks().forEach(t => t.stop()); };
  }, [facingMode, sourceMode]);

  // Handle video file source
  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Stop camera
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    // Revoke previous file URL
    if (videoFileUrlRef.current) URL.revokeObjectURL(videoFileUrlRef.current);
    const url = URL.createObjectURL(file);
    videoFileUrlRef.current = url;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
      videoRef.current.loop = true;
      videoRef.current.play();
    }
    prevLandmarksRef.current = null;
    setSourceMode('video');
    // Reset file input so the same file can be re-selected
    e.target.value = '';
  }, []);

  // Switch back to camera from video
  const switchToCamera = useCallback(() => {
    if (videoFileUrlRef.current) {
      URL.revokeObjectURL(videoFileUrlRef.current);
      videoFileUrlRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.src = '';
      videoRef.current.srcObject = null;
    }
    prevLandmarksRef.current = null;
    setSourceMode('camera');
  }, []);

  // Init matter.js engine, walls, body collision circles
  useEffect(() => {
    const engine = Engine.create({ gravity: { x: 0, y: 1 } });
    engineRef.current = engine;

    const updateWalls = () => {
      const container = cameraWrapRef.current;
      if (!container) return;
      const { width: cw, height: ch } = container.getBoundingClientRect();
      containerSizeRef.current = { w: cw, h: ch };
      if (wallsRef.current.length) Composite.remove(engine.world, wallsRef.current);
      const t = 60;
      const walls = [
        Bodies.rectangle(cw / 2, ch + t / 2, cw * 2, t, { isStatic: true }),
      ];
      if (!sweepRef.current) {
        walls.push(
          Bodies.rectangle(-t / 2, ch / 2, t, ch * 2, { isStatic: true }),
          Bodies.rectangle(cw + t / 2, ch / 2, t, ch * 2, { isStatic: true }),
        );
      }
      wallsRef.current = walls;
      Composite.add(engine.world, wallsRef.current);
    };

    updateWallsRef.current = updateWalls;
    updateWalls();
    const ro = new ResizeObserver(updateWalls);
    if (cameraWrapRef.current) ro.observe(cameraWrapRef.current);

    const circles = [];
    for (let s = 0; s < BODY_SEGMENTS.length; s++) {
      const r = BASE_CIRCLE_RADIUS * BODY_SEGMENTS[s].thickness;
      for (let i = 0; i < CIRCLES_PER_SEGMENT; i++) {
        const c = Bodies.circle(-1000, -1000, r, {
          isStatic: true, friction: 0.3, restitution: 0.4,
        });
        Composite.add(engine.world, c);
        circles.push(c);
      }
    }
    segmentCirclesRef.current = circles;

    const headCircles = [];
    for (let i = 0; i < HEAD_CIRCLES_COUNT; i++) {
      const c = Bodies.circle(-1000, -1000, BASE_CIRCLE_RADIUS, {
        isStatic: true, friction: 0.3, restitution: 0.4,
      });
      Composite.add(engine.world, c);
      headCircles.push(c);
    }
    headCirclesRef.current = headCircles;

    return () => { ro.disconnect(); Engine.clear(engine); engineRef.current = null; };
  }, []);

  // Update body collision from pose landmarks
  const updateBodyFromPose = useCallback((landmarks) => {
    const circles = segmentCirclesRef.current;
    const headCircles = headCirclesRef.current;
    if (!circles.length || !headCircles.length) return;

    // Compute shoulder width for dynamic scaling
    const lShoulder = landmarks[11];
    const rShoulder = landmarks[12];
    const hasShoulders = lShoulder && rShoulder &&
      (lShoulder.visibility ?? 0) > 0.3 && (rShoulder.visibility ?? 0) > 0.3;
    let shoulderWidth = REF_SHOULDER_WIDTH;
    if (hasShoulders) {
      const pLS = landmarkToCss(lShoulder.x, lShoulder.y);
      const pRS = landmarkToCss(rShoulder.x, rShoulder.y);
      shoulderWidth = Math.hypot(pRS.x - pLS.x, pRS.y - pLS.y);
    }
    const bodyScale = Math.max(shoulderWidth / REF_SHOULDER_WIDTH, 0.3);

    // Rescale all collision circles when body scale changes significantly
    const prevScale = circleScaleRef.current;
    const scaleFactor = bodyScale / prevScale;
    if (Math.abs(scaleFactor - 1) > 0.03) {
      for (const c of circles) Body.scale(c, scaleFactor, scaleFactor);
      for (const c of headCircles) Body.scale(c, scaleFactor, scaleFactor);
      circleScaleRef.current = bodyScale;
    }

    for (let s = 0; s < BODY_SEGMENTS.length; s++) {
      const { pair: [a, b] } = BODY_SEGMENTS[s];
      const lmA = landmarks[a];
      const lmB = landmarks[b];
      const baseIdx = s * CIRCLES_PER_SEGMENT;

      if (!lmA || !lmB || (lmA.visibility ?? 0) < 0.3 || (lmB.visibility ?? 0) < 0.3) {
        for (let i = 0; i < CIRCLES_PER_SEGMENT; i++) {
          Body.setPosition(circles[baseIdx + i], { x: -1000, y: -1000 });
        }
        continue;
      }

      const pA = landmarkToCss(lmA.x, lmA.y);
      const pB = landmarkToCss(lmB.x, lmB.y);

      for (let i = 0; i < CIRCLES_PER_SEGMENT; i++) {
        const t = CIRCLES_PER_SEGMENT > 1 ? i / (CIRCLES_PER_SEGMENT - 1) : 0.5;
        const c = circles[baseIdx + i];
        const ox = c.position.x, oy = c.position.y;
        const nx = pA.x + (pB.x - pA.x) * t;
        const ny = pA.y + (pB.y - pA.y) * t;
        Body.setPosition(c, { x: nx, y: ny });
        if (sweepRef.current && ox > -500) Body.setVelocity(c, { x: nx - ox, y: ny - oy });
      }
    }

    // Head — cluster of circles, sized from shoulder width
    const nose = landmarks[0];
    const hasNose = nose && (nose.visibility ?? 0) > 0.3;

    if (hasNose) {
      const pNose = landmarkToCss(nose.x, nose.y);
      const headR = Math.max(shoulderWidth * 0.38, 35);
      const hcx = pNose.x;
      const hcy = pNose.y - headR * 0.65;
      const offsets = [
        [0, 0], [0, -0.75], [0, 0.65],
        [-0.6, -0.2], [0.6, -0.2],
        [-0.4, -0.6], [0.4, -0.6],
        [-0.35, 0.4], [0.35, 0.4],
      ];
      for (let i = 0; i < HEAD_CIRCLES_COUNT; i++) {
        const c = headCircles[i];
        const ox = c.position.x, oy = c.position.y;
        const nx = hcx + offsets[i][0] * headR;
        const ny = hcy + offsets[i][1] * headR;
        Body.setPosition(c, { x: nx, y: ny });
        if (sweepRef.current && ox > -500) Body.setVelocity(c, { x: nx - ox, y: ny - oy });
      }
    } else {
      for (let i = 0; i < HEAD_CIRCLES_COUNT; i++) {
        Body.setPosition(headCircles[i], { x: -1000, y: -1000 });
      }
    }
  }, [landmarkToCss]);

  // Hide body when no pose detected
  const hideBody = useCallback(() => {
    for (const c of segmentCirclesRef.current) Body.setPosition(c, { x: -1000, y: -1000 });
    for (const c of headCirclesRef.current) Body.setPosition(c, { x: -1000, y: -1000 });
  }, []);

  // Drop next line (random x, top-down order)
  const dropLine = useCallback(() => {
    if (!engineRef.current) return;
    const text = inputTextRef.current;
    const lines = text.split('\n').filter(l => l.length > 0);
    if (!lines.length) return;
    const { w: cw } = containerSizeRef.current;
    if (cw === 0) return;

    const idx = dropIndexRef.current % lines.length;
    const line = lines[idx];
    dropIndexRef.current = idx + 1;

    const fs = fontSizeRef.current;
    const trk = trackingRef.current;
    const bodyH = fs * 0.75;
    const lineW = measureLineWidth(line, fs, trk);
    const bodyW = Math.max(lineW, fs * 0.5);

    const margin = bodyW / 2;
    const x = margin + Math.random() * (cw - bodyW);
    const y = -bodyH - Math.random() * 40;

    const body = Bodies.rectangle(x, y, bodyW, bodyH, {
      restitution: 0.25, friction: 0.6, frictionAir: 0.003,
      angle: (Math.random() - 0.5) * 0.15,
    });
    Body.setVelocity(body, { x: (Math.random() - 0.5) * 2, y: 2 + Math.random() * 2 });

    Composite.add(engineRef.current.world, body);
    const color = COLOR_PALETTE[charColorRef.current];
    lineBodiesRef.current.push({ id: ++nextLineId, body, text: line, fontSize: fs, tracking: trk, color });

    while (lineBodiesRef.current.length > MAX_LINES) {
      const oldest = lineBodiesRef.current.shift();
      if (engineRef.current) Composite.remove(engineRef.current.world, oldest.body);
    }
  }, [measureLineWidth]);

  // Cycle to next line in gaze mode (single line, replaces previous)
  const gazeDropLine = useCallback(() => {
    const text = inputTextRef.current;
    const lines = text.split('\n').filter(l => l.length > 0);
    if (!lines.length) return;

    const idx = gazeDropIndexRef.current % lines.length;
    const line = lines[idx];
    gazeDropIndexRef.current = idx + 1;

    const color = COLOR_PALETTE[charColorRef.current];
    gazeLineRef.current = { text: line, fontSize: fontSizeRef.current, tracking: trackingRef.current, color };
  }, []);

  // Main loop: physics + pose detection + canvas rendering
  useEffect(() => {
    if (modelLoading) return;
    let lastTime = performance.now();
    let lastDetectTime = -1;

    function loop(now) {
      const dt = Math.min(now - lastTime, 33);
      lastTime = now;
      const mode = displayModeRef.current;

      // Wind drift in mesh mode
      if (mode === 'mesh' && windRef.current && meshRef.current) {
        const m = meshRef.current;
        const grabbedIdx = new Set(Object.values(m.grabs).map(g => g.idx));
        const t = now * 0.001;
        for (let i = 0; i < m.bodies.length; i++) {
          if (grabbedIdx.has(i)) continue;
          const b = m.bodies[i];
          const phase = i * 0.6;
          const wx = (Math.sin(t * 0.9 + phase) * 0.6 + Math.sin(t * 0.37 + phase * 1.7) * 0.4) * WIND_FORCE;
          const wy = (Math.cos(t * 0.6 + phase * 1.3) * 0.35) * WIND_FORCE;
          Body.applyForce(b, b.position, { x: wx * b.mass, y: wy * b.mass });
        }
      }

      if ((mode === 'fall' || mode === 'mesh') && engineRef.current) Engine.update(engineRef.current, dt);

      const video = videoRef.current;
      const pl = poseLandmarkerRef.current;
      const fl = faceLandmarkerRef.current;
      if (video && pl && video.readyState >= 2 && video.currentTime !== lastDetectTime) {
        lastDetectTime = video.currentTime;
        const res = pl.detectForVideo(video, now);
        if (res.landmarks?.length > 0) {
          const raw = res.landmarks[0];
          const SMOOTH = 0.35;
          if (!prevLandmarksRef.current) {
            prevLandmarksRef.current = raw.map(l => ({ x: l.x, y: l.y, visibility: l.visibility }));
          }
          const smoothed = raw.map((l, i) => {
            const p = prevLandmarksRef.current[i];
            return { ...l, x: p.x + (l.x - p.x) * (1 - SMOOTH), y: p.y + (l.y - p.y) * (1 - SMOOTH) };
          });
          prevLandmarksRef.current = smoothed.map(l => ({ x: l.x, y: l.y, visibility: l.visibility }));
          updateBodyFromPose(smoothed);

          // Hand detection for trace + mesh modes
          if ((mode === 'trace' || mode === 'mesh') && handLandmarkerRef.current) {
            const handRes = handLandmarkerRef.current.detectForVideo(video, now);
            const numHands = handRes.landmarks?.length || 0;

            if (numHands > 0) {
              // Process first hand for trace mode + debug dots
              const h0 = handRes.landmarks[0];
              const thumb0 = landmarkToCss(h0[HAND_THUMB_TIP].x, h0[HAND_THUMB_TIP].y);
              const index0 = landmarkToCss(h0[HAND_INDEX_TIP].x, h0[HAND_INDEX_TIP].y);
              const gap0 = Math.hypot(index0.x - thumb0.x, index0.y - thumb0.y);
              // Collect all finger tips for debug dots
              const debugDots = [];
              for (let hi = 0; hi < numHands; hi++) {
                const h = handRes.landmarks[hi];
                debugDots.push(landmarkToCss(h[HAND_THUMB_TIP].x, h[HAND_THUMB_TIP].y));
                debugDots.push(landmarkToCss(h[HAND_INDEX_TIP].x, h[HAND_INDEX_TIP].y));
              }
              traceDebugRef.current = debugDots;

              // --- Trace mode (first hand only) ---
              if (mode === 'trace') {
                const pinching = gap0 < TRACE_PINCH_CLOSE;
                const tr = traceRef.current;
                if (tr.state === 'idle') { if (pinching) tr.state = 'armed'; }
                else if (tr.state === 'armed') {
                  if (gap0 >= TRACE_PINCH_OPEN) {
                    tr.state = 'tracing';
                    tr.points = [{ x: index0.x, y: index0.y }];
                    tr.cumDist = [0];
                  }
                } else {
                  const last = tr.points[tr.points.length - 1];
                  const d = Math.hypot(index0.x - last.x, index0.y - last.y);
                  if (d > TRACE_MIN_DIST) {
                    tr.points.push({ x: index0.x, y: index0.y });
                    tr.cumDist.push(tr.cumDist[tr.cumDist.length - 1] + d);
                  }
                  if (pinching && tr.points.length > 3) {
                    tr.state = 'armed'; tr.points = []; tr.cumDist = [0];
                  }
                }
              }

              // --- Mesh mode: each hand can grab independently ---
              if (mode === 'mesh' && meshRef.current) {
                const m = meshRef.current;
                const activeHands = new Set();

                for (let hi = 0; hi < numHands; hi++) {
                  const hand = handRes.landmarks[hi];
                  const thumb = landmarkToCss(hand[HAND_THUMB_TIP].x, hand[HAND_THUMB_TIP].y);
                  const idx = landmarkToCss(hand[HAND_INDEX_TIP].x, hand[HAND_INDEX_TIP].y);
                  const gap = Math.hypot(idx.x - thumb.x, idx.y - thumb.y);
                  const pinching = gap < TRACE_PINCH_CLOSE;
                  activeHands.add(hi);

                  if (pinching && !m.grabs?.[hi]) {
                    // Grab nearest vertex
                    let minD = Infinity, minI = 0;
                    for (let i = 0; i < m.bodies.length; i++) {
                      const d2 = Math.hypot(m.bodies[i].position.x - idx.x, m.bodies[i].position.y - idx.y);
                      if (d2 < minD) { minD = d2; minI = i; }
                    }
                    const c = Constraint.create({
                      bodyA: m.bodies[minI], pointB: { x: idx.x, y: idx.y },
                      stiffness: 0.8, damping: 0.2, length: 0,
                    });
                    Composite.add(engineRef.current.world, c);
                    if (!m.grabs) m.grabs = {};
                    m.grabs[hi] = { idx: minI, constraint: c };
                  } else if (pinching && m.grabs?.[hi]) {
                    m.grabs[hi].constraint.pointB = { x: idx.x, y: idx.y };
                  } else if (!pinching && m.grabs?.[hi]) {
                    Composite.remove(engineRef.current.world, m.grabs[hi].constraint);
                    delete m.grabs[hi];
                  }
                }

                // Release grabs for hands no longer detected
                if (m.grabs) {
                  for (const hi of Object.keys(m.grabs)) {
                    if (!activeHands.has(Number(hi))) {
                      Composite.remove(engineRef.current.world, m.grabs[hi].constraint);
                      delete m.grabs[hi];
                    }
                  }
                }
              }
            } else {
              traceDebugRef.current = null;
              // Release all grabs if no hands detected
              if (mode === 'mesh' && meshRef.current?.grabs) {
                for (const g of Object.values(meshRef.current.grabs)) {
                  Composite.remove(engineRef.current.world, g.constraint);
                }
                meshRef.current.grabs = {};
              }
            }
          }
        } else {
          hideBody();
          prevLandmarksRef.current = null;
        }

        // Face detection (zoom-out for close-up faces) → wink → color change
        if (fl) {
          const fc = faceCanvasRef.current;
          const fcCtx = fc.getContext('2d');
          const FACE_SCALE = 0.4;
          const padX = fc.width * (1 - FACE_SCALE) / 2;
          const padY = fc.height * (1 - FACE_SCALE) / 2;
          fcCtx.fillStyle = '#000';
          fcCtx.fillRect(0, 0, fc.width, fc.height);
          fcCtx.drawImage(video, padX, padY, fc.width * FACE_SCALE, fc.height * FACE_SCALE);
          const faceRes = fl.detectForVideo(fc, now);

          // Gaze position: eye-center based, anchored to screen center
          if (displayModeRef.current === 'gaze' && faceRes.faceLandmarks?.length > 0) {
            const fm = faceRes.faceLandmarks[0];
            if (fm.length > RIGHT_IRIS_CENTER) {
              const li = fm[LEFT_IRIS_CENTER];
              const ri = fm[RIGHT_IRIS_CENTER];
              const leOuter = fm[LEFT_EYE_OUTER];
              const reOuter = fm[RIGHT_EYE_OUTER];
              const faceW = Math.hypot(reOuter.x - leOuter.x, reOuter.y - leOuter.y);
              if (faceW > 0.01) {
                const eyeMidX = (leOuter.x + reOuter.x) / 2;
                const eyeMidY = (leOuter.y + reOuter.y) / 2;
                const irisMidX = (li.x + ri.x) / 2;
                const irisMidY = (li.y + ri.y) / 2;
                const normOffX = (irisMidX - eyeMidX) / faceW;
                const normOffY = (irisMidY - eyeMidY) / faceW;
                const { w: cw, h: ch } = containerSizeRef.current;
                const target = {
                  x: cw / 2 + normOffX * cw * GAZE_AMPLIFY,
                  y: ch / 2 + normOffY * ch * GAZE_AMPLIFY,
                };
                const prev = gazePositionRef.current;
                if (!prev) {
                  gazePositionRef.current = target;
                } else {
                  gazePositionRef.current = {
                    x: prev.x + (target.x - prev.x) * GAZE_SMOOTH,
                    y: prev.y + (target.y - prev.y) * GAZE_SMOOTH,
                  };
                }
              }
            }
          }
        }
      }

      // Render lines on canvas
      const canvas = charCanvasRef.current;
      const { w: cw, h: ch } = containerSizeRef.current;
      if (canvas && cw > 0 && ch > 0) {
        const dpr = window.devicePixelRatio || 1;
        const pw = Math.round(cw * dpr);
        const ph = Math.round(ch * dpr);
        if (canvas.width !== pw || canvas.height !== ph) {
          canvas.width = pw; canvas.height = ph;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, pw, ph);
        ctx.save();
        ctx.scale(dpr, dpr);

        if (displayModeRef.current === 'fall') {
          for (const line of lineBodiesRef.current) {
            const { x, y } = line.body.position;
            const angle = line.body.angle;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(angle);
            ctx.font = `${line.fontSize}px 'OTR Grotesk', sans-serif`;
            ctx.fillStyle = line.color;
            ctx.textBaseline = 'middle';
            ctx.textAlign = 'left';

            const chars = line.text;
            const trk = line.tracking;
            let totalW = 0;
            const charWidths = [];
            for (let i = 0; i < chars.length; i++) {
              const w = ctx.measureText(chars[i]).width;
              charWidths.push(w);
              totalW += w;
              if (i < chars.length - 1) totalW += trk;
            }

            let drawX = -totalW / 2;
            for (let i = 0; i < chars.length; i++) {
              ctx.fillText(chars[i], drawX, 0);
              drawX += charWidths[i] + trk;
            }
            ctx.restore();
          }
        } else if (displayModeRef.current === 'gaze' && gazeLineRef.current) {
          const gl = gazeLineRef.current;
          const gaze = gazePositionRef.current || { x: cw / 2, y: ch / 2 };
          ctx.save();
          ctx.translate(gaze.x, gaze.y);
          ctx.font = `${gl.fontSize}px 'OTR Grotesk', sans-serif`;
          ctx.fillStyle = gl.color;
          ctx.textBaseline = 'middle';
          ctx.textAlign = 'left';

          let totalW = 0;
          const charWidths = [];
          for (let i = 0; i < gl.text.length; i++) {
            const w = ctx.measureText(gl.text[i]).width;
            charWidths.push(w);
            totalW += w;
            if (i < gl.text.length - 1) totalW += gl.tracking;
          }
          let drawX = -totalW / 2;
          for (let i = 0; i < gl.text.length; i++) {
            ctx.fillText(gl.text[i], drawX, 0);
            drawX += charWidths[i] + gl.tracking;
          }
          ctx.restore();
        } else if (displayModeRef.current === 'trace') {
          const trail = traceRef.current;
          if (trail.points.length > 1) {
            const text = inputTextRef.current.replace(/\n/g, ' ');
            const fs = fontSizeRef.current;
            const trk = trackingRef.current;
            const color = COLOR_PALETTE[charColorRef.current];
            const totalLen = trail.cumDist[trail.cumDist.length - 1];

            ctx.font = `${fs}px 'OTR Grotesk', sans-serif`;
            ctx.fillStyle = color;
            ctx.textBaseline = 'middle';

            let charDist = 0;
            let segIdx = 0;

            for (let ci = 0; ci < text.length && charDist <= totalLen; ci++) {
              while (segIdx < trail.cumDist.length - 2 && trail.cumDist[segIdx + 1] < charDist) {
                segIdx++;
              }
              if (segIdx >= trail.points.length - 1) break;

              const sd = trail.cumDist[segIdx];
              const segLen = trail.cumDist[segIdx + 1] - sd;
              const t = segLen > 0 ? (charDist - sd) / segLen : 0;
              const p0 = trail.points[segIdx];
              const p1 = trail.points[segIdx + 1];
              const cx = p0.x + (p1.x - p0.x) * t;
              const cy = p0.y + (p1.y - p0.y) * t;
              const angle = Math.atan2(p1.y - p0.y, p1.x - p0.x);

              ctx.save();
              ctx.translate(cx, cy);
              ctx.rotate(angle);
              ctx.fillText(text[ci], 0, 0);
              ctx.restore();

              charDist += ctx.measureText(text[ci]).width + trk;
            }
          }
        } else if (mode === 'mesh' && meshRef.current) {
          const m = meshRef.current;
          const color = COLOR_PALETTE[charColorRef.current];
          ctx.beginPath();
          for (const cmd of m.pathMap) {
            const dp = cmd.pts.map(p => deformPt(p, m.tris, m.bodies));
            switch (cmd.type) {
              case 'M': if (dp[0]) ctx.moveTo(dp[0].x, dp[0].y); break;
              case 'L': if (dp[0]) ctx.lineTo(dp[0].x, dp[0].y); break;
              case 'Q': if (dp[0] && dp[1]) ctx.quadraticCurveTo(dp[0].x, dp[0].y, dp[1].x, dp[1].y); break;
              case 'C': if (dp[0] && dp[1] && dp[2]) ctx.bezierCurveTo(dp[0].x, dp[0].y, dp[1].x, dp[1].y, dp[2].x, dp[2].y); break;
              case 'Z': ctx.closePath(); break;
            }
          }
          ctx.fillStyle = color;
          ctx.fill();
        }

        // Debug: show thumb + index finger dots for all hands
        if ((mode === 'trace' || mode === 'mesh') && traceDebugRef.current) {
          const dots = traceDebugRef.current;
          ctx.fillStyle = '#fff';
          for (let i = 0; i < dots.length; i++) {
            ctx.beginPath();
            ctx.arc(dots[i].x, dots[i].y, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        ctx.restore();
      }

      // Feed the recording canvas
      if (recordingRef.current && recordCtxRef.current && recordCanvasRef.current) {
        drawCompositeRef.current?.(recordCtxRef.current, recordCanvasRef.current.width, recordCanvasRef.current.height);
      }

      animFrameRef.current = requestAnimationFrame(loop);
    }

    animFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [modelLoading, updateBodyFromPose, hideBody, landmarkToCss]);

  // Toggle camera facing mode
  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    prevLandmarksRef.current = null;
  }, []);

  // Destroy mesh bodies
  const destroyMesh = useCallback(() => {
    const m = meshRef.current;
    if (!m || !engineRef.current) { meshRef.current = null; return; }
    for (const c of m.constraints) Composite.remove(engineRef.current.world, c);
    for (const b of m.bodies) Composite.remove(engineRef.current.world, b);
    if (m.grabs) for (const g of Object.values(m.grabs)) Composite.remove(engineRef.current.world, g.constraint);
    meshRef.current = null;
  }, []);

  // Build mesh for mesh mode
  const initMesh = useCallback(() => {
    const font = otFontRef.current;
    const engine = engineRef.current;
    if (!font || !engine) return;

    // Clean previous mesh inline (destroyMesh may not be in scope yet on first call)
    const prev = meshRef.current;
    if (prev) {
      for (const c of prev.constraints) Composite.remove(engine.world, c);
      for (const b of prev.bodies) Composite.remove(engine.world, b);
      if (prev.grabbed) Composite.remove(engine.world, prev.grabbed.constraint);
      meshRef.current = null;
    }

    const text = inputTextRef.current.replace(/\n/g, ' ');
    const fs = fontSizeRef.current;
    const trk = trackingRef.current;
    const { w: cw, h: ch } = containerSizeRef.current;
    const scale = fs / font.unitsPerEm;
    const lineH = fs * 1.3;
    const maxW = cw * 0.85;
    const spaceW = (font.charToGlyph(' ').advanceWidth || 250) * scale + trk;

    // Word-wrap layout
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const lines = [];
    let curLine = [], curW = 0;
    for (const word of words) {
      let ww = 0;
      for (const ch2 of word) ww += (font.charToGlyph(ch2).advanceWidth || 0) * scale + trk;
      ww -= trk;
      if (curW + ww > maxW && curLine.length > 0) {
        lines.push(curLine); curLine = [word]; curW = ww + spaceW;
      } else {
        curLine.push(word); curW += ww + spaceW;
      }
    }
    if (curLine.length > 0) lines.push(curLine);

    // Build combined path with tracking
    const combined = new OTPath();
    let cy2 = 0;
    for (const line of lines) {
      let cx2 = 0;
      for (let wi = 0; wi < line.length; wi++) {
        if (wi > 0) cx2 += spaceW;
        for (const char of line[wi]) {
          const glyph = font.charToGlyph(char);
          const gp = glyph.getPath(cx2, cy2, fs);
          for (const cmd of gp.commands) combined.commands.push(cmd);
          cx2 += (glyph.advanceWidth || 0) * scale + trk;
        }
      }
      cy2 += lineH;
    }

    const bb = combined.getBoundingBox();
    const offX = cw / 2 - (bb.x1 + bb.x2) / 2;
    const offY = ch / 2 - (bb.y1 + bb.y2) / 2;
    const pad = fs * MESH_PAD;
    const mx = bb.x1 + offX - pad, my = bb.y1 + offY - pad;
    const mw = bb.x2 - bb.x1 + pad * 2, mh = bb.y2 - bb.y1 + pad * 2;

    const restVerts = [];
    for (let r = 0; r < MESH_ROWS; r++)
      for (let c = 0; c < MESH_COLS; c++)
        restVerts.push({ x: mx + (c / (MESH_COLS - 1)) * mw, y: my + (r / (MESH_ROWS - 1)) * mh });

    const bodies = restVerts.map(v =>
      Bodies.circle(v.x, v.y, 3, { collisionFilter: MESH_COLLISION, frictionAir: 0.12 })
    );
    Composite.add(engine.world, bodies);

    const constraints = [];
    const addSpring = (i, j, stiff) => {
      constraints.push(Constraint.create({ bodyA: bodies[i], bodyB: bodies[j], stiffness: stiff, damping: MESH_DAMPING }));
    };
    for (let r = 0; r < MESH_ROWS; r++)
      for (let c = 0; c < MESH_COLS - 1; c++) addSpring(r * MESH_COLS + c, r * MESH_COLS + c + 1, MESH_STIFFNESS);
    for (let r = 0; r < MESH_ROWS - 1; r++)
      for (let c = 0; c < MESH_COLS; c++) addSpring(r * MESH_COLS + c, (r + 1) * MESH_COLS + c, MESH_STIFFNESS);
    for (let r = 0; r < MESH_ROWS - 1; r++)
      for (let c = 0; c < MESH_COLS - 1; c++) {
        addSpring(r * MESH_COLS + c, (r + 1) * MESH_COLS + c + 1, MESH_STIFFNESS * 0.5);
        addSpring(r * MESH_COLS + c + 1, (r + 1) * MESH_COLS + c, MESH_STIFFNESS * 0.5);
      }
    Composite.add(engine.world, constraints);

    const corners = [0, MESH_COLS - 1, (MESH_ROWS - 1) * MESH_COLS, MESH_ROWS * MESH_COLS - 1];
    const pins = corners.map(i =>
      Constraint.create({ bodyA: bodies[i], pointB: { x: restVerts[i].x, y: restVerts[i].y }, stiffness: MESH_PIN_STIFFNESS, damping: 0.1, length: 0 })
    );
    Composite.add(engine.world, pins);

    const tris = [];
    for (let r = 0; r < MESH_ROWS - 1; r++)
      for (let c = 0; c < MESH_COLS - 1; c++) {
        const tl = r * MESH_COLS + c, tr2 = tl + 1, bl = tl + MESH_COLS, br = bl + 1;
        tris.push([tl, tr2, bl]);
        tris.push([tr2, br, bl]);
      }

    const pathMap = buildPathMap(combined.commands, offX, offY, restVerts, tris);
    meshRef.current = { restVerts, bodies, constraints: [...constraints, ...pins], tris, pathMap, grabs: {} };
  }, []);
  useEffect(() => { initMeshRef.current = initMesh; }, [initMesh]);

  // Cycle display mode: fall → gaze → trace → mesh → fall
  const toggleDisplayMode = useCallback(() => {
    setDisplayMode(prev => {
      // Cleanup previous mode
      if (prev === 'fall') {
        if (engineRef.current) {
          for (const line of lineBodiesRef.current) Composite.remove(engineRef.current.world, line.body);
        }
        lineBodiesRef.current = [];
      }
      if (prev === 'gaze') { gazePositionRef.current = null; gazeLineRef.current = null; }
      if (prev === 'trace') { traceRef.current = { state: 'idle', points: [], cumDist: [0] }; }
      if (prev === 'mesh') destroyMesh();

      const next = { fall: 'gaze', gaze: 'trace', trace: 'mesh', mesh: 'fall' }[prev];
      if (next === 'gaze') gazeDropIndexRef.current = 0;
      return next;
    });
  }, [destroyMesh]);

  // Init mesh when entering mesh mode
  useEffect(() => {
    if (displayMode === 'mesh') {
      if (engineRef.current) engineRef.current.gravity.y = 0;
      initMesh();
    } else if (engineRef.current) {
      engineRef.current.gravity.y = 1;
    }
  }, [displayMode, initMesh]);

  // Composite video + grain + char canvas into a target-resolution context (center-cropped to target aspect)
  const drawCompositeFrame = useCallback((rctx, targetW, targetH) => {
    const { w: cw, h: ch } = containerSizeRef.current;
    if (cw <= 0 || ch <= 0) return;

    const cropAspect = targetW / targetH;
    const containerAspect = cw / ch;
    let cropW, cropH, cropX, cropY;
    if (containerAspect > cropAspect) {
      cropH = ch; cropW = ch * cropAspect; cropX = (cw - cropW) / 2; cropY = 0;
    } else {
      cropW = cw; cropH = cw / cropAspect; cropX = 0; cropY = (ch - cropH) / 2;
    }
    const scale = targetW / cropW;

    rctx.setTransform(1, 0, 0, 1, 0, 0);
    rctx.fillStyle = '#000';
    rctx.fillRect(0, 0, targetW, targetH);

    // Container→record transform: recordPt = (containerPt - crop) * scale
    const setStage = () => rctx.setTransform(scale, 0, 0, scale, -cropX * scale, -cropY * scale);

    // Video (cover-fit, grayscale filter, slim + optional mirror)
    const video = videoRef.current;
    if (video && video.videoWidth) {
      const vidW = video.videoWidth, vidH = video.videoHeight;
      const cAspect = cw / ch, vAspect = vidW / vidH;
      let dw, dh, ox, oy;
      if (cAspect > vAspect) { dw = cw; dh = cw / vAspect; ox = 0; oy = (ch - dh) / 2; }
      else { dh = ch; dw = ch * vAspect; ox = (cw - dw) / 2; oy = 0; }
      const shouldMirror = sourceModeRef.current === 'camera' && facingModeRef.current === 'user';
      rctx.save();
      rctx.filter = VIDEO_FILTER;
      setStage();
      rctx.translate(cw / 2, 0);
      rctx.scale(VIDEO_SLIM * (shouldMirror ? -1 : 1), 1);
      rctx.translate(-cw / 2, 0);
      rctx.drawImage(video, ox, oy, dw, dh);
      rctx.restore();
    }

    // Grain
    const grain = grainCanvasRef.current;
    if (grain) {
      rctx.setTransform(1, 0, 0, 1, 0, 0);
      const pat = rctx.createPattern(grain, 'repeat');
      if (pat) { rctx.fillStyle = pat; rctx.fillRect(0, 0, targetW, targetH); }
    }

    // Char canvas overlay (drawn in container space, scaled to fill)
    const cc = charCanvasRef.current;
    if (cc) {
      rctx.save();
      setStage();
      rctx.drawImage(cc, 0, 0, cw, ch);
      rctx.restore();
    }
    rctx.setTransform(1, 0, 0, 1, 0, 0);
  }, []);
  useEffect(() => { drawCompositeRef.current = drawCompositeFrame; }, [drawCompositeFrame]);

  // Start / stop video recording
  const toggleRecording = useCallback(() => {
    if (recordingRef.current) {
      mediaRecorderRef.current?.stop();
      recordingRef.current = false;
      setRecording(false);
      return;
    }
    const dims = ASPECT_DIMS[aspectRef.current];
    const canvas = recordCanvasRef.current || document.createElement('canvas');
    recordCanvasRef.current = canvas;
    canvas.width = dims.w; canvas.height = dims.h;
    recordCtxRef.current = canvas.getContext('2d');
    const mime = pickRecordMime();
    let rec;
    try {
      rec = new MediaRecorder(canvas.captureStream(30), { mimeType: mime, videoBitsPerSecond: 12000000 });
    } catch (e) {
      console.error('MediaRecorder init failed:', e);
      return;
    }
    recordedChunksRef.current = [];
    rec.ondataavailable = e => { if (e.data.size) recordedChunksRef.current.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: mime });
      const ext = mime.includes('mp4') ? 'mp4' : 'webm';
      downloadBlob(blob, `they-live-${Date.now()}.${ext}`);
    };
    mediaRecorderRef.current = rec;
    rec.start();
    recordingRef.current = true;
    setRecording(true);
  }, []);

  // Start / stop GIF capture (records until stopped, encoding incrementally)
  const toggleGif = useCallback(() => {
    const st = gifStateRef.current;
    if (st) {
      // Stop: finalize and download
      clearInterval(st.intervalId);
      st.enc.finish();
      downloadBlob(new Blob([st.enc.bytes()], { type: 'image/gif' }), `they-live-${Date.now()}.gif`);
      gifStateRef.current = null;
      setGifRecording(false);
      return;
    }
    // Start
    const dims = ASPECT_DIMS[aspectRef.current];
    const gw = 432;
    const gh = Math.round(gw * dims.h / dims.w);
    const gcanvas = document.createElement('canvas');
    gcanvas.width = gw; gcanvas.height = gh;
    const gctx = gcanvas.getContext('2d', { willReadFrequently: true });
    const fps = 12;
    const delay = Math.round(1000 / fps);
    const enc = GIFEncoder();
    const state = { enc, gctx, gw, gh, delay, first: true, intervalId: 0 };

    state.intervalId = setInterval(() => {
      drawCompositeFrame(gctx, gw, gh);
      const data = gctx.getImageData(0, 0, gw, gh).data;
      const palette = quantize(data, 256);
      const index = applyPalette(data, palette);
      enc.writeFrame(index, gw, gh, { palette, delay });
      state.first = false;
    }, delay);

    gifStateRef.current = state;
    setGifRecording(true);
  }, [drawCompositeFrame]);

  // Clear all lines
  const clearAll = useCallback(() => {
    if (engineRef.current) {
      for (const line of lineBodiesRef.current) Composite.remove(engineRef.current.world, line.body);
    }
    lineBodiesRef.current = [];
    gazeLineRef.current = null;
    gazePositionRef.current = null;
    gazeDropIndexRef.current = 0;
    traceRef.current = { state: 'idle', points: [], cumDist: [0] };
    if (displayMode === 'mesh') { destroyMesh(); initMesh(); }
  }, [displayMode, destroyMesh, initMesh]);

  return (
    <div className="finger-draw-app">
      <div className="fd-stage-area">
        <div
          className="fd-camera-wrap"
          ref={cameraWrapRef}
          style={{ aspectRatio: aspect === '9:16' ? '9 / 16' : '4 / 5' }}
          onClick={displayMode === 'fall' ? dropLine : displayMode === 'gaze' ? gazeDropLine : undefined}
        >
          <video ref={videoRef} autoPlay playsInline muted className={`fd-video${sourceMode === 'camera' && facingMode === 'user' ? ' fd-video-mirrored' : ''}`} />
          <div className="fd-grain" ref={grainRef} />
          <canvas ref={charCanvasRef} className="fd-char-canvas" />

          {modelLoading && (
            <div className="fd-loading">
              <div className="fd-spinner" />
              <span>LOADING POSE TRACKING...</span>
            </div>
          )}
        </div>
      </div>

      <div className="fd-controls">
        <textarea
          className="fd-text-input"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="Enter text (one line per body)..."
          rows={Math.max(2, Math.min(inputText.split('\n').length, 4))}
        />
        <div className="fd-toolbar">
          <div className="fd-sliders">
            <div className="fd-size-control">
              <span className="fd-size-label">{fontSize}px</span>
              <input type="range" className="fd-size-slider" min="24" max="400"
                value={fontSize} onChange={e => setFontSize(Number(e.target.value))} />
            </div>
            <div className="fd-size-control">
              <span className="fd-size-label">{tracking > 0 ? '+' : ''}{tracking}</span>
              <input type="range" className="fd-size-slider" min="-20" max="100"
                value={tracking} onChange={e => setTracking(Number(e.target.value))} />
            </div>
          </div>
          <input ref={fileInputRef} type="file" accept="video/*" hidden
            onChange={handleFileSelect} />
          <div className="fd-colors">
            {COLOR_PALETTE.map((c, i) => (
              <button key={c} className={`fd-color-btn${charColor === i ? ' fd-color-active' : ''}`}
                style={{ background: c }} onClick={() => setCharColor(i)} />
            ))}
          </div>
          <button onClick={() => fileInputRef.current?.click()} className="fd-btn">&#x1F4CE;</button>
          {sourceMode === 'video' ? (
            <button onClick={switchToCamera} className="fd-btn">CAM</button>
          ) : (
            <button onClick={toggleCamera} className="fd-btn">&#x21C6;</button>
          )}
          <button onClick={toggleDisplayMode} className={`fd-btn${displayMode !== 'fall' ? ' fd-btn-active' : ''}`}>
            {displayMode.toUpperCase()}
          </button>
          <button onClick={() => setSweep(s => !s)} className={`fd-btn${sweep ? ' fd-btn-active' : ''}`}>
            SWEEP
          </button>
          {displayMode === 'mesh' && (
            <button onClick={() => setWindOn(w => !w)} className={`fd-btn${windOn ? ' fd-btn-active' : ''}`}>
              WIND
            </button>
          )}
          <button onClick={clearAll} className="fd-btn">CLEAR</button>
        </div>
        <div className="fd-toolbar">
          <button onClick={() => setAspect('9:16')} className={`fd-btn${aspect === '9:16' ? ' fd-btn-active' : ''}`}>9:16</button>
          <button onClick={() => setAspect('4:5')} className={`fd-btn${aspect === '4:5' ? ' fd-btn-active' : ''}`}>4:5</button>
          <button onClick={toggleRecording} className={`fd-btn${recording ? ' fd-btn-rec' : ''}`}>
            {recording ? '■ STOP' : '● REC'}
          </button>
          <button onClick={toggleGif} className={`fd-btn${gifRecording ? ' fd-btn-rec' : ''}`}>
            {gifRecording ? '■ GIF' : 'GIF'}
          </button>
        </div>
      </div>
    </div>
  );
}
