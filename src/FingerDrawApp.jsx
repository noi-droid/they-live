import { useRef, useEffect, useState, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import Matter from 'matter-js';
import './FingerDrawApp.css';

const { Engine, Bodies, Body, Composite, Query } = Matter;

// Body landmark pairs for collision segments
const BODY_SEGMENTS = [
  [11, 12], // shoulders
  [11, 13], // left upper arm
  [13, 15], // left forearm
  [12, 14], // right upper arm
  [14, 16], // right forearm
  [11, 23], // left torso
  [12, 24], // right torso
  [23, 24], // hips
];

const CIRCLES_PER_SEGMENT = 16;
const SEGMENT_CIRCLE_RADIUS = 22;
const HEAD_CIRCLES_COUNT = 9;
const HEAD_CIRCLE_R = 22;
const MAX_LINES = 50;
const CHAR_COLOR = '#C8FF00';

// Camera-based pinch detection — scale-relative thresholds
// Ratios are relative to hand size (wrist→index distance)
const PINCH_GRAB_RATIO = 0.28;   // thumb-index / hand-size to start grab
const PINCH_RELEASE_RATIO = 0.55; // must open wider to release (hysteresis)
const PINCH_CONFIRM_FRAMES = 4;  // consecutive pinch frames before grab triggers
const PINCH_MIN_HAND = 0.04;     // min hand size (normalised) — ignore if too far
const PINCH_HIT_RADIUS = 60;     // CSS-px radius for body hit-test around pinch point
const PINCH_FOLLOW = 0.45;       // lerp factor — 0 = frozen, 1 = instant snap

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
  const engineRef = useRef(null);
  const segmentCirclesRef = useRef([]);
  const headCirclesRef = useRef([]);
  const lineBodiesRef = useRef([]);
  const wallsRef = useRef([]);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const containerSizeRef = useRef({ w: 0, h: 0 });
  const grainRef = useRef(null);
  const prevLandmarksRef = useRef(null);
  const measureCtxRef = useRef(null);
  const dropIndexRef = useRef(0);

  // Camera-based pinch state for each hand
  // { grabbing: false, entry: null, prevPos: null, vel: { x: 0, y: 0 } }
  const pinchStateRef = useRef({
    left:  { grabbing: false, entry: null, prevPos: null, vel: { x: 0, y: 0 }, frames: 0 },
    right: { grabbing: false, entry: null, prevPos: null, vel: { x: 0, y: 0 }, frames: 0 },
  });

  const fileInputRef = useRef(null);
  const videoFileUrlRef = useRef(null);

  const [modelLoading, setModelLoading] = useState(true);
  const [inputText, setInputText] = useState(DEFAULT_TEXT);
  const [fontSize, setFontSize] = useState(48);
  const [tracking, setTracking] = useState(0);
  const [facingMode, setFacingMode] = useState('user');
  const [sourceMode, setSourceMode] = useState('camera'); // 'camera' | 'video'
  const inputTextRef = useRef(DEFAULT_TEXT);
  const fontSizeRef = useRef(48);
  const trackingRef = useRef(0);
  const facingModeRef = useRef('user');
  const sourceModeRef = useRef('camera');

  useEffect(() => {
    inputTextRef.current = inputText;
    dropIndexRef.current = 0;
  }, [inputText]);

  useEffect(() => { fontSizeRef.current = fontSize; }, [fontSize]);
  useEffect(() => { trackingRef.current = tracking; }, [tracking]);
  useEffect(() => { facingModeRef.current = facingMode; }, [facingMode]);
  useEffect(() => { sourceModeRef.current = sourceMode; }, [sourceMode]);

  // Create offscreen canvas for text measurement
  useEffect(() => {
    const c = document.createElement('canvas');
    measureCtxRef.current = c.getContext('2d');
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

  // Init PoseLandmarker
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        if (cancelled) return;
        const pl = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
            delegate: 'GPU',
          },
          numPoses: 1,
          runningMode: 'VIDEO',
        });
        if (cancelled) return;
        poseLandmarkerRef.current = pl;
        setModelLoading(false);
      } catch (e) {
        console.error('PoseLandmarker init error:', e);
      }
    })();
    return () => { cancelled = true; poseLandmarkerRef.current?.close(); };
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
      wallsRef.current = [
        Bodies.rectangle(cw / 2, ch + t / 2, cw * 2, t, { isStatic: true }),
        Bodies.rectangle(-t / 2, ch / 2, t, ch * 2, { isStatic: true }),
        Bodies.rectangle(cw + t / 2, ch / 2, t, ch * 2, { isStatic: true }),
      ];
      Composite.add(engine.world, wallsRef.current);
    };

    updateWalls();
    const ro = new ResizeObserver(updateWalls);
    if (cameraWrapRef.current) ro.observe(cameraWrapRef.current);

    const circles = [];
    for (let i = 0; i < BODY_SEGMENTS.length * CIRCLES_PER_SEGMENT; i++) {
      const c = Bodies.circle(-1000, -1000, SEGMENT_CIRCLE_RADIUS, {
        isStatic: true, friction: 0.3, restitution: 0.4,
      });
      Composite.add(engine.world, c);
      circles.push(c);
    }
    segmentCirclesRef.current = circles;

    const headCircles = [];
    for (let i = 0; i < HEAD_CIRCLES_COUNT; i++) {
      const c = Bodies.circle(-1000, -1000, HEAD_CIRCLE_R, {
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

    for (let s = 0; s < BODY_SEGMENTS.length; s++) {
      const [a, b] = BODY_SEGMENTS[s];
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
        Body.setPosition(circles[baseIdx + i], {
          x: pA.x + (pB.x - pA.x) * t,
          y: pA.y + (pB.y - pA.y) * t,
        });
      }
    }

    // Head — cluster of circles, sized from shoulder width
    const nose = landmarks[0];
    const lShoulder = landmarks[11];
    const rShoulder = landmarks[12];
    const hasNose = nose && (nose.visibility ?? 0) > 0.3;
    const hasShoulders = lShoulder && rShoulder &&
      (lShoulder.visibility ?? 0) > 0.3 && (rShoulder.visibility ?? 0) > 0.3;

    if (hasNose) {
      const pNose = landmarkToCss(nose.x, nose.y);
      let headR = 50;
      if (hasShoulders) {
        const pLS = landmarkToCss(lShoulder.x, lShoulder.y);
        const pRS = landmarkToCss(rShoulder.x, rShoulder.y);
        headR = Math.max(Math.hypot(pRS.x - pLS.x, pRS.y - pLS.y) * 0.38, 35);
      }
      const hcx = pNose.x;
      const hcy = pNose.y - headR * 0.65;
      const offsets = [
        [0, 0], [0, -0.75], [0, 0.65],
        [-0.6, -0.2], [0.6, -0.2],
        [-0.4, -0.6], [0.4, -0.6],
        [-0.35, 0.4], [0.35, 0.4],
      ];
      for (let i = 0; i < HEAD_CIRCLES_COUNT; i++) {
        Body.setPosition(headCircles[i], {
          x: hcx + offsets[i][0] * headR,
          y: hcy + offsets[i][1] * headR,
        });
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

  // Drop next line (random x, reverse order)
  const dropLine = useCallback(() => {
    if (!engineRef.current) return;
    const text = inputTextRef.current;
    const lines = text.split('\n').filter(l => l.length > 0);
    if (!lines.length) return;
    const { w: cw } = containerSizeRef.current;
    if (cw === 0) return;

    const idx = dropIndexRef.current % lines.length;
    const line = lines[lines.length - 1 - idx];
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
    lineBodiesRef.current.push({ id: ++nextLineId, body, text: line, fontSize: fs, tracking: trk });

    while (lineBodiesRef.current.length > MAX_LINES) {
      const oldest = lineBodiesRef.current.shift();
      if (engineRef.current) Composite.remove(engineRef.current.world, oldest.body);
    }
  }, [measureLineWidth]);

  // ── Camera-based pinch detection (runs inside main loop) ──

  const findBodyNear = useCallback((px, py) => {
    // Hit-test: try exact point first, then expand search radius
    const allBodies = lineBodiesRef.current.map(l => l.body);
    const hits = Query.point(allBodies, { x: px, y: py });
    if (hits.length) return lineBodiesRef.current.find(l => l.body === hits[0]) || null;

    // Expanded search: check if any body center is within PINCH_HIT_RADIUS
    let best = null, bestDist = PINCH_HIT_RADIUS;
    for (const line of lineBodiesRef.current) {
      const d = Math.hypot(line.body.position.x - px, line.body.position.y - py);
      if (d < bestDist) { bestDist = d; best = line; }
    }
    return best;
  }, []);

  const updatePinchForHand = useCallback((hand, indexLm, thumbLm, wristLm) => {
    const state = pinchStateRef.current[hand];

    const releasePinch = () => {
      if (state.grabbing && state.entry) {
        Body.setStatic(state.entry.body, false);
        Body.setVelocity(state.entry.body, {
          x: state.vel.x * 0.5, y: state.vel.y * 0.5,
        });
      }
      state.grabbing = false;
      state.entry = null;
      state.prevPos = null;
      state.vel = { x: 0, y: 0 };
      state.frames = 0;
    };

    if (!indexLm || !thumbLm || !wristLm) { releasePinch(); return; }

    const visOk = (indexLm.visibility ?? 0) > 0.3
      && (thumbLm.visibility ?? 0) > 0.3
      && (wristLm.visibility ?? 0) > 0.2;
    if (!visOk) { releasePinch(); return; }

    // Hand size = wrist→index distance (scales with camera distance)
    const handSize = Math.hypot(indexLm.x - wristLm.x, indexLm.y - wristLm.y);
    if (handSize < PINCH_MIN_HAND) { releasePinch(); return; } // too far / too small

    // Dynamic thresholds relative to hand size
    const grabDist = handSize * PINCH_GRAB_RATIO;
    const releaseDist = handSize * PINCH_RELEASE_RATIO;

    // Normalised distance between thumb and index
    const dist = Math.hypot(thumbLm.x - indexLm.x, thumbLm.y - indexLm.y);
    // Midpoint in CSS coords
    const mid = landmarkToCss(
      (indexLm.x + thumbLm.x) / 2,
      (indexLm.y + thumbLm.y) / 2,
    );

    if (!state.grabbing) {
      // Check if pinching — require consecutive frames to avoid false positives
      if (dist < grabDist) {
        state.frames++;
        if (state.frames >= PINCH_CONFIRM_FRAMES) {
          const entry = findBodyNear(mid.x, mid.y);
          if (entry) {
            // Don't steal from other hand
            const other = hand === 'left' ? 'right' : 'left';
            if (pinchStateRef.current[other].entry === entry) return;

            state.grabbing = true;
            state.entry = entry;
            state.prevPos = mid;
            state.vel = { x: 0, y: 0 };
            Body.setStatic(entry.body, true);
            Body.setVelocity(entry.body, { x: 0, y: 0 });
          }
        }
      } else {
        state.frames = 0; // reset if fingers open
      }
    } else {
      // Currently grabbing
      if (dist > releaseDist) {
        // Release — throw with velocity
        if (state.entry) {
          Body.setStatic(state.entry.body, false);
          Body.setVelocity(state.entry.body, {
            x: state.vel.x * 0.5, y: state.vel.y * 0.5,
          });
        }
        state.grabbing = false;
        state.entry = null;
        state.prevPos = null;
        state.vel = { x: 0, y: 0 };
      } else {
        // Move body smoothly toward pinch midpoint (lerp)
        if (state.entry) {
          const bx = state.entry.body.position.x;
          const by = state.entry.body.position.y;
          const nx = bx + (mid.x - bx) * PINCH_FOLLOW;
          const ny = by + (mid.y - by) * PINCH_FOLLOW;
          Body.setPosition(state.entry.body, { x: nx, y: ny });
          // Velocity based on actual body movement (for throw)
          state.vel = { x: nx - bx, y: ny - by };
          state.prevPos = mid;
        }
      }
    }
  }, [landmarkToCss, findBodyNear]);

  // Main loop: physics + pose detection + canvas rendering
  useEffect(() => {
    if (modelLoading) return;
    let lastTime = performance.now();
    let lastDetectTime = -1;

    function loop(now) {
      const dt = Math.min(now - lastTime, 33);
      lastTime = now;
      if (engineRef.current) Engine.update(engineRef.current, dt);

      const video = videoRef.current;
      const pl = poseLandmarkerRef.current;
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

          // Camera-based pinch detection (camera mode only)
          if (sourceModeRef.current === 'camera') {
            updatePinchForHand('left',  smoothed[19], smoothed[21], smoothed[15]);
            updatePinchForHand('right', smoothed[20], smoothed[22], smoothed[16]);
          }
        } else {
          hideBody();
          prevLandmarksRef.current = null;
          // Release any pinch grabs when pose lost
          for (const hand of ['left', 'right']) {
            const ps = pinchStateRef.current[hand];
            if (ps.grabbing && ps.entry) {
              Body.setStatic(ps.entry.body, false);
            }
            ps.grabbing = false;
            ps.entry = null;
            ps.prevPos = null;
            ps.vel = { x: 0, y: 0 };
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

        for (const line of lineBodiesRef.current) {
          const { x, y } = line.body.position;
          const angle = line.body.angle;
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(angle);
          ctx.font = `${line.fontSize}px 'OTR Grotesk', sans-serif`;
          ctx.fillStyle = CHAR_COLOR;
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

        // Draw pinch indicators
        for (const hand of ['left', 'right']) {
          const ps = pinchStateRef.current[hand];
          if (ps.grabbing && ps.prevPos) {
            ctx.save();
            ctx.strokeStyle = CHAR_COLOR;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.arc(ps.prevPos.x, ps.prevPos.y, 18, 0, Math.PI * 2);
            ctx.stroke();
            // Inner dot
            ctx.fillStyle = CHAR_COLOR;
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.arc(ps.prevPos.x, ps.prevPos.y, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
          }
        }

        ctx.restore();
      }

      animFrameRef.current = requestAnimationFrame(loop);
    }

    animFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [modelLoading, updateBodyFromPose, hideBody, updatePinchForHand]);

  // Toggle camera facing mode
  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    prevLandmarksRef.current = null;
  }, []);

  // Clear all lines (also release any pinch grabs)
  const clearAll = useCallback(() => {
    for (const hand of ['left', 'right']) {
      const ps = pinchStateRef.current[hand];
      ps.grabbing = false;
      ps.entry = null;
      ps.prevPos = null;
      ps.vel = { x: 0, y: 0 };
      ps.frames = 0;
    }
    if (engineRef.current) {
      for (const line of lineBodiesRef.current) Composite.remove(engineRef.current.world, line.body);
    }
    lineBodiesRef.current = [];
  }, []);

  return (
    <div className="finger-draw-app">
      <div
        className="fd-camera-wrap"
        ref={cameraWrapRef}
        onClick={dropLine}
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
          <button onClick={() => fileInputRef.current?.click()} className="fd-btn">&#x1F4CE;</button>
          {sourceMode === 'video' ? (
            <button onClick={switchToCamera} className="fd-btn">CAM</button>
          ) : (
            <button onClick={toggleCamera} className="fd-btn">&#x21C6;</button>
          )}
          <button onClick={clearAll} className="fd-btn">CLEAR</button>
        </div>
      </div>
    </div>
  );
}
