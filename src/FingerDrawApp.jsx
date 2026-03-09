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
const TAP_THRESHOLD = 10; // px — movement below this counts as tap

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

  // Pointer / grab tracking
  const pointersRef = useRef(new Map());
  const grabRef = useRef(null);
  const tapStartRef = useRef(null);
  const prevMidRef = useRef(null);
  const velRef = useRef({ x: 0, y: 0 });

  const fileInputRef = useRef(null);
  const videoFileUrlRef = useRef(null);

  const [modelLoading, setModelLoading] = useState(true);
  const [inputText, setInputText] = useState('THEY\nLIVE');
  const [fontSize, setFontSize] = useState(48);
  const [tracking, setTracking] = useState(0);
  const [facingMode, setFacingMode] = useState('user');
  const [sourceMode, setSourceMode] = useState('camera'); // 'camera' | 'video'
  const inputTextRef = useRef('THEY\nLIVE');
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

  // ── Pointer event handlers (tap / 1-finger drag / 2-finger pinch) ──

  const findBodyAt = useCallback((px, py) => {
    const allBodies = lineBodiesRef.current.map(l => l.body);
    const hits = Query.point(allBodies, { x: px, y: py });
    if (!hits.length) return null;
    return lineBodiesRef.current.find(l => l.body === hits[0]) || null;
  }, []);

  const pointerPos = useCallback((e) => {
    const rect = cameraWrapRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const onPointerDown = useCallback((e) => {
    const container = cameraWrapRef.current;
    if (!container) return;
    const pos = pointerPos(e);
    container.setPointerCapture(e.pointerId);
    pointersRef.current.set(e.pointerId, pos);

    const canGrab = sourceModeRef.current === 'camera';

    if (pointersRef.current.size === 1) {
      const entry = canGrab ? findBodyAt(pos.x, pos.y) : null;
      if (entry) {
        // Grab body (camera mode only)
        grabRef.current = {
          entry,
          offset: { x: pos.x - entry.body.position.x, y: pos.y - entry.body.position.y },
          initAngle: null,
          initBodyAngle: entry.body.angle,
        };
        Body.setStatic(entry.body, true);
        Body.setVelocity(entry.body, { x: 0, y: 0 });
        prevMidRef.current = pos;
        velRef.current = { x: 0, y: 0 };
        tapStartRef.current = null;
      } else {
        // Potential tap
        grabRef.current = null;
        tapStartRef.current = pos;
      }
    } else if (pointersRef.current.size === 2 && grabRef.current) {
      // Second finger: enable rotation
      tapStartRef.current = null;
      const pts = Array.from(pointersRef.current.values());
      grabRef.current.initAngle = Math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x);
      grabRef.current.initBodyAngle = grabRef.current.entry.body.angle;
      prevMidRef.current = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
    }
  }, [findBodyAt, pointerPos]);

  const onPointerMove = useCallback((e) => {
    if (!pointersRef.current.has(e.pointerId)) return;
    const pos = pointerPos(e);
    pointersRef.current.set(e.pointerId, pos);

    // Cancel tap if moved too far
    if (tapStartRef.current) {
      const dx = pos.x - tapStartRef.current.x;
      const dy = pos.y - tapStartRef.current.y;
      if (Math.hypot(dx, dy) > TAP_THRESHOLD) tapStartRef.current = null;
    }

    const grab = grabRef.current;
    if (!grab) return;
    const pointers = pointersRef.current;

    if (pointers.size === 1) {
      // Single finger drag
      const nx = pos.x - grab.offset.x;
      const ny = pos.y - grab.offset.y;
      Body.setPosition(grab.entry.body, { x: nx, y: ny });
      if (prevMidRef.current) {
        velRef.current = { x: pos.x - prevMidRef.current.x, y: pos.y - prevMidRef.current.y };
      }
      prevMidRef.current = pos;
    } else if (pointers.size === 2) {
      // Two-finger: rotate + translate
      const pts = Array.from(pointers.values());
      const mid = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
      const angle = Math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x);

      if (grab.initAngle !== null) {
        Body.setAngle(grab.entry.body, grab.initBodyAngle + (angle - grab.initAngle));
      }

      if (prevMidRef.current) {
        const dx = mid.x - prevMidRef.current.x;
        const dy = mid.y - prevMidRef.current.y;
        Body.setPosition(grab.entry.body, {
          x: grab.entry.body.position.x + dx,
          y: grab.entry.body.position.y + dy,
        });
        velRef.current = { x: dx, y: dy };
      }
      prevMidRef.current = mid;
    }
  }, [pointerPos]);

  const onPointerUp = useCallback((e) => {
    pointersRef.current.delete(e.pointerId);
    const grab = grabRef.current;

    if (grab && pointersRef.current.size === 0) {
      // Release: make dynamic and throw
      Body.setStatic(grab.entry.body, false);
      Body.setVelocity(grab.entry.body, {
        x: velRef.current.x * 0.4,
        y: velRef.current.y * 0.4,
      });
      grabRef.current = null;
      prevMidRef.current = null;
      velRef.current = { x: 0, y: 0 };
    } else if (grab && pointersRef.current.size === 1) {
      // 2→1 finger transition: recalculate offset
      const remaining = Array.from(pointersRef.current.values())[0];
      grab.offset = {
        x: remaining.x - grab.entry.body.position.x,
        y: remaining.y - grab.entry.body.position.y,
      };
      grab.initAngle = null;
      prevMidRef.current = remaining;
    } else if (!grab && tapStartRef.current && pointersRef.current.size === 0) {
      // Tap on empty area → drop next line
      dropLine();
      tapStartRef.current = null;
    }
  }, [dropLine]);

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
        } else {
          hideBody();
          prevLandmarksRef.current = null;
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
        ctx.restore();
      }

      animFrameRef.current = requestAnimationFrame(loop);
    }

    animFrameRef.current = requestAnimationFrame(loop);
    return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
  }, [modelLoading, updateBodyFromPose, hideBody]);

  // Toggle camera facing mode
  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    prevLandmarksRef.current = null;
  }, []);

  // Clear all lines
  const clearAll = useCallback(() => {
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
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
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
