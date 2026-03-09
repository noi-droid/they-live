import { useRef, useEffect, useState, useCallback } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import Matter from 'matter-js';
import './FingerDrawApp.css';

const { Engine, Bodies, Body, Composite } = Matter;

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
const HEAD_CIRCLE_RADIUS = 50;
const MAX_LINES = 50;
const CHAR_COLOR = '#C8FF00';

let nextLineId = 0;

export default function FingerDrawApp() {
  const videoRef = useRef(null);
  const charCanvasRef = useRef(null);
  const cameraWrapRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const engineRef = useRef(null);
  const segmentCirclesRef = useRef([]);
  const headCircleRef = useRef(null);
  const lineBodiesRef = useRef([]);
  const wallsRef = useRef([]);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const containerSizeRef = useRef({ w: 0, h: 0 });
  const grainRef = useRef(null);
  const prevLandmarksRef = useRef(null);
  const measureCtxRef = useRef(null);
  const dropIndexRef = useRef(0);

  const [modelLoading, setModelLoading] = useState(true);
  const [inputText, setInputText] = useState('THEY LIVE');
  const [fontSize, setFontSize] = useState(48);
  const [tracking, setTracking] = useState(0);
  const [facingMode, setFacingMode] = useState('user');
  const inputTextRef = useRef('THEY LIVE');
  const fontSizeRef = useRef(48);
  const trackingRef = useRef(0);
  const facingModeRef = useRef('user');

  useEffect(() => {
    inputTextRef.current = inputText;
    dropIndexRef.current = 0;
  }, [inputText]);

  useEffect(() => {
    fontSizeRef.current = fontSize;
  }, [fontSize]);

  useEffect(() => {
    trackingRef.current = tracking;
  }, [tracking]);

  useEffect(() => {
    facingModeRef.current = facingMode;
  }, [facingMode]);

  // Create offscreen canvas for text measurement
  useEffect(() => {
    const c = document.createElement('canvas');
    measureCtxRef.current = c.getContext('2d');
  }, []);

  // Ensure OTR Grotesk is loaded for canvas rendering
  useEffect(() => {
    document.fonts.load("72px 'OTR Grotesk'");
  }, []);

  // Generate grain noise texture
  useEffect(() => {
    const size = 150;
    const c = document.createElement('canvas');
    c.width = size;
    c.height = size;
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
    const adjustedX = facingModeRef.current === 'user' ? 1 - lmX : lmX;
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
    return () => {
      cancelled = true;
      poseLandmarkerRef.current?.close();
    };
  }, []);

  // Start camera (restarts when facingMode changes)
  useEffect(() => {
    let cancelled = false;
    (async () => {
      // Stop previous stream
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
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (e) {
        console.error('Camera error:', e);
      }
    })();
    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, [facingMode]);

  // Init matter.js engine, walls, body collision circles
  useEffect(() => {
    const engine = Engine.create({ gravity: { x: 0, y: 1 } });
    engineRef.current = engine;

    const updateWalls = () => {
      const container = cameraWrapRef.current;
      if (!container) return;
      const { width: cw, height: ch } = container.getBoundingClientRect();
      containerSizeRef.current = { w: cw, h: ch };
      if (wallsRef.current.length) {
        Composite.remove(engine.world, wallsRef.current);
      }
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

    // Create body segment collision circles
    const circles = [];
    for (let i = 0; i < BODY_SEGMENTS.length * CIRCLES_PER_SEGMENT; i++) {
      const c = Bodies.circle(-1000, -1000, SEGMENT_CIRCLE_RADIUS, {
        isStatic: true,
        friction: 0.3,
        restitution: 0.4,
      });
      Composite.add(engine.world, c);
      circles.push(c);
    }
    segmentCirclesRef.current = circles;

    // Create head collision circle
    const head = Bodies.circle(-1000, -1000, HEAD_CIRCLE_RADIUS, {
      isStatic: true,
      friction: 0.3,
      restitution: 0.4,
    });
    Composite.add(engine.world, head);
    headCircleRef.current = head;

    return () => {
      ro.disconnect();
      Engine.clear(engine);
      engineRef.current = null;
    };
  }, []);

  // Update body collision from pose landmarks
  const updateBodyFromPose = useCallback((landmarks) => {
    const circles = segmentCirclesRef.current;
    const head = headCircleRef.current;
    if (!circles.length || !head) return;

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

    // Head
    const nose = landmarks[0];
    if (nose && (nose.visibility ?? 0) > 0.3) {
      const p = landmarkToCss(nose.x, nose.y);
      Body.setPosition(head, { x: p.x, y: p.y - HEAD_CIRCLE_RADIUS * 0.3 });
    } else {
      Body.setPosition(head, { x: -1000, y: -1000 });
    }
  }, [landmarkToCss]);

  // Hide body when no pose detected
  const hideBody = useCallback(() => {
    for (const c of segmentCirclesRef.current) {
      Body.setPosition(c, { x: -1000, y: -1000 });
    }
    if (headCircleRef.current) {
      Body.setPosition(headCircleRef.current, { x: -1000, y: -1000 });
    }
  }, []);

  // Drop one line at a time per tap, cycling through lines in order
  const dropLineAt = useCallback((cx) => {
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

    const x = Math.max(bodyW / 2, Math.min(cw - bodyW / 2, cx));
    const y = -bodyH - Math.random() * 40;

    const body = Bodies.rectangle(x, y, bodyW, bodyH, {
      restitution: 0.25,
      friction: 0.6,
      frictionAir: 0.003,
      angle: (Math.random() - 0.5) * 0.15,
    });

    Body.setVelocity(body, {
      x: (Math.random() - 0.5) * 2,
      y: 2 + Math.random() * 2,
    });

    Composite.add(engineRef.current.world, body);
    lineBodiesRef.current.push({
      id: ++nextLineId, body, text: line, fontSize: fs, tracking: trk,
    });

    // Remove excess (oldest first)
    while (lineBodiesRef.current.length > MAX_LINES) {
      const oldest = lineBodiesRef.current.shift();
      if (engineRef.current) Composite.remove(engineRef.current.world, oldest.body);
    }
  }, [measureLineWidth]);

  // Handle tap on camera area to drop next line
  const handleTap = useCallback((e) => {
    const container = cameraWrapRef.current;
    if (!container) return;
    const cRect = container.getBoundingClientRect();
    const cx = e.clientX - cRect.left;
    dropLineAt(cx);
  }, [dropLineAt]);

  // Main loop: physics + pose detection + canvas rendering
  useEffect(() => {
    if (modelLoading) return;
    let lastTime = performance.now();
    let lastDetectTime = -1;

    function loop(now) {
      // Physics update
      const dt = Math.min(now - lastTime, 33);
      lastTime = now;
      if (engineRef.current) Engine.update(engineRef.current, dt);

      // Pose detection with smoothing
      const video = videoRef.current;
      const pl = poseLandmarkerRef.current;
      if (video && pl && video.readyState >= 2 && video.currentTime !== lastDetectTime) {
        lastDetectTime = video.currentTime;
        const res = pl.detectForVideo(video, now);
        if (res.landmarks?.length > 0) {
          const raw = res.landmarks[0];
          // Exponential smoothing to reduce jitter
          const SMOOTH = 0.35;
          if (!prevLandmarksRef.current) {
            prevLandmarksRef.current = raw.map(l => ({ x: l.x, y: l.y, visibility: l.visibility }));
          }
          const smoothed = raw.map((l, i) => {
            const p = prevLandmarksRef.current[i];
            return {
              ...l,
              x: p.x + (l.x - p.x) * (1 - SMOOTH),
              y: p.y + (l.y - p.y) * (1 - SMOOTH),
            };
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
          canvas.width = pw;
          canvas.height = ph;
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

          // Measure char widths for tracking layout
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
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [modelLoading, updateBodyFromPose, hideBody]);

  // Toggle camera facing mode
  const toggleCamera = useCallback(() => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
    prevLandmarksRef.current = null;
  }, []);

  // Clear all lines
  const clearAll = useCallback(() => {
    if (engineRef.current) {
      for (const line of lineBodiesRef.current) {
        Composite.remove(engineRef.current.world, line.body);
      }
    }
    lineBodiesRef.current = [];
  }, []);

  return (
    <div className="finger-draw-app">
      <div className="fd-camera-wrap" ref={cameraWrapRef} onClick={handleTap}>
        <video ref={videoRef} autoPlay playsInline muted className={`fd-video${facingMode === 'user' ? ' fd-video-mirrored' : ''}`} />
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
          rows={Math.min(inputText.split('\n').length, 3) || 1}
        />
        <div className="fd-toolbar">
          <div className="fd-sliders">
            <div className="fd-size-control">
              <span className="fd-size-label">{fontSize}px</span>
              <input
                type="range"
                className="fd-size-slider"
                min="24"
                max="400"
                value={fontSize}
                onChange={e => setFontSize(Number(e.target.value))}
              />
            </div>
            <div className="fd-size-control">
              <span className="fd-size-label">{tracking > 0 ? '+' : ''}{tracking}</span>
              <input
                type="range"
                className="fd-size-slider"
                min="-20"
                max="100"
                value={tracking}
                onChange={e => setTracking(Number(e.target.value))}
              />
            </div>
          </div>
          <button onClick={toggleCamera} className="fd-btn">
            &#x21C6;
          </button>
          <button onClick={clearAll} className="fd-btn">
            CLEAR
          </button>
        </div>
      </div>
    </div>
  );
}
