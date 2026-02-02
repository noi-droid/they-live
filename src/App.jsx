import { useState, useRef, useEffect } from 'react';
import html2canvas from 'html2canvas';

function App() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [result, setResult] = useState(null);
  const [displayedResult, setDisplayedResult] = useState('');
  const [loading, setLoading] = useState(false);
  const [dots, setDots] = useState('');
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // ローディングドットアニメーション
  useEffect(() => {
    if (!loading) {
      setDots('');
      return;
    }
    
    let count = 0;
    const interval = setInterval(() => {
      count = (count + 1) % 4;
      setDots('.'.repeat(count));
    }, 300);
    
    return () => clearInterval(interval);
  }, [loading]);

  // タイプライター効果
  useEffect(() => {
    if (!result) {
      setDisplayedResult('');
      return;
    }
    
    setDisplayedResult('');
    let index = 0;
    
    const interval = setInterval(() => {
      if (index < result.length) {
        setDisplayedResult(result.slice(0, index + 1));
        index++;
      } else {
        clearInterval(interval);
      }
    }, 50);
    
    return () => clearInterval(interval);
  }, [result]);

  const startCamera = async () => {
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }

      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        } 
      });
      
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      
      setIsStreaming(true);
      setCapturedImage(null);
      setResult(null);
    } catch (e) {
      console.error('Camera access denied:', e);
    }
  };

  const applyFilters = (canvas, ctx) => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // モノクロ
    for (let i = 0; i < data.length; i += 4) {
      const avg = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
      data[i] = avg;
      data[i + 1] = avg;
      data[i + 2] = avg;
    }

    // グレイン
    const intensity = 120;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * intensity;
      data[i] = Math.min(255, Math.max(0, data[i] + noise));
      data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
      data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
    }

    ctx.putImageData(imageData, 0, 0);
  };

  const analyzeImage = async (base64Image) => {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base64Image }),
    });

    if (!response.ok) {
      throw new Error('Server error');
    }

    const data = await response.json();
    return data.result;
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current) return;

    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const rawImageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Image = rawImageData.replace(/^data:image\/\w+;base64,/, '');
    
    applyFilters(canvas, ctx);
    const filteredImageData = canvas.toDataURL('image/jpeg', 0.8);
    
    setCapturedImage(filteredImageData);
    setLoading(true);

    try {
      const resultText = await analyzeImage(base64Image);
      setResult(resultText);
    } catch (e) {
      console.error('API error:', e);
      setResult('OBEY');
    }
    
    setLoading(false);
  };

  const saveImage = async () => {
    const element = document.querySelector('[data-capture]');
    if (!element) return;
    
    const canvas = await html2canvas(element, {
      backgroundColor: null,
      scale: 3,
    });
    
    if (navigator.share && navigator.canShare) {
      canvas.toBlob(async (blob) => {
        const file = new File([blob], `they-live-${Date.now()}.png`, { type: 'image/png' });
        
        if (navigator.canShare({ files: [file] })) {
          try {
            await navigator.share({ files: [file] });
            return;
          } catch (e) {
            console.log('Share cancelled');
          }
        }
      }, 'image/png');
    } else {
      const link = document.createElement('a');
      link.download = `they-live-${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };

  const reset = () => {
    setCapturedImage(null);
    setResult(null);
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'black',
      overflow: 'hidden',
    }}>
      
      {/* Start button */}
      {!isStreaming && !capturedImage && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
        }}>
          <button
            onClick={startCamera}
            style={{
              padding: '16px 32px',
              backgroundColor: 'white',
              color: 'black',
              fontFamily: '"OTR Grotesk", system-ui, sans-serif',
              fontSize: 14,
              fontWeight: 600,
              letterSpacing: '0.05em',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            PUT ON THE GLASSES
          </button>
        </div>
      )}

      {/* Video preview */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{
          display: isStreaming ? 'block' : 'none',
          width: '100vw',
          height: '100vh',
          objectFit: 'cover',
          filter: 'grayscale(100%)',
        }}
      />

      {/* Grain overlay for preview */}
      {isStreaming && !capturedImage && (
        <div style={{
          position: 'absolute',
          inset: 0,
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)'/%3E%3C/svg%3E")`,
          opacity: 0.4,
          mixBlendMode: 'overlay',
          pointerEvents: 'none',
        }}
        />
      )}

      {/* Captured image with overlay */}
      {capturedImage && (
        <div style={{
          position: 'absolute',
          inset: 0,
          zIndex: 5,
        }}>
          <div data-capture style={{
            position: 'relative',
            width: '100%',
            height: '100%',
          }}>
            <img
              src={capturedImage}
              alt="Captured"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
              }}
            />
            
            {/* Result overlay */}
            {result && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                padding: 32,
              }}>
                <div style={{
                  fontFamily: '"OTR Grotesk", system-ui, sans-serif',
                  fontWeight: 900,
                  fontSize: 'clamp(32px, 10vw, 100px)',
                  color: 'white',
                  textAlign: 'center',
                  lineHeight: 0.9,
                  letterSpacing: '-0.02em',
                  whiteSpace: 'pre-wrap',
                  textShadow: '0 0 20px rgba(0,0,0,0.8)',
                }}>
                  {displayedResult}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: 'rgba(0,0,0,0.5)',
          zIndex: 30,
        }}>
          <div style={{
            fontFamily: 'monospace',
            fontWeight: 400,
            fontSize: 14,
            color: 'white',
            letterSpacing: '0.05em',
          }}>
            REVEALING TRUTH{dots}
          </div>
        </div>
      )}

      {/* Hidden canvas */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {/* Capture button */}
      {isStreaming && !capturedImage && (
        <button
          onClick={captureAndAnalyze}
          style={{
            position: 'absolute',
            bottom: 32,
            left: '50%',
            transform: 'translateX(-50%)',
            width: 64,
            height: 64,
            borderRadius: '50%',
            backgroundColor: 'white',
            opacity: 0.5,
            border: 'none',
            cursor: 'pointer',
            zIndex: 10,
          }}
        />
      )}

      {/* Bottom buttons */}
      {capturedImage && !loading && (
        <div style={{
          position: 'absolute',
          bottom: 32,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          gap: 16,
          zIndex: 10,
        }}>
          <button
            onClick={reset}
            style={{
              padding: '12px 24px',
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              fontFamily: 'monospace',
              fontSize: 12,
              border: 'none',
              cursor: 'pointer',
            }}
          >
            WAKE UP
          </button>
          <button
            onClick={saveImage}
            style={{
              padding: '12px 24px',
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              fontFamily: 'monospace',
              fontSize: 12,
              border: 'none',
              cursor: 'pointer',
            }}
          >
            SAVE
          </button>
        </div>
      )}
    </div>
  );
}

export default App;