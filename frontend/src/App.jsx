import { useEffect, useRef, useState } from "react";
import "./App.css";

export default function App() {
  const canvasRef = useRef(null);
  const tinyRef = useRef(null);
  const [out, setOut] = useState("");

  useEffect(() => {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, c.width, c.height);
  }, []);

  function getPos(e) {
    const c = canvasRef.current;
    const r = c.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (c.width / r.width),
      y: (e.clientY - r.top) * (c.height / r.height),
    };
  }

  function clear() {
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, c.width, c.height);
    const t = tinyRef.current.getContext("2d");
    t.clearRect(0, 0, 28, 28);
    setOut("");
  }

  function canvasToPixels28() {
    const c = canvasRef.current;
    const tiny = tinyRef.current;
    const tctx = tiny.getContext("2d");

    tctx.imageSmoothingEnabled = true;
    tctx.clearRect(0, 0, 28, 28);
    tctx.drawImage(c, 0, 0, 28, 28);

    const img = tctx.getImageData(0, 0, 28, 28).data;
    const pixels = new Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      const r = img[i * 4], g = img[i * 4 + 1], b = img[i * 4 + 2];
      const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
      pixels[i] = gray; // 0..1 (black->0, white->1)
      // If your predictions look inverted, try: pixels[i] = 1 - gray;
    }
    return pixels;
  }

  async function predict() {
    setOut("Predicting...");
    const pixels = canvasToPixels28();

    const res = await fetch("http://localhost:8000/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels }),
    });

    if (!res.ok) {
      setOut(`Error: ${res.status}`);
      return;
    }
    const data = await res.json();
    const top = Math.max(...data.probs);
    setOut(`Prediction: ${data.digit} (confidence ${(top * 100).toFixed(1)}%)`);
  }

  // Drawing handlers
  const drawing = useRef(false);
  const last = useRef(null);

  function onDown(e) {
    const c = canvasRef.current;
    c.setPointerCapture(e.pointerId);
    drawing.current = true;
    last.current = getPos(e);
  }

  function onUp() {
    drawing.current = false;
    last.current = null;
  }

  function onMove(e) {
    if (!drawing.current) return;
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    const p = getPos(e);

    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();

    last.current = p;
  }

  return (
    <div style={{ fontFamily: "system-ui", padding: 16 }}>
      <h2>Draw a digit</h2>
      <div style={{ display: "flex", gap: 24, alignItems: "flex-start" }}>
        <div>
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            style={{ border: "1px solid #aaa", touchAction: "none", background: "black" }}
            onPointerDown={onDown}
            onPointerUp={onUp}
            onPointerCancel={onUp}
            onPointerMove={onMove}
          />
          <div style={{ marginTop: 12 }}>
            <button onClick={clear} style={{ marginRight: 8 }}>Clear</button>
            <button onClick={predict}>Predict</button>
          </div>
          <p>{out}</p>
        </div>

        <div>
          <canvas
            ref={tinyRef}
            width={28}
            height={28}
            style={{ width: 140, height: 140, imageRendering: "pixelated", border: "1px solid #ddd" }}
          />
          <p style={{ color: "#666" }}>28×28 preview</p>
        </div>
      </div>
    </div>
  );
}
