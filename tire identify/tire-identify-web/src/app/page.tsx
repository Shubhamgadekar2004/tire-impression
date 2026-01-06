
"use client";
import React, { useRef, useState } from "react";
import NextImage from "next/image";

type VehicleInfo = { make: string; model: string; class: string; tire: string };
type ApiResult = { vehicle: VehicleInfo; confidence: number; features: { mean: number; std: number; skew: number; width: number; height: number } };

// Local preview helpers

function computeGrayscale(data: Uint8ClampedArray) {
  const gray = new Float32Array(data.length / 4);
  for (let i = 0, g = 0; i < data.length; i += 4, g += 1) {
    const r = data[i],
      b = data[i + 1],
      gch = data[i + 2];
    gray[g] = (0.299 * r + 0.587 * b + 0.114 * gch) / 255;
  }
  return gray;
}

function sobelEdges(gray: Float32Array, width: number, height: number) {
  const out = new Uint8ClampedArray(width * height);
  const gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sx = 0, sy = 0;
      let k = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const ix = x + kx;
          const iy = y + ky;
          const idx = iy * width + ix;
          const val = gray[idx];
          sx += gx[k] * val;
          sy += gy[k] * val;
          k++;
        }
      }
      const mag = Math.min(255, Math.sqrt(sx * sx + sy * sy) * 255);
      out[y * width + x] = mag;
    }
  }
  return out;
}

async function callApiPredict(file: File): Promise<ApiResult> {
  const fd = new FormData();
  fd.append("image", file);
  const res = await fetch("/api/predict", { method: "POST", body: fd });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [layout, setLayout] = useState<"Vertical Cards" | "Split Horizontal">("Vertical Cards");
  const [vehicle, setVehicle] = useState<VehicleInfo | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [edgeUrl, setEdgeUrl] = useState<string>("");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleFile = async (f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);

    const img = new window.Image();
    img.src = url;
    await new Promise((res) => (img.onload = () => res(null)));

    const canvas = canvasRef.current!;
    const width = Math.min(900, img.width);
    const scale = width / img.width;
    const height = Math.floor(img.height * scale);
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(img, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);
    const gray = computeGrayscale(imageData.data);
    const edges = sobelEdges(gray, width, height);

    // Build a new image for edges
    const edgeImage = ctx.createImageData(width, height);
    for (let i = 0; i < edges.length; i++) {
      const v = edges[i];
      edgeImage.data[i * 4] = v;
      edgeImage.data[i * 4 + 1] = v;
      edgeImage.data[i * 4 + 2] = v;
      edgeImage.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(edgeImage, 0, 0);
    setEdgeUrl(canvas.toDataURL("image/png"));

    try {
      const result = await callApiPredict(f);
      setVehicle(result.vehicle);
      setConfidence(result.confidence);
    } catch (e) {
      // Keep UI usable even if API fails
      setVehicle({ make: "Unknown", model: "Unknown", class: "Unknown", tire: "N/A" });
      setConfidence(0.2);
    }
  };

  return (
    <div className="min-h-screen w-full">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <header className="mb-6">
          <h1 className="text-3xl font-bold">🔍 Tire Print Identifier</h1>
          <p className="text-slate-300">Upload a tire print image to predict the car model.</p>
        </header>

        <div className="card mb-4">
          <div className="flex items-center justify-between gap-4">
            <span className="badge">Display Options</span>
            <div className="flex gap-2">
              <button
                className={`px-3 py-2 rounded-xl border ${layout === "Vertical Cards" ? "border-sky-400 text-sky-300" : "border-slate-700 text-slate-300"}`}
                onClick={() => setLayout("Vertical Cards")}
              >
                Vertical Cards
              </button>
              <button
                className={`px-3 py-2 rounded-xl border ${layout === "Split Horizontal" ? "border-sky-400 text-sky-300" : "border-slate-700 text-slate-300"}`}
                onClick={() => setLayout("Split Horizontal")}
              >
                Split Horizontal
              </button>
            </div>
          </div>
        </div>

        <div className="card mb-4">
          <div className="badge mb-3">Upload</div>
          <div className="grid sm:grid-cols-[1fr_auto] gap-4 items-center">
            <input
              type="file"
              accept="image/*"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-slate-700 file:text-slate-200 hover:file:bg-slate-600"
            />
            {vehicle && (
              <div className="text-sm text-slate-300">Tip: Try different tire photos for varying outputs.</div>
            )}
          </div>
        </div>

        {file ? (
          layout === "Vertical Cards" ? (
            <div>
              <div className="card mb-4">
                <div className="badge mb-2">Input</div>
                {previewUrl && (
                  <NextImage src={previewUrl} alt="Uploaded Tire Print" width={1200} height={800} className="w-full h-auto rounded-xl" />
                )}
              </div>

              <div className="card mb-4">
                <div className="badge mb-2">Preprocessing</div>
                {edgeUrl && (
                  <NextImage src={edgeUrl} alt="Edge Preview" width={1200} height={800} className="w-full h-auto rounded-xl" />
                )}
              </div>

              <div className="card">
                <div className="badge mb-2">Prediction</div>
                {vehicle && confidence != null && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
                    <div className="md:col-span-2">
                      <div className="accent text-xl">Vehicle: {vehicle.make} {vehicle.model}</div>
                      <div className="mt-1 text-slate-400">Class: {vehicle.class} • Tire: {vehicle.tire}</div>
                      <div className="mt-2 text-slate-300">Confidence</div>
                      <div className="mt-1 h-2 bg-slate-700 rounded-full">
                        <div className="h-2 bg-sky-400 rounded-full" style={{ width: `${Math.round(confidence * 100)}%` }} />
                      </div>
                    </div>
                    <div className="md:col-span-1">
                      <div className="text-sm text-slate-400">Score</div>
                      <div className="text-2xl font-semibold">{confidence.toFixed(2)}</div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="card mb-4">
                  <div className="badge mb-2">Input</div>
                  {previewUrl && (
                    <NextImage src={previewUrl} alt="Uploaded Tire Print" width={1200} height={800} className="w-full h-auto rounded-xl" />
                  )}
                </div>
                <div className="card">
                  <div className="badge mb-2">Preprocessing</div>
                  {edgeUrl && (
                    <NextImage src={edgeUrl} alt="Edge Preview" width={1200} height={800} className="w-full h-auto rounded-xl" />
                  )}
                </div>
              </div>
              <div>
                <div className="card">
                  <div className="badge mb-2">Prediction</div>
                  {vehicle && confidence != null && (
                    <div>
                      <div className="accent text-xl">Vehicle: {vehicle.make} {vehicle.model}</div>
                      <div className="mt-1 text-slate-400">Class: {vehicle.class} • Tire: {vehicle.tire}</div>
                      <div className="mt-2 text-slate-300">Confidence</div>
                      <div className="mt-1 h-2 bg-slate-700 rounded-full">
                        <div className="h-2 bg-sky-400 rounded-full" style={{ width: `${Math.round(confidence * 100)}%` }} />
                      </div>
                      <div className="mt-3 text-slate-400 text-sm">Score</div>
                      <div className="text-2xl font-semibold">{confidence.toFixed(2)}</div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )
        ) : (
          <div className="card">
            <div className="accent text-lg">Upload a tire print to begin.</div>
          </div>
        )}

        {/* Hidden canvas used for edge processing */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    </div>
  );
}
