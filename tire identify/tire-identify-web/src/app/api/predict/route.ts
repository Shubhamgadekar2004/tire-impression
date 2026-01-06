import { NextResponse } from "next/server";

export const runtime = "nodejs";

type VehicleInfo = {
  make: string;
  model: string;
  class: string;
  tire: string;
};

const VEHICLES: VehicleInfo[] = [
  { make: "Toyota", model: "Camry", class: "Sedan", tire: "215/55R17" },
  { make: "Honda", model: "Civic", class: "Compact", tire: "215/50R17" },
  { make: "Ford", model: "F-150", class: "Pickup", tire: "275/65R18" },
  { make: "BMW", model: "3 Series", class: "Sports Sedan", tire: "225/45R18" },
  { make: "Audi", model: "Q5", class: "SUV", tire: "235/55R19" },
];

function computeGrayscale(data: Uint8Array) {
  const gray = new Float32Array(data.length / 4);
  for (let i = 0, g = 0; i < data.length; i += 4, g++) {
    const r = data[i];
    const b = data[i + 1];
    const gb = data[i + 2];
    gray[g] = (0.299 * r + 0.587 * b + 0.114 * gb) / 255;
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

function featureSignature(gray: Float32Array) {
  const n = gray.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += gray[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = gray[i] - mean;
    variance += d * d;
  }
  variance /= n;
  const std = Math.sqrt(variance);
  let skew = 0;
  for (let i = 0; i < n; i++) {
    const z = (gray[i] - mean) / (std + 1e-6);
    skew += z * z * z;
  }
  skew /= n;
  return { mean, std, skew };
}

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("image");
    if (!(file instanceof Blob)) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 });
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Decode using Web APIs via OffscreenCanvas if on-edge, but here we fallback to a simple image decode with sharp.
    // Serverless-friendly approach: use sharp to get raw RGBA.
    const sharp = (await import("sharp")).default;
    const img = sharp(buffer).ensureAlpha();
    const meta = await img.metadata();
    const width = Math.min(900, meta.width || 512);
    const resized = img.resize({ width });
    const raw = await resized.raw().toBuffer({ resolveWithObject: true });
    const height = raw.info.height;
    const rgba = raw.data;

    const gray = computeGrayscale(rgba);
    const edges = sobelEdges(gray, width, height);
    const { mean, std, skew } = featureSignature(gray);

    // Deterministic mapping
    const idx = Math.abs(Math.floor(mean * 7 + std * 11 + skew * 13)) % VEHICLES.length;
    const vehicle = VEHICLES[idx];
    let confidence = 1 / (1 + Math.exp(-std * 2.0));
    confidence = Math.max(0.2, Math.min(0.98, confidence));

    // Basic forensic-oriented metadata
    const features = {
      mean: Number(mean.toFixed(6)),
      std: Number(std.toFixed(6)),
      skew: Number(skew.toFixed(6)),
      width,
      height,
    };

    return NextResponse.json({ vehicle, confidence, features });
  } catch (err: any) {
    return NextResponse.json({ error: err?.message || "Predict failed" }, { status: 500 });
  }
}
