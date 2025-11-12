import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * OpenCV Playground — a HandBrake-style frontend for OpenCV.js
 *
 * Quick start (Vite React):
 *   npm create vite@latest opencv-playground -- --template react
 *   cd opencv-playground && npm install
 *   # In index.html add (before your bundle):
 *   # <script async src="https://docs.opencv.org/4.x/opencv.js"></script>
 *   # Optionally add Tailwind (not required). This component uses basic classes.
 *   # Put this file at src/OpenCVPlayground.jsx and render it from App.jsx
 *
 * Notes:
 * - Loads OpenCV.js from the global `cv` injected by the script tag above.
 * - Drag-and-drop or file picker to load an image.
 * - Build a processing pipeline by stacking operations; re-order, toggle, or delete.
 * - Parameters are auto-generated from each op's schema.
 * - Processing is debounced for smooth sliders.
 * - Memory is carefully managed by deleting Mats created during processing.
 */

function useDebouncedCallback(cb, delay = 120) {
  const to = useRef(null);
  return (...args) => {
    if (to.current) clearTimeout(to.current);
    to.current = setTimeout(() => cb(...args), delay);
  };
}

const DEFAULT_THEME = "light"; // flip to "dark" if you like

// ---- Operation Registry ----------------------------------------------------
// Each op defines a schema (auto-UI) and an apply(cv, src, params, ctx) -> Mat
// Return a NEW Mat and delete all temporaries. Do NOT delete `src`.

function odd(n) { return n % 2 === 0 ? n + 1 : n; }

const OP_REGISTRY = {
  Grayscale: {
    schema: {},
    apply: (cv, src) => {
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const out = new cv.Mat();
      cv.cvtColor(gray, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete();
      return out;
    },
  },
  GaussianBlur: {
    schema: {
      ksize: { type: "int", label: "Kernel", min: 1, max: 99, step: 2, default: 5, enforceOdd: true },
      sigma: { type: "float", label: "Sigma", min: 0, max: 30, step: 0.1, default: 0 },
    },
    apply: (cv, src, p) => {
      const k = p.enforceOdd ? odd(p.ksize) : p.ksize;
      const out = new cv.Mat();
      const ksize = new cv.Size(k, k);
      cv.GaussianBlur(src, out, ksize, p.sigma, p.sigma, cv.BORDER_DEFAULT);
      return out;
    },
  },
  Bilateral: {
    schema: {
      diameter: { type: "int", label: "Diameter", min: 1, max: 25, step: 1, default: 9 },
      sigmaColor: { type: "int", label: "SigmaColor", min: 1, max: 200, step: 1, default: 75 },
      sigmaSpace: { type: "int", label: "SigmaSpace", min: 1, max: 200, step: 1, default: 75 },
    },
    apply: (cv, src, p) => {
      // bilateralFilter requires 1- or 3-channel images. Convert RGBA -> RGB, filter, then back.
      const rgb = new cv.Mat();
      cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB, 0);
      const out3 = new cv.Mat();
      const d = Math.max(1, p.diameter|0);
      const sc = Math.max(1, p.sigmaColor|0);
      const ss = Math.max(1, p.sigmaSpace|0);
      cv.bilateralFilter(rgb, out3, d, sc, ss, cv.BORDER_DEFAULT);
      const out = new cv.Mat();
      cv.cvtColor(out3, out, cv.COLOR_RGB2RGBA, 0);
      rgb.delete(); out3.delete();
      return out;
    },
  },
  Canny: {
    schema: {
      t1: { type: "int", label: "Threshold 1", min: 0, max: 255, step: 1, default: 50 },
      t2: { type: "int", label: "Threshold 2", min: 0, max: 255, step: 1, default: 150 },
    },
    apply: (cv, src, p) => {
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const edges = new cv.Mat();
      cv.Canny(gray, edges, p.t1, p.t2);
      const out = new cv.Mat();
      cv.cvtColor(edges, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete();
      edges.delete();
      return out;
    },
  },
  Threshold: {
    schema: {
      thresh: { type: "int", label: "Threshold", min: 0, max: 255, step: 1, default: 127 },
      type: { type: "select", label: "Type", default: "BINARY", options: ["BINARY","BINARY_INV","TRUNC","TOZERO","TOZERO_INV"] },
    },
    apply: (cv, src, p) => {
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const outGray = new cv.Mat();
      const types = {
        BINARY: cv.THRESH_BINARY,
        BINARY_INV: cv.THRESH_BINARY_INV,
        TRUNC: cv.THRESH_TRUNC,
        TOZERO: cv.THRESH_TOZERO,
        TOZERO_INV: cv.THRESH_TOZERO_INV,
      };
      cv.threshold(gray, outGray, p.thresh, 255, types[p.type]);
      const out = new cv.Mat();
      cv.cvtColor(outGray, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete(); outGray.delete();
      return out;
    },
  },
  CLAHE: {
    schema: {
      clipLimit: { type: "float", label: "Clip Limit", min: 0.1, max: 10, step: 0.1, default: 2.0 },
      tile: { type: "int", label: "Tile Grid", min: 2, max: 16, step: 1, default: 8 },
    },
    apply: (cv, src, p) => {
      const lab = new cv.Mat();
      cv.cvtColor(src, lab, cv.COLOR_RGBA2Lab, 0);
      const labPlanes = new cv.MatVector();
      cv.split(lab, labPlanes);
      const L = labPlanes.get(0);
      const clahe = new cv.CLAHE(p.clipLimit, new cv.Size(p.tile, p.tile));
      const L2 = new cv.Mat();
      clahe.apply(L, L2);
      labPlanes.set(0, L2);
      const merged = new cv.Mat();
      cv.merge(labPlanes, merged);
      const out = new cv.Mat();
      cv.cvtColor(merged, out, cv.COLOR_Lab2RGBA, 0);
      // cleanup
      L.delete(); L2.delete(); labPlanes.delete(); lab.delete(); merged.delete(); clahe.delete();
      return out;
    },
  },
  Morphology: {
    schema: {
      mode: { type: "select", label: "Op", default: "ERODE", options: ["ERODE","DILATE"] },
      ksize: { type: "int", label: "Kernel", min: 1, max: 31, step: 2, default: 3, enforceOdd: true },
      iterations: { type: "int", label: "Iter", min: 1, max: 10, step: 1, default: 1 },
    },
    apply: (cv, src, p) => {
      const k = p.enforceOdd ? odd(p.ksize) : p.ksize;
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(k, k));
      const out = new cv.Mat();
      if (p.mode === "ERODE") cv.erode(src, out, kernel, new cv.Point(-1,-1), p.iterations);
      else cv.dilate(src, out, kernel, new cv.Point(-1,-1), p.iterations);
      kernel.delete();
      return out;
    },
  },
  Rotate: {
    schema: {
      angle: { type: "float", label: "Angle", min: -180, max: 180, step: 0.1, default: 0 },
      border: { type: "select", label: "Border", default: "REFLECT", options: ["REFLECT","REPLICATE","CONSTANT","TRANSPARENT"] },
    },
    apply: (cv, src, p) => {
      const center = new cv.Point(src.cols/2, src.rows/2);
      const M = cv.getRotationMatrix2D(center, p.angle, 1.0);
      const out = new cv.Mat();
      const borderMap = { REFLECT: cv.BORDER_REFLECT, REPLICATE: cv.BORDER_REPLICATE, CONSTANT: cv.BORDER_CONSTANT, TRANSPARENT: cv.BORDER_TRANSPARENT };
      cv.warpAffine(src, out, M, new cv.Size(src.cols, src.rows), cv.INTER_LINEAR, borderMap[p.border], new cv.Scalar());
      M.delete();
      return out;
    },
  },
  Resize: {
    schema: {
      scale: { type: "float", label: "Scale", min: 0.1, max: 5, step: 0.01, default: 1.0 },
      interp: { type: "select", label: "Interp", default: "LINEAR", options: ["NEAREST","LINEAR","CUBIC","AREA","LANCZOS4"] },
    },
    apply: (cv, src, p) => {
      const out = new cv.Mat();
      const map = { NEAREST: cv.INTER_NEAREST, LINEAR: cv.INTER_LINEAR, CUBIC: cv.INTER_CUBIC, AREA: cv.INTER_AREA, LANCZOS4: cv.INTER_LANCZOS4 };
      cv.resize(src, out, new cv.Size(0,0), p.scale, p.scale, map[p.interp]);
      return out;
    },
  },
  Sharpen: {
    schema: {
      amount: { type: "float", label: "Amount", min: 0, max: 3, step: 0.05, default: 1.0 },
      radius: { type: "int", label: "Radius", min: 1, max: 25, step: 2, default: 5, enforceOdd: true },
    },
    apply: (cv, src, p) => {
      const rad = p.enforceOdd ? odd(p.radius) : p.radius;
      const blurred = new cv.Mat();
      cv.GaussianBlur(src, blurred, new cv.Size(rad, rad), 0, 0, cv.BORDER_DEFAULT);
      const out = new cv.Mat();
      // Unsharp mask: out = src*(1+a) + blurred*(-a)
      cv.addWeighted(src, 1 + p.amount, blurred, -p.amount, 0, out);
      blurred.delete();
      return out;
    },
  },
  BrightnessContrast: {
    schema: {
      alpha: { type: "float", label: "Contrast α", min: 0, max: 3, step: 0.01, default: 1.0 },
      beta:  { type: "int", label: "Brightness β", min: -100, max: 100, step: 1, default: 0 },
    },
    apply: (cv, src, p) => {
      const out = new cv.Mat();
      cv.addWeighted(src, p.alpha, src, 0, p.beta, out); // dst = α*src + β
      return out;
    },
  },
  Gamma: {
    schema: {
      gamma: { type: "float", label: "Gamma", min: 0.1, max: 5, step: 0.01, default: 1.0 },
    },
    apply: (cv, src, p) => {
      const g = Math.max(0.1, p.gamma);
      const lutArr = new Uint8Array(256);
      for (let i = 0; i < 256; i++) lutArr[i] = Math.min(255, Math.round(255 * Math.pow(i / 255, 1.0 / g)));
      const lut = cv.matFromArray(1, 256, cv.CV_8UC1, lutArr);
      const planes = new cv.MatVector();
      cv.split(src, planes);
      let b = planes.get(0), gch = planes.get(1), r = planes.get(2), a = planes.size() > 3 ? planes.get(3) : null;
      cv.LUT(b, lut, b); cv.LUT(gch, lut, gch); cv.LUT(r, lut, r);
      const mv = new cv.MatVector(); mv.push_back(b); mv.push_back(gch); mv.push_back(r); if (a) mv.push_back(a);
      const out = new cv.Mat(); cv.merge(mv, out);
      b.delete(); gch.delete(); r.delete(); if (a) a.delete(); mv.delete(); planes.delete(); lut.delete();
      return out;
    },
  },
  MedianBlur: {
    schema: { ksize: { type: "int", label: "Kernel", min: 1, max: 31, step: 2, default: 5, enforceOdd: true } },
    apply: (cv, src, p) => {
      const rgb = new cv.Mat(); cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB, 0);
      const out3 = new cv.Mat(); const k = p.enforceOdd ? odd(p.ksize) : p.ksize;
      cv.medianBlur(rgb, out3, k);
      const out = new cv.Mat(); cv.cvtColor(out3, out, cv.COLOR_RGB2RGBA, 0);
      rgb.delete(); out3.delete();
      return out;
    },
  },
  BoxBlur: {
    schema: { ksize: { type: "int", label: "Kernel", min: 1, max: 99, step: 2, default: 3, enforceOdd: true } },
    apply: (cv, src, p) => {
      const out = new cv.Mat(); const k = p.enforceOdd ? odd(p.ksize) : p.ksize;
      cv.blur(src, out, new cv.Size(k, k), new cv.Point(-1, -1), cv.BORDER_DEFAULT);
      return out;
    },
  },
  Sobel: {
    schema: { ksize: { type: "int", label: "Kernel", min: 1, max: 7, step: 2, default: 3 } },
    apply: (cv, src, p) => {
      const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const dx = new cv.Mat(), dy = new cv.Mat();
      cv.Sobel(gray, dx, cv.CV_16S, 1, 0, p.ksize, 1, 0, cv.BORDER_DEFAULT);
      cv.Sobel(gray, dy, cv.CV_16S, 0, 1, p.ksize, 1, 0, cv.BORDER_DEFAULT);
      const adx = new cv.Mat(), ady = new cv.Mat();
      cv.convertScaleAbs(dx, adx); cv.convertScaleAbs(dy, ady);
      const mag = new cv.Mat(); cv.addWeighted(adx, 0.5, ady, 0.5, 0, mag);
      const out = new cv.Mat(); cv.cvtColor(mag, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete(); dx.delete(); dy.delete(); adx.delete(); ady.delete(); mag.delete();
      return out;
    },
  },
  Laplacian: {
    schema: { ksize: { type: "int", label: "Kernel", min: 1, max: 7, step: 2, default: 3 } },
    apply: (cv, src, p) => {
      const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const lap16 = new cv.Mat(); cv.Laplacian(gray, lap16, cv.CV_16S, p.ksize, 1, 0, cv.BORDER_DEFAULT);
      const abs = new cv.Mat(); cv.convertScaleAbs(lap16, abs);
      const out = new cv.Mat(); cv.cvtColor(abs, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete(); lap16.delete(); abs.delete();
      return out;
    },
  },
  EqualizeHist: {
    schema: {},
    apply: (cv, src) => {
      const ycb = new cv.Mat(); cv.cvtColor(src, ycb, cv.COLOR_RGBA2YCrCb, 0);
      const planes = new cv.MatVector(); cv.split(ycb, planes);
      const Y = planes.get(0); const Ye = new cv.Mat(); cv.equalizeHist(Y, Ye); planes.set(0, Ye);
      const merged = new cv.Mat(); cv.merge(planes, merged);
      const out = new cv.Mat(); cv.cvtColor(merged, out, cv.COLOR_YCrCb2RGBA, 0);
      Y.delete(); Ye.delete(); planes.delete(); ycb.delete(); merged.delete();
      return out;
    },
  },
  HSVAdjust: {
    schema: {
      hue: { type: "int", label: "Hue Shift", min: -180, max: 180, step: 1, default: 0 },
      sat: { type: "float", label: "Sat ×", min: 0, max: 3, step: 0.01, default: 1.0 },
      val: { type: "float", label: "Val ×", min: 0, max: 3, step: 0.01, default: 1.0 },
    },
    apply: (cv, src, p) => {
      const hsv = new cv.Mat(); cv.cvtColor(src, hsv, cv.COLOR_RGBA2HSV, 0);
      const planes = new cv.MatVector(); cv.split(hsv, planes);
      const H = planes.get(0), S = planes.get(1), V = planes.get(2);
      const add = Math.round((((p.hue % 360) + 360) % 360) / 2);
      const Hd = H.data; for (let i = 0; i < Hd.length; i++) Hd[i] = (Hd[i] + add) % 180;
      const sScale = Math.max(0, p.sat), Sd = S.data; for (let i = 0; i < Sd.length; i++) Sd[i] = Math.min(255, Math.round(Sd[i] * sScale));
      const vScale = Math.max(0, p.val), Vd = V.data; for (let i = 0; i < Vd.length; i++) Vd[i] = Math.min(255, Math.round(Vd[i] * vScale));
      planes.set(0, H); planes.set(1, S); planes.set(2, V);
      const merged = new cv.Mat(); cv.merge(planes, merged);
      const out = new cv.Mat(); cv.cvtColor(merged, out, cv.COLOR_HSV2RGBA, 0);
      H.delete(); S.delete(); V.delete(); planes.delete(); hsv.delete(); merged.delete();
      return out;
    },
  },
  Invert: {
    schema: {},
    apply: (cv, src) => {
      const planes = new cv.MatVector(); cv.split(src, planes);
      let b = planes.get(0), g = planes.get(1), r = planes.get(2), a = planes.size() > 3 ? planes.get(3) : null;
      cv.bitwise_not(b, b); cv.bitwise_not(g, g); cv.bitwise_not(r, r);
      const mv = new cv.MatVector(); mv.push_back(b); mv.push_back(g); mv.push_back(r); if (a) mv.push_back(a);
      const out = new cv.Mat(); cv.merge(mv, out);
      b.delete(); g.delete(); r.delete(); if (a) a.delete(); mv.delete(); planes.delete();
      return out;
    },
  },
  Flip: {
    schema: { mode: { type: "select", label: "Mode", default: "H", options: ["H","V","HV"] } },
    apply: (cv, src, p) => {
      const out = new cv.Mat();
      const code = p.mode === 'H' ? 1 : (p.mode === 'V' ? 0 : -1);
      cv.flip(src, out, code);
      return out;
    },
  },
  AdaptiveThreshold: {
    schema: {
      method: { type: "select", label: "Method", default: "MEAN", options: ["MEAN","GAUSSIAN"] },
      type: { type: "select", label: "Type", default: "BINARY", options: ["BINARY","BINARY_INV"] },
      block: { type: "int", label: "Block", min: 3, max: 99, step: 2, default: 11, enforceOdd: true },
      C: { type: "int", label: "C", min: -20, max: 20, step: 1, default: 2 },
    },
    apply: (cv, src, p) => {
      const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const outGray = new cv.Mat();
      const meth = p.method === 'MEAN' ? cv.ADAPTIVE_THRESH_MEAN_C : cv.ADAPTIVE_THRESH_GAUSSIAN_C;
      const typ = p.type === 'BINARY' ? cv.THRESH_BINARY : cv.THRESH_BINARY_INV;
      const block = p.enforceOdd ? odd(p.block) : p.block;
      cv.adaptiveThreshold(gray, outGray, 255, meth, typ, block, p.C);
      const out = new cv.Mat(); cv.cvtColor(outGray, out, cv.COLOR_GRAY2RGBA, 0);
      gray.delete(); outGray.delete();
      return out;
    },
  },
  ColorMap: {
    schema: { map: { type: "select", label: "Map", default: "JET", options: ["AUTUMN","BONE","JET","WINTER","RAINBOW","OCEAN","SUMMER","SPRING","COOL","HSV","PINK","HOT","PARULA","MAGMA","INFERNO","PLASMA","VIRIDIS","CIVIDIS","TURBO"] } },
    apply: (cv, src, p) => {
      const gray = new cv.Mat(); cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      const color = new cv.Mat();
      const cmap = {
        AUTUMN: cv.COLORMAP_AUTUMN, BONE: cv.COLORMAP_BONE, JET: cv.COLORMAP_JET, WINTER: cv.COLORMAP_WINTER,
        RAINBOW: cv.COLORMAP_RAINBOW, OCEAN: cv.COLORMAP_OCEAN, SUMMER: cv.COLORMAP_SUMMER, SPRING: cv.COLORMAP_SPRING,
        COOL: cv.COLORMAP_COOL, HSV: cv.COLORMAP_HSV, PINK: cv.COLORMAP_PINK, HOT: cv.COLORMAP_HOT,
        PARULA: cv.COLORMAP_PARULA, MAGMA: cv.COLORMAP_MAGMA, INFERNO: cv.COLORMAP_INFERNO,
        PLASMA: cv.COLORMAP_PLASMA, VIRIDIS: cv.COLORMAP_VIRIDIS, CIVIDIS: cv.COLORMAP_CIVIDIS, TURBO: cv.COLORMAP_TURBO,
      };
      const code = (cmap[p.map] !== undefined) ? cmap[p.map] : cv.COLORMAP_JET;
      cv.applyColorMap(gray, color, code);
      const out = new cv.Mat(); cv.cvtColor(color, out, cv.COLOR_BGR2RGBA, 0);
      gray.delete(); color.delete();
      return out;
    },
  },
  BlendWithOriginal: {
    schema: {
      alpha: { type: "float", label: "Alpha", min: 0, max: 1, step: 0.01, default: 0.5 },
    },
    apply: (cv, src, p, ctx) => {
      const out = new cv.Mat();
      // src is current pipeline image; ctx.original is the original Mat (RGBA)
      cv.addWeighted(src, p.alpha, ctx.original, 1 - p.alpha, 0, out);
      return out;
    },
  },
};

const OP_ORDER = Object.keys(OP_REGISTRY);

function defaultParams(schema) {
  const p = {};
  for (const [k, v] of Object.entries(schema)) p[k] = v.default;
  return p;
}

let opAutoId = 1;

export default function OpenCVPlayground() {
  const [cvReady, setCvReady] = useState(false);
  const [theme] = useState(DEFAULT_THEME);
  const [ops, setOps] = useState([]); // {id, type, enabled, params}
  const [live, setLive] = useState(true);
  const [status, setStatus] = useState("Load an image to begin");

  const origCanvasRef = useRef(null);
  const procCanvasRef = useRef(null);
  const originalMatRef = useRef(null); // cv.Mat of original RGBA
  const processingRef = useRef(false);
  const processedImageDataRef = useRef(null); // snapshot of processed for fast re-draw

  // File & download state
  const fileInputRef = useRef(null);
  const [fileLabel, setFileLabel] = useState("Open image…");
  const [dlFormat, setDlFormat] = useState("png");

  // Compare/UI state
  const [compareMode, setCompareMode] = useState("slider"); // 'side-by-side' | 'slider'
  const [sliderX, setSliderX] = useState(0.5); // 0..1 divider for slider mode
  const draggingRef = useRef(false);
  const [peekOriginal, setPeekOriginal] = useState(false);

  // "Add op" selector state (controlled)
  const [opToAdd, setOpToAdd] = useState(OP_ORDER[0]);

  // Load OpenCV.js from global
  useEffect(() => {
    let mounted = true;
    function readyCheck() {
      try {
        if (window.cv && window.cv.Mat) {
          if (window.cv['onRuntimeInitialized']) {
            // If runtime still initializing, hook it once.
            const cv = window.cv;
            const prev = cv.onRuntimeInitialized;
            cv.onRuntimeInitialized = () => {
              prev && prev();
              if (mounted) setCvReady(true);
            };
          } else {
            setCvReady(true);
          }
        }
      } catch {}
    }
    // If already loaded, flag ready; otherwise wait until script runs.
    const iv = setInterval(readyCheck, 100);
    readyCheck();
    return () => { mounted = false; clearInterval(iv); };
  }, []);

  // Cleanup originalMat on unmount
  useEffect(() => () => {
    try { originalMatRef.current?.delete?.(); } catch {}
  }, []);

  const debouncedRun = useDebouncedCallback(() => runPipeline(), 120);

  // Spacebar A/B peek
  useEffect(() => {
    function onKeyDown(e){
      if (e.code === 'Space' && !e.repeat) setPeekOriginal(true);
    }
    function onKeyUp(e){
      if (e.code === 'Space') setPeekOriginal(false);
    }
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // Redraw overlay when slider/peek changes
  useEffect(() => {
    refreshCompareOverlay();
  }, [sliderX, compareMode, peekOriginal]);

  // Re-run pipeline when ops change or Live toggles
  useEffect(() => {
    if (!cvReady) return;
    if (live) debouncedRun();
  }, [ops, live, cvReady]);

  // Keep compare overlay crisp on resize
  useEffect(() => {
    const onResize = () => refreshCompareOverlay();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  function onFileChosen(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const oc = origCanvasRef.current;
      oc.width = img.naturalWidth; oc.height = img.naturalHeight;
      const ctx = oc.getContext('2d');
      ctx.drawImage(img, 0, 0);

      // Replace originalMat
      const cv = window.cv;
      const newMat = cv.imread(oc);
      if (originalMatRef.current) try { originalMatRef.current.delete(); } catch {}
      originalMatRef.current = newMat; // RGBA
      processedImageDataRef.current = null;

      // Resize processed canvas to match
      const pc = procCanvasRef.current;
      pc.width = oc.width; pc.height = oc.height;

      setFileLabel(file.name || "Open image…");
      setStatus(`${img.naturalWidth}×${img.naturalHeight} loaded`);
      if (live) debouncedRun(); else drawOriginal();
      URL.revokeObjectURL(url);
    };
    img.onerror = () => setStatus("Failed to load image");
    img.src = url;
  }

  function drawOriginal() {
    const oc = origCanvasRef.current;
    const pc = procCanvasRef.current;
    const pctx = pc.getContext('2d');
    pctx.clearRect(0,0,pc.width,pc.height);
    pctx.drawImage(oc, 0, 0);
  }

  // Draw processed snapshot + overlay original for compare slider
  function refreshCompareOverlay(){
    const pc = procCanvasRef.current;
    const oc = origCanvasRef.current;
    if (!pc || !oc) return;
    const ctx = pc.getContext('2d');

    // If peeking, show full original over processed
    if (peekOriginal) {
      ctx.clearRect(0,0,pc.width,pc.height);
      ctx.drawImage(oc, 0, 0, pc.width, pc.height);
      return;
    }

    if (compareMode !== 'slider') return; // side-by-side uses separate canvases

    // restore processed frame
    if (processedImageDataRef.current) {
      try { ctx.putImageData(processedImageDataRef.current, 0, 0); } catch {}
    }

    // clip and draw original on the left side
    const x = Math.max(0, Math.min(1, sliderX)) * pc.width;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, x, pc.height);
    ctx.clip();
    ctx.drawImage(oc, 0, 0, pc.width, pc.height);
    ctx.restore();

    // draw divider
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, pc.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ffffff';
    ctx.stroke();
    ctx.restore();
  }

  function onProcPointerDown(e){
    if (compareMode !== 'slider') return;
    const rect = procCanvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    setSliderX(Math.max(0, Math.min(1, x)));
    draggingRef.current = true;
  }
  function onProcPointerMove(e){
    if (!draggingRef.current || compareMode !== 'slider') return;
    const rect = procCanvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    setSliderX(Math.max(0, Math.min(1, x)));
  }
  function onProcPointerUp(){ draggingRef.current = false; }

  function runPipeline() {
    const cv = window.cv;
    const original = originalMatRef.current;
    const outCanvas = procCanvasRef.current;
    if (!cvReady || !original || !outCanvas) return;
    if (processingRef.current) return; // avoid overlap
    processingRef.current = true;

    setStatus("Processing…");

    try {
      const enabledOps = ops.filter(o => o.enabled);
      let current = original.clone();
      const ctx = { original };
      if (enabledOps.length === 0) {
        cv.imshow(outCanvas, current);
      } else {
        for (const op of enabledOps) {
          const def = OP_REGISTRY[op.type];
          if (!def) continue;
          const next = def.apply(cv, current, op.params, ctx);
          current.delete();
          current = next;
        }
        cv.imshow(outCanvas, current);
      }
      const ctx2 = outCanvas.getContext('2d');
      try { processedImageDataRef.current = ctx2.getImageData(0, 0, outCanvas.width, outCanvas.height); } catch {}
      // Draw compare overlay (slider/peek)
      refreshCompareOverlay();
      current.delete();
      setStatus(`OK • ${ops.filter(o=>o.enabled).length} ops`);
    } catch (e) {
      console.error(e);
      setStatus("Error: " + (e?.message || String(e)));
    } finally {
      processingRef.current = false;
    }
  }

  function addOp(type) {
    const def = OP_REGISTRY[type];
    if (!def) return;
    const item = { id: opAutoId++, type, enabled: true, params: defaultParams(def.schema) };
    setOps(prev => [...prev, item]);
    if (live) debouncedRun();
  }

  function updateOp(id, patch) {
    setOps(prev => prev.map(o => o.id === id ? { ...o, ...patch } : o));
    if (live && (patch.params || patch.enabled !== undefined)) debouncedRun();
  }

  function removeOp(id) {
    setOps(prev => prev.filter(o => o.id !== id));
    if (live) debouncedRun();
  }

  function moveOp(id, dir) {
    setOps(prev => {
      const idx = prev.findIndex(o => o.id === id);
      if (idx < 0) return prev;
      const arr = prev.slice();
      const target = dir === "up" ? idx - 1 : idx + 1;
      if (target < 0 || target >= arr.length) return prev;
      const [item] = arr.splice(idx, 1);
      arr.splice(target, 0, item);
      return arr;
    });
    if (live) debouncedRun();
  }

  function exportPipeline() {
    const json = JSON.stringify(ops, null, 2);
    try { navigator.clipboard?.writeText(json).catch(() => {}); } catch {}
    alert(`Pipeline JSON copied to clipboard.\n\n${json}`);
  }

  function downloadProcessed(fmt = dlFormat) {
    const pc = procCanvasRef.current;
    if (!pc || !originalMatRef.current) return;
    if (!live) runPipeline();

    const off = document.createElement('canvas');
    off.width = pc.width; off.height = pc.height;
    const octx = off.getContext('2d');
    if (processedImageDataRef.current) {
      try { octx.putImageData(processedImageDataRef.current, 0, 0); }
      catch { octx.drawImage(pc, 0, 0); }
    } else {
      octx.drawImage(pc, 0, 0);
    }

    const mime = fmt === 'jpeg' ? 'image/jpeg' : 'image/png';
    const ext = fmt === 'jpeg' ? 'jpg' : 'png';
    const trigger = (blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `opencv-playground.${ext}`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    };
    if (off.toBlob) off.toBlob(trigger, mime, fmt === 'jpeg' ? 0.92 : undefined);
    else {
      const dataURL = off.toDataURL(mime);
      const a = document.createElement('a');
      a.href = dataURL; a.download = `opencv-playground.${ext}`; a.click();
    }
  }

  function importPipeline() {
    const json = prompt("Paste pipeline JSON");
    if (!json) return;
    try {
      const arr = JSON.parse(json);
      if (!Array.isArray(arr)) throw new Error("Invalid pipeline");
      // Ensure ids unique and track the highest numeric id encountered
      let maxId = -1;
      for (const o of arr) {
        let numericId = Number.parseInt(o?.id, 10);
        if (!Number.isFinite(numericId)) {
          o.id = opAutoId++;
          numericId = o.id;
        } else {
          o.id = numericId;
        }
        if (numericId > maxId) maxId = numericId;
      }
      opAutoId = Math.max(opAutoId, maxId + 1);
      setOps(arr);
      if (live) debouncedRun();
    } catch (e) {
      alert("Failed to import: " + (e?.message || String(e)));
    }
  }

  const dropHandlers = useMemo(() => ({
    onDragOver: (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; },
    onDrop: (e) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      onFileChosen(f);
    },
  }), [onFileChosen]);

  const disabledUI = !cvReady;

  return (
    <div className={`min-h-screen ${theme === "dark" ? 'bg-neutral-900 text-neutral-100' : 'bg-white text-neutral-900'} flex flex-col`} style={{ colorScheme: 'light' }}> 
      <header className="border-b border-neutral-300/50 px-4 py-2 flex items-center gap-3">
        <h1 className="text-xl font-semibold">OpenCV Playground</h1>
        <span className="text-xs opacity-70">HandBrake-style frontend for OpenCV</span>
        <div className="ml-auto flex items-center gap-2">
          <label className="inline-flex items-center gap-2 text-sm">
            <input type="checkbox" checked={live} onChange={e => setLive(e.target.checked)} /> Live
          </label>
          <label className="inline-flex items-center gap-2 text-sm ml-3">
            <span>Compare</span>
            <select className="border rounded p-1 bg-white text-neutral-900" value={compareMode} onChange={e=>setCompareMode(e.target.value)}>
              <option value="side-by-side">Side-by-side</option>
              <option value="slider">Slider</option>
            </select>
            <span className="opacity-60">(hold Space to peek original)</span>
          </label>
          <button className="px-3 py-1 rounded bg-blue-600 text-white disabled:opacity-50" disabled={!cvReady} onClick={runPipeline}>Apply</button>
          <div className="flex items-center gap-2">
            <select className="border rounded p-1 bg-white text-neutral-900" value={dlFormat} onChange={e=>setDlFormat(e.target.value)}>
              <option value="png">PNG</option>
              <option value="jpeg">JPEG</option>
            </select>
            <button className="px-3 py-1 rounded bg-neutral-800 text-white disabled:opacity-50" disabled={!cvReady || !originalMatRef.current} onClick={()=>downloadProcessed()}>Download</button>
          </div>
          <button className="px-3 py-1 rounded bg-neutral-800 text-white" onClick={exportPipeline}>Export</button>
          <button className="px-3 py-1 rounded bg-neutral-800 text-white" onClick={importPipeline}>Import</button>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-12 gap-3 p-3">
        {/* Left panel: Image load + previews */}
        <section className="col-span-8 flex flex-col gap-3">
          <div className="border rounded-lg p-3 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden"
                     onChange={e => onFileChosen(e.target.files?.[0])} disabled={disabledUI} />
              <button className="px-3 py-1 rounded bg-neutral-800 text-white disabled:opacity-50"
                      onClick={()=>fileInputRef.current?.click()} disabled={disabledUI}>Open</button>
              <span className="text-sm opacity-80 truncate max-w-[28ch]" title={fileLabel}>{fileLabel}</span>
              <span className="text-sm opacity-70">{cvReady ? status : 'Loading OpenCV… (ensure <script src=opencv.js>)'}</span>
            </div>
            <div className="text-xs opacity-70">{ops.length} ops</div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="border rounded-lg overflow-hidden">
              <div className="px-2 py-1 text-xs bg-neutral-200/60">Original</div>
              <div className="relative" {...dropHandlers}>
                <canvas ref={origCanvasRef} className="max-w-full h-auto block" />
                <div className="absolute inset-0 pointer-events-none border-2 border-dashed border-transparent" />
              </div>
            </div>
            <div className="border rounded-lg overflow-hidden">
              <div className="px-2 py-1 text-xs bg-neutral-200/60">Processed</div>
              <canvas ref={procCanvasRef} className="max-w-full h-auto block"
                      onPointerDown={onProcPointerDown}
                      onPointerMove={onProcPointerMove}
                      onPointerUp={onProcPointerUp}
                      onPointerLeave={onProcPointerUp}
              />
            </div>
          </div>
        </section>

        {/* Right panel: Operation stack */}
        <aside className="col-span-4 flex flex-col gap-3">
          <div className="border rounded-lg p-2 flex gap-2 items-center">
            <select className="flex-1 border rounded p-2 bg-white text-neutral-900"
              value={opToAdd}
              onChange={e=>setOpToAdd(e.target.value)}
              disabled={disabledUI}>
              {OP_ORDER.map(name => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
            <button className="px-3 py-2 rounded bg-green-600 text-white disabled:opacity-50" disabled={disabledUI} onClick={()=>addOp(opToAdd)}>+ Add</button>
          </div>

          <div className="flex-1 overflow-auto flex flex-col gap-2">
            {ops.length === 0 && (
              <div className="text-sm opacity-70 p-2">Add operations to build your pipeline. Drag-and-drop an image or use the picker.</div>
            )}

            {ops.map(op => (
              <OpCard key={op.id} op={op}
                onChange={(patch)=>updateOp(op.id, patch)}
                onRemove={()=>removeOp(op.id)}
                onMoveUp={()=>moveOp(op.id, 'up')}
                onMoveDown={()=>moveOp(op.id, 'down')}
              />
            ))}
          </div>
        </aside>
      </main>

      <footer className="px-4 py-2 text-xs opacity-70 border-t border-neutral-300/50">
        {cvReady ? status : 'OpenCV not ready'}
      </footer>
    </div>
  );
}

function OpCard({ op, onChange, onRemove, onMoveUp, onMoveDown }) {
  const def = OP_REGISTRY[op.type];
  return (
    <div className="border rounded-lg p-2 bg-white text-neutral-900">
      <div className="flex items-center gap-2">
        <label className="inline-flex items-center gap-2 text-sm">
          <input type="checkbox" checked={op.enabled} onChange={e=>onChange({ enabled: e.target.checked })} />
          <span className="font-medium">{op.type}</span>
        </label>
        <div className="ml-auto flex items-center gap-1">
          <button className="px-2 py-1 text-xs border rounded bg-neutral-800 text-white hover:bg-neutral-700" onClick={onMoveUp}>↑</button>
          <button className="px-2 py-1 text-xs border rounded bg-neutral-800 text-white hover:bg-neutral-700" onClick={onMoveDown}>↓</button>
          <button className="px-2 py-1 text-xs border rounded bg-red-600 text-white" onClick={onRemove}>Delete</button>
        </div>
      </div>

      {def && (
        <div className="mt-2 grid gap-2">
          {Object.entries(def.schema).map(([name, spec]) => (
            <ParamControl key={name} name={name} spec={spec} value={op.params[name]} onChange={(v)=>onChange({ params: { ...op.params, [name]: v } })} />
          ))}
        </div>
      )}
    </div>
  );
}

function ParamControl({ name, spec, value, onChange }) {
  const id = `param-${name}-${Math.random().toString(36).slice(2,7)}`;
  const label = spec.label || name;
  if (spec.type === 'select') {
    return (
      <label className="text-sm flex items-center gap-2">
        <span className="w-28 opacity-80">{label}</span>
        <select className="flex-1 border rounded p-1 bg-white text-neutral-900" value={value} onChange={e=>onChange(e.target.value)}>
          {spec.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
        </select>
      </label>
    );
  }
  if (spec.type === 'int' || spec.type === 'float') {
    const step = spec.step ?? (spec.type === 'int' ? 1 : 0.01);
    const min = spec.min ?? 0; const max = spec.max ?? 100;
    function toNumber(v){ return spec.type === 'int' ? parseInt(v,10) : parseFloat(v); }
    function clamp(v){ if (v < min) v = min; if (v > max) v = max; return v; }
    function maybeOdd(v){ return spec.enforceOdd ? odd(v|0) : v; }
    return (
      <div className="text-sm">
        <div className="flex items-center gap-2">
          <span className="w-28 opacity-80">{label}</span>
          <input id={id} type="range" min={min} max={max} step={step} value={value}
                 onChange={e=>onChange(maybeOdd(toNumber(e.target.value)))} className="flex-1" />
          <input type="number" min={min} max={max} step={step} value={value}
                 onChange={e=>onChange(maybeOdd(clamp(toNumber(e.target.value))))}
                 className="w-20 border rounded p-1 bg-white text-neutral-900" />
        </div>
      </div>
    );
  }
  if (spec.type === 'boolean') {
    return (
      <label className="text-sm inline-flex items-center gap-2">
        <input type="checkbox" checked={!!value} onChange={e=>onChange(e.target.checked)} />
        <span>{label}</span>
      </label>
    );
  }
  return null;
}
