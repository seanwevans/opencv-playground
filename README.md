# OpenCV Playground
HandBrake‑style frontend for **OpenCV.js**. Load an image, stack OpenCV operations, tweak parameters with live previews, and export the pipeline or the processed image.

---

## Using the App
- **Open**: pick a file or drag‑and‑drop into the *Original* panel
- **Live**: when enabled, the pipeline re-runs on every parameter change
- **Apply**: run the pipeline once (useful with Live off)
- **Compare**:
  - *Slider*: drag divider; hold **Space** to show the unprocessed original
  - *Side‑by‑side*: shows original and processed in separate panels
- **Download**: choose **PNG/JPEG**, then **Download** to save the processed image (no overlays)
- **Export**: copy the pipeline JSON to clipboard
- **Import**: paste pipeline JSON to restore a workflow

### Supported Operations
- **Grayscale**
- **GaussianBlur** (kernel, sigma)
- **Bilateral** *(RGBA → RGB → filter → RGBA)*
- **Canny** (t1, t2)
- **Threshold** (binary, inv, trunc, tozero)
- **CLAHE** (clip limit, tile grid)
- **Morphology** (erode/dilate, kernel, iterations)
- **Rotate** (angle, border policy)
- **Resize** (scale, interpolation)
- **Sharpen** (unsharp mask: amount, radius)
- **BrightnessContrast** (α, β)
- **Gamma** (per‑channel LUT; preserves alpha)
- **MedianBlur** *(RGBA → RGB → filter → RGBA)*
- **BoxBlur**
- **Sobel** (edge magnitude)
- **Laplacian**
- **EqualizeHist** (Y channel)
- **HSVAdjust** (hue shift, sat×, val×)
- **Invert** (RGB; alpha preserved)
- **Flip** (H, V, HV)
- **AdaptiveThreshold** (mean/gaussian; block; C)
- **ColorMap** (Jet/Hot/Viridis/Turbo/…)
- **BlendWithOriginal** (mix processed with original)

---

Run:
```bash
npm run dev
```
Build:
```bash
npm run build
```

---

## Add a New Op
Each operation lives in `OP_REGISTRY`. Minimal example:
```js
const OP_REGISTRY = {
  MyOp: {
    schema: { strength: { type: "float", label: "Strength", min: 0, max: 1, step: 0.01, default: 0.5 } },
    apply: (cv, src, p, ctx) => {
      // src is RGBA cv.Mat (DO NOT delete)
      // return a NEW cv.Mat; delete all temporaries you create
      const out = new cv.Mat();
      cv.addWeighted(src, 1 + p.strength, src, -p.strength, 0, out);
      return out;
    },
  },
  // ...other ops
};
```
