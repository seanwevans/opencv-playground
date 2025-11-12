# OpenCV Playground
HandBrake‑style frontend for **OpenCV.js**. Load an image, stack OpenCV operations, tweak parameters with live previews, and export the pipeline or the processed image.

---

## Features
- Drag‑and‑drop or **Open** button to load images
- **Live** processing with debounced sliders (smooth UI)
- **Compare**: *Side‑by‑side* or draggable **Slider**; hold **Space** to “peek” the original
- Pipeline editor: add, reorder, toggle, or delete ops
- **Export / Import** pipeline as JSON
- **Download** processed image (**PNG/JPEG**) without UI overlays
- Memory‑safe: intermediate `cv.Mat`s are cleaned up after each op

---

## Quick Start (Vite + React)
> Requires Node 18+

```bash
# 1) Create a Vite React app (or use your existing project)
npm create vite@latest opencv-playground -- --template react
cd opencv-playground
npm install
```

Add OpenCV.js to **index.html** (before your bundle):
```html
<!-- Load OpenCV.js (4.x) -->
<script async src="https://docs.opencv.org/4.x/opencv.js"></script>
```

*(Optional)* Tailwind – not required, but the sample component uses simple utility classes:
```bash
npm i -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
# configure content paths in tailwind.config.js as usual
```

Copy **`src/OpenCVPlayground.jsx`** from this repo into your project and render it from `App.jsx`:
```jsx
import OpenCVPlayground from "./OpenCVPlayground.jsx";
export default function App() { return <OpenCVPlayground />; }
```

Run:
```bash
npm run dev
```
Build:
```bash
npm run build
```

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

> Ops are safe to chain; color/gray conversions are handled inside each op.

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
**Schema → UI** is automatic (sliders, selects, numbers). If an op needs 1‑ or 3‑channel input, perform the conversion internally (e.g., RGBA→RGB/GRAY then back to RGBA).

---

## Architecture Notes
- Pipeline = ordered list of ops → each `apply()` returns a new `cv.Mat`
- After each step, the previous `Mat` is `delete()`‑d to avoid leaks
- Results are drawn to a canvas; an `ImageData` snapshot is cached to render the compare overlay without recomputing
- UI is fully reactive; changes to ops / ordering / enable flags cause re‑processing (debounced when **Live** is on)

---

## Troubleshooting
- **“OpenCV not ready”**: Ensure the `<script src="…/opencv.js">` tag is present and loads before the React bundle. The app waits for `cv.onRuntimeInitialized` when needed.
- **Bilateral crashes**: In OpenCV.js, `bilateralFilter` accepts 1‑ or 3‑channel images only. This app converts RGBA→RGB before the filter and back to RGBA after.
- **Dark mode contrast**: The app forces `color-scheme: light` for consistent button contrast. Adjust styling as desired.

---

## License
MIT

## Credits
- OpenCV.js — https://opencv.org
- Vite + React
