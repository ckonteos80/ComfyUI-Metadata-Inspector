# ComfyUI Metadata Inspector

A lightweight, browser-based tool for inspecting and comparing metadata embedded in ComfyUI-generated PNG files — no upload, no server, no install.

🔗 **[Live demo](https://ckonteos80.github.io/ComfyUI-Metadata-Inspector)**

---

## Features

- **Drag & drop** or click to load multiple PNGs at once
- Extracts: seed, steps, CFG, sampler, scheduler, resolution, model, LoRAs, positive/negative prompts
- Supports a wide range of node types: KSampler, KSamplerAdvanced, Flux, SDXL, SD3, custom schedulers, and more
- LoRA detection from loader nodes and inline `<lora:...>` tags in prompts
- Sortable, resizable columns with per-column visibility toggles
- **Row selection** — click a thumbnail to select/deselect a row; Shift+click to select a continuous range
- **Delete selected** — remove individual rows from the table without clearing everything
- **Hover-to-copy** — a copy button appears on any cell when hovered; copies the full untruncated value
- Export to **CSV** or **JSON** (only the remaining rows are exported)
- Thumbnail preview for each image
- Dark mode (follows system preference)
- 100% client-side — your images never leave your machine

---

## Usage

1. Open the [live page](https://ckonteos80.github.io/ComfyUI-Metadata-Inspector)
2. Drop ComfyUI PNG files onto the drop zone, or click to browse
3. Inspect, sort, and compare metadata across images
4. Click a thumbnail to select rows; Shift+click to select a range; use **Delete selected** to remove them
5. Hover any cell to reveal a copy button — copies the full value even if the cell is visually truncated
6. Optionally export the table as CSV or JSON

---

## Supported Node Types

| Category | Nodes |
|---|---|
| Samplers | KSampler, KSamplerAdvanced, SamplerCustom, SamplerCustomAdvanced, KSamplerEfficient |
| Schedulers | BasicScheduler, KarrasScheduler |
| Models | CheckpointLoaderSimple, UNETLoader, UnetLoaderGGUF, DiffusionModelLoader |
| LoRAs | LoraLoader, LoraLoaderStack, Power Lora Loader (rgthree), CR LoRA Stack, and more |
| Text Encoders | CLIPTextEncode, CLIPTextEncodeSDXL, CLIPTextEncodeFlux, CLIPTextEncodeSD3, and more |
| Guidance | FluxGuidance |

---

## Local Use

No build step needed. Just open `index.html` directly in any modern browser.

---

## License

MIT
