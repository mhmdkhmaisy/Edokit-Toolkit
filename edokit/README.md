# Edokit

A fast, native Java alternative to [Alt1 Toolkit](https://runeapps.org/alt1) for real-time RuneScape interface detection and buff-bar reading.

## What it does

Edokit captures the live RuneScape client window at ~30 FPS using the Windows GDI API (no screen-recording permissions required), locates the buff/debuff bar using OpenCV template matching, and OCRs the countdown timer overlay on each active buff slot using a pixel-exact raster text engine ported directly from Alt1's TypeScript source.

The result is printed to stdout every frame a known buff is detected:

```
[Edokit Engine] Active Buff: "Overload" | Timer: "4:32" | Slot: 1
```

## Architecture

```
EdokitEngineRunner          — 30 FPS capture loop, anchor discovery, orchestration
├── NativeCaptureEngine     — GDI BitBlt → raw RGBA byte[]  (JNA, no OpenCV)
├── SubImageMatcher         — TM_CCOEFF_NORMED template match (OpenCV)
├── BuffReader              — grid discovery + colour fingerprint matching
│   └── RasterOCR           — Alt1-faithful canblend() scorer, readLine(), removeTextPixels()
└── FontSheet               — font pipeline: unblend → generateFont() → FontDef/GlyphDef
```

### Font pipeline (Alt1-equivalent)

`FontSheet.load()` reproduces Alt1's `generateFont()` in `src/ocr/index.ts`:

1. **Parse** `.fontmeta.json` — reads `chars`, `seconds`, `bonus`, `color`, `shadow`, `unblendmode`, `treshold`, `basey`, `spacewidth`.
2. **Unblend** the glyph sprite sheet — three modes: `raw` (transparency-encoded), `blackbg`, `removebg` (two-row pairs).
3. **Scan boundary row** — the last row of the composite image marks glyph column extents (pixels where `R=255` and `A=255`).
4. **generateFont** — two passes:
   - Pass 1: compute tight `miny`/`maxy` across all glyphs.
   - Pass 2: collect above-threshold pixels into flat `[x, y, matchStrength, (shadowStrength)?]` arrays.
   - Output: `FontDef` with `GlyphDef[]` entries, adjusted `basey`, tight `height`, font colour.

### OCR scoring model (Alt1-faithful `canblend`)

Rather than measuring colour distance to a fixed expected colour, the engine asks: *"could this screen pixel result from blending the font colour with some plausible background?"*

```
p  = strength / 255          // template pixel opacity
m  = min(50, p / (1 - p))   // extrapolation leverage
bg = screenPixel + (screenPixel - fontColour) × m   // implied background
penalty = max(0, how far bg lies outside [0, 255]³)
```

`penalty = 0` means the observation is perfectly consistent with the font being present. Accumulated penalty > 400 disqualifies a glyph candidate.

Winner per cursor position = lowest `sizeScore` (biased toward larger glyphs to prevent small chars matching inside larger ones).

## Requirements

- **Java 17+**
- **Windows** (GDI capture via JNA; Linux/macOS not yet supported)
- **RuneScape** running in a window titled exactly `"RuneScape"`
- Maven 3.6+

## Build

```bash
cd edokit
mvn package -q
```

Produces `target/edokit-*.jar` with all dependencies shaded in.

## Run

```bash
java -jar target/edokit-*.jar
```

The engine will:
1. Load `pixel_8px_digits` font from the bundled classpath resources.
2. Bind the GDI capture engine to the RuneScape window.
3. Every ~5 seconds: run OpenCV template matching against `buffborder.data.png` to locate the buff bar.
4. Every frame (~33 ms): OCR the countdown timer on each discovered buff slot and fingerprint-match the icon colour.

Press **Ctrl+C** to exit; a JVM shutdown hook cleanly releases all GDI resources.

## Adding buff fingerprints

Edit `EdokitEngineRunner.buildBuffDatabase()`. Each `BuffDefinition` records 3–5 pixel sample offsets inside the 27×27 icon cell and the expected `0xRRGGBB` colour at each:

```java
database.add(new BuffDefinition(
    "My Buff",
    new int[]{ 13, 18, 10 },          // sampleOffsetsX
    new int[]{  7, 11, 15 },          // sampleOffsetsY
    new int[]{ 0xB03010, 0xE06020, 0xC04818 }  // expectedRGBs
));
```

Choose offsets from areas of the icon that are visually unique to that buff and not overlapped by the countdown timer digits (which appear along the bottom edge).

## Project structure

```
edokit/
├── src/main/java/edokit/
│   ├── engine/   EdokitEngineRunner.java
│   ├── base/
│   │   ├── capture/  NativeCaptureEngine.java, JNA bindings
│   │   ├── io/       EdokitImageLoader.java
│   │   ├── models/   EdokitImage.java
│   │   └── vision/   SubImageMatcher.java
│   ├── ocr/      FontSheet.java, RasterOCR.java
│   └── buffs/    BuffReader.java
└── src/main/resources/
    ├── fonts/    pixel_8px_digits.fontmeta.json + data.png
    └── buffs/    buffborder.data.png
```

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| JNA | 5.14.0 | GDI BitBlt screen capture |
| OpenCV | 4.9.0-0 | TM_CCOEFF_NORMED template matching |
| Gson | 2.11.0 | `.fontmeta.json` parsing |

## Debug: anchor confidence logging

If the buff bar is not being found, run with the engine's built-in debug mode (currently active). Every 150 frames it prints:

```
[Edokit Debug] Best match confidence found on screen: 0.6234 at (X: 812, Y: 47)
```

A production-quality match scores ≥ 0.87. Typical failure causes:

| Symptom | Likely cause |
|---------|-------------|
| Confidence < 0.5 | UI scaling mismatch — resample `buffborder.data.png` to match client DPI |
| Confidence 0.5–0.8 | Colour channel compression — verify the PNG was saved losslessly |
| `findBuffGrid failed` logged | Anchor coordinate is offset — the border heuristic expects the exact top-left corner of the buff cell |

## License

MIT
