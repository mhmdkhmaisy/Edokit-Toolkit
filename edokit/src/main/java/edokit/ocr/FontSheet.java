package edokit.ocr;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import edokit.base.io.EdokitImageLoader;
import edokit.base.models.EdokitImage;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * FontSheet — Alt1-compatible font schema parser and glyph registry.
 *
 * <h2>What this is</h2>
 * Alt1 distributes its UI fonts as a pair of files per font:
 * <ul>
 *   <li>A {@code .fontmeta.json} descriptor carrying global metadata (baseline,
 *       space width, character order, font colour, shadow flag, unblend mode).</li>
 *   <li>A {@code data.png} sprite sheet containing all glyphs packed into a
 *       single horizontal strip, loaded via {@link EdokitImageLoader} to
 *       guarantee raw, ICC-stripped pixels.</li>
 * </ul>
 * {@code FontSheet} parses both and exposes them in a form that
 * {@link RasterOCR} can scan efficiently with zero per-frame allocation.
 *
 * <h2>Alt1 font JSON schema</h2>
 * <pre>
 *   {
 *     "basey":       7,                 // baseline y offset (pixels from top of glyph strip)
 *     "spacewidth":  3,                 // advance width for the space character
 *     "chars":       "0123456789m()hr", // characters in left-to-right sprite order
 *     "treshold":    0.9,              // minimum match strength [0-1] to retain a pixel
 *     "color":       [255, 255, 255],  // font glyph colour in the template image
 *     "shadow":      true,             // whether the font carries a black drop-shadow
 *     "unblendmode": "raw"             // how to interpret the template pixel data
 *   }
 * </pre>
 *
 * <h2>Sprite sheet format</h2>
 * The last row of every {@code data.png} is a <em>boundary marker row</em> — it is
 * never rendered but encodes glyph column extents.  A pixel in that row with
 * {@code R == 255} and {@code A == 255} belongs to the current glyph region;
 * any other value is a gap between glyphs.  This is identical to how Alt1's
 * {@code generateFont()} function in {@code src/ocr/index.ts} locates glyphs.
 *
 * <h2>Unblending</h2>
 * Before boundary scanning, the glyph pixel rows are "unblended" into
 * per-pixel <em>match-strength</em> values (0–255) according to
 * {@code "unblendmode"}:
 * <ul>
 *   <li>{@code "raw"} — background already removed; match strength = alpha channel.</li>
 *   <li>{@code "blackbg"} — glyphs on black; strength = 1 − max colour deviation.</li>
 *   <li>{@code "removebg"} — the PNG encodes a background reference in its lower
 *       half; strength is derived via {@code decompose2col} linear decomposition.</li>
 * </ul>
 *
 * <h2>Memory layout</h2>
 * The full font sprite sheet is held as a single {@link EdokitImage} whose
 * {@code byte[]} is addressed by every {@link GlyphDefinition} via its
 * {@code (x, y)} origin + {@code (width, height)} extent.  No per-glyph
 * sub-image is allocated.
 */
public final class FontSheet {

    // =========================================================================
    // Public fields
    // =========================================================================

    /**
     * The full font sprite sheet in raw RGBA format.
     * All {@link GlyphDefinition} coordinates refer to regions within this image.
     */
    public final EdokitImage fontImage;

    /**
     * Vertical baseline offset in pixels.
     * When reading a text line starting at screen Y, the effective top of each
     * glyph in the sprite sheet is aligned using this value.
     * Default: 0.
     */
    public final int baseY;

    /**
     * Vertical step between consecutive text lines in pixels.
     * Used by multi-line block parsing.  Default: 0.
     */
    public final int lineHeight;

    /**
     * Fallback advance width for the space character ({@code ' '}).
     * Applied when the glyph map does not contain an explicit space entry.
     * Default: 4.
     */
    public final int spaceWidth;

    /**
     * Human-readable font identifier from the {@code "fontname"} JSON field.
     * Default: {@code "Alt1 Font"}.
     */
    public final String fontName;

    // =========================================================================
    // Private state
    // =========================================================================

    /**
     * Character → glyph metadata.  {@link LinkedHashMap} preserves the
     * left-to-right sprite declaration order so that {@link RasterOCR} tries
     * glyphs in a stable, deterministic sequence.
     */
    private final Map<Character, GlyphDefinition> glyphMap;

    // =========================================================================
    // Construction (private — use the static factory)
    // =========================================================================

    private FontSheet(EdokitImage fontImage,
                      Map<Character, GlyphDefinition> glyphMap,
                      int baseY,
                      int lineHeight,
                      int spaceWidth,
                      String fontName) {
        this.fontImage  = fontImage;
        this.glyphMap   = glyphMap;
        this.baseY      = baseY;
        this.lineHeight = lineHeight;
        this.spaceWidth = spaceWidth;
        this.fontName   = fontName;
    }

    // =========================================================================
    // Static factory
    // =========================================================================

    /**
     * Parses an Alt1 font descriptor and loads its accompanying sprite sheet.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>Parse JSON — extract {@code basey}, {@code spacewidth}, {@code chars},
     *       {@code treshold}, {@code color}, {@code shadow}, {@code unblendmode}.</li>
     *   <li>Load the raw PNG via {@link EdokitImageLoader}.</li>
     *   <li>Compute {@code pxheight} (glyph content rows, excluding the boundary
     *       marker row — and the background reference half for {@code removebg}).</li>
     *   <li>Extract the glyph pixel sub-image and (for {@code removebg}) the
     *       background reference sub-image.</li>
     *   <li>Unblend: convert raw RGB/A pixels into per-pixel match-strength values
     *       using the mode-appropriate function.</li>
     *   <li>Build the composite "unblended" image: unblended glyph rows followed by
     *       the original boundary marker row.</li>
     *   <li>Scan the boundary marker row to locate glyph column extents and assign
     *       them to characters in {@code "chars"} order.</li>
     * </ol>
     *
     * @param jsonStream the {@code .fontmeta.json} descriptor stream
     * @param imgStream  the {@code data.png} sprite sheet stream
     * @return a fully initialised {@code FontSheet}
     * @throws IOException              if either stream cannot be read or the image
     *                                  format is unsupported
     * @throws IllegalArgumentException if the JSON is malformed or the boundary row
     *                                  yields no glyph entries
     */
    public static FontSheet load(InputStream jsonStream, InputStream imgStream)
            throws IOException {

        // ── 1. Parse JSON descriptor ──────────────────────────────────────────
        final JsonObject root;
        try (Reader reader = new InputStreamReader(jsonStream, StandardCharsets.UTF_8)) {
            root = JsonParser.parseReader(reader).getAsJsonObject();
        }

        final String fontName   = getStringOrDefault(root, "fontname",    "Alt1 Font");
        final int    baseY      = getIntOrDefault(root,    "basey",        0);
        final int    lineHeight = getIntOrDefault(root,    "lineheight",   0);
        final int    spaceWidth = getIntOrDefault(root,    "spacewidth",   4);
        // treshold [0-1]: minimum match-strength proportion to retain a pixel.
        // Converted to 0-255 scale to match Alt1's generateFont() multiply.
        final int    treshold   = (int) (getDoubleOrDefault(root, "treshold", 0.6) * 255.0);
        final boolean shadow    = getBoolOrDefault(root, "shadow", false);
        final String unblendMode = getStringOrDefault(root, "unblendmode", "raw");
        final int[]  color      = getColorOrDefault(root);   // [R, G, B]

        // "chars" — characters in left-to-right sprite order.
        final JsonElement charsEl = root.get("chars");
        if (charsEl == null || charsEl.isJsonNull()) {
            throw new IllegalArgumentException(
                    "Font JSON for \"" + fontName + "\" is missing the required "
                    + "\"chars\" field.");
        }
        final String charsStr = charsEl.getAsString();
        if (charsStr.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font JSON for \"" + fontName + "\" has an empty \"chars\" string.");
        }

        // ── 2. Load sprite sheet (ICC-stripped raw RGBA) ──────────────────────
        final EdokitImage fontImage = EdokitImageLoader.load(imgStream);
        final int imgWidth  = fontImage.width;
        final int imgHeight = fontImage.height;

        // ── 3. Compute pxheight and boundary row index ────────────────────────
        //
        // For "raw" / "blackbg":
        //   PNG layout:   [ glyph rows 0..H-1 | boundary row H ]
        //   pxheight = H = imgHeight - 1
        //   boundary row in original PNG = imgHeight - 1
        //
        // For "removebg":
        //   PNG layout:   [ glyph rows 0..H-1 | boundary row H | bg rows H+1..2H ]
        //   pxheight = H = (imgHeight - 1) / 2
        //   boundary row in original PNG = H
        //
        final int pxheight;
        final int boundaryRowInOrig;
        if ("removebg".equals(unblendMode)) {
            pxheight         = (imgHeight - 1) / 2;
            boundaryRowInOrig = pxheight;               // middle row
        } else {
            pxheight         = imgHeight - 1;
            boundaryRowInOrig = imgHeight - 1;           // last row
        }

        if (pxheight <= 0) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\" sprite sheet is too small ("
                    + imgWidth + "×" + imgHeight + " px) for unblendmode=\""
                    + unblendMode + "\".  Expected at least height 2.");
        }

        // ── 4. Extract sub-images ─────────────────────────────────────────────
        final EdokitImage inImg = fontImage.cloneSubImage(0, 0, imgWidth, pxheight);

        // ── 5. Unblend: raw pixels → per-pixel match-strength values ──────────
        final EdokitImage outImg;
        if ("removebg".equals(unblendMode)) {
            final EdokitImage bgImg =
                    fontImage.cloneSubImage(0, pxheight + 1, imgWidth, pxheight);
            outImg = unblendKnownBg(inImg, bgImg, shadow, color[0], color[1], color[2]);
        } else if ("blackbg".equals(unblendMode)) {
            outImg = unblendBlackBackground(inImg, color[0], color[1], color[2]);
        } else {
            // "raw" — match strength is the alpha channel directly
            outImg = unblendTrans(inImg, shadow, color[0], color[1], color[2]);
        }

        // ── 6. Build composite "unblended" image ──────────────────────────────
        //   rows 0..pxheight-1  : unblended match-strength pixels
        //   row  pxheight       : original boundary marker row (R=255,A=255 = glyph)
        final byte[] compositeData = new byte[(pxheight + 1) * imgWidth * 4];
        // Copy unblended glyph rows
        System.arraycopy(outImg.data, 0, compositeData, 0, outImg.data.length);
        // Append boundary row verbatim from the original PNG
        System.arraycopy(
                fontImage.data, boundaryRowInOrig * imgWidth * 4,
                compositeData,  pxheight          * imgWidth * 4,
                imgWidth * 4);
        final EdokitImage unblended = new EdokitImage(imgWidth, pxheight + 1, compositeData);

        // ── 7. Scan boundary row → glyph map ─────────────────────────────────
        final Map<Character, GlyphDefinition> glyphMap = new LinkedHashMap<>();
        scanGlyphBoundaries(unblended, charsStr, glyphMap, fontName, pxheight);

        if (glyphMap.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\" yielded 0 glyph entries.  "
                    + "Verify that the PNG's boundary marker row (last row for "
                    + "\"raw\"/\"blackbg\", middle row for \"removebg\") contains "
                    + "at least one pixel with R=255 and A=255.");
        }

        return new FontSheet(fontImage,
                Collections.unmodifiableMap(glyphMap),
                baseY, lineHeight, spaceWidth, fontName);
    }

    // =========================================================================
    // Public accessors
    // =========================================================================

    /**
     * Returns the {@link GlyphDefinition} for {@code ch}, or {@code null} if
     * the font does not define that character.
     *
     * @param ch the character to look up
     * @return glyph definition, or {@code null}
     */
    public GlyphDefinition getGlyph(char ch) {
        return glyphMap.get(ch);
    }

    /**
     * Returns an unmodifiable view of the full glyph map.
     *
     * <p>Iteration order reflects the left-to-right sprite sheet declaration
     * order (preserved by the {@link LinkedHashMap} backing store).
     *
     * @return unmodifiable {@code Character → GlyphDefinition} map
     */
    public Map<Character, GlyphDefinition> glyphMap() {
        return glyphMap; // already wrapped unmodifiable in load()
    }

    /** Returns the number of glyph entries in this font. */
    public int glyphCount() {
        return glyphMap.size();
    }

    @Override
    public String toString() {
        return "FontSheet{name=\"" + fontName + "\", glyphs=" + glyphMap.size()
               + ", baseY=" + baseY + ", spaceWidth=" + spaceWidth + "}";
    }

    // =========================================================================
    // Private — unblending pipeline
    // =========================================================================

    /**
     * Unblends a sprite sheet that already has its background removed
     * ({@code unblendmode = "raw"}).
     *
     * <p>For each pixel, the <em>match strength</em> is the alpha channel value —
     * fully opaque pixels ({@code A = 255}) are pure glyph colour; fully
     * transparent pixels ({@code A = 0}) are pure background.
     * When {@code shadow = true}, the green output channel carries the
     * <em>shadow strength</em>: the luminance of the source pixel normalised to
     * the font's declared colour luminance ({@code R+G+B}).
     *
     * <p>Output pixel layout:
     * <pre>
     *   R = match strength (= source alpha)
     *   G = shadow strength (= source luminance / font-colour luminance × 255, or = R if no shadow)
     *   B = match strength (copy of R)
     *   A = 255 (always fully opaque)
     * </pre>
     *
     * @param img    glyph pixel sub-image (rows 0..pxheight-1 of the original PNG)
     * @param shadow {@code true} if the font carries a black drop-shadow
     * @param r      font colour red channel [0-255]
     * @param g      font colour green channel [0-255]
     * @param b      font colour blue channel [0-255]
     * @return a new {@link EdokitImage} of the same dimensions with match-strength data
     */
    private static EdokitImage unblendTrans(EdokitImage img,
                                             boolean shadow,
                                             int r, int g, int b) {
        final int    pxlum = r + g + b;
        final byte[] out   = new byte[img.data.length];
        for (int i = 0; i < img.data.length; i += 4) {
            final int matchStr = img.data[i + 3] & 0xFF;  // source alpha = match strength
            final int shadowStr;
            if (shadow && pxlum > 0) {
                // Shadow strength = source pixel luminance / font-colour luminance
                final int lum = (img.data[i]     & 0xFF)
                              + (img.data[i + 1] & 0xFF)
                              + (img.data[i + 2] & 0xFF);
                shadowStr = Math.min(255, lum * 255 / pxlum);
            } else {
                shadowStr = matchStr;
            }
            out[i]     = (byte) matchStr;   // R = match strength
            out[i + 1] = (byte) shadowStr;  // G = shadow strength
            out[i + 2] = (byte) matchStr;   // B = match strength (copy)
            out[i + 3] = (byte) 0xFF;       // A = always opaque
        }
        return new EdokitImage(img.width, img.height, out);
    }

    /**
     * Unblends a sprite sheet where glyphs are drawn on a solid black background
     * ({@code unblendmode = "blackbg"}, {@code shadow = false} only).
     *
     * <p>Match strength = {@code 1 − maxChannelDeviation / 255}, where
     * {@code maxChannelDeviation = max(|R−r|, |G−g|, |B−b|)}.  A pixel that
     * exactly matches the font colour gets strength 255; a black pixel gets 0.
     *
     * @param img the glyph pixel sub-image
     * @param r   font colour red channel [0-255]
     * @param g   font colour green channel [0-255]
     * @param b   font colour blue channel [0-255]
     * @return a new {@link EdokitImage} with match-strength data in all channels
     */
    private static EdokitImage unblendBlackBackground(EdokitImage img,
                                                       int r, int g, int b) {
        final byte[] out = new byte[img.data.length];
        for (int i = 0; i < img.data.length; i += 4) {
            final int rp = img.data[i]     & 0xFF;
            final int gp = img.data[i + 1] & 0xFF;
            final int bp = img.data[i + 2] & 0xFF;
            final int maxDif = Math.max(Math.abs(rp - r),
                               Math.max(Math.abs(gp - g), Math.abs(bp - b)));
            final int strength = 255 - maxDif;  // (1 − maxDif/255) × 255
            out[i]     = (byte) strength;
            out[i + 1] = (byte) strength;
            out[i + 2] = (byte) strength;
            out[i + 3] = (byte) 0xFF;
        }
        return new EdokitImage(img.width, img.height, out);
    }

    /**
     * Unblends a sprite sheet that encodes both the character screenshots and
     * the background reference within the same PNG
     * ({@code unblendmode = "removebg"}).
     *
     * <p>Each pixel is decomposed into proportions of the font colour and the
     * background colour using {@link #decompose2col} (Cramer's rule on a 3-basis
     * colour space).  The font-colour proportion becomes the match strength.
     *
     * <p>Output pixel layout (identical to {@link #unblendTrans} for downstream
     * compatibility):
     * <pre>
     *   R = match strength
     *   G = shadow strength (or = R when shadow = false)
     *   B = match strength (copy)
     *   A = 255
     * </pre>
     *
     * @param img    glyph pixel sub-image (top half of the original PNG)
     * @param bgImg  background reference sub-image (bottom half of the original PNG)
     * @param shadow {@code true} if the font carries a black drop-shadow
     * @param r      font colour red channel
     * @param g      font colour green channel
     * @param b      font colour blue channel
     * @return a new {@link EdokitImage} with match-strength data
     */
    private static EdokitImage unblendKnownBg(EdokitImage img, EdokitImage bgImg,
                                               boolean shadow,
                                               int r, int g, int b) {
        final byte[] out = new byte[img.data.length];
        for (int i = 0; i < img.data.length; i += 4) {
            final double rp = img.data[i]       & 0xFF;
            final double gp = img.data[i + 1]   & 0xFF;
            final double bp = img.data[i + 2]   & 0xFF;
            final double r2 = bgImg.data[i]     & 0xFF;
            final double g2 = bgImg.data[i + 1] & 0xFF;
            final double b2 = bgImg.data[i + 2] & 0xFF;

            // col[0] = proportion of font colour (= match strength factor)
            // col[1] = proportion of shadow/black colour
            // col[2] = error component (orthogonal noise)
            final double[] col = decompose2col(rp, gp, bp, r, g, b, r2, g2, b2);

            final int matchStr;
            final int shadowStr;
            if (shadow) {
                // m = main-colour proportion; avoid divide-by-zero if near 0
                final double m = 1.0 - col[1] - Math.abs(col[2]);
                matchStr  = clamp255(m * 255.0);
                shadowStr = (m > 1e-4) ? clamp255(col[0] / m * 255.0) : 0;
            } else {
                matchStr  = clamp255(col[0] * 255.0);
                shadowStr = matchStr;
            }
            out[i]     = (byte) matchStr;
            out[i + 1] = (byte) shadowStr;
            out[i + 2] = (byte) matchStr;
            out[i + 3] = (byte) 0xFF;
        }
        return new EdokitImage(img.width, img.height, out);
    }

    /**
     * Decomposes pixel {@code [rp, gp, bp]} into proportions of two given basis
     * colours plus an orthogonal error component, using Cramer's rule.
     *
     * <p>A third basis vector is constructed as the cross-product of the two input
     * colours (then normalised to length 255) to span the error dimension.
     * {@link #decompose3col} is then called with all three bases.
     *
     * @param rp the pixel red channel to decompose
     * @param gp the pixel green channel
     * @param bp the pixel blue channel
     * @param r1 first basis colour red
     * @param g1 first basis colour green
     * @param b1 first basis colour blue
     * @param r2 second basis colour red
     * @param g2 second basis colour green
     * @param b2 second basis colour blue
     * @return {@code [x, y, z]} where {@code pixel ≈ x·C1 + y·C2 + z·C3}
     */
    private static double[] decompose2col(double rp, double gp, double bp,
                                          double r1, double g1, double b1,
                                          double r2, double g2, double b2) {
        // Cross-product of C1 and C2 gives an orthogonal third basis vector.
        double r3 = g1 * b2 - g2 * b1;
        double g3 = b1 * r2 - b2 * r1;
        double b3 = r1 * g2 - r2 * g1;
        final double len = Math.sqrt(r3 * r3 + g3 * g3 + b3 * b3);
        if (len < 1e-10) {
            // Degenerate: font colour and background colour are collinear (identical).
            return new double[]{0.0, 0.0, 0.0};
        }
        final double norm = 255.0 / len;
        r3 *= norm; g3 *= norm; b3 *= norm;
        return decompose3col(rp, gp, bp, r1, g1, b1, r2, g2, b2, r3, g3, b3);
    }

    /**
     * Decomposes pixel {@code [rp, gp, bp]} into proportions of three given basis
     * colours using Cramer's rule (direct solution of the 3×3 linear system
     * {@code M·w = p}).
     *
     * <p>Port of Alt1's {@code decompose3col()} from {@code src/ocr/index.ts}.
     *
     * @return {@code [x, y, z]} where {@code pixel ≈ x·C1 + y·C2 + z·C3}
     */
    private static double[] decompose3col(double rp, double gp, double bp,
                                          double r1, double g1, double b1,
                                          double r2, double g2, double b2,
                                          double r3, double g3, double b3) {
        // Cofactors of the column matrix [C1 | C2 | C3]
        final double A = g2 * b3 - b2 * g3;
        final double B = g3 * b1 - b3 * g1;
        final double C = g1 * b2 - b1 * g2;
        final double D = b2 * r3 - r2 * b3;
        final double E = b3 * r1 - r3 * b1;
        final double F = b1 * r2 - r1 * b2;
        final double G = r2 * g3 - g2 * r3;
        final double H = r3 * g1 - g3 * r1;
        final double I = r1 * g2 - g1 * r2;
        final double det = r1 * A + g1 * D + b1 * G;
        if (Math.abs(det) < 1e-10) {
            return new double[]{0.0, 0.0, 0.0};
        }
        return new double[]{
            (A * rp + D * gp + G * bp) / det,
            (B * rp + E * gp + H * bp) / det,
            (C * rp + F * gp + I * bp) / det
        };
    }

    // =========================================================================
    // Private — glyph boundary scanner
    // =========================================================================

    /**
     * Reads the boundary marker row (last row of {@code unblended}) and maps
     * each discovered glyph region to the corresponding character in
     * {@code chars}.
     *
     * <h3>Boundary marker protocol</h3>
     * A pixel at column {@code x} of the last row is inside a glyph region when
     * <pre>
     *   R == 255 AND A == 255
     * </pre>
     * Any other value is treated as a gap between glyphs.  This matches Alt1's
     * {@code generateFont()} glyph-detection loop in {@code src/ocr/index.ts}
     * exactly.
     *
     * <h3>GlyphDefinition fields</h3>
     * <ul>
     *   <li>{@code x} — left edge of the glyph in the sprite sheet (column of
     *       first boundary-marked pixel).</li>
     *   <li>{@code y} — always 0 (all glyphs start at the top of the content
     *       rows).</li>
     *   <li>{@code width} — exact pixel width from the boundary markers
     *       ({@code de - ds} in Alt1 terminology).</li>
     *   <li>{@code height} — {@code pxheight} (content rows only, boundary
     *       marker row excluded).</li>
     *   <li>{@code advX} — set to {@code width}; the cursor advances by the
     *       exact glyph pixel width (no +1 approximation needed because the
     *       boundary markers already encode the natural spacing).</li>
     * </ul>
     *
     * @param unblended composite image: unblended glyph rows + boundary marker row
     * @param chars     ordered character string from the {@code "chars"} JSON field
     * @param target    destination map (insertion order preserved via LinkedHashMap)
     * @param fontName  font identifier for diagnostic messages
     * @param pxheight  number of glyph content rows (= {@code unblended.height - 1})
     */
    private static void scanGlyphBoundaries(EdokitImage unblended,
                                             String chars,
                                             Map<Character, GlyphDefinition> target,
                                             String fontName,
                                             int pxheight) {
        final int    imgWidth = unblended.width;
        final byte[] data     = unblended.data;

        int     charIndex  = 0;
        boolean inGlyph    = false;
        int     glyphStart = 0;

        // Iterate to x = imgWidth inclusive so an open glyph at the right edge is closed.
        for (int x = 0; x <= imgWidth; x++) {
            final boolean marker;
            if (x < imgWidth) {
                // Boundary row is at y = pxheight (the last row of the unblended image).
                // Pixel offset: (x + imgWidth * pxheight) * 4
                final int off = (x + imgWidth * pxheight) * 4;
                marker = ((data[off]     & 0xFF) == 255)   // R == 255
                      && ((data[off + 3] & 0xFF) == 255);  // A == 255
            } else {
                marker = false;  // sentinel — closes any glyph still open at the far edge
            }

            if (marker && !inGlyph) {
                // gap → glyph: record start column
                inGlyph    = true;
                glyphStart = x;
            } else if (!marker && inGlyph) {
                // glyph → gap: commit GlyphDefinition for this region
                inGlyph = false;
                final int glyphWidth = x - glyphStart;
                if (glyphWidth > 0 && charIndex < chars.length()) {
                    final char ch = chars.charAt(charIndex);
                    // advX = glyphWidth: exact advance derived from boundary markers.
                    target.put(ch, new GlyphDefinition(
                            glyphStart, 0, glyphWidth, pxheight, glyphWidth));
                    charIndex++;
                }
            }
        }

        if (charIndex == chars.length()) {
            System.out.printf(
                    "[FontSheet] \"%s\": successfully mapped %d/%d glyphs from boundary row.%n",
                    fontName, charIndex, chars.length());
        } else {
            System.err.printf(
                    "[FontSheet] Warning: \"%s\" — boundary row yielded %d glyph(s) "
                    + "but \"chars\" declares %d.  Verify that the PNG boundary "
                    + "marker row (R=255, A=255 marks glyph columns) is intact.%n",
                    fontName, charIndex, chars.length());
        }
    }

    // =========================================================================
    // Private — null-safe JSON accessors
    // =========================================================================

    private static int getIntOrDefault(JsonObject obj, String key, int def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsInt() : def;
    }

    private static double getDoubleOrDefault(JsonObject obj, String key, double def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsDouble() : def;
    }

    private static boolean getBoolOrDefault(JsonObject obj, String key, boolean def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsBoolean() : def;
    }

    private static String getStringOrDefault(JsonObject obj, String key, String def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsString() : def;
    }

    /**
     * Reads the {@code "color"} JSON array as an {@code int[3]} (R, G, B).
     * Defaults to white {@code [255, 255, 255]} if absent or malformed.
     */
    private static int[] getColorOrDefault(JsonObject obj) {
        final JsonElement el = obj.get("color");
        if (el != null && !el.isJsonNull() && el.isJsonArray()) {
            final JsonArray arr = el.getAsJsonArray();
            if (arr.size() >= 3) {
                return new int[]{
                    arr.get(0).getAsInt(),
                    arr.get(1).getAsInt(),
                    arr.get(2).getAsInt()
                };
            }
        }
        return new int[]{255, 255, 255};
    }

    /**
     * Clips a floating-point value to the unsigned byte range [0, 255] and
     * truncates to {@code int}.  Used throughout the unblend pipeline to convert
     * linear-algebra results back into pixel channel values.
     *
     * @param v the value to clamp
     * @return {@code v} clamped to [0, 255] as an {@code int}
     */
    private static int clamp255(double v) {
        return (int) Math.max(0.0, Math.min(255.0, v));
    }

    // =========================================================================
    // Nested value type
    // =========================================================================

    /**
     * Immutable descriptor for a single character's position and metrics inside
     * the font sprite sheet.
     *
     * <p>All coordinates are in font-sheet pixel space and refer to
     * {@link FontSheet#fontImage}{@code .data}.  {@link RasterOCR} accesses the
     * font pixel at column {@code x + dx}, row {@code y + dy} via
     * {@link EdokitImage#pixelOffset}{@code (x + dx, y + dy)}.
     *
     * @param x      left edge of the glyph inside the sprite sheet (inclusive)
     * @param y      top edge of the glyph inside the sprite sheet (inclusive)
     * @param width  glyph pixel width derived from the boundary marker row
     * @param height glyph pixel height (content rows only; boundary row excluded)
     * @param advX   horizontal cursor advance after this character is consumed
     */
    public record GlyphDefinition(int x, int y, int width, int height, int advX) {

        @Override
        public String toString() {
            return String.format("GlyphDef{x=%d, y=%d, w=%d, h=%d, advX=%d}",
                    x, y, width, height, advX);
        }
    }
}
