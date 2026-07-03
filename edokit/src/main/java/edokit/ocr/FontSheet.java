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
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * FontSheet — Alt1-compatible font loader.
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Parse {@code .fontmeta.json} — extracts {@code basey}, {@code spacewidth},
 *       {@code chars}, {@code seconds}, {@code bonus}, {@code treshold}, {@code color},
 *       {@code shadow}, {@code unblendmode}.</li>
 *   <li>Load the companion {@code data.png} via {@link EdokitImageLoader}
 *       (ICC-stripped raw RGBA).</li>
 *   <li>Compute {@code pxheight} (glyph content rows) based on {@code unblendmode}:
 *       {@code imgHeight-1} for {@code raw}/{@code blackbg};
 *       {@code (imgHeight-1)/2} for {@code removebg}.</li>
 *   <li>Unblend the glyph rows into per-pixel match-strength values.</li>
 *   <li>Composite: unblended rows followed by the original boundary marker row.</li>
 *   <li>Scan the boundary row ({@code R==255 AND A==255} marks glyph columns) to
 *       discover glyph extents — identical to Alt1's {@code generateFont()} loop.</li>
 *   <li>Run {@code generateFont}: compute tight height, collect above-threshold
 *       pixels into flat arrays, build {@link FontDef} and its {@link GlyphDef}s.</li>
 * </ol>
 *
 * <h2>Data model (Alt1-equivalent)</h2>
 * {@link FontDef} ↔ {@code FontDefinition}; {@link GlyphDef} ↔ {@code Charinfo}.
 * The {@code pixels[]} flat array uses stride 3 (no shadow) or 4 (shadow):
 * {@code [x, y, matchStrength, (shadowStrength)?]}.
 */
public final class FontSheet {

    // =========================================================================
    // Public fields
    // =========================================================================

    /** Original font sprite sheet in raw RGBA (retained for debugging). */
    public final EdokitImage fontImage;

    /** Raw baseline offset from the JSON {@code "basey"} field. */
    public final int baseY;

    /** Multi-line vertical step; 0 for most Alt1 fonts. */
    public final int lineHeight;

    /** Space-character advance width from the JSON {@code "spacewidth"} field. */
    public final int spaceWidth;

    /** Human-readable font name; defaults to {@code "Alt1 Font"} if absent. */
    public final String fontName;

    // =========================================================================
    // Private state
    // =========================================================================

    /** The fully-processed font data ready for {@link RasterOCR}. */
    private final FontDef fontDef;

    // =========================================================================
    // Construction
    // =========================================================================

    private FontSheet(EdokitImage fontImage,
                      FontDef fontDef,
                      int baseY,
                      int lineHeight,
                      int spaceWidth,
                      String fontName) {
        this.fontImage  = fontImage;
        this.fontDef    = fontDef;
        this.baseY      = baseY;
        this.lineHeight = lineHeight;
        this.spaceWidth = spaceWidth;
        this.fontName   = fontName;
    }

    // =========================================================================
    // Static factory
    // =========================================================================

    /**
     * Loads and processes a font from its JSON descriptor and sprite sheet PNG.
     *
     * @param jsonStream the {@code .fontmeta.json} descriptor
     * @param imgStream  the {@code data.png} sprite sheet
     * @return a fully initialised {@code FontSheet}
     * @throws IOException              if either stream cannot be read
     * @throws IllegalArgumentException if the font produces zero usable glyphs
     */
    public static FontSheet load(InputStream jsonStream, InputStream imgStream)
            throws IOException {

        // ── 1. Parse JSON ─────────────────────────────────────────────────────
        final JsonObject root;
        try (Reader r = new InputStreamReader(jsonStream, StandardCharsets.UTF_8)) {
            root = JsonParser.parseReader(r).getAsJsonObject();
        }

        final String  fontName    = getStringOrDefault(root, "fontname",    "Alt1 Font");
        final int     baseY       = getIntOrDefault(root,    "basey",        0);
        final int     lineHeight  = getIntOrDefault(root,    "lineheight",   0);
        final int     spaceWidth  = getIntOrDefault(root,    "spacewidth",   4);
        // treshold [0-1] → converted to 0-255 scale to match Alt1's generateFont multiply
        final int     treshold    = (int)(getDoubleOrDefault(root, "treshold", 0.6) * 255.0);
        final boolean shadow      = getBoolOrDefault(root, "shadow", false);
        final String  unblendMode = getStringOrDefault(root, "unblendmode", "raw");
        final int[]   color       = getColorOrDefault(root);       // [R, G, B]
        final String  seconds     = getStringOrDefault(root, "seconds", "");
        final Map<Character, Integer> bonusMap = getCharBonusMap(root);

        final JsonElement charsEl = root.get("chars");
        if (charsEl == null || charsEl.isJsonNull()) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\": missing required \"chars\" field.");
        }
        final String charsStr = charsEl.getAsString();
        if (charsStr.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\": \"chars\" field is empty.");
        }

        // ── 2. Load PNG ───────────────────────────────────────────────────────
        final EdokitImage fontImage = EdokitImageLoader.load(imgStream);
        final int imgWidth  = fontImage.width;
        final int imgHeight = fontImage.height;

        // ── 3. Compute pxheight and boundary row index ────────────────────────
        final int pxheight;
        final int boundaryRowInOrig;
        if ("removebg".equals(unblendMode)) {
            pxheight          = (imgHeight - 1) / 2;
            boundaryRowInOrig = pxheight;
        } else {
            pxheight          = imgHeight - 1;
            boundaryRowInOrig = imgHeight - 1;
        }

        if (pxheight <= 0) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\": sprite sheet too small ("
                    + imgWidth + "×" + imgHeight + ") for unblendmode=\"" + unblendMode + "\".");
        }

        // ── 4. Extract and unblend ────────────────────────────────────────────
        final EdokitImage inImg = fontImage.cloneSubImage(0, 0, imgWidth, pxheight);
        final EdokitImage outImg;
        if ("removebg".equals(unblendMode)) {
            final EdokitImage bgImg =
                    fontImage.cloneSubImage(0, pxheight + 1, imgWidth, pxheight);
            outImg = unblendKnownBg(inImg, bgImg, shadow, color[0], color[1], color[2]);
        } else if ("blackbg".equals(unblendMode)) {
            outImg = unblendBlackBackground(inImg, color[0], color[1], color[2]);
        } else {
            outImg = unblendTrans(inImg, shadow, color[0], color[1], color[2]);
        }

        // ── 5. Composite: unblended rows + original boundary row ──────────────
        final byte[] compositeData = new byte[(pxheight + 1) * imgWidth * 4];
        System.arraycopy(outImg.data, 0, compositeData, 0, outImg.data.length);
        System.arraycopy(fontImage.data, boundaryRowInOrig * imgWidth * 4,
                         compositeData,  pxheight          * imgWidth * 4,
                         imgWidth * 4);
        final EdokitImage unblended = new EdokitImage(imgWidth, pxheight + 1, compositeData);

        // ── 6. Boundary scan → glyph regions ─────────────────────────────────
        final List<int[]> regions = new ArrayList<>();
        scanGlyphBoundaries(unblended, charsStr, regions, fontName);

        // ── 7. generateFont → FontDef ─────────────────────────────────────────
        final FontDef fontDef = generateFont(
                unblended, charsStr, seconds, bonusMap, regions,
                baseY, spaceWidth, treshold, shadow, color[0], color[1], color[2]);

        if (fontDef.chars().length == 0) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\": no usable glyphs produced.  "
                    + "Verify the PNG boundary row (last row: R=255 A=255 = glyph columns).");
        }

        return new FontSheet(fontImage, fontDef, baseY, lineHeight, spaceWidth, fontName);
    }

    // =========================================================================
    // Public accessors
    // =========================================================================

    /** Returns the fully-processed {@link FontDef} used by {@link RasterOCR}. */
    public FontDef fontDef() { return fontDef; }

    /** Returns the number of glyph entries in this font. */
    public int glyphCount() { return fontDef.chars().length; }

    @Override
    public String toString() {
        return "FontSheet{name=\"" + fontName + "\", glyphs=" + fontDef.chars().length
               + ", baseY=" + baseY + ", spaceWidth=" + spaceWidth + "}";
    }

    // =========================================================================
    // Private — generateFont (Alt1 generateFont equivalent)
    // =========================================================================

    /**
     * Converts the unblended image + glyph regions into a {@link FontDef}.
     *
     * <p>Pass 1 computes the tight content bounding box ({@code miny..maxy}) across
     * ALL glyphs; pass 2 collects per-pixel {@code [x, y, matchStrength, (shadowStrength)?]}
     * tuples for every pixel whose match strength meets {@code treshold}.
     *
     * <p>This is a direct Java port of Alt1's {@code generateFont()} in
     * {@code src/ocr/index.ts}.
     *
     * @param unblended   composite image (unblended rows + boundary marker row)
     * @param chars       ordered character string from JSON {@code "chars"}
     * @param seconds     secondary-character string from JSON {@code "seconds"}
     * @param bonusMap    per-character score nudges from JSON {@code "bonus"}
     * @param regions     glyph column extents from {@link #scanGlyphBoundaries}
     * @param rawBaseY    raw {@code "basey"} from JSON (adjusted by miny here)
     * @param spaceWidth  space advance
     * @param treshold    match-strength threshold in 0-255 scale
     * @param shadow      font has drop-shadow (pixels[] stride = 4 vs 3)
     * @param fontR/G/B   declared font colour
     * @return a fully populated {@link FontDef}
     */
    private static FontDef generateFont(EdokitImage unblended,
                                         String chars,
                                         String seconds,
                                         Map<Character, Integer> bonusMap,
                                         List<int[]> regions,
                                         int rawBaseY,
                                         int spaceWidth,
                                         int treshold,
                                         boolean shadow,
                                         int fontR, int fontG, int fontB) {
        final int BONUS_PER_PIXEL = 5;
        final int pxheight = unblended.height - 1;   // content rows
        final int imgWidth = unblended.width;
        final byte[] data  = unblended.data;

        final int nGlyphs = Math.min(regions.size(), chars.length());

        // ── Pass 1: compute tight content bounding box (miny..maxy) ───────────
        int miny = pxheight - 1;
        int maxy = 0;

        for (int ci = 0; ci < nGlyphs; ci++) {
            final int ds = regions.get(ci)[0];
            final int de = regions.get(ci)[1];
            for (int x = 0; x < de - ds; x++) {
                for (int y = 0; y < pxheight; y++) {
                    final int off = (x + ds + imgWidth * y) * 4;
                    if ((data[off] & 0xFF) >= treshold) {
                        if (y < miny) miny = y;
                        if (y > maxy) maxy = y;
                    }
                }
            }
        }

        if (miny > maxy) {
            // No pixels passed threshold — use full height as fallback.
            miny = 0;
            maxy = pxheight - 1;
        }

        final int contentHeight  = maxy + 1 - miny;
        final int adjustedBaseY  = rawBaseY - miny;
        int       maxGlyphWidth  = 0;

        // ── Pass 2: collect per-glyph pixel arrays ────────────────────────────
        final GlyphDef[] glyphDefs = new GlyphDef[nGlyphs];

        for (int ci = 0; ci < nGlyphs; ci++) {
            final int  ds         = regions.get(ci)[0];
            final int  de         = regions.get(ci)[1];
            final int  glyphWidth = de - ds;
            final char ch         = chars.charAt(ci);
            final boolean secondary = seconds.indexOf(ch) >= 0;
            final int jsonBonus   = bonusMap.getOrDefault(ch, 0);

            int bonus = jsonBonus;
            // Pre-size with an initial capacity; exact count unknown until scan.
            final List<Integer> pixList = new ArrayList<>(glyphWidth * contentHeight);

            for (int x = 0; x < glyphWidth; x++) {
                for (int y = 0; y < contentHeight; y++) {
                    final int off = (x + ds + imgWidth * (y + miny)) * 4;
                    final int strength = data[off] & 0xFF;   // R channel = match strength
                    if (strength >= treshold) {
                        pixList.add(x);
                        pixList.add(y);
                        pixList.add(strength);
                        if (shadow) {
                            pixList.add(data[off + 1] & 0xFF); // G channel = shadow strength
                        }
                        bonus += BONUS_PER_PIXEL;
                    }
                }
            }

            final int[] pixels = new int[pixList.size()];
            for (int k = 0; k < pixels.length; k++) pixels[k] = pixList.get(k);

            glyphDefs[ci] = new GlyphDef(ch, glyphWidth, secondary, bonus, pixels);
            if (glyphWidth > maxGlyphWidth) maxGlyphWidth = glyphWidth;
        }

        return new FontDef(glyphDefs, maxGlyphWidth, contentHeight, adjustedBaseY,
                           spaceWidth, shadow, BONUS_PER_PIXEL, fontR, fontG, fontB);
    }

    // =========================================================================
    // Private — unblending pipeline (steps 1 & 2, unchanged)
    // =========================================================================

    private static EdokitImage unblendTrans(EdokitImage img,
                                             boolean shadow,
                                             int r, int g, int b) {
        final int    pxlum = r + g + b;
        final byte[] out   = new byte[img.data.length];
        for (int i = 0; i < img.data.length; i += 4) {
            final int matchStr = img.data[i + 3] & 0xFF;
            final int shadowStr;
            if (shadow && pxlum > 0) {
                final int lum = (img.data[i]     & 0xFF)
                              + (img.data[i + 1] & 0xFF)
                              + (img.data[i + 2] & 0xFF);
                shadowStr = Math.min(255, lum * 255 / pxlum);
            } else {
                shadowStr = matchStr;
            }
            out[i]     = (byte) matchStr;
            out[i + 1] = (byte) shadowStr;
            out[i + 2] = (byte) matchStr;
            out[i + 3] = (byte) 0xFF;
        }
        return new EdokitImage(img.width, img.height, out);
    }

    private static EdokitImage unblendBlackBackground(EdokitImage img,
                                                       int r, int g, int b) {
        final byte[] out = new byte[img.data.length];
        for (int i = 0; i < img.data.length; i += 4) {
            final int rp = img.data[i]     & 0xFF;
            final int gp = img.data[i + 1] & 0xFF;
            final int bp = img.data[i + 2] & 0xFF;
            final int maxDif = Math.max(Math.abs(rp - r),
                               Math.max(Math.abs(gp - g), Math.abs(bp - b)));
            final int strength = 255 - maxDif;
            out[i]     = (byte) strength;
            out[i + 1] = (byte) strength;
            out[i + 2] = (byte) strength;
            out[i + 3] = (byte) 0xFF;
        }
        return new EdokitImage(img.width, img.height, out);
    }

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
            final double[] col = decompose2col(rp, gp, bp, r, g, b, r2, g2, b2);
            final int matchStr;
            final int shadowStr;
            if (shadow) {
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

    private static double[] decompose2col(double rp, double gp, double bp,
                                          double r1, double g1, double b1,
                                          double r2, double g2, double b2) {
        double r3 = g1 * b2 - g2 * b1;
        double g3 = b1 * r2 - b2 * r1;
        double b3 = r1 * g2 - r2 * g1;
        final double len = Math.sqrt(r3 * r3 + g3 * g3 + b3 * b3);
        if (len < 1e-10) return new double[]{0.0, 0.0, 0.0};
        final double norm = 255.0 / len;
        r3 *= norm; g3 *= norm; b3 *= norm;
        return decompose3col(rp, gp, bp, r1, g1, b1, r2, g2, b2, r3, g3, b3);
    }

    private static double[] decompose3col(double rp, double gp, double bp,
                                          double r1, double g1, double b1,
                                          double r2, double g2, double b2,
                                          double r3, double g3, double b3) {
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
        if (Math.abs(det) < 1e-10) return new double[]{0.0, 0.0, 0.0};
        return new double[]{
            (A * rp + D * gp + G * bp) / det,
            (B * rp + E * gp + H * bp) / det,
            (C * rp + F * gp + I * bp) / det
        };
    }

    // =========================================================================
    // Private — boundary scanner
    // =========================================================================

    /**
     * Reads the boundary marker row (last row of {@code unblended}) and appends
     * one {@code {ds, de}} int-pair to {@code regions} per discovered glyph run.
     *
     * <p>A pixel in the last row marks a glyph column when {@code R==255 AND A==255};
     * anything else is a gap.  This is Alt1's {@code generateFont()} detection loop.
     */
    private static void scanGlyphBoundaries(EdokitImage unblended,
                                             String chars,
                                             List<int[]> regions,
                                             String fontName) {
        final int    imgWidth = unblended.width;
        final int    pxheight = unblended.height - 1;
        final byte[] data     = unblended.data;

        boolean inGlyph    = false;
        int     glyphStart = 0;

        for (int x = 0; x <= imgWidth; x++) {
            final boolean marker;
            if (x < imgWidth) {
                final int off = (x + imgWidth * pxheight) * 4;
                marker = ((data[off]     & 0xFF) == 255)
                      && ((data[off + 3] & 0xFF) == 255);
            } else {
                marker = false;
            }

            if (marker && !inGlyph) {
                inGlyph    = true;
                glyphStart = x;
            } else if (!marker && inGlyph) {
                inGlyph = false;
                final int width = x - glyphStart;
                if (width > 0 && regions.size() < chars.length()) {
                    regions.add(new int[]{glyphStart, x});
                }
            }
        }

        if (regions.size() == chars.length()) {
            System.out.printf("[FontSheet] \"%s\": mapped %d/%d glyphs from boundary row.%n",
                    fontName, regions.size(), chars.length());
        } else {
            System.err.printf(
                    "[FontSheet] Warning: \"%s\" — boundary row yielded %d glyph(s) "
                    + "but \"chars\" declares %d.%n",
                    fontName, regions.size(), chars.length());
        }
    }

    // =========================================================================
    // Private — JSON accessors
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

    private static int[] getColorOrDefault(JsonObject obj) {
        final JsonElement el = obj.get("color");
        if (el != null && !el.isJsonNull() && el.isJsonArray()) {
            final JsonArray arr = el.getAsJsonArray();
            if (arr.size() >= 3) {
                return new int[]{arr.get(0).getAsInt(),
                                  arr.get(1).getAsInt(),
                                  arr.get(2).getAsInt()};
            }
        }
        return new int[]{255, 255, 255};
    }

    /**
     * Reads the optional {@code "bonus"} JSON object ({@code {"char": nudgeInt, ...}})
     * into a {@code Character → Integer} map.  Characters not listed default to 0.
     */
    private static Map<Character, Integer> getCharBonusMap(JsonObject obj) {
        final Map<Character, Integer> map = new LinkedHashMap<>();
        final JsonElement el = obj.get("bonus");
        if (el != null && !el.isJsonNull() && el.isJsonObject()) {
            for (Map.Entry<String, JsonElement> e : el.getAsJsonObject().entrySet()) {
                if (!e.getKey().isEmpty()) {
                    map.put(e.getKey().charAt(0), e.getValue().getAsInt());
                }
            }
        }
        return Collections.unmodifiableMap(map);
    }

    private static int clamp255(double v) {
        return (int) Math.max(0.0, Math.min(255.0, v));
    }

    // =========================================================================
    // Nested value types (Alt1 FontDefinition / Charinfo equivalents)
    // =========================================================================

    /**
     * Processed font descriptor — the Alt1 {@code FontDefinition} equivalent.
     *
     * <p>All fields are derived during {@link FontSheet#load} and are immutable.
     *
     * @param chars        processed glyph list in sprite-sheet left-to-right order
     * @param width        max glyph pixel width across all chars
     * @param height       tight content height ({@code maxy + 1 - miny})
     * @param basey        adjusted baseline ({@code rawBaseY - miny})
     * @param spaceWidth   advance for the space character
     * @param shadow       font carries a black drop-shadow
     * @param bonusPerPixel hardcoded to 5 (matches Alt1)
     * @param fontR/G/B    declared font colour from JSON {@code "color"}
     */
    public record FontDef(GlyphDef[] chars,
                          int width,
                          int height,
                          int basey,
                          int spaceWidth,
                          boolean shadow,
                          int bonusPerPixel,
                          int fontR,
                          int fontG,
                          int fontB) {

        /**
         * Linear search for a {@link GlyphDef} by character.
         *
         * @param ch the character to find
         * @return the matching {@link GlyphDef}, or {@code null} if not present
         */
        public GlyphDef findGlyph(char ch) {
            for (GlyphDef g : chars) {
                if (g.chr() == ch) return g;
            }
            return null;
        }
    }

    /**
     * Processed per-character descriptor — the Alt1 {@code Charinfo} equivalent.
     *
     * <h3>pixels[] layout</h3>
     * Without shadow: {@code [x, y, matchStrength, x, y, matchStrength, ...]} (stride 3).<br>
     * With shadow:    {@code [x, y, matchStrength, shadowStrength, ...]} (stride 4).<br>
     * Coordinates are relative to the glyph's left edge ({@code x = 0}) and to the
     * top of the content bounding box ({@code y = 0}).
     *
     * @param chr       the character this glyph represents
     * @param width     pixel width from the boundary marker row
     * @param secondary {@code true} if listed in the JSON {@code "seconds"} field
     * @param bonus     accumulated score offset: {@code jsonNudge + bonusPerPixel × N}
     * @param pixels    flat pixel array as described above
     */
    public record GlyphDef(char chr,
                            int width,
                            boolean secondary,
                            int bonus,
                            int[] pixels) {
        @Override
        public String toString() {
            final int stride = 3; // shadow unknown here; approximate
            return String.format("GlyphDef{chr='%c', w=%d, secondary=%b, bonus=%d, pixels=%d}",
                    chr, width, secondary, bonus, pixels.length / stride);
        }
    }
}
