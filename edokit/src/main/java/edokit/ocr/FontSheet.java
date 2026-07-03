package edokit.ocr;

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
 *   <li>A {@code .fontmeta.json} descriptor that names every character, its
 *       bounding box inside the sprite sheet, and its advance width.</li>
 *   <li>A {@code data.png} sprite sheet containing all glyphs packed into a
 *       single uncompressed image (loaded via {@link EdokitImageLoader} to
 *       guarantee raw, ICC-stripped pixels).</li>
 * </ul>
 * {@code FontSheet} parses both and exposes them in a form that
 * {@link RasterOCR} can scan efficiently with zero per-frame allocation.
 *
 * <h2>Alt1 font JSON schema</h2>
 * Alt1 {@code .fontmeta.json} files store only global metadata at the root level.
 * Per-character bounding boxes are <em>not</em> present in the JSON — they are
 * derived at load time by scanning the accompanying sprite sheet for contiguous
 * non-empty pixel column runs:
 * <pre>
 *   {
 *     "basey":      7,                  // baseline y offset inside the sprite sheet
 *     "spacewidth": 3,                  // advance width for the space character
 *     "chars":      "0123456789m()hr",  // characters in left-to-right sprite order
 *     "treshold":   0.9,               // (unused by Edokit — Alt1 internal)
 *     "color":      [255, 255, 255],   // (unused by Edokit — Alt1 internal)
 *     "shadow":     true               // (unused by Edokit — Alt1 internal)
 *   }
 * </pre>
 * {@code FontSheet.load()} reads {@code "chars"} to know which character occupies
 * each glyph run, then scans the PNG column-by-column to locate each run's
 * {@code x}, {@code width}, and advance width automatically.
 *
 * <h2>Memory layout</h2>
 * The full font sprite sheet is held as a single {@link EdokitImage} whose
 * {@code byte[]} is addressed by every {@link GlyphDefinition} via its
 * {@code (x, y)} origin + {@code (width, height)} extent.  No per-glyph
 * sub-image is allocated — {@link RasterOCR} indexes directly into
 * {@link #fontImage}{@code .data} using {@link EdokitImage#pixelOffset}.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 *   try (InputStream json = getClass().getResourceAsStream("/fonts/pixel_8px_digits.fontmeta.json");
 *        InputStream img  = getClass().getResourceAsStream("/fonts/pixel_8px_digits.data.png")) {
 *       FontSheet digits = FontSheet.load(json, img);
 *   }
 * }</pre>
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
     * Default: {@code "unknown"}.
     */
    public final String fontName;

    // =========================================================================
    // Private state
    // =========================================================================

    /**
     * Character → glyph metadata.  {@link LinkedHashMap} preserves JSON
     * declaration order so that {@link RasterOCR#readLine} tries glyphs
     * in a stable, deterministic sequence.
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
     * Parses an Alt1 font schema and loads its accompanying sprite sheet.
     *
     * <p>This is the only public way to obtain a {@code FontSheet} instance.
     * Both streams are read fully inside this call; the caller is responsible
     * for closing them afterwards.
     *
     * @param jsonStream the {@code .fontmeta.json} descriptor stream; must not
     *                   be {@code null}
     * @param imgStream  the {@code data.png} sprite sheet stream; must not be
     *                   {@code null}; processed by {@link EdokitImageLoader} for
     *                   ICC-stripped raw RGBA pixels
     * @return a fully initialised {@code FontSheet}
     * @throws IOException              if either stream cannot be read, or the
     *                                  image format is unsupported
     * @throws IllegalArgumentException if the JSON is malformed or contains no
     *                                  parseable glyph entries
     */
    public static FontSheet load(InputStream jsonStream, InputStream imgStream)
            throws IOException {

        // ── Parse JSON descriptor ─────────────────────────────────────────────
        final JsonObject root;
        try (Reader reader = new InputStreamReader(jsonStream, StandardCharsets.UTF_8)) {
            root = JsonParser.parseReader(reader).getAsJsonObject();
        }

        // "fontname" is an optional label; absent from all shipped Alt1 assets.
        final String fontName   = getStringOrDefault(root, "fontname",   "Alt1 Font");
        // "basey" — vertical baseline offset inside the sprite sheet (pixels from top).
        final int    baseY      = getIntOrDefault(root,    "basey",       0);
        // "lineheight" — multi-line step; absent from most Alt1 assets, default 0.
        final int    lineHeight = getIntOrDefault(root,    "lineheight",  0);
        // Alt1 uses "spacewidth" (no underscore) for the space-character advance.
        final int    spaceWidth = getIntOrDefault(root,    "spacewidth",  4);

        // ── Read the "chars" character-order string ───────────────────────────
        // Alt1 font JSONs carry NO per-character x/y/width/height data.
        // "chars" is the sole source of character identity: chars.charAt(i) names
        // the i-th glyph run in the sprite sheet (left-to-right order).
        final JsonElement charsEl = root.get("chars");
        if (charsEl == null || charsEl.isJsonNull()) {
            throw new IllegalArgumentException(
                    "Font JSON for \"" + fontName + "\" is missing the required "
                    + "\"chars\" field.  This field must list every character rendered "
                    + "in the accompanying sprite sheet, in left-to-right order.");
        }
        final String charsStr = charsEl.getAsString();
        if (charsStr.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font JSON for \"" + fontName + "\" has an empty \"chars\" string — "
                    + "at least one character must be listed.");
        }

        // ── Load sprite sheet ─────────────────────────────────────────────────
        // EdokitImageLoader strips ICC/gamma profiles so pixel values are the
        // raw, uncorrected digital colour values used by the Alt1 schema.
        final EdokitImage fontImage = EdokitImageLoader.load(imgStream);

        // ── Derive per-glyph bounds by scanning the sprite sheet ──────────────
        // Glyphs are packed into a single horizontal strip with no embedded
        // coordinate data.  scanGlyphs() locates each glyph by finding
        // contiguous non-empty pixel column runs and assigns them to characters
        // in "chars" declaration order.
        final Map<Character, GlyphDefinition> glyphMap = new LinkedHashMap<>();
        scanGlyphs(fontImage, charsStr, glyphMap, fontName, spaceWidth);

        if (glyphMap.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\" yielded 0 glyph entries after sprite "
                    + "sheet scan.  Verify that the PNG contains visible glyph pixels.");
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
     * <p>Iteration order reflects the declaration order of glyphs in the
     * original JSON file (preserved by the {@link LinkedHashMap} backing store).
     * {@link RasterOCR#readLine} uses this to try glyphs in a stable sequence.
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
    // Private parsing helpers
    // =========================================================================

    /**
     * Scans {@code img} left-to-right for contiguous non-empty pixel column runs
     * and maps each discovered run to the corresponding character in {@code chars}.
     *
     * <h3>Empty-column strategy</h3>
     * <ul>
     *   <li><b>Alpha-channel fonts</b> — if any pixel in the entire sprite sheet
     *       has {@code A == 0}, the sheet uses a transparent background.  A column
     *       is then empty when every pixel in it has {@code A == 0}.</li>
     *   <li><b>Opaque-background fonts</b> — all pixels have {@code A == 0xFF}
     *       (e.g. sheets decoded from 3-channel PNGs).  A column is empty when
     *       every pixel reads solid black ({@code R == G == B == 0}).</li>
     * </ul>
     *
     * <h3>Advance width</h3>
     * The Alt1 JSON carries no per-character advance width.  Each glyph's
     * {@code advX} is computed as {@code pixelWidth + 1}, matching the one-pixel
     * inter-character gap Alt1's renderer inserts between consecutive glyphs.
     *
     * @param img        the font sprite sheet in raw RGBA format
     * @param chars      ordered character string from the {@code "chars"} JSON field
     * @param target     destination map (insertion order preserved via LinkedHashMap)
     * @param fontName   font identifier used in diagnostic log messages only
     * @param spaceWidth space-character advance width from the JSON descriptor
     */
    private static void scanGlyphs(EdokitImage img,
                                    String chars,
                                    Map<Character, GlyphDefinition> target,
                                    String fontName,
                                    int spaceWidth) {
        final int    imgWidth  = img.width;
        final int    imgHeight = img.height;
        final byte[] data      = img.data;

        // Choose empty-column strategy once, based on whether any transparent
        // pixel exists anywhere in the sprite sheet.
        final boolean useAlpha = hasAnyTransparentPixel(data);

        int charIndex = 0;
        int x         = 0;

        while (x < imgWidth && charIndex < chars.length()) {

            // ── Skip inter-glyph gap (empty columns) ──────────────────────────
            while (x < imgWidth && isColumnEmpty(data, x, imgWidth, imgHeight, useAlpha)) {
                x++;
            }
            if (x >= imgWidth) break;

            // ── Accumulate glyph body (consecutive non-empty columns) ──────────
            final int glyphStartX = x;
            while (x < imgWidth && !isColumnEmpty(data, x, imgWidth, imgHeight, useAlpha)) {
                x++;
            }
            final int glyphWidth = x - glyphStartX;
            if (glyphWidth <= 0) continue;

            // ── Commit GlyphDefinition ─────────────────────────────────────────
            // y = 0: all glyphs sit at the top of the single-row sprite strip.
            // advX = pixelWidth + 1 mirrors Alt1's one-pixel inter-glyph cursor advance.
            // Fallback: if width is somehow 0 (degenerate), use spaceWidth as advX.
            final int advX = (glyphWidth > 0) ? (glyphWidth + 1) : (spaceWidth + 1);
            final char ch  = chars.charAt(charIndex);
            target.put(ch, new GlyphDefinition(glyphStartX, 0, glyphWidth, imgHeight, advX));
            charIndex++;
        }

        if (charIndex < chars.length()) {
            System.err.printf(
                    "[FontSheet] Warning: \"%s\" — sprite sheet yielded %d glyph(s) "
                    + "but \"chars\" declares %d.  The PNG may be clipped or "
                    + "contain fewer glyph runs than expected.%n",
                    fontName, charIndex, chars.length());
        }
    }

    /**
     * Returns {@code true} if any pixel in the flat RGBA byte array has an alpha
     * value of {@code 0} (fully transparent).
     *
     * <p>Used once per {@link #load} call to select the background-detection
     * strategy for {@link #isColumnEmpty}: transparent-background sheet if any
     * transparent pixel is found; opaque-black-background sheet otherwise.
     *
     * @param data flat RGBA byte array from an {@link EdokitImage}
     * @return {@code true} if at least one pixel has {@code A == 0}
     */
    private static boolean hasAnyTransparentPixel(byte[] data) {
        // Alpha is at index 3 of every 4-byte pixel (RGBA layout).
        for (int i = 3; i < data.length; i += 4) {
            if ((data[i] & 0xFF) == 0) return true;
        }
        return false;
    }

    /**
     * Returns {@code true} if every pixel in column {@code col} of the sprite
     * sheet is considered empty (not part of any rendered glyph).
     *
     * <p>The emptiness criterion depends on the background strategy:
     * <ul>
     *   <li>{@code useAlpha = true}: pixel is empty when {@code A == 0}.</li>
     *   <li>{@code useAlpha = false}: pixel is empty when
     *       {@code R == 0 && G == 0 && B == 0} (solid black background).</li>
     * </ul>
     *
     * <p>No objects are allocated here — every access is a direct index into
     * the backing {@code byte[]} with a branch-free unsigned comparison.
     *
     * @param data      flat RGBA byte array from an {@link EdokitImage}
     * @param col       x-coordinate of the column to test (0-based)
     * @param imgWidth  sprite sheet width in pixels (used as row stride)
     * @param imgHeight sprite sheet height in pixels (number of rows to check)
     * @param useAlpha  {@code true} to test the alpha channel;
     *                  {@code false} to test RGB channels against black
     * @return {@code true} if the entire column contains no glyph pixels
     */
    private static boolean isColumnEmpty(byte[] data,
                                         int col,
                                         int imgWidth,
                                         int imgHeight,
                                         boolean useAlpha) {
        for (int row = 0; row < imgHeight; row++) {
            final int off = (col + imgWidth * row) * 4;
            if (useAlpha) {
                // Transparent-background: any non-zero alpha = glyph pixel present.
                if ((data[off + 3] & 0xFF) != 0) return false;
            } else {
                // Opaque-background: any non-zero RGB channel = glyph pixel present.
                if ((data[off]     & 0xFF) != 0) return false;
                if ((data[off + 1] & 0xFF) != 0) return false;
                if ((data[off + 2] & 0xFF) != 0) return false;
            }
        }
        return true;
    }

    // ── Null-safe JSON primitive accessors ────────────────────────────────────

    private static int getIntOrDefault(JsonObject obj, String key, int def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsInt() : def;
    }

    private static String getStringOrDefault(JsonObject obj, String key, String def) {
        final JsonElement el = obj.get(key);
        return (el != null && !el.isJsonNull()) ? el.getAsString() : def;
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
     * @param width  glyph pixel width (bounding box, may include side-bearing)
     * @param height glyph pixel height (bounding box)
     * @param advX   horizontal cursor advance: how many pixels to move the write
     *               cursor after this character is consumed (≥ 1)
     */
    public record GlyphDefinition(int x, int y, int width, int height, int advX) {

        @Override
        public String toString() {
            return String.format("GlyphDef{x=%d, y=%d, w=%d, h=%d, advX=%d}",
                    x, y, width, height, advX);
        }
    }
}
