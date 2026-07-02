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
 * <h2>Supported JSON schemas</h2>
 * The parser handles <em>both</em> glyph-list layouts used across Alt1's font
 * assets:
 * <pre>
 *   Array form   — "glyphs": [ {"ch":"0","x":0,"y":0,"width":8,"height":8,"advx":9}, … ]
 *   Object form  — "glyphs": { "0": {"x":0,"y":0,"width":8,"height":8,"advx":9}, … }
 * </pre>
 * All top-level schema fields are optional with documented defaults so that
 * partial or minimal font definitions load without errors.
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

        final String fontName   = getStringOrDefault(root, "fontname",     "unknown");
        final int    baseY      = getIntOrDefault(root,    "basey",         0);
        final int    lineHeight = getIntOrDefault(root,    "lineheight",    0);
        final int    spaceWidth = getIntOrDefault(root,    "space_width",   4);

        // ── Parse glyph definitions ───────────────────────────────────────────
        // LinkedHashMap preserves JSON declaration order for stable readLine iteration.
        final Map<Character, GlyphDefinition> glyphMap = new LinkedHashMap<>();

        final JsonElement glyphsEl = root.get("glyphs");
        if (glyphsEl == null || glyphsEl.isJsonNull()) {
            throw new IllegalArgumentException(
                    "Font JSON for \"" + fontName + "\" contains no \"glyphs\" field.");
        }

        if (glyphsEl.isJsonArray()) {
            /*
             * Array form:
             *   "glyphs": [ {"ch":"A","x":0,"y":0,"width":7,"height":9,"advx":8}, … ]
             *
             * The "ch" field carries the character.  "id" is parsed but not stored;
             * it is an Alt1 internal index not needed for pixel-level matching.
             */
            for (JsonElement el : glyphsEl.getAsJsonArray()) {
                if (!el.isJsonObject()) continue;
                parseGlyphObject(el.getAsJsonObject(), null, glyphMap, fontName);
            }

        } else if (glyphsEl.isJsonObject()) {
            /*
             * Object / map form:
             *   "glyphs": { "A": {"x":0,"y":0,"width":7,"height":9,"advx":8}, … }
             *
             * The map key is the character string; "ch" inside the object is
             * optional and, if present, takes precedence over the key.
             */
            for (Map.Entry<String, JsonElement> entry : glyphsEl.getAsJsonObject().entrySet()) {
                if (!entry.getValue().isJsonObject()) continue;
                parseGlyphObject(entry.getValue().getAsJsonObject(),
                                 entry.getKey(), glyphMap, fontName);
            }

        } else {
            throw new IllegalArgumentException(
                    "\"glyphs\" in font \"" + fontName
                    + "\" must be a JSON array or object, got: "
                    + glyphsEl.getClass().getSimpleName());
        }

        if (glyphMap.isEmpty()) {
            throw new IllegalArgumentException(
                    "Font \"" + fontName + "\" parsed 0 valid glyph entries.");
        }

        // ── Load sprite sheet ─────────────────────────────────────────────────
        // EdokitImageLoader strips ICC/gamma profiles so pixel values are the
        // raw, uncorrected digital colour values used by the Alt1 schema.
        final EdokitImage fontImage = EdokitImageLoader.load(imgStream);

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
     * Parses one glyph JSON object and inserts it into {@code target}.
     *
     * <p>Character resolution priority:
     * <ol>
     *   <li>{@code "ch"} field inside the object (supports multi-byte chars in
     *       fonts that encode them as strings).</li>
     *   <li>{@code mapKey} — the outer object key when using map-form JSON.</li>
     * </ol>
     * Entries with no resolvable character, or with zero width/height, are
     * silently skipped (they represent internal placeholder slots).
     *
     * @param obj     the glyph JSON object
     * @param mapKey  outer key if this came from an object-form glyphs field,
     *                {@code null} for array-form
     * @param target  destination map
     * @param fontName font name for diagnostics only
     */
    private static void parseGlyphObject(JsonObject obj,
                                         String mapKey,
                                         Map<Character, GlyphDefinition> target,
                                         String fontName) {
        // Resolve character: prefer "ch" inside the object, fall back to mapKey.
        final String chStr;
        if (obj.has("ch") && !obj.get("ch").isJsonNull()) {
            chStr = obj.get("ch").getAsString();
        } else if (mapKey != null && !mapKey.isEmpty()) {
            chStr = mapKey;
        } else {
            return; // No character identity — skip
        }

        if (chStr.isEmpty()) return;
        final char ch = chStr.charAt(0);

        final int x      = getIntOrDefault(obj, "x",      0);
        final int y      = getIntOrDefault(obj, "y",      0);
        final int width  = getIntOrDefault(obj, "width",  0);
        final int height = getIntOrDefault(obj, "height", 0);

        // Skip degenerate entries (width=0 or height=0 means no renderable pixels).
        if (width <= 0 || height <= 0) return;

        // "advx" is the horizontal cursor advance after this character is drawn.
        // If absent, default to the glyph's own pixel width (monospaced fallback).
        final int advX = getIntOrDefault(obj, "advx", width);

        target.put(ch, new GlyphDefinition(x, y, width, height, advX));
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
