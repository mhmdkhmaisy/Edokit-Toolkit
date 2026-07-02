package edokit.ocr;

import edokit.base.models.EdokitImage;

import java.util.Map;

/**
 * RasterOCR — Pixel-exact raster text scanner for RuneScape's UI fonts.
 *
 * <h2>What this is</h2>
 * A pure-Java, no-external-engine OCR implementation tuned for RuneScape's
 * interface fonts.  These fonts use hard-edged, shadow-composited glyphs on
 * a predictable coloured background — a property that makes pixel-exact colour
 * distance matching far more reliable and faster than probabilistic OCR.
 *
 * <p>The engine mirrors Alt1's {@code ocr} module: it walks a horizontal scan
 * line, tries every known glyph at each position, and advances the cursor
 * by the glyph's tracked advance width on a successful match.
 *
 * <h2>Performance contract</h2>
 * <ul>
 *   <li>No objects are allocated inside the hot pixel-checking loops.
 *       All intermediate values are stack-local {@code int} primitives.</li>
 *   <li>Pixel addressing uses {@link EdokitImage#pixelOffset} (a final method
 *       the JIT inlines to a bare multiply-add).</li>
 *   <li>The early-exit {@code coldif} check short-circuits the inner loops the
 *       moment any pixel exceeds the colour distance — no wasted comparisons.</li>
 * </ul>
 *
 * <h2>Thread safety</h2>
 * {@code RasterOCR} is stateless — all state lives in the caller-supplied
 * {@link EdokitImage} and {@link FontSheet} arguments.  Multiple threads may
 * call {@link #readLine} concurrently on different source images without
 * synchronisation.
 */
public final class RasterOCR {

    // =========================================================================
    // readLine
    // =========================================================================

    /**
     * Reads a single horizontal text line from {@code source} starting at
     * {@code (startX, startY)}, using the glyph templates in {@code font}.
     *
     * <h3>Algorithm</h3>
     * <pre>
     *   cursorX ← startX
     *   while cursorX &lt; source.width:
     *     for each glyph G in font.glyphMap (declaration order):
     *       if matchesGlyph(source, cursorX, startY, G, maxColorDistance):
     *         if gap since last match ≥ font.spaceWidth → append ' '
     *         append G.character
     *         cursorX += G.advX
     *         gapPixels ← 0
     *         break
     *     else (no glyph matched):
     *       cursorX += 1
     *       gapPixels += 1
     *       if gapPixels &gt; END_OF_LINE_THRESHOLD → stop (end of text)
     * </pre>
     *
     * <h3>Colour distance ({@code coldif})</h3>
     * For each non-transparent pixel in the font glyph template, the
     * per-channel absolute difference against the corresponding source pixel
     * is computed:
     * <pre>
     *   deltaR = |fontR − srcR|
     *   deltaG = |fontG − srcG|
     *   deltaB = |fontB − srcB|
     * </pre>
     * A glyph matches only if <em>all</em> visible pixels satisfy
     * {@code deltaR ≤ maxColorDistance ∧ deltaG ≤ maxColorDistance ∧ deltaB ≤ maxColorDistance}.
     * The check short-circuits on the first failing pixel.
     *
     * <h3>Space detection</h3>
     * When the horizontal gap between two consecutive matched glyphs reaches or
     * exceeds {@link FontSheet#spaceWidth} pixels, a single {@code ' '} is
     * inserted before the next character.  Trailing gaps (after the last glyph)
     * do not produce trailing spaces.
     *
     * <h3>Termination</h3>
     * Scanning stops when the cursor reaches {@code source.width} or when
     * {@code NO_MATCH_STOP_PIXELS} consecutive non-matching pixels have been
     * seen — the latter heuristic prevents scanning through large empty regions
     * of the frame after the text ends.
     *
     * @param source           the live frame to read from
     * @param startX           left edge of the scan region (inclusive, 0-based)
     * @param startY           top edge of the scan region (inclusive, 0-based);
     *                         caller should account for {@link FontSheet#baseY}
     *                         when positioning this relative to a UI element
     * @param font             the font schema and sprite sheet to match against
     * @param maxColorDistance maximum per-channel RGB delta for a pixel to be
     *                         considered "matching" (typically 30–80; lower =
     *                         stricter)
     * @return the recognised text string; empty if no glyphs were found
     */
    public String readLine(EdokitImage source,
                           int startX,
                           int startY,
                           FontSheet font,
                           int maxColorDistance) {

        // ── Guard: scan region must have at least 1 valid pixel row ───────────
        if (startX >= source.width || startY >= source.height || startX < 0 || startY < 0) {
            return "";
        }

        final StringBuilder result   = new StringBuilder(32);
        final byte[]        srcData  = source.data;
        final int           srcWidth = source.width;

        // Snapshot of the glyph map entry set — stable iteration, no allocation.
        final Map<Character, FontSheet.GlyphDefinition> glyphs = font.glyphMap();

        int cursorX   = startX;
        int gapPixels = 0;       // pixels advanced since last confirmed glyph match

        // How many consecutive non-matching pixels before we declare end-of-line.
        // Set to a generous value (3× the largest typical advance width) so that
        // proportional fonts with wide inter-character gaps are handled correctly.
        final int stopAfter = Math.max(font.spaceWidth * 4, 16);

        outer:
        while (cursorX < srcWidth) {

            // ── Try every glyph at the current cursor position ─────────────────
            for (Map.Entry<Character, FontSheet.GlyphDefinition> entry : glyphs.entrySet()) {
                final char                     ch    = entry.getKey();
                final FontSheet.GlyphDefinition glyph = entry.getValue();

                // Bounds guard: skip if glyph would extend beyond source edges.
                if (cursorX + glyph.width()  > srcWidth
                 || startY  + glyph.height() > source.height) {
                    continue;
                }

                if (glyphMatchesColdif(srcData, source.width,
                                       cursorX, startY,
                                       font.fontImage.data, font.fontImage.width,
                                       glyph, maxColorDistance)) {

                    // ── Successful match ───────────────────────────────────────
                    // Insert a space if a large enough gap preceded this glyph.
                    if (!result.isEmpty() && gapPixels >= font.spaceWidth) {
                        result.append(' ');
                    }

                    result.append(ch);
                    cursorX   += glyph.advX();
                    gapPixels  = 0;
                    continue outer; // restart outer loop at updated cursor
                }
            }

            // ── No glyph matched at this position ──────────────────────────────
            cursorX++;
            gapPixels++;

            // End-of-line heuristic: stop scanning after a long unmatched run.
            if (gapPixels >= stopAfter) break;
        }

        return result.toString();
    }

    // =========================================================================
    // removeTextPixels
    // =========================================================================

    /**
     * Zeroes out the source pixels that form {@code verifiedText} starting at
     * {@code (startX, startY)}, using the same cursor-advance logic as
     * {@link #readLine}.
     *
     * <h3>Purpose</h3>
     * After reading an overlay label (e.g. a yellow countdown timer on a buff
     * icon), the digit pixels must be erased from the source buffer before
     * downstream colour-sampling routines inspect the same region.  This
     * prevents the text pixels from biasing colour-distance checks performed by
     * the buff-detection pipeline.
     *
     * <h3>What "zero out" means</h3>
     * Only pixels where the corresponding font template pixel has
     * {@code alpha > 0} are cleared.  Each such pixel in {@code source.data}
     * is set to {@code [R=0, G=0, B=0, A=0]} — fully transparent black.  This
     * matches Alt1's {@code removeTextPixels} masking technique.
     *
     * <h3>Cursor advance</h3>
     * The cursor advances by {@link FontSheet.GlyphDefinition#advX()} for each
     * matched character, mirroring the exact advance used during {@link #readLine}.
     * Space characters ({@code ' '}) advance by {@link FontSheet#spaceWidth}.
     * Unknown characters (not in the glyph map) advance by 1 pixel.
     *
     * @param source        the live frame to modify in-place
     * @param startX        left edge of the text region (same value used in
     *                      the corresponding {@link #readLine} call)
     * @param startY        top edge of the text region
     * @param verifiedText  the string previously returned by {@link #readLine};
     *                      must be the exact sequence of characters to erase
     * @param font          the font schema whose glyph bounding boxes define
     *                      which pixels to zero out
     */
    public void removeTextPixels(EdokitImage source,
                                 int startX,
                                 int startY,
                                 String verifiedText,
                                 FontSheet font) {

        if (verifiedText == null || verifiedText.isEmpty()) return;
        if (startX < 0 || startY < 0
         || startX >= source.width || startY >= source.height) return;

        final byte[] srcData     = source.data;
        final byte[] fontData    = font.fontImage.data;
        final int    srcWidth    = source.width;
        final int    fontImgWidth = font.fontImage.width;

        int cursorX = startX;

        for (int ci = 0; ci < verifiedText.length(); ci++) {
            final char ch = verifiedText.charAt(ci);

            // ── Space character: advance without zeroing ───────────────────────
            if (ch == ' ') {
                cursorX += font.spaceWidth;
                continue;
            }

            final FontSheet.GlyphDefinition glyph = font.getGlyph(ch);

            // ── Unknown character: advance 1 pixel and skip ────────────────────
            if (glyph == null) {
                cursorX++;
                continue;
            }

            final int gw = glyph.width();
            final int gh = glyph.height();
            final int gx = glyph.x();
            final int gy = glyph.y();

            // ── Zero out every visible (alpha > 0) font pixel in source ────────
            // All index arithmetic reuses primitive locals — zero heap allocation.
            for (int dy = 0; dy < gh; dy++) {
                // Pre-compute row start offsets once per row.
                final int fontRowStart = (4 * gx) + (4 * fontImgWidth * (gy + dy));
                final int srcY         = startY + dy;

                // Bounds: skip rows that fall outside the source image vertically.
                if (srcY >= source.height) break;

                final int srcRowStart = (4 * cursorX) + (4 * srcWidth * srcY);

                for (int dx = 0; dx < gw; dx++) {
                    // Bounds: skip columns outside the source image horizontally.
                    final int srcX = cursorX + dx;
                    if (srcX >= srcWidth) break;

                    final int fontOffset = fontRowStart + (dx * 4);
                    final int fontAlpha  = fontData[fontOffset + 3] & 0xFF;

                    // Only overwrite pixels where the font template is visible.
                    if (fontAlpha == 0) continue;

                    final int srcOffset = srcRowStart + (dx * 4);
                    srcData[srcOffset]     = 0; // R → 0
                    srcData[srcOffset + 1] = 0; // G → 0
                    srcData[srcOffset + 2] = 0; // B → 0
                    srcData[srcOffset + 3] = 0; // A → 0  (fully transparent)
                }
            }

            cursorX += glyph.advX();
        }
    }

    // =========================================================================
    // Internal: coldif pixel-level glyph match check
    // =========================================================================

    /**
     * Returns {@code true} if the glyph template, positioned with its top-left
     * corner at {@code (srcX, srcY)} in the source image, matches within
     * {@code maxColorDistance} on every non-transparent template pixel.
     *
     * <h3>Why separate data/width parameters instead of EdokitImage</h3>
     * Passing the raw {@code byte[]} arrays and stride widths directly avoids
     * two object-field loads ({@code .data}, {@code .width}) per loop iteration.
     * At 30 FPS with hundreds of glyph candidates per frame, this keeps the
     * inner loop tight enough for the JIT to fully unroll short glyphs.
     *
     * <h3>Short-circuit guarantee</h3>
     * The method returns {@code false} the instant any visible pixel's
     * per-channel delta exceeds {@code maxColorDistance}.  Only fully matching
     * glyphs traverse their entire bounding box.
     *
     * <h3>Minimum visible-pixel requirement</h3>
     * A glyph with zero visible pixels (fully transparent sprite sheet region)
     * always returns {@code false} — it must not match every candidate position.
     *
     * @param srcData          {@link EdokitImage#data} of the source frame
     * @param srcStride        {@link EdokitImage#width} of the source frame
     * @param srcX             left edge of the candidate match in source space
     * @param srcY             top edge of the candidate match in source space
     * @param fontData         {@link EdokitImage#data} of the font sprite sheet
     * @param fontStride       {@link EdokitImage#width} of the font sprite sheet
     * @param glyph            glyph bounding box and origin in the sprite sheet
     * @param maxColorDistance maximum per-channel delta for a passing pixel
     * @return {@code true} if all visible glyph pixels are within the distance
     */
    private static boolean glyphMatchesColdif(byte[] srcData,  int srcStride,
                                              int    srcX,      int srcY,
                                              byte[] fontData,  int fontStride,
                                              FontSheet.GlyphDefinition glyph,
                                              int maxColorDistance) {
        final int gw = glyph.width();
        final int gh = glyph.height();
        final int gx = glyph.x();
        final int gy = glyph.y();

        // Pre-compute row-start offsets for the first row of each operand.
        // Inside the loop we increment by stride (4*width bytes) each row.
        int fontRowBase = (4 * gx) + (4 * fontStride * gy);
        int srcRowBase  = (4 * srcX) + (4 * srcStride * srcY);

        int visiblePixels = 0; // must have at least one to be a valid match

        for (int dy = 0; dy < gh; dy++, fontRowBase += fontStride * 4,
                                        srcRowBase  += srcStride  * 4) {
            for (int dx = 0; dx < gw; dx++) {

                // ── Read font pixel ────────────────────────────────────────────
                final int fOff  = fontRowBase + (dx * 4);
                final int fontA = fontData[fOff + 3] & 0xFF;

                // Transparent font pixel → skip (not part of glyph shape).
                if (fontA == 0) continue;

                visiblePixels++;

                final int fontR = fontData[fOff]     & 0xFF;
                final int fontG = fontData[fOff + 1] & 0xFF;
                final int fontB = fontData[fOff + 2] & 0xFF;

                // ── Read source pixel ──────────────────────────────────────────
                final int sOff = srcRowBase + (dx * 4);
                final int srcR = srcData[sOff]     & 0xFF;
                final int srcG = srcData[sOff + 1] & 0xFF;
                final int srcB = srcData[sOff + 2] & 0xFF;

                // ── coldif: per-channel absolute difference ────────────────────
                // Short-circuit: return false on the first out-of-range channel.
                final int deltaR = fontR - srcR;
                final int deltaG = fontG - srcG;
                final int deltaB = fontB - srcB;

                // Using conditional-free absolute value: (x ^ (x>>31)) - (x>>31)
                // is equivalent to Math.abs(x) but avoids a branch on most JITs.
                if (((deltaR ^ (deltaR >> 31)) - (deltaR >> 31)) > maxColorDistance) return false;
                if (((deltaG ^ (deltaG >> 31)) - (deltaG >> 31)) > maxColorDistance) return false;
                if (((deltaB ^ (deltaB >> 31)) - (deltaB >> 31)) > maxColorDistance) return false;
            }
        }

        // A glyph with no visible pixels must not match arbitrary positions.
        return visiblePixels > 0;
    }
}
