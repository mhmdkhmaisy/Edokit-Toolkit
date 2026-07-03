package edokit.ocr;

import edokit.base.models.EdokitImage;

/**
 * RasterOCR — Alt1-faithful raster text scanner for RuneScape's UI fonts.
 *
 * <h2>Scoring model</h2>
 * This engine mirrors Alt1's {@code readChar} / {@code canblend} pipeline from
 * {@code src/ocr/index.ts} exactly.  Rather than comparing template pixel colours
 * to screen colours directly, it asks a more robust question:
 *
 * <blockquote><i>"Could this screen pixel be the result of blending the font
 * colour with <em>some</em> plausible background colour?"</i></blockquote>
 *
 * The answer is computed by {@link #canblend}: extrapolate from the screen pixel
 * away from the font colour by the leverage factor implied by the template pixel's
 * match strength.  If the extrapolated background lands inside {@code [0,255]³}
 * the penalty is 0; otherwise the penalty equals how far outside it lands.
 *
 * <h2>Per-glyph scoring</h2>
 * For each candidate glyph:
 * <ul>
 *   <li>{@code score} — raw penalty sum; short-circuits at 400.</li>
 *   <li>{@code sizeScore = -bonus + Σ(penalty − bonusPerPixel)} — biased so larger
 *       glyphs with more pixels and lower per-pixel error win over smaller,
 *       ambiguous ones.  The winner is the glyph with the lowest sizeScore.</li>
 * </ul>
 * A glyph is rejected if fewer than 60 % of its template pixels were in-bounds
 * and checked (or all pixels for very small glyphs with ≤ 6 template points).
 *
 * <h2>Secondary characters</h2>
 * Glyphs marked {@link FontSheet.GlyphDef#secondary()} (from the JSON
 * {@code "seconds"} field) are tried alongside primary glyphs in
 * {@link #readLine}.  They participate in the scoring competition on equal terms
 * — but because they typically have fewer template pixels their natural bonus is
 * smaller, so they only win when nothing else fits.
 *
 * <h2>Thread safety</h2>
 * {@code RasterOCR} is stateless.  Multiple threads may call {@link #readLine}
 * and {@link #removeTextPixels} concurrently without synchronisation.
 */
public final class RasterOCR {

    // =========================================================================
    // readLine
    // =========================================================================

    /**
     * Reads a single horizontal text line from {@code source} starting at
     * {@code (startX, startY)}, using the processed glyph data in {@code font}.
     *
     * <h3>Algorithm</h3>
     * <pre>
     *   cursorX ← startX
     *   while cursorX &lt; source.width:
     *     match ← readChar(source, font, cursorX, startY, allowSecondary=true)
     *     if match ≠ null:
     *       if gap ≥ font.spaceWidth → append ' '
     *       append match.chr
     *       cursorX += match.width
     *       gapPixels ← 0
     *     else:
     *       cursorX += 1; gapPixels += 1
     *       if gapPixels ≥ stopAfter → stop
     * </pre>
     *
     * <h3>Space detection</h3>
     * When {@code gapPixels} reaches {@link FontSheet.FontDef#spaceWidth()} between
     * two matched glyphs a single {@code ' '} is inserted.  Trailing gaps do not
     * produce trailing spaces.
     *
     * <h3>Termination</h3>
     * Scanning stops when the cursor reaches {@code source.width} or after
     * {@code max(spaceWidth × 3, 16)} consecutive unmatched pixels.
     *
     * @param source  the live frame to read from
     * @param startX  left edge of the scan region (inclusive)
     * @param startY  top of the glyph content rows in the source image.
     *                This is the top-of-content coordinate, <em>not</em> the
     *                baseline — i.e. the same Y the caller has been using to
     *                position the text region within the frame.
     * @param font    the loaded {@link FontSheet}
     * @return the recognised text; empty string if no glyphs matched
     */
    public String readLine(EdokitImage source,
                           int startX,
                           int startY,
                           FontSheet font) {

        if (startX < 0 || startY < 0
                || startX >= source.width || startY >= source.height) {
            return "";
        }

        final FontSheet.FontDef fd        = font.fontDef();
        final StringBuilder     result    = new StringBuilder(32);
        final byte[]            srcData   = source.data;
        final int               srcWidth  = source.width;
        final int               srcHeight = source.height;

        int cursorX   = startX;
        int gapPixels = 0;
        // Stop once this many consecutive pixels fail to match any glyph.
        final int stopAfter = Math.max(fd.spaceWidth() * 3, 16);

        while (cursorX < srcWidth) {

            final CharMatch match =
                    readChar(srcData, srcWidth, srcHeight, fd, cursorX, startY, true);

            if (match != null) {
                // Insert a space when a gap at least spaceWidth wide preceded this glyph.
                if (!result.isEmpty() && gapPixels >= fd.spaceWidth()) {
                    result.append(' ');
                }
                result.append(match.glyph().chr());
                cursorX  += match.glyph().width();
                gapPixels = 0;
            } else {
                cursorX++;
                gapPixels++;
                if (gapPixels >= stopAfter) break;
            }
        }

        return result.toString();
    }

    // =========================================================================
    // removeTextPixels
    // =========================================================================

    /**
     * Zeroes out every template pixel from {@code verifiedText} in the source
     * image, using the same cursor-advance logic as {@link #readLine}.
     *
     * <p>Only pixels listed in {@link FontSheet.GlyphDef#pixels()} (i.e. those
     * whose match strength exceeded the load-time threshold) are cleared.  Each
     * cleared pixel is set to fully-transparent black {@code [0, 0, 0, 0]}.
     *
     * <p>This matches Alt1's {@code removeTextPixels} masking technique: after
     * reading a countdown timer, erase its digit pixels so the underlying icon
     * colour is available for buff-fingerprint matching.
     *
     * @param source        the live frame to mutate in-place
     * @param startX        left edge of the text region (same as the {@link #readLine} call)
     * @param startY        top of the glyph content rows (same as the {@link #readLine} call)
     * @param verifiedText  the string returned by the preceding {@link #readLine} call
     * @param font          the loaded {@link FontSheet}
     */
    public void removeTextPixels(EdokitImage source,
                                 int startX,
                                 int startY,
                                 String verifiedText,
                                 FontSheet font) {

        if (verifiedText == null || verifiedText.isEmpty()) return;
        if (startX < 0 || startY < 0
                || startX >= source.width || startY >= source.height) return;

        final FontSheet.FontDef fd        = font.fontDef();
        final byte[]            srcData   = source.data;
        final int               srcWidth  = source.width;
        final int               srcHeight = source.height;
        final int               stride    = fd.shadow() ? 4 : 3;

        int cursorX = startX;

        for (int ci = 0; ci < verifiedText.length(); ci++) {
            final char ch = verifiedText.charAt(ci);

            if (ch == ' ') {
                cursorX += fd.spaceWidth();
                continue;
            }

            final FontSheet.GlyphDef glyph = fd.findGlyph(ch);
            if (glyph == null) {
                cursorX++;
                continue;
            }

            final int[] pixels = glyph.pixels();

            for (int a = 0; a < pixels.length; a += stride) {
                final int srcX = cursorX + pixels[a];
                final int srcY = startY  + pixels[a + 1];

                if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) continue;

                final int off = (srcX + srcWidth * srcY) * 4;
                srcData[off]     = 0; // R → 0
                srcData[off + 1] = 0; // G → 0
                srcData[off + 2] = 0; // B → 0
                srcData[off + 3] = 0; // A → 0  (fully transparent)
            }

            cursorX += glyph.width();
        }
    }

    // =========================================================================
    // Internal: readChar
    // =========================================================================

    /**
     * Tries every glyph in {@code fd} at screen position {@code (x, y)} and
     * returns the best match, or {@code null} if nothing passes scoring.
     *
     * <h3>Scoring summary</h3>
     * <pre>
     *   sizeScore₀ = −bonus + N × bonusPerPixel
     *   for each template pixel p:
     *     penalty ← canblend(screenPixel, fontColour × lum_p, strength_p)
     *     score     += penalty
     *     sizeScore += penalty − bonusPerPixel
     *     if score &gt; 400 → discard this glyph
     *   if checkedPixels &lt; N × 0.6 → discard
     *   winner = glyph with lowest sizeScore (score ≤ 400)
     * </pre>
     *
     * <p>All arithmetic uses {@code double} to match Alt1's JavaScript floats.
     *
     * @param srcData       raw RGBA bytes of the source frame
     * @param srcWidth      stride of the source frame
     * @param srcHeight     height of the source frame
     * @param fd            processed font definition
     * @param x             cursor X — left edge of the candidate glyph in source
     * @param y             cursor Y — top of content rows in source
     * @param allowSecondary {@code true} to include secondary-flagged glyphs
     * @return the best-matching {@link CharMatch}, or {@code null}
     */
    private static CharMatch readChar(byte[] srcData, int srcWidth, int srcHeight,
                                      FontSheet.FontDef fd,
                                      int x, int y,
                                      boolean allowSecondary) {

        final double MAX_SCORE    = 400.0;
        final int    stride       = fd.shadow() ? 4 : 3;
        final int    bpp          = fd.bonusPerPixel();
        final double fontR        = fd.fontR();
        final double fontG        = fd.fontG();
        final double fontB        = fd.fontB();
        final boolean shadow      = fd.shadow();

        FontSheet.GlyphDef bestGlyph     = null;
        double             bestSizeScore = Double.MAX_VALUE;
        double             bestScore     = 0.0;

        for (FontSheet.GlyphDef chr : fd.chars()) {

            if (chr.secondary() && !allowSecondary) continue;

            final int[] pixels  = chr.pixels();
            final int   nPixels = pixels.length / stride;
            if (nPixels == 0) continue;

            // Alt1: sizescore = -chrobj.bonus + originalpixels * bonusperpixel
            // chrobj.bonus = jsonNudge + bpp * N, so init = -jsonNudge
            double score     = 0.0;
            double sizeScore = -chr.bonus() + (double) nPixels * bpp;
            int    checked   = 0;
            boolean exceeded = false;

            for (int a = 0; a < pixels.length; a += stride) {
                final int px       = pixels[a];
                final int py       = pixels[a + 1];
                final int strength = pixels[a + 2];

                final int srcX = x + px;
                final int srcY = y + py;
                if (srcX < 0 || srcX >= srcWidth || srcY < 0 || srcY >= srcHeight) continue;

                final int off = (srcX + srcWidth * srcY) * 4;
                final int rm  = srcData[off]     & 0xFF;
                final int gm  = srcData[off + 1] & 0xFF;
                final int bm  = srcData[off + 2] & 0xFF;

                final double penalty;
                if (shadow) {
                    // Scale font colour by shadow luminance for this pixel.
                    // Alt1: penalty = canblend(src, col*lum, col*lum, col*lum, strength/255)
                    final double lum = pixels[a + 3] / 255.0;
                    penalty = canblend(rm, gm, bm, fontR * lum, fontG * lum, fontB * lum, strength);
                } else {
                    penalty = canblend(rm, gm, bm, fontR, fontG, fontB, strength);
                }

                score     += penalty;
                sizeScore += penalty - bpp;
                checked++;

                if (score > MAX_SCORE) {
                    exceeded = true;
                    break;
                }
            }

            if (exceeded) continue;

            // Reject if fewer than 60 % of template pixels were in-bounds and checked.
            // Also reject very small glyphs (≤ 6 pixels) unless every pixel was checked.
            if (checked < nPixels * 0.6) continue;
            if (nPixels <= 6 && checked < nPixels) continue;

            if (sizeScore < bestSizeScore) {
                bestSizeScore = sizeScore;
                bestScore     = score;
                bestGlyph     = chr;
            }
        }

        if (bestGlyph == null || bestScore > MAX_SCORE) return null;
        return new CharMatch(bestGlyph, bestScore, bestSizeScore);
    }

    // =========================================================================
    // Internal: canblend
    // =========================================================================

    /**
     * Determines whether the observed screen pixel {@code [rm, gm, bm]} is
     * consistent with a blend of the font colour {@code [r1, g1, b1]} at
     * proportion {@code strength/255} over some unknown background colour.
     *
     * <p>Port of Alt1's {@code canblend()} from {@code src/ocr/index.ts}:
     * <pre>
     *   p = strength / 255
     *   m = min(50, p / (1 − p))
     *   bg = screenPixel + (screenPixel − fontColour) × m   // extrapolated background
     *   penalty = max(0, −bg.r, −bg.g, −bg.b, bg.r−255, bg.g−255, bg.b−255)
     * </pre>
     * A penalty of 0 means the required background colour falls inside
     * {@code [0,255]³} — the observation is consistent with the font being
     * present.  A positive penalty measures how far outside {@code [0,255]³}
     * the implied background would lie, i.e. how implausible the match is.
     *
     * @param rm       observed red channel of the screen pixel
     * @param gm       observed green channel
     * @param bm       observed blue channel
     * @param r1       effective font red channel (may be pre-scaled by shadow lum)
     * @param g1       effective font green channel
     * @param b1       effective font blue channel
     * @param strength template pixel match strength [0–255]
     * @return penalty ≥ 0 (0 = perfect plausibility)
     */
    private static double canblend(int rm, int gm, int bm,
                                   double r1, double g1, double b1,
                                   int strength) {
        final double p = strength / 255.0;
        // m = leverage: how far to extrapolate to reach the implied background.
        // Capped at 50 to bound numerical extremes for strength near 255.
        final double m = Math.min(50.0, p / (1.0 - p + 1e-9));

        final double r = rm + (rm - r1) * m;
        final double g = gm + (gm - g1) * m;
        final double b = bm + (bm - b1) * m;

        // Penalty = how far the implied background colour lies outside [0, 255]³.
        return Math.max(0.0, Math.max(-r, Math.max(-g, Math.max(-b,
                        Math.max(r - 255.0, Math.max(g - 255.0, b - 255.0))))));
    }

    // =========================================================================
    // Private value type
    // =========================================================================

    /**
     * Holds the result of a successful {@link #readChar} call.
     *
     * @param glyph     the winning {@link FontSheet.GlyphDef}
     * @param score     raw accumulated penalty (≤ 400)
     * @param sizeScore size-adjusted score used to rank multiple candidates
     */
    private record CharMatch(FontSheet.GlyphDef glyph, double score, double sizeScore) {}
}
