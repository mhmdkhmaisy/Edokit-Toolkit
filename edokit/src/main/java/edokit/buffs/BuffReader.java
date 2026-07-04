package edokit.buffs;

import edokit.base.models.EdokitImage;
import edokit.ocr.FontSheet;
import edokit.ocr.RasterOCR;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * BuffReader — Buff-bar grid discovery, overlay-timer OCR, and colour-fingerprint
 * buff identification.
 *
 * <h2>What this is</h2>
 * A native-Java port of the programmatic logic in Alt1's {@code src/buffs/index.ts}.
 * RuneScape renders active buffs/debuffs as a vertical (or horizontal) grid of
 * fixed-size icon cells. Each cell may show:
 * <ul>
 *   <li>A 27×27 icon texture identifying the effect (e.g. Overload, Weapon Poison).</li>
 *   <li>An optional countdown-timer overlay (e.g. {@code "4:32"}) rendered directly
 *       on top of the icon pixels in a bright, high-contrast colour.</li>
 * </ul>
 * {@code BuffReader} performs three passes per frame:
 * <ol>
 *   <li>{@link #findBuffGrid} — structurally discovers which grid cells are
 *       actually occupied by a buff icon (as opposed to empty background).</li>
 *   <li>{@link #readBuffBar} step A/B — OCRs and then erases the countdown-timer
 *       overlay so it cannot bias the icon colour fingerprint underneath.</li>
 *   <li>{@link #readBuffBar} step C — matches the now-clean icon pixels against a
 *       database of known {@link BuffDefinition} colour fingerprints.</li>
 * </ol>
 *
 * <h2>Performance contract</h2>
 * <ul>
 *   <li>No objects are allocated inside the hot pixel-sampling loops — all
 *       intermediate values are stack-local {@code int} primitives.</li>
 *   <li>Colour channel extraction from packed {@code 0xRRGGBB} template values uses
 *       bitwise shifts ({@code >> 16}, {@code >> 8}, {@code & 0xFF}) rather than
 *       object-based colour types.</li>
 *   <li>Pixel addressing uses {@link EdokitImage#pixelOffset}, a {@code final}
 *       method the JIT inlines to a bare multiply-add.</li>
 * </ul>
 *
 * <h2>Thread safety</h2>
 * {@code BuffReader} holds a single internal {@link RasterOCR} instance, which is
 * itself stateless. All other state lives in caller-supplied arguments, so a
 * {@code BuffReader} may be shared across threads scanning different frames
 * without external synchronisation.
 */
public final class BuffReader {

    // =========================================================================
    // Configuration constants (Alt1-exact layout math)
    // =========================================================================

    /** Pixel width/height of a single buff icon (the visible texture square). */
    public static final int BUFF_SIZE = 27;

    /**
     * Pixel step size between horizontal/vertical grid slots, including the
     * 3px inter-cell padding baked into RuneScape's buff bar layout
     * ({@code GRID_SIZE - BUFF_SIZE == 3}).
     */
    public static final int GRID_SIZE = 30;

    // -------------------------------------------------------------------------
    // Grid-frame detection tuning
    // -------------------------------------------------------------------------

    /**
     * Maximum per-channel value for a pixel to be classified as part of the
     * buff cell's outer 2px near-black bounding frame.
     */
    private static final int OUTER_BORDER_MAX_CHANNEL = 40;

    /** Lower/upper bounds for the inner blue-grey border gradient (R and G channels). */
    private static final int INNER_BORDER_RG_MIN = 45;
    private static final int INNER_BORDER_RG_MAX = 115;

    /** Lower/upper bounds for the inner blue-grey border gradient (B channel). */
    private static final int INNER_BORDER_B_MIN = 65;
    private static final int INNER_BORDER_B_MAX = 150;

    /** Minimum lead the blue channel must have over red/green for a "blue-grey" read. */
    private static final int INNER_BORDER_BLUE_LEAD = 4;

    /** Consecutive empty columns in a row before that row's horizontal scan stops. */
    private static final int MAX_EMPTY_COLS_BEFORE_ROW_STOP = 1;

    // =========================================================================
    // Internal state
    // =========================================================================

    /** Stateless OCR engine reused across every {@link #readBuffBar} call. */
    private final RasterOCR ocr = new RasterOCR();

    // =========================================================================
    // Specification 1: Structural grid bounds & discovery
    // =========================================================================

    /**
     * Structurally scans the interface for occupied buff grid cells, starting
     * from an anchor coordinate (typically resolved by
     * {@link edokit.base.vision.SubImageMatcher} against the buff bar's header
     * sprite) and stepping outward in {@link #GRID_SIZE} increments.
     *
     * <h3>Detection heuristic</h3>
     * A cell at {@code (x, y)} is considered an occupied buff slot when both of
     * the following hold:
     * <ol>
     *   <li><b>Outer bounding frame</b> — the top-left 2px L-shaped border
     *       (top edge + left edge) reads near-black
     *       ({@code max(R,G,B) <= OUTER_BORDER_MAX_CHANNEL}).</li>
     *   <li><b>Inner border gradient</b> — the pixel just inside that frame
     *       reads a distinctive blue-grey tone ({@code R} and {@code G} within
     *       a mid-grey band, {@code B} moderately higher than {@code R}/{@code G}).</li>
     * </ol>
     * Both checks must pass for the slot to be reported; an icon rendered with
     * <em>no</em> frame (i.e. empty background) fails the outer check
     * immediately and is skipped with no further per-pixel work.
     *
     * <h3>Scan order &amp; early termination</h3>
     * Cells are scanned row-major (top to bottom, left to right within a row),
     * mirroring the fact that RuneScape always fills buff slots contiguously
     * starting at the anchor. A row's horizontal scan stops early once
     * {@link #MAX_EMPTY_COLS_BEFORE_ROW_STOP} consecutive empty columns are
     * seen (the remainder of that row is assumed empty), and the vertical scan
     * stops entirely the first time an entire row produces zero matches.
     *
     * @param screen  the live frame to scan
     * @param anchorX left pixel coordinate of grid column 0
     * @param anchorY top pixel coordinate of grid row 0
     * @param maxHor  maximum number of columns to probe
     * @param maxVer  maximum number of rows to probe
     * @return a list of {@link Rectangle} bounds (each {@code BUFF_SIZE x BUFF_SIZE})
     *         for every discovered occupied slot, in row-major scan order
     */
    public List<Rectangle> findBuffGrid(EdokitImage screen,
                                        int anchorX,
                                        int anchorY,
                                        int maxHor,
                                        int maxVer) {

        final List<Rectangle> slots = new ArrayList<>();
        if (screen == null || maxHor <= 0 || maxVer <= 0) {
            return slots;
        }

        rows:
        for (int row = 0; row < maxVer; row++) {
            final int slotY = anchorY + (row * GRID_SIZE);
            boolean rowHadMatch = false;
            int emptyStreak = 0;

            for (int col = 0; col < maxHor; col++) {
                final int slotX = anchorX + (col * GRID_SIZE);

                if (isValidBuffFrame(screen, slotX, slotY)) {
                    slots.add(new Rectangle(slotX, slotY, BUFF_SIZE, BUFF_SIZE));
                    rowHadMatch = true;
                    emptyStreak = 0;
                } else {
                    emptyStreak++;
                    if (emptyStreak > MAX_EMPTY_COLS_BEFORE_ROW_STOP) {
                        break; // rest of this row assumed empty
                    }
                }
            }

            if (!rowHadMatch) {
                break rows; // rest of the grid assumed empty
            }
        }

        return slots;
    }

    /**
     * Tests whether the {@code BUFF_SIZE x BUFF_SIZE} cell whose top-left corner
     * sits at {@code (x, y)} carries a genuine buff-frame border.
     *
     * <p>No objects are allocated here — every pixel read goes straight through
     * {@link EdokitImage#pixelOffset} into the backing {@code byte[]}.
     *
     * @param screen the frame being scanned
     * @param x      candidate slot left edge
     * @param y      candidate slot top edge
     * @return {@code true} if the outer near-black frame and inner blue-grey
     *         gradient are both present
     */
    private static boolean isValidBuffFrame(EdokitImage screen, int x, int y) {
        // Bounds guard: the full cell must fit inside the frame.
        if (x < 0 || y < 0
                || x + BUFF_SIZE > screen.width
                || y + BUFF_SIZE > screen.height) {
            return false;
        }

        // ── Outer 2px near-black L-frame: sample top edge + left edge ──────────
        // Top edge, both border rows, spanning the full cell width.
        for (int dx = 0; dx < BUFF_SIZE; dx++) {
            if (!isNearBlack(screen, x + dx, y)) return false;
            if (!isNearBlack(screen, x + dx, y + 1)) return false;
        }
        // Left edge, both border columns, spanning the full cell height.
        for (int dy = 0; dy < BUFF_SIZE; dy++) {
            if (!isNearBlack(screen, x, y + dy)) return false;
            if (!isNearBlack(screen, x + 1, y + dy)) return false;
        }

        // ── Inner blue-grey border gradient, just inside the black frame ───────
        return isBlueGreyBorder(screen, x + 2, y + 2)
            && isBlueGreyBorder(screen, x + BUFF_SIZE - 3, y + 2)
            && isBlueGreyBorder(screen, x + 2, y + BUFF_SIZE - 3);
    }

    /** Returns {@code true} if the pixel at {@code (x, y)} reads near-black. */
    private static boolean isNearBlack(EdokitImage screen, int x, int y) {
        final int off = screen.pixelOffset(x, y);
        final byte[] data = screen.data;
        final int r = data[off]     & 0xFF;
        final int g = data[off + 1] & 0xFF;
        final int b = data[off + 2] & 0xFF;
        final int maxChannel = Math.max(r, Math.max(g, b));
        return maxChannel <= OUTER_BORDER_MAX_CHANNEL;
    }

    /** Returns {@code true} if the pixel at {@code (x, y)} reads blue-grey. */
    private static boolean isBlueGreyBorder(EdokitImage screen, int x, int y) {
        final int off = screen.pixelOffset(x, y);
        final byte[] data = screen.data;
        final int r = data[off]     & 0xFF;
        final int g = data[off + 1] & 0xFF;
        final int b = data[off + 2] & 0xFF;

        if (r < INNER_BORDER_RG_MIN || r > INNER_BORDER_RG_MAX) return false;
        if (g < INNER_BORDER_RG_MIN || g > INNER_BORDER_RG_MAX) return false;
        if (b < INNER_BORDER_B_MIN  || b > INNER_BORDER_B_MAX)  return false;

        return (b - r) >= INNER_BORDER_BLUE_LEAD && (b - g) >= INNER_BORDER_BLUE_LEAD;
    }

    // =========================================================================
    // Specification 2: State tracking & the target matching loop
    // =========================================================================

    /**
     * Reads every discovered buff slot: extracts and erases its countdown-timer
     * overlay, then fingerprint-matches its icon texture against a database of
     * known buffs.
     *
     * <h3>Per-slot algorithm</h3>
     * <pre>
     *   for each slot (index i):
     *     A. text ← RasterOCR.readLine(bottom half of slot)
     *     B. if text non-empty: RasterOCR.removeTextPixels(same region)
     *     C. for each BuffDefinition in database (declaration order):
     *          if countMatch(slot, definition) → register TrackedBuff(i, name, text); stop
     * </pre>
     *
     * <h3>Step A/B — OCR extraction &amp; masking</h3>
     * Countdown timers are rendered along the bottom edge of the icon. The scan
     * origin for both the read and the erase is the bottom half of the slot
     * rectangle: {@code (slot.x, slot.y + slot.height / 2)}. Erasing immediately
     * after reading (rather than deferring it) guarantees the colour fingerprint
     * step in the same iteration never samples timer-digit pixels.
     *
     * <h3>Step C — countMatch colour fingerprint</h3>
     * For each {@link BuffDefinition}, every tracked offset is sampled from
     * {@code screenFrame} at {@code (slot.x + offsetX, slot.y + offsetY)} and its
     * RGB channels are compared against the definition's expected colour at that
     * offset. A definition matches only when <em>every</em> offset's per-channel
     * delta is {@code <= colorTolerance}. The first matching definition in
     * declaration order wins for that slot (mirrors Alt1's first-match
     * short-circuit); remaining definitions are skipped for that slot.
     *
     * @param screenFrame    the live frame to read and mutate (timer pixels are
     *                       erased in place)
     * @param activeSlots    slot rectangles previously returned by
     *                       {@link #findBuffGrid}
     * @param timerFont      font schema used to OCR the countdown overlay
     * @param database       known buff colour fingerprints, tried in order
     * @param colorTolerance maximum per-channel RGB delta for a fingerprint
     *                       offset to be considered a match
     * @return one {@link TrackedBuff} per slot that both matched a known
     *         fingerprint; slots with no fingerprint match are omitted
     */
    public List<TrackedBuff> readBuffBar(EdokitImage screenFrame,
                                         List<Rectangle> activeSlots,
                                         FontSheet timerFont,
                                         List<BuffDefinition> database,
                                         int colorTolerance) {

        final List<TrackedBuff> tracked = new ArrayList<>();
        if (screenFrame == null || activeSlots == null || activeSlots.isEmpty()) {
            return tracked;
        }

        for (int i = 0; i < activeSlots.size(); i++) {
            final Rectangle slot = activeSlots.get(i);

            // ── Step A: OCR the countdown timer along the bottom half of the icon ──
            final int ocrX = slot.x;
            final int ocrY = slot.y + (slot.height / 2);
            String timerText = "";
            if (timerFont != null) {
                timerText = ocr.readLine(screenFrame, ocrX, ocrY, timerFont);

                // ── Step B: mask the recognised digits out of the live frame ───────
                if (!timerText.isEmpty()) {
                    ocr.removeTextPixels(screenFrame, ocrX, ocrY, timerText, timerFont);
                }
            }

            // ── Step C: colour-fingerprint match against the known buff database ──
            if (database != null) {
                for (int d = 0; d < database.size(); d++) {
                    final BuffDefinition def = database.get(d);
                    if (countMatch(screenFrame, slot, def, colorTolerance)) {
                        tracked.add(new TrackedBuff(i, def.name(), timerText));
                        break; // first matching definition wins this slot
                    }
                }
            }
        }

        return tracked;
    }

    /**
     * Alt1-style {@code countMatch}: returns {@code true} only if every tracked
     * offset of {@code def} reads within {@code colorTolerance} of its expected
     * colour inside {@code slot}.
     *
     * <p>Recycles primitive locals across iterations — no per-offset object
     * allocation. Expected colours are unpacked from a single {@code 0xRRGGBB}
     * int via bitwise shifts rather than a boxed colour type.
     *
     * @param screenFrame    frame to sample from
     * @param slot           the candidate buff cell bounds
     * @param def            colour fingerprint under test
     * @param colorTolerance maximum per-channel delta for a passing offset
     * @return {@code true} if all of {@code def}'s sample offsets match
     */
    private static boolean countMatch(EdokitImage screenFrame,
                                      Rectangle slot,
                                      BuffDefinition def,
                                      int colorTolerance) {

        final int[] offsetsX   = def.sampleOffsetsX();
        final int[] offsetsY   = def.sampleOffsetsY();
        final int[] expectedRGBs = def.expectedRGBs();
        final int sampleCount  = offsetsX.length;

        if (sampleCount == 0 || sampleCount != offsetsY.length || sampleCount != expectedRGBs.length) {
            return false;
        }

        final byte[] data   = screenFrame.data;
        final int    width  = screenFrame.width;
        final int    height = screenFrame.height;

        for (int s = 0; s < sampleCount; s++) {
            final int px = slot.x + offsetsX[s];
            final int py = slot.y + offsetsY[s];

            if (px < 0 || py < 0 || px >= width || py >= height) {
                return false;
            }

            final int off = (4 * px) + (4 * width * py);
            final int srcR = data[off]     & 0xFF;
            final int srcG = data[off + 1] & 0xFF;
            final int srcB = data[off + 2] & 0xFF;

            // Unpack the packed 0xRRGGBB expected colour via bitwise shifts.
            final int expected = expectedRGBs[s];
            final int expR = (expected >> 16) & 0xFF;
            final int expG = (expected >> 8)  & 0xFF;
            final int expB =  expected        & 0xFF;

            final int deltaR = expR - srcR;
            final int deltaG = expG - srcG;
            final int deltaB = expB - srcB;

            // Branch-free absolute value, mirroring RasterOCR's coldif check.
            if (((deltaR ^ (deltaR >> 31)) - (deltaR >> 31)) > colorTolerance) return false;
            if (((deltaG ^ (deltaG >> 31)) - (deltaG >> 31)) > colorTolerance) return false;
            if (((deltaB ^ (deltaB >> 31)) - (deltaB >> 31)) > colorTolerance) return false;
        }

        return true;
    }

    // =========================================================================
    // Nested value types
    // =========================================================================

    /**
     * A named colour fingerprint used to identify one specific buff/debuff icon.
     *
     * <p>Each fingerprint tracks 3–5 highly specific local pixel offsets inside
     * the {@code BUFF_SIZE x BUFF_SIZE} icon square and the exact colour expected
     * at each — chosen from parts of the icon art that are visually unique to
     * that buff (so overlapping timer digits or generic icon-frame pixels are
     * deliberately avoided).
     *
     * <p>The three arrays are parallel: {@code sampleOffsetsX[i]},
     * {@code sampleOffsetsY[i]}, and {@code expectedRGBs[i]} together describe
     * one sample point. {@code expectedRGBs} values are packed as
     * {@code 0xRRGGBB} (alpha ignored).
     *
     * @param name            human-readable buff identifier (e.g. {@code "Overload"})
     * @param sampleOffsetsX  local X offsets (0..BUFF_SIZE-1) inside the icon square
     * @param sampleOffsetsY  local Y offsets (0..BUFF_SIZE-1) inside the icon square
     * @param expectedRGBs    expected colour at each offset, packed as {@code 0xRRGGBB}
     */
    public record BuffDefinition(String name,
                                 int[] sampleOffsetsX,
                                 int[] sampleOffsetsY,
                                 int[] expectedRGBs) {

        /**
         * Compact constructor — validates that all three parallel arrays share
         * the same length so {@link #countMatch} never has to bounds-check them
         * per sample.
         */
        public BuffDefinition {
            final int len = sampleOffsetsX == null ? -1 : sampleOffsetsX.length;
            if (sampleOffsetsX == null || sampleOffsetsY == null || expectedRGBs == null
                    || sampleOffsetsY.length != len || expectedRGBs.length != len) {
                throw new IllegalArgumentException(
                        "BuffDefinition \"" + name + "\": sampleOffsetsX, sampleOffsetsY, "
                        + "and expectedRGBs must be non-null and equal length.");
            }
            if (len < 3 || len > 5) {
                throw new IllegalArgumentException(
                        "BuffDefinition \"" + name + "\" must track 3-5 sample offsets, got: " + len);
            }
        }

        @Override
        public String toString() {
            return "BuffDefinition{name=\"" + name + "\", samples=" + sampleOffsetsX.length + "}";
        }
    }

    /**
     * One identified buff occupying a specific slot in the discovered grid.
     *
     * @param slotIndex index of the matched slot within the {@code activeSlots}
     *                  list passed to {@link #readBuffBar} (top-to-bottom,
     *                  left-to-right scan order)
     * @param buffName  {@link BuffDefinition#name()} of the identified buff
     * @param timerText the countdown string OCR'd from the slot's overlay before
     *                  it was masked out; empty if no timer was present
     */
    public record TrackedBuff(int slotIndex, String buffName, String timerText) {

        @Override
        public String toString() {
            return "TrackedBuff{slot=" + slotIndex + ", buff=\"" + buffName
                   + "\", timer=\"" + timerText + "\"}";
        }
    }

    // Silence "unused import" style checks in stricter lint configurations —
    // Collections.emptyList() usage kept out of hot paths intentionally.
    static {
        assert Collections.emptyList().isEmpty();
    }
}
