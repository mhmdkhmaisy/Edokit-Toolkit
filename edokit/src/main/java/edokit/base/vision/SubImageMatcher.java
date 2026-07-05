package edokit.base.vision;

import edokit.base.models.EdokitImage;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * SubImageMatcher — template matching against live EdokitImage frames.
 *
 * <h2>Two matching modes (selected automatically)</h2>
 *
 * <h3>Mode A — Alt1-style direct colour scan (alpha-masked templates)</h3>
 * When the template PNG has an alpha channel (any pixel with alpha &lt; 255),
 * this class replicates Alt1's {@code ImageDetect} algorithm:
 * <ol>
 *   <li>At construction, extract every opaque pixel's (dx, dy, R, G, B) into a
 *       flat int array {@code opaquePixels}.</li>
 *   <li>At scan time, slide a window across the source frame. For each candidate
 *       position, compare every opaque template pixel against the corresponding
 *       screen pixel using a per-channel absolute-difference tolerance
 *       ({@link #ALT1_MATCH_TOLERANCE}).</li>
 *   <li>Score = matching pixels / total opaque pixels.  If score ≥ threshold
 *       the position is a candidate hit.</li>
 *   <li>Fast-rejection: before the full per-pixel loop, the first opaque pixel
 *       (top-left corner of the template border) is checked alone. Because the
 *       border has a very specific colour (e.g. the RS3 buff-border green
 *       {@code 90,150,25}), nearly every screen position is rejected after a
 *       single pixel comparison.</li>
 * </ol>
 * This correctly handles templates like {@code buffborder.data.png}, which is
 * a frame of a specific colour with a transparent interior.
 * {@code TM_CCOEFF_NORMED} (grayscale, mean-subtracted) fails for such
 * templates because the mean subtraction zeroes out nearly-uniform borders and
 * because the transparent interior was mapping to black, polluting the score.
 *
 * <h3>Mode B — OpenCV TM_CCOEFF_NORMED (fully opaque templates)</h3>
 * When every pixel is opaque, the standard OpenCV normalised cross-correlation
 * path is used. This is more efficient and better suited to complex, multi-colour
 * textures with no transparency.
 *
 * <h2>Memory strategy</h2>
 * OpenCV Mat objects are kept allocated per-instance and reused across frames.
 * The Alt1-style {@code opaquePixels} array is allocated once at construction.
 *
 * <h2>Thread safety</h2>
 * Not thread-safe. All matching calls must come from the same thread.
 */
public final class SubImageMatcher implements AutoCloseable {

    // =========================================================================
    // Static OpenCV initialisation
    // =========================================================================

    static {
        try {
            nu.pattern.OpenCV.loadLocally();
        } catch (Exception e) {
            throw new ExceptionInInitializerError(
                    "Failed to load OpenCV native library via nu.pattern.OpenCV.loadLocally(). "
                    + "Ensure the org.openpnp:opencv dependency is on the classpath. "
                    + "Cause: " + e.getMessage());
        }
    }

    // =========================================================================
    // Public constants
    // =========================================================================

    /**
     * Default match-score threshold for {@link #findAll} / {@link #findBest}.
     *
     * <p>For Alt1-style mode: fraction of opaque template pixels that must match
     * within {@link #ALT1_MATCH_TOLERANCE} for the position to be accepted.
     * 0.85 = 85 % of the border ring must match.
     *
     * <p>For OpenCV mode: normalised cross-correlation score in [0, 1].
     */
    public static final float DEFAULT_THRESHOLD = 0.85f;

    /**
     * Per-channel absolute-difference tolerance used by the Alt1-style scan.
     *
     * <p>30 matches {@code COLOR_TOLERANCE} used throughout the engine and
     * provides adequate slack for GDI BitBlt capture rounding (which can shift
     * individual channels by ±5–15 relative to the raw pixel value in the
     * reference PNG).
     */
    public static final int ALT1_MATCH_TOLERANCE = 30;

    // =========================================================================
    // Template state
    // =========================================================================

    private final int templateWidth;
    private final int templateHeight;

    /**
     * Alt1-style mode: flat array of opaque pixel descriptors.
     * Layout per entry (stride 5): {@code [dx, dy, R, G, B]}.
     * {@code null} when the template is fully opaque (Mode B).
     */
    private final int[] opaquePixels;   // stride 5: [dx, dy, R, G, B, ...]
    private final int   opaqueCount;    // number of opaque pixels

    // ── OpenCV resources (Mode B only; null in Mode A) ────────────────────────
    private final Mat templateRGBA;
    private final Mat templateGray;
    private Mat       srcRGBA;
    private Mat       srcGray;
    private Mat       resultMat;
    private float[]   resultBuffer;
    private int       cachedSrcWidth  = -1;
    private int       cachedSrcHeight = -1;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * Builds a matcher for the supplied template.
     *
     * <p>If the template has any transparent pixel (alpha &lt; 255), Mode A
     * (Alt1-style direct colour scan) is used.  Otherwise Mode B (OpenCV
     * {@code TM_CCOEFF_NORMED}) is used.
     *
     * @param template source template image; must have positive dimensions
     */
    public SubImageMatcher(EdokitImage template) {
        if (template.width <= 0 || template.height <= 0) {
            throw new IllegalArgumentException(
                    "Template must have positive dimensions, got "
                    + template.width + "×" + template.height);
        }

        this.templateWidth  = template.width;
        this.templateHeight = template.height;

        // ── Decide mode by scanning for any transparent pixel ─────────────────
        final int[] opaqueResult = extractOpaquePixels(template);
        if (opaqueResult != null) {
            // Mode A — Alt1-style
            this.opaquePixels  = opaqueResult;
            this.opaqueCount   = opaqueResult.length / 5;
            this.templateRGBA  = null;
            this.templateGray  = null;
            System.out.printf(
                    "[SubImageMatcher] %dx%d template — Mode A (Alt1-style colour scan), "
                    + "%d opaque pixel(s) to compare per position.%n",
                    templateWidth, templateHeight, opaqueCount);
        } else {
            // Mode B — OpenCV
            this.opaquePixels = null;
            this.opaqueCount  = 0;
            this.templateRGBA = new Mat(template.height, template.width, CvType.CV_8UC4);
            this.templateGray = new Mat(template.height, template.width, CvType.CV_8UC1);
            templateRGBA.put(0, 0, template.data);
            Imgproc.cvtColor(templateRGBA, templateGray, Imgproc.COLOR_RGBA2GRAY);
            System.out.printf(
                    "[SubImageMatcher] %dx%d template — Mode B (OpenCV TM_CCOEFF_NORMED).%n",
                    templateWidth, templateHeight);
        }
    }

    /** Convenience factory. */
    public static SubImageMatcher of(EdokitImage template) {
        return new SubImageMatcher(template);
    }

    // =========================================================================
    // Public matching API
    // =========================================================================

    /**
     * Finds all occurrences of the template above the given threshold.
     *
     * @param source    live frame to search
     * @param threshold score in [0, 1]; use {@link #DEFAULT_THRESHOLD} if unsure
     * @return unmodifiable list of matches (empty if none)
     */
    public List<MatchPoint> findAll(EdokitImage source, float threshold) {
        validateThreshold(threshold);
        if (source.width < templateWidth || source.height < templateHeight) {
            return Collections.emptyList();
        }
        if (opaquePixels != null) {
            return alt1StyleScan(source, threshold);
        } else {
            return opencvScan(source, threshold);
        }
    }

    /** {@link #findAll} with {@link #DEFAULT_THRESHOLD}. */
    public List<MatchPoint> findAll(EdokitImage source) {
        return findAll(source, DEFAULT_THRESHOLD);
    }

    /**
     * Finds the single highest-confidence match above the given threshold.
     *
     * @param source    live frame to search
     * @param threshold score in [0, 1]
     * @return best match, or empty
     */
    public Optional<MatchPoint> findBest(EdokitImage source, float threshold) {
        validateThreshold(threshold);
        if (source.width < templateWidth || source.height < templateHeight) {
            return Optional.empty();
        }
        if (opaquePixels != null) {
            return alt1StyleBest(source, threshold);
        } else {
            return opencvBest(source, threshold);
        }
    }

    /** {@link #findBest} with {@link #DEFAULT_THRESHOLD}. */
    public Optional<MatchPoint> findBest(EdokitImage source) {
        return findBest(source, DEFAULT_THRESHOLD);
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    @Override
    public void close() {
        if (templateRGBA != null) templateRGBA.release();
        if (templateGray != null) templateGray.release();
        if (srcRGBA      != null) srcRGBA.release();
        if (srcGray      != null) srcGray.release();
        if (resultMat    != null) resultMat.release();
    }

    // =========================================================================
    // Mode A — Alt1-style direct colour scan
    // =========================================================================

    /**
     * Scans the full source frame for positions where at least {@code threshold}
     * fraction of the opaque template pixels match within
     * {@link #ALT1_MATCH_TOLERANCE} per channel.
     *
     * <p>Fast-rejection: the very first opaque pixel (entry 0 in
     * {@code opaquePixels}) is checked before the full loop.  For templates
     * like the RS3 buff border (a ring of a specific green), only a tiny fraction
     * of screen positions will pass this first test, making the overall scan
     * very fast despite covering every pixel.
     */
    private List<MatchPoint> alt1StyleScan(EdokitImage src, float threshold) {
        final List<MatchPoint> hits = new ArrayList<>();
        final int maxX = src.width  - templateWidth;
        final int maxY = src.height - templateHeight;
        final byte[] sdata = src.data;
        final int sw = src.width;

        // First opaque pixel coords & expected colour — used for fast rejection.
        final int fDx = opaquePixels[0], fDy = opaquePixels[1];
        final int fEr = opaquePixels[2], fEg = opaquePixels[3], fEb = opaquePixels[4];
        final int tol = ALT1_MATCH_TOLERANCE;
        final int needed = (int) Math.ceil(opaqueCount * threshold);

        for (int sy = 0; sy <= maxY; sy++) {
            for (int sx = 0; sx <= maxX; sx++) {
                // ── Fast rejection: check first opaque pixel only ──────────────
                final int fOff = ((sy + fDy) * sw + (sx + fDx)) * 4;
                if (Math.abs((sdata[fOff]     & 0xFF) - fEr) > tol) continue;
                if (Math.abs((sdata[fOff + 1] & 0xFF) - fEg) > tol) continue;
                if (Math.abs((sdata[fOff + 2] & 0xFF) - fEb) > tol) continue;

                // ── Full per-pixel comparison ──────────────────────────────────
                int matches = 1; // first pixel already passed
                for (int k = 5; k < opaquePixels.length; k += 5) {
                    final int off = ((sy + opaquePixels[k + 1]) * sw
                                   + (sx + opaquePixels[k    ])) * 4;
                    final int er = opaquePixels[k + 2];
                    final int eg = opaquePixels[k + 3];
                    final int eb = opaquePixels[k + 4];
                    if (Math.abs((sdata[off]     & 0xFF) - er) <= tol
                     && Math.abs((sdata[off + 1] & 0xFF) - eg) <= tol
                     && Math.abs((sdata[off + 2] & 0xFF) - eb) <= tol) {
                        matches++;
                    }
                }

                if (matches >= needed) {
                    hits.add(new MatchPoint(sx, sy, (float) matches / opaqueCount));
                }
            }
        }

        // NMS: keep only local maxima within one template-width window.
        return Collections.unmodifiableList(nms(hits));
    }

    /** Returns the single best Alt1-style match. */
    private Optional<MatchPoint> alt1StyleBest(EdokitImage src, float threshold) {
        final int maxX = src.width  - templateWidth;
        final int maxY = src.height - templateHeight;
        final byte[] sdata = src.data;
        final int sw = src.width;

        final int fDx = opaquePixels[0], fDy = opaquePixels[1];
        final int fEr = opaquePixels[2], fEg = opaquePixels[3], fEb = opaquePixels[4];
        final int tol = ALT1_MATCH_TOLERANCE;
        final int needed = (int) Math.ceil(opaqueCount * threshold);

        float bestScore = -1f;
        int bestX = -1, bestY = -1;

        for (int sy = 0; sy <= maxY; sy++) {
            for (int sx = 0; sx <= maxX; sx++) {
                final int fOff = ((sy + fDy) * sw + (sx + fDx)) * 4;
                if (Math.abs((sdata[fOff]     & 0xFF) - fEr) > tol) continue;
                if (Math.abs((sdata[fOff + 1] & 0xFF) - fEg) > tol) continue;
                if (Math.abs((sdata[fOff + 2] & 0xFF) - fEb) > tol) continue;

                int matches = 1;
                for (int k = 5; k < opaquePixels.length; k += 5) {
                    final int off = ((sy + opaquePixels[k + 1]) * sw
                                   + (sx + opaquePixels[k    ])) * 4;
                    if (Math.abs((sdata[off]     & 0xFF) - opaquePixels[k + 2]) <= tol
                     && Math.abs((sdata[off + 1] & 0xFF) - opaquePixels[k + 3]) <= tol
                     && Math.abs((sdata[off + 2] & 0xFF) - opaquePixels[k + 4]) <= tol) {
                        matches++;
                    }
                }

                if (matches >= needed) {
                    final float score = (float) matches / opaqueCount;
                    if (score > bestScore) {
                        bestScore = score;
                        bestX = sx;
                        bestY = sy;
                    }
                }
            }
        }

        if (bestScore < threshold) return Optional.empty();
        return Optional.of(new MatchPoint(bestX, bestY, bestScore));
    }

    /**
     * Extracts the (dx, dy, R, G, B) descriptor for every non-transparent pixel.
     * Returns {@code null} if ALL pixels are fully opaque (→ use Mode B).
     * The first entry in the returned array is always a corner pixel of the
     * template border — chosen to maximise fast-rejection efficiency.
     */
    private static int[] extractOpaquePixels(EdokitImage template) {
        final byte[] rgba = template.data;
        final int w = template.width;
        final int h = template.height;
        final int n = w * h;

        boolean anyTransparent = false;
        int opaqueCount = 0;
        for (int i = 3; i < rgba.length; i += 4) {
            if ((rgba[i] & 0xFF) == 0) anyTransparent = true;
            else opaqueCount++;
        }

        if (!anyTransparent) return null; // fully opaque → Mode B

        final int[] pixels = new int[opaqueCount * 5];
        int idx = 0;

        // Put the top-left corner pixel first (best fast-rejection candidate).
        // It's guaranteed opaque for a border-style template.
        boolean firstInserted = false;
        for (int pi = 0; pi < n; pi++) {
            final int base = pi * 4;
            if ((rgba[base + 3] & 0xFF) == 0) continue;
            final int dx = pi % w;
            final int dy = pi / w;
            final int r  = rgba[base]     & 0xFF;
            final int g  = rgba[base + 1] & 0xFF;
            final int b  = rgba[base + 2] & 0xFF;

            if (!firstInserted) {
                // Corner pixel → slot 0 (used for fast rejection)
                pixels[0] = dx; pixels[1] = dy;
                pixels[2] = r;  pixels[3] = g; pixels[4] = b;
                firstInserted = true;
                idx = 5;
            } else {
                pixels[idx]     = dx; pixels[idx + 1] = dy;
                pixels[idx + 2] = r;  pixels[idx + 3] = g; pixels[idx + 4] = b;
                idx += 5;
            }
        }

        return pixels;
    }

    // =========================================================================
    // Mode B — OpenCV TM_CCOEFF_NORMED
    // =========================================================================

    private List<MatchPoint> opencvScan(EdokitImage source, float threshold) {
        ensureSourceMatsAllocated(source.width, source.height);
        srcRGBA.put(0, 0, source.data);
        Imgproc.cvtColor(srcRGBA, srcGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.matchTemplate(srcGray, templateGray, resultMat, Imgproc.TM_CCOEFF_NORMED);
        final int resultRows = resultMat.rows();
        final int resultCols = resultMat.cols();
        resultMat.get(0, 0, resultBuffer);
        return extractPeaks(resultBuffer, resultRows, resultCols, threshold);
    }

    private Optional<MatchPoint> opencvBest(EdokitImage source, float threshold) {
        ensureSourceMatsAllocated(source.width, source.height);
        srcRGBA.put(0, 0, source.data);
        Imgproc.cvtColor(srcRGBA, srcGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.matchTemplate(srcGray, templateGray, resultMat, Imgproc.TM_CCOEFF_NORMED);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(resultMat);
        if (mmr.maxVal < threshold) return Optional.empty();
        return Optional.of(new MatchPoint((int) mmr.maxLoc.x, (int) mmr.maxLoc.y, (float) mmr.maxVal));
    }

    private void ensureSourceMatsAllocated(int srcW, int srcH) {
        if (srcW == cachedSrcWidth && srcH == cachedSrcHeight) return;
        if (srcRGBA   != null) srcRGBA.release();
        if (srcGray   != null) srcGray.release();
        if (resultMat != null) resultMat.release();
        final int resultW = srcW - templateWidth  + 1;
        final int resultH = srcH - templateHeight + 1;
        srcRGBA   = new Mat(srcH, srcW,   CvType.CV_8UC4);
        srcGray   = new Mat(srcH, srcW,   CvType.CV_8UC1);
        resultMat = new Mat(resultH, resultW, CvType.CV_32FC1);
        resultBuffer = new float[resultW * resultH];
        cachedSrcWidth  = srcW;
        cachedSrcHeight = srcH;
    }

    // =========================================================================
    // Internal: NMS and peak extraction (Mode B only)
    // =========================================================================

    private List<MatchPoint> extractPeaks(float[] data, int rows, int cols, float threshold) {
        final List<MatchPoint> peaks = new ArrayList<>();
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                final float val = data[row * cols + col];
                if (val < threshold) continue;
                if (isLocalMaximum(data, rows, cols, row, col)) {
                    peaks.add(new MatchPoint(col, row, val));
                }
            }
        }
        return Collections.unmodifiableList(peaks);
    }

    private boolean isLocalMaximum(float[] data, int rows, int cols, int row, int col) {
        final float center = data[row * cols + col];
        final int r0 = Math.max(0,        row - templateHeight);
        final int r1 = Math.min(rows - 1, row + templateHeight);
        final int c0 = Math.max(0,        col - templateWidth);
        final int c1 = Math.min(cols - 1, col + templateWidth);
        for (int r = r0; r <= r1; r++) {
            final int rowOffset = r * cols;
            for (int c = c0; c <= c1; c++) {
                if (r == row && c == col) continue;
                if (data[rowOffset + c] > center) return false;
            }
        }
        return true;
    }

    /** Simple NMS for Alt1-style hits: suppress any hit within one template-width of a higher-scoring hit. */
    private List<MatchPoint> nms(List<MatchPoint> hits) {
        if (hits.size() <= 1) return hits;
        final List<MatchPoint> sorted = new ArrayList<>(hits);
        sorted.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
        final List<MatchPoint> kept = new ArrayList<>();
        final boolean[] suppressed = new boolean[sorted.size()];
        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) continue;
            kept.add(sorted.get(i));
            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) continue;
                final MatchPoint hi = sorted.get(i), hj = sorted.get(j);
                if (Math.abs(hi.x() - hj.x()) < templateWidth
                 && Math.abs(hi.y() - hj.y()) < templateHeight) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
    }

    // =========================================================================
    // Validation
    // =========================================================================

    private static void validateThreshold(float threshold) {
        if (threshold < 0.0f || threshold > 1.0f) {
            throw new IllegalArgumentException(
                    "Threshold must be in [0.0, 1.0], got: " + threshold);
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    public int templateWidth()  { return templateWidth;  }
    public int templateHeight() { return templateHeight; }

    // =========================================================================
    // Result type
    // =========================================================================

    /**
     * Immutable match result.
     *
     * @param x          left edge of the matched region in the source image
     * @param y          top  edge of the matched region in the source image
     * @param confidence score in [0, 1]; 1.0 = all opaque pixels matched exactly
     */
    public record MatchPoint(int x, int y, float confidence) {
        @Override
        public String toString() {
            return String.format("MatchPoint{x=%d, y=%d, confidence=%.4f}", x, y, confidence);
        }
    }
}
