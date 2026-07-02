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
 * SubImageMatcher — OpenCV-backed template matching against live EdokitImage frames.
 *
 * <h2>What this does</h2>
 * Locates all occurrences of a fixed template image inside a (typically larger) source
 * frame by running OpenCV's {@code Imgproc.matchTemplate} with the
 * {@code TM_CCOEFF_NORMED} algorithm, then applying non-maximum suppression (NMS) to
 * return only genuine, spatially distinct matches above a confidence threshold.
 *
 * <h2>Memory strategy</h2>
 * OpenCV stores {@code Mat} data in native (off-heap) memory.  This class minimises
 * the number of native allocations by:
 * <ol>
 *   <li><b>Template Mats</b> — allocated once at construction, never reallocated.</li>
 *   <li><b>Source and result Mats</b> — lazily allocated on the first call and reused
 *       for every subsequent frame at the same resolution.  A resize only triggers
 *       reallocation when the source dimensions change.</li>
 *   <li><b>Result float buffer</b> — one {@code float[]} pre-allocated to match the
 *       result Mat's element count; reused for every extraction pass.</li>
 * </ol>
 *
 * <p><b>Java heap → native Mat transfer:</b> {@code EdokitImage.data} is a Java
 * {@code byte[]} living on the JVM heap; OpenCV {@code Mat} data lives in native
 * (off-heap) memory.  The standard OpenCV Java binding provides no way to pin a
 * Java array into native memory without a copy.  The transfer is performed via
 * {@code Mat.put(0, 0, byte[])}, which delegates to a single JNI {@code memcpy}
 * call — one copy per frame at the native layer, with no intermediate Java-level
 * array allocation.
 *
 * <h2>Pixel format pipeline</h2>
 * <pre>
 *   EdokitImage (RGBA, CV_8UC4)
 *       │
 *       ├─ srcMat.put()        → srcMat   (CV_8UC4, native)
 *       │  cvtColor RGBA→GRAY  → srcGray  (CV_8UC1, native)
 *       │
 *       └─ (template)          → templateGray (CV_8UC1, pre-loaded)
 *
 *   matchTemplate(srcGray, templateGray, resultMat, TM_CCOEFF_NORMED)
 *       → resultMat (CV_32FC1, values in [-1.0, +1.0])
 *
 *   resultMat.get() → resultBuffer (float[], Java heap)
 *   NMS peak scan   → List&lt;MatchPoint&gt;
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 *   EdokitImage template = EdokitImageLoader.load(new File("prayer_globe.png"));
 *
 *   try (SubImageMatcher matcher = new SubImageMatcher(template)) {
 *       // Inside your capture loop:
 *       EdokitImage frame = captureEngine.capture();
 *       List<SubImageMatcher.MatchPoint> hits = matcher.findAll(frame);
 *       // or for a single best match:
 *       matcher.findBest(frame).ifPresent(pt -> System.out.println("Found at " + pt));
 *   }
 * }</pre>
 *
 * <h2>Thread safety</h2>
 * Not thread-safe.  All calls to {@link #findAll} / {@link #findBest} must come
 * from the same thread.  {@link #close()} may be called from any thread after
 * the matching loop exits.
 */
public final class SubImageMatcher implements AutoCloseable {

    // =========================================================================
    // Static OpenCV initialisation
    // =========================================================================

    static {
        /*
         * org.openpnp:opencv bundles platform-native .so/.dll alongside the Java
         * JNI wrapper.  loadLocally() extracts the appropriate native library to a
         * temp directory and loads it — idempotent across multiple calls.
         *
         * If your deployment uses a system-installed OpenCV instead of the openpnp
         * bundle, replace this with: System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
         */
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
     * Default confidence threshold for {@link #findAll(EdokitImage)} and
     * {@link #findBest(EdokitImage)}.
     *
     * <p>0.90 corresponds to a 90% normalised cross-correlation score.  For
     * pixel-perfect game UI assets this is conservative; raise to 0.95–0.98
     * if false positives appear against patterned backgrounds.
     */
    public static final float DEFAULT_THRESHOLD = 0.90f;

    // =========================================================================
    // Template state (allocated once, immutable after construction)
    // =========================================================================

    /** Width of the fixed template in pixels. */
    private final int templateWidth;

    /** Height of the fixed template in pixels. */
    private final int templateHeight;

    /**
     * RGBA template Mat (CV_8UC4).  Kept alive to allow future colour-mode
     * extensions; currently only the grayscale version is used for matching.
     */
    private final Mat templateRGBA;

    /**
     * Grayscale template Mat (CV_8UC1).  This is the actual operand passed to
     * {@code matchTemplate} on every call — allocated once, never changed.
     */
    private final Mat templateGray;

    // =========================================================================
    // Source and result state (lazily allocated, reused per source resolution)
    // =========================================================================

    private Mat    srcRGBA;      // CV_8UC4  — source frame, native memory
    private Mat    srcGray;      // CV_8UC1  — grayscale source, native memory
    private Mat    resultMat;    // CV_32FC1 — matchTemplate output, native memory

    /** Pre-allocated Java-heap float array for reading resultMat into. */
    private float[] resultBuffer;

    /** Cached source dimensions — used to detect window resizes. */
    private int cachedSrcWidth  = -1;
    private int cachedSrcHeight = -1;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * Builds a matcher pre-loaded with the supplied template image.
     *
     * <p>The template's pixels are converted to grayscale and stored in native
     * OpenCV memory immediately.  All subsequent calls to {@link #findAll} and
     * {@link #findBest} reuse this native Mat — the template conversion cost is
     * paid exactly once.
     *
     * @param template the image to search for; must not be {@code null} and must
     *                 have positive, non-zero dimensions
     * @throws IllegalArgumentException if the template has zero dimensions
     */
    public SubImageMatcher(EdokitImage template) {
        if (template.width <= 0 || template.height <= 0) {
            throw new IllegalArgumentException(
                    "Template must have positive dimensions, got: "
                    + template.width + "×" + template.height);
        }

        this.templateWidth  = template.width;
        this.templateHeight = template.height;

        // Allocate native Mats for the template.
        templateRGBA = new Mat(template.height, template.width, CvType.CV_8UC4);
        templateGray = new Mat(template.height, template.width, CvType.CV_8UC1);

        // Push template pixels into native memory (single JNI memcpy).
        templateRGBA.put(0, 0, template.data);

        // Convert RGBA → grayscale once; stored permanently in templateGray.
        Imgproc.cvtColor(templateRGBA, templateGray, Imgproc.COLOR_RGBA2GRAY);
    }

    /**
     * Convenience static factory.
     *
     * @param template the image to search for
     * @return a new {@code SubImageMatcher} for the given template
     */
    public static SubImageMatcher of(EdokitImage template) {
        return new SubImageMatcher(template);
    }

    // =========================================================================
    // Public matching API
    // =========================================================================

    /**
     * Locates all occurrences of the template in {@code source} whose confidence
     * meets or exceeds the supplied threshold, with non-maximum suppression applied
     * to eliminate duplicate detections of the same region.
     *
     * <p>The returned list is sorted neither by position nor confidence — apply
     * {@code Comparator} sorting at the call site if order matters.
     *
     * @param source    the live frame to search within
     * @param threshold minimum normalised correlation score in [0.0, 1.0];
     *                  use {@link #DEFAULT_THRESHOLD} (0.90) if unsure
     * @return unmodifiable list of distinct match locations; empty if none found
     * @throws IllegalArgumentException if threshold is outside [0.0, 1.0]
     */
    public List<MatchPoint> findAll(EdokitImage source, float threshold) {
        validateThreshold(threshold);

        // Template larger than source → impossible match.
        if (source.width < templateWidth || source.height < templateHeight) {
            return Collections.emptyList();
        }

        // Allocate or reuse native Mats sized to this source resolution.
        ensureSourceMatsAllocated(source.width, source.height);

        // ── Transfer + convert source ─────────────────────────────────────────
        // mat.put() is a single JNI memcpy; no intermediate Java array is created.
        srcRGBA.put(0, 0, source.data);
        Imgproc.cvtColor(srcRGBA, srcGray, Imgproc.COLOR_RGBA2GRAY);

        // ── Template matching ─────────────────────────────────────────────────
        // resultMat is pre-allocated to the correct size; matchTemplate will
        // reuse its native storage without triggering a GDI reallocation.
        Imgproc.matchTemplate(srcGray, templateGray, resultMat, Imgproc.TM_CCOEFF_NORMED);

        // ── Read result into Java heap ────────────────────────────────────────
        // resultMat.get() copies the entire float matrix to our pre-allocated
        // resultBuffer in one JNI call.
        final int resultRows = resultMat.rows();
        final int resultCols = resultMat.cols();
        resultMat.get(0, 0, resultBuffer);

        // ── Extract local peaks above threshold (with NMS) ────────────────────
        return extractPeaks(resultBuffer, resultRows, resultCols, threshold);
    }

    /**
     * Locates all template occurrences using the {@link #DEFAULT_THRESHOLD}.
     *
     * @param source live frame to search within
     * @return unmodifiable list of distinct match locations
     */
    public List<MatchPoint> findAll(EdokitImage source) {
        return findAll(source, DEFAULT_THRESHOLD);
    }

    /**
     * Returns the single highest-confidence match in {@code source}, or
     * {@link Optional#empty()} if no match exceeds the threshold.
     *
     * <p>This path uses {@link Core#minMaxLoc} instead of a full matrix scan,
     * avoiding the cost of reading the entire {@code resultBuffer} array when
     * only the global maximum is needed.
     *
     * @param source    live frame to search within
     * @param threshold minimum confidence score in [0.0, 1.0]
     * @return the best match, or empty
     */
    public Optional<MatchPoint> findBest(EdokitImage source, float threshold) {
        validateThreshold(threshold);

        if (source.width < templateWidth || source.height < templateHeight) {
            return Optional.empty();
        }

        ensureSourceMatsAllocated(source.width, source.height);

        srcRGBA.put(0, 0, source.data);
        Imgproc.cvtColor(srcRGBA, srcGray, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.matchTemplate(srcGray, templateGray, resultMat, Imgproc.TM_CCOEFF_NORMED);

        // Core.minMaxLoc allocates one MinMaxLocResult (cheap) but avoids
        // reading the entire result matrix into Java heap.
        Core.MinMaxLocResult mmr = Core.minMaxLoc(resultMat);

        if (mmr.maxVal < threshold) {
            return Optional.empty();
        }

        return Optional.of(new MatchPoint(
                (int) mmr.maxLoc.x,
                (int) mmr.maxLoc.y,
                (float) mmr.maxVal));
    }

    /**
     * Returns the best match using the {@link #DEFAULT_THRESHOLD}.
     *
     * @param source live frame to search within
     * @return the best match, or empty
     */
    public Optional<MatchPoint> findBest(EdokitImage source) {
        return findBest(source, DEFAULT_THRESHOLD);
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /**
     * Releases all OpenCV native Mat memory held by this matcher.
     *
     * <p>Must be called exactly once when the matcher is no longer needed.
     * The try-with-resources pattern is strongly recommended.  Calling
     * {@code close()} more than once is safe (Mat.release() is idempotent).
     *
     * <p>Failure to call {@code close()} will leak native memory — the JVM GC
     * does not manage native Mat allocations.
     */
    @Override
    public void close() {
        // Template Mats (always allocated).
        templateRGBA.release();
        templateGray.release();

        // Source and result Mats (null until first findAll/findBest call).
        if (srcRGBA   != null) srcRGBA.release();
        if (srcGray   != null) srcGray.release();
        if (resultMat != null) resultMat.release();
    }

    // =========================================================================
    // Internal: Mat lifecycle management
    // =========================================================================

    /**
     * Ensures that {@link #srcRGBA}, {@link #srcGray}, {@link #resultMat}, and
     * {@link #resultBuffer} are allocated and sized for a source of
     * {@code srcW × srcH} pixels.
     *
     * <p>On the first call, all four objects are allocated.  On subsequent calls
     * at the same resolution, this method returns immediately (O(1) check).  If
     * the source resolution changes (e.g. window resize), the old native Mats are
     * explicitly released before the new ones are allocated — no native leak.
     */
    private void ensureSourceMatsAllocated(int srcW, int srcH) {
        if (srcW == cachedSrcWidth && srcH == cachedSrcHeight) {
            return; // steady state — zero work
        }

        // Release stale native resources before reallocating.
        if (srcRGBA   != null) srcRGBA.release();
        if (srcGray   != null) srcGray.release();
        if (resultMat != null) resultMat.release();

        final int resultW = srcW - templateWidth  + 1;
        final int resultH = srcH - templateHeight + 1;

        srcRGBA   = new Mat(srcH, srcW,   CvType.CV_8UC4);
        srcGray   = new Mat(srcH, srcW,   CvType.CV_8UC1);
        resultMat = new Mat(resultH, resultW, CvType.CV_32FC1);

        // Pre-allocate the Java-heap read buffer for the result matrix.
        resultBuffer = new float[resultW * resultH];

        cachedSrcWidth  = srcW;
        cachedSrcHeight = srcH;
    }

    // =========================================================================
    // Internal: Peak extraction with Non-Maximum Suppression
    // =========================================================================

    /**
     * Scans {@code data} (the flattened {@code TM_CCOEFF_NORMED} result matrix)
     * and returns all pixel positions that:
     * <ol>
     *   <li>Have a correlation value &ge; {@code threshold}.</li>
     *   <li>Are a strict local maximum within a neighbourhood equal to the
     *       template dimensions — i.e., no neighbouring pixel in that window
     *       has a higher score.</li>
     * </ol>
     *
     * <h3>NMS neighbourhood size</h3>
     * The suppression window is {@code ±templateWidth} columns × {@code ±templateHeight}
     * rows around each candidate.  This ensures that two returned matches are
     * always at least one full template-width apart horizontally and one
     * template-height apart vertically — preventing duplicate detections of the
     * same UI element.
     *
     * @param data      flattened float array from {@code resultMat.get(0, 0, data)}
     * @param rows      number of rows in the result matrix
     * @param cols      number of columns in the result matrix
     * @param threshold minimum score for a candidate to be considered
     * @return list of confirmed peaks as {@link MatchPoint} instances
     */
    private List<MatchPoint> extractPeaks(float[] data, int rows, int cols, float threshold) {
        final List<MatchPoint> peaks = new ArrayList<>();

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                final float val = data[row * cols + col];

                // Fast rejection: skip anything below threshold before the NMS scan.
                if (val < threshold) continue;

                // Confirm this pixel is a local maximum in the template-sized window.
                if (isLocalMaximum(data, rows, cols, row, col)) {
                    // (col, row) in result space = top-left corner of the match
                    // in source image space (result position maps 1:1 to source pixel).
                    peaks.add(new MatchPoint(col, row, val));
                }
            }
        }

        return Collections.unmodifiableList(peaks);
    }

    /**
     * Returns {@code true} if the pixel at {@code (row, col)} has the highest
     * value within the suppression neighbourhood of size
     * {@code ±templateWidth × ±templateHeight}.
     *
     * <p>The scan uses a flat index into {@code data} (row-major) to avoid
     * 2D array overhead.  The check short-circuits on the first neighbour found
     * with a strictly higher value — no need to scan the entire window once a
     * disqualifier is found.
     *
     * @param data   flattened result matrix (row-major)
     * @param rows   total rows in the result matrix (bounds for clamping)
     * @param cols   total columns in the result matrix (bounds for clamping)
     * @param row    candidate row index
     * @param col    candidate column index
     * @return {@code true} if this position is the local maximum
     */
    private boolean isLocalMaximum(float[] data, int rows, int cols, int row, int col) {
        final float center = data[row * cols + col];

        // Clamp neighbourhood bounds to valid result matrix extents.
        final int r0 = Math.max(0,      row - templateHeight);
        final int r1 = Math.min(rows - 1, row + templateHeight);
        final int c0 = Math.max(0,      col - templateWidth);
        final int c1 = Math.min(cols - 1, col + templateWidth);

        for (int r = r0; r <= r1; r++) {
            final int rowOffset = r * cols;
            for (int c = c0; c <= c1; c++) {
                // Skip self-comparison.
                if (r == row && c == col) continue;

                // Short-circuit: a single stronger neighbour disqualifies the candidate.
                if (data[rowOffset + c] > center) return false;
            }
        }
        return true;
    }

    // =========================================================================
    // Internal: argument validation
    // =========================================================================

    private static void validateThreshold(float threshold) {
        if (threshold < 0.0f || threshold > 1.0f) {
            throw new IllegalArgumentException(
                    "Threshold must be in [0.0, 1.0], got: " + threshold);
        }
    }

    // =========================================================================
    // Result type
    // =========================================================================

    /**
     * Immutable value object representing a single confirmed template match.
     *
     * <p>{@code x} and {@code y} are the coordinates of the <em>top-left corner</em>
     * of the matched region in the source image (not the centre).  To obtain the
     * centre, add half the template dimensions:
     * <pre>{@code
     *   int cx = point.x() + matcher.templateWidth()  / 2;
     *   int cy = point.y() + matcher.templateHeight() / 2;
     * }</pre>
     *
     * @param x          left edge of the matched region in the source image (pixels)
     * @param y          top edge of the matched region in the source image (pixels)
     * @param confidence normalised correlation score in [0.0, 1.0];
     *                   1.0 = pixel-perfect match
     */
    public record MatchPoint(int x, int y, float confidence) {

        /**
         * Returns a human-readable summary for logging and debugging.
         *
         * @return e.g. {@code "MatchPoint{x=142, y=87, confidence=0.9741}"}
         */
        @Override
        public String toString() {
            return String.format("MatchPoint{x=%d, y=%d, confidence=%.4f}", x, y, confidence);
        }
    }

    // =========================================================================
    // Accessors (for external centre-coordinate calculation)
    // =========================================================================

    /** Returns the template width in pixels. */
    public int templateWidth()  { return templateWidth;  }

    /** Returns the template height in pixels. */
    public int templateHeight() { return templateHeight; }
}
