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
 * <h2>Alpha-mask handling</h2>
 * When the template PNG has an alpha channel (e.g. {@code buffborder.data.png}),
 * transparent interior pixels must NOT participate in the correlation — otherwise
 * OpenCV's {@code COLOR_RGBA2GRAY} maps them to black (0) and the match score
 * collapses because the live screen has non-black icon content at those positions.
 *
 * <p>This class solves that by:
 * <ol>
 *   <li>Extracting the alpha channel from the template RGBA Mat into a dedicated
 *       {@code templateMask} (CV_8UC1, 0=exclude / 255=include).</li>
 *   <li>Zeroing out the grayscale template pixels wherever the mask is 0, so the
 *       grayscale template is fully consistent with the mask.</li>
 *   <li>Passing {@code templateMask} as the 5th argument to {@code matchTemplate}
 *       when any non-zero mask pixels exist.  {@code TM_CCOEFF_NORMED} with a mask
 *       ignores masked-out pixels during both template and image normalisation.</li>
 * </ol>
 * If every mask pixel is 255 (fully opaque template), the mask path is skipped and
 * the standard 4-argument {@code matchTemplate} is used instead.
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
 * (off-heap) memory.  The transfer is performed via
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
 *       └─ (template)          → templateGray (CV_8UC1, pre-loaded, zeros at α=0)
 *                               → templateMask (CV_8UC1, alpha channel, or null if opaque)
 *
 *   matchTemplate(srcGray, templateGray, resultMat, TM_CCOEFF_NORMED[, templateMask])
 *       → resultMat (CV_32FC1, values in [-1.0, +1.0])
 *
 *   resultMat.get() → resultBuffer (float[], Java heap)
 *   NMS peak scan   → List&lt;MatchPoint&gt;
 * </pre>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 *   EdokitImage template = EdokitImageLoader.load(new File("buffborder.data.png"));
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
     * <p>0.85 provides a practical balance for game UI assets: high enough to
     * reject noise, with a small margin for minor GPU anti-aliasing variation
     * across different client zoom levels.  The alpha-mask path eliminates the
     * need to drop this below 0.80 just to compensate for transparent-interior
     * template poisoning.
     */
    public static final float DEFAULT_THRESHOLD = 0.85f;

    // =========================================================================
    // Template state (allocated once, immutable after construction)
    // =========================================================================

    /** Width of the fixed template in pixels. */
    private final int templateWidth;

    /** Height of the fixed template in pixels. */
    private final int templateHeight;

    /**
     * RGBA template Mat (CV_8UC4).  Kept alive to allow future colour-mode
     * extensions; currently the grayscale and mask versions are derived from it.
     */
    private final Mat templateRGBA;

    /**
     * Grayscale template Mat (CV_8UC1).  Transparent pixels (alpha=0) are
     * zeroed out after RGBA→GRAY conversion so they do not contribute spurious
     * correlation energy when the mask path is active.
     */
    private final Mat templateGray;

    /**
     * Alpha-channel mask Mat (CV_8UC1), or {@code null} if the template is
     * fully opaque.  When non-null, every pixel whose alpha=0 has value 0
     * (exclude from match); alpha=255 pixels have value 255 (include).
     *
     * <p>Passed as the optional 5th argument to {@code Imgproc.matchTemplate}
     * with {@code TM_CCOEFF_NORMED}, which honours masks for both CCOEFF and
     * CCORR normalised methods.
     */
    private final Mat templateMask; // null = fully opaque → standard 4-arg matchTemplate

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
     * <p>If the template contains any pixel with alpha &lt; 255, an alpha-mask
     * Mat is extracted and retained.  The grayscale template has its transparent
     * pixel positions zeroed so it is fully consistent with the mask.  All
     * subsequent {@link #findAll}/{@link #findBest} calls use the masked
     * {@code matchTemplate} path automatically.
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

        // ── 1. Load RGBA template into native memory ──────────────────────────
        templateRGBA = new Mat(template.height, template.width, CvType.CV_8UC4);
        templateRGBA.put(0, 0, template.data);

        // ── 2. Convert to grayscale ───────────────────────────────────────────
        //    COLOR_RGBA2GRAY = 0.299R + 0.587G + 0.114B, alpha is IGNORED.
        //    Transparent pixels (R=G=B=A=0) → gray=0 (black).  We correct this
        //    after extracting the mask.
        templateGray = new Mat(template.height, template.width, CvType.CV_8UC1);
        Imgproc.cvtColor(templateRGBA, templateGray, Imgproc.COLOR_RGBA2GRAY);

        // ── 3. Build alpha mask and zero-correct the grayscale template ────────
        //    Scan the raw RGBA bytes to determine whether ANY pixel is transparent.
        //    If so, extract the alpha channel into a dedicated mask Mat and set
        //    the corresponding grayscale pixels to 0.
        final Mat mask = buildAlphaMask(template, templateGray);
        this.templateMask = mask; // null if fully opaque

        System.out.printf(
                "[SubImageMatcher] Template %dx%d loaded — alpha mask: %s%n",
                templateWidth, templateHeight,
                (mask != null ? "ACTIVE (transparent pixels excluded from match)"
                              : "none (fully opaque)"));
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
     *                  use {@link #DEFAULT_THRESHOLD} (0.85) if unsure
     * @return unmodifiable list of distinct match locations; empty if none found
     * @throws IllegalArgumentException if threshold is outside [0.0, 1.0]
     */
    public List<MatchPoint> findAll(EdokitImage source, float threshold) {
        validateThreshold(threshold);

        if (source.width < templateWidth || source.height < templateHeight) {
            return Collections.emptyList();
        }

        ensureSourceMatsAllocated(source.width, source.height);

        srcRGBA.put(0, 0, source.data);
        Imgproc.cvtColor(srcRGBA, srcGray, Imgproc.COLOR_RGBA2GRAY);

        runMatchTemplate();

        final int resultRows = resultMat.rows();
        final int resultCols = resultMat.cols();
        resultMat.get(0, 0, resultBuffer);

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

        runMatchTemplate();

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
        templateRGBA.release();
        templateGray.release();
        if (templateMask != null) templateMask.release();

        if (srcRGBA   != null) srcRGBA.release();
        if (srcGray   != null) srcGray.release();
        if (resultMat != null) resultMat.release();
    }

    // =========================================================================
    // Internal: alpha mask construction
    // =========================================================================

    /**
     * Scans the raw RGBA bytes of {@code template} to determine whether any pixel
     * has alpha &lt; 255.  If so, builds a CV_8UC1 mask Mat (0 = transparent /
     * excluded, 255 = opaque / included) and zeroes out the corresponding pixels
     * in {@code grayMat} so the template is consistent with the mask.
     *
     * <p>If all pixels are fully opaque, returns {@code null} — the standard
     * 4-argument {@code matchTemplate} is used and no mask overhead is paid.
     *
     * @param template source RGBA image
     * @param grayMat  the already-converted grayscale Mat to mutate in place
     * @return the mask Mat, or {@code null} if the template is fully opaque
     */
    private static Mat buildAlphaMask(EdokitImage template, Mat grayMat) {
        final int w = template.width;
        final int h = template.height;
        final byte[] rgba = template.data;
        final int nPix = w * h;

        // Build mask byte array in a single pass.
        // Also collect a zeroed grayscale patch for transparent positions.
        final byte[] maskBytes = new byte[nPix];
        final byte[] grayPatch = new byte[nPix]; // read-modify-write into grayMat

        grayMat.get(0, 0, grayPatch); // pull existing grayscale into Java heap

        boolean anyTransparent = false;
        for (int i = 0, src = 3; i < nPix; i++, src += 4) { // src points at alpha channel
            final int alpha = rgba[src] & 0xFF;
            if (alpha < 255) {
                anyTransparent = true;
                maskBytes[i]   = 0;       // exclude from match
                grayPatch[i]   = 0;       // zero the grayscale value at this position
            } else {
                maskBytes[i]   = (byte) 255; // include in match
                // grayPatch[i] already holds the correct grayscale value
            }
        }

        if (!anyTransparent) {
            return null; // template is fully opaque — no mask needed
        }

        // Push the corrected grayscale back (only the zeroed positions changed).
        grayMat.put(0, 0, grayPatch);

        // Allocate and populate the mask Mat.
        final Mat mask = new Mat(h, w, CvType.CV_8UC1);
        mask.put(0, 0, maskBytes);
        return mask;
    }

    // =========================================================================
    // Internal: Mat lifecycle management
    // =========================================================================

    /**
     * Ensures that {@link #srcRGBA}, {@link #srcGray}, {@link #resultMat}, and
     * {@link #resultBuffer} are allocated and sized for a source of
     * {@code srcW × srcH} pixels.
     */
    private void ensureSourceMatsAllocated(int srcW, int srcH) {
        if (srcW == cachedSrcWidth && srcH == cachedSrcHeight) {
            return;
        }

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

    /**
     * Runs {@code Imgproc.matchTemplate} using the alpha mask when available.
     *
     * <p>When {@link #templateMask} is non-null (template has transparent pixels),
     * the 5-argument overload is used so that masked-out pixels are excluded from
     * both template and image normalisation.  When null (fully opaque template),
     * the standard 4-argument overload is used to avoid any overhead.
     */
    private void runMatchTemplate() {
        if (templateMask != null) {
            Imgproc.matchTemplate(srcGray, templateGray, resultMat,
                                  Imgproc.TM_CCOEFF_NORMED, templateMask);
        } else {
            Imgproc.matchTemplate(srcGray, templateGray, resultMat,
                                  Imgproc.TM_CCOEFF_NORMED);
        }
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
     */
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

    /**
     * Returns {@code true} if the pixel at {@code (row, col)} has the highest
     * value within the suppression neighbourhood of size
     * {@code ±templateWidth × ±templateHeight}.
     */
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
