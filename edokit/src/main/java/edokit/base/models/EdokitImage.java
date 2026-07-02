package edokit.base.models;

/**
 * EdokitImage — Raw heap-backed RGBA image buffer.
 *
 * <p>Equivalent to Alt1's {@code ImageData}. Pixel data is stored as a flat,
 * single-dimensional {@code byte[]} in interleaved RGBA order:
 *
 * <pre>
 *   index = (4 * x) + (4 * width * y)
 *   data[index + 0] = R
 *   data[index + 1] = G
 *   data[index + 2] = B
 *   data[index + 3] = A
 * </pre>
 *
 * <p><b>Performance contract:</b>
 * <ul>
 *   <li>All fields are {@code final} — the JIT can treat them as constants after
 *       construction and eliminate repeated field-load overhead in hot loops.</li>
 *   <li>Hot methods are marked {@code final} to allow aggressive inlining.</li>
 *   <li>No collections, no boxing, no varargs allocations in pixel-level paths.</li>
 *   <li>Sub-image cropping uses {@link System#arraycopy} for block memory moves
 *       rather than element-wise copies.</li>
 * </ul>
 */
public class EdokitImage {

    /** Width of the image in pixels. */
    public final int width;

    /** Height of the image in pixels. */
    public final int height;

    /**
     * Raw pixel data in interleaved RGBA order.
     * Length is always {@code width * height * 4}.
     */
    public final byte[] data;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Wraps an existing byte array as an EdokitImage.
     *
     * @param width  image width in pixels (must be &gt; 0)
     * @param height image height in pixels (must be &gt; 0)
     * @param data   raw RGBA bytes; length must equal {@code width * height * 4}
     * @throws IllegalArgumentException if dimensions are non-positive or the
     *                                  array length does not match
     */
    public EdokitImage(int width, int height, byte[] data) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException(
                    "Dimensions must be positive, got: " + width + "x" + height);
        }
        int expected = width * height * 4;
        if (data == null || data.length != expected) {
            throw new IllegalArgumentException(
                    "data.length must be width * height * 4 = " + expected
                    + ", got: " + (data == null ? "null" : data.length));
        }
        this.width  = width;
        this.height = height;
        this.data   = data;
    }

    /**
     * Allocates a zeroed (transparent black) EdokitImage of the given size.
     *
     * @param width  image width in pixels
     * @param height image height in pixels
     */
    public EdokitImage(int width, int height) {
        this(width, height, new byte[width * height * 4]);
    }

    // -------------------------------------------------------------------------
    // Core pixel addressing
    // -------------------------------------------------------------------------

    /**
     * Returns the byte-array index of the first channel (R) of the pixel at
     * {@code (x, y)}.
     *
     * <p>Marked {@code final} so the JIT can inline this trivial arithmetic
     * directly into callers, eliminating the method-call frame in tight loops.
     *
     * @param x column (0-based, must be in range [0, width))
     * @param y row    (0-based, must be in range [0, height))
     * @return byte offset of the R channel for the requested pixel
     */
    public final int pixelOffset(int x, int y) {
        return (4 * x) + (4 * width * y);
    }

    // -------------------------------------------------------------------------
    // Pixel read utilities
    // -------------------------------------------------------------------------

    /**
     * Reads a single pixel as an unsigned RGBA quad.
     *
     * <p>Signed {@code byte} values are widened to {@code int} in the range
     * [0, 255] via {@code & 0xFF} masking — no {@link Byte#toUnsignedInt}
     * wrapper call, which avoids a potential boxing path on older JIT tiers.
     *
     * @param x column (0-based)
     * @param y row    (0-based)
     * @return new {@code int[4]} containing {@code [R, G, B, A]}, each in [0, 255]
     */
    public final int[] getRGBA(int x, int y) {
        int i = pixelOffset(x, y);
        return new int[]{
            data[i]     & 0xFF,   // R
            data[i + 1] & 0xFF,   // G
            data[i + 2] & 0xFF,   // B
            data[i + 3] & 0xFF    // A
        };
    }

    /**
     * Reads a single pixel into a caller-supplied array, avoiding allocation
     * entirely in hot loops.
     *
     * @param x      column (0-based)
     * @param y      row    (0-based)
     * @param target pre-allocated {@code int[4]} to write into
     */
    public final void getRGBA(int x, int y, int[] target) {
        int i = pixelOffset(x, y);
        target[0] = data[i]     & 0xFF;
        target[1] = data[i + 1] & 0xFF;
        target[2] = data[i + 2] & 0xFF;
        target[3] = data[i + 3] & 0xFF;
    }

    /**
     * Packs the pixel at {@code (x, y)} into a single {@code int} as
     * {@code 0xAARRGGBB} (standard Java/AWT color integer format).
     *
     * <p>Useful when passing pixel values to AWT/Swing components without
     * creating an intermediate array.
     *
     * @param x column (0-based)
     * @param y row    (0-based)
     * @return packed ARGB integer
     */
    public final int getARGB(int x, int y) {
        int i = pixelOffset(x, y);
        int r = data[i]     & 0xFF;
        int g = data[i + 1] & 0xFF;
        int b = data[i + 2] & 0xFF;
        int a = data[i + 3] & 0xFF;
        return (a << 24) | (r << 16) | (g << 8) | b;
    }

    // -------------------------------------------------------------------------
    // Pixel write utilities
    // -------------------------------------------------------------------------

    /**
     * Writes RGBA channels for the pixel at {@code (x, y)}.
     *
     * @param x column (0-based)
     * @param y row    (0-based)
     * @param r red   channel [0, 255]
     * @param g green channel [0, 255]
     * @param b blue  channel [0, 255]
     * @param a alpha channel [0, 255]
     */
    public final void setRGBA(int x, int y, int r, int g, int b, int a) {
        int i = pixelOffset(x, y);
        data[i]     = (byte) r;
        data[i + 1] = (byte) g;
        data[i + 2] = (byte) b;
        data[i + 3] = (byte) a;
    }

    // -------------------------------------------------------------------------
    // Region operations
    // -------------------------------------------------------------------------

    /**
     * Crops a rectangular sub-region of this image into a new {@link EdokitImage}.
     *
     * <p>Uses {@link System#arraycopy} for each scan-line row so that the JVM
     * can delegate to a native {@code memmove} call rather than interpreting a
     * byte-by-byte loop.
     *
     * @param x         left edge of the sub-region (0-based, inclusive)
     * @param y         top edge of the sub-region (0-based, inclusive)
     * @param subWidth  width of the sub-region in pixels (must be &gt; 0)
     * @param subHeight height of the sub-region in pixels (must be &gt; 0)
     * @return a new {@link EdokitImage} containing only the requested region
     * @throws IllegalArgumentException if the sub-region extends beyond image bounds
     */
    public EdokitImage cloneSubImage(int x, int y, int subWidth, int subHeight) {
        if (x < 0 || y < 0 || subWidth <= 0 || subHeight <= 0
                || x + subWidth > width || y + subHeight > height) {
            throw new IllegalArgumentException(
                    "Sub-region [" + x + ", " + y + ", " + subWidth + "x" + subHeight
                    + "] is out of bounds for image " + width + "x" + height);
        }

        byte[] dest = new byte[subWidth * subHeight * 4];

        // Bytes occupied by one full source row and one destination row.
        int srcRowBytes  = width    * 4;
        int destRowBytes = subWidth * 4;

        // Starting byte offset of the first pixel we want in the source array.
        int srcRowStart = pixelOffset(x, y);

        for (int row = 0; row < subHeight; row++) {
            System.arraycopy(
                data, srcRowStart  + row * srcRowBytes,  // source position
                dest, row * destRowBytes,                // destination position
                destRowBytes                             // bytes to copy (one row)
            );
        }

        return new EdokitImage(subWidth, subHeight, dest);
    }

    // -------------------------------------------------------------------------
    // Object overrides
    // -------------------------------------------------------------------------

    @Override
    public String toString() {
        return "EdokitImage{width=" + width + ", height=" + height
               + ", bytes=" + data.length + "}";
    }
}
