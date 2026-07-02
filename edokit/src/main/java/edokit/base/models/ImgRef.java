package edokit.base.models;

/**
 * ImgRef — Abstraction over different image memory residences.
 *
 * <p>Equivalent to Alt1's {@code ImgRef} concept. In Alt1, images may live in
 * the JavaScript heap, in Alt1's native memory, or as a canvas-backed layout.
 * In Edokit the analogous split is:
 *
 * <ul>
 *   <li><b>Heap-resident</b> — an {@link EdokitImage} backed by a Java
 *       {@code byte[]} on the JVM heap (managed, GC-tracked).</li>
 *   <li><b>Native-resident</b> — an OpenCV {@code Mat} allocated in off-heap
 *       native memory via the OpenCV JNI bridge (unmanaged, must be explicitly
 *       released).</li>
 * </ul>
 *
 * <p>All callers that need raw pixel access should go through
 * {@link #toImageData()}, which performs a download/conversion to a heap
 * {@link EdokitImage} if the data is currently native. Callers that can
 * work directly with the native handle should guard on {@link #isNative()}
 * first to avoid unnecessary copies.
 *
 * <p><b>Lifecycle note:</b> Implementations that hold a native handle (e.g., a
 * future {@code NativeMatImgRef}) must also implement {@link AutoCloseable} and
 * release the underlying {@code Mat} in {@code close()}.  This interface
 * intentionally does not extend {@code AutoCloseable} so that heap-only
 * implementations stay free of that contract.
 */
public interface ImgRef {

    /**
     * Returns the image width in pixels.
     *
     * <p>Must not trigger a native download or any expensive conversion.
     *
     * @return image width in pixels
     */
    int getWidth();

    /**
     * Returns the image height in pixels.
     *
     * <p>Must not trigger a native download or any expensive conversion.
     *
     * @return image height in pixels
     */
    int getHeight();

    /**
     * Returns {@code true} if this reference currently points to live OpenCV
     * native {@code Mat} data in off-heap memory.
     *
     * <p>Callers can use this flag to skip the heap copy when the underlying
     * pipeline can accept a native Mat directly (e.g., passing it to an OpenCV
     * {@code matchTemplate} call).
     *
     * @return {@code true} if backed by a native Mat handle; {@code false} if
     *         backed by a Java heap byte array
     */
    boolean isNative();

    /**
     * Forces the image into a heap-resident {@link EdokitImage}.
     *
     * <p>If this ref is already heap-resident, implementations <em>should</em>
     * return the existing {@link EdokitImage} without copying (zero cost).
     * If this ref is native, implementations <em>must</em> download the pixel
     * data from the native Mat into a new {@link EdokitImage} (one copy cost).
     *
     * <p>The returned {@link EdokitImage} is valid for the lifetime of this
     * {@code ImgRef} unless the implementation documents otherwise.
     *
     * @return a heap-backed {@link EdokitImage} containing this image's pixels
     */
    EdokitImage toImageData();
}
