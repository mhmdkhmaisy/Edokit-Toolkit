package edokit.base.models;

import java.util.Objects;

/**
 * HeapImgRef — JVM heap-backed implementation of {@link ImgRef}.
 *
 * <p>This is the simplest and most common {@link ImgRef} variant: the image
 * lives entirely inside a Java {@code byte[]} on the JVM heap, managed by the
 * garbage collector.  No native handles are involved.
 *
 * <p>Typical use cases:
 * <ul>
 *   <li>Images captured from an AWT {@code BufferedImage} and converted to RGBA.</li>
 *   <li>Static reference images loaded from disk for template matching.</li>
 *   <li>Cropped sub-regions produced by {@link EdokitImage#cloneSubImage}.</li>
 * </ul>
 *
 * <p><b>Thread safety:</b> Instances are effectively immutable after construction
 * (the internal {@link EdokitImage} reference cannot be reassigned, and
 * {@link EdokitImage}'s fields are all {@code final}).  However, the underlying
 * {@code byte[]} is mutable; external mutation of {@code image.data} is not
 * guarded here — callers are responsible for synchronisation if they share and
 * mutate pixel data across threads.
 */
public final class HeapImgRef implements ImgRef {

    /** The wrapped heap image. Never null after successful construction. */
    private final EdokitImage image;

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /**
     * Wraps an existing {@link EdokitImage}.
     *
     * @param image the heap image to wrap; must not be {@code null}
     * @throws NullPointerException if {@code image} is {@code null}
     */
    public HeapImgRef(EdokitImage image) {
        this.image = Objects.requireNonNull(image, "image must not be null");
    }

    /**
     * Convenience factory: allocates a new zeroed {@link EdokitImage} of the
     * given dimensions and wraps it.
     *
     * @param width  image width in pixels (must be &gt; 0)
     * @param height image height in pixels (must be &gt; 0)
     * @return a {@code HeapImgRef} backed by a fresh, transparent-black image
     */
    public static HeapImgRef ofSize(int width, int height) {
        return new HeapImgRef(new EdokitImage(width, height));
    }

    /**
     * Convenience factory: wraps raw RGBA bytes without an extra copy.
     *
     * @param width  image width in pixels
     * @param height image height in pixels
     * @param data   raw RGBA bytes; length must equal {@code width * height * 4}
     * @return a {@code HeapImgRef} backed by the provided byte array
     */
    public static HeapImgRef ofBytes(int width, int height, byte[] data) {
        return new HeapImgRef(new EdokitImage(width, height, data));
    }

    // -------------------------------------------------------------------------
    // ImgRef implementation
    // -------------------------------------------------------------------------

    /**
     * {@inheritDoc}
     *
     * <p>Returns the underlying {@link EdokitImage} directly — zero copies,
     * zero allocations.
     */
    @Override
    public EdokitImage toImageData() {
        return image;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Always returns {@code false} — this implementation is entirely
     * JVM-heap-resident and holds no native Mat handle.
     */
    @Override
    public boolean isNative() {
        return false;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates directly to the wrapped {@link EdokitImage#width} field.
     * The JIT will inline this to a field read with no method overhead.
     */
    @Override
    public int getWidth() {
        return image.width;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Delegates directly to the wrapped {@link EdokitImage#height} field.
     */
    @Override
    public int getHeight() {
        return image.height;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /**
     * Direct access to the underlying {@link EdokitImage}.
     *
     * <p>Prefer this over {@link #toImageData()} inside Edokit-internal code
     * that already knows this is a {@code HeapImgRef}, as it communicates
     * intent clearly and avoids the {@link ImgRef} dispatch.
     *
     * @return the wrapped image; never {@code null}
     */
    public EdokitImage getImage() {
        return image;
    }

    // -------------------------------------------------------------------------
    // Object overrides
    // -------------------------------------------------------------------------

    @Override
    public String toString() {
        return "HeapImgRef{" + image + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof HeapImgRef other)) return false;
        return image == other.image;
    }

    @Override
    public int hashCode() {
        return System.identityHashCode(image);
    }
}
