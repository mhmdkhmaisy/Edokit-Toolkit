package edokit.base.io;

import edokit.base.models.EdokitImage;

import javax.imageio.ImageIO;
import javax.imageio.ImageReadParam;
import javax.imageio.ImageReader;
import javax.imageio.stream.FileImageInputStream;
import javax.imageio.stream.ImageInputStream;
import javax.imageio.stream.MemoryCacheImageInputStream;
import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Iterator;

/**
 * EdokitImageLoader — ICC/gamma-stripped image ingestion pipeline.
 *
 * <h2>The ICC/Gamma problem</h2>
 * {@code ImageIO.read()} is a convenience wrapper that silently applies embedded
 * color-profile corrections ({@code gAMA}, {@code iCCP}, {@code cHRM} PNG chunks,
 * JPEG JFIF/Exif profiles) via the image's {@code ColorModel} before handing
 * data back to the caller.  For RuneScape template matching, we need the raw,
 * uncorrected digital values exactly as stored in the file — the same bytes that
 * would appear if you read the compressed stream with a hex editor.
 *
 * <h2>How this class neutralizes it</h2>
 * <ol>
 *   <li><b>Reader-level suppression:</b> {@code ImageReader.setInput(...,
 *       ignoreMetadata=true)} instructs the PNG/JPEG decoder to skip the
 *       parsing of all metadata chunks — including gamma and ICC profile
 *       records — so the decoder never builds a corrective {@code ColorModel}
 *       in the first place.</li>
 *   <li><b>DataBuffer bypass:</b> Pixel values are extracted directly from
 *       the raw {@code DataBuffer} of the {@code Raster}, never through
 *       {@code BufferedImage.getRGB()} or any {@code ColorModel} path.
 *       Even if the JVM's decoder applied a correction step internally, this
 *       route returns the stored byte values, not the color-managed floats.</li>
 * </ol>
 *
 * <h2>Output contract</h2>
 * The resulting {@link EdokitImage} always contains a flat {@code byte[]} in
 * strict interleaved RGBA order: {@code [R₀, G₀, B₀, A₀, R₁, G₁, B₁, A₁, …]}.
 * Opaque images (no alpha channel) receive {@code 0xFF} alpha for every pixel.
 *
 * <h2>Memory discipline</h2>
 * <ul>
 *   <li>All {@code ImageReader} and {@code ImageInputStream} resources are
 *       released in {@code finally} / try-with-resources blocks.</li>
 *   <li>The fast paths (known {@code BufferedImage} types) cast the
 *       {@code DataBuffer} and walk the raw array once — one allocation (the
 *       output {@code byte[]}) per call, zero GC pressure inside the loop.</li>
 *   <li>The fallback path calls {@code Raster.getPixels()} once, allocating a
 *       single scratch {@code int[]} reused across all pixels.</li>
 * </ul>
 *
 * <h2>Supported formats</h2>
 * Any format for which a {@code javax.imageio.ImageReader} SPI is registered
 * on the current JVM (PNG, JPEG, BMP, GIF, TIFF with the right plugin).
 * PNG is the primary target for Edokit UI template assets.
 */
public final class EdokitImageLoader {

    private EdokitImageLoader() { /* utility class — no instances */ }

    // =========================================================================
    // Public entry points
    // =========================================================================

    /**
     * Loads an image from the file system.
     *
     * <p>Uses a {@link FileImageInputStream} backed by a {@code RandomAccessFile}
     * — no heap buffering, no intermediate copy of the compressed stream.
     *
     * @param file the image file to load (PNG recommended)
     * @return a raw-pixel {@link EdokitImage} in RGBA byte layout
     * @throws IOException              if the file cannot be read or the format
     *                                  is not supported
     * @throws IllegalArgumentException if the decoded image has an unsupported
     *                                  band count
     */
    public static EdokitImage load(File file) throws IOException {
        try (FileImageInputStream iis = new FileImageInputStream(file)) {
            return readFromStream(iis);
        }
    }

    /**
     * Loads an image from a {@link Path}.
     *
     * @param path path to the image file
     * @return a raw-pixel {@link EdokitImage} in RGBA byte layout
     * @throws IOException if the file cannot be read or the format is not supported
     */
    public static EdokitImage load(Path path) throws IOException {
        return load(path.toFile());
    }

    /**
     * Loads an image from an {@link InputStream}.
     *
     * <p>Typically used for classpath resources:
     * <pre>{@code
     *   try (InputStream in = MyClass.class.getResourceAsStream("/templates/bar.png")) {
     *       EdokitImage tpl = EdokitImageLoader.load(in);
     *   }
     * }</pre>
     *
     * <p><b>Note:</b> Non-seekable streams are wrapped in a
     * {@link MemoryCacheImageInputStream}, which buffers the compressed bytes
     * in heap memory.  For large assets, prefer the {@link #load(File)} or
     * {@link #load(Path)} overloads instead.
     *
     * @param in the source stream; the caller is responsible for closing it
     * @return a raw-pixel {@link EdokitImage} in RGBA byte layout
     * @throws IOException if the stream cannot be read or the format is not supported
     */
    public static EdokitImage load(InputStream in) throws IOException {
        // MemoryCacheImageInputStream does NOT close the underlying stream on
        // its own close(), so the caller's try-with-resources on 'in' is safe.
        try (MemoryCacheImageInputStream iis = new MemoryCacheImageInputStream(in)) {
            return readFromStream(iis);
        }
    }

    /**
     * Converts an already-decoded {@link BufferedImage} into an
     * {@link EdokitImage} using the same raw-DataBuffer extraction path.
     *
     * <p>Intended for use by the upcoming {@code NativeCaptureEngine}, which
     * will produce {@code BufferedImage} instances from GDI {@code BitBlt}
     * output and needs to convert them into our canonical format without
     * re-encoding or running through {@code getRGB()}.
     *
     * @param src the source image; must not be {@code null}
     * @return a raw-pixel {@link EdokitImage} in RGBA byte layout
     * @throws IOException if the image band layout is not supported
     */
    public static EdokitImage fromBufferedImage(BufferedImage src) throws IOException {
        return extractRGBA(src);
    }

    // =========================================================================
    // Internal pipeline
    // =========================================================================

    /**
     * Drives the reader, keeping metadata suppressed and releasing resources.
     */
    private static EdokitImage readFromStream(ImageInputStream iis) throws IOException {
        Iterator<ImageReader> it = ImageIO.getImageReaders(iis);
        if (!it.hasNext()) {
            throw new IOException("No ImageReader found for the provided input stream. "
                    + "Ensure the file is a supported format (e.g. PNG, JPEG, BMP).");
        }

        ImageReader reader = it.next();
        try {
            /*
             * seekForwardOnly = true  → reader will not seek backwards; slightly
             *                          faster for formats that support it.
             * ignoreMetadata  = true  → decoder skips gAMA, iCCP, cHRM, Exif, XMP
             *                          and similar color-profile chunks entirely,
             *                          preventing gamma/ICC correction from being
             *                          baked into the constructed ColorModel.
             */
            reader.setInput(iis, true, true);

            ImageReadParam param = reader.getDefaultReadParam();
            BufferedImage decoded = reader.read(0, param);
            return extractRGBA(decoded);
        } finally {
            // Releases the reader's internal native/JNI resources if any.
            reader.dispose();
        }
    }

    /**
     * Converts a {@link BufferedImage} to a flat RGBA {@code byte[]} without
     * passing through any {@code ColorModel} conversion.
     *
     * <h3>Fast paths (cast DataBuffer directly)</h3>
     * <ul>
     *   <li>{@code TYPE_4BYTE_ABGR} / {@code TYPE_4BYTE_ABGR_PRE} — 4 channels,
     *       stored order is {@code [A, B, G, R]} per pixel → reorder to RGBA.</li>
     *   <li>{@code TYPE_3BYTE_BGR} — 3 channels, stored order {@code [B, G, R]}
     *       → reorder to RGBA, synthesize {@code 0xFF} alpha.</li>
     *   <li>{@code TYPE_BYTE_GRAY} — 1 channel → broadcast to R=G=B, alpha
     *       {@code 0xFF}.</li>
     *   <li>{@code TYPE_INT_ARGB} / {@code TYPE_INT_ARGB_PRE} /
     *       {@code TYPE_INT_RGB} — packed {@code int} — unpack to RGBA bytes.</li>
     *   <li>{@code TYPE_BYTE_INDEXED} — palette lookup; handles transparency
     *       entries in {@link IndexColorModel}.</li>
     * </ul>
     *
     * <h3>Fallback path</h3>
     * Unknown types fall back to {@code Raster.getPixels()}, which allocates
     * one {@code int[]} of size {@code width × height × bands} and then exits.
     * Still zero per-pixel allocations.
     */
    private static EdokitImage extractRGBA(BufferedImage src) throws IOException {
        final int w = src.getWidth();
        final int h = src.getHeight();
        final int numPixels = w * h;
        final byte[] rgba = new byte[numPixels * 4];

        final int type = src.getType();
        final WritableRaster raster = src.getRaster();

        switch (type) {

            // ------------------------------------------------------------------
            // 4-byte ABGR — most common result for RGBA PNGs under Java ImageIO
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_4BYTE_ABGR:
            case BufferedImage.TYPE_4BYTE_ABGR_PRE: {
                // DataBuffer raw layout: [A₀, B₀, G₀, R₀,  A₁, B₁, G₁, R₁, …]
                final byte[] raw = ((DataBufferByte) raster.getDataBuffer()).getData();
                for (int src4 = 0, dst4 = 0; src4 < raw.length; src4 += 4, dst4 += 4) {
                    rgba[dst4]     = raw[src4 + 3]; // R  (index 3 in ABGR)
                    rgba[dst4 + 1] = raw[src4 + 2]; // G
                    rgba[dst4 + 2] = raw[src4 + 1]; // B
                    rgba[dst4 + 3] = raw[src4];     // A  (index 0 in ABGR)
                }
                break;
            }

            // ------------------------------------------------------------------
            // 3-byte BGR — common for opaque PNGs and most JPEGs
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_3BYTE_BGR: {
                // DataBuffer raw layout: [B₀, G₀, R₀,  B₁, G₁, R₁, …]
                final byte[] raw = ((DataBufferByte) raster.getDataBuffer()).getData();
                for (int src3 = 0, dst4 = 0; src3 < raw.length; src3 += 3, dst4 += 4) {
                    rgba[dst4]     = raw[src3 + 2]; // R
                    rgba[dst4 + 1] = raw[src3 + 1]; // G
                    rgba[dst4 + 2] = raw[src3];     // B
                    rgba[dst4 + 3] = (byte) 0xFF;   // A — fully opaque
                }
                break;
            }

            // ------------------------------------------------------------------
            // Grayscale — broadcast single channel to R, G, B
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_BYTE_GRAY: {
                final byte[] raw = ((DataBufferByte) raster.getDataBuffer()).getData();
                for (int i = 0, dst4 = 0; i < raw.length; i++, dst4 += 4) {
                    rgba[dst4]     = raw[i]; // R = gray
                    rgba[dst4 + 1] = raw[i]; // G = gray
                    rgba[dst4 + 2] = raw[i]; // B = gray
                    rgba[dst4 + 3] = (byte) 0xFF;
                }
                break;
            }

            // ------------------------------------------------------------------
            // Packed int ARGB — produced by some decoders (e.g. GIF, BMP)
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_INT_ARGB:
            case BufferedImage.TYPE_INT_ARGB_PRE: {
                // DataBuffer raw layout: each int = 0xAARRGGBB
                final int[] raw = ((DataBufferInt) raster.getDataBuffer()).getData();
                for (int i = 0, dst4 = 0; i < raw.length; i++, dst4 += 4) {
                    final int px = raw[i];
                    rgba[dst4]     = (byte) ((px >> 16) & 0xFF); // R
                    rgba[dst4 + 1] = (byte) ((px >> 8)  & 0xFF); // G
                    rgba[dst4 + 2] = (byte) (px          & 0xFF); // B
                    rgba[dst4 + 3] = (byte) ((px >> 24) & 0xFF); // A
                }
                break;
            }

            // ------------------------------------------------------------------
            // Packed int RGB — no alpha channel, 0x00RRGGBB
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_INT_RGB: {
                final int[] raw = ((DataBufferInt) raster.getDataBuffer()).getData();
                for (int i = 0, dst4 = 0; i < raw.length; i++, dst4 += 4) {
                    final int px = raw[i];
                    rgba[dst4]     = (byte) ((px >> 16) & 0xFF); // R
                    rgba[dst4 + 1] = (byte) ((px >> 8)  & 0xFF); // G
                    rgba[dst4 + 2] = (byte) (px          & 0xFF); // B
                    rgba[dst4 + 3] = (byte) 0xFF;                 // A
                }
                break;
            }

            // ------------------------------------------------------------------
            // BGR packed int — less common but produced by some BMP variants
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_INT_BGR: {
                final int[] raw = ((DataBufferInt) raster.getDataBuffer()).getData();
                for (int i = 0, dst4 = 0; i < raw.length; i++, dst4 += 4) {
                    final int px = raw[i];
                    // Stored as 0x00BBGGRR
                    rgba[dst4]     = (byte) (px          & 0xFF); // R
                    rgba[dst4 + 1] = (byte) ((px >> 8)  & 0xFF); // G
                    rgba[dst4 + 2] = (byte) ((px >> 16) & 0xFF); // B
                    rgba[dst4 + 3] = (byte) 0xFF;
                }
                break;
            }

            // ------------------------------------------------------------------
            // Indexed color (palette) — 8-bit PNG with optional transparency
            // ------------------------------------------------------------------
            case BufferedImage.TYPE_BYTE_INDEXED: {
                /*
                 * DataBuffer contains palette indices [0..255], not RGB values.
                 * IndexColorModel holds the actual color table; getRGBs() gives
                 * us the full packed-ARGB palette in one call, without running
                 * any per-pixel conversion.
                 */
                final IndexColorModel icm = (IndexColorModel) src.getColorModel();
                final int mapSize = icm.getMapSize();
                final int[] palette = new int[mapSize]; // one allocation, reused per pixel
                icm.getRGBs(palette);                   // fills with 0xAARRGGBB entries

                final byte[] indices = ((DataBufferByte) raster.getDataBuffer()).getData();
                for (int i = 0, dst4 = 0; i < indices.length; i++, dst4 += 4) {
                    final int entry = palette[indices[i] & 0xFF]; // & 0xFF: unsigned index
                    rgba[dst4]     = (byte) ((entry >> 16) & 0xFF); // R
                    rgba[dst4 + 1] = (byte) ((entry >> 8)  & 0xFF); // G
                    rgba[dst4 + 2] = (byte) (entry          & 0xFF); // B
                    rgba[dst4 + 3] = (byte) ((entry >> 24) & 0xFF); // A (from tRNS chunk)
                }
                break;
            }

            // ------------------------------------------------------------------
            // Fallback — unknown / rare types (e.g. 16-bit USHORT_GRAY, USHORT_555)
            // ------------------------------------------------------------------
            default: {
                final int numBands = raster.getNumBands();
                /*
                 * Raster.getPixels() allocates a single int[] for all pixel
                 * samples and fills it in one JNI-backed call — still far cheaper
                 * than per-pixel getRGB() which would invoke ColorModel
                 * conversion on every iteration.
                 *
                 * For 16-bit samples the values are in [0, 65535]; we scale down
                 * to [0, 255] with an arithmetic right-shift by 8.
                 */
                final DataBuffer db = raster.getDataBuffer();
                final boolean is16bit = (db.getDataType() == DataBuffer.TYPE_USHORT);
                final int[] samples = raster.getPixels(0, 0, w, h, (int[]) null);

                switch (numBands) {
                    case 4 -> {
                        for (int i = 0, dst4 = 0; i < samples.length; i += 4, dst4 += 4) {
                            rgba[dst4]     = downscale(samples[i],     is16bit);
                            rgba[dst4 + 1] = downscale(samples[i + 1], is16bit);
                            rgba[dst4 + 2] = downscale(samples[i + 2], is16bit);
                            rgba[dst4 + 3] = downscale(samples[i + 3], is16bit);
                        }
                    }
                    case 3 -> {
                        for (int i = 0, dst4 = 0; i < samples.length; i += 3, dst4 += 4) {
                            rgba[dst4]     = downscale(samples[i],     is16bit);
                            rgba[dst4 + 1] = downscale(samples[i + 1], is16bit);
                            rgba[dst4 + 2] = downscale(samples[i + 2], is16bit);
                            rgba[dst4 + 3] = (byte) 0xFF;
                        }
                    }
                    case 1 -> {
                        for (int i = 0, dst4 = 0; i < samples.length; i++, dst4 += 4) {
                            final byte gray = downscale(samples[i], is16bit);
                            rgba[dst4]     = gray;
                            rgba[dst4 + 1] = gray;
                            rgba[dst4 + 2] = gray;
                            rgba[dst4 + 3] = (byte) 0xFF;
                        }
                    }
                    default -> throw new IOException(
                            "Unsupported image band count: " + numBands
                            + " (BufferedImage type=" + type + "). "
                            + "Only 1, 3, and 4-band images are supported.");
                }
                break;
            }
        }

        return new EdokitImage(w, h, rgba);
    }

    /**
     * Converts a raw raster sample to a single unsigned byte.
     *
     * <p>For 8-bit images ({@code is16bit=false}) the sample is already in
     * [0, 255]; just cast.  For 16-bit images the sample is in [0, 65535];
     * arithmetic shift by 8 maps it to [0, 255] preserving linearity.
     *
     * @param sample  raw raster sample value
     * @param is16bit {@code true} if the source DataBuffer is {@code USHORT}
     * @return the sample as a (signed) byte whose unsigned interpretation is
     *         in [0, 255]
     */
    private static byte downscale(int sample, boolean is16bit) {
        return (byte) (is16bit ? (sample >> 8) : sample);
    }
}
