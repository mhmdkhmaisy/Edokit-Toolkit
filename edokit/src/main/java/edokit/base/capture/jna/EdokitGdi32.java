package edokit.base.capture.jna;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.platform.win32.WinDef.HBITMAP;
import com.sun.jna.platform.win32.WinDef.HDC;
import com.sun.jna.platform.win32.WinDef.HGDIOBJ;
import com.sun.jna.platform.win32.WinGDI.BITMAPINFO;
import com.sun.jna.win32.StdCallLibrary;
import com.sun.jna.win32.W32APIOptions;

/**
 * Minimal JNA binding for the Win32 {@code GDI32.dll} functions used by
 * the Edokit GDI capture pipeline.
 *
 * <h2>Capture pipeline call order</h2>
 * <pre>
 *   GetDC(hwnd)                            → windowDC
 *   CreateCompatibleDC(windowDC)           → memDC
 *   CreateCompatibleBitmap(windowDC, w, h) → hBitmap
 *   SelectObject(memDC, hBitmap)           → prevBitmap  (save for restore)
 *
 *   ── per-frame hot path ──────────────────────────────────────────────
 *   BitBlt(memDC, 0,0,w,h, windowDC, 0,0, SRCCOPY)
 *   GetDIBits(memDC, hBitmap, 0, h, pixelPtr, bitmapInfo, DIB_RGB_COLORS)
 *   ── end hot path ────────────────────────────────────────────────────
 *
 *   SelectObject(memDC, prevBitmap)        (restore on close)
 *   DeleteObject(hBitmap)
 *   DeleteDC(memDC)
 *   ReleaseDC(hwnd, windowDC)             (via User32)
 * </pre>
 *
 * <p><b>Constants declared here are scoped to the GDI namespace</b> to avoid
 * polluting the top-level package; callers should reference them as
 * {@code EdokitGdi32.SRCCOPY}, {@code EdokitGdi32.DIB_RGB_COLORS}, etc.
 */
public interface EdokitGdi32 extends StdCallLibrary {

    /** Singleton loaded once at class-init time. */
    EdokitGdi32 INSTANCE = Native.load("gdi32", EdokitGdi32.class,
            W32APIOptions.DEFAULT_OPTIONS);

    // =========================================================================
    // Raster-operation constants used with BitBlt
    // =========================================================================

    /**
     * {@code SRCCOPY} (0x00CC0020) — copies the source rectangle directly to
     * the destination.  This is the fastest and most common raster operation
     * for screen capture.
     */
    int SRCCOPY = 0x00CC0020;

    // =========================================================================
    // GetDIBits usage constant
    // =========================================================================

    /**
     * {@code DIB_RGB_COLORS} (0) — the colour table in the {@code BITMAPINFO}
     * structure contains literal RGB values (as opposed to palette indices).
     * Always use this for 32bpp BI_RGB captures.
     */
    int DIB_RGB_COLORS = 0;

    // =========================================================================
    // Off-screen device context management
    // =========================================================================

    /**
     * Creates a memory device context (DC) compatible with the specified DC.
     *
     * <p>A memory DC is an off-screen surface.  Before it can be used for
     * drawing or copying, a bitmap must be selected into it via
     * {@link #SelectObject}.
     *
     * @param hDC reference DC (window DC); the memory DC will be compatible
     *            with this surface's colour depth and format
     * @return memory DC handle; {@code null} on failure
     */
    HDC CreateCompatibleDC(HDC hDC);

    /**
     * Creates a bitmap compatible with the specified DC's pixel format and
     * large enough to hold a {@code width × height} image.
     *
     * <p>The bitmap is created in unmanaged GDI memory.  It must be freed
     * with {@link #DeleteObject} when no longer needed.
     *
     * @param hDC    reference DC (determines colour depth / format)
     * @param nWidth width of the bitmap in pixels
     * @param nHeight height of the bitmap in pixels
     * @return bitmap handle; {@code null} on failure
     */
    HBITMAP CreateCompatibleBitmap(HDC hDC, int nWidth, int nHeight);

    /**
     * Selects a GDI object (bitmap, pen, brush, font, etc.) into the
     * specified DC, replacing the previously selected object of the same type.
     *
     * <p>The return value is the handle of the <em>previously</em> selected
     * object.  Callers <em>must</em> save this and restore it before deleting
     * the DC to avoid GDI resource leaks.
     *
     * @param hDC    memory DC to modify
     * @param hObject new GDI object to select (e.g. a {@link HBITMAP})
     * @return handle to the previously selected object; save it for restore
     */
    HGDIOBJ SelectObject(HDC hDC, HGDIOBJ hObject);

    // =========================================================================
    // Pixel copy
    // =========================================================================

    /**
     * Performs a bit-block transfer — copies pixel data from the source DC
     * into the destination DC.
     *
     * <p>With {@code dwRop = SRCCOPY}, this is a direct GPU-assisted pixel
     * copy with no compositing overhead.  It operates entirely in kernel/driver
     * space, making it the fastest available GDI capture mechanism.
     *
     * @param hObject    destination DC (our memory DC)
     * @param nXDest     destination X origin (0 for full frame)
     * @param nYDest     destination Y origin (0 for full frame)
     * @param nWidth     number of columns to copy
     * @param nHeight    number of rows to copy
     * @param hObjectSource source DC (the window's device context)
     * @param nXSrc      source X origin
     * @param nYSrc      source Y origin
     * @param dwRop      raster operation code; use {@link #SRCCOPY} for
     *                   straight pixel copy
     * @return {@code true} on success; {@code false} if the call fails
     *         (e.g. occluded or minimised window)
     */
    boolean BitBlt(HDC hObject,
                   int nXDest, int nYDest,
                   int nWidth, int nHeight,
                   HDC hObjectSource,
                   int nXSrc, int nYSrc,
                   int dwRop);

    // =========================================================================
    // Pixel extraction
    // =========================================================================

    /**
     * Retrieves pixel bits from a GDI bitmap and copies them into a
     * caller-supplied buffer in the DIB format described by {@code lpbi}.
     *
     * <h3>32bpp top-down layout (Edokit standard)</h3>
     * Configure {@code lpbi.bmiHeader} with:
     * <ul>
     *   <li>{@code biBitCount = 32}</li>
     *   <li>{@code biCompression = 0} (BI_RGB)</li>
     *   <li>{@code biHeight = -height} (negative → top-down; row 0 is the top)</li>
     * </ul>
     * The output layout is {@code [B, G, R, X]} per pixel (RGBQUAD order),
     * where {@code X} is an unused padding byte (always 0 from BitBlt output).
     * Edokit converts this to {@code [R, G, B, A]} in a subsequent in-place pass.
     *
     * @param hdc        memory DC that owns the bitmap
     * @param hbmp       handle of the GDI bitmap containing source pixels
     * @param uStartScan first scan line to retrieve (0 = top row)
     * @param cScanLines number of scan lines to retrieve
     * @param lpvBits    pre-allocated native buffer to receive raw pixel bytes
     *                   (must be at least {@code width × height × 4} bytes for 32bpp)
     * @param lpbi       {@code BITMAPINFO} describing the desired output format
     * @param uUsage     colour table type; always {@link #DIB_RGB_COLORS}
     * @return number of scan lines successfully written; 0 indicates failure
     */
    int GetDIBits(HDC hdc,
                  HBITMAP hbmp,
                  int uStartScan,
                  int cScanLines,
                  Pointer lpvBits,
                  BITMAPINFO lpbi,
                  int uUsage);

    // =========================================================================
    // Resource cleanup
    // =========================================================================

    /**
     * Deletes a memory device context created by {@link #CreateCompatibleDC}.
     *
     * <p>The DC must have its default bitmap reselected (via
     * {@link #SelectObject}) before calling this, or the original bitmap will
     * be leaked.
     *
     * @param hDC memory DC to delete
     * @return {@code true} if the DC was deleted; {@code false} on error
     */
    boolean DeleteDC(HDC hDC);

    /**
     * Deletes a GDI object and frees all system resources it consumed.
     * Used to release bitmaps created by {@link #CreateCompatibleBitmap}.
     *
     * <p>The object must <em>not</em> be selected into any DC when this is
     * called; doing so produces undefined behaviour and a GDI resource leak.
     *
     * @param hObject GDI object handle to delete (bitmap, pen, brush, etc.)
     * @return {@code true} if the object was deleted; {@code false} on error
     */
    boolean DeleteObject(HGDIOBJ hObject);
}
