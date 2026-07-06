package edokit.base.capture;

import com.sun.jna.Memory;
import com.sun.jna.platform.win32.WinDef.HBITMAP;
import com.sun.jna.platform.win32.WinDef.HDC;
import edokit.base.capture.jna.EdokitGdi32.HGDIOBJ;
import com.sun.jna.platform.win32.WinDef.HWND;
import com.sun.jna.platform.win32.WinDef.RECT;
import com.sun.jna.platform.win32.WinGDI.BITMAPINFO;
import edokit.base.capture.jna.EdokitGdi32;
import edokit.base.capture.jna.EdokitUser32;
import edokit.base.models.EdokitImage;

/**
 * NativeCaptureEngine — Zero-allocation GDI screen-capture loop.
 *
 * <h2>What this does</h2>
 * Binds to a live Win32 window by title, establishes a persistent GDI
 * off-screen device-context pipeline, and captures raw pixel frames at
 * 30+ FPS into a reused {@link EdokitImage} buffer — without allocating
 * a single object inside the hot path after the first frame.
 *
 * <h2>GDI pipeline</h2>
 * <pre>
 *   Target window (HWND)
 *       │
 *       ├─ GetDC(hwnd) ─────────────────────────── windowDC
 *       │       │
 *       │   CreateCompatibleDC(windowDC) ─────────  memDC
 *       │   CreateCompatibleBitmap(windowDC, w, h) ─ hBitmap ─→ SelectObject(memDC)
 *       │
 *       ╔══════════════ per-frame hot path (zero allocs) ══════════════╗
 *       ║  BitBlt(memDC ← windowDC, SRCCOPY)                          ║
 *       ║  GetDIBits(memDC, hBitmap → pixelBuffer[BGRX])              ║
 *       ║  pixelBuffer.read() → currentFrame.data[BGRX]               ║
 *       ║  in-place swap: [B,G,R,X] → [R,G,B,0xFF]                   ║
 *       ╚══════════════════════════════════════════════════════════════╝
 *
 *   close(): SelectObject(prev) → DeleteObject(bitmap) →
 *            DeleteDC(memDC) → ReleaseDC(hwnd, windowDC)
 * </pre>
 *
 * <h2>BGRX → RGBA</h2>
 * GDI's 32-bit BI_RGB layout stores each pixel as an {@code RGBQUAD}:
 * {@code [B, G, R, X]} (B at the lowest address). {@code X} is a reserved
 * padding byte — BitBlt always writes 0 there; it carries no alpha
 * information. The in-place pass swaps R and B and fills A with {@code 0xFF}
 * (fully opaque), producing the RGBA layout expected by {@link EdokitImage}.
 *
 * <h2>Thread safety</h2>
 * <strong>Not thread-safe.</strong> Call {@link #capture()} from a single
 * dedicated capture thread. {@link #close()} may be called from any thread
 * after the capture loop has exited.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 *   try (NativeCaptureEngine engine = new NativeCaptureEngine("RuneScape")) {
 *       while (!Thread.currentThread().isInterrupted()) {
 *           EdokitImage frame = engine.capture();
 *           // process frame ...
 *       }
 *   }
 * }</pre>
 */
public final class NativeCaptureEngine implements AutoCloseable {

    // ── JNA library singletons (loaded once per JVM) ─────────────────────────
    private static final EdokitUser32 USER32 = EdokitUser32.INSTANCE;
    private static final EdokitGdi32  GDI32  = EdokitGdi32.INSTANCE;

    // ── Window handle — resolved once, valid for the engine lifetime ──────────
    private final HWND hwnd;

    // ── GDI pipeline handles — allocated in initGdiResources() ───────────────
    private HDC     windowDC;   // GetDC(hwnd)      — window display surface
    private HDC     memDC;      // CreateCompatibleDC — off-screen memory context
    private HBITMAP hBitmap;    // CreateCompatibleBitmap — our capture surface
    private HGDIOBJ prevBitmap; // SelectObject return — original bitmap, restored on close

    // ── Pre-allocated frame state (zero-alloc hot path) ──────────────────────
    private Memory      pixelBuffer;  // JNA native heap: receives raw BGRX bytes from GDI
    private EdokitImage currentFrame; // reused every capture(); .data[] is the live buffer

    // ── Cached dimensions for resize detection ────────────────────────────────
    private int captureWidth;
    private int captureHeight;

    // ── Reusable structures (written once on init/resize, reused per-frame) ───
    private final BITMAPINFO bitmapInfo = new BITMAPINFO();
    private final RECT       windowRect = new RECT();

    // ── Lifecycle guard ───────────────────────────────────────────────────────
    private volatile boolean closed = false;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * Constructs a capture engine bound to the first top-level window whose
     * title exactly matches {@code windowTitle}.
     *
     * <h3>Composite capture</h3>
     * To include overlay windows rendered on top of the game (e.g. Alt1 Toolkit's
     * transparent timer overlays), this engine uses the <em>desktop device context</em>
     * ({@code GetDC(null)}) rather than the game window's own DC.  Each frame is
     * captured via {@code BitBlt} from the desktop DC at the game window's current
     * screen coordinates.  This composites everything the user sees — game content
     * plus any overlay windows — into a single frame, exactly matching what a
     * screenshot would show.
     *
     * <p>Initialization steps:
     * <ol>
     *   <li>Calls {@code FindWindow(null, windowTitle)} to locate the HWND.</li>
     *   <li>Reads the initial window bounds via {@code GetWindowRect}.</li>
     *   <li>Opens the desktop DC via {@code GetDC(null)}.</li>
     *   <li>Allocates the GDI pipeline and native pixel buffer sized to those
     *       bounds.</li>
     * </ol>
     *
     * @param windowTitle exact Win32 window title to match (e.g. {@code "RuneScape"})
     * @throws IllegalStateException if no matching window is found, or if the
     *                               initial window bounds are zero/invalid
     */
    public NativeCaptureEngine(String windowTitle) {
        hwnd = USER32.FindWindow(null, windowTitle);
        if (hwnd == null) {
            throw new IllegalStateException(
                    "No window found with title \"" + windowTitle + "\". "
                    + "Ensure the game client is running and visible.");
        }

        // Read initial bounds so we can size the GDI resources correctly.
        if (!USER32.GetWindowRect(hwnd, windowRect)) {
            throw new IllegalStateException(
                    "GetWindowRect failed immediately after FindWindow — "
                    + "window handle may be stale.");
        }

        int w = windowRect.right  - windowRect.left;
        int h = windowRect.bottom - windowRect.top;

        if (w <= 0 || h <= 0) {
            throw new IllegalStateException(
                    "Target window reports zero/negative dimensions (" + w + "×" + h + "). "
                    + "Is the window minimised?");
        }

        initGdiResources(w, h);
    }

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * Captures the current contents of the target window into the reused
     * internal {@link EdokitImage} and returns it.
     *
     * <h3>Zero-allocation guarantee</h3>
     * After the first call, no heap objects are allocated inside this method
     * unless the window has been resized since the last call.  On a resize
     * event, GDI resources are reallocated once, then the zero-alloc contract
     * resumes for all subsequent frames at the new size.
     *
     * <h3>Ownership</h3>
     * The returned {@link EdokitImage} is owned by this engine and its
     * {@code data} array is overwritten on every call.  If you need to
     * retain the pixels beyond the next {@code capture()} call, copy them
     * via {@link EdokitImage#cloneSubImage}.
     *
     * @return the engine-owned frame buffer containing the latest RGBA pixels
     * @throws IllegalStateException if the engine is closed, if {@code BitBlt}
     *                               fails (e.g. window minimised), or if
     *                               {@code GetDIBits} returns 0 scan lines
     */
    public EdokitImage capture() {
        if (closed) {
            throw new IllegalStateException(
                    "NativeCaptureEngine.capture() called after close().");
        }

        // ── 1. Query current window bounds ────────────────────────────────────
        if (!USER32.GetWindowRect(hwnd, windowRect)) {
            throw new IllegalStateException(
                    "GetWindowRect failed — has the RuneScape window been closed?");
        }

        final int w = windowRect.right  - windowRect.left;
        final int h = windowRect.bottom - windowRect.top;

        if (w <= 0 || h <= 0) {
            throw new IllegalStateException(
                    "Window reported zero/negative size (" + w + "×" + h + "). "
                    + "Capture is not possible while the window is minimised.");
        }

        // ── 2. Resize GDI resources if the window dimensions changed ──────────
        //    This block allocates; all other paths below are zero-alloc.
        if (w != captureWidth || h != captureHeight) {
            initGdiResources(w, h);
        }

        // ══════════════════════════════════════════════════════════════════════
        //  ZERO-ALLOCATION HOT PATH — no new objects from here to return
        // ══════════════════════════════════════════════════════════════════════

        // ── 3. BitBlt: GPU/driver-level copy from desktop DC → memory DC ──────
        //    Source origin is the game window's current screen position so we
        //    capture exactly the screen region the game occupies — including any
        //    overlay windows (Alt1, etc.) composited on top by the window manager.
        if (!GDI32.BitBlt(memDC, 0, 0, captureWidth, captureHeight,
                          windowDC, windowRect.left, windowRect.top,
                          EdokitGdi32.SRCCOPY | EdokitGdi32.CAPTUREBLT)) {
            throw new IllegalStateException(
                    "BitBlt from desktop DC failed — the window may be minimised. "
                    + "Ensure the game is in windowed or borderless-windowed mode.");
        }

        // ── 4. GetDIBits: extract BGRX bytes from GDI bitmap → native buffer ──
        //    biHeight is negative (top-down), so row 0 in pixelBuffer = top row.
        //    With 32bpp BI_RGB, each pixel is stored as [B, G, R, X] (RGBQUAD).
        final int linesWritten = GDI32.GetDIBits(
                memDC, hBitmap,
                0, captureHeight,
                pixelBuffer, bitmapInfo,
                EdokitGdi32.DIB_RGB_COLORS);

        if (linesWritten == 0) {
            throw new IllegalStateException(
                    "GetDIBits returned 0 scan lines — pixel extraction failed. "
                    + "BITMAPINFO may be misconfigured.");
        }

        // ── 5. Bulk transfer: JNA native heap → Java heap byte[] ──────────────
        //    Memory.read() delegates to a single native memcpy — no element-wise
        //    loop in Java.  After this, currentFrame.data holds raw BGRX pixels.
        final byte[] frameData = currentFrame.data;
        final int    bufSize   = captureWidth * captureHeight * 4;
        pixelBuffer.read(0L, frameData, 0, bufSize);

        // ── 6. In-place BGRX → RGBA channel reorder ───────────────────────────
        //    GDI RGBQUAD layout : [B₀, G₀, R₀, X₀,  B₁, G₁, R₁, X₁, …]
        //    EdokitImage target : [R₀, G₀, B₀, A₀,  R₁, G₁, B₁, A₁, …]
        //
        //    G is already at position [i+1] — no move needed.
        //    X (GDI reserved padding, always 0 from BitBlt) → A = 0xFF (opaque).
        //
        //    Two-register swap of B and R avoids a third temp variable on most
        //    JIT backends.  The loop body is branch-free.
        for (int i = 0; i < bufSize; i += 4) {
            final byte blue       = frameData[i];     // B at BGRX[0]
            frameData[i]          = frameData[i + 2]; // R → RGBA[0]
            frameData[i + 2]      = blue;             // B → RGBA[2]
            frameData[i + 3]      = (byte) 0xFF;      // A (fully opaque)
            // frameData[i + 1] = G — unchanged
        }

        return currentFrame;
    }

    /**
     * Returns the width of the most recently captured frame, in pixels.
     * Returns 0 if no frame has been captured yet.
     */
    public int getCaptureWidth()  { return captureWidth;  }

    /**
     * Returns the height of the most recently captured frame, in pixels.
     * Returns 0 if no frame has been captured yet.
     */
    public int getCaptureHeight() { return captureHeight; }

    /**
     * Returns {@code true} if this engine has been closed.
     */
    public boolean isClosed() { return closed; }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /**
     * Releases all unmanaged GDI resources held by this engine.
     *
     * <p>Must be called exactly once when the capture loop exits; the
     * try-with-resources pattern is strongly recommended.  Calling
     * {@code close()} more than once is safe (idempotent).
     *
     * <p>Resource release order (critical — reversing this causes GDI leaks):
     * <ol>
     *   <li>Restore the original bitmap with {@code SelectObject}.</li>
     *   <li>{@code DeleteObject(hBitmap)} — bitmap must not be selected into
     *       any DC at this point.</li>
     *   <li>{@code DeleteDC(memDC)} — frees the off-screen context.</li>
     *   <li>{@code ReleaseDC(hwnd, windowDC)} — returns the shared window DC
     *       to the system pool.</li>
     * </ol>
     */
    @Override
    public void close() {
        if (closed) return; // idempotent
        closed = true;
        freeGdiResources();
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /**
     * Allocates (or reallocates) all GDI objects and pre-allocated Java buffers
     * for a frame of size {@code width × height}.
     *
     * <p>This is the <em>only</em> method in the capture pipeline that performs
     * heap allocation.  It is called once at construction time and then again
     * only if the window is resized between frames.
     */
    private void initGdiResources(int width, int height) {
        // Release any previously allocated GDI objects first.
        freeGdiResources();

        captureWidth  = width;
        captureHeight = height;

        // Obtain the DESKTOP display surface (GetDC(null)) rather than the game
        // window's own DC.  This gives us a composite view of the full screen at
        // the game window's coordinates, which includes Alt1's transparent overlay
        // window rendered on top.  Using the game-window DC alone would skip any
        // overlay content drawn by other processes.
        windowDC = USER32.GetDC(null);
        if (windowDC == null) {
            throw new IllegalStateException("GetDC(desktop) returned null — unable to access screen surface.");
        }

        // Create an off-screen memory DC compatible with the window's colour depth.
        memDC = GDI32.CreateCompatibleDC(windowDC);
        if (memDC == null) {
            USER32.ReleaseDC(hwnd, windowDC);
            throw new IllegalStateException("CreateCompatibleDC returned null.");
        }

        // Create a bitmap surface to receive BitBlt output.
        hBitmap = GDI32.CreateCompatibleBitmap(windowDC, width, height);
        if (hBitmap == null) {
            GDI32.DeleteDC(memDC);
            USER32.ReleaseDC(hwnd, windowDC);
            throw new IllegalStateException(
                    "CreateCompatibleBitmap returned null for size " + width + "×" + height + ".");
        }

        // Select our bitmap into the memory DC; save the default bitmap handle
        // so we can restore it before deleting the DC (mandatory to avoid leaks).
        // HBITMAP is wrapped as HGDIOBJ because SelectObject accepts any GDI
        // object handle; our locally-defined HGDIOBJ bridges the JNA type gap.
        prevBitmap = GDI32.SelectObject(memDC, new HGDIOBJ(hBitmap.getPointer()));

        // ── BITMAPINFO setup ──────────────────────────────────────────────────
        // Configure once per size; reused for every GetDIBits call on this size.
        //
        // biHeight < 0  → top-down DIB (row 0 = top of image).  Positive biHeight
        //                 produces a bottom-up DIB which would require row reversal.
        // biBitCount=32 → 32 bits per pixel (BGRX / RGBQUAD layout).
        // biCompression=0 → BI_RGB, no compression; colour table is empty.
        // biSizeImage=0   → permissible for BI_RGB; GDI computes it internally.
        bitmapInfo.bmiHeader.biSize        = bitmapInfo.bmiHeader.size();
        bitmapInfo.bmiHeader.biWidth       = width;
        bitmapInfo.bmiHeader.biHeight      = -height; // negative = top-down
        bitmapInfo.bmiHeader.biPlanes      = 1;
        bitmapInfo.bmiHeader.biBitCount    = 32;
        bitmapInfo.bmiHeader.biCompression = 0;       // BI_RGB
        bitmapInfo.bmiHeader.biSizeImage   = 0;

        // ── Native pixel buffer ───────────────────────────────────────────────
        // JNA Memory is allocated in unmanaged heap and auto-freed when GC'd,
        // but we hold a strong reference so it lives exactly as long as the engine.
        pixelBuffer = new Memory((long) width * height * 4);

        // ── Java-heap frame buffer ────────────────────────────────────────────
        // Allocate a fresh EdokitImage; its data[] is the live write target for
        // every subsequent capture() call at this resolution.
        currentFrame = new EdokitImage(width, height);
    }

    /**
     * Frees all currently held GDI handles in the correct teardown order.
     * Safe to call even if some handles were never allocated (null-checked).
     */
    private void freeGdiResources() {
        // Step 1: Restore the original bitmap before touching the DC or bitmap.
        //         Deleting a bitmap that is still selected causes a GDI resource leak.
        if (memDC != null && prevBitmap != null) {
            GDI32.SelectObject(memDC, prevBitmap);
            prevBitmap = null;
        }

        // Step 2: Delete our capture bitmap (now safely deselected).
        if (hBitmap != null) {
            GDI32.DeleteObject(new HGDIOBJ(hBitmap.getPointer()));
            hBitmap = null;
        }

        // Step 3: Delete the off-screen memory DC.
        if (memDC != null) {
            GDI32.DeleteDC(memDC);
            memDC = null;
        }

        // Step 4: Return the desktop DC to the system pool.
        //         Must use null HWND since that is what was passed to GetDC().
        //         Failure to release this leaks one of Windows' finite DC slots.
        if (windowDC != null) {
            USER32.ReleaseDC(null, windowDC);
            windowDC = null;
        }

        // Null-out Java references; the JNA Memory will be GC-eligible once
        // pixelBuffer is cleared, which triggers its internal free().
        pixelBuffer  = null;
        currentFrame = null;
    }
}
