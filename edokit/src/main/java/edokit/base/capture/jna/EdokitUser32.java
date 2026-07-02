package edokit.base.capture.jna;

import com.sun.jna.Native;
import com.sun.jna.platform.win32.WinDef.HDC;
import com.sun.jna.platform.win32.WinDef.HWND;
import com.sun.jna.platform.win32.WinDef.RECT;
import com.sun.jna.win32.StdCallLibrary;
import com.sun.jna.win32.W32APIOptions;

/**
 * Minimal JNA binding for the Win32 {@code User32.dll} functions used by
 * the Edokit capture pipeline.
 *
 * <p>Only the functions required by {@link edokit.base.capture.NativeCaptureEngine}
 * are declared here — keeping the interface surface small reduces JNA's
 * proxy-generation overhead at class-load time.
 *
 * <p><b>Calling convention:</b> {@link W32APIOptions#DEFAULT_OPTIONS} is applied
 * to the entire interface, which sets:
 * <ul>
 *   <li>StdCall calling convention (required for all Win32 API functions).</li>
 *   <li>Unicode (Wide) string marshalling — {@code FindWindow} resolves to
 *       {@code FindWindowW} automatically.</li>
 *   <li>{@code BOOL} return types are mapped to Java {@code boolean}.</li>
 * </ul>
 *
 * <p><b>Thread safety:</b> The {@code INSTANCE} singleton is safe to call
 * from multiple threads (JNA proxies are stateless). The callers are
 * responsible for synchronising access to the handles they receive.
 */
public interface EdokitUser32 extends StdCallLibrary {

    /**
     * Singleton loaded once at class-init time.  JNA uses a native proxy
     * under the hood; obtaining it is cheap after the first call.
     */
    EdokitUser32 INSTANCE = Native.load("user32", EdokitUser32.class,
            W32APIOptions.DEFAULT_OPTIONS);

    // =========================================================================
    // Window discovery
    // =========================================================================

    /**
     * Finds the first top-level window whose class name and/or window title
     * match the supplied strings.  Either argument may be {@code null} to act
     * as a wildcard.
     *
     * <p>Maps to {@code FindWindowW} (Unicode) via DEFAULT_OPTIONS.
     *
     * @param lpClassName  window class name filter, or {@code null}
     * @param lpWindowName window title filter, or {@code null}
     * @return window handle, or {@code null} if no match is found
     */
    HWND FindWindow(String lpClassName, String lpWindowName);

    // =========================================================================
    // Window geometry
    // =========================================================================

    /**
     * Retrieves the bounding rectangle of the specified window in screen
     * coordinates.  The rectangle includes the window frame, title bar, and
     * all non-client decoration.
     *
     * <p>Use {@code GetClientRect} + {@code ClientToScreen} if you need the
     * inner client area without window chrome.
     *
     * @param hWnd window handle
     * @param rect output structure filled with {@code left}, {@code top},
     *             {@code right}, {@code bottom} screen coordinates
     * @return {@code true} on success
     */
    boolean GetWindowRect(HWND hWnd, RECT rect);

    // =========================================================================
    // Device context management
    // =========================================================================

    /**
     * Returns a handle to a display device context (DC) for the client area of
     * the specified window.
     *
     * <p>The returned DC is a shared, non-owned resource.  It <em>must</em>
     * be released with {@link #ReleaseDC} after use; failing to do so leaks a
     * system-wide GDI object slot (Windows has a finite pool).
     *
     * @param hWnd window handle, or {@code null} for the entire screen DC
     * @return device context handle; {@code null} indicates failure
     */
    HDC GetDC(HWND hWnd);

    /**
     * Releases a device context previously obtained with {@link #GetDC}.
     *
     * <p>The return value indicates whether the DC was released (1) or the
     * call had no effect (0, e.g. the DC is a class or private DC).
     *
     * @param hWnd window handle originally passed to {@code GetDC}
     * @param hDC  device context to release
     * @return 1 if released, 0 otherwise
     */
    int ReleaseDC(HWND hWnd, HDC hDC);
}
