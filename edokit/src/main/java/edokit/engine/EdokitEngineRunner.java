package edokit.engine;

import edokit.base.capture.NativeCaptureEngine;
import edokit.base.io.EdokitImageLoader;
import edokit.base.models.EdokitImage;
import edokit.buffs.BuffReader;
import edokit.buffs.BuffReader.BuffDefinition;
import edokit.buffs.BuffReader.TrackedBuff;
import edokit.ocr.FontSheet;

import java.awt.Rectangle;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * EdokitEngineRunner — Central orchestrator and entry point for the Edokit tracking pipeline.
 *
 * <h2>Anchor discovery — structural scan (no template image required)</h2>
 * The previous approach used OpenCV {@code TM_CCOEFF_NORMED} with the bundled
 * {@code buffborder.data.png}.  That failed for two reasons:
 * <ol>
 *   <li>The bundled PNG is a placeholder green frame {@code (90,150,25)} — it
 *       does not represent the actual RS3 buff-slot border and therefore matches
 *       nothing on screen.</li>
 *   <li>{@code TM_CCOEFF_NORMED} is poorly suited to thin 1-px border templates;
 *       its mean-subtraction step produces near-zero variance for nearly-uniform
 *       dark borders, making the confidence score unreliable regardless of which
 *       template image is used.</li>
 * </ol>
 * The replacement is {@link BuffReader#scanForBuffBar(EdokitImage)}, which slides
 * a {@code 27×27} window across every pixel of the captured frame and tests each
 * position against the 8-point border heuristic.  Because the scan covers the full
 * frame, the buff bar is found wherever the player has placed it — no fixed
 * coordinates, no hardcoded offsets, fully compatible with RS3's movable UI.
 *
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>Load font assets and build the buff definition database.</li>
 *   <li>Bind the GDI capture engine to the live RuneScape window.</li>
 *   <li>Register a JVM shutdown hook to guarantee safe GDI resource release.</li>
 *   <li>Enter the 30 FPS capture loop: full-screen structural anchor scan every
 *       {@link #ANCHOR_RESCAN_INTERVAL} frames, buff-bar reading every frame once
 *       the grid is locked.</li>
 * </ol>
 *
 * <h2>Thread safety</h2>
 * All capture and matching work executes on a single dedicated background thread.
 * The main thread registers the shutdown hook and then parks; the capture thread
 * responds to {@link Thread#interrupt()} for graceful termination.
 */
public final class EdokitEngineRunner {

    // =========================================================================
    // Configuration
    // =========================================================================

    /** Exact Win32 window title of the RuneScape client. */
    private static final String WINDOW_TITLE = "RuneScape";

    /**
     * Classpath location of the Alt1 timer font JSON descriptor.
     * {@code pixel_8px_digits} is the dedicated digit-only font used by Alt1
     * to render buff countdown overlays.
     */
    private static final String FONT_JSON_RESOURCE = "/fonts/pixel_8px_digits.fontmeta.json";

    /** Classpath location of the Alt1 timer font sprite sheet PNG. */
    private static final String FONT_IMG_RESOURCE = "/fonts/pixel_8px_digits.data.png";

    /** Maximum horizontal slot columns to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_COLS = 20;

    /** Maximum vertical slot rows to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_ROWS = 2;

    /**
     * Frames between full structural anchor re-scans (~5 seconds at 30 FPS).
     * Keeping this high minimises idle CPU overhead while still adapting to
     * interface layout changes made by the user.
     */
    private static final int ANCHOR_RESCAN_INTERVAL = 150;

    /**
     * Maximum per-channel RGB delta for a buff fingerprint sample point to be
     * considered a colour match.
     */
    private static final int COLOR_TOLERANCE = 30;

    /** Target frame duration in milliseconds to maintain a stable 30 FPS loop. */
    private static final long FRAME_SLEEP_MS = 33L;

    /** Sleep duration in milliseconds when the target window is unavailable. */
    private static final long WINDOW_RETRY_SLEEP_MS = 1000L;

    // =========================================================================
    // Entry point
    // =========================================================================

    /**
     * Application entry point.
     *
     * @param args command-line arguments (currently unused)
     * @throws IOException          if any required classpath resource cannot be loaded
     * @throws InterruptedException if the main thread is interrupted while waiting
     */
    public static void main(String[] args) throws IOException, InterruptedException {

        // ── Step 1: Load timer font ───────────────────────────────────────────
        System.out.println("[Edokit Engine] Loading timer font...");
        final FontSheet timerFont = loadFontSheet();
        System.out.println("[Edokit Engine] Font loaded: " + timerFont);

        // ── Step 2: Build buff definition database ────────────────────────────
        final List<BuffDefinition> buffDatabase = buildBuffDatabase();
        System.out.println("[Edokit Engine] Buff database initialised: "
                + buffDatabase.size() + " definition(s).");

        // ── Step 3: Bind the GDI capture engine ───────────────────────────────
        System.out.println("[Edokit Engine] Binding capture engine to window: \""
                + WINDOW_TITLE + "\"...");
        final NativeCaptureEngine captureEngine = new NativeCaptureEngine(WINDOW_TITLE);
        System.out.println("[Edokit Engine] Capture engine ready ("
                + captureEngine.getCaptureWidth() + "×"
                + captureEngine.getCaptureHeight() + " px).");

        // ── Step 4: Register JVM shutdown hook ────────────────────────────────
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Edokit Engine] Shutdown hook triggered — releasing GDI resources.");
            captureEngine.close();
            System.out.println("[Edokit Engine] GDI resources released. Goodbye.");
        }, "edokit-shutdown-hook"));

        // ── Step 5: Run the capture loop on a dedicated thread ────────────────
        final Thread captureThread = new Thread(
                () -> runCaptureLoop(captureEngine, timerFont, buffDatabase),
                "edokit-capture");
        captureThread.setDaemon(false);
        captureThread.start();

        captureThread.join();
        System.out.println("[Edokit Engine] Capture thread exited. Main thread returning.");
    }

    // =========================================================================
    // Core capture loop
    // =========================================================================

    /**
     * The primary 30 FPS background tracking loop.
     *
     * <h3>Loop structure per frame</h3>
     * <ol>
     *   <li>Capture the live GDI frame; retry on window error.</li>
     *   <li>Every {@link #ANCHOR_RESCAN_INTERVAL} frames: run a full-screen
     *       structural scan via {@link BuffReader#scanForBuffBar} to (re-)discover
     *       the buff grid anchor.  This replaces OpenCV template matching entirely —
     *       the scan is position-independent and works with RS3's movable UI.</li>
     *   <li>If grid slots are known: invoke {@link BuffReader#readBuffBar} and
     *       print identified buffs to stdout.</li>
     *   <li>Sleep {@link #FRAME_SLEEP_MS} ms to pace the loop to ~30 FPS.</li>
     * </ol>
     *
     * @param captureEngine the bound GDI capture engine
     * @param timerFont     the OCR font for countdown-timer overlay reading
     * @param buffDatabase  known buff colour fingerprints, tried in order
     */
    private static void runCaptureLoop(NativeCaptureEngine captureEngine,
                                       FontSheet timerFont,
                                       List<BuffDefinition> buffDatabase) {

        final BuffReader buffReader = new BuffReader();

        boolean gridDiscovered = false;
        List<Rectangle> activeSlots = null;
        int frameCount = 0;

        System.out.println("[Edokit Engine] Entering capture loop. Press Ctrl+C to exit.");
        System.out.println("[Edokit Engine] Anchor strategy: full-screen structural scan "
                + "(position-independent, works with movable RS3 UI).");

        while (!Thread.currentThread().isInterrupted()) {

            // ── 1. Grab live frame ────────────────────────────────────────────
            final EdokitImage frame;
            try {
                frame = captureEngine.capture();
            } catch (IllegalStateException captureError) {
                System.err.println("[Edokit Engine] Capture failed (window unavailable): "
                        + captureError.getMessage()
                        + " — retrying in " + WINDOW_RETRY_SLEEP_MS + " ms.");
                gridDiscovered = false;
                activeSlots    = null;
                try {
                    Thread.sleep(WINDOW_RETRY_SLEEP_MS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
                continue;
            }

            // ── 2. Anchor detection (first run or periodic rescan) ─────────────
            // Structural scan: slides across every pixel of the frame looking for
            // the RS3 buff-slot border pattern.  Runs on the first frame and then
            // every ANCHOR_RESCAN_INTERVAL frames (~5 s at 30 FPS).
            final boolean rescanDue = !gridDiscovered
                    || (frameCount % ANCHOR_RESCAN_INTERVAL == 0 && frameCount > 0);

            if (rescanDue) {
                final long scanStart = System.currentTimeMillis();
                final Optional<int[]> anchor = buffReader.scanForBuffBar(frame);
                final long scanMs = System.currentTimeMillis() - scanStart;

                if (anchor.isPresent()) {
                    final int anchorX = anchor.get()[0];
                    final int anchorY = anchor.get()[1];

                    System.out.printf(
                            "[Edokit Debug] Structural scan found buff border at (X:%d, Y:%d) in %d ms%n",
                            anchorX, anchorY, scanMs);

                    final List<Rectangle> discovered =
                            buffReader.findBuffGrid(frame, anchorX, anchorY,
                                    MAX_BUFF_COLS, MAX_BUFF_ROWS);

                    if (!discovered.isEmpty()) {
                        activeSlots    = discovered;
                        gridDiscovered = true;
                        System.out.printf(
                                "[Edokit Engine] Buff grid discovered at (%d, %d) — "
                                + "%d active slot(s) cached.%n",
                                anchorX, anchorY, activeSlots.size());
                    } else {
                        // scanForBuffBar found a frame but findBuffGrid mapped zero slots —
                        // this can happen if the single frame the scanner hit is a non-buff
                        // dark border element.  Continue scanning on the next interval.
                        System.err.printf(
                                "[Edokit Debug] Border found at (%d,%d) but findBuffGrid "
                                + "returned 0 occupied slots — will retry on next rescan.%n",
                                anchorX, anchorY);
                        activeSlots    = null;
                        gridDiscovered = false;
                    }
                } else {
                    System.out.printf(
                            "[Edokit Debug] Structural scan: no buff border found on screen "
                            + "(%dx%d frame, scan took %d ms). No active buffs or HUD hidden.%n",
                            frame.width, frame.height, scanMs);
                    if (gridDiscovered) {
                        System.err.println("[Edokit Engine] Anchor lost — retrying on next interval.");
                    }
                    activeSlots    = null;
                    gridDiscovered = false;
                }
            }

            // ── 3. Fast-path buff reading (every frame, grid must be known) ──
            if (gridDiscovered && activeSlots != null && !activeSlots.isEmpty()) {
                final List<TrackedBuff> activeBuff =
                        buffReader.readBuffBar(frame, activeSlots,
                                timerFont, buffDatabase, COLOR_TOLERANCE);

                System.out.printf("[Edokit Engine] Active buffs detected: %d%n", activeBuff.size());
                for (final TrackedBuff buff : activeBuff) {
                    final String timer = buff.timerText().isEmpty() ? "N/A" : buff.timerText();
                    System.out.printf("[Edokit Engine]   %s | Timer: %s%n",
                            buff.buffName(), timer);
                }
            }

            // ── 4. Frame pacing — ~30 FPS ─────────────────────────────────────
            frameCount++;
            try {
                Thread.sleep(FRAME_SLEEP_MS);
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }

        System.out.println("[Edokit Engine] Capture loop terminated after "
                + frameCount + " frame(s).");
    }

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /**
     * Loads the Alt1 timer font from the {@code /fonts/} classpath directory.
     *
     * @return the fully parsed font sheet
     * @throws IOException if either resource is missing or the JSON is malformed
     */
    private static FontSheet loadFontSheet() throws IOException {
        final InputStream jsonStream = EdokitEngineRunner.class.getResourceAsStream(FONT_JSON_RESOURCE);
        if (jsonStream == null) {
            throw new IOException(
                    "Font JSON resource not found on classpath: " + FONT_JSON_RESOURCE);
        }

        final InputStream imgStream = EdokitEngineRunner.class.getResourceAsStream(FONT_IMG_RESOURCE);
        if (imgStream == null) {
            jsonStream.close();
            throw new IOException(
                    "Font image resource not found on classpath: " + FONT_IMG_RESOURCE);
        }

        try (jsonStream; imgStream) {
            return FontSheet.load(jsonStream, imgStream);
        }
    }

    /**
     * Loads an {@link EdokitImage} from a classpath resource path.
     * Kept as a utility for any callers that still need to load PNG assets
     * (e.g., future Alt1-style direct-colour matchers).
     *
     * @param resourcePath classpath-absolute resource path (e.g. {@code "/buffs/buffborder.data.png"})
     * @return loaded image
     * @throws IOException if the resource is missing or the format is unsupported
     */
    @SuppressWarnings("unused")
    static EdokitImage loadResource(String resourcePath) throws IOException {
        final InputStream stream = EdokitEngineRunner.class.getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new IOException("Resource not found on classpath: " + resourcePath);
        }
        try (stream) {
            return EdokitImageLoader.load(stream);
        }
    }

    /**
     * Constructs the buff definition database.
     *
     * <p>Each {@link BuffDefinition} encodes 3–5 pixel sample offsets within
     * the {@code 27×27} buff icon square and the expected colour at each offset,
     * packed as {@code 0xRRGGBB}.  Sample coordinates are chosen from areas of
     * the icon art that are visually unique to that buff and free from timer-digit
     * overlap (which is always cleared by {@link BuffReader#readBuffBar} before
     * fingerprint matching).
     *
     * <p><strong>Production note:</strong> replace these placeholder entries with
     * real fingerprints extracted from live RuneScape icon textures using the
     * Edokit pixel sampler utility.
     *
     * @return a mutable list of buff definitions tried in declaration order
     */
    private static List<BuffDefinition> buildBuffDatabase() {
        final List<BuffDefinition> database = new ArrayList<>();

        database.add(new BuffDefinition(
                "Overload",
                new int[]{ 13, 18, 10, 13 },
                new int[]{  7, 11, 15, 20 },
                new int[]{ 0xB03010, 0xE06020, 0xC04818, 0x6C1808 }
        ));

        database.add(new BuffDefinition(
                "Prayer Renewal",
                new int[]{ 13, 18,  9, 13 },
                new int[]{  6, 10, 16, 22 },
                new int[]{ 0x5090D8, 0x70B0E8, 0x3870B8, 0x184878 }
        ));

        return database;
    }
}
