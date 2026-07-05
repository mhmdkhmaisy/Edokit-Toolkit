package edokit.engine;

import edokit.base.capture.NativeCaptureEngine;
import edokit.base.io.EdokitImageLoader;
import edokit.base.models.EdokitImage;
import edokit.base.vision.SubImageMatcher;
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
 * <h2>Anchor discovery — dual strategy</h2>
 *
 * <h3>Primary: Alt1-style template scan via {@link SubImageMatcher}</h3>
 * Loads {@code buffborder.data.png} — the actual RS3 buff-slot border asset,
 * an outer 1-px ring of olive green {@code (90, 150, 25)} around a 27×27 cell.
 * {@link SubImageMatcher} detects that the template has alpha transparency and
 * automatically switches to Mode A (Alt1-style direct colour comparison):
 * for every candidate screen position it checks how many of the template's
 * opaque border pixels match within {@link SubImageMatcher#ALT1_MATCH_TOLERANCE}
 * per channel.  This correctly replicates Alt1's {@code ImageDetect} algorithm.
 *
 * <h3>Why OpenCV {@code TM_CCOEFF_NORMED} was wrong</h3>
 * <ol>
 *   <li>The template was converted to greyscale before matching, discarding all
 *       colour information — the olive-green border reads as a very similar grey
 *       to many other game UI elements.</li>
 *   <li>{@code TM_CCOEFF_NORMED}'s mean-subtraction step normalises nearly-uniform
 *       colour blocks (like a 1-px border ring) to near-zero variance, producing
 *       unreliable confidence scores regardless of match quality.</li>
 *   <li>The transparent interior of the template was being mapped to black,
 *       polluting the cross-correlation with a spurious dark 25×25 square.</li>
 * </ol>
 *
 * <h3>Fallback: structural scan via {@link BuffReader#scanForBuffBar}</h3>
 * If the template file is missing or the Alt1 scan finds nothing (e.g. because
 * the border was partially clipped by the frame edge), a pixel-level structural
 * scan slides across every position of the captured frame and calls
 * {@link BuffReader#isValidBuffFrame} (which checks the same green colour at
 * 8 border points).
 *
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>Load font assets and build the buff definition database.</li>
 *   <li>Load the buff border template and construct a {@link SubImageMatcher}.</li>
 *   <li>Bind the GDI capture engine to the live RuneScape window.</li>
 *   <li>Register a JVM shutdown hook to guarantee safe GDI resource release.</li>
 *   <li>Enter the 30 FPS capture loop: anchor re-scan every
 *       {@link #ANCHOR_RESCAN_INTERVAL} frames, buff-bar reading every frame
 *       once the grid is locked.</li>
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

    /**
     * Classpath location of the RS3 buff-slot border template.
     * This is the actual Alt1 asset: a 27×27 RGBA image whose outer 1-px ring
     * is olive green {@code (90, 150, 25, 255)} and whose interior is transparent.
     */
    private static final String BUFF_BORDER_RESOURCE = "/buffs/buffborder.data.png";

    /** Maximum horizontal slot columns to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_COLS = 20;

    /** Maximum vertical slot rows to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_ROWS = 2;

    /**
     * Frames between full anchor re-scans (~5 seconds at 30 FPS).
     * Keeping this high minimises idle CPU overhead while still adapting to
     * interface layout changes made by the user.
     */
    private static final int ANCHOR_RESCAN_INTERVAL = 150;

    /**
     * Maximum per-channel RGB delta for a buff fingerprint sample point to be
     * considered a colour match during icon identification.
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

        // ── Step 3: Load buff border template & build matcher ─────────────────
        // buffborder.data.png contains the actual RS3 buff-slot border:
        // 27x27 RGBA, outer 1px ring = (90, 150, 25, 255) olive green, interior transparent.
        // SubImageMatcher detects the alpha channel and auto-selects Alt1-style Mode A.
        System.out.println("[Edokit Engine] Loading buff border template: " + BUFF_BORDER_RESOURCE);
        final SubImageMatcher borderMatcher;
        {
            final InputStream stream = EdokitEngineRunner.class.getResourceAsStream(BUFF_BORDER_RESOURCE);
            if (stream == null) {
                throw new IOException("Buff border template not found on classpath: " + BUFF_BORDER_RESOURCE);
            }
            EdokitImage borderTemplate;
            try (stream) {
                borderTemplate = EdokitImageLoader.load(stream);
            }
            borderMatcher = new SubImageMatcher(borderTemplate);
            System.out.printf(
                    "[Edokit Engine] Border template loaded (%dx%d px, Mode A — Alt1-style colour scan).%n",
                    borderTemplate.width, borderTemplate.height);
        }

        // ── Step 4: Bind the GDI capture engine ───────────────────────────────
        System.out.println("[Edokit Engine] Binding capture engine to window: \""
                + WINDOW_TITLE + "\"...");
        final NativeCaptureEngine captureEngine = new NativeCaptureEngine(WINDOW_TITLE);
        System.out.println("[Edokit Engine] Capture engine ready ("
                + captureEngine.getCaptureWidth() + "×"
                + captureEngine.getCaptureHeight() + " px).");

        // ── Step 5: Register JVM shutdown hook ────────────────────────────────
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Edokit Engine] Shutdown hook triggered — releasing GDI resources.");
            captureEngine.close();
            borderMatcher.close();
            System.out.println("[Edokit Engine] Resources released. Goodbye.");
        }, "edokit-shutdown-hook"));

        // ── Step 6: Run the capture loop on a dedicated thread ────────────────
        final Thread captureThread = new Thread(
                () -> runCaptureLoop(captureEngine, timerFont, buffDatabase, borderMatcher),
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
     * <h3>Anchor discovery (every {@link #ANCHOR_RESCAN_INTERVAL} frames)</h3>
     * <ol>
     *   <li><b>Primary:</b> Alt1-style template scan via {@link SubImageMatcher#findBest}.
     *       The {@code buffborder.data.png} template's opaque pixels (olive-green border ring)
     *       are compared directly against each screen position using per-channel colour
     *       distance.  This is the same algorithm Alt1 uses internally.</li>
     *   <li><b>Fallback:</b> If the template scan finds nothing (no buffs visible or
     *       border partially clipped), {@link BuffReader#scanForBuffBar} performs a
     *       pixel-level structural scan checking the same olive-green colour at 8
     *       border points across every screen position.</li>
     * </ol>
     *
     * @param captureEngine the bound GDI capture engine
     * @param timerFont     the OCR font for countdown-timer overlay reading
     * @param buffDatabase  known buff colour fingerprints, tried in order
     * @param borderMatcher pre-built Alt1-style matcher for the buff border template
     */
    private static void runCaptureLoop(NativeCaptureEngine captureEngine,
                                       FontSheet timerFont,
                                       List<BuffDefinition> buffDatabase,
                                       SubImageMatcher borderMatcher) {

        final BuffReader buffReader = new BuffReader();

        boolean gridDiscovered = false;
        List<Rectangle> activeSlots = null;
        int frameCount = 0;

        System.out.println("[Edokit Engine] Entering capture loop. Press Ctrl+C to exit.");
        System.out.println("[Edokit Engine] Primary anchor: Alt1-style colour scan (buffborder.data.png).");
        System.out.println("[Edokit Engine] Fallback anchor: structural pixel scan (isValidBuffFrame).");

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
            final boolean rescanDue = !gridDiscovered
                    || (frameCount % ANCHOR_RESCAN_INTERVAL == 0 && frameCount > 0);

            if (rescanDue) {
                final int[] anchor = resolveAnchor(frame, borderMatcher, buffReader);

                if (anchor != null) {
                    final int anchorX = anchor[0];
                    final int anchorY = anchor[1];

                    final List<Rectangle> discovered =
                            buffReader.findBuffGrid(frame, anchorX, anchorY,
                                    MAX_BUFF_COLS, MAX_BUFF_ROWS);

                    if (!discovered.isEmpty()) {
                        activeSlots    = discovered;
                        gridDiscovered = true;
                        System.out.printf(
                                "[Edokit Engine] Buff grid at (%d, %d) — %d active slot(s).%n",
                                anchorX, anchorY, activeSlots.size());
                    } else {
                        // Anchor found but grid has no occupied slots.
                        // Could be a false positive; clear and retry next interval.
                        System.err.printf(
                                "[Edokit Debug] Anchor at (%d,%d) found but findBuffGrid "
                                + "returned 0 occupied slots — retrying on next rescan.%n",
                                anchorX, anchorY);
                        activeSlots    = null;
                        gridDiscovered = false;
                    }
                } else {
                    if (gridDiscovered) {
                        System.err.println("[Edokit Engine] Buff bar no longer visible — retrying.");
                    } else if (frameCount % ANCHOR_RESCAN_INTERVAL == 0) {
                        System.out.printf(
                                "[Edokit Debug] No buff border found on %dx%d frame. "
                                + "No active buffs or HUD hidden.%n",
                                frame.width, frame.height);
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
    // Anchor resolution (primary + fallback)
    // =========================================================================

    /**
     * Resolves the buff-bar anchor position using two strategies in order.
     *
     * <ol>
     *   <li><b>Primary</b> — {@link SubImageMatcher#findBest}: Alt1-style colour scan
     *       against {@code buffborder.data.png}. Finds the position where the most
     *       border pixels match the olive-green ring within
     *       {@link SubImageMatcher#ALT1_MATCH_TOLERANCE} per channel.</li>
     *   <li><b>Fallback</b> — {@link BuffReader#scanForBuffBar}: structural pixel scan,
     *       checks the same colour at 8 border positions per candidate cell.</li>
     * </ol>
     *
     * @param frame         the live captured frame
     * @param borderMatcher pre-built Alt1-style matcher
     * @param buffReader    structural scan provider
     * @return {@code {x, y}} of the first buff slot's top-left corner, or {@code null}
     */
    private static int[] resolveAnchor(EdokitImage frame,
                                       SubImageMatcher borderMatcher,
                                       BuffReader buffReader) {
        // ── Primary: Alt1-style template scan ────────────────────────────────
        final long t0 = System.currentTimeMillis();
        final Optional<SubImageMatcher.MatchPoint> templateHit =
                borderMatcher.findBest(frame, SubImageMatcher.DEFAULT_THRESHOLD);
        final long tTemplate = System.currentTimeMillis() - t0;

        if (templateHit.isPresent()) {
            final SubImageMatcher.MatchPoint mp = templateHit.get();
            System.out.printf(
                    "[Edokit Debug] Alt1 template scan hit: (%d, %d) confidence=%.3f in %d ms%n",
                    mp.x(), mp.y(), mp.confidence(), tTemplate);
            return new int[]{ mp.x(), mp.y() };
        }

        System.out.printf(
                "[Edokit Debug] Alt1 template scan: no match above threshold in %d ms. "
                + "Trying structural scan fallback.%n", tTemplate);

        // ── Fallback: structural pixel scan ───────────────────────────────────
        final long t1 = System.currentTimeMillis();
        final Optional<int[]> structuralHit = buffReader.scanForBuffBar(frame);
        final long tStructural = System.currentTimeMillis() - t1;

        if (structuralHit.isPresent()) {
            final int[] pos = structuralHit.get();
            System.out.printf(
                    "[Edokit Debug] Structural scan hit: (%d, %d) in %d ms%n",
                    pos[0], pos[1], tStructural);
            return pos;
        }

        System.out.printf(
                "[Edokit Debug] Structural scan: no buff border found in %d ms.%n",
                tStructural);
        return null;
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
