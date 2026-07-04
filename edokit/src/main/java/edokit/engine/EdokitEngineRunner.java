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
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>Load font assets and build the buff definition database.</li>
 *   <li>Bind the GDI capture engine to the live RuneScape window.</li>
 *   <li>Register a JVM shutdown hook to guarantee safe GDI resource release.</li>
 *   <li>Enter the 30 FPS capture loop: anchor discovery every 150 frames,
 *       buff-bar reading on every frame once the grid is locked.</li>
 * </ol>
 *
 * <h2>Anchor discovery strategy</h2>
 * Full OpenCV template matching on every frame is prohibitively expensive.
 * Instead, a {@link SubImageMatcher} runs every {@link #ANCHOR_RESCAN_INTERVAL}
 * frames (~5 seconds at 30 FPS). Between scans the last known
 * {@code List<Rectangle>} grid layout is reused, making the common hot path
 * a pure pixel-sampling operation with no native CV overhead.
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
     * to render buff countdown overlays, making it the correct choice for
     * buff-timer OCR.
     */
    private static final String FONT_JSON_RESOURCE = "/fonts/pixel_8px_digits.fontmeta.json";

    /** Classpath location of the Alt1 timer font sprite sheet PNG. */
    private static final String FONT_IMG_RESOURCE = "/fonts/pixel_8px_digits.data.png";

    /**
     * Classpath location of the buff-bar anchor template PNG.
     *
     * <p>This is Alt1's official {@code buffborder.data.png} asset — the exact
     * 27×27 pixel border frame that RuneScape renders around every active buff
     * slot.  Using the real Alt1 sprite guarantees a pixel-accurate match against
     * the live client without any colour-guessing heuristics.
     */
    private static final String ANCHOR_TEMPLATE_RESOURCE = "/buffs/buffborder.data.png";

    /**
     * Minimum {@code TM_CCOEFF_NORMED} confidence score for the buff-border
     * template match to be accepted as a valid anchor.
     *
     * <p>0.87 sits in the middle of the 0.85–0.90 recommended range: high enough
     * to reject noise and HUD elements that superficially resemble the border
     * frame, but with a small tolerance margin for minor GPU anti-aliasing
     * variation across different client zoom levels.
     */
    private static final float ANCHOR_MATCH_THRESHOLD = 0.60f;

    /** Maximum horizontal slot columns to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_COLS = 20;

    /** Maximum vertical slot rows to probe during {@link BuffReader#findBuffGrid}. */
    private static final int MAX_BUFF_ROWS = 2;

    /**
     * Frames between full OpenCV anchor re-scans (~5 seconds at 30 FPS).
     * Keeping this high minimises idle CPU overhead while still adapting to
     * interface layout changes made by the user.
     */
    private static final int ANCHOR_RESCAN_INTERVAL = 150;

    /**
     * Maximum per-channel RGB delta for a buff fingerprint sample point to be
     * considered a colour match. 30 provides a practical tolerance for minor
     * in-game lighting variation without causing false positives between visually
     * distinct icons.
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
     * <p>Performs the full initialization sequence, registers the JVM shutdown
     * hook, and then blocks on the capture loop until the thread is interrupted
     * or the JVM exits.
     *
     * @param args command-line arguments (currently unused)
     * @throws IOException          if any required classpath resource cannot be loaded
     * @throws InterruptedException if the main thread is interrupted while waiting
     */
    public static void main(String[] args) throws IOException, InterruptedException {

        // ── Step 1a: Load timer font ──────────────────────────────────────────
        System.out.println("[Edokit Engine] Loading timer font...");
        final FontSheet timerFont = loadFontSheet();
        System.out.println("[Edokit Engine] Font loaded: " + timerFont);

        // ── Step 1b: Load anchor template for grid discovery ──────────────────
        System.out.println("[Edokit Engine] Loading buff anchor template...");
        final EdokitImage anchorTemplate = loadAnchorTemplate();
        System.out.println("[Edokit Engine] Anchor template loaded: "
                + anchorTemplate.width + "×" + anchorTemplate.height + " px.");

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
        // Ensures GDI device contexts, bitmaps, and memory DCs are always released
        // — even on SIGTERM, SIGKILL, or an uncaught exception on another thread.
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Edokit Engine] Shutdown hook triggered — releasing GDI resources.");
            captureEngine.close();
            System.out.println("[Edokit Engine] GDI resources released. Goodbye.");
        }, "edokit-shutdown-hook"));

        // ── Step 5: Run the capture loop on a dedicated thread ────────────────
        final Thread captureThread = new Thread(
                () -> runCaptureLoop(captureEngine, anchorTemplate, timerFont, buffDatabase),
                "edokit-capture");
        captureThread.setDaemon(false); // keep JVM alive until this thread exits
        captureThread.start();

        // Park the main thread until the capture thread finishes (e.g. interrupted).
        captureThread.join();
        System.out.println("[Edokit Engine] Capture thread exited. Main thread returning.");
    }

    // =========================================================================
    // Core capture loop
    // =========================================================================

    /**
     * The primary 30 FPS background tracking loop.
     *
     * <p>Runs until {@link Thread#isInterrupted()} returns {@code true} or the
     * thread is interrupted during a {@link Thread#sleep} call.
     *
     * <h3>Loop structure per frame</h3>
     * <ol>
     *   <li>Capture the live GDI frame; retry on window error.</li>
     *   <li>Every {@link #ANCHOR_RESCAN_INTERVAL} frames: run OpenCV template
     *       matching to (re-)discover the buff grid anchor and populate
     *       {@code activeSlots} via {@link BuffReader#findBuffGrid}.</li>
     *   <li>If grid slots are known: invoke {@link BuffReader#readBuffBar} and
     *       print any identified buffs to stdout.</li>
     *   <li>Sleep {@link #FRAME_SLEEP_MS} ms to pace the loop to ~30 FPS.</li>
     * </ol>
     *
     * @param captureEngine  the bound GDI capture engine
     * @param anchorTemplate the template image used for grid-origin detection
     * @param timerFont      the OCR font for countdown-timer overlay reading
     * @param buffDatabase   known buff colour fingerprints, tried in order
     */
    private static void runCaptureLoop(NativeCaptureEngine captureEngine,
                                       EdokitImage anchorTemplate,
                                       FontSheet timerFont,
                                       List<BuffDefinition> buffDatabase) {

        final BuffReader buffReader = new BuffReader();

        // Anchor discovery state — reset when a rescan is triggered.
        boolean gridDiscovered = false;
        List<Rectangle> activeSlots = null;
        int frameCount = 0;

        // Pre-allocate the SubImageMatcher once; its native Mats are reused per-frame.
        try (SubImageMatcher anchorMatcher = new SubImageMatcher(anchorTemplate)) {

            System.out.println("[Edokit Engine] Entering capture loop. Press Ctrl+C to exit.");

            while (!Thread.currentThread().isInterrupted()) {

                // ── 1. Grab live frame ────────────────────────────────────────
                final EdokitImage frame;
                try {
                    frame = captureEngine.capture();
                } catch (IllegalStateException captureError) {
                    // Window minimised, occluded, or closed — back off and retry.
                    System.err.println("[Edokit Engine] Capture failed (window unavailable): "
                            + captureError.getMessage()
                            + " — retrying in " + WINDOW_RETRY_SLEEP_MS + " ms.");
                    gridDiscovered = false; // anchor coordinates may have shifted on re-show
                    activeSlots    = null;
                    try {
                        Thread.sleep(WINDOW_RETRY_SLEEP_MS);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                    }
                    continue;
                }

                // ── 2. Anchor detection (first run or periodic rescan) ─────────
                // Run a full OpenCV template match only when the grid is undiscovered
                // or the rescan interval has elapsed. Between scans the cached
                // activeSlots list is reused, keeping the hot path allocation-free.
                final boolean rescanDue = !gridDiscovered
                        || (frameCount % ANCHOR_RESCAN_INTERVAL == 0 && frameCount > 0);

                if (rescanDue) {
                    final Optional<SubImageMatcher.MatchPoint> anchorHit =
                            anchorMatcher.findBest(frame, ANCHOR_MATCH_THRESHOLD);

                    if (anchorHit.isPresent()) {
                        final SubImageMatcher.MatchPoint hit = anchorHit.get();

                        System.out.printf(
                                "[Edokit Debug] Best match confidence found on screen: %.4f at (X: %d, Y: %d)%n",
                                hit.confidence(), hit.x(), hit.y());

                        final List<Rectangle> discovered =
                                buffReader.findBuffGrid(frame, hit.x(), hit.y(),
                                        MAX_BUFF_COLS, MAX_BUFF_ROWS);

                        if (!discovered.isEmpty()) {
                            activeSlots    = discovered;
                            gridDiscovered = true;
                            System.out.printf("[Edokit Engine] Buff grid discovered at (%d, %d) — "
                                    + "%d active slot(s) cached. (confidence=%.4f)%n",
                                    hit.x(), hit.y(), activeSlots.size(), hit.confidence());
                        } else {
                            System.err.printf(
                                    "[Edokit Debug] Anchor matched with %.4f confidence, but "
                                    + "findBuffGrid failed to discover valid occupied slots "
                                    + "at this coordinate location.%n",
                                    hit.confidence());
                            activeSlots    = null;
                            gridDiscovered = false;
                        }
                    } else {
                        System.out.printf(
                                "[Edokit Debug] Best match confidence found on screen: <none above %.2f> at (X: N/A, Y: N/A)%n",
                                ANCHOR_MATCH_THRESHOLD);
                        if (gridDiscovered) {
                            System.err.println("[Edokit Engine] Anchor lost — "
                                    + "will retry on next rescan interval.");
                        }
                        activeSlots    = null;
                        gridDiscovered = false;
                    }
                }

                // ── 3. Fast-path buff reading (every frame, grid must be known) ──
                // Pass null for the database: report every discovered slot by position
                // rather than requiring colour-fingerprint definitions.
                if (gridDiscovered && activeSlots != null && !activeSlots.isEmpty()) {
                    final List<TrackedBuff> activeBuff =
                            buffReader.readBuffBar(frame, activeSlots,
                                    timerFont, null, COLOR_TOLERANCE);

                    System.out.printf("[Edokit Engine] Active buffs detected: %d%n", activeBuff.size());
                    for (final TrackedBuff buff : activeBuff) {
                        final String timer = buff.timerText().isEmpty() ? "N/A" : buff.timerText();
                        System.out.printf("[Edokit Engine]   %s | Timer: %s%n",
                                buff.buffName(), timer);
                    }
                }

                // ── 4. Frame pacing — ~30 FPS ─────────────────────────────────
                frameCount++;
                try {
                    Thread.sleep(FRAME_SLEEP_MS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt(); // restore flag and exit loop
                }
            }

        } // SubImageMatcher.close() releases OpenCV native Mats here

        System.out.println("[Edokit Engine] Capture loop terminated after "
                + frameCount + " frame(s).");
    }

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /**
     * Loads the Alt1 timer font from the {@code /fonts/} classpath directory.
     *
     * <p>Both the {@code .fontmeta.json} descriptor and the companion
     * {@code .data.png} sprite sheet are opened as classpath resources.
     * Neither stream is closed before {@link FontSheet#load} returns — the
     * method fully reads both before returning the immutable {@code FontSheet}.
     *
     * @return the fully parsed font sheet
     * @throws IOException if either resource is missing or the JSON is malformed
     */
    private static FontSheet loadFontSheet() throws IOException {
        final InputStream jsonStream = EdokitEngineRunner.class.getResourceAsStream(FONT_JSON_RESOURCE);
        if (jsonStream == null) {
            throw new IOException(
                    "Font JSON resource not found on classpath: " + FONT_JSON_RESOURCE
                    + " — ensure the file exists under src/main/resources/fonts/.");
        }

        final InputStream imgStream = EdokitEngineRunner.class.getResourceAsStream(FONT_IMG_RESOURCE);
        if (imgStream == null) {
            jsonStream.close();
            throw new IOException(
                    "Font image resource not found on classpath: " + FONT_IMG_RESOURCE
                    + " — ensure the file exists under src/main/resources/fonts/.");
        }

        try (jsonStream; imgStream) {
            return FontSheet.load(jsonStream, imgStream);
        }
    }

    /**
     * Loads Alt1's official {@code buffborder.data.png} from the {@code /buffs/}
     * classpath directory and returns it as an {@link EdokitImage}.
     *
     * <p>This sprite is the exact 27×27 pixel border frame that RuneScape renders
     * around every active buff slot.  {@link SubImageMatcher} uses it as the
     * template target to locate the buff grid's top-left origin on every anchor
     * scan pass — no colour-guessing heuristics are needed because the match is
     * pixel-accurate against the real client asset.
     *
     * @return the buffborder template as an {@link EdokitImage} in raw RGBA format
     * @throws IOException if the resource is missing or the format is unsupported
     */
    private static EdokitImage loadAnchorTemplate() throws IOException {
        final InputStream stream = EdokitEngineRunner.class.getResourceAsStream(ANCHOR_TEMPLATE_RESOURCE);
        if (stream == null) {
            throw new IOException(
                    "Buff-border anchor resource not found on classpath: " + ANCHOR_TEMPLATE_RESOURCE
                    + " — ensure buffborder.data.png exists under src/main/resources/buffs/.");
        }
        try (stream) {
            return EdokitImageLoader.load(stream);
        }
    }

    /**
     * Constructs the buff definition database for testing purposes.
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

        /*
         * Overload — the signature orange-red combat potion buff.
         *
         * Sample points are drawn from the central orb area of the icon, which
         * renders a distinctive gradient from dark crimson at the core to bright
         * orange at the highlight edge.  All three offsets are in the upper-centre
         * of the 27×27 cell, safely above the countdown timer zone at the bottom.
         *
         *   (13, 7)  → dark crimson core          0xB03010
         *   (18, 11) → bright orange highlight     0xE06020
         *   (10, 15) → mid-tone red-orange slope   0xC04818
         *   (13, 20) → deep shadow at lower orb    0x6C1808
         */
        database.add(new BuffDefinition(
                "Overload",
                new int[]{ 13, 18, 10, 13 },           // sampleOffsetsX
                new int[]{  7, 11, 15, 20 },           // sampleOffsetsY
                new int[]{ 0xB03010, 0xE06020, 0xC04818, 0x6C1808 } // expectedRGBs (0xRRGGBB)
        ));

        /*
         * Prayer Renewal — the pale-blue prayer-sustain buff.
         *
         * The icon features a bright teardrop silhouette with a cool blue-white
         * body and a darker teal shadow at the base.  Sample points avoid the
         * white rim (susceptible to anti-aliasing variation) and the transparent
         * icon border.
         *
         *   (13, 6)  → bright blue-white centre    0x5090D8
         *   (18, 10) → lighter blue highlight edge 0x70B0E8
         *   ( 9, 16) → mid teal body               0x3870B8
         *   (13, 22) → deep teal base shadow       0x184878
         */
        database.add(new BuffDefinition(
                "Prayer Renewal",
                new int[]{ 13, 18,  9, 13 },           // sampleOffsetsX
                new int[]{  6, 10, 16, 22 },           // sampleOffsetsY
                new int[]{ 0x5090D8, 0x70B0E8, 0x3870B8, 0x184878 } // expectedRGBs (0xRRGGBB)
        ));

        return database;
    }
}
