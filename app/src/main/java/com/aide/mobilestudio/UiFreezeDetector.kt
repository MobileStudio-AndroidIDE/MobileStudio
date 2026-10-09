package com.aide.mobilestudio

import android.app.Activity
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.*
import android.util.Log
import android.view.*
import android.widget.*
import kotlinx.coroutines.*
import java.io.File
import java.lang.ref.WeakReference
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * UI Freeze Detector - Android Studio-style ANR detection
 *
 * Monitors the main/UI thread for responsiveness. When the main thread
 * is blocked for >= 3 seconds, shows a diagnostic overlay.
 *
 * The detector itself NEVER blocks the main thread. All diagnostics
 * run on background threads. The freeze-cause work continues running
 * and when it completes, the UI recovers automatically.
 */
class UiFreezeDetector(private val context: Context) {

    companion object {
        private const val TAG = "UiFreezeDetector"
        private const val WARNING_THRESHOLD_MS = 1000L
        private const val CRITICAL_THRESHOLD_MS = 3000L
        private const val HEARTBEAT_INTERVAL_MS = 200L
        private const val HEARTBEAT_TIMEOUT_MS = 500L
    }

    interface FreezeListener {
        fun onFreezeDetected(durationMs: Long)
        fun onFreezeResolved(durationMs: Long)
        fun onDumpCaptured(dumpFile: File)
    }

    @Volatile private var listener: FreezeListener? = null
    @Volatile private var isMonitoring = false
    @Volatile private var isFrozen = false
    @Volatile private var freezeStartTime = 0L
    @Volatile private var overlayView: View? = null

    private val mainHandler = Handler(Looper.getMainLooper())
    private val watchdogThread = HandlerThread("UiFreezeWatchdog").apply { start() }
    private val watchdogHandler = Handler(watchdogThread.looper)
    private val heartbeatToken = AtomicBoolean(false)
    private val heartbeatPending = AtomicBoolean(false)
    private val dumpInProgress = AtomicBoolean(false)
    private val heartbeatCount = AtomicLong(0)
    private val consecutiveFailures = AtomicLong(0)

    private var warningShown = false
    private var activityRef: WeakReference<Activity>? = null

    fun setListener(l: FreezeListener?) { listener = l }

    fun startMonitoring(activity: Activity) {
        if (isMonitoring) return
        isMonitoring = true
        activityRef = WeakReference(activity)
        Log.i(TAG, "Freeze detector started")
        scheduleHeartbeat()
    }

    fun stopMonitoring() {
        isMonitoring = false
        mainHandler.removeCallbacksAndMessages(null)
        watchdogHandler.removeCallbacksAndMessages(null)
        hideOverlayOnMainThread()
        Log.i(TAG, "Freeze detector stopped")
    }

    private fun scheduleHeartbeat() {
        if (!isMonitoring) return
        watchdogHandler.postDelayed({
            if (!isMonitoring) return@postDelayed
            sendHeartbeat()
        }, HEARTBEAT_INTERVAL_MS)
    }

    private fun sendHeartbeat() {
        if (!isMonitoring) return
        if (heartbeatPending.getAndSet(true)) {
            val failures = consecutiveFailures.incrementAndGet()
            val estimatedBlockMs = failures * HEARTBEAT_INTERVAL_MS

            if (estimatedBlockMs >= WARNING_THRESHOLD_MS && !warningShown) {
                warningShown = true
                Log.w(TAG, "Main thread responsive delay detected: ~${estimatedBlockMs}ms")
            }

            if (estimatedBlockMs >= CRITICAL_THRESHOLD_MS && !isFrozen) {
                isFrozen = true
                freezeStartTime = System.currentTimeMillis() - estimatedBlockMs
                Log.e(TAG, "UI FREEZE detected: ~${estimatedBlockMs}ms")
                listener?.onFreezeDetected(estimatedBlockMs)
                showOverlayOnMainThread(estimatedBlockMs)
            }

            if (isFrozen) {
                val duration = System.currentTimeMillis() - freezeStartTime
                updateOverlayDuration(duration)
            }

            watchdogHandler.postDelayed({
                if (isMonitoring) sendHeartbeat()
            }, HEARTBEAT_INTERVAL_MS)
            return
        }

        heartbeatToken.set(false)

        mainHandler.post {
            heartbeatToken.set(true)
        }

        watchdogHandler.postDelayed({
            if (!isMonitoring) return@postDelayed
            heartbeatPending.set(false)

            if (!heartbeatToken.get()) {
                val failures = consecutiveFailures.incrementAndGet()
                val estimatedBlockMs = failures * HEARTBEAT_INTERVAL_MS

                if (estimatedBlockMs >= WARNING_THRESHOLD_MS && !warningShown) {
                    warningShown = true
                }

                if (estimatedBlockMs >= CRITICAL_THRESHOLD_MS && !isFrozen) {
                    isFrozen = true
                    freezeStartTime = System.currentTimeMillis() - estimatedBlockMs
                    Log.e(TAG, "UI FREEZE detected: ~${estimatedBlockMs}ms")
                    listener?.onFreezeDetected(estimatedBlockMs)
                    showOverlayOnMainThread(estimatedBlockMs)
                }

                if (isFrozen) {
                    val duration = System.currentTimeMillis() - freezeStartTime
                    updateOverlayDuration(duration)
                }
            } else {
                consecutiveFailures.set(0)
                if (isFrozen) {
                    val duration = System.currentTimeMillis() - freezeStartTime
                    isFrozen = false
                    warningShown = false
                    Log.i(TAG, "UI freeze resolved after ${duration}ms")
                    listener?.onFreezeResolved(duration)
                    hideOverlayOnMainThread()
                }
            }

            if (isMonitoring) scheduleHeartbeat()
        }, HEARTBEAT_TIMEOUT_MS)
    }

    private fun showOverlayOnMainThread(durationMs: Long) {
        mainHandler.post {
            try {
                val activity = activityRef?.get() ?: return@post
                if (activity.isFinishing || activity.isDestroyed) return@post
                if (overlayView != null) return@post

                val overlay = createFreezeOverlay(activity, durationMs)
                overlayView = overlay

                val rootView = activity.findViewById<ViewGroup>(android.R.id.content)
                val params = FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                rootView.addView(overlay, params)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to show freeze overlay", e)
            }
        }
    }

    private fun hideOverlayOnMainThread() {
        mainHandler.post {
            try {
                overlayView?.let { view ->
                    val parent = view.parent as? ViewGroup
                    parent?.removeView(view)
                }
                overlayView = null
            } catch (e: Exception) {
                Log.e(TAG, "Failed to hide freeze overlay", e)
            }
        }
    }

    private fun updateOverlayDuration(durationMs: Long) {
        mainHandler.post {
            try {
                overlayView?.findViewById<TextView>(R.id.freeze_duration_text)?.text =
                    "UI not responding for ${formatDuration(durationMs)}"
            } catch (_: Exception) {}
        }
    }

    private fun createFreezeOverlay(activity: Activity, durationMs: Long): View {
        val dp = { px: Int -> (px * activity.resources.displayMetrics.density).toInt() }

        val container = FrameLayout(activity).apply {
            setBackgroundColor(Color.parseColor("#80000000"))
            setPadding(dp(24), dp(24), dp(24), dp(24))
        }

        val card = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            background = createRoundedRect(Color.parseColor("#FF1E1E2E"), dp(16).toFloat())
            setPadding(dp(24), dp(20), dp(24), dp(20))
            val cardParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT
            )
            cardParams.gravity = Gravity.CENTER
            cardParams.leftMargin = dp(16)
            cardParams.rightMargin = dp(16)
            layoutParams = cardParams
        }

        card.addView(TextView(activity).apply {
            text = "UI Not Responding"
            setTextColor(Color.parseColor("#FFFF6B6B"))
            textSize = 18f
            setTypeface(null, android.graphics.Typeface.BOLD)
        })

        card.addView(TextView(activity).apply {
            id = R.id.freeze_duration_text
            text = "UI not responding for ${formatDuration(durationMs)}"
            setTextColor(Color.parseColor("#FFDCE3EE"))
            textSize = 14f
            setPadding(0, dp(8), 0, dp(4))
        })

        card.addView(TextView(activity).apply {
            text = "The main thread is blocked. This is usually caused by a long-running\n" +
                    "operation on the UI thread (network, disk I/O, heavy computation).\n" +
                    "The app will recover once the blocking operation completes."
            setTextColor(Color.parseColor("#FF8E99AC"))
            textSize = 12f
            setPadding(0, dp(4), 0, dp(12))
        })

        val buttonRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }

        buttonRow.addView(createButton(activity, "Dump Threads", Color.parseColor("#FF4F8EF7")) {
            captureThreadDump(activity)
        })

        buttonRow.addView(createButton(activity, "Fix", Color.parseColor("#FF98C379")) {
            attemptRecovery(activity)
        })

        buttonRow.addView(createButton(activity, "Dismiss", Color.parseColor("#FF636D83")) {
            hideOverlayOnMainThread()
        })

        card.addView(buttonRow)
        container.addView(card)

        return container
    }

    private fun createButton(context: Context, text: String, color: Int, onClick: () -> Unit): Button {
        val dp = { px: Int -> (px * context.resources.displayMetrics.density).toInt() }
        return Button(context).apply {
            this.text = text
            setTextColor(Color.WHITE)
            textSize = 13f
            setBackgroundColor(color)
            setPadding(dp(16), dp(8), dp(16), dp(8))
            val params = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            params.leftMargin = dp(8)
            layoutParams = params
            setOnClickListener { onClick() }
        }
    }

    private fun captureThreadDump(activity: Activity) {
        if (dumpInProgress.getAndSet(true)) return

        Thread {
            try {
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val dumpDir = File(activity.filesDir, "freeze_dumps")
                dumpDir.mkdirs()
                val dumpFile = File(dumpDir, "freeze_dump_$timestamp.txt")

                val dump = buildThreadDump()
                dumpFile.writeText(dump)

                Log.i(TAG, "Thread dump saved to: ${dumpFile.absolutePath}")
                listener?.onDumpCaptured(dumpFile)

                mainHandler.post {
                    Toast.makeText(activity, "Dump saved: ${dumpFile.name}", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to capture thread dump", e)
            } finally {
                dumpInProgress.set(false)
            }
        }.start()
    }

    private fun attemptRecovery(activity: Activity) {
        Thread {
            try {
                // Try to interrupt blocked threads
                val allThreads = Thread.getAllStackTraces()
                var interrupted = 0
                for ((thread, _) in allThreads) {
                    if (thread.state == Thread.State.BLOCKED || thread.state == Thread.State.WAITING) {
                        try {
                            thread.interrupt()
                            interrupted++
                        } catch (_: Exception) {}
                    }
                }

                // Force GC to free memory
                System.gc()

                // Try to restart shell if it's the cause
                if (activity is MainActivity) {
                    mainHandler.post {
                        try {
                            // Hide the overlay first
                            hideOverlayOnMainThread()
                            Toast.makeText(activity, "Attempting recovery... interrupted $interrupted threads", Toast.LENGTH_SHORT).show()

                            // Try to hide bottom panel if visible
                            val bottomPanel = activity.findViewById<View>(R.id.bottom_panel)
                            if (bottomPanel?.visibility == View.VISIBLE) {
                                CoroutineScope(Dispatchers.Main + SupervisorJob()).launch {
                                    try {
                                        kotlinx.coroutines.delay(300)
                                        bottomPanel.visibility = View.GONE
                                        kotlinx.coroutines.delay(200)
                                        Toast.makeText(activity, "Bottom panel hidden. Try reopening.", Toast.LENGTH_SHORT).show()
                                    } catch (_: Exception) {}
                                }
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Recovery attempt failed", e)
                        }
                    }
                } else {
                    mainHandler.post {
                        hideOverlayOnMainThread()
                        Toast.makeText(activity, "Interrupted $interrupted threads. App may recover.", Toast.LENGTH_SHORT).show()
                    }
                }

                Log.i(TAG, "Recovery attempt: interrupted $interrupted threads")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to attempt recovery", e)
                mainHandler.post {
                    hideOverlayOnMainThread()
                }
            }
        }.start()
    }

    private fun buildThreadDump(): String {
        return buildString {
            appendLine("=== MobileStudio UI Freeze Thread Dump ===")
            appendLine("Timestamp: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())}")
            appendLine("Freeze duration: ${formatDuration(System.currentTimeMillis() - freezeStartTime)}")
            appendLine()

            val allThreads = Thread.getAllStackTraces()
            appendLine("=== All Threads (${allThreads.size}) ===")
            appendLine()

            val mainThread = Thread.currentThread()
            val mainTrace = allThreads[mainThread]
            appendLine("--- MAIN THREAD (ID=${mainThread.id}, State=${mainThread.state}) ---")
            mainTrace?.forEach { element ->
                appendLine("  at $element")
            }
            appendLine()

            val gradleThreads = mutableMapOf<Thread, Array<StackTraceElement>>()
            val indexingThreads = mutableMapOf<Thread, Array<StackTraceElement>>()
            val ioThreads = mutableMapOf<Thread, Array<StackTraceElement>>()
            val otherThreads = mutableMapOf<Thread, Array<StackTraceElement>>()

            for ((thread, trace) in allThreads) {
                if (thread == mainThread) continue
                val name = thread.name.lowercase()
                when {
                    name.contains("gradle") || name.contains("daemon") -> gradleThreads[thread] = trace
                    name.contains("index") || name.contains("symbol") -> indexingThreads[thread] = trace
                    name.contains("io") || name.contains("read") || name.contains("write") ||
                    name.contains("network") || name.contains("http") -> ioThreads[thread] = trace
                    else -> otherThreads[thread] = trace
                }
            }

            if (gradleThreads.isNotEmpty()) {
                appendLine("--- GRADLE THREADS (${gradleThreads.size}) ---")
                for ((thread, trace) in gradleThreads) {
                    appendLine("  Thread: ${thread.name} (ID=${thread.id}, State=${thread.state})")
                    trace?.take(20)?.forEach { appendLine("    at $it") }
                    appendLine()
                }
            }

            if (indexingThreads.isNotEmpty()) {
                appendLine("--- INDEXING THREADS (${indexingThreads.size}) ---")
                for ((thread, trace) in indexingThreads) {
                    appendLine("  Thread: ${thread.name} (ID=${thread.id}, State=${thread.state})")
                    trace?.take(20)?.forEach { appendLine("    at $it") }
                    appendLine()
                }
            }

            if (ioThreads.isNotEmpty()) {
                appendLine("--- I/O THREADS (${ioThreads.size}) ---")
                for ((thread, trace) in ioThreads) {
                    appendLine("  Thread: ${thread.name} (ID=${thread.id}, State=${thread.state})")
                    trace?.take(10)?.forEach { appendLine("    at $it") }
                    appendLine()
                }
            }

            if (otherThreads.isNotEmpty()) {
                appendLine("--- OTHER THREADS (${otherThreads.size}) ---")
                for ((thread, trace) in otherThreads) {
                    appendLine("  Thread: ${thread.name} (ID=${thread.id}, State=${thread.state})")
                    trace?.take(10)?.forEach { appendLine("    at $it") }
                    appendLine()
                }
            }

            appendLine("=== LOCK/BLOCK ANALYSIS ===")
            val blockedThreads = allThreads.filter { it.key.state == Thread.State.BLOCKED }
            if (blockedThreads.isNotEmpty()) {
                for ((thread, trace) in blockedThreads) {
                    appendLine("  BLOCKED: ${thread.name} (ID=${thread.id})")
                    trace?.take(5)?.forEach { appendLine("    at $it") }
                    appendLine()
                }
            } else {
                appendLine("  No blocked threads detected")
            }

            appendLine("=== MEMORY ===")
            val runtime = Runtime.getRuntime()
            appendLine("  Max memory: ${runtime.maxMemory() / 1024 / 1024}MB")
            appendLine("  Total memory: ${runtime.totalMemory() / 1024 / 1024}MB")
            appendLine("  Free memory: ${runtime.freeMemory() / 1024 / 1024}MB")
            appendLine("  Used memory: ${(runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024}MB")

            appendLine()
            appendLine("=== SUMMARY ===")
            appendLine("  Total threads: ${allThreads.size}")
            appendLine("  Main thread state: ${mainThread.state}")
            appendLine("  Blocked threads: ${blockedThreads.size}")
            appendLine("  Waiting threads: ${allThreads.count { it.key.state == Thread.State.WAITING }}")
            appendLine("  Timed waiting threads: ${allThreads.count { it.key.state == Thread.State.TIMED_WAITING }}")
            appendLine("  Runnable threads: ${allThreads.count { it.key.state == Thread.State.RUNNABLE }}")
        }
    }

    private fun createRoundedRect(color: Int, radius: Float): GradientDrawable {
        return GradientDrawable().apply {
            setColor(color)
            cornerRadius = radius
        }
    }

    private fun formatDuration(ms: Long): String {
        return when {
            ms < 1000 -> "${ms}ms"
            ms < 60000 -> String.format("%.1fs", ms / 1000.0)
            else -> String.format("%dm %ds", ms / 60000, (ms % 60000) / 1000)
        }
    }

    fun destroy() {
        stopMonitoring()
        watchdogThread.quitSafely()
        try {
            watchdogThread.join(1000)
        } catch (_: Exception) {}
    }
}
