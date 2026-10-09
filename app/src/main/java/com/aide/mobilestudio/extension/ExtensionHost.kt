package com.aide.mobilestudio.extension

import android.content.Context
import android.util.Log
import dalvik.system.PathClassLoader
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Extension runtime host: lazy-loads extension entry points from installed .msext
 * packages and isolates every lifecycle/error so a crashing extension never takes
 * the IDE down.
 */
object ExtensionHost {

    private const val TAG = "ExtensionHost"
    private val loaded = ConcurrentHashMap<String, LoadedExtension>()
    private var appContext: Context? = null

    fun init(context: Context) {
        if (appContext == null) appContext = context.applicationContext
    }

    /**
     * Lazily loads (or returns the already-loaded) extension by id.
     * Returns null if not installed, disabled, or unloadable (error isolated).
     */
    fun loadOrGet(context: Context, extensionId: String): LoadedExtension? {
        appContext?.let { init(it) }
        loaded[extensionId]?.let { return if (it.isAlive()) it else null }

        val manager = ExtensionManager.get(context)
        val entry = manager.installed().firstOrNull { it.manifest.id == extensionId } ?: return null
        if (!entry.enabled) return null
        val manifest = entry.manifest
        if (manifest.entryPoint.isBlank()) {
            // Metadata-only extension (no code) — still a valid "installed" state
            return LoadedExtension(manifest, null, null).also { loaded[extensionId] = it }
        }

        return try {
            val dexFile = findDex(entry.dir) ?: run {
                Log.w(TAG, "no classes.dex in ${entry.dir.name}")
                return null
            }
            val loader = PathClassLoader(dexFile.absolutePath, context.classLoader)
            val clazz = loader.loadClass(manifest.entryPoint)
            val instance = clazz.getDeclaredConstructor().newInstance() as MobileStudioExtension
            val extContext = DefaultExtensionContext(context, manifest)
            LoadedExtension(manifest, instance, extContext).also { loaded[extensionId] = it }
        } catch (t: Throwable) {
            Log.e(TAG, "extension load failed: $extensionId", t)
            val le = LoadedExtension(manifest, null, null)
            le.crashed = t
            loaded[extensionId] = le
            null
        }
    }

    /** Dispatches a lifecycle call with error isolation; reports crashes via listener. */
    fun dispatch(extensionId: String, call: String, block: (MobileStudioExtension, ExtensionContext) -> Unit) {
        val context = appContext ?: return
        val le = loaded[extensionId] ?: return
        val instance = le.instance ?: return
        val extContext = le.context ?: return
        try {
            block(instance, extContext)
        } catch (t: Throwable) {
            Log.e(TAG, "extension $call crashed: $extensionId", t)
            le.crashed = t
            CrashReporter.onExtensionCrash(le, t)
        }
    }

    fun dispatchUninstall(extensionId: String) {
        dispatch(extensionId, "onUninstall") { ext, ctx -> ext.onUninstall(ctx) }
    }

    fun dispatchFileOpen(extensionId: String, file: File) {
        dispatch(extensionId, "onFileOpen") { ext, ctx -> ext.onFileOpen(ctx, file) }
    }

    /** Dispatches onFileOpen to every enabled installed extension (lazy, isolated). */
    fun dispatchFileOpenForAll(file: File) {
        val context = appContext ?: return
        ExtensionManager.get(context).installed().forEach { entry ->
            if (entry.enabled && !entry.manifest.entryPoint.isBlank()) dispatchFileOpen(entry.manifest.id, file)
        }
    }

    fun dispatchFileSave(extensionId: String, file: File) {
        dispatch(extensionId, "onFileSave") { ext, ctx -> ext.onFileSave(ctx, file) }
    }

    fun dispatchBuildStart(extensionId: String, tasks: List<String>) {
        dispatch(extensionId, "onBuildStart") { ext, ctx -> ext.onBuildStart(ctx, tasks) }
    }

    fun dispatchBuildFinish(extensionId: String, success: Boolean) {
        dispatch(extensionId, "onBuildFinish") { ext, ctx -> ext.onBuildFinish(ctx, success) }
    }

    /** Drops a crashed instance so it can be restarted cleanly. */
    fun restart(extensionId: String) {
        loaded.remove(extensionId)
    }

    private fun findDex(dir: File): File? {
        val direct = File(dir, "classes.dex")
        if (direct.isFile) return direct
        // .msext zip: extracted at install time, but accept the zip itself if it holds the dex
        return dir.listFiles()?.firstOrNull { it.isFile && it.name.endsWith(".msext") }
    }
}

/**
 * Collects extension crash reports and writes them to the extension error log
 * (files/extensions/.error-log.txt) so [[[ERROR]]] dialogs always have content.
 */
object CrashReporter {

    private const val TAG = "ExtensionCrash"
    private var listener: ((LoadedExtension, Throwable) -> Unit)? = null
    private var appContext: Context? = null

    fun init(context: Context) {
        if (appContext == null) appContext = context.applicationContext
    }

    



    fun setListener(l: (LoadedExtension, Throwable) -> Unit) { listener = l }

    fun onExtensionCrash(extension: LoadedExtension, error: Throwable) {
        appContext?.let { init(it) }
        listener?.invoke(extension, error)
        try {
            val ctx = appContext ?: return
            val logFile = File(File(ctx.filesDir, "extensions"), ".error-log.txt")
            logFile.parentFile?.mkdirs()
            val entry = buildString {
                append("=== Extension crash ===\n")
                append("Extension: ${extension.manifest.name} V${extension.manifest.version} (${extension.manifest.id})\n")
                append("Time: ${java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())}\n")
                append("Error: ${error.javaClass.name}: ${error.message}\n")
                append(error.stackTrace.take(30).joinToString("\n") { "  at $it" })
                append("\n\n")
            }
            logFile.appendText(entry)
        } catch (e: Exception) {
            Log.w(TAG, "error log write failed", e)
        }
    }

    fun readLog(context: Context): String {
        val logFile = File(File(context.filesDir, "extensions"), ".error-log.txt")
        return if (logFile.isFile) logFile.readText() else "No extension errors recorded."
    }

    fun clearLog(context: Context) {
        val logFile = File(File(context.filesDir, "extensions"), ".error-log.txt")
        if (logFile.isFile) logFile.delete()
    }
}

/** Default ExtensionContext implementation backed by the core app. */
class DefaultExtensionContext(
    private val context: Context,
    private val manifest: ExtensionManifest
) : ExtensionContext {

    private fun hasPermission(id: String): Boolean = manifest.permissions.contains(id)

    private fun require(permission: String) {
        if (!hasPermission(permission)) {
            throw SecurityException("Extension '${manifest.id}' lacks permission '$permission'")
        }
    }

    override val androidContext: Context get() = context

    // -------- Editor API --------
    override fun currentFile(): File? =
        ExtensionBridge.currentFile?.invoke()

    override fun openFile(file: File) {
        ExtensionBridge.openFile?.invoke(file)
    }

    override fun insertText(text: String) {
        require("editor")
        ExtensionBridge.insertText?.invoke(text)
    }

    override fun selectedText(): String? = ExtensionBridge.selectedText?.invoke()

    override fun registerSyntaxHighlighter(languageId: String, keywords: List<String>, lineComment: String?) {
        require("ui")
        ExtensionBridge.registerSyntaxHighlighter?.invoke(languageId, keywords, lineComment)
    }

    override fun registerAutocompleteProvider(languageId: String, words: List<String>) {
        require("ui")
        ExtensionBridge.registerAutocompleteWords?.invoke(languageId, words)
    }

    // -------- Project API --------
    override fun projectDir(): File? = ExtensionBridge.projectDir?.invoke()

    override fun readFile(path: String): String? {
        require("project.read")
        val root = projectDir() ?: return null
        val f = File(root, path)
        if (!f.canonicalPath.startsWith(root.canonicalPath)) throw SecurityException("path escape")
        return if (f.isFile) f.readText() else null
    }

    override fun writeFile(path: String, content: String): Boolean {
        require("project.write")
        val root = projectDir() ?: return false
        val f = File(root, path)
        if (!f.canonicalPath.startsWith(root.canonicalPath)) throw SecurityException("path escape")
        return runCatching { f.parentFile?.mkdirs(); f.writeText(content); true }.getOrDefault(false)
    }

    override fun listFiles(path: String, maxDepth: Int): List<File> {
        require("project.read")
        val root = projectDir() ?: return emptyList()
        val start = File(root, path)
        if (!start.canonicalPath.startsWith(root.canonicalPath)) throw SecurityException("path escape")
        if (!start.exists()) return emptyList()
        return runCatching { start.walkTopDown().maxDepth(maxDepth).take(500).toList() }.getOrDefault(emptyList())
    }

    // -------- UI / Notification API --------
    override fun showToast(message: String) {
        ExtensionBridge.showToast?.invoke(message)
    }

    override fun notify(message: String, isError: Boolean) {
        ExtensionBridge.notify?.invoke(message, isError)
    }

    override fun progress(title: String, fraction: Float) {
        ExtensionBridge.progress?.invoke(title, fraction)
    }

    // -------- Terminal API --------
    override fun runCommand(command: String, onLine: (String) -> Unit, onExit: (Int) -> Unit) {
        require("terminal")
        ExtensionBridge.runCommand?.invoke(command, onLine, onExit)
    }

    // -------- Storage API --------
    override fun dataDir(): File {
        require("storage")
        val dir = File(File(context.filesDir, "extensions/data"), manifest.id)
        dir.mkdirs()
        return dir
    }

    override fun cacheDir(): File {
        require("storage")
        val dir = File(File(context.cacheDir, "extensions"), manifest.id)
        dir.mkdirs()
        return dir
    }
}
