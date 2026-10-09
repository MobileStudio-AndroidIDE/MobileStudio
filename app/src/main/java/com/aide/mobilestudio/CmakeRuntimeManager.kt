package com.aide.mobilestudio

import android.content.Context
import android.os.StatFs
import android.util.Log
import java.io.File
import java.net.URL

/**
 * CmakeRuntimeManager — downloads the user-ported CMake 3.22.1 Android runtime
 * (arm64-v8a) from the MobileStudio GitHub release.
 *
 * Version management: exactly ONE version is distributed (cmake-android-v3.22.1);
 * bump the constants below when a new release is published and reinstall.
 *
 * Security policy (same as LlamaRuntimeManager / ToolchainArchiveInstaller):
 *  - HTTPS download only, SHA-256 verification MANDATORY
 *  - safe tar.gz extraction with path traversal protection
 *  - installed runtime is never re-downloaded on app restart
 *
 * Install layout (archive root flattened during extraction):
 *   <filesDir>/toolchain/cmake/3.22.1/bin/cmake, cpack, ctest
 *   <filesDir>/toolchain/cmake/3.22.1/share/cmake-3.22/...
 *
 * ToolchainManager.Kind.CMAKE resolves <filesDir>/toolchain/cmake/<version>,
 * so the installed port is picked up by the build pipeline automatically.
 */
class CmakeRuntimeManager private constructor(private val context: Context?) {
    companion object {
        private const val TAG = "CmakeRuntimeManager"

        /** The only distributed version (user-ported). */
        const val CMAKE_VERSION = "3.22.1"
        const val DOWNLOAD_URL =
            "https://github.com/MobileStudio-AndroidIDE/MobileStudio_AndroidIDE/releases/download/cmake-android-v3.22.1/cmake-3.22.1-android-arm64.tar.gz"

        /** SHA-256 of the release archive (verified against the actual download). */
        const val EXPECTED_SHA256 = "092133e9440b88bf5dcbe916594cf2f69c7dd35104f77c4551eb34c73460eea0"

        /** Actual archive size (verified against the GitHub release). */
        const val ESTIMATED_ARCHIVE_BYTES = 12_919_431L

        private const val MARKER = ".cmake-runtime-version"

        @Volatile private var appContext: Context? = null
        @Volatile private var shared: CmakeRuntimeManager? = null

        /** Call early so context-free access works. */
        fun initAppContext(ctx: Context) {
            appContext = ctx.applicationContext
        }

        /** Shared instance (context optional once initAppContext was called). */
        fun get(context: Context? = null): CmakeRuntimeManager {
            shared?.let { return it }
            synchronized(this) {
                shared?.let { return it }
                val ctx = context?.applicationContext ?: appContext
                    ?: error("CmakeRuntimeManager: no context (call initAppContext)")
                return CmakeRuntimeManager(ctx).also { shared = it }
            }
        }
    }

    private val ctx: Context get() = context ?: appContext
        ?: error("CmakeRuntimeManager: no context (call initAppContext)")
    private val cmakeRoot by lazy { File(ToolchainManager(ctx).rootDir, "cmake") }
    private val versionDir get() = File(cmakeRoot, CMAKE_VERSION)
    private val markerFile get() = File(versionDir, MARKER)
    private val tmpFile get() = File(cmakeRoot, "download.tmp")
    private val installLock = Any()

    /** Directory containing bin/cmake — passed to the build pipeline. */
    fun cmakeDir(): File = versionDir

    fun installedVersion(): String? = if (isInstalled()) CMAKE_VERSION else null

    /**
     * True when installed: marker present AND bin/cmake is a real ELF.
     * Symlinks and non-ELF files count as damaged — on-device symlink-repair
     * passes previously replaced the binary with self-referential links.
     */
    fun isInstalled(): Boolean {
        if (!markerFile.isFile) return false
        if (markerFile.readText().trim() != CMAKE_VERSION) return false
        val cmakeBin = File(versionDir, "bin/cmake")
        if (!cmakeBin.isFile) return false
        if (runCatching { java.nio.file.Files.isSymbolicLink(cmakeBin.toPath()) }.getOrDefault(false)) return false
        return NativeToolWrappers.isElf(cmakeBin)
    }

    /**
     * Ensures the CMake runtime is installed. Non-blocking (background thread):
     *  1. already installed → done immediately (no re-download)
     *  2. damaged partial download (.tmp) → deleted
     *  3. download with progress ("4 MB / 12 MB")
     *  4. SHA-256 verification (mismatch → delete + error)
     *  5. safe extraction (path traversal protected, archive root flattened)
     *  6. version marker + executable bits written
     *
     * onDone(success, error): on failure the caller may retry by calling again.
     */
    fun ensureInstalled(
        onProgress: (bytesRead: Long, totalBytes: Long) -> Unit,
        onLog: (String) -> Unit,
        onDone: (Boolean, String?) -> Unit
    ) {
        if (isInstalled()) {
            onLog("[CMake] Installed: $CMAKE_VERSION (no re-download)")
            onDone(true, null)
            return
        }
        Thread {
            try {
                installBlocking(onProgress, onLog)
                onLog("[CMake] Installed: $CMAKE_VERSION")
                onLog("[CMake] Path: ${versionDir.absolutePath}")
                onDone(true, null)
            } catch (e: Exception) {
                Log.e(TAG, "CMake runtime install failed", e)
                tmpFile.delete()
                onLog("[CMake ERROR] Install failed: ${e.message}")
                onDone(false, e.message)
            }
        }.start()
    }

    /**
     * Blocking variant for background threads that already run off the UI
     * thread (the Gradle build flow). Returns immediately when installed;
     * throws on failure. Concurrent callers serialize on [installLock].
     */
    fun ensureInstalledBlocking(
        onProgress: (bytesRead: Long, totalBytes: Long) -> Unit = { _, _ -> },
        onLog: (String) -> Unit = {}
    ) {
        if (isInstalled()) return
        synchronized(installLock) {
            if (isInstalled()) return
            try {
                installBlocking(onProgress, onLog)
                onLog("[CMake] Installed: $CMAKE_VERSION (${versionDir.absolutePath})")
            } catch (e: Exception) {
                Log.e(TAG, "CMake runtime install failed", e)
                tmpFile.delete()
                throw e
            }
        }
    }

    /** Download + verify + extract (GitHub release URL, SHA-256 mandatory). */
    private fun installBlocking(
        onProgress: (bytesRead: Long, totalBytes: Long) -> Unit,
        onLog: (String) -> Unit
    ) {
        synchronized(installLock) {
            if (tmpFile.isFile) {
                onLog("[CMake] Removing damaged partial download...")
                tmpFile.delete()
            }

            val stat = StatFs(ctx.filesDir.path)
            val available = stat.availableBlocksLong * stat.blockSizeLong
            if (available < ESTIMATED_ARCHIVE_BYTES + 256L * 1024 * 1024) {
                throw IllegalStateException(
                    "Insufficient storage for CMake runtime: available " +
                        ToolchainArchiveInstaller.formatSize(available)
                )
            }

            onLog("[CMake] Downloading CMake $CMAKE_VERSION (arm64-v8a)...")
            ToolchainArchiveInstaller.download(tmpFile, URL(DOWNLOAD_URL), onProgress)
            onLog("[CMake] Download complete (${ToolchainArchiveInstaller.formatSize(tmpFile.length())})")

            onLog("[CMake] Verifying SHA-256...")
            ToolchainArchiveInstaller.verifySha256(tmpFile, EXPECTED_SHA256)
            onLog("[CMake] SHA-256 OK")

            onLog("[CMake] Extracting (archive root flattened)...")
            // Extract into a FRESH staging dir. versionDir may hold leftovers
            // from older installs (a nested archive root, a stray share/)
            // which defeat the single-root flattening pass and leave the
            // tree without a top-level bin/cmake. Swap only after the staged
            // tree is verified.
            val stageDir = File(cmakeRoot, "extract.tmp")
            stageDir.deleteRecursively()
            ToolchainArchiveInstaller.extractTarGz(tmpFile, stageDir) {
                onLog("[CMake] $it")
            }
            tmpFile.delete()

            val cmakeBin = File(stageDir, "bin/cmake")
            if (!cmakeBin.isFile || !NativeToolWrappers.isElf(cmakeBin)) {
                val found = stageDir.walkTopDown().filter { it.isFile }
                    .map { it.name }.toList()
                onLog("[CMake ERROR] Extraction incomplete — bin/cmake missing")
                onLog("[CMake ERROR]   found: ${found.take(20).joinToString(", ")}")
                stageDir.deleteRecursively()
                throw IllegalStateException("Extraction incomplete: bin/cmake missing")
            }

            if (versionDir.exists()) versionDir.deleteRecursively()
            if (!stageDir.renameTo(versionDir)) {
                stageDir.copyRecursively(versionDir, overwrite = true)
                stageDir.deleteRecursively()
            }

            // Executable bits for bin/* (cmake/cpack/ctest)
            File(versionDir, "bin").listFiles()?.forEach { it.setExecutable(true, false) }

            // The release archive ships cmake/cpack/ctest only (tar listing has
            // no ninja entry). Copy the app's arm64 ninja ELF into bin/ so
            // CMAKE_MAKE_PROGRAM and in-tree lookups resolve inside the install
            // (copy, not symlink — symlinks end in a direct execve() that
            // Android denies).
            val ninjaDest = File(versionDir, "bin/ninja")
            if (!ninjaDest.isFile || !NativeToolWrappers.isElf(ninjaDest)) {
                val ninjaSrc = NativeToolWrappers
                    .scanToolCandidates(File(ctx.filesDir, "toolchain"), "ninja")
                    .firstOrNull { !it.absolutePath.startsWith(versionDir.absolutePath) }
                if (ninjaSrc != null) {
                    runCatching {
                        if (ninjaDest.exists() ||
                            runCatching { java.nio.file.Files.isSymbolicLink(ninjaDest.toPath()) }.getOrDefault(false)
                        ) {
                            ninjaDest.delete()
                        }
                        ninjaSrc.copyTo(ninjaDest, overwrite = true)
                        ninjaDest.setExecutable(true, false)
                        onLog("[CMake] ninja: ${ninjaDest.absolutePath} -> copy of ${ninjaSrc.absolutePath}")
                    }.onFailure {
                        onLog("[CMake WARN] ninja copy failed: ${it.message}")
                    }
                } else {
                    onLog("[CMake WARN] no arm64 ninja ELF found — AGP will use the SDK wrapper if present")
                }
            }

            markerFile.writeText(CMAKE_VERSION)
        }
    }

    /** Removes the installed CMake runtime (for version replacement / reinstall). */
    fun uninstall() {
        versionDir.deleteRecursively()
        Log.i(TAG, "CMake runtime uninstalled: ${versionDir.path}")
    }

    fun formatSize(bytes: Long): String = ToolchainArchiveInstaller.formatSize(bytes)
}
