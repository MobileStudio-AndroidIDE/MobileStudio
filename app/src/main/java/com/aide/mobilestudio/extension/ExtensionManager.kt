package com.aide.mobilestudio.extension

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * MobileStudio Extension Manager.
 *
 * Extensions are .msext (ZIP) packages distributed via GitHub Releases.
 * The registry is the GitHub Releases list of the official
 * MobileStudio-AndroidIDE/MobileStudio-Extensions repository (Releases API
 * with ETag conditional requests). Downloads are always SHA-256 verified
 * before installation. No auto-update: the user picks.
 */
class ExtensionManager(private val context: Context) {

    companion object {
        private const val TAG = "ExtensionManager"
        /**
         * Registry source of truth: GitHub Releases of the official extensions
         * repository (one Release = one extension version, only .msext assets
         * count). Responses are ETag-cached, so an unchanged registry costs a
         * single 304 Not Modified request.
         */
        private const val RELEASES_API =
            "https://api.github.com/repos/MobileStudio-AndroidIDE/MobileStudio-Extensions/releases?per_page=100"
        private const val PREFS = "extensions"
        private const val KEY_INSTALLED = "installed"
        private const val KEY_DISABLED = "disabled"
        private const val KEY_CLOSED_BANNERS = "closed_banners"
        private const val KEY_REPO_ETAG = "repo_etag"
        private const val MANIFEST_NAME = "extension.json"

        /** Release tag: `<full-extension-id>-v<version>` (matches the publish flow). */
        private val TAG_RE = Regex("^(.+)-v(\\d+\\.\\d+\\.\\d+(?:[-+][0-9A-Za-z.-]+)?)$")
        private val ID_RE = Regex("^[a-z0-9][a-z0-9._-]{2,63}$")
        private val VERSION_RE = Regex("^\\d+\\.\\d+\\.\\d+(?:[-+][0-9A-Za-z.-]+)?$")

        /** Files treated as text even without an extension (`file(1)`-style). */
        private val HEURISTIC_TEXT_NAMES = setOf(
            "license", "licence", "readme", "changelog", "notice", "authors",
            "contributors", "install", "copying", "makefile", "dockerfile",
            "gradlew", "cmakelists.txt", "gradle.properties", ".gitignore",
            ".gitattributes", ".editorconfig", "proguard-rules.pro", "todo"
        )
        private val HEURISTIC_TEXT_PREFIXES = listOf("license", "readme", "notice", "authors", "changelog")

        @Volatile private var instance: ExtensionManager? = null
        fun get(context: Context): ExtensionManager =
            instance ?: synchronized(this) { instance ?: ExtensionManager(context.applicationContext).also { instance = it } }
    }

    // ------------------------- installed state -------------------------

    data class InstalledExtension(
        val manifest: ExtensionManifest,
        val dir: File,
        val enabled: Boolean
    )

    fun installed(): List<InstalledExtension> {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_INSTALLED, null) ?: return emptyList()
        val disabled = prefs.getStringSet(KEY_DISABLED, emptySet()) ?: emptySet()
        val out = mutableListOf<InstalledExtension>()
        try {
            val arr = JSONArray(raw)
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                val dir = File(obj.optString("dir"))
                val manifest = readManifest(dir) ?: continue
                out.add(InstalledExtension(manifest, dir, obj.optString("id") !in disabled))
            }
        } catch (e: Exception) {
            Log.w(TAG, "installed() parse failed", e)
        }
        return out
    }

    fun isEnabled(id: String): Boolean = installed().firstOrNull { it.manifest.id == id }?.enabled ?: false
    fun isInstalled(id: String): Boolean = installed().any { it.manifest.id == id }

    fun findInstalledForLanguage(languageId: String): InstalledExtension? =
        installed().firstOrNull { it.enabled && it.manifest.languages.any { l -> l.equals(languageId, ignoreCase = true) } }

    fun findInstalledForFile(file: File, contentSample: String? = null): InstalledExtension? {
        val name = file.name.lowercase()
        val ext = name.substringAfterLast('.', "")
        for (e in installed()) {
            if (!e.enabled) continue
            val langs = e.manifest.languages
            val matchesExt = langs.any { l ->
                when (l.lowercase()) {
                    "shell", "bash", "sh" -> ext == "sh" || ext == "bash" || ext == "zsh"
                    "javascript" -> ext == "js" || ext == "mjs"
                    "typescript" -> ext == "ts" || ext == "tsx"
                    "markdown" -> ext == "md" || ext == "markdown"
                    "kotlin" -> ext == "kt" || ext == "kts"
                    "cpp", "c++" -> ext == "cpp" || ext == "cc" || ext == "c" || ext == "h" || ext == "hpp"
                    else -> ext == l.lowercase()
                }
            }
            if (matchesExt) return e
        }
        // Heuristic: extension-less files whose content is text (shebang / known names)
        if (ext.isEmpty() || ext == file.name) {
            val shebang = contentSample?.lineSequence()?.firstOrNull()?.trim()
            if (shebang != null && shebang.startsWith("#!")) {
                return installed().firstOrNull { e ->
                    e.enabled && e.manifest.languages.any { l -> l.lowercase() in setOf("shell", "bash", "sh") }
                }
            }
            val base = name.substringBeforeLast('.')
            if (base in HEURISTIC_TEXT_NAMES || HEURISTIC_TEXT_PREFIXES.any { base.startsWith(it) }) {
                return null // text-based but no extension claims it; core opens as text
            }
        }
        return null
    }

    // ------------------------- enable / disable / uninstall -------------------------

    fun setEnabled(id: String, enabled: Boolean): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val disabled = prefs.getStringSet(KEY_DISABLED, emptySet())?.toMutableSet() ?: mutableSetOf()
        val raw = prefs.getString(KEY_INSTALLED, null) ?: return false
        val arr = JSONArray(raw)
        val newArr = JSONArray()
        var found = false
        for (i in 0 until arr.length()) {
            val obj = arr.getJSONObject(i)
            if (obj.optString("id") == id) {
                found = true
                if (enabled) disabled.remove(id) else disabled.add(id)
            }
            newArr.put(obj)
        }
        if (!found) return false
        prefs.edit().putString(KEY_INSTALLED, newArr.toString())
            .putStringSet(KEY_DISABLED, disabled).apply()
        return true
    }

    fun uninstall(id: String): Boolean {
        val entry = installed().firstOrNull { it.manifest.id == id } ?: return false
        // Give the extension a chance to clean up (isolated)
        try { ExtensionHost.dispatchUninstall(id) } catch (_: Exception) {}
        val ok = entry.dir.deleteRecursively()
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val raw = prefs.getString(KEY_INSTALLED, null) ?: return ok
        val arr = JSONArray(raw)
        val newArr = JSONArray()
        for (i in 0 until arr.length()) {
            if (arr.getJSONObject(i).optString("id") != id) newArr.put(arr.getJSONObject(i))
        }
        val disabled = prefs.getStringSet(KEY_DISABLED, emptySet())?.toMutableSet() ?: mutableSetOf()
        disabled.remove(id)
        prefs.edit().putString(KEY_INSTALLED, newArr.toString())
            .putStringSet(KEY_DISABLED, disabled).apply()
        return ok
    }

    // ------------------------- banner state -------------------------

    /** Marks a banner closed so it is not shown again for the same file while open. */
    fun markBannerClosed(extensionId: String, filePath: String) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val closed = prefs.getStringSet(KEY_CLOSED_BANNERS, emptySet())?.toMutableSet() ?: mutableSetOf()
        closed.add("$extensionId::$filePath")
        prefs.edit().putStringSet(KEY_CLOSED_BANNERS, closed).apply()
    }

    fun isBannerClosed(extensionId: String, filePath: String): Boolean {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val closed = prefs.getStringSet(KEY_CLOSED_BANNERS, emptySet()) ?: emptySet()
        return "$extensionId::$filePath" in closed
    }

    // ------------------------- repository (Phase 3) -------------------------

    data class RepoEntry(
        val id: String,
        val name: String,
        val version: String,
        val description: String,
        val author: String,
        val download: String,
        val size: Long,
        val sha256: String,
        val minStudioVersion: String,
        val permissions: List<String>,
        val languages: List<String>
    ) {
        fun versionCode(): List<Int> = version.split(".").map { it.toIntOrNull() ?: 0 }
    }

    private val repoCache = AtomicReference<List<RepoEntry>>(emptyList())
    private val fetching = AtomicBoolean(false)

    fun cachedRepo(): List<RepoEntry> = repoCache.get()

    /**
     * Fetches the Releases registry (cached; ETag conditional request; on
     * network failure the last known list is served instead of going blank).
     */
    fun refreshRepo(force: Boolean = false, onDone: (List<RepoEntry>?, String?) -> Unit) {
        if (!force && repoCache.get().isNotEmpty()) { onDone(repoCache.get(), null); return }
        if (!fetching.compareAndSet(false, true)) { onDone(null, "이미 가져오는 중"); return }
        Thread {
            var result: List<RepoEntry>? = null
            var error: String? = null
            var conn: HttpURLConnection? = null
            try {
                val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                // Only send If-None-Match when the in-memory cache can answer a 304
                val etag = if (!force && repoCache.get().isNotEmpty())
                    prefs.getString(KEY_REPO_ETAG, null) else null
                conn = URL(RELEASES_API).openConnection() as HttpURLConnection
                conn.connectTimeout = 15_000
                conn.readTimeout = 30_000
                conn.setRequestProperty("Accept", "application/vnd.github+json")
                conn.setRequestProperty("X-GitHub-Api-Version", "2022-11-28")
                conn.setRequestProperty("User-Agent", "MobileStudio")
                if (etag != null) conn.setRequestProperty("If-None-Match", etag)
                when (val code = conn.responseCode) {
                    200 -> {
                        val text = conn.inputStream.bufferedReader().readText()
                        val parsed = parseReleases(text)
                        repoCache.set(parsed)
                        result = parsed
                        val newEtag = conn.getHeaderField("ETag")
                        prefs.edit().apply {
                            if (newEtag != null) putString(KEY_REPO_ETAG, newEtag)
                            else remove(KEY_REPO_ETAG)
                        }.apply()
                    }
                    304 -> result = repoCache.get()   // not modified: cache is current
                    else -> error = "HTTP $code"
                }
            } catch (e: Exception) {
                error = e.message ?: "네트워크 오류"
                Log.w(TAG, "refreshRepo failed", e)
            } finally {
                conn?.disconnect()
            }
            if (result == null) {
                // Network failure: serve the last known list instead of going blank
                val cached = repoCache.get()
                if (cached.isNotEmpty()) {
                    result = cached
                    error = null
                }
            }
            fetching.set(false)
            android.os.Handler(context.mainLooper).post { onDone(result, error) }
        }.start()
    }

    /**
     * Builds registry entries from the GitHub Releases payload.
     *
     * Rules (kept in sync with the server registry):
     *  - draft/prerelease Releases are not listed
     *  - only .msext assets are extension packages
     *  - the Release body carries extension.json as JSON; when the body is
     *    plain text the tag `<id>-v<version>` is the fallback
     *  - duplicate ids collapse to the highest SemVer
     */
    private fun parseReleases(text: String): List<RepoEntry> {
        val best = LinkedHashMap<String, RepoEntry>()
        val arr = JSONArray(text)
        for (i in 0 until arr.length()) {
            val rel = arr.optJSONObject(i) ?: continue
            if (rel.optBoolean("draft") || rel.optBoolean("prerelease")) continue
            val tag = rel.optString("tag_name")
            val assets = rel.optJSONArray("assets") ?: continue
            var assetUrl = ""
            var assetSize = 0L
            for (j in 0 until assets.length()) {
                val asset = assets.optJSONObject(j) ?: continue
                if (asset.optString("name").endsWith(".msext", ignoreCase = true)) {
                    assetUrl = asset.optString("browser_download_url")
                    assetSize = asset.optLong("size")
                    break
                }
            }
            if (assetUrl.isBlank()) continue

            val meta = try { JSONObject(rel.optString("body")) } catch (_: Exception) { null }
            var id = meta?.optString("id") ?: ""
            var version = meta?.optString("version") ?: ""
            val fromTag = TAG_RE.matchEntire(tag)
            if (!ID_RE.matches(id) || !VERSION_RE.matches(version)) {
                id = fromTag?.groupValues?.get(1) ?: ""
                version = fromTag?.groupValues?.get(2) ?: ""
            }
            if (!ID_RE.matches(id) || !VERSION_RE.matches(version)) continue

            val entry = RepoEntry(
                id = id,
                name = (meta?.optString("name") ?: "").ifEmpty { tag },
                version = version,
                description = meta?.optString("description") ?: "",
                author = (meta?.optString("author") ?: "").ifEmpty {
                    rel.optJSONObject("author")?.optString("login") ?: ""
                },
                download = assetUrl,
                size = if (assetSize > 0) assetSize else meta?.optLong("size") ?: 0L,
                sha256 = meta?.optString("sha256") ?: "",
                minStudioVersion = (meta?.optString("minStudioVersion") ?: "").ifEmpty {
                    (meta?.optString("minMobileStudioVersion") ?: "").ifEmpty { "1.0.0" }
                },
                permissions = stringList(meta?.optJSONArray("permissions")),
                languages = stringList(meta?.optJSONArray("languages"))
            )
            val prev = best[entry.id]
            if (prev == null || isNewerVersion(entry.version, prev.version)) best[entry.id] = entry
        }
        return best.values.toList()
    }

    private fun stringList(arr: JSONArray?): List<String> {
        val out = mutableListOf<String>()
        if (arr != null) for (j in 0 until arr.length()) out.add(arr.optString(j))
        return out
    }

    private fun isNewerVersion(candidate: String, current: String): Boolean {
        val a = candidate.split('.', '-', '+').map { it.toIntOrNull() ?: 0 }
        val b = current.split('.', '-', '+').map { it.toIntOrNull() ?: 0 }
        for (i in 0 until maxOf(a.size, b.size)) {
            val x = a.getOrElse(i) { 0 }
            val y = b.getOrElse(i) { 0 }
            if (x != y) return x > y
        }
        return false
    }

    /** Updates available: installed version < repo version for the same id. */
    fun updates(): List<Pair<InstalledExtension, RepoEntry>> {
        val repo = repoCache.get()
        if (repo.isEmpty()) return emptyList()
        val out = mutableListOf<Pair<InstalledExtension, RepoEntry>>()
        for (inst in installed()) {
            val entry = repo.firstOrNull { it.id == inst.manifest.id } ?: continue
            val newV = entry.versionCode()
            val curV = inst.manifest.versionCode()
            val newer = newV.indices.any { i -> (newV[i] > (curV.getOrElse(i) { 0 })) }
            if (newer) out.add(inst to entry)
        }
        return out
    }

    // ------------------------- install (Phase 2) -------------------------

    fun install(entry: RepoEntry, onProgress: (String) -> Unit, onDone: (Boolean, String) -> Unit) {
        Thread {
            try {
                onProgress("다운로드 중: ${entry.name}")
                val tmp = File(context.cacheDir, "ext-${entry.id}-${System.currentTimeMillis()}.msext")
                downloadFile(entry.download, tmp, onProgress)

                if (entry.sha256.isNotBlank()) {
                    onProgress("SHA-256 검증 중...")
                    val hash = sha256Of(tmp)
                    if (!hash.equals(entry.sha256, ignoreCase = true)) {
                        tmp.delete()
                        onDone(false, "SHA-256 불일치 (기대 ${entry.sha256.take(12)}…, 실제 ${hash.take(12)}…)")
                        return@Thread
                    }
                }

                onProgress("설치 중: ${entry.name}")
                val extDir = File(File(context.filesDir, "extensions"), entry.id)
                if (extDir.exists()) extDir.deleteRecursively()
                extDir.mkdirs()
                extractMsext(tmp, extDir)
                tmp.delete()

                val manifest = readManifest(extDir)
                    ?: run { extDir.deleteRecursively(); onDone(false, "extension.json 없음 또는 손상"); return@Thread }

                if (manifest.extensionApiVersion > ExtensionApi.API_VERSION) {
                    extDir.deleteRecursively()
                    onDone(false, "Extension API ${manifest.extensionApiVersion} 필요 (이 앱은 API ${ExtensionApi.API_VERSION})")
                    return@Thread
                }

                val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                val raw = prefs.getString(KEY_INSTALLED, null) ?: "[]"
                val arr = JSONArray(raw)
                val newArr = JSONArray()
                for (i in 0 until arr.length()) {
                    if (arr.getJSONObject(i).optString("id") != entry.id) newArr.put(arr.getJSONObject(i))
                }
                val obj = JSONObject()
                    .put("id", manifest.id)
                    .put("dir", extDir.absolutePath)
                    .put("version", manifest.version)
                newArr.put(obj)
                val disabled = prefs.getStringSet(KEY_DISABLED, emptySet())?.toMutableSet() ?: mutableSetOf()
                disabled.remove(manifest.id)
                prefs.edit().putString(KEY_INSTALLED, newArr.toString())
                    .putStringSet(KEY_DISABLED, disabled).apply()

                onProgress("설치 완료: ${manifest.name} ${manifest.version}")
                onDone(true, "${manifest.name} ${manifest.version} 설치됨")
            } catch (e: Exception) {
                Log.e(TAG, "install failed", e)
                onDone(false, "설치 실패: ${e.message}")
            }
        }.start()
    }

    private fun downloadFile(urlStr: String, target: File, onProgress: (String) -> Unit) {
        var connection = URL(urlStr).openConnection() as HttpURLConnection
        connection.connectTimeout = 20_000
        connection.readTimeout = 120_000
        var redirects = 0
        while (redirects < 5) {
            val code = connection.responseCode
            if (code in 301..308) {
                val location = connection.getHeaderField("Location") ?: break
                connection.disconnect()
                connection = URL(location).openConnection() as HttpURLConnection
                connection.connectTimeout = 20_000
                connection.readTimeout = 120_000
                redirects++
            } else break
        }
        if (connection.responseCode != 200) throw Exception("HTTP ${connection.responseCode}: $urlStr")
        val total = connection.contentLengthLong
        connection.inputStream.use { input ->
            FileOutputStream(target).use { out ->
                val buf = ByteArray(64 * 1024)
                var read: Int
                var totalRead = 0L
                while (input.read(buf).also { read = it } != -1) {
                    out.write(buf, 0, read)
                    totalRead += read
                    if (total > 0) onProgress("다운로드 중: ${totalRead * 100 / total}%")
                }
            }
        }
        connection.disconnect()
    }

    fun extractMsext(archive: File, dest: File) {
        ZipInputStream(archive.inputStream().buffered()).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                val target = File(dest, entry.name)
                if (entry.isDirectory) {
                    target.mkdirs()
                } else {
                    if (!target.canonicalPath.startsWith(dest.canonicalPath)) {
                        throw SecurityException("Zip path escape: ${entry.name}")
                    }
                    target.parentFile?.mkdirs()
                    FileOutputStream(target).use { out -> zip.copyTo(out) }
                }
                entry = zip.nextEntry
            }
        }
    }

    /**
     * Packs a project folder (extension.json + sources/) into a .msext package
     * (ZIP) plus a companion .msextinf metadata file. Used by the in-app
     * "Build .msext" action on extension projects.
     */
    fun packMsext(projectDir: File, outputFile: File, onProgress: (String) -> Unit): Pair<File, File>? {
        val manifestFile = File(projectDir, "extension.json")
        if (!manifestFile.isFile) throw Exception("extension.json 없음 — 확장 프로젝트 형식이 아닙니다")
        val manifest = readManifest(projectDir) ?: throw Exception("extension.json 손상")
        onProgress("패키징 중: ${manifest.name}")

        // Copy extension.json + all non-build files into the package
        val packageDir = File(projectDir, "build/.msext-work")
        if (packageDir.exists()) packageDir.deleteRecursively()
        packageDir.mkdirs()
        manifestFile.copyTo(File(packageDir, "extension.json"), overwrite = true)
        listOf("resources", "icons", "templates", "tools").forEach { sub ->
            val src = File(projectDir, sub)
            if (src.isDirectory) src.copyRecursively(File(packageDir, sub), overwrite = true)
        }
        // Include compiled dex if present
        val dex = File(projectDir, "build/classes.dex")
        if (dex.isFile) dex.copyTo(File(packageDir, "classes.dex"), overwrite = true)

        // ZIP it
        // Package name follows the Release asset convention: <full-extension-id>-v<version>.msext
        val msextFile = File(projectDir, "${manifest.id}-v${manifest.version}.msext")
        ZipOutputStream(FileOutputStream(msextFile).buffered()).use { zip ->
            packageDir.walkTopDown().filter { it.isFile }.forEach { f ->
                val name = packageDir.toPath().relativize(f.toPath()).toString().replace('\\', '/')
                val entry = java.util.zip.ZipEntry(name)
                zip.putNextEntry(entry)
                f.inputStream().use { it.copyTo(zip) }
                zip.closeEntry()
            }
        }
        packageDir.deleteRecursively()

        // .msextinf companion (text-based metadata: name + description ≤100 chars)
        val infFile = File(msextFile.parentFile, msextFile.name + "inf")
        val desc = if (manifest.description.length > 100) manifest.description.take(100) else manifest.description
        infFile.writeText(
            "# MobileStudio Extension Info (.msextinf)\n" +
                "id=${manifest.id}\n" +
                "name=${manifest.name}\n" +
                "version=${manifest.version}\n" +
                "author=${manifest.author}\n" +
                "description=$desc\n"
        )
        onProgress("패키징 완료: ${msextFile.name}")
        return msextFile to infFile
    }

    fun readManifest(dir: File): ExtensionManifest? {
        val f = File(dir, MANIFEST_NAME)
        if (!f.isFile) return null
        return try {
            val o = JSONObject(f.readText())
            val perms = mutableListOf<String>()
            o.optJSONArray("permissions")?.let { p -> for (j in 0 until p.length()) perms.add(p.getString(j)) }
            val langs = mutableListOf<String>()
            o.optJSONArray("languages")?.let { l -> for (j in 0 until l.length()) langs.add(l.getString(j)) }
            ExtensionManifest(
                id = o.getString("id"),
                name = o.getString("name"),
                version = o.getString("version"),
                author = o.optString("author"),
                description = o.optString("description"),
                minStudioVersion = o.optString("minStudioVersion", o.optString("minMobileStudioVersion", "1.0.0")),
                maxStudioVersion = o.optString("maxStudioVersion", "*"),
                entryPoint = o.optString("entryPoint"),
                permissions = perms,
                languages = langs,
                extensionApiVersion = o.optInt("extensionApiVersion", ExtensionApi.API_VERSION)
            )
        } catch (e: Exception) {
            Log.w(TAG, "manifest parse failed: ${dir.name}", e)
            null
        }
    }

    private fun sha256Of(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buf = ByteArray(64 * 1024)
            var read: Int
            while (input.read(buf).also { read = it } != -1) digest.update(buf, 0, read)
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
