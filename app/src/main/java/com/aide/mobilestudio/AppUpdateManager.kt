package com.aide.mobilestudio

import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import org.json.JSONArray
import org.json.JSONObject

object AppUpdateManager {
    private const val TAG = "AppUpdateManager"
    private const val GITHUB_REPO = "MobileStudio-AndroidIDE/MobileStudio_AndroidIDE"
    private const val GITHUB_API = "https://api.github.com/repos/$GITHUB_REPO/releases?per_page=100"

    data class UpdateInfo(
        val latestVersion: String,
        val latestTag: String,
        val apkUrl: String,
        val releaseNotes: String,
        val apkSize: Long,
        val publishedAt: String
    )

    @Volatile
    private var cachedUpdate: UpdateInfo? = null

    @Volatile
    private var isChecking = false

    fun getCachedUpdate(): UpdateInfo? = cachedUpdate

    fun clearCache() { cachedUpdate = null }

    fun checkForUpdates(
        context: Context,
        currentVersionName: String,
        onResult: (UpdateInfo?) -> Unit
    ) {
        if (isChecking) { onResult(null); return }
        isChecking = true

        Thread {
            try {
                val conn = URL(GITHUB_API).openConnection() as HttpURLConnection
                conn.connectTimeout = 10000
                conn.readTimeout = 10000
                conn.setRequestProperty("Accept", "application/vnd.github+json")
                conn.setRequestProperty("User-Agent", "MobileStudio-AndroidIDE")

                if (conn.responseCode != 200) {
                    Log.w(TAG, "releases list HTTP ${conn.responseCode}")
                    conn.disconnect()
                    isChecking = false
                    onResult(null)
                    return@Thread
                }

                val json = conn.inputStream.bufferedReader().readText()
                conn.disconnect()

                val releases = JSONArray(json)

                var bestTag: String? = null
                var bestInfo: UpdateInfo? = null

                for (i in 0 until releases.length()) {
                    val release = releases.getJSONObject(i)
                    if (release.optBoolean("draft", false)) continue
                    if (release.optBoolean("prerelease", false)) continue

                    val tagName = release.optString("tag_name", "").removePrefix("v")
                    if (tagName.isBlank()) continue

                    val assets = release.optJSONArray("assets") ?: JSONArray()
                    var apkUrl: String? = null
                    var apkSize: Long = 0
                    for (j in 0 until assets.length()) {
                        val asset = assets.getJSONObject(j)
                        val name = asset.optString("name", "")
                        if (!name.endsWith(".apk")) continue
                        val prefUrl = asset.optString("browser_download_url")
                        if (prefUrl.isBlank()) continue
                        if (name.startsWith("MobileStudio_") || apkUrl == null) {
                            apkUrl = prefUrl
                            apkSize = asset.optLong("size", 0)
                        }
                        if (name.startsWith("MobileStudio_")) break
                    }
                    if (apkUrl == null) continue

                    if (bestTag == null || isNewerVersion(bestTag, tagName)) {
                        bestTag = tagName
                        bestInfo = UpdateInfo(
                            latestVersion = tagName,
                            latestTag = tagName,
                            apkUrl = apkUrl!!,
                            releaseNotes = release.optString("body", ""),
                            apkSize = apkSize,
                            publishedAt = release.optString("published_at", "")
                        )
                    }
                }

                isChecking = false
                val info = bestInfo ?: run {
                    onResult(null)
                    return@Thread
                }

                val currentNumeric = currentVersionName.split(" ").firstOrNull() ?: ""

                if (!isNewerVersion(currentNumeric, info.latestTag)) {
                    onResult(null)
                    return@Thread
                }

                cachedUpdate = info
                onResult(info)
            } catch (e: Exception) {
                Log.w(TAG, "checkForUpdates failed: ${e.message}")
                isChecking = false
                onResult(null)
            }
        }.apply { isDaemon = true; start() }
    }

    fun isNewerVersion(current: String, latest: String): Boolean {
        val curParts = current.split(".").map { it.toIntOrNull() ?: 0 }
        val latParts = latest.split(".").map { it.toIntOrNull() ?: 0 }
        for (i in 0 until maxOf(curParts.size, latParts.size)) {
            val c = curParts.getOrElse(i) { 0 }
            val l = latParts.getOrElse(i) { 0 }
            if (l > c) return true
            if (l < c) return false
        }
        return false
    }

    fun downloadAndInstall(
        context: Context,
        info: UpdateInfo,
        onProgress: (Int) -> Unit,
        onDone: (Boolean, String?) -> Unit
    ) {
        Thread {
            try {
                val apkFile = File(context.cacheDir, "update_${info.latestTag}.apk")
                apkFile.delete()

                val conn = URL(info.apkUrl).openConnection() as HttpURLConnection
                conn.connectTimeout = 15000
                conn.readTimeout = 60000
                conn.instanceFollowRedirects = true
                conn.connect()

                if (conn.responseCode !in 200..299) {
                    conn.disconnect()
                    onDone(false, "HTTP ${conn.responseCode}")
                    return@Thread
                }

                val total = conn.contentLengthLong
                val input = conn.inputStream
                val output = FileOutputStream(apkFile)
                val buffer = ByteArray(64 * 1024)
                var downloaded = 0L

                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    output.write(buffer, 0, read)
                    downloaded += read
                    if (total > 0) {
                        onProgress((downloaded * 100 / total).toInt())
                    }
                }
                output.close()
                input.close()
                conn.disconnect()

                onProgress(100)

                Handler(Looper.getMainLooper()).post {
                    try {
                        val uri = FileProvider.getUriForFile(
                            context,
                            "${context.packageName}.provider",
                            apkFile
                        )
                        val intent = Intent(Intent.ACTION_VIEW).apply {
                            setDataAndType(uri, "application/vnd.android.package-archive")
                            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
                        }
                        context.startActivity(intent)
                        onDone(true, null)
                    } catch (e: Exception) {
                        onDone(false, e.message)
                    }
                }
            } catch (e: Exception) {
                onDone(false, e.message)
            }
        }.apply { isDaemon = true; start() }
    }
}
