package com.aide.mobilestudio.ai.agent

import android.content.Context
import android.os.StatFs
import android.util.Log
import com.aide.mobilestudio.ai.GgufInfo
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.util.*

class ModelManager(private val context: Context) {
    data class ModelInfo(
        val id: String,
        val name: String,
        val path: String,
        val sizeBytes: Long = 0,
        val downloadUrl: String? = null,
        val quantization: String = "Q4_K_M",
        val contextLength: Int = 4096,
        val contextLimit: Int? = null,
        /**
         * The model's own maximum context, read from the GGUF header.
         *
         * Null when the file does not declare one, or has not been downloaded:
         * never a guess. [contextLength] is the window the app loads, this is
         * what the model can actually hold, and conflating the two is how a
         * model got loaded at a size it was never trained for.
         */
        val maxContext: Int? = null,
    )

    /**
     * GGUF header reads are cached for the life of the manager: listModels()
     * runs on every request and the header does not change underneath us.
     */
    private val ggufCache = java.util.concurrent.ConcurrentHashMap<String, GgufInfo.Info?>()

    fun ggufInfo(path: String): GgufInfo.Info? {
        if (ggufCache.containsKey(path)) return ggufCache[path]
        val info = runCatching { GgufInfo.read(File(path)) }.getOrNull()
        ggufCache[path] = info
        return info
    }

    private val prefs = context.getSharedPreferences("ai_models", Context.MODE_PRIVATE)
    private val modelDir = File(context.filesDir, "AI/models").apply { mkdirs() }

    fun listRemoteModels(): List<ModelInfo> {
        return listOf(
            ModelInfo("qwen-1b", "1B (Qwen2.5-Coder 1.5B)", File(modelDir, "qwen2.5-coder-1.5b-q4.gguf").absolutePath, 1_100_000_000L, 
                "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
            ModelInfo("qwen-3b", "3B (Qwen2.5-Coder 3B)", File(modelDir, "qwen2.5-coder-3b-q4.gguf").absolutePath, 2_100_000_000L,
                "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"),
            ModelInfo("ds-7b", "5B (DeepSeek-Coder 6.7B)", File(modelDir, "deepseek-coder-6.7b-q4.gguf").absolutePath, 4_300_000_000L,
                "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"),
            ModelInfo("qwen-7b", "7B (Qwen2.5-Coder 7B)", File(modelDir, "qwen2.5-coder-7b-q4.gguf").absolutePath, 4_700_000_000L,
                "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
            ModelInfo("llama-8b", "10B (Llama-3.1 8B)", File(modelDir, "llama-3.1-8b-q4.gguf").absolutePath, 5_300_000_000L,
                "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
        )
    }

    fun isModelDownloaded(model: ModelInfo): Boolean {
        val file = File(model.path)
        return file.exists() && file.length() > (model.sizeBytes * 0.9)
    }

    fun checkStorageSpace(requiredSize: Long): Pair<Boolean, Long> {
        val stat = StatFs(context.filesDir.path)
        val availableBytes = stat.availableBlocksLong * stat.blockSizeLong
        val bufferBytes = 3L * 1024 * 1024 * 1024 // 3GB
        val success = availableBytes > (requiredSize + bufferBytes)
        return success to availableBytes
    }

    fun downloadModel(model: ModelInfo, onProgress: (Float) -> Unit, onDone: (Boolean, String?) -> Unit) {
        val (hasSpace, available) = checkStorageSpace(model.sizeBytes)
        if (!hasSpace) {
            val required = (model.sizeBytes + 3L * 1024 * 1024 * 1024) / (1024 * 1024 * 1024)
            val current = available / (1024 * 1024 * 1024)
            onDone(false, "Insufficient storage. Need at least ${required}GB free (including 3GB buffer). Current available: ${current}GB.")
            return
        }

        Thread {
            try {
                val url = URL(model.downloadUrl)
                val conn = url.openConnection() as HttpURLConnection
                conn.connect()
                
                val length = conn.contentLengthLong
                val input = conn.inputStream
                val file = File(model.path)
                file.parentFile?.mkdirs()
                
                val output = FileOutputStream(file)
                val buffer = ByteArray(64 * 1024)
                var downloaded = 0L
                
                while (true) {
                    val read = input.read(buffer)
                    if (read <= 0) break
                    output.write(buffer, 0, read)
                    downloaded += read
                    onProgress(downloaded.toFloat() / length)
                }
                
                output.close()
                input.close()
                onDone(true, null)
            } catch (e: Exception) {
                Log.e("ModelManager", "Download failed", e)
                onDone(false, e.message)
            }
        }.start()
    }

    fun listModels(): List<ModelInfo> {
        val json = prefs.getString("models", "[]") ?: "[]"
        val array = JSONArray(json)
        val list = mutableListOf<ModelInfo>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            list.add(ModelInfo(
                obj.getString("id"),
                obj.getString("name"),
                obj.getString("path"),
                obj.optLong("sizeBytes"),
                obj.optString("downloadUrl"),
                obj.optString("quantization", "Q4_K_M"),
                obj.optInt("contextLength", 4096),
                ggufInfo(obj.getString("path"))?.contextLength,
            ))
        }
        val remote = listRemoteModels()
        remote.forEach { r ->
            if (isModelDownloaded(r) && list.none { it.id == r.id }) {
                list.add(r.copy(maxContext = ggufInfo(r.path)?.contextLength))
            }
        }
        return list
    }

    fun addModel(name: String, path: String) {
        val id = File(path).nameWithoutExtension
        val models = listModels().toMutableList()
        if (models.any { it.id == id }) return
        
        models.add(ModelInfo(id, name, path, maxContext = ggufInfo(path)?.contextLength))
        saveModels(models)
    }

    fun removeModel(id: String) {
        val models = listModels().filter { it.id != id }
        saveModels(models)
    }

    private fun saveModels(list: List<ModelInfo>) {
        val array = JSONArray()
        list.forEach {
            array.put(JSONObject().apply {
                put("id", it.id)
                put("name", it.name)
                put("path", it.path)
                put("sizeBytes", it.sizeBytes)
                put("downloadUrl", it.downloadUrl)
                put("quantization", it.quantization)
                put("contextLength", it.contextLength)
            })
        }
        prefs.edit().putString("models", array.toString()).apply()
    }
}
