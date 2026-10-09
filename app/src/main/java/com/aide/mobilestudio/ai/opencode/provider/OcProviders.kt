package com.aide.mobilestudio.ai.opencode.provider

import android.content.Context
import android.util.Log
import com.aide.mobilestudio.ai.api.AIProvider
import com.aide.mobilestudio.ai.agent.ModelManager
import com.aide.mobilestudio.ai.opencode.config.OpencodeConfig
import kotlinx.coroutines.channels.awaitClose
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.net.HttpURLConnection
import java.net.URL

private fun org.json.JSONObject.optStringOrNull(key: String): String? {
    if (!has(key) || isNull(key)) return null
    return optString(key, "").ifEmpty { null }
}

data class OcModelRef(val providerID: String, val modelID: String) {
    override fun toString(): String = "$providerID/$modelID"

    companion object {
        fun parse(ref: String): OcModelRef? {
            val idx = ref.indexOf('/')
            if (idx <= 0 || idx >= ref.length - 1) return null
            return OcModelRef(ref.substring(0, idx), ref.substring(idx + 1))
        }
    }
}

data class OcModelInfo(
    val ref: OcModelRef,
    val name: String,
    val contextLimit: Int,
    val outputLimit: Int?,
    val toolCall: Boolean,
    val reasoning: Boolean,
    val local: Boolean,
    val ggufPath: String? = null,
)

class OcProviderRegistry(
    private val context: Context,
    private val configProvider: () -> OpencodeConfig,
    private val modelManager: ModelManager,
) {

    fun resolve(ref: String?): OcModelInfo? {
        val parsed = ref?.let { OcModelRef.parse(it) }
            ?: configProvider().model?.let { OcModelRef.parse(it) }
            ?: return null
        return getModel(parsed)
    }

    fun listProviders(): List<Pair<String, String>> {
        val result = mutableListOf<Pair<String, String>>()
        result.add("gemini" to "Google Gemini (API)")
        result.add("openai" to "OpenAI (API)")
        result.add("anthropic" to "Anthropic (API)")
        result.add("local" to "Local Models (llama.cpp)")
        for ((name, provider) in configProvider().providers) {
            result.add(name to (provider.optString("name", name)))
        }
        return result
    }

    fun listModels(providerID: String): List<OcModelInfo> {
        return when (providerID) {
            "gemini" -> {
                val ids = linkedSetOf("gemini-1.5-flash", "gemini-2.0-flash")
                val configured = com.aide.mobilestudio.ai.AiSettings(context).geminiModel
                if (configured.isNotEmpty()) ids.add(configured)
                ids.map { id ->
                    OcModelInfo(
                        OcModelRef("gemini", id), "Gemini $id",
                        1_048_576, 8_192, true, false, false,
                    )
                }
            }
            "openai" -> {
                val settings = com.aide.mobilestudio.ai.AiSettings(context)
                val id = settings.openaiModel
                listOf(
                    OcModelInfo(
                        OcModelRef("openai", id), "OpenAI $id",
                        128_000, settings.openaiMaxTokens, true, false, false,
                    ),
                )
            }
            "anthropic" -> {
                val settings = com.aide.mobilestudio.ai.AiSettings(context)
                val id = settings.anthropicModel
                listOf(
                    OcModelInfo(
                        OcModelRef("anthropic", id), "Claude $id",
                        200_000, 8_192, true, false, false,
                    ),
                )
            }
            "local" -> modelManager.listModels().map { model ->
                OcModelInfo(
                    OcModelRef("local", model.id), model.name,
                    model.contextLength, null, true, false, true, model.path,
                )
            }
            else -> {
                val provider = configProvider().providers[providerID] ?: return emptyList()
                val models = provider.optJSONObject("models") ?: return emptyList()
                val out = mutableListOf<OcModelInfo>()
                val keys = models.keys()
                while (keys.hasNext()) {
                    val id = keys.next()
                    val model = models.getJSONObject(id)
                    out.add(
                        OcModelInfo(
                            OcModelRef(providerID, id),
                            model.optString("name", id),
                            model.optJSONObject("limit")?.optInt("context", 128_000) ?: 128_000,
                            model.optJSONObject("limit")?.let { if (it.has("output")) it.optInt("output") else null },
                            model.optBoolean("tool_call", true),
                            model.optBoolean("reasoning", false),
                            false,
                        ),
                    )
                }
                out
            }
        }
    }

    fun getModel(ref: OcModelRef): OcModelInfo? {
        for (model in listModels(ref.providerID)) {
            if (model.ref.modelID == ref.modelID) return model
        }
        return null
    }

    fun createProvider(ref: OcModelRef, runtimeOptions: JSONObject): AIProvider? {
        val model = getModel(ref) ?: return null
        return when (ref.providerID) {
            "gemini" -> {
                val apiKey = com.aide.mobilestudio.ai.AiSettings(context).apiKey ?: return null
                OcGeminiProvider(apiKey, ref.modelID, model)
            }
            "openai" -> {
                val settings = com.aide.mobilestudio.ai.AiSettings(context)
                val apiKey = settings.openaiApiKey
                if (apiKey.isNullOrEmpty()) return null
                OcOpenAiCompatibleProvider(settings.openaiBaseUrl, apiKey, ref.modelID, model)
            }
            "anthropic" -> {
                val settings = com.aide.mobilestudio.ai.AiSettings(context)
                val apiKey = settings.anthropicApiKey
                if (apiKey.isNullOrEmpty()) return null
                OcAnthropicProvider(apiKey, ref.modelID, model)
            }
            "local" -> {
                val gguf = model.ggufPath ?: return null
                val modelInfo = ModelManager.ModelInfo(
                    id = ref.modelID,
                    name = model.name,
                    path = gguf,
                    contextLength = model.contextLimit,
                )
                OcLocalLlamaProvider(modelInfo)
            }
            else -> {
                val provider = configProvider().providers[ref.providerID] ?: return null
                val options = provider.optJSONObject("options") ?: JSONObject()
                val baseURL = options.optString("baseURL", "")
                if (baseURL.isBlank()) return null
                val apiKey = options.optStringOrNull("apiKey")
                OcOpenAiCompatibleProvider(baseURL, apiKey, ref.modelID, model)
            }
        }
    }
}

class OcGeminiProvider(
    private val apiKey: String,
    private val modelId: String,
    private val model: OcModelInfo,
) : AIProvider {
    override val id: String = "gemini/$modelId"
    override val name: String = "Google Gemini ($modelId)"

    private val baseUrl
        get() = "https://generativelanguage.googleapis.com/v1beta/models/$modelId:streamGenerateContent?alt=sse&key=$apiKey"

    override fun generateStream(
        history: JSONArray,
        tools: JSONArray?,
        options: JSONObject?,
    ): kotlinx.coroutines.flow.Flow<AIProvider.StreamEvent> = kotlinx.coroutines.flow.flow {
        val systemText = extractSystem(history)
        val contents = JSONArray()
        for (i in 0 until history.length()) {
            val msg = history.getJSONObject(i)
            val role = msg.optString("role")
            if (role == "system") continue
            val parts = msg.optJSONArray("parts") ?: continue
            val gParts = JSONArray()
            val functionResponses = JSONArray()
            for (j in 0 until parts.length()) {
                val part = parts.getJSONObject(j)
                when (part.optString("type")) {
                    "text" -> {
                        val text = part.optString("text")
                        if (text.isNotEmpty()) gParts.put(JSONObject().put("text", text))
                    }
                    "tool" -> {
                        val name = part.optString("tool")
                        val state = part.optJSONObject("state")
                        if (role == "assistant") {
                            val input = state?.optJSONObject("input") ?: JSONObject()
                            if (name.isNotEmpty()) {
                                gParts.put(
                                    JSONObject().put(
                                        "functionCall",
                                        JSONObject().put("name", name).put("args", input),
                                    ),
                                )
                            }
                            val status = state?.optString("status", "completed") ?: "completed"
                            val output = state?.optString("output")?.takeIf { it.isNotEmpty() }
                                ?: state?.optString("error", "")
                                ?: ""
                            if (name.isNotEmpty() && state != null) {
                                functionResponses.put(
                                    JSONObject().put(
                                        "functionResponse",
                                        JSONObject().put("name", name)
                                            .put(
                                                "response",
                                                JSONObject().put("content", output)
                                                    .put("status", status),
                                            ),
                                    ),
                                )
                            }
                        } else {
                            val output = state?.optString("output", state.optString("error", "")) ?: ""
                            if (name.isNotEmpty()) {
                                functionResponses.put(
                                    JSONObject().put(
                                        "functionResponse",
                                        JSONObject().put("name", name)
                                            .put("response", JSONObject().put("content", output)),
                                    ),
                                )
                            }
                        }
                    }
                    "file" -> {
                        val data = part.optString("data")
                        if (data.isNotEmpty()) {
                            gParts.put(
                                JSONObject().put(
                                    "inlineData",
                                    JSONObject()
                                        .put("mimeType", part.optString("mime", "image/jpeg"))
                                        .put("data", data),
                                ),
                            )
                        } else {
                            gParts.put(
                                JSONObject().put(
                                    "text",
                                    "[Attached ${part.optString("mime")}: ${part.optString("filename", "file")}]",
                                ),
                            )
                        }
                    }
                    else -> {}
                }
            }
            if (gParts.length() == 0 && functionResponses.length() == 0) continue
            if (gParts.length() > 0) {
                contents.put(
                    JSONObject().put("role", if (role == "assistant") "model" else "user")
                        .put("parts", gParts),
                )
            }
            if (functionResponses.length() > 0) {
                contents.put(JSONObject().put("role", "user").put("parts", functionResponses))
            }
        }
        val body = JSONObject().apply {
            if (systemText != null) put("systemInstruction", JSONObject().put("parts", JSONArray().put(JSONObject().put("text", systemText))))
            put("contents", contents)
            if (tools != null && tools.length() > 0) {
                put("tools", JSONArray().put(JSONObject().put("function_declarations", tools)))
            }
        }

        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            var conn: HttpURLConnection? = null
            try {
                conn = (URL(baseUrl).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    setRequestProperty("Content-Type", "application/json")
                    doOutput = true
                    connect()
                }

                java.io.OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }

                var line: String? = null
                var planBuffer = StringBuilder()
                var isInPlan = false

                val reader = BufferedReader(InputStreamReader(conn.inputStream))
                while (reader.readLine()?.also { line = it } != null) {
                    val currentLine = line ?: continue
                    if (currentLine.startsWith("data: ")) {
                        val data = currentLine.substring(6)
                        if (data == "[DONE]") break
                        val json = JSONObject(data)
                        val candidates = json.optJSONArray("candidates")
                        if (candidates != null && candidates.length() > 0) {
                            val candidate = candidates.getJSONObject(0)
                            val content = candidate.optJSONObject("content")
                            val parts = content?.optJSONArray("parts")
                            if (parts != null) {
                                for (i in 0 until parts.length()) {
                                    val part = parts.getJSONObject(i)
                                    val text = part.optString("text")
                                    if (text.isNotEmpty()) {
                                        var remainingText = text
                                        if (!isInPlan && remainingText.contains("<plan>")) {
                                            val split = remainingText.split("<plan>", limit = 2)
                                            if (split[0].isNotEmpty()) emit(AIProvider.StreamEvent.Text(split[0]))
                                            isInPlan = true
                                            remainingText = split[1]
                                        }
                                        if (isInPlan) {
                                            if (remainingText.contains("</plan>")) {
                                                val split = remainingText.split("</plan>", limit = 2)
                                                planBuffer.append(split[0])
                                                emit(AIProvider.StreamEvent.Plan(planBuffer.toString()))
                                                isInPlan = false
                                                planBuffer = StringBuilder()
                                                if (split[1].isNotEmpty()) emit(AIProvider.StreamEvent.Text(split[1]))
                                            } else {
                                                planBuffer.append(remainingText)
                                            }
                                        } else {
                                            emit(AIProvider.StreamEvent.Text(remainingText))
                                        }
                                    }
                                    if (part.optBoolean("thought", false) && text.isNotEmpty()) {
                                        emit(AIProvider.StreamEvent.Reasoning(text))
                                    }
                                    val toolCall = part.optJSONObject("functionCall")
                                    if (toolCall != null) {
                                        emit(
                                            AIProvider.StreamEvent.ToolCall(
                                                toolCall.getString("name"),
                                                toolCall.optJSONObject("args") ?: JSONObject(),
                                                null,
                                            ),
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
                emit(AIProvider.StreamEvent.Done)
            } catch (e: Exception) {
                Log.e("OcGeminiProvider", "Stream error", e)
                emit(AIProvider.StreamEvent.Error(e.message ?: "Unknown error"))
            } finally {
                conn?.disconnect()
            }
        }
    }

    private fun extractSystem(history: JSONArray): String? {
        var system: String? = null
        for (i in 0 until history.length()) {
            val msg = history.getJSONObject(i)
            if (msg.optString("role") != "system") continue
            val parts = msg.optJSONArray("parts") ?: continue
            for (j in 0 until parts.length()) {
                val part = parts.getJSONObject(j)
                if (part.optString("type") == "text") {
                    system = ((system ?: "") + part.optString("text")).ifEmpty { null }
                }
            }
        }
        return system
    }

    override fun getCapabilities(): JSONObject = JSONObject().apply {
        put("streaming", true)
        put("tools", true)
    }
}

class OcOpenAiCompatibleProvider(
    private val baseURL: String,
    private val apiKey: String?,
    private val modelId: String,
    private val model: OcModelInfo,
) : AIProvider {
    override val id: String = "openai-compatible/$modelId"
    override val name: String = "OpenAI Compatible ($modelId)"

    override fun generateStream(
        history: JSONArray,
        tools: JSONArray?,
        options: JSONObject?,
    ): kotlinx.coroutines.flow.Flow<AIProvider.StreamEvent> = kotlinx.coroutines.flow.flow {
        val body = JSONObject().apply {
            put("model", modelId)
            put("messages", convertMessages(history))
            put("stream", true)
            if (tools != null && tools.length() > 0) {
                put("tools", JSONArray().apply {
                    for (i in 0 until tools.length()) {
                        val tool = tools.getJSONObject(i)
                        put(JSONObject().apply {
                            put("type", "function")
                            put("function", JSONObject().apply {
                                put("name", tool.optString("name"))
                                put("description", tool.optString("description"))
                                put("parameters", tool.optJSONObject("parameters") ?: JSONObject().put("type", "object"))
                            })
                        })
                    }
                })
            }
        }

        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            var conn: HttpURLConnection? = null
            try {
                val urlStr = baseURL.trimEnd('/') + "/chat/completions"
                conn = (URL(urlStr).openConnection() as HttpURLConnection).apply {
                    requestMethod = "POST"
                    setRequestProperty("Content-Type", "application/json")
                    if (apiKey != null) setRequestProperty("Authorization", "Bearer $apiKey")
                    doOutput = true
                    connect()
                }

                java.io.OutputStreamWriter(conn.outputStream).use { it.write(body.toString()) }

                val reader = BufferedReader(InputStreamReader(conn.inputStream))

                val toolCalls = mutableMapOf<Int, Pair<String, StringBuilder>>()
                var line: String? = null
                while (reader.readLine()?.also { line = it } != null) {
                    val currentLine = line ?: continue
                    if (!currentLine.startsWith("data: ")) continue
                    val data = currentLine.substring(6).trim()
                    if (data == "[DONE]") break
                    val json = runCatching { JSONObject(data) }.getOrNull() ?: continue
                    val choices = json.optJSONArray("choices") ?: continue
                    if (choices.length() == 0) continue
                    val delta = choices.getJSONObject(0).optJSONObject("delta") ?: continue
                    val reasoning = delta.optStringOrNull("reasoning_content")
                        ?: delta.optStringOrNull("reasoning")
                    if (!reasoning.isNullOrEmpty()) emit(AIProvider.StreamEvent.Reasoning(reasoning))
                    val content = delta.optStringOrNull("content")
                    if (!content.isNullOrEmpty()) emit(AIProvider.StreamEvent.Text(content))
                    val toolCallsArr = delta.optJSONArray("tool_calls")
                    if (toolCallsArr != null) {
                        for (i in 0 until toolCallsArr.length()) {
                            val tc = toolCallsArr.getJSONObject(i)
                            val index = tc.optInt("index", 0)
                            val id = tc.optStringOrNull("id")
                            val fn = tc.optJSONObject("function") ?: continue
                            val name = fn.optStringOrNull("name") ?: ""
                            val args = fn.optStringOrNull("arguments") ?: ""
                            val existing = toolCalls[index]
                            if (existing == null) {
                                toolCalls[index] = Pair(name, StringBuilder(args))
                                if (id != null) toolCallIds[index] = id
                            } else {
                                val mergedName = if (name.isNotEmpty()) name else existing.first
                                toolCalls[index] = Pair(mergedName, existing.second.append(args))
                            }
                        }
                    }
                }
                for ((index, tc) in toolCalls.toSortedMap()) {
                    val argsJson = runCatching { JSONObject(tc.second.toString()) }.getOrDefault(JSONObject())
                    emit(AIProvider.StreamEvent.ToolCall(tc.first, argsJson, toolCallIds[index]))
                }
                emit(AIProvider.StreamEvent.Done)
            } catch (e: Exception) {
                Log.e("OcOpenAiCompatible", "Stream error", e)
                emit(AIProvider.StreamEvent.Error(e.message ?: "Unknown error"))
            } finally {
                conn?.disconnect()
            }
        }
    }

    private val toolCallIds = mutableMapOf<Int, String>()

    private fun convertMessages(history: JSONArray): JSONArray {
        val messages = JSONArray()
        for (i in 0 until history.length()) {
            val msg = history.getJSONObject(i)
            val role = msg.optString("role")
            val parts = msg.optJSONArray("parts") ?: continue
            if (parts.length() == 0) continue
            if (role == "system") {
                val systemContent = StringBuilder()
                for (j in 0 until parts.length()) {
                    val part = parts.getJSONObject(j)
                    if (part.optString("type") == "text") {
                        if (systemContent.isNotEmpty()) systemContent.append("\n")
                        systemContent.append(part.optString("text"))
                    }
                }
                if (systemContent.isNotEmpty()) {
                    messages.put(JSONObject().put("role", "system").put("content", systemContent.toString()))
                }
                continue
            }
            if (role == "assistant") {
                val content = StringBuilder()
                val toolCallsArr = JSONArray()
                for (j in 0 until parts.length()) {
                    val part = parts.getJSONObject(j)
                    when (part.optString("type")) {
                        "text" -> if (!part.optBoolean("ignored") && part.optString("text").isNotEmpty()) {
                            if (content.isNotEmpty()) content.append("\n")
                            content.append(part.optString("text"))
                        }
                        "tool" -> {
                            val state = part.optJSONObject("state") ?: JSONObject()
                            val input = state.optJSONObject("input") ?: JSONObject()
                            toolCallsArr.put(JSONObject().apply {
                                put("id", part.optString("callId"))
                                put("type", "function")
                                put("function", JSONObject().apply {
                                    put("name", part.optString("tool"))
                                    put("arguments", input.toString())
                                })
                            })
                        }
                    }
                }
                val msgOut = JSONObject().apply {
                    put("role", "assistant")
                    if (content.isNotEmpty()) put("content", content.toString())
                    if (toolCallsArr.length() > 0) put("tool_calls", toolCallsArr)
                }
                messages.put(msgOut)
                for (j in 0 until parts.length()) {
                    val part = parts.getJSONObject(j)
                    if (part.optString("type") != "tool") continue
                    val state = part.optJSONObject("state") ?: JSONObject()
                    messages.put(JSONObject().apply {
                        put("role", "tool")
                        put("tool_call_id", part.optString("callId"))
                        put("content", state.optString("output"))
                    })
                }
            } else {
                val content = StringBuilder()
                for (j in 0 until parts.length()) {
                    val part = parts.getJSONObject(j)
                    when (part.optString("type")) {
                        "text" -> if (!part.optBoolean("ignored") && part.optString("text").isNotEmpty()) {
                            if (content.isNotEmpty()) content.append("\n")
                            content.append(part.optString("text"))
                        }
                        "file" -> content.append("[Attached ").append(part.optString("mime")).append(": ").append(part.optString("filename", "file")).append("]")
                    }
                }
                if (content.isNotEmpty()) {
                    messages.put(JSONObject().put("role", "user").put("content", content.toString()))
                }
            }
        }
        return messages
    }

    override fun getCapabilities(): JSONObject = JSONObject().apply {
        put("streaming", true)
        put("tools", model.toolCall)
    }
}

class OcLocalLlamaProvider(private val model: ModelManager.ModelInfo) : AIProvider {
    override val id: String = "local/" + model.id
    override val name: String = "Local: " + model.name

    private val llama = com.aide.mobilestudio.ai.api.LlamaAndroid()
    private var modelHandle: Long = 0L
    private var activeBridge: com.aide.mobilestudio.ai.api.LlamaRuntimeBridge? = null

    companion object {
        private val TOOL_OPEN = "\uD83E\uDD16"
        private val TOOL_CLOSE = "\uE003"
    }

    override fun generateStream(
        history: JSONArray,
        tools: JSONArray?,
        options: JSONObject?,
    ): kotlinx.coroutines.flow.Flow<AIProvider.StreamEvent> = kotlinx.coroutines.flow.callbackFlow {
        if (modelHandle == 0L) {
            val nGpu = options?.optInt("nGpuLayers", 32) ?: 32
            val nThreads = options?.optInt("nThreads", 4) ?: 4
            val backendName = options?.optString("backend", "AUTO") ?: "AUTO"
            val runtimeManager = com.aide.mobilestudio.ai.LlamaRuntimeManager.get()
            val runtimeAvailable = com.aide.mobilestudio.ai.api.LlamaRuntimeBridge.isBridgeAvailable() &&
                runtimeManager.isInstalled()
            if (!runtimeAvailable && !com.aide.mobilestudio.ai.api.LlamaAndroid.isNativeAvailable()) {
                Log.e("OcLocalLlamaProvider", "No AI runtime installed and legacy lib unavailable")
                trySend(
                    AIProvider.StreamEvent.Error(
                        "AI runtime is not installed.\nInstall it from AI Settings (llama.cpp runtime download).",
                    ),
                )
                close()
                return@callbackFlow
            }
            Log.i("OcLocalLlamaProvider", "Loading with $nGpu layers, $nThreads threads, runtime=$runtimeAvailable")
            val loaded = runCatching {
                kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
                    if (runtimeAvailable) {
                        val bridge = com.aide.mobilestudio.ai.api.LlamaRuntimeBridge()
                        if (runtimeManager.ensureNativeDeps() == null) {
                            Log.w("OcLocalLlamaProvider", "libc++_shared.so could not be staged")
                        }
                        if (!bridge.isRuntimeLoaded() &&
                            !bridge.loadRuntime(
                                runtimeManager.runtimeLibDir().absolutePath,
                                runtimeManager.appNativeLibDir(),
                            )
                        ) {
                            throw IllegalStateException(bridge.lastError() ?: "runtime load failed")
                        }
                        val vulkan = bridge.isVulkanAvailable()
                        val ctx = model.contextLength.coerceIn(512, 4096)
                        Log.i("OcLocalLlamaProvider", "Loading model at ${model.path} (ctx=$ctx, nGpu=$nGpu, threads=$nThreads, vulkan=$vulkan)")
                        val result = bridge.loadModelWithFallback(model.path, ctx, nGpu, nThreads, vulkan)
                        activeBridge = bridge
                        Log.i("OcLocalLlamaProvider", "Model loaded: handle=${result.handle}, backend=${result.backend}, gpuLayers=${result.nGpuLayersUsed}")
                        result.handle
                    } else {
                        if (!com.aide.mobilestudio.ai.api.LlamaAndroid.isNativeAvailable()) {
                            throw UnsatisfiedLinkError(
                                com.aide.mobilestudio.ai.api.LlamaAndroid.getLoadError() ?: "libllama_android.so unavailable",
                            )
                        }
                        Log.i("OcLocalLlamaProvider", "Loading model at ${model.path} (legacy, ctx=${model.contextLength}, threads=$nThreads, backend=$backendName)")
                        llama.loadModel(model.path, model.contextLength.coerceIn(512, 2048), 0, nThreads, backendName)
                    }
                }
            }
            modelHandle = loaded.getOrElse { e ->
                Log.e("OcLocalLlamaProvider", "Model load failed: ${e.message}", e)
                activeBridge = null
                trySend(
                    AIProvider.StreamEvent.Error(
                        "Failed to load model: ${e.message}\nInstall or reinstall the AI runtime from AI Settings.",
                    ),
                )
                close()
                return@callbackFlow
            }
            if (modelHandle == 0L) {
                trySend(AIProvider.StreamEvent.Error("Failed to load model at " + model.path))
                close()
                return@callbackFlow
            }
        }

        val prompt = buildPrompt(history, tools)
        val temp = options?.optDouble("temperature", 0.7)?.toFloat() ?: 0.7f

        var planBuffer = StringBuilder()
        var isInPlan = false
        var toolCallBuffer = StringBuilder()
        var isInToolCall = false

        kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) {
            val result = generate(prompt, temp) { token ->
                var current = token
                if (!isInPlan && current.contains("<plan>")) {
                    val split = current.split("<plan>", limit = 2)
                    if (split[0].isNotEmpty()) trySend(AIProvider.StreamEvent.Text(split[0]))
                    isInPlan = true
                    current = split[1]
                }
                if (isInPlan) {
                    if (current.contains("</plan>")) {
                        val split = current.split("</plan>", limit = 2)
                        planBuffer.append(split[0])
                        trySend(AIProvider.StreamEvent.Plan(planBuffer.toString()))
                        isInPlan = false
                        planBuffer = StringBuilder()
                        if (split[1].isNotEmpty()) trySend(AIProvider.StreamEvent.Text(split[1]))
                    } else {
                        planBuffer.append(current)
                    }
                    return@generate true
                }
                if (!isInToolCall && current.contains(TOOL_OPEN)) {
                    isInToolCall = true
                    current = current.substringAfter(TOOL_OPEN)
                }
                if (isInToolCall) {
                    if (current.contains(TOOL_CLOSE)) {
                        toolCallBuffer.append(current.substringBefore(TOOL_CLOSE))
                        runCatching {
                            val json = JSONObject(toolCallBuffer.toString())
                            trySend(
                                AIProvider.StreamEvent.ToolCall(
                                    json.getString("name"),
                                    json.optJSONObject("arguments") ?: JSONObject(),
                                ),
                            )
                        }
                        isInToolCall = false
                        toolCallBuffer = StringBuilder()
                    } else {
                        toolCallBuffer.append(current)
                    }
                    return@generate true
                }
                trySend(AIProvider.StreamEvent.Text(current))
                true
            }
            if (result != null && result.startsWith("Error")) {
                trySend(AIProvider.StreamEvent.Error(result))
            }
        }
        trySend(AIProvider.StreamEvent.Done)
        close()
        awaitClose { }
    }

    private fun buildPrompt(history: JSONArray, tools: JSONArray?): String {
        val sb = StringBuilder()
        sb.append("<|im_start|>system\n")
        var first = true
        for (i in 0 until history.length()) {
            val msg = history.getJSONObject(i)
            if (msg.optString("role") == "system" && first) {
                val parts = msg.optJSONArray("parts")
                if (parts != null && parts.length() > 0) {
                    sb.append(parts.getJSONObject(0).optString("text"))
                    first = false
                }
            }
        }
        if (first) sb.append("You are a professional AI Coding Agent. You have access to tools.\n")
        if (tools != null && tools.length() > 0) {
            sb.append("\nAvailable tools: ").append(tools.toString(2)).append("\n")
            sb.append("To call a tool, use: ").append(TOOL_OPEN)
                .append("{\"name\": \"...\", \"arguments\": {...}}").append(TOOL_CLOSE).append("\n")
        }
        sb.append("<|im_end|>\n")

        for (i in 0 until history.length()) {
            val msg = history.getJSONObject(i)
            val role = msg.optString("role")
            if (role == "system") continue
            val parts = msg.optJSONArray("parts") ?: continue
            if (parts.length() == 0) continue
            val content = StringBuilder()
            for (j in 0 until parts.length()) {
                val part = parts.getJSONObject(j)
                when (part.optString("type")) {
                    "text" -> if (!part.optBoolean("ignored")) {
                        if (content.isNotEmpty()) content.append("\n")
                        content.append(part.optString("text"))
                    }
                    "tool" -> {
                        val state = part.optJSONObject("state") ?: JSONObject()
                        val status = state.optString("status")
                        val input = state.optJSONObject("input") ?: JSONObject()
                        content.append(TOOL_OPEN)
                            .append("{\"name\": \"").append(part.optString("tool"))
                            .append("\", \"arguments\": ").append(input.toString()).append("}").append(TOOL_CLOSE)
                        if (status == "completed") {
                            content.append("\nTool result: ").append(state.optString("output"))
                        } else if (status == "error") {
                            content.append("\nTool error: ").append(state.optString("error"))
                        }
                    }
                }
            }
            if (content.isEmpty()) continue
            val displayRole = if (role == "assistant") "assistant" else "user"
            sb.append("<|im_start|>").append(displayRole).append("\n").append(content).append("<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }

    override fun getCapabilities(): JSONObject = JSONObject().apply {
        put("streaming", true)
        put("tools", true)
        put("local", true)
    }

    private fun generate(prompt: String, temp: Float, onToken: (String) -> Boolean): String {
        return try {
            val bridge = activeBridge
            val stops = arrayOf("</s>", "</plan>")
            if (bridge != null) {
                bridge.generate(modelHandle, prompt, temp, 0.9f, 40, 1.1f, -1, stops, onToken)
            } else {
                llama.generate(modelHandle, prompt, temp, 0.9f, 40, 1.1f, -1, stops, onToken)
            }
        } catch (e: Throwable) {
            Log.e("OcLocalLlamaProvider", "AI generation failed: ${e.message}", e)
            "Error: AI generation failed: ${e.message}"
        }
    }

    fun unload() {
        if (modelHandle != 0L) {
            val bridge = activeBridge
            try {
                if (bridge != null) bridge.unload() else llama.unloadModel(modelHandle)
            } catch (e: Throwable) {
                Log.e("OcLocalLlamaProvider", "unload failed: ${e.message}")
            }
            modelHandle = 0L
            activeBridge = null
        }
    }
}
