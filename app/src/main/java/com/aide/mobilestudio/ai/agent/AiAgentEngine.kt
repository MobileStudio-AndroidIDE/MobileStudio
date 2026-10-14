package com.aide.mobilestudio.ai.agent

import android.content.Context
import android.util.Log
import com.aide.mobilestudio.ai.AiSettings
import com.aide.mobilestudio.ai.api.AIProvider
import com.aide.mobilestudio.ai.api.GeminiClient
import com.aide.mobilestudio.ai.api.OpenAiClient
import com.aide.mobilestudio.ai.api.AnthropicClient
import com.aide.mobilestudio.ai.api.LocalLlamaProvider
import com.aide.mobilestudio.analysis.ErrorFingerprint
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Superseded by [com.aide.mobilestudio.ai.opencode.agent.OcEngine], which the UI
 * now uses. Kept for reference; not referenced by the shipping source set.
 */
@Deprecated("Use com.aide.mobilestudio.ai.opencode.agent.OcEngine instead")
class AiAgentEngine(private val context: Context) {
    private val settings = AiSettings(context)
    private val storage = AiConversationStorage(context)
    private var currentSessionId = storage.generateId()
    private val history = JSONArray()
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var currentJob: Job? = null

    private var currentFileContext: String? = null
    private var lastProviderId: String? = null

    var provider: AIProvider? = null
        set(value) {
            val oldId = field?.id
            val newId = value?.id
            if (oldId != newId) {
                field = value
                value?.let {
                    Log.i("AiAgentEngine", "Switched provider from $oldId to ${it.name}")
                    settings.lastModelId = it.id
                    // Preserve conversation history across model switches
                    if (oldId != null && history.length() > 0) {
                        lastProviderId = oldId
                        addSystemMessage("[Model switched to ${it.name}. Previous conversation context preserved.]")
                    }
                }
            }
        }

    private val fixAttempts = mutableMapOf<String, Int>()

    var runtimeOptions = JSONObject().apply {
        put("nGpuLayers", 99) // -ngl 99: offload all layers by default
        put("nThreads", 4)
        put("maxTokens", 8192)
    }

    private var agentMode = "build"

    fun setAgentMode(mode: String) {
        agentMode = mode
        Log.i("AiAgentEngine", "Agent mode set to: $mode")
    }

    interface Listener {
        fun onMessage(text: String, isComplete: Boolean)
        fun onThinking(status: String)
        fun onToolCall(name: String, args: JSONObject, risk: RiskLevel, onApproval: (Boolean, Boolean) -> Unit)
        fun onToolStart(name: String, args: JSONObject)
        fun onToolEnd(name: String, result: String)
        fun onPlanUpdate(rawPlan: String)
        fun onStateChange(state: AgentState)
        fun onError(message: String)
        fun onCommandOutput(commandId: String, line: String)
    }

    enum class RiskLevel { READ_ONLY, BUILD, MODIFY, DANGEROUS }
    enum class AgentState { IDLE, THINKING, GENERATING, WAITING_TOOL_APPROVAL, EXECUTING_TOOL, PROCESSING_TOOL_RESULT, COMPLETED, ERROR, CANCELLED }

    var listener: Listener? = null

    init {
        // Sub-agent delegation: runs a bounded, read-only research loop in the
        // background thread that ToolRegistry spawns for the 'task' tool.
        ToolRegistry.taskHandler = { prompt, ctx -> runSubagent(prompt, ctx) }
    }

    /** Read-only tool set exposed to sub-agents (no writes, no task-in-task recursion). */
    private val SUBAGENT_TOOLS = setOf(
        "read_file", "read_files", "list_files", "search_files", "search_codebase",
        "glob", "grep", "lsp", "web_search", "web_fetch", "get_project_context",
        "git_diff", "git_status", "view_diff", "find_resource", "analyze_dependencies"
    )

    private fun subagentToolDefinitions(): JSONArray {
        val out = JSONArray()
        val all = ToolRegistry.getToolDefinitions()
        for (i in 0 until all.length()) {
            val t = all.getJSONObject(i)
            if (t.optString("name") in SUBAGENT_TOOLS) out.put(t)
        }
        return out
    }

    private fun resolveProviderOrNull(): AIProvider? = when (settings.activeProvider) {
        "openai" -> settings.openaiApiKey?.let { OpenAiClient(it, settings.openaiBaseUrl, settings.openaiModel) }
        "anthropic" -> settings.anthropicApiKey?.let { AnthropicClient(it, settings.anthropicModel) }
        else -> settings.apiKey?.let { GeminiClient(it, settings.geminiModel) }
    }

    /**
     * Runs a bounded sub-agent conversation (max 6 rounds of tool calls) and
     * returns its final text. Runs on the calling (background) thread.
     */
    private fun runSubagent(taskPrompt: String, extraContext: String): String = runBlocking {
        val subProvider = provider ?: resolveProviderOrNull()
            ?: return@runBlocking "Sub-agent unavailable: no AI provider configured."
        val toolDefs = subagentToolDefinitions()
        val subHistory = JSONArray().apply {
            put(JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray().put(JSONObject().put("text",
                    "You are a read-only research sub-agent of MobileStudio. Complete the task using only " +
                    "the read-only tools provided, then give a thorough final report as plain text. " +
                    "Never attempt to modify files.\n\nTASK:\n$taskPrompt" +
                    if (extraContext.isNotBlank()) "\n\nCONTEXT:\n$extraContext" else ""
                )))
            })
        }
        var finalText = ""
        var errorMsg: String? = null
        for (round in 0 until 6) {
            val text = StringBuilder()
            val toolCalls = mutableListOf<AIProvider.StreamEvent.ToolCall>()
            try {
                subProvider.generateStream(subHistory, toolDefs, runtimeOptions).collect { ev ->
                    when (ev) {
                        is AIProvider.StreamEvent.Text -> text.append(ev.text)
                        is AIProvider.StreamEvent.ToolCall -> toolCalls.add(ev)
                        is AIProvider.StreamEvent.Error -> errorMsg = ev.message
                        else -> {}
                    }
                }
            } catch (e: Exception) {
                errorMsg = e.message
            }
            if (errorMsg != null) return@runBlocking "Sub-agent error: $errorMsg"
            if (text.isNotEmpty()) {
                finalText = text.toString()
                subHistory.put(JSONObject().apply {
                    put("role", "model")
                    put("parts", JSONArray().put(JSONObject().put("text", finalText)))
                })
            }
            if (toolCalls.isEmpty()) break
            // Execute read-only tool calls synchronously (we are on a plain thread)
            for (tc in toolCalls) {
                val latch = java.util.concurrent.CountDownLatch(1)
                var result = ""
                if (tc.name in SUBAGENT_TOOLS) {
                    ToolRegistry.executeTool(context, tc.name, tc.args) { r -> result = r; latch.countDown() }
                    latch.await(90, java.util.concurrent.TimeUnit.SECONDS)
                } else {
                    result = "Tool not available to sub-agent: ${tc.name}"
                    latch.countDown()
                }
                subHistory.put(JSONObject().apply {
                    put("role", "user")
                    put("parts", JSONArray().put(JSONObject().put("functionResponse", JSONObject().apply {
                        put("name", tc.name)
                        put("response", JSONObject().put("content", result.take(20_000)))
                    })))
                })
            }
        }
        finalText.ifBlank { "(sub-agent finished without a report)" }
    }

    fun loadSession(sessionId: String) {
        val conv = storage.loadConversation(sessionId) ?: return
        currentSessionId = sessionId
        history.apply {
            while (length() > 0) remove(0)
            for (i in 0 until conv.messages.length()) {
                put(conv.messages[i])
            }
        }
    }

    fun newSession() {
        currentSessionId = storage.generateId()
        history.apply { while (length() > 0) remove(0) }
        fixAttempts.clear()
        currentFileContext = null
        lastProviderId = null
    }

    fun setCurrentFileContext(filePath: String?, content: String?, selection: String? = null) {
        if (filePath == null || content == null || content.isBlank()) {
            currentFileContext = null
            return
        }
        val sb = StringBuilder()
        sb.append("Current file: ").append(filePath).append("\n\n")
        if (selection != null && selection.isNotBlank()) {
            sb.append("Selected text:\n```\n").append(selection).append("\n```\n\n")
        }
        // Mobile/local models commonly have a 4k context. A bounded file
        // excerpt leaves room for the request, instructions and response.
        val excerpt = content.take(6_000)
        sb.append("File content:\n```\n").append(excerpt).append("\n```")
        if (content.length > excerpt.length) sb.append("\n[File excerpt truncated]")
        currentFileContext = sb.toString()
    }

    fun sendMessage(text: String, imageBase64: String? = null) {
        val messageText = if (currentFileContext != null) {
            text + "\n\n---CONTEXT---\n" + currentFileContext
        } else {
            text
        }
        addUserMessage(messageText, imageBase64)
        runAgentLoop()
    }

    private fun addUserMessage(text: String, imageBase64: String? = null) {
        history.put(JSONObject().apply {
            put("role", "user")
            put("parts", JSONArray().apply {
                put(JSONObject().put("text", text))
                if (imageBase64 != null) {
                    put(JSONObject().apply {
                        put("inlineData", JSONObject().apply {
                            put("mimeType", "image/jpeg")
                            put("data", imageBase64)
                        })
                    })
                }
            })
        })
        saveSession()
    }

    private fun addAiMessage(text: String) {
        history.put(JSONObject().apply {
            put("role", "model")
            put("parts", JSONArray().put(JSONObject().put("text", text)))
        })
        saveSession()
    }

    private fun addSystemMessage(text: String) {
        history.put(JSONObject().apply {
            put("role", "user")
            put("parts", JSONArray().put(JSONObject().put("text", text)))
        })
        saveSession()
    }

    private fun addToolResultMessage(toolName: String, result: String) {
        history.put(JSONObject().apply {
            put("role", "user")
            put("parts", JSONArray().put(JSONObject().put("functionResponse", JSONObject().apply {
                put("name", toolName)
                put("response", JSONObject().put("content", result))
            })))
        })
        saveSession()
    }

    private fun saveSession() {
        storage.saveConversation(currentSessionId, "AI Conversation", history, provider?.getCapabilities())
    }

    private fun buildSystemPrompt(): String {
        val sb = StringBuilder()
        sb.append("""
            You are MobileStudio, an Android coding agent. Use tools to inspect
            before editing, keep a short <plan> for multi-step tasks, and explain
            changes concisely. Never access outside the project. Verify edits with
            diagnostics or Gradle when appropriate. Use async commands for builds.
            Build context surgically: search with glob/grep/lsp first and read only
            the files you actually need — never dump whole directories or the whole
            project into the conversation. Prefer 'task' for research sub-goals and
            ask 'question' when requirements are ambiguous. If a build fails, read
            the errors, fix the code and rebuild — repeat until it passes.
        """.trimIndent())

        // Add error context if available
        val errorCount = getGlobalErrorCount()
        if (errorCount > 0) {
            sb.append("\n\nCURRENT PROJECT STATE: $errorCount compilation error(s) detected.")
            val errorsText = getGlobalErrorsText()
            if (errorsText.isNotEmpty()) {
                sb.append("\nErrors:\n${errorsText.take(3_000)}")
            }
        } else {
            sb.append("\n\nCURRENT PROJECT STATE: No compilation errors.")
        }

        return sb.toString()
    }

    private fun runAgentLoop() {
        if (history.length() == 1) {
            val systemPrompt = buildSystemPrompt()
            val newHistory = JSONArray()
            newHistory.put(JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray().put(JSONObject().put("text", systemPrompt)))
            })
            for (i in 0 until history.length()) newHistory.put(history.get(i))
            history.apply {
                while (length() > 0) remove(0)
                for (i in 0 until newHistory.length()) put(newHistory.get(i))
            }
        }

        // Add the mode instruction only for a user turn. Re-adding it after
        // every tool result silently grew local-model prompts until they failed.
        if (history.length() <= 2) when (agentMode) {
            "plan" -> addSystemMessage("[Plan mode: read only. Inspect and return a step-by-step plan; do not edit or run commands.]")
            "explore" -> addSystemMessage("[Explore mode: read only. Quickly inspect relevant files and report findings.]")
        }

        val activeProvider = provider ?: run {
            val providerName = settings.activeProvider
            when (providerName) {
                "openai" -> {
                    val key = settings.openaiApiKey
                    if (key != null) {
                        OpenAiClient(key, settings.openaiBaseUrl, settings.openaiModel).also { provider = it }
                    } else null
                }
                "anthropic" -> {
                    val key = settings.anthropicApiKey
                    if (key != null) {
                        AnthropicClient(key, settings.anthropicModel).also { provider = it }
                    } else null
                }
                else -> {
                    val key = settings.apiKey
                    if (key != null) {
                        GeminiClient(key, settings.geminiModel).also { provider = it }
                    } else null
                }
            }
        } ?: run {
            listener?.onError("AI Provider not configured. Open AI Settings to add an API key.")
            return
        }

        currentJob?.cancel()
        currentJob = scope.launch {
            listener?.onStateChange(AgentState.THINKING)
            listener?.onThinking("Thinking...")
            var aiResponseText = ""
            val toolCalls = mutableListOf<AIProvider.StreamEvent.ToolCall>()

            try {
                activeProvider.generateStream(history, ToolRegistry.getToolDefinitions(), runtimeOptions).collect { event ->
                    when (event) {
                        is AIProvider.StreamEvent.Text -> {
                            if (aiResponseText.isEmpty()) {
                                listener?.onStateChange(AgentState.GENERATING)
                            }
                            aiResponseText += event.text
                            listener?.onMessage(aiResponseText, false)
                        }
                        is AIProvider.StreamEvent.ToolCall -> {
                            toolCalls.add(event)
                        }
                        is AIProvider.StreamEvent.Plan -> {
                            listener?.onPlanUpdate(event.rawPlan)
                        }
                        is AIProvider.StreamEvent.Error -> {
                            listener?.onError(event.message)
                        }
                        is AIProvider.StreamEvent.Reasoning -> {
                            listener?.onThinking("Thinking...")
                        }
                        is AIProvider.StreamEvent.Done -> {
                            if (aiResponseText.isNotEmpty()) {
                                addAiMessage(aiResponseText)
                                listener?.onMessage(aiResponseText, true)
                            }
                            handleToolCalls(toolCalls)
                        }
                    }
                }
            } catch (e: Exception) {
                listener?.onError(e.message ?: "Unknown error during agent loop")
            }
            listener?.onStateChange(AgentState.IDLE)
        }
    }

    private suspend fun handleToolCalls(calls: List<AIProvider.StreamEvent.ToolCall>) {
        if (calls.isEmpty()) return

        for (call in calls) {
            listener?.onThinking("Executing: ${call.name}")
            val risk = getRiskLevel(call.name, call.args)

            val beforeFingerprints = getGlobalErrorFingerprints()
            listener?.onToolStart(call.name, call.args)

            // Commands the user previously approved with "Allow Always" skip the prompt
            val commandAlwaysAllowed = call.name == "run_command" &&
                settings.isCommandAlwaysAllowed(call.args.optString("command", ""))

            val result = if (risk == RiskLevel.READ_ONLY || commandAlwaysAllowed) {
                listener?.onStateChange(AgentState.EXECUTING_TOOL)
                executeToolSync(call)
            } else if ((call.name == "write_file") && settings.autoApplyChanges) {
                executeToolSync(call)
            } else {
                listener?.onStateChange(AgentState.WAITING_TOOL_APPROVAL)
                val approved = CompletableDeferred<Pair<Boolean, Boolean>>()
                withContext(Dispatchers.Main) {
                    listener?.onToolCall(call.name, call.args, risk) { allow, always ->
                        approved.complete(allow to always)
                    }
                }
                val (allow, always) = approved.await()
                if (allow) {
                    if (always && call.name == "run_command") {
                        val command = call.args.optString("command", "")
                        settings.allowCommandAlways(command)
                    }
                    listener?.onStateChange(AgentState.EXECUTING_TOOL)
                    executeToolSync(call)
                } else {
                    "User rejected execution."
                }
            }

            listener?.onToolEnd(call.name, result)
            addToolResultMessage(call.name, result)

            if (call.name == "write_file" && result.startsWith("Successfully")) {
                handlePostWriteAnalysis(beforeFingerprints)
            }
        }
        runAgentLoop()
    }

    private suspend fun handlePostWriteAnalysis(beforeFingerprints: List<ErrorFingerprint>) {
        listener?.onThinking("Analyzing changes...")
        val activity = context as? com.aide.mobilestudio.MainActivity
        activity?.triggerLiveDiagnosticsAndWait()?.join()

        delay(1000)
        val afterErrors = getGlobalErrorCount()
        val afterFingerprints = getGlobalErrorFingerprints()

        val fixed = beforeFingerprints.filter { it !in afterFingerprints }.size
        val introduced = afterFingerprints.filter { it !in beforeFingerprints }.size

        if (introduced > 0) {
            val errorDetails = getGlobalErrorsText()
            listener?.onMessage("⚠️ This change introduced $introduced error(s).", false)
            addToolResultMessage("compiler_feedback", "The previous write introduced $introduced new errors:\n$errorDetails\nPlease fix them.")

            val fingerprintKey = afterFingerprints.joinToString(",") { "${it.file}:${it.line}:${it.normalizedMessage}" }
            val currentFixCount = fixAttempts.getOrDefault(fingerprintKey, 0)
            if (currentFixCount < 5) {
                fixAttempts[fingerprintKey] = currentFixCount + 1
                listener?.onThinking("Attempting auto-fix (${currentFixCount + 1}/5)...")
                runAgentLoop()
            } else {
                listener?.onMessage("❌ Auto-fix attempts (5) exceeded. Manual intervention required.", false)
            }
        } else if (fixed > 0) {
            listener?.onMessage("✅ This change fixed $fixed error(s).", false)
            if (afterErrors == 0) {
                listener?.onMessage("🚀 All errors resolved!", false)
            }
        } else if (afterErrors == 0) {
            listener?.onMessage("✅ No errors introduced.", false)
        }
    }

    private suspend fun executeToolSync(call: AIProvider.StreamEvent.ToolCall): String {
        val deferred = CompletableDeferred<String>()

        // Set up command output streaming for async commands
        if (call.name == "run_command_async" || call.name == "run_gradle") {
            val prevListener = ToolRegistry.commandOutputListener
            ToolRegistry.commandOutputListener = { cmdId, line ->
                listener?.onCommandOutput(cmdId, line)
            }
            ToolRegistry.executeTool(context, call.name, call.args) { result ->
                // Restore previous listener after command starts
                ToolRegistry.commandOutputListener = prevListener
                deferred.complete(result)
            }
        } else {
            ToolRegistry.executeTool(context, call.name, call.args) { result ->
                deferred.complete(result)
            }
        }

        return deferred.await()
    }

    private fun getRiskLevel(name: String, args: JSONObject): RiskLevel {
        return when (name) {
            "read_file", "list_files", "search_files", "search_codebase",
            "read_files", "get_project_context", "check_command",
            "git_status", "git_diff", "recall_memory", "todowrite", "view_diff",
            "question", "lsp", "glob", "grep", "logcat_search",
            "find_resource", "analyze_dependencies" -> RiskLevel.READ_ONLY
            "run_command" -> {
                val cmd = args.optString("command", "").lowercase()
                when {
                    cmd.contains("rm -rf") || cmd.contains("chmod 777") -> RiskLevel.DANGEROUS
                    cmd.startsWith("rm ") || cmd.startsWith("mv ") || cmd.startsWith("cp ") -> RiskLevel.MODIFY
                    else -> RiskLevel.BUILD
                }
            }
            "run_command_async", "run_gradle", "run_tests" -> RiskLevel.BUILD
            "write_file", "write_files", "save_memory", "apply_patch",
            "edit_manifest", "undo_last_edit", "task" -> RiskLevel.MODIFY
            "git_commit", "git_branch" -> RiskLevel.MODIFY
            else -> RiskLevel.DANGEROUS
        }
    }

    private fun getGlobalErrorCount(): Int {
        val activity = context as? com.aide.mobilestudio.MainActivity ?: return 0
        return activity.getTotalErrorCount()
    }

    private fun getGlobalErrorsText(): String {
        val activity = context as? com.aide.mobilestudio.MainActivity ?: return ""
        return activity.getErrorsAsText()
    }

    private fun getGlobalErrorFingerprints(): List<ErrorFingerprint> {
        val activity = context as? com.aide.mobilestudio.MainActivity ?: return emptyList()
        return activity.getErrorFingerprints()
    }

    fun stop() {
        currentJob?.cancel()
        currentJob = null
        listener?.onStateChange(AgentState.CANCELLED)
    }

    /** Get conversation token estimate (rough: 1 token ~ 4 chars) */
    fun estimateTokens(): Int {
        var totalChars = 0
        for (i in 0 until history.length()) {
            val msg = history.optJSONObject(i) ?: continue
            val parts = msg.optJSONArray("parts") ?: continue
            for (j in 0 until parts.length()) {
                val part = parts.optJSONObject(j) ?: continue
                totalChars += part.optString("text", "").length
            }
        }
        return totalChars / 4
    }
}
