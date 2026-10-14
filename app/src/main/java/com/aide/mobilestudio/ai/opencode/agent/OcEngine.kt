package com.aide.mobilestudio.ai.opencode.agent

import android.content.Context
import com.aide.mobilestudio.ai.AiSettings
import com.aide.mobilestudio.ai.agent.CheckpointFileDelta
import com.aide.mobilestudio.ai.agent.CheckpointOutcome
import com.aide.mobilestudio.ai.agent.CheckpointStore
import com.aide.mobilestudio.ai.agent.FileMentions
import com.aide.mobilestudio.ai.agent.ModelManager
import com.aide.mobilestudio.ai.api.AIProvider
import com.aide.mobilestudio.ai.opencode.command.OcCommandInfo
import com.aide.mobilestudio.ai.opencode.command.OcCommandRegistry
import com.aide.mobilestudio.ai.opencode.config.OpencodeConfig
import com.aide.mobilestudio.ai.opencode.mcp.McpToolAdapter
import com.aide.mobilestudio.ai.opencode.mcp.OcMcpClient
import com.aide.mobilestudio.ai.opencode.permission.Permission
import com.aide.mobilestudio.ai.opencode.permission.PermissionRule
import com.aide.mobilestudio.ai.opencode.provider.OcModelInfo
import com.aide.mobilestudio.ai.opencode.provider.OcModelRef
import com.aide.mobilestudio.ai.opencode.provider.OcProviderRegistry
import com.aide.mobilestudio.ai.opencode.session.OcCompaction
import com.aide.mobilestudio.ai.opencode.session.OcInstructions
import com.aide.mobilestudio.ai.opencode.session.OcMessage
import com.aide.mobilestudio.ai.opencode.session.OcSessionStore
import com.aide.mobilestudio.ai.opencode.session.OcSnapshot
import com.aide.mobilestudio.ai.opencode.skill.OcSkill
import com.aide.mobilestudio.ai.opencode.tool.*
import com.aide.mobilestudio.ai.validation.DiagnosticDiffEngine
import com.aide.mobilestudio.ai.validation.LiveDiagnostic
import com.aide.mobilestudio.ai.validation.LiveDiagnostics
import com.aide.mobilestudio.ai.resource.AndroidMemoryProbe
import com.aide.mobilestudio.ai.resource.TaskScheduler
import kotlinx.coroutines.*
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class OcEngine(private val context: Context) : EngineBridge {

    interface Listener {
        fun onMessage(text: String, isComplete: Boolean)
        fun onThinking(status: String)
        /** Live reasoning text of the current turn (accumulated so far). */
        fun onReasoning(text: String) {}
        /** Reasoning of the current turn ended (response text or a tool started). */
        fun onReasoningDone() {}
        fun onToolCall(name: String, args: JSONObject, risk: RiskLevel, onApproval: (Boolean, Boolean) -> Unit)
        fun onToolStart(name: String, args: JSONObject)
        fun onToolEnd(name: String, result: String)
        fun onPlanUpdate(rawPlan: String)
        fun onStateChange(state: AgentState)
        fun onError(message: String)
        fun onTodos(todos: JSONArray) {}
        fun onAskQuestion(questions: JSONArray, onAnswer: (JSONArray?) -> Unit) {
            onAnswer(null)
        }
        fun onCommandOutput(commandId: String, line: String) {}
        /** Fired after a write/edit/patch tool completed, with the affected path. */
        fun onFileChanged(path: String) {}
        /**
         * Inline, non-error notice (unknown /command, unresolved @file, ...).
         * Surfaces render it as a system bubble; message flow is never blocked.
         */
        fun onNotice(text: String) {}
    }

    enum class RiskLevel { READ_ONLY, BUILD, MODIFY, DANGEROUS }

    enum class AgentState {
        IDLE, THINKING, GENERATING, WAITING_TOOL_APPROVAL, EXECUTING_TOOL,
        PROCESSING_TOOL_RESULT, COMPLETED, ERROR, CANCELLED,
    }

    private val settings = AiSettings(context)
    private val store = OcSessionStore(context)
    private val snapshots = OcSnapshot(context)
    private val instructions = OcInstructions(context)
    private val modelManager = ModelManager(context)
    private val mcpClient = OcMcpClient()

    private var config: OpencodeConfig = OpencodeConfig.load(context, worktreeOf(context))
    private var agentRegistry = OcAgentRegistry(context, { config })
    private var toolRegistry = OcToolRegistry()
    private var skillManager = OcSkill(context) { config }
    private var commandRegistry = OcCommandRegistry(context, { config }, { skillManager }, { mcpServers }, mcpClient)
    private var providerRegistry = OcProviderRegistry(context, { config }, modelManager)

    private var mcpServers: List<OcMcpClient.ServerInfo> = emptyList()
    private var mcpReady = false

    var provider: AIProvider? = null
    var modelRef: String? = null
    var agentName: String = agentRegistry.defaultAgent()
    var listener: Listener? = null
    val runtimeOptions = JSONObject().apply {
        put("nGpuLayers", 32)
        put("nThreads", 4)
    }

    private var currentSession: OcSessionStore.Session? = null
    private val readFiles = mutableSetOf<String>()
    private val toolCallCounts = mutableMapOf<String, Int>()
    private var activeMessageId: String = ""
    private var subagentDepth = 0

    // --- Checkpoints (/undo //redo): persisted per session under filesDir/ai/checkpoints ---
    private val checkpointStore = CheckpointStore(File(context.filesDir, "ai/checkpoints"))

    /** Groups every file mutation of one user request so /undo rolls back the whole turn. */
    @Volatile
    private var currentTurnId: String = ""

    // --- @file mentions: resolved once per queued message, attached at request time ---
    @Volatile
    private var pendingMentionContext: String? = null

    private data class PendingMessage(val text: String, val imageBase64: String? = null)

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var currentJob: Job? = null
    private val queue = mutableListOf<PendingMessage>()
    private var isRunning = false

    private var currentFileContext: String? = null
    private var currentFilePath: String? = null

    fun setAgentMode(mode: String) {
        agentName = mode
    }

    fun setCurrentFileContext(filePath: String?, content: String?, selection: String? = null) {
        currentFilePath = filePath
        if (filePath == null || content == null || content.isBlank()) {
            currentFileContext = null
            return
        }
        val sb = StringBuilder()
        sb.append("Current file: ").append(filePath).append("\n\n")
        if (selection != null && selection.isNotBlank()) {
            sb.append("Selected text:\n```\n").append(selection).append("\n```\n\n")
        }
        sb.append(content.take(6_000))
        currentFileContext = sb.toString()
    }

    /** Absolute path of the file set via [setCurrentFileContext], if any. */
    fun currentFileContextPath(): String? = currentFilePath

    /** Project root the agent operates on (used by @file mentions). */
    fun workspaceRoot(): File = projectDir ?: config.worktree

    fun estimateTokens(): Int {
        val session = currentSession ?: return 0
        var total = 0
        for (msg in store.messages(session)) {
            val parts = OcMessage.parts(msg)
            for (j in 0 until parts.length()) {
                val part = parts.optJSONObject(j) ?: continue
                total += part.optString("text", "").length
            }
        }
        return total / 4
    }

    fun reloadConfig() {
        config = OpencodeConfig.load(context, worktreeOf(context))
        agentRegistry = OcAgentRegistry(context, { config })
        skillManager = OcSkill(context) { config }
        commandRegistry = OcCommandRegistry(context, { config }, { skillManager }, { mcpServers }, mcpClient)
        providerRegistry = OcProviderRegistry(context, { config }, modelManager)
        agentName = agentRegistry.defaultAgent()
        mcpReady = false
        initTools()
    }

    private fun worktreeOf(context: Context): File {
        projectDir?.let { return it }
        val project = (context as? com.aide.mobilestudio.MainActivity)?.currentProjectDir
        return project ?: File(context.filesDir, "workspace").apply { mkdirs() }
    }

    /** Project root override for hosts that are not [com.aide.mobilestudio.MainActivity]. */
    var projectDir: File? = null

    private val toolContext: ToolContext
        get() = toolContextFor(currentSession?.id ?: "")

    private fun toolContextFor(sessionId: String): ToolContext = ToolContext(
        androidContext = context,
        worktree = config.worktree,
        sessionId = sessionId,
        config = config,
        bridge = this,
        tmpDir = File(context.filesDir, "tmp").apply { mkdirs() },
    )

    fun init() {
        initTools()
    }

    private fun initTools() {
        toolRegistry = OcToolRegistry()
        toolRegistry.registerAll(
            listOf(
                BashTool(),
                ReadTool(),
                WriteTool(),
                EditTool(),
                GlobTool(),
                GrepTool(),
                PatchTool(),
                TodoWriteTool(),
                TaskTool(),
                QuestionTool(),
                SkillTool { skillManager },
                WebFetchTool(),
                WebSearchTool(),
            ),
        )
        mcpServers = runCatching { mcpClient.parseServers(config.mcp) }.getOrDefault(emptyList())
    }

    private suspend fun ensureMcp() {
        if (mcpReady || mcpServers.isEmpty()) {
            mcpReady = true
            return
        }
        withContext(Dispatchers.IO) {
            for (server in mcpServers) {
                runCatching {
                    for (mcpTool in mcpClient.listTools(server)) {
                        toolRegistry.register(McpToolAdapter(server, mcpTool, mcpClient))
                    }
                }
            }
        }
        mcpReady = true
    }

    fun currentSessionId(): String? = currentSession?.id

    fun newSession() {
        val session = store.create(
            agent = agentName,
            model = currentModelRef()?.toString()?.let { JSONObject().put("ref", it) },
        )
        currentSession = session
        activeMessageId = session.messages.optJSONObject(session.messages.length() - 1)?.optString("id") ?: ""
        readFiles.clear()
        toolCallCounts.clear()
        currentTurnId = ""
        pendingMentionContext = null
        snapshots.clear(session.id)
        listener?.onStateChange(AgentState.IDLE)
    }

    fun loadSession(sessionId: String) {
        val session = store.load(sessionId) ?: return
        currentSession = session
        agentName = session.agent.ifEmpty { agentRegistry.defaultAgent() }
        activeMessageId = ""
        readFiles.clear()
        toolCallCounts.clear()
        listener?.onStateChange(AgentState.IDLE)
    }

    fun listSessions(includeArchived: Boolean = false): List<OcSessionStore.Session> = store.list(includeArchived)

    fun sessionTitle(sessionId: String): String = store.load(sessionId)?.title ?: "Untitled"

    fun sessionMessages(sessionId: String): List<JSONObject> {
        val session = store.load(sessionId) ?: return emptyList()
        return store.messages(session)
    }

    fun renameSession(sessionId: String, title: String) {
        store.load(sessionId)?.let { store.setTitle(it, title) }
    }

    fun forkSession(sessionId: String, messageID: String? = null): String? {
        val forked = store.fork(sessionId, messageID) ?: return null
        return forked.id
    }

    fun archiveSession(sessionId: String, archived: Boolean): Boolean {
        return if (archived) store.archive(sessionId) else store.unarchive(sessionId)
    }

    fun deleteSession(sessionId: String): Boolean = store.delete(sessionId)

    fun sessionMessages(): List<JSONObject> {
        val session = currentSession ?: return emptyList()
        return store.messages(session)
    }

    fun revert(): List<String> {
        val session = currentSession ?: return emptyList()
        val restored = snapshots.restore(session.id)
        if (restored.isNotEmpty()) listener?.onMessage("Reverted ${restored.size} file(s).", false)
        return restored
    }

    fun listCommands(): List<OcCommandInfo> = runCatching { commandRegistry.list() }.getOrDefault(emptyList())

    fun listProviders(): List<Pair<String, String>> = providerRegistry.listProviders()

    fun listProviderModels(providerId: String): List<OcModelInfo> = providerRegistry.listModels(providerId)

    fun currentModelLabel(): String = currentModelRef()?.toString() ?: "none"

    fun currentTodos(): JSONArray = currentSession?.let { runCatching { store.load(it.id)?.todos }.getOrNull() }
        ?: JSONArray()

    /** Replaces the current session's todo list (used by structural plan editing). */
    fun updateTodos(todos: JSONArray) {
        val session = currentSession ?: return
        store.setTodos(session, todos)
        listener?.onTodos(todos)
    }

    fun latestBackupForCurrentSession(path: String): File? =
        currentSession?.let { runCatching { snapshots.latestBackup(it.id, path) }.getOrNull() }

    fun mcpServerNames(): List<String> = mcpServers.map { it.name }

    /** Names of tools currently registered with the engine. */
    fun toolNames(): List<String> = toolRegistry.names()

    /** Adds a local MCP server to the global config and reloads. */
    fun addMcpServer(name: String, command: List<String>) {
        OpencodeConfig.upsertMcpServer(context, name, command)
        reloadConfig()
    }

    fun removeMcpServer(name: String): Boolean {
        val removed = OpencodeConfig.removeMcpServer(context, name)
        if (removed) reloadConfig()
        return removed
    }

    /** Probes an MCP server's `tools/list`. Runs on IO; returns a human-readable status. */
    suspend fun testMcpServer(name: String): String {
        val server = mcpServers.firstOrNull { it.name == name }
            ?: return "Unknown MCP server: $name"
        return withContext(Dispatchers.IO) {
            val tools = runCatching { mcpClient.listTools(server) }.getOrNull()
            if (tools == null) {
                "MCP '$name': no response (server not running or not reachable)"
            } else {
                "MCP '$name': ${tools.size} tool(s)" +
                    if (tools.isNotEmpty()) " — ${tools.take(6).joinToString(", ") { it.name }}" else ""
            }
        }
    }

    fun listAgents(): List<OcAgent> = agentRegistry.listVisible()

    fun currentAgent(): OcAgent = agentRegistry.get(agentName)

    fun setAgent(name: String) {
        if (agentRegistry.agents().containsKey(name)) agentName = name
    }

    fun currentModelRef(): OcModelRef? {
        modelRef?.let { OcModelRef.parse(it) }?.let { if (providerRegistry.getModel(it) != null) return it }
        config.model?.let { OcModelRef.parse(it) }?.let { if (providerRegistry.getModel(it) != null) return it }
        val apiKey = settings.apiKey
        if (apiKey != null) return OcModelRef("gemini", "gemini-1.5-flash")
        val local = providerRegistry.listModels("local").firstOrNull() ?: return null
        return local.ref
    }

    fun setModel(ref: String) {
        modelRef = ref
        settings.lastModelId = ref
        provider = null
    }

    fun resolveModelInfo(): OcModelInfo? = currentModelRef()?.let { providerRegistry.getModel(it) }

    fun stop() {
        currentJob?.cancel()
        currentJob = null
        queue.clear()
        listener?.onStateChange(AgentState.CANCELLED)
    }

    fun sendMessage(text: String, imageBase64: String? = null) {
        if (text.isBlank() && imageBase64 == null) return
        if (currentSession == null) newSession()
        if (text.trimStart().startsWith("/")) {
            if (executeCommand(text.trim())) return
        }
        queue.add(PendingMessage(text, imageBase64))
        pumpQueue()
    }

    private fun executeCommand(text: String): Boolean {
        val body = text.removePrefix("/").trim()
        val name = body.substringBefore(" ").trim()
        val args = body.substringAfter(" ", "").trim()
        val cmd = runCatching { commandRegistry.get(name) }.getOrNull() ?: return false
        scope.launch {
            try {
                val template = cmd.template()
                if (template.isBlank()) {
                    listener?.onError("Command /$name has an empty template")
                    return@launch
                }
                val rendered = commandRegistry.interpolate(template, args)
                if (cmd.model != null) setModel(cmd.model)
                if (cmd.subtask) {
                    listener?.onThinking("Running /$name as subagent...")
                    val subagentType = cmd.agent ?: "general"
                    val result = spawnSubagent(subagentType, rendered, name, null, currentSession?.id ?: "", false)
                    listener?.onMessage(result, true)
                } else {
                    if (cmd.agent != null) setAgent(cmd.agent)
                    queue.add(PendingMessage(rendered))
                    pumpQueue()
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                listener?.onError("Command /$name failed: ${e.message}")
            }
        }
        return true
    }

    private fun pumpQueue() {
        if (isRunning) return
        val text = if (queue.isEmpty()) {
            listener?.onStateChange(AgentState.IDLE)
            return
        } else {
            queue.removeAt(0)
        }
        val session = currentSession ?: return
        isRunning = true
        currentJob = scope.launch {
            try {
                // Resolve @file mentions off the main thread, once per message:
                // the block is attached to every request of this agent turn.
                val resolution = withContext(Dispatchers.IO) { resolveMentions(text.text) }
                pendingMentionContext = resolution.block
                if (resolution.unknown.isNotEmpty()) {
                    listener?.onNotice(
                        "[@file] not found: " +
                            resolution.unknown.joinToString(", ") { "@$it" } +
                            " — message sent without it.",
                    )
                }
                currentTurnId = OcMessage.nextId("turn")
                addUserMessage(text.text, text.imageBase64)
                runLoop(session, agentRegistry.get(agentName), 0)
                listener?.onStateChange(AgentState.COMPLETED)
            } catch (e: CancellationException) {
                listener?.onStateChange(AgentState.CANCELLED)
            } catch (e: Exception) {
                listener?.onError(e.message ?: "Unknown error during agent loop")
                listener?.onStateChange(AgentState.ERROR)
            } finally {
                pendingMentionContext = null
                isRunning = false
                if (queue.isNotEmpty() && isActive) {
                    pumpQueue()
                } else {
                    listener?.onStateChange(AgentState.IDLE)
                }
            }
        }
    }

    private fun addUserMessage(text: String, imageBase64: String? = null) {
        val session = currentSession ?: return
        val modelJson = currentModelRef()?.let { JSONObject().put("ref", it.toString()) }
        val msg = OcMessage.userMessage(session.id, agentName, modelJson)
        OcMessage.addPart(msg, OcMessage.textPart(text))
        if (imageBase64 != null) {
            OcMessage.addPart(
                msg,
                JSONObject().apply {
                    put("type", "file")
                    put("id", OcMessage.nextId("part"))
                    put("mime", "image/jpeg")
                    put("filename", "pasted-image.jpg")
                    put("data", imageBase64)
                },
            )
        }
        store.updateMessage(session, msg)
        activeMessageId = msg.optString("id")
    }

    private suspend fun runLoop(session: OcSessionStore.Session, agent: OcAgent, depth: Int): String {
        var iterations = 0
        val maxSteps = agent.steps ?: 80
        var lastText = ""
        var didCompact = false
        val ctx = toolContextFor(session.id)
        while (true) {
            ensureMcp()
            val model = resolveModelFor(agent)
            if (model == null) {
                listener?.onError("No model configured. Set an API key or select a local model.")
                return lastText
            }
            val systemText = buildSystemPrompt(agent, model, session)
            val messages = store.messages(session)
            if (!didCompact && config.compactionAuto() && messages.count { OcMessage.isUser(it) } >= 2) {
                if (OcCompaction.compactIfNeeded(
                        messages, systemText, "{}", config.compaction, true,
                        model.contextLimit, null, model.outputLimit, config.compactionReserved(),
                    )
                ) {
                    didCompact = true
                    OcCompaction.prune(store, session.id, config.compactionPrune())
                    runCompaction(session, model, agent)
                    continue
                }
            }
            val history = buildHistory(session, systemText, messages)
            val visible = visibleTools(agent)
            val toolDefs = toolRegistry.definitions(visible, ctx)
            val provider = providerRegistry.createProvider(model.ref, runtimeOptions)
            if (provider == null) {
                listener?.onError("Unable to create provider for ${model.ref}")
                return lastText
            }

            listener?.onStateChange(AgentState.GENERATING)
            val textBuf = StringBuilder()
            val reasoningBuf = StringBuilder()
            val toolCalls = mutableListOf<AIProvider.StreamEvent.ToolCall>()
            var streamError: String? = null

            provider.generateStream(history, toolDefs, runtimeOptions).collect { event ->
                when (event) {
                    is AIProvider.StreamEvent.Text -> {
                        if (textBuf.isEmpty()) listener?.onReasoningDone()
                        textBuf.append(event.text)
                        listener?.onMessage(textBuf.toString(), false)
                    }
                    is AIProvider.StreamEvent.Reasoning -> {
                        reasoningBuf.append(event.text)
                        listener?.onReasoning(reasoningBuf.toString())
                        val snippet = reasoningBuf.toString().takeLast(120).replace("\n", " ")
                        listener?.onThinking(snippet)
                    }
                    is AIProvider.StreamEvent.Plan -> listener?.onPlanUpdate(event.rawPlan)
                    is AIProvider.StreamEvent.ToolCall -> toolCalls.add(event)
                    is AIProvider.StreamEvent.Error -> streamError = event.message
                    AIProvider.StreamEvent.Done -> {}
                }
            }

            if (streamError != null) {
                listener?.onError(streamError)
                listener?.onStateChange(AgentState.ERROR)
                return lastText
            }

            val messagesNow = store.messages(session)
            val lastUser = messagesNow.lastOrNull { OcMessage.isUser(it) }
            val assistant = OcMessage.assistantMessage(
                session.id, agent.name, modelJsonOf(model), parentID = lastUser?.optString("id"),
            )
            val parts = assistant.optJSONArray("parts") ?: JSONArray()
            if (reasoningBuf.isNotEmpty()) parts.put(OcMessage.reasoningPart(reasoningBuf.toString()))
            if (textBuf.isNotEmpty()) parts.put(OcMessage.textPart(textBuf.toString()))
            val pendingTools = mutableListOf<Pair<JSONObject, AIProvider.StreamEvent.ToolCall>>()
            for (call in toolCalls) {
                val callId = call.id ?: OcMessage.nextCallId()
                val part = OcMessage.toolPart(call.name, callId, call.args)
                parts.put(part)
                pendingTools.add(part to call)
            }
            store.updateMessage(session, assistant)
            activeMessageId = assistant.optString("id")
            lastText = textBuf.toString()
            if (textBuf.isNotEmpty()) listener?.onMessage(textBuf.toString(), true)

            if (pendingTools.isEmpty()) {
                listener?.onStateChange(AgentState.COMPLETED)
                maybeGenerateTitle(session, lastText)
                return lastText
            }

            val canParallelize = depth == 0 &&
                pendingTools.size >= 2 &&
                pendingTools.all { it.second.name in READ_ONLY_TOOLS } &&
                pendingTools.all { (_, call) ->
                    val tool = toolRegistry.get(call.name) ?: return@all false
                    val hash = call.args?.toString() ?: "{}"
                    permissionActionFor(tool, call, agent, ctx, currentToolCallCount(call.name, hash)) == "allow"
                }

            if (canParallelize) {
                executeReadOnlyToolsInParallel(session, assistant, pendingTools, ctx)
                if (!isActiveLoop()) {
                    listener?.onStateChange(AgentState.CANCELLED)
                    return lastText
                }
            } else {
            for ((part, call) in pendingTools) {
                if (!isActiveLoop()) {
                    listener?.onStateChange(AgentState.CANCELLED)
                    return lastText
                }
                val beforeDiagnostics = if (depth == 0) getGlobalDiagnostics() else emptyList()
                listener?.onStateChange(AgentState.PROCESSING_TOOL_RESULT)
                val state = executeToolCall(call, agent, ctx)
                part.put("state", state)
                store.updatePart(session, assistant.optString("id"), part)
                val output = state.optString("output", state.optString("error", ""))
                listener?.onToolEnd(call.name, output.take(2000))
                if (depth == 0 && call.name in setOf("write", "edit", "patch") &&
                    state.optString("status") == "completed"
                ) {
                    handlePostWriteAnalysis(beforeDiagnostics, session)
                }
                if (call.name in setOf("write", "edit", "patch") && state.optString("status") == "completed") {
                    val changed = call.args.optString("filePath", "")
                        .ifEmpty { call.args.optString("path", "") }
                    if (changed.isNotEmpty()) listener?.onFileChanged(changed)
                }
            }
            }

            iterations++
            if (iterations >= maxSteps) {
                val notice = "Maximum number of agentic iterations ($maxSteps) reached. Respond with text only."
                val assistant2 = OcMessage.assistantMessage(
                    session.id, agent.name, modelJsonOf(model), parentID = assistant.optString("id"),
                )
                OcMessage.addPart(assistant2, OcMessage.textPart(notice))
                store.updateMessage(session, assistant2)
                listener?.onMessage(notice, true)
                return notice
            }
        }
    }

    private fun isActiveLoop(): Boolean = currentJob?.isActive != false || isRunning

    private fun resolveModelFor(agent: OcAgent): OcModelInfo? {
        val ref = agent.model?.let { OcModelRef.parse(it) }
            ?: currentModelRef()
            ?: return null
        providerRegistry.getModel(ref)?.let { return it }
        return providerRegistry.resolve(config.model)
    }

    private fun modelJsonOf(model: OcModelInfo): JSONObject = JSONObject().apply {
        put("ref", model.ref.toString())
        put("limit", JSONObject().apply {
            put("context", model.contextLimit)
            if (model.outputLimit != null) put("output", model.outputLimit)
        })
    }

    private class MentionResolution(val block: String?, val unknown: List<String>)

    /** Resolves @path tokens in [text] into a context block (runs on IO). */
    private fun resolveMentions(text: String): MentionResolution {
        val mentions = FileMentions.extract(text)
        if (mentions.isEmpty()) return MentionResolution(null, emptyList())
        val resolved = runCatching { FileMentions.resolve(workspaceRoot(), mentions) }.getOrNull()
            ?: return MentionResolution(null, mentions)
        val block = FileMentions.formatContextBlock(resolved)
        val unknown = resolved.filter { it.path == null }.map { it.mention }
        return MentionResolution(block.ifEmpty { null }, unknown)
    }

    private fun buildHistory(
        session: OcSessionStore.Session?,
        systemText: String,
        messages: List<JSONObject>,
    ): JSONArray {
        val arr = JSONArray()
        arr.put(
            JSONObject().apply {
                put("role", "system")
                put("parts", JSONArray().put(OcMessage.textPart(systemText)))
            },
        )
        // Request-time context (@file mentions + the open editor file) is
        // appended to a COPY of the last user message: every request of the
        // turn carries it, while the stored transcript stays clean.
        val isTopLevel = session != null && session.id == currentSession?.id
        val contextParts = mutableListOf<String>()
        if (isTopLevel) {
            pendingMentionContext?.let { contextParts.add(it) }
            currentFileContext?.let { contextParts.add("---EDITOR CONTEXT---\n$it") }
        }
        val contextText = contextParts.joinToString("\n\n")
        var lastUserIndex = -1
        if (contextText.isNotEmpty()) {
            for (i in messages.indices) if (OcMessage.isUser(messages[i])) lastUserIndex = i
        }
        for ((i, msg) in messages.withIndex()) {
            if (i == lastUserIndex) {
                val copy = JSONObject(msg.toString())
                val part = OcMessage.textPart(contextText, synthetic = true)
                part.put("ignored", true)
                OcMessage.addPart(copy, part)
                arr.put(copy)
            } else {
                arr.put(msg)
            }
        }
        return arr
    }

    private fun visibleTools(agent: OcAgent): Set<String> {
        val all = toolRegistry.names().toSet()
        val result = all.toMutableSet()
        result.removeAll(Permission.disabledTools(all.toList(), agent.permission))
        val configTools = config.tools
        if (configTools != null) {
            for (name in all) {
                if (configTools.has(name) && !configTools.optBoolean(name, true)) result.remove(name)
            }
        }
        val primaryOnly = config.experimental?.optJSONArray("primary_tools")
        if (primaryOnly != null && agent.mode == "subagent") {
            for (i in 0 until primaryOnly.length()) result.remove(primaryOnly.optString(i))
        }
        return result
    }

    private suspend fun executeToolCall(
        call: AIProvider.StreamEvent.ToolCall,
        agent: OcAgent,
        ctx: ToolContext,
    ): JSONObject {
        val state = JSONObject().put("input", call.args)
        val tool = toolRegistry.get(call.name)
        if (tool == null) {
            return state.put("status", "error")
                .put("output", "Unknown tool: ${call.name}. Available tools: ${toolRegistry.names().sorted().joinToString(", ")}")
        }
        val argsJson = call.args?.toString() ?: "{}"
        val callCount = recordToolCall(call.name, argsJson)
        listener?.onToolStart(call.name, call.args ?: JSONObject())
        state.put("status", "running")
        listener?.onStateChange(AgentState.WAITING_TOOL_APPROVAL)

        val permissionKey = tool.permissionKey()
        val pattern = tool.resourcePattern(call.args, ctx)
        val risk = tool.riskHint(call.args, ctx)
        val rules = agent.permission + savedPermissions()
        var action = permissionActionFor(tool, call, agent, ctx, callCount)
        val external = externalResource(tool, call.args, ctx)

        if (action == "deny") {
            listener?.onStateChange(AgentState.PROCESSING_TOOL_RESULT)
            return state.put("status", "error")
                .put("output", "Permission denied by policy for tool '$permissionKey' (pattern: $pattern).")
        }
        if (action == "ask") {
            val decision = askApproval(call.name, call.args ?: JSONObject(), risk)
            if (!decision.first) {
                listener?.onStateChange(AgentState.PROCESSING_TOOL_RESULT)
                return state.put("status", "error")
                    .put("output", "User rejected the tool call '${call.name}'. Continue and adjust.")
            }
            if (decision.second) {
                val rule = PermissionRule(permissionKey, "allow", pattern)
                addSavedPermission(rule)
                if (external != null) addSavedPermission(PermissionRule("external_directory", "allow", external))
            }
        }

        listener?.onStateChange(AgentState.EXECUTING_TOOL)
        // Single capture hook for /undo //redo: before ANY file-mutating tool
        // runs, checkpoint the affected file's pre-edit content.
        if (call.name in CHECKPOINT_TOOLS) captureCheckpoint(call.args, ctx)
        val result = try {
            tool.execute(call.args ?: JSONObject(), ctx)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            ToolResult("${call.name} failed: ${e.message}", isError = true)
        }
        state.put("status", if (result.isError) "error" else "completed")
        state.put("output", result.output)
        return state
    }

    /** Tools whose successful run mutates a project file. */
    private val CHECKPOINT_TOOLS = setOf("write", "edit", "patch")

    /**
     * Checkpoints the pre-edit content of the file a mutating tool is about to
     * touch. Content is read on IO; missing files are recorded as "created"
     * (undo deletes them). One call site: [executeToolCall].
     */
    private suspend fun captureCheckpoint(args: JSONObject?, ctx: ToolContext) {
        val path = args?.optString("filePath", "")?.ifEmpty { args.optString("path", "") }
        if (path.isNullOrEmpty()) return
        val turn = currentTurnId.ifEmpty { OcMessage.nextId("turn") }
        withContext(Dispatchers.IO) {
            runCatching { checkpointStore.capture(ctx.sessionId, turn, ctx.resolveInWorkspace(path)) }
        }
    }

    /** Files the next /undo would restore, with line-count deltas (IO). */
    fun pendingUndoDeltas(): List<CheckpointFileDelta> {
        val sessionId = currentSession?.id ?: return emptyList()
        return runCatching { checkpointStore.describeUndo(sessionId) }.getOrDefault(emptyList())
    }

    /** Rolls back all files of the newest checkpointed turn. Blocking IO. */
    fun undoLastTurn(): CheckpointOutcome {
        val sessionId = currentSession?.id ?: return CheckpointOutcome.NothingToUndo
        val outcome = runCatching { checkpointStore.undo(sessionId) }.getOrElse {
            return CheckpointOutcome.Failed(it.message ?: "undo failed", emptyList())
        }
        if (outcome is CheckpointOutcome.Undone) {
            for (path in outcome.files) listener?.onFileChanged(path)
        }
        return outcome
    }

    /** Re-applies the most recently undone turn. Blocking IO. */
    fun redoLastTurn(): CheckpointOutcome {
        val sessionId = currentSession?.id ?: return CheckpointOutcome.NothingToRedo
        val outcome = runCatching { checkpointStore.redo(sessionId) }.getOrElse {
            return CheckpointOutcome.Failed(it.message ?: "redo failed", emptyList())
        }
        if (outcome is CheckpointOutcome.Redone) {
            for (path in outcome.files) listener?.onFileChanged(path)
        }
        return outcome
    }

    /** Shared permission decision (used by both sequential and parallel execution). */
    private fun permissionActionFor(
        tool: OcTool,
        call: AIProvider.StreamEvent.ToolCall,
        agent: OcAgent,
        ctx: ToolContext,
        callCount: Int,
    ): String {
        val rules = agent.permission + savedPermissions()
        var action = Permission.evaluate(tool.permissionKey(), tool.resourcePattern(call.args, ctx), rules).action
        val external = externalResource(tool, call.args, ctx)
        if (external != null) {
            val extAction = Permission.evaluate("external_directory", external, rules).action
            if (extAction == "deny") action = "deny"
            else if (extAction == "ask" && action != "deny") action = "ask"
        }
        val doom = Permission.evaluate("doom_loop", "*", rules).action
        if (doom == "deny") action = "deny"
        else if (doom == "ask" && callCount >= 3) action = "ask"
        return action
    }

    private fun currentToolCallCount(tool: String, argsHash: String): Int =
        toolCallCounts["$tool|$argsHash"] ?: 0

    /**
     * Runs several auto-allowed read-only tools concurrently with a parallelism
     * chosen by the resource policy. Only reached when every call is read-only
     * and already permitted (no approval prompt is bypassed).
     */
    private suspend fun executeReadOnlyToolsInParallel(
        session: OcSessionStore.Session,
        assistant: JSONObject,
        pendingTools: List<Pair<JSONObject, AIProvider.StreamEvent.ToolCall>>,
        ctx: ToolContext,
    ) {
        listener?.onStateChange(AgentState.EXECUTING_TOOL)
        val snapshot = AndroidMemoryProbe.snapshot(context)
        val tasks = pendingTools.map { (part, call) ->
            suspend {
                listener?.onToolStart(call.name, call.args)
                val tool = toolRegistry.get(call.name)
                val result = if (tool == null) {
                    ToolResult("Unknown tool: ${call.name}", isError = true)
                } else {
                    try {
                        tool.execute(call.args, ctx)
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        ToolResult("${call.name} failed: ${e.message}", isError = true)
                    }
                }
                Triple(part, call, result)
            }
        }
        val outcome = taskScheduler.runAll(tasks, snapshot, loadedLocalModels = 0, buildRunning = false)
        for ((part, call, result) in outcome.results) {
            val state = JSONObject().put("input", call.args)
                .put("status", if (result.isError) "error" else "completed")
                .put("output", result.output)
            part.put("state", state)
            store.updatePart(session, assistant.optString("id"), part)
            listener?.onToolEnd(call.name, result.output.take(2000))
        }
    }

    private fun externalResource(tool: OcTool, args: JSONObject?, ctx: ToolContext): String? {
        val path = when (tool.name) {
            "read", "write", "edit", "patch" -> args?.optString("path")?.takeIf { it.isNotEmpty() }
            else -> null
        } ?: return null
        val resolved = ctx.resolveInWorkspace(path)
        return if (ctx.isInsideWorkspace(resolved)) null else resolved.absolutePath
    }

    override suspend fun askApproval(
        tool: String,
        args: JSONObject,
        risk: RiskHint,
    ): Pair<Boolean, Boolean> {
        val deferred = CompletableDeferred<Pair<Boolean, Boolean>>()
        withContext(Dispatchers.Main) {
            listener?.onToolCall(tool, args, risk.toRiskLevel()) { allow, always ->
                deferred.complete(allow to always)
            }
        }
        return deferred.await()
    }

    private fun savedPermissions(): List<PermissionRule> = runCatching {
        val json = JSONArray(AiSettings(context).ocPermissionsJson())
        val out = mutableListOf<PermissionRule>()
        for (i in 0 until json.length()) {
            val o = json.optJSONObject(i) ?: continue
            out.add(
                PermissionRule(
                    o.optString("permission"), o.optString("action", "allow"), o.optString("pattern"),
                ),
            )
        }
        out
    }.getOrDefault(emptyList())

    private fun addSavedPermission(rule: PermissionRule) {
        val settings = AiSettings(context)
        val arr = runCatching { JSONArray(settings.ocPermissionsJson()) }.getOrDefault(JSONArray())
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            if (o.optString("permission") == rule.permission && o.optString("pattern") == rule.pattern) {
                o.put("action", rule.action)
                settings.setOcPermissionsJson(arr.toString())
                return
            }
        }
        arr.put(JSONObject().put("permission", rule.permission).put("action", rule.action).put("pattern", rule.pattern))
        settings.setOcPermissionsJson(arr.toString())
    }

    override suspend fun spawnSubagent(
        subagentType: String,
        prompt: String,
        description: String,
        taskId: String?,
        parentSessionId: String,
        background: Boolean,
    ): String {
        val agents = agentRegistry.agents()
        val agent = agents[subagentType]
            ?: return "Error: unknown subagent '$subagentType'. Available: ${agents.keys.sorted().joinToString(", ")}"
        if (subagentDepth >= config.subagentDepth) {
            return "Error: Maximum subagent nesting depth (${config.subagentDepth}) reached. Continue the task yourself instead of spawning more subagents."
        }
        if (background) {
            scope.launch { runSubagent(agent, prompt, taskId, background = true) }
            return "The task is running in the background. You will be notified when it finishes. task_id: ${taskId ?: "pending"}"
        }
        return runSubagent(agent, prompt, taskId, background = false)
    }

    private suspend fun runSubagent(
        agent: OcAgent,
        prompt: String,
        taskId: String?,
        background: Boolean,
    ): String {
        val session = if (taskId != null) {
            store.load(taskId) ?: return "Error: failed to load subagent session '$taskId'"
        } else {
            val modelJson = agent.model?.let { JSONObject().put("ref", it) }
                ?: currentModelRef()?.let { JSONObject().put("ref", it.toString()) }
            store.create(agent.name, modelJson)
        }
        subagentDepth++
        try {
            val modelJson = agent.model?.let { JSONObject().put("ref", it) }
            val msg = OcMessage.userMessage(session.id, agent.name, modelJson)
            OcMessage.addPart(msg, OcMessage.textPart(prompt))
            store.updateMessage(session, msg)
            val result = runLoop(session, agent, subagentDepth)
            if (background) {
                withContext(Dispatchers.Main) {
                    listener?.onMessage(
                        "[background task ${session.id} finished]\n$result", true,
                    )
                }
            }
            return result
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return "Error: subagent failed: ${e.message}"
        } finally {
            subagentDepth--
        }
    }

    private suspend fun runCompaction(session: OcSessionStore.Session, model: OcModelInfo, agent: OcAgent) {
        listener?.onThinking("Compacting conversation...")
        listener?.onStateChange(AgentState.GENERATING)
        val smallRef = config.smallModel?.let { OcModelRef.parse(it) } ?: model.ref
        val llm = object : OcCompaction.Llm {
            override suspend fun generate(prompt: String, outputTokens: Int): String? {
                val prov = providerRegistry.createProvider(smallRef, runtimeOptions) ?: return null
                val history = JSONArray().put(
                    JSONObject().apply {
                        put("role", "system")
                        put("parts", JSONArray().put(OcMessage.textPart(prompt)))
                    },
                )
                val opts = JSONObject(runtimeOptions.toString())
                    .put("maxOutputTokens", outputTokens).put("temperature", 0.5)
                val sb = StringBuilder()
                prov.generateStream(history, JSONArray(), opts).collect { event ->
                    if (event is AIProvider.StreamEvent.Text) sb.append(event.text)
                    if (event is AIProvider.StreamEvent.Error) return@collect
                }
                return sb.toString().ifBlank { null }
            }
        }
        OcCompaction.compact(
            session.id, session, store, modelJsonOf(model), false, llm,
            agent.model?.let { JSONObject().put("ref", it) },
        )
        listener?.onMessage("[conversation compacted]", true)
    }

    private suspend fun maybeGenerateTitle(session: OcSessionStore.Session, firstResponse: String) {
        if (session.id.startsWith("subagent-")) return
        val current = store.load(session.id) ?: return
        if (current.title.isNotBlank() && current.title != "New session") return
        val messages = store.messages(current)
        val firstUser = messages.firstOrNull { OcMessage.isUser(it) } ?: return
        val text = OcMessage.textOf(firstUser)?.take(400) ?: return
        val smallRef = config.smallModel?.let { OcModelRef.parse(it) } ?: currentModelRef() ?: return
        val titlePrompt = OcAssets.read(context, "ai/prompt/agent/title.txt") ?: "Generate a title."
        val generated = generateText(
            system = titlePrompt,
            user = "Generate a title (3-5 words) for this conversation:\n\n$text",
            modelRef = smallRef,
            temperature = 0.5f,
        ) ?: return
        val title = generated.trim().lineSequence().firstOrNull()
            ?.trim('"', ' ', '\n')?.take(80)
        if (!title.isNullOrBlank()) store.setTitle(current, title)
    }

    suspend fun generateText(
        system: String,
        user: String,
        modelRef: OcModelRef,
        temperature: Float? = null,
    ): String? {
        val prov = providerRegistry.createProvider(modelRef, runtimeOptions) ?: return null
        val history = JSONArray().apply {
            put(JSONObject().apply {
                put("role", "system")
                put("parts", JSONArray().put(OcMessage.textPart(system)))
            })
            put(JSONObject().apply {
                put("role", "user")
                put("parts", JSONArray().put(OcMessage.textPart(user)))
            })
        }
        val opts = JSONObject(runtimeOptions.toString())
        if (temperature != null) opts.put("temperature", temperature.toDouble())
        val sb = StringBuilder()
        try {
            prov.generateStream(history, JSONArray(), opts).collect { event ->
                when (event) {
                    is AIProvider.StreamEvent.Text -> sb.append(event.text)
                    is AIProvider.StreamEvent.Error -> throw IllegalStateException(event.message)
                    else -> {}
                }
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            return null
        }
        return sb.toString().ifBlank { null }
    }

    private fun buildSystemPrompt(agent: OcAgent, model: OcModelInfo, session: OcSessionStore.Session): String {
        val sb = StringBuilder()
        sb.append(agent.prompt ?: modelSystemPrompt(model)).append("\n\n")
        sb.append(envBlock(model))
        if ("skill" in visibleTools(agent)) {
            val skills = runCatching { skillManager.list() }.getOrDefault(emptyList())
            val denied = skills.filter { Permission.evaluate("skill", it.name, agent.permission).action == "deny" }
            val fmt = runCatching {
                skillManager.fmt(skillManager.availableFor(skills, denied.map { it.name }.toSet()), verbose = true)
            }.getOrDefault("")
            if (fmt.isNotBlank()) {
                sb.append("\n\n").append(OcAssets.read(context, "ai/prompt/tool/skill.txt")?.substringBefore("<skill>"))
                    .append(fmt)
            }
        }
        val instructions = runCatching { instructions.collect(config) }.getOrDefault(emptyList())
        if (instructions.isNotEmpty()) sb.append("\n\n").append(instructions.joinToString("\n\n"))
        if (agent.name == "plan") sb.append("\n\n").append(planModeReminder(session))
        return sb.toString().trim() + "\n"
    }

    private fun modelSystemPrompt(model: OcModelInfo): String {
        val id = model.ref.modelID.lowercase()
        val file = when {
            id.contains("claude") -> "ai/prompt/session/anthropic.txt"
            id.contains("gemini") -> "ai/prompt/session/gemini.txt"
            id.contains("gpt") || id.contains("o1") || id.contains("o3") -> "ai/prompt/session/gpt.txt"
            id.contains("llama") || id.contains("local") -> "ai/prompt/session/default.txt"
            else -> "ai/prompt/session/default.txt"
        }
        return OcAssets.read(context, file) ?: OcAssets.read(context, "ai/prompt/session/default.txt")
            ?: "You are an AI coding assistant."
    }

    private fun envBlock(model: OcModelInfo): String {
        val root = config.worktree
        val git = File(root, ".git").exists()
        val today = java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US)
            .format(java.util.Date())
        return "You are powered by the model named ${model.ref.modelID}. " +
            "The exact model ID is ${model.ref}\nHere is some useful information " +
            "about the environment you are running in:\n<env>\n" +
            "  Working directory: ${root.absolutePath}\n" +
            "  Workspace root folder: ${root.absolutePath}\n" +
            "  Is directory a git repo: ${if (git) "yes" else "no"}\n" +
            "  Platform: android (arm64-v8a)\n" +
            "  Today's date: $today\n" +
            "</env>"
    }

    private fun planModeReminder(session: OcSessionStore.Session): String {
        val plansDir = agentRegistry.plansDir()
        plansDir.mkdirs()
        val slug = (session.title.ifBlank { session.id })
            .lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-').take(48)
        val planFile = File(plansDir, "$slug.md")
        val info = if (planFile.exists()) {
            "Plan file: ${planFile.absolutePath}"
        } else {
            "No plan file exists yet. You should create your plan at ${planFile.absolutePath} " +
                "using the write tool. (You have write access only to this path.)"
        }
        val template = OcAssets.read(context, "ai/prompt/session/plan-mode.txt") ?: return ""
        return template.replace("\${planInfo}", info)
    }

    private val fixAttempts = mutableMapOf<String, Int>()
    private val validationDiff = DiagnosticDiffEngine()
    private val taskScheduler = TaskScheduler()

    /** Tools that only read state and can safely run concurrently. */
    private val READ_ONLY_TOOLS = setOf("read", "glob", "grep", "webfetch", "websearch")

    private suspend fun handlePostWriteAnalysis(
        beforeDiagnostics: List<LiveDiagnostic>,
        session: OcSessionStore.Session,
    ) {
        listener?.onThinking("Analyzing changes...")
        val activity = context as? com.aide.mobilestudio.MainActivity
        activity?.triggerLiveDiagnosticsAndWait()?.join()
        delay(1000)
        val after = getGlobalDiagnostics()
        val diff = validationDiff.diff(beforeDiagnostics, after)
        val introduced = diff.newErrorCount
        val fixed = diff.resolvedErrorCount
        val remaining = diff.remainingErrorCount

        if (introduced > 0) {
            listener?.onMessage("[compiler] ${diff.report().replace("\n", " · ")}", false)
            val newDetails = diff.new.joinToString("\n") { d ->
                "${d.file}:${d.line}:${d.column}: ${d.message}"
            }
            val fb = OcMessage.userMessage(session.id, "compiler_feedback", null)
            OcMessage.addPart(
                fb,
                OcMessage.textPart(
                    "[compiler_feedback] The previous write introduced $introduced new error(s):\n" +
                        "$newDetails\nPlease fix them.",
                ),
            )
            store.updateMessage(session, fb)
            activeMessageId = fb.optString("id")
            val key = diff.new.joinToString(",") { "${it.file}:${it.line}:${it.message}" }
            val attempts = fixAttempts.getOrDefault(key, 0)
            if (attempts >= 5) {
                listener?.onMessage("[compiler] Auto-fix attempts (5) exceeded. Manual intervention required.", false)
            } else {
                fixAttempts[key] = attempts + 1
            }
        } else if (fixed > 0) {
            listener?.onMessage("[compiler] This change fixed $fixed error(s).", false)
            if (remaining == 0) listener?.onMessage("[compiler] All errors resolved.", false)
        } else if (remaining == 0 && activity != null) {
            listener?.onMessage("[compiler] No errors introduced.", false)
        }
        if (diff.uncertain.isNotEmpty()) {
            listener?.onMessage(
                "[compiler] ${diff.uncertain.size} diagnostic(s) matched only by similarity; classification is uncertain.",
                false,
            )
        }
    }

    private fun getGlobalDiagnostics(): List<LiveDiagnostic> {
        val activity = context as? com.aide.mobilestudio.MainActivity ?: return emptyList()
        return LiveDiagnostics.fromProblems(activity.getAllProblems(), activity.currentProjectDir)
    }

    private fun RiskHint.toRiskLevel(): RiskLevel = when (this) {
        RiskHint.READ_ONLY -> RiskLevel.READ_ONLY
        RiskHint.BUILD -> RiskLevel.BUILD
        RiskHint.MODIFY -> RiskLevel.MODIFY
        RiskHint.DANGEROUS -> RiskLevel.DANGEROUS
    }

    override fun currentMessageId(): String = activeMessageId

    override suspend fun askQuestion(questions: JSONArray): JSONArray? {
        val deferred = CompletableDeferred<JSONArray?>()
        withContext(Dispatchers.Main) {
            listener?.onAskQuestion(questions) { answers -> deferred.complete(answers) }
        }
        return deferred.await()
    }

    override fun onTodos(todos: JSONArray) {
        currentSession?.let { store.setTodos(it, todos) }
        listener?.onTodos(todos)
    }

    override fun onToolStart(name: String, args: JSONObject) {
        listener?.onToolStart(name, args)
    }

    override fun onToolEnd(name: String, result: String) {
        listener?.onToolEnd(name, result)
    }

    override fun markFileRead(path: String) {
        readFiles.add(path)
    }

    override fun hasReadFile(path: String): Boolean = path in readFiles

    override fun snapshotFile(messageId: String, file: File) {
        val sid = currentSession?.id ?: return
        runCatching { snapshots.snapshot(sid, messageId, file) }
    }

    override fun recordToolCall(tool: String, argsHash: String): Int {
        val key = "$tool|$argsHash"
        val count = (toolCallCounts[key] ?: 0) + 1
        toolCallCounts[key] = count
        return count
    }
}
