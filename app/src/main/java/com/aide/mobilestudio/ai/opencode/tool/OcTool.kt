package com.aide.mobilestudio.ai.opencode.tool

import android.content.Context
import com.aide.mobilestudio.ai.opencode.config.OpencodeConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

class ToolResult(val output: String, val isError: Boolean = false)

interface EngineBridge {
    fun currentMessageId(): String
    suspend fun askApproval(tool: String, args: JSONObject, risk: RiskHint): Pair<Boolean, Boolean>
    suspend fun askQuestion(questions: JSONArray): JSONArray?
    fun onTodos(todos: JSONArray)
    fun onToolStart(name: String, args: JSONObject)
    fun onToolEnd(name: String, result: String)
    fun markFileRead(path: String)
    fun hasReadFile(path: String): Boolean
    fun snapshotFile(messageId: String, file: File)
    fun recordToolCall(tool: String, argsHash: String): Int
    suspend fun spawnSubagent(
        subagentType: String,
        prompt: String,
        description: String,
        taskId: String?,
        parentSessionId: String,
        background: Boolean,
    ): String
}

enum class RiskHint { READ_ONLY, BUILD, MODIFY, DANGEROUS }

class ToolContext(
    val androidContext: Context,
    val worktree: File,
    val sessionId: String,
    val config: OpencodeConfig,
    val bridge: EngineBridge,
    val tmpDir: File,
) {
    fun resolveInWorkspace(path: String): File {
        val f = File(path)
        return if (f.isAbsolute) f else File(worktree, path)
    }

    fun isInsideWorkspace(file: File): Boolean {
        val canonical = runCatching { file.canonicalFile }.getOrDefault(file)
        val root = runCatching { worktree.canonicalFile }.getOrDefault(worktree)
        return canonical.path == root.path || canonical.path.startsWith(root.path + File.separator)
    }
}

interface OcTool {
    val name: String
    fun description(ctx: ToolContext): String
    fun parameters(): JSONObject
    suspend fun execute(args: JSONObject, ctx: ToolContext): ToolResult

    fun permissionKey(): String = name

    fun resourcePattern(args: JSONObject, ctx: ToolContext): String = "*"

    fun riskHint(args: JSONObject, ctx: ToolContext): RiskHint = RiskHint.READ_ONLY

    /**
     * A one-line version of [description] for small context windows.
     *
     * The full descriptions carry worked examples and edge cases meant for a
     * frontier model, and they are re-sent on every single request. A small
     * window cannot afford them, so it gets the opening sentence instead -
     * enough to choose the right tool - while the parameter schema stays
     * whole, because without it the call cannot be made at all.
     */
    fun shortDescription(ctx: ToolContext): String = firstSentence(description(ctx))
}

/** Length cap for [OcTool.shortDescription]; long enough to describe, short enough to afford. */
private const val SHORT_DESCRIPTION_MAX = 200

/**
 * Collapses a description to its opening sentence.
 *
 * Whitespace is normalised first so multi-line descriptions do not smuggle
 * their blank lines into the prompt, and the result is capped rather than
 * trusted: a tool whose first sentence runs long still has to fit.
 */
private fun firstSentence(text: String): String {
    val normalised = text.replace(Regex("\\s+"), " ").trim()
    val terminator = normalised.indexOf(". ")
    val sentence = if (terminator > 0) normalised.substring(0, terminator + 1) else normalised
    return if (sentence.length <= SHORT_DESCRIPTION_MAX) sentence
    else sentence.take(SHORT_DESCRIPTION_MAX).trimEnd() + "…"
}

class OcToolRegistry {

    private val tools = LinkedHashMap<String, OcTool>()

    fun register(tool: OcTool) {
        tools[tool.name] = tool
    }

    fun registerAll(tools: List<OcTool>) {
        for (tool in tools) register(tool)
    }

    fun get(name: String): OcTool? = tools[name]

    fun names(): List<String> = tools.keys.toList()

    fun all(): List<OcTool> = tools.values.toList()

    /**
     * Schemas for the tools the agent may call.
     *
     * [compact] swaps each description for its opening sentence. The parameter
     * schema is never shortened: a tool the model cannot call correctly is
     * worse than a tool it never knew about.
     */
    fun definitions(visible: Set<String>, ctx: ToolContext, compact: Boolean = false): JSONArray {
        val result = JSONArray()
        for (tool in tools.values) {
            if (visible.contains(tool.name)) {
                result.put(JSONObject().apply {
                    put("name", tool.name)
                    put("description", if (compact) tool.shortDescription(ctx) else tool.description(ctx))
                    put("parameters", tool.parameters())
                })
            }
        }
        return result
    }
}
