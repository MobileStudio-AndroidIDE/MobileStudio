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

    fun definitions(visible: Set<String>, ctx: ToolContext): JSONArray {
        val result = JSONArray()
        for (tool in tools.values) {
            if (visible.contains(tool.name)) {
                result.put(JSONObject().apply {
                    put("name", tool.name)
                    put("description", tool.description(ctx))
                    put("parameters", tool.parameters())
                })
            }
        }
        return result
    }
}
