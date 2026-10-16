package com.aide.mobilestudio.ai

import android.content.Context
import android.content.SharedPreferences
import com.aide.mobilestudio.SecurePrefs

class AiSettings(private val context: Context) {
    // API keys live in an encrypted store; existing plaintext values are migrated once.
    private val prefs: SharedPreferences = SecurePrefs.get(context, SECURE_PREFS).also {
        SecurePrefs.migrateAll(context, PREFS, it)
    }

    companion object {
        const val PREFS = "ai_settings"
        const val SECURE_PREFS = "ai_settings_secure"
        const val KEY_OC_PERMISSIONS = "oc_saved_permissions"
    }

    /** Raw JSON of saved opencode permission rules (read by OcEngine). */
    fun ocPermissionsJson(): String = prefs.getString(KEY_OC_PERMISSIONS, "[]") ?: "[]"

    fun setOcPermissionsJson(json: String) = prefs.edit().putString(KEY_OC_PERMISSIONS, json).apply()

    var apiKey: String?
        get() = prefs.getString("gemini_api_key", null)
        set(value) = prefs.edit().putString("gemini_api_key", value).apply()

    var lastModelId: String?
        get() = prefs.getString("last_model_id", "gemini")
        set(value) = prefs.edit().putString("last_model_id", value).apply()

    /**
     * The last model chosen in the AI workspace (OcEngine).
     *
     * It is a separate preference from [lastModelId] on purpose: that one
     * belongs to the bottom-panel agent, and the two keep entirely different
     * model lists whose ids do not mean the same thing. Sharing one key meant
     * opening either screen silently overwrote the other's choice, which is
     * how the workspace could reopen on a model you had not picked.
     */
    var lastWorkspaceModelId: String?
        get() = prefs.getString("last_workspace_model_id", null)
        set(value) = prefs.edit().putString("last_workspace_model_id", value).apply()

    var activeProvider: String
        get() = prefs.getString("active_provider", "gemini") ?: "gemini"
        set(value) = prefs.edit().putString("active_provider", value).apply()

    var openaiApiKey: String?
        get() = prefs.getString("openai_api_key", null)
        set(value) = prefs.edit().putString("openai_api_key", value).apply()

    var openaiBaseUrl: String
        get() = prefs.getString("openai_base_url", "https://api.openai.com/v1") ?: "https://api.openai.com/v1"
        set(value) = prefs.edit().putString("openai_base_url", value).apply()

    var openaiModel: String
        get() = prefs.getString("openai_model", "gpt-4o") ?: "gpt-4o"
        set(value) = prefs.edit().putString("openai_model", value).apply()

    var anthropicApiKey: String?
        get() = prefs.getString("anthropic_api_key", null)
        set(value) = prefs.edit().putString("anthropic_api_key", value).apply()

    var anthropicModel: String
        get() = prefs.getString("anthropic_model", "claude-sonnet-4-20250514") ?: "claude-sonnet-4-20250514"
        set(value) = prefs.edit().putString("anthropic_model", value).apply()

    var geminiModel: String
        get() = prefs.getString("gemini_model", "gemini-2.0-flash") ?: "gemini-2.0-flash"
        set(value) = prefs.edit().putString("gemini_model", value).apply()

    /** Max output tokens for OpenAI-compatible providers (per-model setting). */
    var openaiMaxTokens: Int
        get() = prefs.getInt("openai_max_tokens", 8192)
        set(value) = prefs.edit().putInt("openai_max_tokens", value.coerceIn(256, 131072)).apply()

    /**
     * Context window of the OpenAI-compatible endpoint, in tokens.
     *
     * Zero means "work it out from the base URL" - see the provider registry.
     * The hosted API's 128k is real, but this client also points at whatever
     * the Base URL says, and a local Ollama or llama.cpp server behind a
     * custom URL typically runs with 4k or 8k. Trusting 128k there meant the
     * engine never compacted, shipped prompts far past what the server could
     * hold - minutes of prefill - and then got refused with "Prompt exceeds
     * context size".
     */
    var openaiContextLimit: Int
        get() = prefs.getInt("openai_context_limit", 0)
        set(value) = prefs.edit().putInt("openai_context_limit", value.coerceIn(0, 2_000_000)).apply()

    fun clearApiKey() = prefs.edit().remove("gemini_api_key").apply()

    fun isCommandAlwaysAllowed(command: String): Boolean {
        val whitelist = prefs.getStringSet("allowed_commands", emptySet()) ?: emptySet()
        return whitelist.contains(command)
    }

    fun allowCommandAlways(command: String) {
        val whitelist = (prefs.getStringSet("allowed_commands", emptySet()) ?: emptySet()).toMutableSet()
        whitelist.add(command)
        prefs.edit().putStringSet("allowed_commands", whitelist).apply()
    }

    var autoApplyChanges: Boolean
        get() = prefs.getBoolean("auto_apply_changes", false)
        set(value) = prefs.edit().putBoolean("auto_apply_changes", value).apply()

    var webSearchEnabled: Boolean
        get() = prefs.getBoolean("web_search_enabled", true)
        set(value) = prefs.edit().putBoolean("web_search_enabled", value).apply()

    var systemPrompt: String
        get() = prefs.getString("system_prompt", "") ?: ""
        set(value) = prefs.edit().putString("system_prompt", value).apply()

    // ── Local AI (llama.cpp runtime) ─────────────────────────────────

    /** GPU layer offload: default 99 (-ngl 99 = all layers on GPU for any model), 0 = CPU, -1 = max. */
    var nGpuLayers: Int
        get() = prefs.getInt("llama_n_gpu_layers", 99)
        set(value) = prefs.edit().putInt("llama_n_gpu_layers", value).apply()

    var nThreads: Int
        get() = prefs.getInt("llama_n_threads", 4)
        set(value) = prefs.edit().putInt("llama_n_threads", value.coerceIn(1, 8)).apply()

    var contextLength: Int
        get() = prefs.getInt("llama_context", 4096)
        set(value) = prefs.edit().putInt("llama_context", value.coerceIn(512, 16384)).apply()
}
