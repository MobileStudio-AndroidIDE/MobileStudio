package com.aide.mobilestudio.ai

import android.content.Context
import android.content.SharedPreferences

class AiSettings(private val context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    companion object {
        const val PREFS = "ai_settings"
        const val KEY_OC_PERMISSIONS = "oc_saved_permissions"
    }

    var apiKey: String?
        get() = prefs.getString("gemini_api_key", null)
        set(value) = prefs.edit().putString("gemini_api_key", value).apply()

    var lastModelId: String?
        get() = prefs.getString("last_model_id", "gemini")
        set(value) = prefs.edit().putString("last_model_id", value).apply()

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
