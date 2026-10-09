package com.aide.mobilestudio.ai.api

import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.flow.Flow

interface AIProvider {
    val id: String
    val name: String

    sealed class StreamEvent {
        data class Text(val text: String) : StreamEvent()
        data class Reasoning(val text: String) : StreamEvent()
        data class ToolCall(val name: String, val args: JSONObject, val id: String? = null) : StreamEvent()
        data class Plan(val rawPlan: String) : StreamEvent()
        data class Error(val message: String) : StreamEvent()
        object Done : StreamEvent()
    }

    fun generateStream(
        history: JSONArray,
        tools: JSONArray? = null,
        options: JSONObject? = null
    ): Flow<StreamEvent>

    fun getCapabilities(): JSONObject
}
