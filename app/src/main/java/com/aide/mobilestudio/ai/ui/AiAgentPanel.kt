package com.aide.mobilestudio.ai.ui

import android.content.Context
import android.content.Intent
import android.provider.OpenableColumns
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.VibrationEffect
import android.os.Vibrator
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.text.style.TypefaceSpan
import android.util.Base64
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import com.aide.mobilestudio.R
import com.aide.mobilestudio.StudioUi
import com.aide.mobilestudio.ai.opencode.agent.OcEngine as AiAgentEngine
import com.aide.mobilestudio.ai.api.AIProvider
import com.aide.mobilestudio.ai.agent.ModelManager
import com.aide.mobilestudio.ai.agent.HardwareManager
import com.aide.mobilestudio.ai.agent.ToolRegistry
import com.aide.mobilestudio.ai.opencode.command.OcCommandInfo
import com.aide.mobilestudio.ai.opencode.session.OcMessage
import com.aide.mobilestudio.ai.opencode.session.OcSessionStore
import com.aide.mobilestudio.ai.agent.CheckpointOutcome
import com.aide.mobilestudio.ai.agent.ProjectFiles
import com.aide.mobilestudio.ai.agent.PromptContextBadge
import com.aide.mobilestudio.ai.agent.StatusLine
import com.aide.mobilestudio.ai.command.SlashCommand
import com.aide.mobilestudio.ai.command.SlashCommandAction
import com.aide.mobilestudio.ai.command.SlashCommandRegistry
import com.aide.mobilestudio.ai.command.SlashInvocation
import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.*

class AiAgentPanel(context: Context) : LinearLayout(context) {
    private val settings = com.aide.mobilestudio.ai.AiSettings(context)
    val engine = AiAgentEngine(context)
    private val modelManager = ModelManager(context)
    private val conversationStorage = com.aide.mobilestudio.ai.agent.AiConversationStorage(context)
    private val hardwareManager = HardwareManager(context)

    private val outputContainer: LinearLayout
    private val scrollView: ScrollView
    private val approvalContainer: LinearLayout
    private val inputField: EditText
    private val sendButton: ImageButton
    private lateinit var attachButton: ImageButton
    private var pendingImageBase64: String? = null
    private var pendingImagePreview: TextView? = null
    private var slashPopup: ListPopupWindow? = null
    private var slashMatches: List<OcCommandInfo> = emptyList()
    private var slashCommandCache: List<OcCommandInfo>? = null
    private val slashRegistry = SlashCommandRegistry()
    private var slashLocalMatches: List<SlashCommand> = emptyList()
    private var mentionPopup: ListPopupWindow? = null
    private var mentionFiles: List<String> = emptyList()
    private var mentionFilesRoot: File? = null
    private var mentionFilesLoading = false
    private var activeMentionToken: String? = null
    private lateinit var modeSelector: Spinner
    private lateinit var contextBadge: TextView
    private val thinkingText: TextView
    private val thinkingCard: LinearLayout
    private val thinkingHeader: TextView
    private val thinkingBody: TextView
    private val thinkingLiveBuf = StringBuilder()
    private var thinkingExpanded = false
    private var thinkingLiveDone = true
    private var thinkingFlushPending = false
    private val thinkingFlushRunnable = Runnable { renderThinkingCard() }

    // Streaming assistant message: full markdown rebuild only on fence
    // transitions, otherwise the delta is appended to the growing TextView.
    private var streamText = ""
    private var streamView: TextView? = null
    private val planContainer: LinearLayout
    private val planToggle: TextView
    private val modelSelector: Spinner
    private val runtimeSelector: Spinner
    private val tokenCounter: TextView
    private var aiStatusText: TextView? = null

    // Command output streaming views
    private val commandOutputContainer: LinearLayout
    private val commandOutputScroll: ScrollView
    private val commandOutputText: TextView
    private val commandOutputToggle: TextView

    private var thinkingDots = 0
    private var thinkingStatus = "Thinking"
    private val thinkingHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val thinkingRunnable = object : Runnable {
        override fun run() {
            thinkingDots = (thinkingDots + 1) % 4
            val dots = ".".repeat(thinkingDots)
            thinkingText.text = "🧠 $thinkingStatus$dots"
            thinkingHandler.postDelayed(this, 500)
        }
    }

    /** Accumulated reasoning of the current turn arrived from [AiAgentEngine.onReasoning]. */
    private fun onReasoningLive(text: String) {
        thinkingLiveDone = false
        thinkingLiveBuf.setLength(0)
        thinkingLiveBuf.append(text)
        thinkingCard.visibility = View.VISIBLE
        // Throttle UI writes: reasoning streams token by token (frame-coalesced).
        if (!thinkingFlushPending) {
            thinkingFlushPending = true
            thinkingHandler.postDelayed(thinkingFlushRunnable, 150)
        }
    }

    private fun onReasoningFinished() {
        thinkingHandler.removeCallbacks(thinkingFlushRunnable)
        thinkingFlushPending = false
        thinkingLiveDone = true
        renderThinkingCard()
    }

    private fun renderThinkingCard() {
        thinkingFlushPending = false
        val full = thinkingLiveBuf.toString()
        if (full.isEmpty()) {
            thinkingCard.visibility = View.GONE
            return
        }
        thinkingCard.visibility = View.VISIBLE
        val capped = if (full.length > THINKING_MAX_CHARS) {
            "…(earlier thinking truncated)\n" + full.takeLast(THINKING_MAX_CHARS)
        } else {
            full
        }
        thinkingBody.text = if (thinkingExpanded) capped else thinkingTail(capped)
        thinkingHeader.text = when {
            thinkingExpanded -> "🧠 Thinking ▴ (tap to collapse)"
            thinkingLiveDone -> "🧠 Thinking ▾ done · tap to expand"
            else -> "🧠 Thinking ▾ streaming · tap to expand"
        }
    }

    private fun thinkingTail(s: String): String {
        if (s.length <= THINKING_TAIL_CHARS) return s
        val tail = s.substring(s.length - THINKING_TAIL_CHARS)
        val nl = tail.indexOf('\n')
        return if (nl in 0 until tail.length - 1) "…\n" + tail.substring(nl + 1) else "…\n$tail"
    }

    /**
     * Stream one command-output line. Re-attaches the panel when a previous
     * completion cleared it (removeAllViews detached the scroll), and reveals
     * the Show more/less toggle once the output reaches [OUTPUT_CLAMP_LINES].
     */
    private fun appendCommandOutput(line: String) {
        val restart = commandOutputText.text.isEmpty()
        if (restart) {
            if (commandOutputToggle.parent == null) commandOutputContainer.addView(commandOutputToggle)
            commandOutputToggle.text = "Show more ⌵"
            commandOutputToggle.tag = false
            commandOutputToggle.visibility = View.GONE
        }
        if (commandOutputScroll.parent == null) {
            commandOutputContainer.addView(commandOutputScroll)
        }
        commandOutputScroll.visibility = View.VISIBLE
        commandOutputText.append(line + "\n")
        if (commandOutputText.lineCount >= OUTPUT_CLAMP_LINES) {
            commandOutputToggle.visibility = View.VISIBLE
        }
        commandOutputScroll.post { commandOutputScroll.fullScroll(View.FOCUS_DOWN) }
    }

    companion object {
        private const val ID_CONTENT = 1001
        private const val ID_DIFF = 2001
        private const val ID_CODE_BLOCK = 3001
        private const val ID_CMD_OUTPUT = 4001
        private const val ID_SCROLL = 5001
        private const val ID_SHOW_MORE = 5002
        private const val MESSAGE_CLAMP_LINES = 10
        private const val OUTPUT_CLAMP_LINES = 10
        private const val THINKING_MAX_CHARS = 8000
        private const val THINKING_TAIL_CHARS = 700
        private const val MAX_PROMPT_CHARS = 5000
    }

    init {
        // Local AI runtime: register app context early (context-free access)
        com.aide.mobilestudio.ai.LlamaRuntimeManager.initAppContext(context)
        syncRuntimeOptions()
        orientation = VERTICAL
        layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT)
        setBackgroundColor(StudioUi.palette.bg)

        // Header
        val headerRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(StudioUi.palette.surface)
            setPadding(dp(12), dp(4), dp(4), dp(4))
        }

        modelSelector = Spinner(context).apply {
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
            background = null
        }

        // Agent mode selector (build/plan/explore)
        modeSelector = Spinner(context).apply {
            id = View.generateViewId()
            val modes = listOf("Build (Full Access)", "Plan (Read-Only)", "Explore (Fast)")
            adapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, modes)
            setSelection(0)
            onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    val mode = when(position) { 1 -> "plan"; 2 -> "explore"; else -> "build" }
                    engine.setAgentMode(mode)
                    addMessage("system", "Agent mode: ${modes[position]}")
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }

        runtimeSelector = Spinner(context).apply {
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT)
            background = null
            visibility = View.GONE
        }

        val newConvBtn = ImageButton(context).apply {
            setImageResource(android.R.drawable.ic_menu_add)
            background = null
            setOnClickListener { startNewConversation() }
        }
        val historyBtn = ImageButton(context).apply {
            setImageResource(android.R.drawable.ic_menu_recent_history)
            background = null
            setOnClickListener { showHistoryDialog() }
        }
        val settingsBtn = ImageButton(context).apply {
            setImageResource(android.R.drawable.ic_menu_preferences)
            background = null
            setOnClickListener { showSettingsDialog() }
        }
        headerRow.addView(modelSelector)
        headerRow.addView(runtimeSelector)
        headerRow.addView(newConvBtn)
        headerRow.addView(historyBtn)
        headerRow.addView(settingsBtn)
        addView(headerRow)

        // Local AI status row: 현재 로드된 모델 + backend/GPU layers (실측값만 표시)
        aiStatusText = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#8B949E"))
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setPadding(dp(12), dp(2), dp(12), dp(2))
            visibility = GONE
            layoutParams = LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        }
        addView(aiStatusText)

        // Quick Actions bar
        val quickActionsRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setBackgroundColor(StudioUi.palette.surface)
            layoutParams = LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
        }
        val quickActions = listOf(
            "Explain" to "\u2139\ufe0f Explain this code",
            "Fix" to "\ud83d\udd27 Fix the errors in this file",
            "Refactor" to "\ud83d\udd04 Refactor this code",
            "Review" to "\ud83d\udd0d Review this code for bugs, security issues, and maintainability. Give prioritized findings.",
            "Test" to "\ud83e\uddea Write unit tests for this code",
            "Doc" to "\ud83d\udcdd Write documentation for this code",
            "Optimize" to "\u26a1 Optimize this code for performance"
        )
        for ((label, prompt) in quickActions) {
            val btn = TextView(context).apply {
                text = label
                textSize = 10f
                setTextColor(StudioUi.palette.accent)
                setPadding(dp(6), dp(3), dp(6), dp(3))
                background = StudioUi.rounded(context, fill = "#21252B", radiusDp = 12)
                layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
                    marginEnd = dp(4)
                }
                isClickable = true
                setOnClickListener {
                    inputField.setText(prompt)
                    sendMessage()
                }
            }
            quickActionsRow.addView(btn)
        }
        addView(quickActionsRow)

        // Plan Area
        planToggle = TextView(context).apply {
            this.text = "Show Plan ⌵"
            textSize = 12f
            setTextColor(StudioUi.palette.accent)
            setPadding(dp(12), dp(4), dp(12), dp(4))
            visibility = View.GONE
            setOnClickListener {
                planContainer.visibility = if (planContainer.visibility == View.VISIBLE) View.GONE else View.VISIBLE
                this.text = if (planContainer.visibility == View.VISIBLE) "Hide Plan ▴" else "Show Plan ⌵"
            }
        }
        planContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(24), dp(4), dp(24), dp(8))
            visibility = View.GONE
            setBackgroundColor(StudioUi.palette.surface)
        }
        addView(planToggle)
        addView(planContainer)

        // Status + Token counter row
        val statusRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        thinkingText = TextView(context).apply {
            this.text = "AI Agent Ready"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            setPadding(dp(12), dp(4), dp(12), dp(4))
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
        }
        tokenCounter = TextView(context).apply {
            this.text = "0 tokens"
            textSize = 10f
            setTextColor(StudioUi.palette.muted)
            setPadding(dp(4), dp(4), dp(12), dp(4))
        }
        statusRow.addView(thinkingText)
        statusRow.addView(tokenCounter)
        addView(statusRow)

        // Live reasoning card: streams the model's thinking in real time.
        // Pinned above the message list; tap to expand/collapse.
        thinkingHeader = TextView(context).apply {
            text = "🧠 Thinking"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(12), dp(6), dp(12), 0)
        }
        thinkingBody = TextView(context).apply {
            textSize = 11f
            setTextColor(StudioUi.palette.muted)
            setLineSpacing(2f, 1f)
            setPadding(dp(12), dp(2), dp(12), dp(6))
        }
        thinkingCard = LinearLayout(context).apply {
            orientation = VERTICAL
            setBackgroundColor(StudioUi.palette.surface)
            visibility = View.GONE
            addView(thinkingHeader)
            addView(thinkingBody)
            setOnClickListener {
                thinkingExpanded = !thinkingExpanded
                renderThinkingCard()
            }
        }
        addView(thinkingCard)

        // Messages
        scrollView = ScrollView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, 0, 1f)
            isFillViewport = true
        }
        outputContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(12), 0, dp(12), dp(8))
        }
        scrollView.addView(outputContainer)
        addView(scrollView)

        // Command output panel (10-line preview + Show more/less)
        commandOutputToggle = TextView(context).apply {
            this.text = "Show more ⌵"
            textSize = 11f
            setTextColor(StudioUi.palette.muted)
            setPadding(dp(12), dp(4), dp(12), dp(2))
            visibility = View.GONE
            isClickable = true
            setOnClickListener {
                val expanded = tag == true
                commandOutputScroll.layoutParams = commandOutputScroll.layoutParams.apply {
                    height = if (expanded) dp(150) else resources.displayMetrics.heightPixels * 2 / 5
                }
                this.text = if (expanded) "Show more ⌵" else "Show less ▴"
                tag = !expanded
            }
        }
        commandOutputScroll = ScrollView(context).apply {
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, dp(150))
            visibility = View.GONE
            setBackgroundColor(StudioUi.palette.bg)
        }
        commandOutputText = TextView(context).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setTextColor(StudioUi.palette.text)
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setTextIsSelectable(true)
        }
        commandOutputScroll.addView(commandOutputText)
        commandOutputContainer = LinearLayout(context).apply {
            orientation = VERTICAL
        }
        commandOutputContainer.addView(commandOutputToggle)
        commandOutputContainer.addView(commandOutputScroll)
        addView(commandOutputContainer)

        // Diff viewer
        val diffContainer = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
            id = View.generateViewId()
        }
        val diffHeader = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setBackgroundColor(Color.parseColor("#1E1E2E"))
        }
        val diffTitle = TextView(context).apply {
            text = "Diffs"
            setTextColor(Color.parseColor("#89B4FA"))
            textSize = 12f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        val diffCloseBtn = TextView(context).apply {
            text = "X"
            setTextColor(Color.parseColor("#F38BA8"))
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            setPadding(dp(8), 0, dp(8), 0)
            setOnClickListener { diffContainer.visibility = View.GONE }
        }
        diffHeader.addView(diffTitle)
        diffHeader.addView(diffCloseBtn)
        diffContainer.addView(diffHeader)
        val diffScroll = ScrollView(context).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(200)) }
        val diffText = TextView(context).apply {
            setTextColor(Color.parseColor("#CDD6F4"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setPadding(dp(8), dp(4), dp(8), dp(4))
            setTextIsSelectable(true)
        }
        diffScroll.addView(diffText)
        diffContainer.addView(diffScroll)
        addView(diffContainer)

        // Approval
        approvalContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            background = StudioUi.rounded(context, fill = "#1C212B", stroke = "#F05C5C", radiusDp = 8)
            setPadding(dp(12), dp(12), dp(12), dp(12))
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                setMargins(dp(12), dp(8), dp(12), dp(8))
            }
            visibility = View.GONE
        }
        addView(approvalContainer)

        // Context badge: compact "what will be sent" line above the input
        // (open editor file · @mention count · rough token size).
        contextBadge = TextView(context).apply {
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(StudioUi.palette.muted)
            setPadding(dp(12), dp(2), dp(12), dp(2))
            visibility = GONE
        }
        addView(contextBadge)

        // Input
        val inputRow = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(8), dp(8), dp(8))
            setBackgroundColor(StudioUi.palette.surface)
        }
        // Attach image button
        attachButton = ImageButton(context).apply {
            setImageResource(android.R.drawable.ic_menu_gallery)
            background = null
            setColorFilter(StudioUi.palette.muted)
            setPadding(dp(8), dp(8), dp(8), dp(8))
            setOnClickListener { pickImage() }
            contentDescription = "Attach image"
        }
        inputField = EditText(context).apply {
            hint = "Ask AI Agent..."
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.muted)
            background = StudioUi.rounded(context, fill = String.format("#%06X", 0xFFFFFF and StudioUi.palette.bg), stroke = String.format("#%06X", 0xFFFFFF and StudioUi.palette.border), radiusDp = StudioUi.RadiusXl)
            setPadding(dp(12), dp(8), dp(12), dp(8))
            // Bounded: 7 visible lines with internal scrolling instead of
            // growing forever; pastes capped at MAX_PROMPT_CHARS.
            inputType = android.text.InputType.TYPE_CLASS_TEXT or
                android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE
            maxLines = 7
            minLines = 1
            gravity = Gravity.TOP or Gravity.START
            filters = arrayOf(android.text.InputFilter.LengthFilter(MAX_PROMPT_CHARS))
            layoutParams = LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f)
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                    sendMessage()
                    true
                } else false
            }
            addTextChangedListener(object : android.text.TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = Unit
                override fun afterTextChanged(s: android.text.Editable?) {
                    val text = s?.toString() ?: ""
                    updateSlashSuggestions(text)
                    updateMentionSuggestions(text)
                    updateContextBadge(text)
                }
            })
        }
        sendButton = ImageButton(context).apply {
            setImageResource(android.R.drawable.ic_menu_send)
            background = null
            setColorFilter(StudioUi.palette.accent)
            setOnClickListener {
                if (engineState == AiAgentEngine.AgentState.GENERATING) engine.stop()
                else sendMessage()
            }
        }
        inputRow.addView(attachButton)
        inputRow.addView(inputField)
        inputRow.addView(sendButton)
        addView(inputRow)

        // Set up command output streaming
        engine.listener = null // Will be set in setupEngine
        ToolRegistry.commandOutputListener = { _, line ->
            post { appendCommandOutput(line) }
        }

        // Question tool: AI asks the user directly (blocks the tool thread until answered)
        ToolRegistry.questionHandler = { question, options ->
            val latch = java.util.concurrent.CountDownLatch(1)
            var answer = "No answer provided."
            post {
                try {
                    val builder = android.app.AlertDialog.Builder(context)
                    builder.setTitle("AI Question")
                    builder.setMessage(question)
                    if (options.isNotEmpty()) {
                        builder.setItems(options.toTypedArray()) { _, which ->
                            answer = options[which]
                            latch.countDown()
                        }
                    } else {
                        val input = android.widget.EditText(context)
                        input.hint = "Type your answer…"
                        builder.setView(input)
                        builder.setPositiveButton("Answer") { _, _ ->
                            answer = input.text.toString().ifBlank { "No answer provided." }
                            latch.countDown()
                        }
                    }
                    builder.setOnCancelListener { latch.countDown() }
                    builder.show()
                } catch (e: Exception) {
                    latch.countDown()
                }
            }
            latch.await(300, java.util.concurrent.TimeUnit.SECONDS)
            answer
        }

        post {
            updateModelList()
            setupEngine()
            updateTokenCount()
        }
    }

    private var engineState = AiAgentEngine.AgentState.IDLE

    private fun updateModelList() {
        val models = mutableListOf<String>()
        val providerLabels = mutableListOf<String>()

        // Cloud providers
        if (settings.apiKey?.isNotEmpty() == true) {
            models.add("gemini:${settings.geminiModel}")
            providerLabels.add("Gemini: ${settings.geminiModel}")
        }
        if (settings.openaiApiKey?.isNotEmpty() == true) {
            models.add("openai:${settings.openaiModel}")
            providerLabels.add("OpenAI: ${settings.openaiModel}")
        }
        if (settings.anthropicApiKey?.isNotEmpty() == true) {
            models.add("anthropic:${settings.anthropicModel}")
            providerLabels.add("Claude: ${settings.anthropicModel}")
        }

        // If no providers configured, show placeholder
        if (providerLabels.isEmpty()) {
            providerLabels.add("No API key configured - tap Settings")
            models.add("none")
        }

        // Local models
        val remoteModels = modelManager.listRemoteModels()
        val localModels = modelManager.listModels().filter { local -> remoteModels.none { it.id == local.id } }
        for (m in remoteModels) {
            models.add("local:${m.id}")
            providerLabels.add("Local: ${m.name}")
        }
        for (m in localModels) {
            models.add("local:${m.id}")
            providerLabels.add("Imported: ${m.name}")
        }

        val modelAdapter = ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, providerLabels)
        modelSelector.adapter = modelAdapter

        // Restore last selection
        val lastId = settings.lastModelId ?: settings.activeProvider
        val idx = models.indexOfFirst { it == lastId || it.startsWith("$lastId:") }
        if (idx >= 0) modelSelector.setSelection(idx)

        modelSelector.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, pos: Int, p3: Long) {
                val selected = models[pos]
                if (selected == "none") {
                    showSettingsDialog()
                    return
                }
                val parts = selected.split(":", limit = 2)
                val providerType = parts[0]
                val modelId = parts.getOrNull(1) ?: ""

                if (providerType != "local") aiStatusText?.visibility = GONE

                settings.lastModelId = selected
                settings.activeProvider = providerType

                when (providerType) {
                    "gemini" -> {
                        if (settings.apiKey != null) {
                            engine.setModel("gemini/${modelId.ifEmpty { settings.geminiModel }}")
                        } else {
                            showSettingsDialog()
                        }
                    }
                    "openai" -> {
                        if (settings.openaiApiKey != null) {
                            engine.setModel("openai/${modelId.ifEmpty { settings.openaiModel }}")
                        } else {
                            showSettingsDialog()
                        }
                    }
                    "anthropic" -> {
                        if (settings.anthropicApiKey != null) {
                            engine.setModel("anthropic/${modelId.ifEmpty { settings.anthropicModel }}")
                        } else {
                            showSettingsDialog()
                        }
                    }
                    "local" -> {
                        val modelInfo = (remoteModels + localModels).find { it.id == modelId }
                        if (modelInfo != null) {
                            if (modelManager.isModelDownloaded(modelInfo)) {
                                engine.setModel("local/$modelId")
                                aiStatusText?.let { st ->
                                    post {
                                        st.visibility = VISIBLE
                                        st.text = "Local: ${modelInfo.name}"
                                    }
                                }
                                Toast.makeText(context, "Using local model: ${modelInfo.name}", Toast.LENGTH_SHORT).show()
                            } else {
                                showDownloadDialog(modelInfo)
                            }
                        }
                    }
                }
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        setupEngine()
    }

    private fun showDownloadDialog(model: ModelManager.ModelInfo) {
        val sizeGbNum = model.sizeBytes.toDouble() / (1024 * 1024 * 1024)
        val sizeGb = String.format(Locale.US, "%.1f", sizeGbNum)
        android.app.AlertDialog.Builder(context)
            .setTitle("Download Model")
            .setMessage("This model (${model.name}) needs to be downloaded ($sizeGb GB).\n\nRequires at least ${String.format(Locale.US, "%.1f", sizeGbNum + 3)} GB free space.\n\nContinue?")
            .setPositiveButton("Download") { _, _ -> startDownload(model) }
            .setNegativeButton("Cancel") { _, _ -> modelSelector.setSelection(0) }
            .show()
    }

    private fun startDownload(model: ModelManager.ModelInfo) {
        val progressDialog = android.app.ProgressDialog(context).apply {
            setTitle("Downloading ${model.name}")
            setMessage("Preparing...")
            setProgressStyle(android.app.ProgressDialog.STYLE_HORIZONTAL)
            max = 100
            setCancelable(false)
            show()
        }

        modelManager.downloadModel(model, { progress ->
            post {
                progressDialog.progress = (progress * 100).toInt()
                progressDialog.setMessage("Downloading... ${(progress * 100).toInt()}%")
            }
        }, { success, error ->
            post {
                progressDialog.dismiss()
                if (success) {
                    Toast.makeText(context, "Download complete!", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    modelSelector.setSelection(modelManager.listRemoteModels().indexOfFirst { it.id == model.id } + 1)
                } else {
                    android.app.AlertDialog.Builder(context)
                        .setTitle("Download Failed")
                        .setMessage(error ?: "Unknown error")
                        .setPositiveButton("OK") { _, _ -> modelSelector.setSelection(0) }
                        .show()
                }
            }
        })
    }

    private fun setupEngine() {
        engine.listener = object : AiAgentEngine.Listener {
            override fun onMessage(text: String, isComplete: Boolean) {
                post {
                    updateLastMessage(text)
                    if (isComplete) {
                        finalizeLastMessage()
                        updateTokenCount()
                        // Clear command output after AI finishes
                        if (commandOutputText.text.isNotEmpty()) {
                            postDelayed({
                                commandOutputContainer.removeAllViews()
                                commandOutputScroll.visibility = View.GONE
                                commandOutputText.text = ""
                            }, 3000)
                        }
                    }
                }
            }

            override fun onThinking(status: String) {
                post {
                    thinkingStatus = status.removeSuffix("...").ifBlank { "Thinking" }
                    thinkingHandler.removeCallbacks(thinkingRunnable)
                    thinkingHandler.post(thinkingRunnable)
                }
            }

            override fun onReasoning(text: String) {
                post { onReasoningLive(text) }
            }

            override fun onReasoningDone() {
                post { onReasoningFinished() }
            }

            override fun onToolCall(name: String, args: JSONObject, risk: AiAgentEngine.RiskLevel, onApproval: (Boolean, Boolean) -> Unit) {
                post {
                    onReasoningFinished()
                    thinkingHandler.removeCallbacks(thinkingRunnable)
                    showApprovalUI(name, args, risk, onApproval)
                    vibrateForApproval()
                }
            }

            override fun onToolStart(name: String, args: JSONObject) {
                post {
                    onReasoningFinished()
                    thinkingHandler.removeCallbacks(thinkingRunnable)
                    thinkingText.text = "⚙️ Running: $name"
                    addTimelineToolItem(name, args)
                }
            }

            override fun onToolEnd(name: String, result: String) {
                post {
                    thinkingText.text = "⚙️ Completed: $name"
                }
            }

            override fun onPlanUpdate(rawPlan: String) {
                post { updatePlanView(rawPlan) }
            }

            override fun onStateChange(state: AiAgentEngine.AgentState) {
                post {
                    engineState = state
                    when (state) {
                        AiAgentEngine.AgentState.GENERATING -> {
                            thinkingStatus = "Writing response"
                            sendButton.setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
                            thinkingText.visibility = View.VISIBLE
                            thinkingHandler.post(thinkingRunnable)
                        }
                        AiAgentEngine.AgentState.THINKING -> {
                            thinkingStatus = "Preparing local model"
                            sendButton.setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
                            thinkingText.visibility = View.VISIBLE
                            thinkingHandler.post(thinkingRunnable)
                        }
                        else -> {
                            sendButton.setImageResource(android.R.drawable.ic_menu_send)
                            thinkingHandler.removeCallbacks(thinkingRunnable)
                            if (state == AiAgentEngine.AgentState.IDLE) {
                                thinkingText.text = "AI Agent Ready"
                            }
                        }
                    }
                }
            }

            override fun onError(message: String) {
                post {
                    thinkingHandler.removeCallbacks(thinkingRunnable)
                    thinkingText.text = "AI Agent Ready"
                    addMessage("system", "Error: $message")
                }
            }

            override fun onCommandOutput(commandId: String, line: String) {
                post { appendCommandOutput(line) }
            }

            override fun onNotice(text: String) {
                post { addMessage("system", text) }
            }

            override fun onTodos(todos: JSONArray) {
                post {
                    if (todos.length() == 0) return@post
                    var done = 0
                    for (i in 0 until todos.length()) {
                        val status = todos.optJSONObject(i)?.optString("status", "") ?: ""
                        if (status == "completed" || status == "cancelled") done++
                    }
                    thinkingStatus = "Tasks $done/${todos.length()} done"
                    thinkingHandler.removeCallbacks(thinkingRunnable)
                    thinkingHandler.post(thinkingRunnable)
                }
            }

            override fun onAskQuestion(questions: JSONArray, onAnswer: (JSONArray?) -> Unit) {
                post { showQuestionDialog(questions, onAnswer) }
            }
        }
    }

    private fun showQuestionDialog(questions: JSONArray, onAnswer: (JSONArray?) -> Unit) {
        var answered = false
        val answerOnce: (JSONArray?) -> Unit = { result ->
            if (!answered) {
                answered = true
                onAnswer(result)
            }
        }
        val answerProviders = mutableListOf<() -> String?>()

        val scroll = ScrollView(context)
        val column = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        scroll.addView(column)

        for (i in 0 until questions.length()) {
            val q = questions.optJSONObject(i) ?: continue
            val header = q.optString("header", "Question").ifBlank { "Question" }
            val text = q.optString("question", "")
            val multiple = q.optBoolean("multiple", false)
            val options = q.optJSONArray("options")

            column.addView(TextView(context).apply {
                this.text = header.uppercase(Locale.getDefault())
                textSize = 11f
                setTypeface(null, Typeface.BOLD)
                setTextColor(StudioUi.palette.accent)
                setPadding(0, dp(16), 0, dp(4))
            })
            if (text.isNotEmpty()) {
                column.addView(TextView(context).apply {
                    this.text = text
                    textSize = 15f
                    setTextColor(StudioUi.palette.text)
                    setPadding(0, dp(2), 0, dp(8))
                })
            }

            if (options == null || options.length() == 0) {
                val freeInput = EditText(context).apply {
                    hint = "Type your answer"
                    setTextColor(StudioUi.palette.text)
                    setHintTextColor(StudioUi.palette.muted)
                }
                column.addView(freeInput)
                answerProviders.add { freeInput.text.toString().trim().ifEmpty { null } }
                continue
            }

            if (multiple) {
                val selected = mutableListOf<Pair<CheckBox, String>>()
                for (j in 0 until options.length()) {
                    val opt = options.optJSONObject(j) ?: continue
                    val label = opt.optString("label").ifBlank { "Option ${j + 1}" }
                    val desc = opt.optString("description")
                    val box = CheckBox(context).apply {
                        this.text = label
                        setTextColor(StudioUi.palette.text)
                        setPadding(0, 0, 0, 0)
                    }
                    column.addView(box)
                    selected.add(box to label)
                    if (desc.isNotEmpty()) {
                        column.addView(TextView(context).apply {
                            this.text = "    $desc"
                            textSize = 12f
                            setTextColor(StudioUi.palette.muted)
                            setPadding(dp(36), 0, 0, dp(4))
                        })
                    }
                }
                answerProviders.add {
                    val labels = selected.filter { it.first.isChecked }.map { it.second }
                    labels.joinToString(", ").ifEmpty { null }
                }
            } else {
                val optionsBox = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
                val radios = mutableListOf<Pair<RadioButton, String>>()
                for (j in 0 until options.length()) {
                    val opt = options.optJSONObject(j) ?: continue
                    val label = opt.optString("label").ifBlank { "Option ${j + 1}" }
                    val desc = opt.optString("description")
                    val radio = RadioButton(context).apply {
                        this.text = label
                        tag = label
                        setTextColor(StudioUi.palette.text)
                    }
                    radio.setOnCheckedChangeListener { button, isChecked ->
                        if (isChecked) {
                            for ((other, _) in radios) {
                                if (other !== button) other.isChecked = false
                            }
                        }
                    }
                    optionsBox.addView(radio)
                    radios.add(radio to label)
                    if (desc.isNotEmpty()) {
                        optionsBox.addView(TextView(context).apply {
                            this.text = "    $desc"
                            textSize = 12f
                            setTextColor(StudioUi.palette.muted)
                            setPadding(dp(36), 0, 0, dp(2))
                        })
                    }
                }
                column.addView(optionsBox)
                answerProviders.add { radios.firstOrNull { it.first.isChecked }?.second }
            }
        }

        android.app.AlertDialog.Builder(context)
            .setTitle("AI Question")
            .setView(scroll)
            .setCancelable(true)
            .setPositiveButton("Answer") { _, _ ->
                val result = JSONArray()
                for (provider in answerProviders) result.put(provider() ?: "")
                answerOnce(result)
            }
            .setNegativeButton("Skip") { _, _ -> answerOnce(null) }
            .setOnCancelListener { answerOnce(null) }
            .show()
    }

    private fun updatePlanView(rawPlan: String) {
        planToggle.visibility = View.VISIBLE
        planContainer.removeAllViews()
        val lines = rawPlan.trim().lines()
        lines.forEach { line ->
            val clean = line.trim().removePrefix("-").trim()
            if (clean.isEmpty()) return@forEach
            val statusIcon = if (clean.contains("☑")) "☑️" else if (clean.contains("⏺")) "⏺️" else "✖️"
            val text = clean.replace("☑", "").replace("⏺", "").replace("✖", "").trim()

            planContainer.addView(LinearLayout(context).apply {
                orientation = HORIZONTAL
                addView(TextView(context).apply { this.text = statusIcon; setPadding(0, 0, dp(8), 0) })
                addView(TextView(context).apply {
                    this.text = text
                    textSize = 12f
                    setTextColor(Color.parseColor("#D1D5DB"))
                })
            })
        }
    }

    private fun updateSlashSuggestions(text: String) {
        if (!text.startsWith("/") || text.contains(' ')) {
            hideSlashSuggestions()
            return
        }
        val query = text.substring(1)
        slashLocalMatches = slashRegistry.filter(query)
        val commands = slashCommandCache
            ?: runCatching { engine.listCommands() }.getOrDefault(emptyList()).also { slashCommandCache = it }
        slashMatches = commands.filter { cmd ->
            cmd.name.startsWith(query, ignoreCase = true) ||
                (cmd.description ?: "").contains(query, ignoreCase = true)
        }
        if (slashLocalMatches.isEmpty() && slashMatches.isEmpty()) {
            hideSlashSuggestions()
            return
        }
        val labels = ArrayList<String>(slashLocalMatches.size + slashMatches.size)
        slashLocalMatches.forEach { labels.add("/${it.name}  —  ${it.description}") }
        slashMatches.forEach { cmd ->
            val desc = cmd.description
            labels.add(if (desc.isNullOrBlank()) "/${cmd.name}" else "/${cmd.name}  —  $desc")
        }
        val popup = slashPopup ?: ListPopupWindow(context).apply {
            anchorView = inputField
            isModal = false
            setOnItemClickListener { _, _, position, _ ->
                hideSlashSuggestions()
                val local = slashLocalMatches.getOrNull(position)
                if (local != null) {
                    inputField.setText("")
                    runSlashCommand(local)
                } else {
                    val cmd = slashMatches.getOrNull(position - slashLocalMatches.size)
                        ?: return@setOnItemClickListener
                    inputField.setText("/${cmd.name} ")
                    inputField.setSelection(inputField.text.length)
                }
            }
            slashPopup = this
        }
        popup.setContentWidth(inputField.width.coerceAtLeast(dp(220)))
        popup.setAdapter(ArrayAdapter(context, android.R.layout.simple_list_item_1, labels))
        popup.show()
    }

    private fun hideSlashSuggestions() {
        slashPopup?.dismiss()
    }

    // ---------------------------------------------------- @file mention picker

    private fun updateMentionSuggestions(text: String) {
        val token = mentionTokenAt(text) ?: run {
            mentionPopup?.dismiss()
            return
        }
        ensureMentionFiles {
            val matches = ProjectFiles.filter(mentionFiles, token.removePrefix("@"))
            if (matches.isEmpty()) {
                mentionPopup?.dismiss()
                return@ensureMentionFiles
            }
            activeMentionToken = token
            val labels = matches.map { "@$it" }
            val popup = mentionPopup ?: ListPopupWindow(context).apply {
                anchorView = inputField
                isModal = false
                setOnItemClickListener { _, _, position, _ ->
                    mentionPopup?.dismiss()
                    val relative = matches.getOrNull(position) ?: return@setOnItemClickListener
                    val current = activeMentionToken ?: return@setOnItemClickListener
                    val editable = inputField.text ?: return@setOnItemClickListener
                    val start = (editable.length - current.length).coerceAtLeast(0)
                    editable.replace(start, editable.length, "@$relative ")
                    inputField.setSelection(editable.length)
                    activeMentionToken = null
                }
                mentionPopup = this
            }
            popup.setContentWidth(inputField.width.coerceAtLeast(dp(220)))
            popup.setAdapter(ArrayAdapter(context, android.R.layout.simple_list_item_1, labels))
            popup.show()
        }
    }

    /** The "@..." token currently being typed, or null. */
    private fun mentionTokenAt(text: String): String? {
        if (text.isEmpty()) return null
        val start = text.lastIndexOf('@')
        if (start < 0) return null
        if (start > 0) {
            val prev = text[start - 1]
            if (prev.isLetterOrDigit() || prev == '_' || prev == '.' || prev == '/') return null
        }
        val token = text.substring(start)
        return if (token.contains(' ') || token.contains('\n')) null else token
    }

    private fun ensureMentionFiles(onReady: () -> Unit) {
        val root = engine.workspaceRoot()
        if (mentionFilesRoot == root && mentionFiles.isNotEmpty()) {
            onReady()
            return
        }
        if (mentionFilesLoading) return
        mentionFilesLoading = true
        Thread {
            val files = runCatching { ProjectFiles.list(root) }.getOrDefault(emptyList())
            post {
                mentionFiles = files
                mentionFilesRoot = root
                mentionFilesLoading = false
                onReady()
            }
        }.start()
    }

    private fun updateContextBadge(text: String) {
        val badge = PromptContextBadge.format(engine.currentFileContextPath(), text)
        contextBadge.text = badge
        contextBadge.visibility = if (badge.isEmpty()) View.GONE else View.VISIBLE
    }

    // ------------------------------------------------------- slash dispatch

    private fun sendMessage() {
        val text = inputField.text.toString().trim()
        if (text.isEmpty() && pendingImageBase64 == null) return
        if (text.startsWith("/") && handleSlashInput(text)) return
        inputField.setText("")
        val imageB64 = pendingImageBase64
        pendingImageBase64 = null
        pendingImagePreview?.visibility = View.GONE
        attachButton.setColorFilter(Color.parseColor("#6B7280"))
        val displayText = if (imageB64 != null) "$text\n[Image attached]" else text
        addMessage("user", displayText)
        // Clear previous command output
        commandOutputText.text = ""
        commandOutputScroll.visibility = View.GONE
        engine.sendMessage(text, imageBase64 = imageB64)
    }

    /**
     * Runs a local slash command. Returns true when consumed; false lets the
     * text flow to the engine (unknown-but-template commands like /init).
     */
    private fun handleSlashInput(text: String): Boolean {
        return when (val invocation = slashRegistry.parse(text)) {
            is SlashInvocation.Known -> {
                inputField.setText("")
                runSlashCommand(invocation.command)
                true
            }
            is SlashInvocation.Unknown -> {
                val knownTemplate = runCatching {
                    engine.listCommands().any { it.name.equals(invocation.name, ignoreCase = true) }
                }.getOrDefault(false)
                if (knownTemplate) false else {
                    inputField.setText("")
                    addMessage("system", "Unknown command: /${invocation.name} — type / to see commands.")
                    true
                }
            }
            SlashInvocation.NotACommand -> false
        }
    }

    private fun runSlashCommand(command: SlashCommand) {
        when (command.action) {
            SlashCommandAction.HELP -> addMessage("system", SlashCommandRegistry.helpText())
            SlashCommandAction.CLEAR -> confirmClearConversation()
            SlashCommandAction.NEW_SESSION -> startNewConversation()
            SlashCommandAction.SHOW_SESSIONS -> showHistoryDialog()
            SlashCommandAction.SHOW_MODELS -> showSettingsDialog()
            SlashCommandAction.TOGGLE_PLAN -> togglePlanMode()
            SlashCommandAction.UNDO -> confirmUndo()
            SlashCommandAction.REDO -> performRedo()
            SlashCommandAction.SHOW_CHANGES -> listChangedFilesNotice()
            SlashCommandAction.STATUS -> addMessage(
                "system",
                StatusLine.format(
                    project = engine.workspaceRoot().absolutePath,
                    openFile = engine.currentFileContextPath(),
                    model = engine.currentModelLabel(),
                    sessionId = engine.currentSessionId(),
                    agent = engine.agentName,
                ),
            )
        }
    }

    private fun startNewConversation() {
        engine.newSession()
        outputContainer.removeAllViews()
        commandOutputContainer.removeAllViews()
        commandOutputScroll.visibility = View.GONE
        planToggle.visibility = View.GONE
        planContainer.visibility = View.GONE
        thinkingText.text = "AI Agent Ready"
        thinkingCard.visibility = View.GONE
        thinkingLiveBuf.setLength(0)
        thinkingLiveDone = true
        updateTokenCount()
        addMessage("system", "Started a new conversation.")
    }

    private fun confirmClearConversation() {
        android.app.AlertDialog.Builder(context)
            .setTitle("Clear conversation?")
            .setMessage("This clears the visible conversation and starts a new session. The previous session stays in History.")
            .setPositiveButton("Clear") { _, _ -> startNewConversation() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun togglePlanMode() {
        val targetPosition = if (engine.agentName == "plan") 0 else 1 // 0 build, 1 plan
        if (modeSelector.selectedItemPosition == targetPosition) {
            engine.setAgentMode(if (targetPosition == 1) "plan" else "build")
            addMessage(
                "system",
                if (targetPosition == 1) "Agent mode: Plan (Read-Only)" else "Agent mode: Build (Full Access)",
            )
        } else {
            // The spinner's listener applies the mode and reports it.
            modeSelector.setSelection(targetPosition)
        }
    }

    private fun confirmUndo() {
        Thread {
            val deltas = runCatching { engine.pendingUndoDeltas() }.getOrDefault(emptyList())
            post {
                if (deltas.isEmpty()) {
                    addMessage("system", "Nothing to undo.")
                    return@post
                }
                val listing = deltas.joinToString("\n") { d ->
                    val sign = if (d.lineDelta >= 0) "+" else ""
                    "${File(d.path).name}  ($sign${d.lineDelta} lines)" + if (d.existedBefore) "" else " (created)"
                }
                android.app.AlertDialog.Builder(context)
                    .setTitle("Undo the agent's last change?")
                    .setMessage(listing)
                    .setPositiveButton("Undo") { _, _ ->
                        Thread {
                            val outcome = runCatching { engine.undoLastTurn() }
                                .getOrElse { CheckpointOutcome.Failed(it.message ?: "undo failed", emptyList()) }
                            post { reportUndoRedo(outcome, redo = false) }
                        }.start()
                    }
                    .setNegativeButton("Cancel", null)
                    .show()
            }
        }.start()
    }

    private fun performRedo() {
        Thread {
            val outcome = runCatching { engine.redoLastTurn() }
                .getOrElse { CheckpointOutcome.Failed(it.message ?: "redo failed", emptyList()) }
            post { reportUndoRedo(outcome, redo = true) }
        }.start()
    }

    private fun reportUndoRedo(outcome: CheckpointOutcome, redo: Boolean) {
        val message = when (outcome) {
            is CheckpointOutcome.Undone -> "Undid ${outcome.files.size} file(s)."
            is CheckpointOutcome.Redone -> "Re-applied ${outcome.files.size} file(s)."
            CheckpointOutcome.NothingToUndo -> if (redo) "Nothing to redo." else "Nothing to undo."
            CheckpointOutcome.NothingToRedo -> if (redo) "Nothing to redo." else "Nothing to undo."
            is CheckpointOutcome.Failed -> "⚠ ${if (redo) "Redo" else "Undo"} failed: ${outcome.message}"
        }
        addMessage("system", message)
    }

    /** /files for the panel (no CHANGES tab): list what this session changed. */
    private fun listChangedFilesNotice() {
        val sessionId = engine.currentSessionId()
        val files = if (sessionId == null) emptyList() else engine.sessionMessages(sessionId)
            .flatMap { msg ->
                val parts = OcMessage.parts(msg)
                (0 until parts.length()).mapNotNull { i ->
                    val part = parts.optJSONObject(i) ?: return@mapNotNull null
                    if (part.optString("type") != "tool") return@mapNotNull null
                    if (part.optString("tool") !in setOf("write", "edit", "patch")) return@mapNotNull null
                    val toolInput = part.optJSONObject("state")?.optJSONObject("input")
                    val path = toolInput?.optString("filePath", "")?.ifEmpty { toolInput.optString("path", "") }
                    path?.takeIf { it.isNotEmpty() }
                }
            }
            .distinct()
        addMessage(
            "system",
            if (files.isEmpty()) {
                "No files changed in this session yet. (Full diff view: AI Workspace → Changes.)"
            } else {
                "Files changed in this session:\n" + files.joinToString("\n") { "• $it" }
            },
        )
    }

    private fun pickImage() {
        val intent = Intent(Intent.ACTION_PICK).apply {
            type = "image/*"
        }
        (context as? android.app.Activity)?.startActivityForResult(intent, 9999)
    }

    fun onImagePicked(uri: Uri?) {
        if (uri == null) return
        try {
            val inputStream = context.contentResolver.openInputStream(uri) ?: return
            val bitmap = BitmapFactory.decodeStream(inputStream)
            inputStream.close()
            if (bitmap == null) return
            // Resize if too large (max 1024px on longest side)
            val maxDim = maxOf(bitmap.width, bitmap.height)
            val scaled = if (maxDim > 1024) {
                val ratio = 1024f / maxDim
                Bitmap.createScaledBitmap(bitmap, (bitmap.width * ratio).toInt(), (bitmap.height * ratio).toInt(), true)
            } else bitmap
            val baos = ByteArrayOutputStream()
            scaled.compress(Bitmap.CompressFormat.JPEG, 80, baos)
            pendingImageBase64 = Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
            attachButton.setColorFilter(Color.parseColor("#61AFEF"))
            showToast("Image attached")
        } catch (e: Exception) {
            showToast("Failed to load image")
        }
    }

    private fun addMessage(role: String, text: String) {
        val messageView = createMessageView(role, text)
        outputContainer.addView(messageView)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun updateLastMessage(text: String) {
        val last = outputContainer.getChildAt(outputContainer.childCount - 1) as? LinearLayout
        if (last == null || last.tag != "ai") {
            val view = createMessageView("ai", text)
            view.tag = "ai"
            outputContainer.addView(view)
            streamText = text
            streamView = findGrowingView(view)
            scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
            return
        }

        // Fast path: pure append with no ``` change - just extend the growing
        // TextView instead of rebuilding the whole markdown view hierarchy
        // (that full rebuild per token was O(n^2) and froze the panel).
        val delta = when {
            text.startsWith(streamText) -> text.substring(streamText.length)
            else -> null
        }
        if (delta != null && !delta.contains("```")) {
            if (delta.isNotEmpty()) {
                val inner = (last.findViewById<ScrollView>(ID_SCROLL)?.getChildAt(0)) as? LinearLayout
                val target = streamView
                    ?: TextView(context).apply {
                        textSize = 13f
                        setTextColor(Color.parseColor("#D1D5DB"))
                        setLineSpacing(0f, 1.2f)
                        inner?.addView(this)
                    }.also { streamView = it }
                target.append(renderInlineMarkdown(delta))
                streamText = text
                scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
            }
            return
        }

        // Fence opened/closed or discontinuity: one full rebuild.
        rebuildLastMessage(last, text)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    /** Final formatting pass +10-line clamp once the assistant message is complete. */
    private fun finalizeLastMessage() {
        val last = outputContainer.getChildAt(outputContainer.childCount - 1) as? LinearLayout ?: return
        if (last.tag != "ai") return
        if (streamText.isNotEmpty()) rebuildLastMessage(last, streamText)
        clampMessage(last)
    }

    private fun rebuildLastMessage(root: LinearLayout, text: String) {
        val inner = (root.findViewById<ScrollView>(ID_SCROLL)?.getChildAt(0)) as? LinearLayout
        if (inner == null) {
            root.removeAllViews()
            renderMarkdownContent(text, root)
        } else {
            inner.removeAllViews()
            if (text.isNotEmpty()) renderMarkdownContent(text, inner)
        }
        streamText = text
        streamView = findGrowingView(root)
    }

    /** The TextView currently receiving more text (last text view, or code view inside a fence). */
    private fun findGrowingView(root: View): TextView? {
        val inner = (root.findViewById<ScrollView>(ID_SCROLL)?.getChildAt(0)) as? LinearLayout ?: return null
        if (inner.childCount == 0) return null
        val lastChild = inner.getChildAt(inner.childCount - 1)
        return when (lastChild) {
            is TextView -> lastChild
            is LinearLayout -> lastChild.getChildAt(lastChild.childCount - 1) as? TextView
            else -> null
        }
    }

    private fun updateTokenCount() {
        val tokens = engine.estimateTokens()
        tokenCounter.text = when {
            tokens > 1000 -> "${tokens / 1000}.${(tokens % 1000) / 100}k tokens"
            else -> "$tokens tokens"
        }
    }

    private fun addTimelineToolItem(name: String, args: JSONObject) {
        val icon = when (name) {
            "read_file", "list_files", "search_files", "search_codebase" -> "🔍"
            "write_file", "write_files" -> "📝"
            "edit_file" -> "✏️"
            "run_command", "run_command_async" -> "⚙️"
            "run_gradle" -> "🔨"
            "run_tests" -> "🧪"
            "check_command" -> "📊"
            "get_project_context" -> "📋"
            "git_diff", "git_status" -> "🔀"
            "git_commit" -> "💾"
            "git_branch" -> "🌿"
            "save_memory" -> "🧠"
            "recall_memory" -> "📚"
            "web_search" -> "🌐"
            "web_fetch" -> "📱"
            else -> "🔧"
        }
        val label = when (name) {
            "read_file" -> "Read"
            "write_file" -> "Edited"
            "edit_file" -> "Replaced"
            "run_command" -> "Running"
            "run_command_async" -> "Started"
            "run_gradle" -> "Gradle"
            "run_tests" -> "Testing"
            "check_command" -> "Checking"
            "get_project_context" -> "Context"
            "git_commit" -> "Committed"
            "git_status" -> "Git Status"
            "git_branch" -> "Branch"
            "git_diff" -> "Diff"
            "save_memory" -> "Saved"
            "recall_memory" -> "Recalled"
            "web_search" -> "Searching"
            "web_fetch" -> "Fetching"
            else -> "Tool"
        }
        val target = args.optString("path", args.optString("command", args.optString("task", args.optString("query", ""))))
        val display = if (target.length > 40) target.take(37) + "..." else target

        val item = LinearLayout(context)
        item.apply {
            orientation = VERTICAL
            setPadding(0, dp(4), 0, dp(4))
            val row = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(TextView(context).apply { this.text = icon; textSize = 14f; setPadding(0, 0, dp(8), 0) })
                addView(TextView(context).apply { this.text = "$label "; textSize = 13f; setTextColor(Color.parseColor("#9CA3AF")) })
                addView(TextView(context).apply {
                    this.text = display
                    textSize = 13f
                    setTextColor(Color.parseColor("#4F8EF7"))
                    setTypeface(null, Typeface.BOLD)
                    isClickable = true
                    if (name == "write_file") {
                        setOnClickListener { toggleDiff(item, args) }
                    } else if (name == "run_command_async" || name == "run_gradle") {
                        setOnClickListener {
                            val cmdId = args.optString("commandId", "")
                            if (cmdId.isNotEmpty()) {
                                addMessage("system", "Checking command: $cmdId")
                                engine.sendMessage("check_command $cmdId")
                            }
                        }
                    }
                })
            }
            addView(row)
        }
        outputContainer.addView(item)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun toggleDiff(item: LinearLayout, args: JSONObject) {
        val existing = item.findViewById<View>(ID_DIFF)
        if (existing != null) {
            item.removeView(existing)
            return
        }
        val diffContainer = LinearLayout(context).apply {
            id = ID_DIFF
            orientation = VERTICAL
            setPadding(dp(24), dp(8), dp(8), dp(8))
            background = StudioUi.rounded(context, fill = "#171C25", radiusDp = 8)
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply { topMargin = dp(4) }
        }
        val path = args.getString("path")
        val newContent = args.getString("content")
        val oldContent = runCatching<String> { File(path).readText() }.getOrDefault("")
        computeSimpleDiff(oldContent, newContent).forEach { line ->
            diffContainer.addView(TextView(context).apply {
                this.text = line.text
                textSize = 11f
                typeface = Typeface.MONOSPACE
                setTextColor(Color.parseColor(line.color))
                setBackgroundColor(Color.parseColor(line.bg))
            })
        }
        item.addView(diffContainer)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private data class DiffLine(val text: String, val color: String, val bg: String)
    private fun computeSimpleDiff(old: String, new: String): List<DiffLine> {
        val oldLines = old.lines(); val newLines = new.lines()
        val result = mutableListOf<DiffLine>()
        var i = 0; var j = 0
        while (i < oldLines.size || j < newLines.size) {
            if (i < oldLines.size && j < newLines.size && oldLines[i] == newLines[j]) { i++; j++ }
            else if (j < newLines.size && (i >= oldLines.size || !oldLines.drop(i).contains(newLines[j]))) {
                result.add(DiffLine("+ ${newLines[j]}", "#98C379", "#12251A")); j++
            } else { result.add(DiffLine("- ${oldLines[i]}", "#E06C75", "#2A1418")); i++ }
            if (result.size > 20) { result.add(DiffLine("... truncated ...", "#8E99AC", "#171C25")); break }
        }
        return result
    }

    private fun createMessageView(role: String, text: String): View {
        val isUser = role == "user"
        val root = LinearLayout(context).apply {
            orientation = VERTICAL
            // User messages hug the right edge as a speech bubble.
            gravity = if (isUser) Gravity.END else Gravity.START
            setPadding(0, dp(8), 0, dp(8))

            // Long-press to copy entire message
            setOnLongClickListener {
                val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                val clip = android.content.ClipData.newPlainText("AI message", text)
                clipboard.setPrimaryClip(clip)
                Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                true
            }
            isClickable = true
            isFocusable = true

            if (role == "ai" && text.contains("change introduced") && text.startsWith("⚠️")) {
                background = StudioUi.rounded(context, fill = "#2A1418", radiusDp = 8)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            } else if (role == "ai" && text.contains("change fixed") && text.startsWith("✅")) {
                background = StudioUi.rounded(context, fill = "#12251A", radiusDp = 8)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
        }

        if (role == "ai" && text.startsWith("💭")) {
            root.addView(LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                addView(TextView(context).apply { this.text = "🧠"; textSize = 14f; setPadding(0, 0, dp(8), 0) })
                addView(TextView(context).apply { this.text = "Thought"; textSize = 13f; setTextColor(Color.parseColor("#9CA3AF")) })
            })
            root.addView(TextView(context).apply { id = ID_CONTENT; this.text = text.removePrefix("💭").trim(); textSize = 13f; setTextColor(Color.parseColor("#D1D5DB")); setPadding(dp(22), dp(4), 0, 0) })
            return root
        }

        // User: rounded bubble (label inside), capped at 85% width, right-aligned.
        // The bubble lives in a MATCH_PARENT horizontal row with gravity END and
        // is itself WRAP_CONTENT so short messages stay narrow. (Giving the
        // bubble weight directly would measure it 0 px wide: this root is
        // VERTICAL, so weight distributes height, not width.)
        val host: LinearLayout
        if (isUser) {
            val bubble = LinearLayout(context).apply {
                orientation = VERTICAL
                background = StudioUi.rounded(context, fill = "#24405F", radiusDp = 14)
                setPadding(dp(12), dp(8), dp(12), dp(8))
            }
            val row = LinearLayout(context).apply {
                orientation = HORIZONTAL
                gravity = Gravity.END
            }
            row.addView(bubble, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT))
            root.addView(row, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
            host = bubble
            capUserBubbleWidthOnceLaidOut(row, bubble)
        } else {
            host = root
        }

        // Role label
        host.addView(TextView(context).apply {
            this.text = when (role) {
                "user" -> "You"
                "system" -> "System"
                else -> "AI"
            }
            textSize = 11f; setTypeface(null, Typeface.BOLD)
            setTextColor(when (role) {
                "user" -> Color.parseColor("#98C379")
                "system" -> Color.parseColor("#E5C07B")
                else -> Color.parseColor("#61AFEF")
            })
        })

        // Render content with code blocks inside a height-capped scroll area
        // (10-line preview + Show more/less toggle for long messages).
        val contentInner = LinearLayout(context).apply { orientation = VERTICAL }
        if (role == "ai" || role == "user") {
            renderMarkdownContent(text, contentInner)
        } else {
            contentInner.addView(TextView(context).apply {
                id = ID_CONTENT
                this.text = text
                textSize = 13f
                setTextColor(Color.parseColor("#D1D5DB"))
            })
        }
        val contentScroll = ScrollView(context).apply {
            id = ID_SCROLL
            layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT)
            isHorizontalScrollBarEnabled = false
            isFillViewport = false
            overScrollMode = View.OVER_SCROLL_NEVER
            addView(contentInner)
        }
        host.addView(contentScroll)
        val showMore = TextView(context).apply {
            id = ID_SHOW_MORE
            this.text = "Show more ⌵"
            textSize = 11f
            setTextColor(StudioUi.palette.muted)
            setPadding(0, dp(4), 0, 0)
            visibility = View.GONE
            isClickable = true
            setOnClickListener {
                val expanded = tag == true
                val cap = (contentScroll.tag as? Int) ?: contentScroll.height
                if (expanded) {
                    contentScroll.layoutParams = contentScroll.layoutParams.apply { height = cap }
                    this.text = "Show more ⌵"
                } else {
                    contentScroll.layoutParams =
                        contentScroll.layoutParams.apply { height = LayoutParams.WRAP_CONTENT }
                    this.text = "Show less ▴"
                }
                tag = !expanded
            }
        }
        host.addView(showMore)
        clampMessage(root)
        return root
    }

    /**
     * Clamp the message content to [MESSAGE_CLAMP_LINES] visible lines once it
     * overflows, revealing a Show more/less toggle. No-op while the user has
     * expanded the message or while it still fits.
     */
    private fun clampMessage(messageRoot: View) {
        val scroll = messageRoot.findViewById<ScrollView>(ID_SCROLL) ?: return
        val toggle = messageRoot.findViewById<TextView>(ID_SHOW_MORE) ?: return
        if (toggle.tag == true) return // user expanded it - leave alone
        scroll.post {
            if (toggle.visibility == View.VISIBLE && scroll.tag != null) return@post
            val firstLine = findFirstLineHeight(scroll)
            val cap = MESSAGE_CLAMP_LINES * firstLine + scroll.paddingTop + scroll.paddingBottom
            if (scroll.height > cap) {
                scroll.tag = cap
                scroll.layoutParams = scroll.layoutParams.apply { height = cap }
                toggle.visibility = View.VISIBLE
            }
        }
    }

    private fun findFirstLineHeight(scroll: ScrollView): Int {
        val inner = scroll.getChildAt(0)
        fun walk(v: View?): Int {
            when (v) {
                is TextView -> return if (v.lineHeight > 0) v.lineHeight else dp(16)
                is ViewGroup -> for (i in 0 until v.childCount) {
                    val h = walk(v.getChildAt(i))
                    if (h > 0) return h
                }
            }
            return 0
        }
        return walk(inner).coerceAtLeast(dp(16))
    }

    private fun renderMarkdownContent(text: String, container: LinearLayout) {
        val parts = text.split("```")
        if (parts.size == 1) {
            container.addView(TextView(context).apply {
                id = ID_CONTENT
                setText(renderInlineMarkdown(text), android.widget.TextView.BufferType.SPANNABLE)
                textSize = 13f
                setTextColor(Color.parseColor("#D1D5DB"))
                setLineSpacing(0f, 1.2f)
            })
            return
        }

        for (i in parts.indices) {
            if (i % 2 == 0) {
                val textPart = parts[i].trim()
                if (textPart.isNotEmpty()) {
                    container.addView(TextView(context).apply {
                        setText(renderInlineMarkdown(textPart), android.widget.TextView.BufferType.SPANNABLE)
                        textSize = 13f
                        setTextColor(Color.parseColor("#D1D5DB"))
                        setLineSpacing(0f, 1.2f)
                    })
                }
            } else {
                val langEnd = parts[i].indexOf('\n')
                val lang = if (langEnd > 0) parts[i].substring(0, langEnd).trim() else ""
                val code = if (langEnd > 0) parts[i].substring(langEnd + 1) else parts[i]

                if (lang.isNotEmpty()) {
                    container.addView(TextView(context).apply {
                        this.text = lang
                        textSize = 10f
                        setTextColor(Color.parseColor("#6B7280"))
                        setBackgroundColor(Color.parseColor("#161B22"))
                        setPadding(dp(8), dp(4), dp(8), dp(4))
                        typeface = Typeface.MONOSPACE
                    })
                }

                val codeBlock = LinearLayout(context).apply {
                    orientation = VERTICAL
                    setBackgroundColor(Color.parseColor("#0D1117"))
                    setPadding(dp(8), dp(8), dp(8), dp(8))
                    layoutParams = LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                        topMargin = if (lang.isNotEmpty()) 0 else dp(4)
                        bottomMargin = dp(4)
                    }
                }

                val codeView = TextView(context).apply {
                    id = ID_CODE_BLOCK
                    this.text = code.trimEnd()
                    textSize = 11f
                    typeface = Typeface.MONOSPACE
                    setTextColor(Color.parseColor("#C9D1D9"))
                    setTextIsSelectable(true)
                    setLineSpacing(0f, 1.3f)
                }
                codeBlock.addView(codeView)

                // Copy button
                val copyBtn = TextView(context).apply {
                    this.text = "📋 Copy"
                    textSize = 10f
                    setTextColor(Color.parseColor("#4F8EF7"))
                    setPadding(dp(4), dp(4), 0, 0)
                    setOnClickListener {
                        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        val clip = android.content.ClipData.newPlainText("code", code.trimEnd())
                        clipboard.setPrimaryClip(clip)
                        Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                    }
                }
                codeBlock.addView(copyBtn)

                container.addView(codeBlock)
            }
        }
    }

    private fun renderInlineMarkdown(text: String): android.text.SpannableStringBuilder {
        val ssb = android.text.SpannableStringBuilder(text)

        // Bold: **text** or __text__
        val boldPattern = Regex("""(\*\*|__)(.*?)\1""")
        for (match in boldPattern.findAll(text)) {
            val start = match.range.first
            val end = match.range.last + 1
            val contentStart = start + 2
            val contentEnd = end - 2
            ssb.setSpan(android.text.style.StyleSpan(android.graphics.Typeface.BOLD),
                contentStart, contentEnd, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            ssb.delete(end - 2, end)
            ssb.delete(start, start + 2)
        }

        // Italic: *text* or _text_
        val italicPattern = Regex("""(?<!\*)(\*)(?!\*)(.*?)(?<!\*)(\*)(?!\*)""")
        for (match in italicPattern.findAll(ssb.toString())) {
            val start = match.range.first
            val end = match.range.last + 1
            val contentStart = start + 1
            val contentEnd = end - 1
            if (contentEnd > contentStart) {
                ssb.setSpan(android.text.style.StyleSpan(android.graphics.Typeface.ITALIC),
                    contentStart, contentEnd, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                ssb.delete(end - 1, end)
                ssb.delete(start, start + 1)
            }
        }

        // Inline code: `text`
        val codePattern = Regex("""`(.*?)`""")
        for (match in codePattern.findAll(ssb.toString())) {
            val start = match.range.first
            val end = match.range.last + 1
            val contentStart = start + 1
            val contentEnd = end - 1
            ssb.setSpan(android.text.style.TypefaceSpan(Typeface.MONOSPACE),
                contentStart, contentEnd, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            ssb.setSpan(android.text.style.ForegroundColorSpan(Color.parseColor("#E06C75")),
                contentStart, contentEnd, android.text.Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            ssb.delete(end - 1, end)
            ssb.delete(start, start + 1)
        }

        return ssb
    }

    private fun showApprovalUI(name: String, args: JSONObject, risk: AiAgentEngine.RiskLevel, onApproval: (Boolean, Boolean) -> Unit) {
        approvalContainer.removeAllViews()
        approvalContainer.visibility = View.VISIBLE

        val riskColor = when (risk) {
            AiAgentEngine.RiskLevel.READ_ONLY -> "#2E8549"
            AiAgentEngine.RiskLevel.BUILD -> "#4F8EF7"
            AiAgentEngine.RiskLevel.MODIFY -> "#E3B963"
            AiAgentEngine.RiskLevel.DANGEROUS -> "#F05C5C"
        }
        val riskLabel = when (risk) {
            AiAgentEngine.RiskLevel.READ_ONLY -> "READ ONLY"
            AiAgentEngine.RiskLevel.BUILD -> "BUILD"
            AiAgentEngine.RiskLevel.MODIFY -> "MODIFY"
            AiAgentEngine.RiskLevel.DANGEROUS -> "DANGEROUS"
        }

        approvalContainer.addView(TextView(context).apply {
            this.text = "⚠️ AI requests: $name"
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.WHITE)
            textSize = 14f
        })
        approvalContainer.addView(TextView(context).apply {
            this.text = "Risk: $riskLabel"
            textSize = 11f
            setTextColor(Color.parseColor(riskColor))
            setPadding(0, dp(4), 0, dp(4))
        })

        val argsPreview = when (name) {
            "run_command", "run_command_async" -> "Command: ${args.optString("command", "")}"
            "run_gradle" -> "Task: ${args.optString("task", "")}"
            "write_file" -> "File: ${args.optString("path", "")}"
            else -> "Args: $args"
        }
        approvalContainer.addView(TextView(context).apply {
            this.text = argsPreview
            textSize = 12f
            setTextColor(Color.parseColor("#D1D5DB"))
            setPadding(0, dp(4), 0, dp(8))
            typeface = Typeface.MONOSPACE
        })

        if (name == "write_file") {
            approvalContainer.addView(Button(context).apply {
                text = "Show Diff"
                textSize = 11f
                setOnClickListener {
                    val p = args.getString("path")
                    val n = args.getString("content")
                    val o = runCatching<String> { File(p).readText() }.getOrDefault("")
                    com.aide.mobilestudio.DiffViewerActivity.start(context, "Current", o, "AI Proposal", n)
                }
            })
        }

        val row = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.END
        }
        row.addView(Button(context).apply {
            text = "Reject"
            setOnClickListener {
                approvalContainer.visibility = View.GONE
                onApproval(false, false)
            }
        })
        row.addView(Button(context).apply {
            text = "Allow Once"
            setOnClickListener {
                approvalContainer.visibility = View.GONE
                onApproval(true, false)
            }
        })
        row.addView(Button(context).apply {
            text = "Allow Always"
            isEnabled = risk != AiAgentEngine.RiskLevel.DANGEROUS
            setOnClickListener {
                approvalContainer.visibility = View.GONE
                onApproval(true, true)
            }
        })
        approvalContainer.addView(row)
    }

    private fun showHistoryDialog() {
        val conversations = engine.listSessions()
        val dialog = android.app.AlertDialog.Builder(context).setTitle("Conversation History")

        if (conversations.isEmpty()) {
            dialog.setMessage("No saved conversations.")
            dialog.setPositiveButton("OK", null)
            dialog.show()
            return
        }

        val titles = conversations.map {
            val date = SimpleDateFormat("MM/dd HH:mm", Locale.getDefault()).format(Date(it.timeUpdated))
            val tokens = " (~${it.messages.length()} msgs)"
            "${it.title.ifBlank { "Untitled" }} ($date)$tokens"
        }.toTypedArray()

        dialog.setItems(titles) { _, which ->
            showSessionActions(conversations[which])
        }
        dialog.setNeutralButton("Export All") { _, _ ->
            val exportDir = File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS), "MobileStudio/AI_Export")
            if (conversationStorage.exportTo(exportDir)) {
                Toast.makeText(context, "Exported to ${exportDir.absolutePath}", Toast.LENGTH_LONG).show()
            }
        }
        dialog.setPositiveButton("Import") { _, _ ->
            val srcDir = File(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DOCUMENTS), "MobileStudio/AI_Import")
            if (conversationStorage.importFrom(srcDir)) {
                Toast.makeText(context, "Imported from ${srcDir.absolutePath}", Toast.LENGTH_LONG).show()
                showHistoryDialog()
            } else {
                Toast.makeText(context, "Import failed. Place JSON files in Documents/MobileStudio/AI_Import", Toast.LENGTH_LONG).show()
            }
        }
        dialog.show()
    }

    private fun showSessionActions(conv: OcSessionStore.Session) {
        val archiveLabel = if (conv.archived) "Unarchive" else "Archive"
        val actions = arrayOf("Open", "Fork", archiveLabel, "Delete", "Cancel")
        android.app.AlertDialog.Builder(context)
            .setTitle(conv.title.ifBlank { "Untitled" })
            .setItems(actions) { _, which ->
                when (which) {
                    0 -> loadSessionIntoUi(conv)
                    1 -> {
                        val forkedId = engine.forkSession(conv.id)
                        if (forkedId == null) {
                            Toast.makeText(context, "Fork failed", Toast.LENGTH_SHORT).show()
                        } else {
                            val forked = engine.listSessions(true).firstOrNull { it.id == forkedId } ?: conv
                            loadSessionIntoUi(forked)
                            Toast.makeText(context, "Forked to new session", Toast.LENGTH_SHORT).show()
                        }
                    }
                    2 -> {
                        val nowArchived = !conv.archived
                        engine.archiveSession(conv.id, nowArchived)
                        Toast.makeText(
                            context,
                            if (nowArchived) "Session archived" else "Session restored",
                            Toast.LENGTH_SHORT,
                        ).show()
                        showHistoryDialog()
                    }
                    3 -> android.app.AlertDialog.Builder(context)
                        .setTitle("Delete session?")
                        .setMessage(conv.title.ifBlank { "Untitled" })
                        .setPositiveButton("Delete") { _, _ ->
                            engine.deleteSession(conv.id)
                            Toast.makeText(context, "Session deleted", Toast.LENGTH_SHORT).show()
                            showHistoryDialog()
                        }
                        .setNegativeButton("Cancel", null)
                        .show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun loadSessionIntoUi(conv: OcSessionStore.Session) {
        engine.loadSession(conv.id)
        outputContainer.removeAllViews()
        commandOutputText.text = ""
        commandOutputScroll.visibility = View.GONE
        for (msg in engine.sessionMessages(conv.id)) {
            val role = msg.optString("role")
            if (role == "system") continue
            val parts = msg.optJSONArray("parts") ?: continue
            val textBuf = StringBuilder()
            for (i in 0 until parts.length()) {
                val part = parts.optJSONObject(i) ?: continue
                if (part.optString("type") == "text" && !part.optBoolean("synthetic", false)) {
                    if (textBuf.isNotEmpty()) textBuf.append("\n")
                    textBuf.append(part.optString("text"))
                }
            }
            val text = textBuf.toString()
            if (text.isNotEmpty()) {
                if (role == "assistant" || role == "model") updateLastMessage(text)
                else addMessage("user", text)
            }
        }
        updateTokenCount()
    }

    private fun syncRuntimeOptions() {
        // GPU offload slider → n_gpu_layers, CPU threads, context size, output tokens
        engine.runtimeOptions.put("nGpuLayers", settings.nGpuLayers)
        engine.runtimeOptions.put("nThreads", settings.nThreads)
        engine.runtimeOptions.put("context", settings.contextLength)
        engine.runtimeOptions.put("maxTokens", settings.openaiMaxTokens)
    }

    private fun showSettingsDialog() {
        val dialog = android.app.AlertDialog.Builder(context).setTitle("AI Settings")
        val layout = LinearLayout(context).apply { orientation = VERTICAL; setPadding(dp(20), dp(16), dp(20), dp(16)) }

        // Gemini section
        layout.addView(TextView(context).apply { text = "Google Gemini"; setTextColor(Color.parseColor("#4285F4")); setTypeface(null, Typeface.BOLD); textSize = 14f })
        layout.addView(TextView(context).apply { text = "API Key"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val geminiKeyInput = EditText(context).apply {
            setText(settings.apiKey)
            hint = "Gemini API key"
            textSize = 13f
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(geminiKeyInput)
        layout.addView(TextView(context).apply { text = "Model"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val geminiModelInput = EditText(context).apply {
            setText(settings.geminiModel)
            hint = "gemini-2.0-flash"
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(geminiModelInput)
        layout.addView(Button(context).apply {
            text = "Get Gemini API Key"
            textSize = 10f
            setTextColor(Color.parseColor("#4285F4"))
            background = null
            setOnClickListener { runCatching { context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://aistudio.google.com/app/apikey"))) } }
        })

        // OpenAI section
        layout.addView(View(context).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(16)) })
        layout.addView(TextView(context).apply { text = "OpenAI (ChatGPT)"; setTextColor(Color.parseColor("#10A37F")); setTypeface(null, Typeface.BOLD); textSize = 14f })
        layout.addView(TextView(context).apply { text = "API Key"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val openaiKeyInput = EditText(context).apply {
            setText(settings.openaiApiKey)
            hint = "sk-..."
            textSize = 13f
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(openaiKeyInput)
        layout.addView(TextView(context).apply { text = "Model"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val openaiModelInput = EditText(context).apply {
            setText(settings.openaiModel)
            hint = "gpt-4o"
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(openaiModelInput)
        layout.addView(TextView(context).apply { text = "Base URL (optional, for compatible APIs)"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val openaiUrlInput = EditText(context).apply {
            setText(settings.openaiBaseUrl)
            hint = "https://api.openai.com/v1"
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(openaiUrlInput)
        // Provider presets: one-tap Base URL + Model fill for OpenAI-compatible APIs
        layout.addView(TextView(context).apply { text = "Provider Preset"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val presetNames = arrayOf(
            "(keep current)", "OpenAI", "GLM (Zhipu)", "Qwen (DashScope)",
            "NVIDIA NIM", "DeepSeek", "Moonshot (Kimi)", "Groq", "Ollama (local)"
        )
        val presetUrls = arrayOf(
            "", "https://api.openai.com/v1", "https://open.bigmodel.cn/api/paas/v4",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "https://integrate.api.nvidia.com/v1", "https://api.deepseek.com/v1",
            "https://api.moonshot.cn/v1", "https://api.groq.com/openai/v1",
            "http://127.0.0.1:11434/v1"
        )
        val presetModels = arrayOf(
            "", "gpt-4o", "glm-4-plus", "qwen-max",
            "meta/llama-3.1-70b-instruct", "deepseek-chat", "moonshot-v1-8k",
            "llama-3.3-70b-versatile", "qwen2.5-coder:7b"
        )
        val presetSpinner = android.widget.Spinner(context).apply {
            adapter = android.widget.ArrayAdapter(context, android.R.layout.simple_spinner_dropdown_item, presetNames)
            onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                var first = true
                override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                    if (first) { first = false; return }
                    if (position == 0) return
                    openaiUrlInput.setText(presetUrls[position])
                    openaiModelInput.setText(presetModels[position])
                    Toast.makeText(context, "${presetNames[position]} preset applied — enter API key below", Toast.LENGTH_SHORT).show()
                }
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            }
        }
        layout.addView(presetSpinner)
        layout.addView(TextView(context).apply { text = "Max Output Tokens (per-model)"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val maxTokensInput = EditText(context).apply {
            setText(settings.openaiMaxTokens.toString())
            hint = "8192"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(maxTokensInput)

        // Anthropic section
        layout.addView(View(context).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(16)) })
        layout.addView(TextView(context).apply { text = "Anthropic (Claude)"; setTextColor(Color.parseColor("#D4A574")); setTypeface(null, Typeface.BOLD); textSize = 14f })
        layout.addView(TextView(context).apply { text = "API Key"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val anthropicKeyInput = EditText(context).apply {
            setText(settings.anthropicApiKey)
            hint = "sk-ant-..."
            textSize = 13f
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(anthropicKeyInput)
        layout.addView(TextView(context).apply { text = "Model"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f })
        val anthropicModelInput = EditText(context).apply {
            setText(settings.anthropicModel)
            hint = "claude-sonnet-4-20250514"
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(anthropicModelInput)

        // Auto Apply
        layout.addView(View(context).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(16)) })
        layout.addView(TextView(context).apply {
            text = "Auto Apply Changes"
            setPadding(0, dp(8), 0, 4)
            textSize = 12f
            setTextColor(Color.parseColor("#9CA3AF"))
        })
        val autoApplySwitch = androidx.appcompat.widget.SwitchCompat(context).apply { isChecked = settings.autoApplyChanges }
        layout.addView(autoApplySwitch)

        // Local AI (llama.cpp runtime) section
        layout.addView(View(context).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(16)) })
        layout.addView(TextView(context).apply { text = "Local AI (llama.cpp, CPU + Vulkan)"; setTextColor(Color.parseColor("#A78BFA")); setTypeface(null, Typeface.BOLD); textSize = 14f })
        val runtimeManager = com.aide.mobilestudio.ai.LlamaRuntimeManager.get(context)
        val runtimeStatusText = TextView(context).apply {
            text = if (runtimeManager.isInstalled())
                "Runtime: ${runtimeManager.installedVersion()} (installed)"
            else
                "Runtime: not installed"
            setTextColor(Color.parseColor("#9CA3AF"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
        }
        layout.addView(runtimeStatusText)
        val runtimeProgressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        layout.addView(runtimeProgressBar)
        val runtimeProgressText = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#7EE787"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
        }
        layout.addView(runtimeProgressText)
        layout.addView(Button(context).apply {
            text = if (runtimeManager.isInstalled()) "Re-download AI Runtime" else "Download AI Runtime"
            textSize = 11f
            setOnClickListener {
                runtimeProgressBar.progress = 0
                runtimeProgressText.text = "Downloading AI Runtime..."
                runtimeManager.ensureInstalled(
                    onProgress = { read, total ->
                        post {
                            val percent = if (total > 0) ((read * 100) / total).toInt() else 0
                            runtimeProgressBar.progress = percent.coerceIn(0, 100)
                            runtimeProgressText.text = "${runtimeManager.formatSize(read)} / ${runtimeManager.formatSize(total)}"
                        }
                    },
                    onLog = { line -> post { runtimeProgressText.text = line } },
                    onDone = { success, error ->
                        post {
                            if (success) {
                                runtimeProgressText.text = "[AI Runtime] Installed (no re-download on restart)"
                                text = "Re-download AI Runtime"
                            } else {
                                runtimeProgressText.text = "Install failed: ${error ?: "unknown"} (retry available)"
                            }
                        }
                    }
                )
            }
        })
        layout.addView(TextView(context).apply { text = "Compute: CPU <====|=============|===> GPU"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f; setPadding(0, dp(10), 0, dp(2)) })
        val gpuSlider = SeekBar(context).apply { max = 100 }
        val gpuSliderLabel = TextView(context).apply {
            text = ""
            setTextColor(Color.parseColor("#C9D1D9"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
        }
        fun updateGpuSliderLabel(progress: Int) {
            // n_gpu_layers levels only (NOT a CPU/GPU compute percentage):
            // 0 = CPU, 99 = -ngl 99 (전체 layer GPU offload), N = N layers on GPU
            when {
                progress == 0 -> { settings.nGpuLayers = 0; gpuSliderLabel.text = "n_gpu_layers = 0 (CPU 위주)" }
                progress >= 100 -> { settings.nGpuLayers = 99; gpuSliderLabel.text = "n_gpu_layers = 99 (-ngl 99: 전체 layer GPU offload)" }
                else -> {
                    val layers = (progress / 5).coerceAtLeast(1)
                    settings.nGpuLayers = layers
                    gpuSliderLabel.text = "n_gpu_layers = $layers (일부 layer를 Vulkan으로 offload)"
                }
            }
        }
        val savedGpu = settings.nGpuLayers
        gpuSlider.progress = when {
            savedGpu >= 99 || savedGpu == -1 -> 100
            savedGpu <= 0 -> 0
            else -> (savedGpu * 5).coerceIn(0, 95)
        }
        updateGpuSliderLabel(gpuSlider.progress)
        gpuSlider.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                updateGpuSliderLabel(progress)
            }
            override fun onStartTrackingTouch(sb: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(sb: android.widget.SeekBar?) {}
        })
        layout.addView(gpuSlider)
        layout.addView(gpuSliderLabel)
        layout.addView(TextView(context).apply { text = "CPU Threads (1-8)"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f; setPadding(0, dp(8), 0, dp(2)) })
        val threadsInput = EditText(context).apply {
            setText(settings.nThreads.toString())
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(threadsInput)
        layout.addView(TextView(context).apply { text = "Context Size (512-16384)"; setTextColor(Color.parseColor("#9CA3AF")); textSize = 11f; setPadding(0, dp(8), 0, dp(2)) })
        val contextInput = EditText(context).apply {
            setText(settings.contextLength.toString())
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
            textSize = 13f
            background = StudioUi.rounded(context, fill = "#0F1724", stroke = "#2F3B4D", radiusDp = 8)
            setPadding(dp(12), dp(8), dp(12), dp(8))
        }
        layout.addView(contextInput)
        layout.addView(TextView(context).apply {
            text = "모델 로딩 실패 시 GPU offload를 자동으로 낮춰 재시도합니다 (최대 4회, 이후 CPU 폴백)."
            setTextColor(Color.parseColor("#484F58"))
            textSize = 10f
            setPadding(0, dp(8), 0, 0)
        })

        // Scroll wrapper
        val scrollWrapper = ScrollView(context).apply { addView(layout) }

        dialog.setView(scrollWrapper).setPositiveButton("Save") { _, _ ->
            settings.apiKey = geminiKeyInput.text.toString().trim()
            settings.geminiModel = geminiModelInput.text.toString().trim().ifEmpty { "gemini-2.0-flash" }
            settings.openaiApiKey = openaiKeyInput.text.toString().trim()
            settings.openaiModel = openaiModelInput.text.toString().trim().ifEmpty { "gpt-4o" }
            settings.openaiBaseUrl = openaiUrlInput.text.toString().trim().ifEmpty { "https://api.openai.com/v1" }
            settings.anthropicApiKey = anthropicKeyInput.text.toString().trim()
            settings.anthropicModel = anthropicModelInput.text.toString().trim().ifEmpty { "claude-sonnet-4-20250514" }
            settings.openaiMaxTokens = maxTokensInput.text.toString().toIntOrNull() ?: 8192
            settings.autoApplyChanges = autoApplySwitch.isChecked
            settings.nThreads = threadsInput.text.toString().toIntOrNull() ?: 4
            settings.contextLength = contextInput.text.toString().toIntOrNull() ?: 4096
            syncRuntimeOptions()
            Toast.makeText(context, "Settings saved", Toast.LENGTH_SHORT).show()
            updateModelList()
        }.setNegativeButton("Cancel", null).show()
    }

    private fun importGgufModel() {
        val activity = context as? android.app.Activity ?: return
        val picker = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("application/octet-stream"))
            addCategory(Intent.CATEGORY_OPENABLE)
        }
        activity.startActivityForResult(picker, 0x4247)
    }

    fun onGgufImportResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == 0x4247 && resultCode == android.app.Activity.RESULT_OK) {
            data?.data?.let { uri ->
                val displayName = runCatching {
                    context.contentResolver.query(uri, null, null, null, null).use { cursor ->
                        if (cursor?.moveToFirst() == true) {
                            cursor.getString(cursor.getColumnIndexOrThrow(OpenableColumns.DISPLAY_NAME)) ?: "model.gguf"
                        } else {
                            "model.gguf"
                        }
                    }
                }.getOrElse { "model.gguf" }
                copyAndRegisterModel(uri, displayName)
            }
        }
    }

    private fun copyAndRegisterModel(uri: Uri, displayName: String) {
        Thread {
            try {
                val modelDir = File(context.filesDir, "AI/models").apply { mkdirs() }
                val dest = File(modelDir, displayName)
                context.contentResolver.openInputStream(uri)?.use { input ->
                    dest.outputStream().use { output ->
                        input.copyTo(output)
                    }
                } ?: throw IOException("Failed to open input stream for $uri")
                val size = dest.length()
                modelManager.addModel(displayName.removeSuffix(".gguf"), dest.absolutePath)
                post {
                    Toast.makeText(context, "Model imported: ${displayName} (${formatSize(size)})", Toast.LENGTH_LONG).show()
                    updateModelList()
                }
            } catch (e: Exception) {
                post { Toast.makeText(context, "Import failed: ${e.message}", Toast.LENGTH_SHORT).show() }
            }
        }.start()
    }

    private fun formatSize(bytes: Long): String = when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024 * 1024 -> "${bytes / 1024} KB"
        bytes < 1024 * 1024 * 1024 -> String.format(Locale.US, "%.1f MB", bytes / 1024.0 / 1024.0)
        else -> String.format(Locale.US, "%.2f GB", bytes / 1024.0 / 1024.0 / 1024.0)
    }

    private fun vibrateForApproval() {
        val v = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
        val p = longArrayOf(0, 500, 300, 500)
        v.vibrate(VibrationEffect.createWaveform(p, -1))
    }

    fun updateFileContext(filePath: String?, content: String?, selection: String? = null) {
        engine.setCurrentFileContext(filePath, content, selection)
    }

    private fun dp(px: Int): Int = (px * context.resources.displayMetrics.density).toInt()

    private fun showToast(msg: String) {
        android.widget.Toast.makeText(context, msg, android.widget.Toast.LENGTH_SHORT).show()
    }
}
