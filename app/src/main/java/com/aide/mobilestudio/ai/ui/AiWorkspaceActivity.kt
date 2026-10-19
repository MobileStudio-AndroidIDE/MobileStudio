package com.aide.mobilestudio.ai.ui

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ListPopupWindow
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.aide.mobilestudio.DiffViewerActivity
import com.aide.mobilestudio.ShellEnvironment
import com.aide.mobilestudio.ShellRunner
import com.aide.mobilestudio.StudioUi
import com.aide.mobilestudio.ai.AiSettings
import com.aide.mobilestudio.ai.agent.CheckpointOutcome
import com.aide.mobilestudio.ai.agent.PromptContextBadge
import com.aide.mobilestudio.ai.agent.ProjectFiles
import com.aide.mobilestudio.ai.agent.StatusLine
import com.aide.mobilestudio.ai.command.SlashCommand
import com.aide.mobilestudio.ai.command.SlashCommandAction
import com.aide.mobilestudio.ai.command.SlashCommandRegistry
import com.aide.mobilestudio.ai.command.SlashInvocation
import com.aide.mobilestudio.ai.opencode.agent.OcEngine
import com.aide.mobilestudio.ai.opencode.provider.OcModelInfo
import com.aide.mobilestudio.ai.opencode.session.OcMessage
import com.aide.mobilestudio.ai.opencode.session.OcSessionStore
import com.aide.mobilestudio.ai.provider.AndroidHttpTransport
import com.aide.mobilestudio.ai.provider.AuthStyle
import com.aide.mobilestudio.ai.provider.ConnectionTester
import com.aide.mobilestudio.ai.provider.ProviderEndpoint
import com.aide.mobilestudio.ai.provider.ProviderModelList
import com.aide.mobilestudio.ai.resource.AndroidMemoryProbe
import com.aide.mobilestudio.ai.resource.ResourceManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

/**
 * New AI workspace. Replaces the old chat panel as the primary AI entry point.
 *
 * The screen is a small IDE for the agent: Chat, Plan, Changes, Diagnostics,
 * Sessions, Models, Settings and Terminal. It talks to the same [OcEngine] the
 * old panel used (no engine fork) and adds its own live validation through
 * [WorkspaceValidation] so it works outside MainActivity as well.
 */
class AiWorkspaceActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_PROJECT_DIR = "project_dir"

        /** File currently open in the editor, shown as context in the top bar. */
        const val EXTRA_CURRENT_FILE = "current_file"

        /**
         * Message submitted to the agent as soon as the workspace opens. Used
         * by editor actions that used to type into the old bottom-panel AI
         * ("Explain this file", crash analysis) so they land in the workspace
         * conversation instead.
         */
        const val EXTRA_INITIAL_MESSAGE = "initial_message"

        /**
         * Extra consumed by MainActivity's onCreate ("open_file" → open that
         * absolute path in the editor, optionally with "open_line"). Used by
         * [openInEditor]; no MainActivity change is required for it.
         */
        const val EXTRA_OPEN_IN_EDITOR = "open_file"

        /** Tools whose successful run means a project file was modified. */
        private val FILE_CHANGING_TOOLS = setOf("write", "edit", "patch")

        private const val REQUEST_ATTACHMENT = 4001
        private const val REQUEST_MIC = 4002

        /**
         * Open the full-screen AI workspace - the primary AI surface.
         *
         * @param projectDir  project root the agent operates on; the editor
         *   AI button passes the open project here.
         * @param currentFile file open in the editor, shown in the top bar as
         *   conversational context. Optional with a null default so existing
         *   callers keep compiling; callers wanting file context add the
         *   argument (see [EXTRA_CURRENT_FILE]).
         * @param initialMessage prompt submitted right after the workspace
         *   opens (see [EXTRA_INITIAL_MESSAGE]); null leaves the composer to
         *   the user.
         */
        fun start(
            context: Context,
            projectDir: String? = null,
            currentFile: String? = null,
            initialMessage: String? = null,
        ) {
            val intent = Intent(context, AiWorkspaceActivity::class.java)
            if (projectDir != null) intent.putExtra(EXTRA_PROJECT_DIR, projectDir)
            if (currentFile != null) intent.putExtra(EXTRA_CURRENT_FILE, currentFile)
            if (!initialMessage.isNullOrBlank()) intent.putExtra(EXTRA_INITIAL_MESSAGE, initialMessage)
            context.startActivity(intent)
        }
    }

    private enum class Tab(val label: String, val labelKo: String) {
        FILES("Files", "프로젝트"),
        CHAT("Chat", "채팅"),
        PLAN("Plan", "계획"),
        CHANGES("Changes", "변경"),
        SEARCH("Search", "검색"),
        GIT("Git", "Git"),
        DIAGNOSTICS("Diagnostics", "진단"),
        TOOLS("Tools", "도구"),
        RESOURCES("Resources", "자원"),
        SESSIONS("Sessions", "세션"),
        MODELS("Models", "모델"),
        SETTINGS("Settings", "설정"),
        TERMINAL("Terminal", "터미널"),
    }

    /**
     * The five destinations of the left navigation rail (프로젝트 / 검색 / Git /
     * AI 어시스턴트 / 설정). Several [Tab]s share one rail destination; the AI and
     * settings destinations show a segmented sub-bar to switch between them.
     */
    private enum class Screen(val label: String, val labelKo: String, val icon: String) {
        PROJECT("Files", "프로젝트", "🗂"),
        SEARCH("Search", "검색", "🔍"),
        GIT("Git", "Git", "⑂"),
        AI("AI", "AI", "✦"),
        SETTINGS("Settings", "설정", "⚙"),
    }

    private fun screenOf(tab: Tab): Screen = when (tab) {
        Tab.FILES -> Screen.PROJECT
        Tab.SEARCH -> Screen.SEARCH
        Tab.GIT -> Screen.GIT
        Tab.SETTINGS, Tab.SESSIONS, Tab.MODELS -> Screen.SETTINGS
        else -> Screen.AI
    }

    /**
     * Built in [onCreate], **not** as a field initializer. Activity field
     * initializers run before the framework attaches the base context, so
     * constructing the engine here crashed on device with
     * `getFilesDir() on a null object reference` (OcSessionStore → OcEngine).
     */
    private lateinit var engine: OcEngine
    private lateinit var validation: WorkspaceValidation
    private val settings by lazy { AiSettings(this) }
    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val handler = Handler(Looper.getMainLooper())

    private var projectDir: File? = null

    /** File open in the editor when the workspace was launched, if provided. */
    private var currentFile: File? = null
    private var activeTab = Tab.CHAT
    private val changedFiles = linkedSetOf<String>()
    private val READ_ONLY_HINT = setOf("read", "glob", "grep", "webfetch", "websearch")

    // top bar
    private lateinit var titleText: TextView
    private lateinit var modelText: TextView
    private lateinit var statusText: TextView
    private lateinit var projectText: TextView
    private lateinit var statusPill: TextView
    private lateinit var tokenCounter: TextView
    private lateinit var railRow: LinearLayout
    private lateinit var content: FrameLayout
    private lateinit var statusBar: TextView

    // assistant header chips (kept so pickers can refresh their labels)
    private lateinit var agentChip: TextView
    private lateinit var modelChip: TextView

    // project file browser (프로젝트 rail destination)
    private var browseDir: File? = null
    private lateinit var filesList: LinearLayout

    // git screen output
    private lateinit var gitOutput: TextView

    // composer attachments
    private var pendingImageBase64: String? = null

    // chat (persistent)
    private lateinit var chatScroll: ScrollView
    private lateinit var chatMessages: LinearLayout
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var thinkingView: TextView
    private var streamingView: TextView? = null
    private var streamingCard: LinearLayout? = null

    private var engineState = OcEngine.AgentState.IDLE
    private var thinkDots = 0
    private var thinkBase = "Ready"

    // Generation speed (t/s), ported from the old bottom-panel counter: the
    // clock starts when the engine enters GENERATING and the character count
    // is the length of the streamed reply, which onMessage hands over in full
    // on every chunk. Tokens are estimated the same way as everywhere else in
    // the app - four characters to the token.
    private var rateStartMs = 0L
    private var rateChars = 0L
    private var lastTokensPerSec = 0.0
    private var lastRatePaintMs = 0L
    private val tokensPerChar = 4.0

    // onMessage arrives once per streamed chunk, well above reading speed.
    private val ratePaintIntervalMs = 200L

    // --- slash commands + @file mentions + context badge ---
    private val slashRegistry = SlashCommandRegistry()
    private var slashPopup: ListPopupWindow? = null
    private var mentionPopup: ListPopupWindow? = null
    private var mentionFiles: List<String> = emptyList()
    private var mentionFilesRoot: File? = null
    private var mentionFilesLoading = false
    private var activeMentionToken: String? = null
    private lateinit var contextBadge: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Belt and braces with MobileStudioApp: the workspace is the single AI
        // surface now and must be able to reach the local runtime even if the
        // Application-level init is ever lost.
        com.aide.mobilestudio.ai.LlamaRuntimeManager.initAppContext(applicationContext)
        // The base context is attached by the time onCreate runs, so the
        // engine/storage may safely touch filesDir and SharedPreferences here.
        engine = OcEngine(this)
        validation = WorkspaceValidation(this)
        com.aide.mobilestudio.UiTheme.resolve(this)
        setContentView(buildRoot())

        projectDir = intent.getStringExtra(EXTRA_PROJECT_DIR)?.let { File(it) }?.takeIf { it.isDirectory }
        currentFile = intent.getStringExtra(EXTRA_CURRENT_FILE)?.let { File(it) }?.takeIf { it.isFile }
        engine.projectDir = projectDir
        engine.reloadConfig()
        // The composer belongs to the chat screen and is built lazily by
        // chatRoot() the first time that screen is shown - which happens well
        // after this point. The engine reports its state the instant a session
        // is loaded, though, and loadSession calls back synchronously, from
        // inside resumeLatestSessionOrStartNew() below. That first report
        // therefore landed on a composer that did not exist yet and threw
        // UninitializedPropertyAccessException on sendButton, killing the
        // workspace on every launch that had a stored session to resume. Build
        // the chat views now: chatRoot() reuses them, so the screen switch
        // later on is unchanged.
        buildChatViews()
        setupEngine()
        resumeLatestSessionOrStartNew()
        renderSessionHistory()
        resumedSessionTitle?.let {
            addSystemBubble(t("Resumed previous conversation", "이전 대화를 복원했습니다") + ": " + it)
        }
        showTab(Tab.CHAT)
        updateModelLabel()

        // Editor actions (explain file / crash report) used to type into the
        // old bottom-panel AI; they now hand the prompt over to this screen.
        intent.getStringExtra(EXTRA_INITIAL_MESSAGE)?.takeIf { it.isNotBlank() }?.let { submit(it) }

        if (projectDir != null) {
            scope.launch {
                withContext(Dispatchers.IO) { validation.beginBaseline(projectDir!!) }
                if (activeTab == Tab.DIAGNOSTICS) showTab(Tab.DIAGNOSTICS)
            }
        }
    }

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        runCatching { engine.stop() }
        scope.coroutineContext[kotlinx.coroutines.Job]?.cancel()
        super.onDestroy()
    }

    /**
     * Reopening the workspace must not wipe the conversation: a fresh
     * [OcEngine] starts with no session, so unconditionally calling
     * [OcEngine.newSession] here created a new empty one on every relaunch
     * even though the session store still had the transcript on disk. Resume
     * the newest stored session instead; only first launches (nothing stored
     * yet) start empty.
     */
    private fun resumeLatestSessionOrStartNew() {
        if (engine.currentSessionId() != null) return
        val latest = engine.listSessions(includeArchived = false).firstOrNull()
        if (latest == null) {
            engine.newSession()
        } else {
            engine.loadSession(latest.id)
            resumedSessionTitle = latest.title
        }
    }

    /** Title of the stored session resumed on launch, shown once after render. */
    private var resumedSessionTitle: String? = null

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    // ---------------------------------------------------------------- root UI

    private fun buildRoot(): View {
        val p = StudioUi.palette
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        }
        root.addView(buildTopBar())
        val body = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        body.addView(buildNavRail(), LinearLayout.LayoutParams(dp(62), ViewGroup.LayoutParams.MATCH_PARENT))
        content = FrameLayout(this).apply { setBackgroundColor(p.bg) }
        body.addView(
            content,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT),
        )
        root.addView(body, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        statusBar = TextView(this).apply {
            setTextColor(p.muted)
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setPadding(dp(12), dp(4), dp(12), dp(4))
            background = AiUiKit.rounded(p.surface, 0)
        }
        root.addView(statusBar, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        updateStatusBar()
        return root
    }

    private fun updateStatusBar() {
        if (!::statusBar.isInitialized) return
        val state = when (engineState) {
            OcEngine.AgentState.THINKING -> t("Thinking", "생각 중")
            OcEngine.AgentState.GENERATING -> t("Writing", "작성 중")
            OcEngine.AgentState.EXECUTING_TOOL, OcEngine.AgentState.PROCESSING_TOOL_RESULT -> t("Working", "작업 중")
            OcEngine.AgentState.WAITING_TOOL_APPROVAL -> t("Approval", "승인 대기")
            OcEngine.AgentState.ERROR -> t("Error", "오류")
            OcEngine.AgentState.CANCELLED -> t("Cancelled", "취소")
            else -> t("Ready", "준비")
        }
        statusBar.text = "$state  ·  ${engine.estimateTokens()} tok  ·  ${engine.currentModelLabel()}  ·  UTF-8"
    }

    private fun buildTopBar(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = AiUiKit.rounded(p.surface, 0)
            setPadding(dp(14), dp(12), dp(14), dp(8))
        }
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row.addView(AiUiKit.circle(this, "AI", p.accent, p.onAccent, 30))
        titleText = TextView(this).apply {
            text = t("AI Workspace", "AI 작업공간")
            setTextColor(p.text)
            textSize = StudioUi.TypeHeading
            setTypeface(null, Typeface.BOLD)
            setPadding(dp(10), 0, dp(8), 0)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        projectText = AiUiKit.pill(this, "", p.surface2, p.textSoft)
        row.addView(titleText)
        row.addView(projectText)
        // Quick navigation + build actions, as in the IDE design (검색 / 실행 / 정리).
        row.addView(
            AiUiKit.actionChip(this, "🔍") { _ -> showTab(Tab.SEARCH) },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginStart = dp(6) },
        )
        val runButton = AiUiKit.pill(this, "▶", AiUiKit.withAlpha(p.success, 0.85f), p.onAccent)
        runButton.setOnClickListener { runGradleBuild() }
        row.addView(
            runButton,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginStart = dp(6) },
        )
        row.addView(
            AiUiKit.actionChip(this, "🧹") { _ -> runGradleClean() },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginStart = dp(6) },
        )
        // Persistent escape hatch on every tab: finish the workspace and drop
        // back on the editor exactly as it was. The transcript lives in the
        // session store, so reopening the workspace resumes it (see
        // resumeLatestSessionOrStartNew) and scrolls to the bottom.
        row.addView(
            AiUiKit.actionChip(this, t("‹ Editor", "‹ 편집기")) { _ -> finish() },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginStart = dp(8) },
        )
        column.addView(row)

        val row2 = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(8), 0, 0)
        }
        modelText = TextView(this).apply {
            setTextColor(p.accent)
            textSize = StudioUi.TypeSmall
            setOnClickListener { showModelPicker() }
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        tokenCounter = TextView(this).apply {
            this.text = "— t/s"
            setTextColor(p.muted)
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setPadding(dp(6), 0, dp(8), 0)
        }
        statusPill = AiUiKit.pill(this, t("Ready", "대기"), p.surface2, p.textSoft)
        row2.addView(modelText)
        row2.addView(tokenCounter)
        row2.addView(statusPill)
        column.addView(row2)

        statusText = TextView(this).apply { visibility = View.GONE }
        column.addView(statusText)
        updateProjectChip()
        return column
    }

    private fun updateProjectChip() {
        val name = projectDir?.name ?: t("No project", "프로젝트 없음")
        val file = currentFile?.name
        projectText.text = if (file != null) "$name · $file" else name
    }

    private fun updateStatusPill(text: String, color: Int) {
        statusPill.text = text
        statusPill.setTextColor(color)
        statusPill.background = AiUiKit.rounded(AiUiKit.withAlpha(color, 0.15f), dp(999))
    }

    private fun buildNavRail(): View {
        val p = StudioUi.palette
        railRow = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = AiUiKit.rounded(p.surface, 0)
            setPadding(dp(4), dp(10), dp(4), dp(10))
        }
        for (screen in Screen.entries) {
            val item = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(2), dp(9), dp(2), dp(9))
                tag = screen
                setOnClickListener {
                    when (screen) {
                        Screen.PROJECT -> showTab(Tab.FILES)
                        Screen.SEARCH -> showTab(Tab.SEARCH)
                        Screen.GIT -> showTab(Tab.GIT)
                        Screen.AI -> showTab(Tab.CHAT)
                        Screen.SETTINGS -> showTab(Tab.SETTINGS)
                    }
                }
            }
            item.addView(TextView(this).apply {
                text = screen.icon
                textSize = 16f
                gravity = Gravity.CENTER
            })
            item.addView(TextView(this).apply {
                text = t(screen.label, screen.labelKo)
                textSize = 9f
                gravity = Gravity.CENTER
                setTextColor(p.textSoft)
            })
            railRow.addView(
                item,
                LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT),
            )
        }
        return railRow
    }

    private fun highlightRail() {
        val p = StudioUi.palette
        if (!::railRow.isInitialized) return
        val active = screenOf(activeTab)
        for (i in 0 until railRow.childCount) {
            val item = railRow.getChildAt(i) as LinearLayout
            val selected = item.tag == active
            item.background = if (selected) AiUiKit.rounded(AiUiKit.withAlpha(p.accent, 0.18f), dp(10)) else null
            (item.getChildAt(0) as TextView).setTextColor(if (selected) p.accent else p.textSoft)
            (item.getChildAt(1) as TextView).setTextColor(if (selected) p.text else p.textSoft)
        }
    }

    private fun segmentedRow(items: List<Triple<String, Boolean, () -> Unit>>): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(10), dp(2), dp(10), dp(4))
        }
        for ((label, selected, onClick) in items) {
            val tv = TextView(this).apply {
                text = label
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setPadding(dp(14), dp(7), dp(14), dp(7))
                background = AiUiKit.rippleContent(
                    this@AiWorkspaceActivity,
                    if (selected) StudioUi.palette.accent else StudioUi.palette.surface2,
                    dp(999),
                )
                setTextColor(if (selected) StudioUi.palette.onAccent else StudioUi.palette.textSoft)
                setOnClickListener { onClick() }
            }
            row.addView(
                tv,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { marginEnd = dp(6) },
            )
        }
        return row
    }

    private enum class AiGroup { CHAT, PLAN, PROJECT, TOOLS }

    private fun aiGroupOf(tab: Tab): AiGroup = when (tab) {
        Tab.PLAN -> AiGroup.PLAN
        Tab.CHANGES, Tab.DIAGNOSTICS -> AiGroup.PROJECT
        Tab.TOOLS, Tab.RESOURCES, Tab.TERMINAL -> AiGroup.TOOLS
        else -> AiGroup.CHAT
    }

    private fun buildAiHeader(): View {
        val p = StudioUi.palette
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(4))
        }
        row.addView(TextView(this).apply {
            text = t("AI Assistant", "AI 어시스턴트")
            setTextColor(p.text)
            textSize = 15f
            setTypeface(null, Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        // Start a fresh conversation without hunting through the Sessions tab.
        row.addView(
            AiUiKit.actionChip(this, "＋") { _ -> newChat() }.apply {
                contentDescription = t("New chat", "새 대화")
            },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginEnd = dp(6) },
        )
        // Agent mode (build / plan / ...) — OpenCode-style agent switching.
        agentChip = AiUiKit.pill(this, engine.currentAgentName(), AiUiKit.withAlpha(p.accent, 0.15f), p.accent)
        agentChip.setOnClickListener { showAgentPicker() }
        row.addView(
            agentChip,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginEnd = dp(6) },
        )
        // On-demand conversation compaction.
        row.addView(
            AiUiKit.actionChip(this, t("Compact", "압축")) { _ ->
                scope.launch {
                    val ok = runCatching { engine.compactNow() }.getOrDefault(false)
                    Toast.makeText(
                        this@AiWorkspaceActivity,
                        if (ok) t("Conversation compacted", "대화를 압축했습니다") else t("Compaction unavailable", "압축을 사용할 수 없습니다"),
                        Toast.LENGTH_SHORT,
                    ).show()
                }
            },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginEnd = dp(6) },
        )
        modelChip = AiUiKit.pill(this, compactModelLabel(), p.surface2, p.accent)
        modelChip.setOnClickListener { showModelPicker() }
        row.addView(modelChip)
        return row
    }

    private fun showAgentPicker() {
        val agents = runCatching { engine.listAgents() }.getOrDefault(emptyList())
        if (agents.isEmpty()) {
            Toast.makeText(this, t("No agents available", "사용 가능한 에이전트가 없습니다"), Toast.LENGTH_SHORT).show()
            return
        }
        val names = agents.map { it.name }.toTypedArray()
        val current = engine.currentAgentName()
        AlertDialog.Builder(this)
            .setTitle(t("Agent", "에이전트"))
            .setSingleChoiceItems(names, names.indexOf(current)) { dialog, which ->
                engine.setAgent(names[which])
                if (::agentChip.isInitialized) agentChip.text = engine.currentAgentName()
                dialog.dismiss()
                showTab(Tab.CHAT)
            }
            .setNegativeButton(t("Cancel", "취소"), null)
            .show()
    }

    /**
     * Model picker for the header chip: switches to the unified Models tab,
     * where every provider's models sit in one list with context limits and
     * the current one checked, next to keys, endpoints and fetch.
     */
    private fun showModelPicker() {
        // The chip opens the unified picker tab: API models, installed GGUFs
        // and Hugging Face downloads in one place, with a safe switch that
        // reverts to the previous model when the new one cannot be created.
        showTab(Tab.MODELS)
    }

    /** Model label short enough for the assistant header chip. */
    private fun compactModelLabel(): String {
        val ref = runCatching { engine.currentModelRef() }.getOrNull() ?: return engine.currentModelLabel()
        val id = ref.modelID.substringAfterLast('/')
        val short = if (id.length > 18) id.take(17) + "…" else id
        return "${ref.providerID}/$short"
    }

    /** Fresh conversation: stop the current turn, new engine session, empty chat. */
    private fun newChat() {
        runCatching { engine.stop() }
        runCatching { engine.newSession() }
        changedFiles.clear()
        runCatching { validation.reset() }
        streamingView = null
        streamingCard = null
        chatMessages.removeAllViews()
        thinkingView.visibility = View.GONE
        contextBadge.visibility = View.GONE
        renderSessionHistory()
        resetTokenCounter()
        showTab(Tab.CHAT)
    }

    private fun resetTokenCounter() {
        rateStartMs = 0L
        rateChars = 0L
        lastTokensPerSec = 0.0
        if (::tokenCounter.isInitialized) tokenCounter.text = "— t/s"
    }

    private fun updateTokenCount() {
        if (!::tokenCounter.isInitialized) return
        val live = rateStartMs > 0
        val rate = if (live) {
            val elapsedSec = (System.currentTimeMillis() - rateStartMs).coerceAtLeast(1L) / 1000.0
            (rateChars / tokensPerChar) / elapsedSec
        } else {
            lastTokensPerSec
        }
        tokenCounter.text = if (rate > 0) {
            String.format(Locale.US, "%.1f t/s", rate)
        } else {
            "— t/s"
        }
    }

    /** AI destination: header + (채팅/계획/프로젝트/도구) segmented bar + content. */
    private fun aiScreen(tab: Tab): View {
        val p = StudioUi.palette
        val group = aiGroupOf(tab)
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        column.addView(buildAiHeader())
        column.addView(
            segmentedRow(
                listOf(
                    Triple(t("Chat", "채팅"), group == AiGroup.CHAT) { showTab(Tab.CHAT) },
                    Triple(t("Plan", "계획"), group == AiGroup.PLAN) { showTab(Tab.PLAN) },
                    Triple(
                        t("Project", "프로젝트"),
                        group == AiGroup.PROJECT,
                    ) { showTab(if (tab == Tab.DIAGNOSTICS) Tab.DIAGNOSTICS else Tab.CHANGES) },
                    Triple(
                        t("Tools", "도구"),
                        group == AiGroup.TOOLS,
                    ) { showTab(if (tab == Tab.TERMINAL) Tab.TERMINAL else Tab.TOOLS) },
                ),
            ),
        )
        when (group) {
            AiGroup.PROJECT -> column.addView(
                segmentedRow(
                    listOf(
                        Triple(t("Changes", "변경"), tab == Tab.CHANGES) { showTab(Tab.CHANGES) },
                        Triple(t("Diagnostics", "진단"), tab == Tab.DIAGNOSTICS) { showTab(Tab.DIAGNOSTICS) },
                    ),
                ),
            )
            AiGroup.TOOLS -> column.addView(
                segmentedRow(
                    listOf(
                        Triple(t("Tools", "도구"), tab == Tab.TOOLS) { showTab(Tab.TOOLS) },
                        Triple(t("Resources", "자원"), tab == Tab.RESOURCES) { showTab(Tab.RESOURCES) },
                        Triple(t("Terminal", "터미널"), tab == Tab.TERMINAL) { showTab(Tab.TERMINAL) },
                    ),
                ),
            )
            else -> {}
        }
        val frame = FrameLayout(this)
        frame.addView(
            when (tab) {
                Tab.PLAN -> buildPlanTab()
                Tab.CHANGES -> buildChangesTab()
                Tab.DIAGNOSTICS -> buildDiagnosticsTab()
                Tab.TOOLS -> buildToolsTab()
                Tab.RESOURCES -> buildResourcesTab()
                Tab.TERMINAL -> buildTerminalTab()
                else -> chatRoot()
            },
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT),
        )
        column.addView(frame, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        return column
    }

    /** Settings destination: (설정/세션/모델) segmented bar + content. */
    private fun settingsScreen(tab: Tab): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        column.addView(
            segmentedRow(
                listOf(
                    Triple(t("Settings", "설정"), tab == Tab.SETTINGS) { showTab(Tab.SETTINGS) },
                    Triple(t("Sessions", "세션"), tab == Tab.SESSIONS) { showTab(Tab.SESSIONS) },
                    Triple(t("Models", "모델"), tab == Tab.MODELS) { showTab(Tab.MODELS) },
                ),
            ),
        )
        val frame = FrameLayout(this)
        frame.addView(
            when (tab) {
                Tab.SESSIONS -> buildSessionsTab()
                Tab.MODELS -> buildModelsTab()
                else -> buildSettingsTab()
            },
            FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT),
        )
        column.addView(frame, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        return column
    }

    private fun showTab(tab: Tab) {
        activeTab = tab
        highlightRail()
        content.removeAllViews()
        content.addView(
            when (screenOf(tab)) {
                Screen.AI -> aiScreen(tab)
                Screen.SETTINGS -> settingsScreen(tab)
                else -> when (tab) {
                    Tab.FILES -> buildFilesTab()
                    Tab.GIT -> buildGitTab()
                    else -> buildSearchTab()
                }
            },
        )
        updateStatusBar()
    }

    // ---------------------------------------------------------------- Chat tab

    private fun chatRoot(): View {
        if (!::chatWrapper.isInitialized) buildChatViews()
        (chatWrapper.parent as? ViewGroup)?.removeView(chatWrapper)
        return chatWrapper
    }

    private fun buildChatViews() {
        val p = StudioUi.palette
        chatWrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        chatMessages = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(8), dp(8), dp(8), dp(8))
        }
        chatScroll = ScrollView(this).apply { addView(chatMessages) }
        chatWrapper.addView(chatScroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        thinkingView = TextView(this).apply {
            setTextColor(p.textSoft)
            textSize = StudioUi.TypeSmall
            setPadding(dp(12), dp(4), dp(12), dp(4))
            visibility = View.GONE
        }
        chatWrapper.addView(thinkingView)

        // Context badge: compact line showing what the next request carries
        // (open editor file, @mention count, rough token size).
        contextBadge = TextView(this).apply {
            setTextColor(p.muted)
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setPadding(dp(14), dp(2), dp(14), dp(2))
            visibility = View.GONE
        }
        chatWrapper.addView(contextBadge)
        chatWrapper.addView(buildQuickActions())

        chatWrapper.addView(buildInputRow())
    }

    /**
     * One-tap prompts under the assistant header (프로젝트 생성 / 문제 해결 /
     * 리팩토링 / 파일 생성 / 분석 / 테스트), matching the IDE design.
     */
    private fun buildQuickActions(): View {
        val actions = listOf(
            t("Create project", "프로젝트 생성하기") to t(
                "Create a new project here. Ask me for the details you need first.",
                "여기에 새 프로젝트를 만들어줘. 필요한 정보를 먼저 물어봐줘.",
            ),
            t("Fix problems", "문제 해결하기") to t(
                "Check the current diagnostics and fix the errors and warnings.",
                "현재 진단 오류와 경고를 확인하고 수정해줘.",
            ),
            t("Refactor", "리팩토링하기") to t(
                "Refactor this project. Propose a plan first and wait for my approval.",
                "이 프로젝트를 리팩토링해줘. 먼저 계획을 제시하고 승인을 받은 뒤 진행해줘.",
            ),
            t("New file", "파일 생성하기") to t(
                "Create a new file. Ask me what to create first.",
                "새 파일을 만들어줘. 무엇을 만들지 먼저 물어봐줘.",
            ),
            t("Analyze project", "프로젝트 분석하기") to t(
                "Analyze the project structure, dependencies and build setup, then summarise it.",
                "프로젝트 구조, 의존성, 빌드 설정을 분석하고 요약해줘.",
            ),
            t("Write tests", "테스트 작성하기") to t(
                "Write unit tests for the core code of this project.",
                "이 프로젝트의 핵심 코드에 대한 단위 테스트를 작성해줘.",
            ),
        )
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(10), dp(4), dp(10), dp(4))
        }
        var row: LinearLayout? = null
        for ((index, action) in actions.withIndex()) {
            if (index % 3 == 0) {
                row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                column.addView(
                    row,
                    LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT),
                )
            }
            row!!.addView(
                AiUiKit.actionChip(this, action.first) { _ -> submit(action.second) },
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { setMargins(0, dp(3), dp(6), dp(3)) },
            )
        }
        return column
    }

    private lateinit var chatWrapper: LinearLayout

    private fun buildInputRow(): View {
        val p = StudioUi.palette
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = AiUiKit.rounded(p.surface, 0)
            setPadding(dp(10), dp(8), dp(10), dp(8))
        }
        input = EditText(this).apply {
            hint = t("Ask the agent…", "에이전트에게 요청…")
            setHintTextColor(p.muted)
            setTextColor(p.text)
            textSize = StudioUi.TypeBody
            maxLines = 6
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
            imeOptions = EditorInfo.IME_ACTION_SEND
            background = AiUiKit.rounded(p.surface2, dp(14), strokeColor = p.border, strokePx = dp(1))
            setPadding(dp(14), dp(10), dp(14), dp(10))
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) { send(); true } else false
            }
            // Multi-line composer: Enter sends, Shift+Enter (hardware
            // keyboards) inserts a newline. Without this the IME's Enter key
            // only ever produced newlines and the send action never fired.
            setOnKeyListener { _, keyCode, event ->
                if (keyCode == android.view.KeyEvent.KEYCODE_ENTER &&
                    event.action == android.view.KeyEvent.ACTION_DOWN &&
                    !event.isShiftPressed
                ) {
                    send(); true
                } else false
            }
            addTextChangedListener(object : android.text.TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
                override fun afterTextChanged(s: android.text.Editable?) {
                    val text = s?.toString().orEmpty()
                    updateSlashSuggestions(text)
                    updateMentionSuggestions(text, input.selectionStart)
                    updateContextBadge(text)
                }
            })
        }
        sendButton = Button(this).apply {
            text = t("Send", "전송")
            isAllCaps = false
            setTextColor(p.onAccent)
            background = AiUiKit.rippleContent(this@AiWorkspaceActivity, p.accent, dp(14))
            setOnClickListener {
                if (engineState == OcEngine.AgentState.GENERATING || engineState == OcEngine.AgentState.THINKING) engine.stop()
                else send()
            }
        }
        row.addView(
            AiUiKit.actionChip(this, "📎") { _ -> pickAttachment() },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginEnd = dp(4) },
        )
        row.addView(
            AiUiKit.actionChip(this, "🎤") { _ -> startVoiceInput() },
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { marginEnd = dp(4) },
        )
        row.addView(input)
        row.addView(sendButton, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(8) })
        return row
    }

    private fun send() {
        submit(input.text?.toString()?.trim().orEmpty())
    }

    /**
     * Sends [text] (plus any pending attachment) to the engine after the local
     * slash-command pass. Shared by the composer, quick-action chips and file
     * actions so everything uses one entry point.
     */
    private fun submit(text: String) {
        val image = pendingImageBase64
        if (text.isBlank() && image == null) return
        if (text.startsWith("/") && image == null && handleSlashInput(text)) return
        input.setText("")
        pendingImageBase64 = null
        addUserBubble(if (image != null) "🖼 " + text.ifBlank { t("(image attached)", "(이미지 첨부)") } else text)
        streamingView = null
        streamingCard = null
        engine.sendMessage(
            text.ifBlank { t("What is in this image?", "이 이미지에는 무엇이 있나요?") },
            imageBase64 = image,
        )
    }

    // -------------------------------------------------------- attach / voice

    private fun pickAttachment() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("image/*", "text/*", "application/json"))
        }
        runCatching { startActivityForResult(intent, REQUEST_ATTACHMENT) }
            .onFailure {
                Toast.makeText(this, t("No file picker available", "파일 선택기를 사용할 수 없습니다"), Toast.LENGTH_SHORT).show()
            }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_ATTACHMENT || resultCode != RESULT_OK) return
        val uri = data?.data ?: return
        runCatching {
            val mime = contentResolver.getType(uri).orEmpty()
            if (mime.startsWith("image/")) {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                if (bytes == null) {
                    Toast.makeText(this, t("Could not read file", "파일을 읽을 수 없습니다"), Toast.LENGTH_SHORT).show()
                    return@runCatching
                }
                pendingImageBase64 = android.util.Base64.encodeToString(bytes, android.util.Base64.NO_WRAP)
                Toast.makeText(this, t("Image attached", "이미지 첨부됨"), Toast.LENGTH_SHORT).show()
            } else {
                val text = contentResolver.openInputStream(uri)?.use { it.bufferedReader().readText() }
                if (text == null) {
                    Toast.makeText(this, t("Could not read file", "파일을 읽을 수 없습니다"), Toast.LENGTH_SHORT).show()
                    return@runCatching
                }
                val excerpt = text.take(4000)
                val label = uri.lastPathSegment ?: "file"
                input.setText(
                    input.text?.toString().orEmpty() + "\n\n" +
                        t("[Attached $label]", "[첨부 $label]") + "\n```\n$excerpt\n```",
                )
                Toast.makeText(this, t("File attached", "파일 첨부됨"), Toast.LENGTH_SHORT).show()
            }
        }.onFailure {
            Toast.makeText(this, t("Attach failed: ${it.message}", "첨부 실패: ${it.message}"), Toast.LENGTH_SHORT).show()
        }
    }

    private fun startVoiceInput() {
        val available = runCatching { android.speech.SpeechRecognizer.isRecognitionAvailable(this) }.getOrDefault(false)
        if (!available) {
            Toast.makeText(this, t("Speech recognition is not available", "음성 인식을 사용할 수 없습니다"), Toast.LENGTH_SHORT).show()
            return
        }
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), REQUEST_MIC)
            return
        }
        listen()
    }

    private fun listen() {
        val recognizer = android.speech.SpeechRecognizer.createSpeechRecognizer(this)
        val intent = android.content.Intent(android.speech.RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE_MODEL, android.speech.RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault().language)
        }
        recognizer.setRecognitionListener(object : android.speech.RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                Toast.makeText(this@AiWorkspaceActivity, t("Listening…", "듣는 중…"), Toast.LENGTH_SHORT).show()
            }

            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onError(error: Int) {
                Toast.makeText(this@AiWorkspaceActivity, t("Voice input failed", "음성 입력 실패"), Toast.LENGTH_SHORT).show()
                recognizer.destroy()
            }

            override fun onResults(results: Bundle?) {
                val text = results
                    ?.getStringArrayList(android.speech.SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.firstOrNull().orEmpty()
                if (text.isNotBlank()) {
                    input.setText(text)
                    input.setSelection(text.length)
                }
                recognizer.destroy()
            }

            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}
        })
        recognizer.startListening(intent)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_MIC && grantResults.firstOrNull() == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            listen()
        }
    }

    /**
     * Intercepts slash input before sending. Returns true when the text was
     * consumed by a local command or an inline notice (unknown command);
     * false when it should still flow to the engine (template commands).
     */
    private fun handleSlashInput(text: String): Boolean {
        return when (val invocation = slashRegistry.parse(text)) {
            is SlashInvocation.Known -> {
                input.setText("")
                runSlashCommand(invocation.command, invocation.argument)
                true
            }
            is SlashInvocation.Unknown -> {
                // Template commands (init, review, skills...) still go through
                // the engine; anything else becomes an inline notice.
                val knownTemplate = runCatching {
                    engine.listCommands().any { it.name.equals(invocation.name, ignoreCase = true) }
                }.getOrDefault(false)
                if (knownTemplate) false else {
                    input.setText("")
                    addSystemBubble(
                        t(
                            "Unknown command: /${invocation.name} — type / to see commands.",
                            "알 수 없는 명령어: /${invocation.name} — / 를 입력하면 명령어 목록이 표시됩니다.",
                        ),
                    )
                    true
                }
            }
            SlashInvocation.NotACommand -> false
        }
    }

    private fun runSlashCommand(command: SlashCommand, argument: String) {
        when (command.action) {
            SlashCommandAction.HELP -> addSystemBubble(SlashCommandRegistry.helpText())
            SlashCommandAction.CLEAR -> confirmClearConversation()
            SlashCommandAction.NEW_SESSION -> newChat()
            SlashCommandAction.SHOW_SESSIONS -> showTab(Tab.SESSIONS)
            SlashCommandAction.SHOW_MODELS -> showTab(Tab.MODELS)
            SlashCommandAction.TOGGLE_PLAN -> togglePlanMode()
            SlashCommandAction.UNDO -> confirmUndo()
            SlashCommandAction.REDO -> performRedo()
            SlashCommandAction.SHOW_CHANGES -> showTab(Tab.CHANGES)
            SlashCommandAction.STATUS -> addSystemBubble(
                StatusLine.format(
                    project = projectDir?.absolutePath,
                    openFile = currentFile?.absolutePath,
                    model = engine.currentModelLabel(),
                    sessionId = engine.currentSessionId(),
                    agent = engine.agentName,
                ),
            )
        }
    }

    private fun togglePlanMode() {
        val enabling = engine.agentName != "plan"
        engine.setAgentMode(if (enabling) "plan" else "build")
        addSystemBubble(
            if (enabling) {
                t("Plan mode ON — the agent can only read and plan.", "계획 모드 켜짐 — 에이전트는 읽기와 계획만 할 수 있습니다.")
            } else {
                t("Plan mode OFF — back to build.", "계획 모드 꺼짐 — 빌드 모드로 돌아갑니다.")
            },
        )
        if (enabling) showTab(Tab.PLAN)
    }

    private fun confirmClearConversation() {
        AlertDialog.Builder(this)
            .setTitle(t("Clear conversation?", "대화를 지울까요?"))
            .setMessage(
                t(
                    "This clears the visible conversation and starts a new session. The previous session stays in Sessions.",
                    "화면의 대화를 지우고 새 세션을 시작합니다. 이전 세션은 세션 목록에 남아 있습니다.",
                ),
            )
            .setPositiveButton(t("Clear", "지우기")) { _, _ ->
                newChat()
                addSystemBubble(t("Conversation cleared.", "대화가 지워졌습니다."))
            }
            .setNegativeButton(t("Cancel", "취소"), null)
            .show()
    }

    /** /undo: confirm dialog listing what will be restored (path + line delta). */
    private fun confirmUndo() {
        scope.launch {
            val deltas = withContext(Dispatchers.IO) { engine.pendingUndoDeltas() }
            if (deltas.isEmpty()) {
                addSystemBubble(t("Nothing to undo.", "되돌릴 변경이 없습니다."))
                return@launch
            }
            val listing = deltas.joinToString("\n") { d ->
                val sign = if (d.lineDelta >= 0) "+" else ""
                val created = if (d.existedBefore) "" else t(" (created)", " (새 파일)")
                "${relativePath(d.path)}  ($sign${d.lineDelta} lines)$created"
            }
            AlertDialog.Builder(this@AiWorkspaceActivity)
                .setTitle(t("Undo the agent's last change?", "에이전트의 마지막 변경을 되돌릴까요?"))
                .setMessage(listing)
                .setPositiveButton(t("Undo", "되돌리기")) { _, _ -> performUndo() }
                .setNegativeButton(t("Cancel", "취소"), null)
                .show()
        }
    }

    private fun performUndo() {
        scope.launch {
            when (val outcome = withContext(Dispatchers.IO) { engine.undoLastTurn() }) {
                is CheckpointOutcome.Undone ->
                    addSystemBubble(t("Undid ${outcome.files.size} file(s).", "${outcome.files.size}개 파일을 되돌렸습니다."))
                CheckpointOutcome.NothingToUndo ->
                    addSystemBubble(t("Nothing to undo.", "되돌릴 변경이 없습니다."))
                CheckpointOutcome.NothingToRedo -> {}
                is CheckpointOutcome.Redone -> {}
                is CheckpointOutcome.Failed ->
                    addSystemBubble("⚠ " + t("Undo failed: ", "되돌리기 실패: ") + outcome.message)
            }
        }
    }

    private fun performRedo() {
        scope.launch {
            when (val outcome = withContext(Dispatchers.IO) { engine.redoLastTurn() }) {
                is CheckpointOutcome.Redone ->
                    addSystemBubble(t("Re-applied ${outcome.files.size} file(s).", "${outcome.files.size}개 파일을 다시 적용했습니다."))
                CheckpointOutcome.NothingToRedo ->
                    addSystemBubble(t("Nothing to redo.", "다시 적용할 변경이 없습니다."))
                CheckpointOutcome.NothingToUndo -> {}
                is CheckpointOutcome.Undone -> {}
                is CheckpointOutcome.Failed ->
                    addSystemBubble("⚠ " + t("Redo failed: ", "다시 적기 실패: ") + outcome.message)
            }
        }
    }

    private fun updateContextBadge(text: String) {
        val label = engine.currentFileContextPath()
        val badge = PromptContextBadge.format(label, text)
        contextBadge.text = badge
        contextBadge.visibility = if (badge.isEmpty()) View.GONE else View.VISIBLE
    }

    // ------------------------------------------------- "/" command suggestions

    /**
     * Filterable "/" popup: local commands first, then the engine's template
     * commands (init, review, skills...). Typing filters; selecting a local
     * command runs it, selecting a template command fills the input.
     */
    private fun updateSlashSuggestions(text: String) {
        val body = text.trimStart()
        if (!body.startsWith("/") || body.contains(' ')) {
            slashPopup?.dismiss()
            return
        }
        val query = body.substring(1)
        val local = slashRegistry.filter(query)
        val templates = runCatching { engine.listCommands() }.getOrDefault(emptyList()).filter { cmd ->
            cmd.name.startsWith(query, ignoreCase = true) ||
                (cmd.description ?: "").contains(query, ignoreCase = true)
        }
        if (local.isEmpty() && templates.isEmpty()) {
            slashPopup?.dismiss()
            return
        }
        val labels = ArrayList<String>(local.size + templates.size)
        local.forEach { labels.add("/${it.name} — ${it.description}") }
        templates.forEach { labels.add("/${it.name} — ${it.description ?: "template command"}") }

        val popup = slashPopup ?: ListPopupWindow(this).apply {
            anchorView = input
            isModal = false
            setOnItemClickListener { _, _, position, _ ->
                slashPopup?.dismiss()
                if (position < local.size) {
                    val command = local[position]
                    input.setText("")
                    runSlashCommand(command, "")
                } else {
                    val cmd = templates.getOrNull(position - local.size) ?: return@setOnItemClickListener
                    input.setText("/${cmd.name} ")
                    input.setSelection(input.text.length)
                }
            }
            slashPopup = this
        }
        popup.setContentWidth(input.width.coerceAtLeast(dp(220)))
        popup.setAdapter(android.widget.ArrayAdapter(this, android.R.layout.simple_list_item_1, labels))
        runCatching { popup.show() }
    }

    // -------------------------------------------------- @file mention picker

    /**
     * Shows the project-file picker while the caret is inside an "@path"
     * token. Selection replaces the token with "@relative/path ".
     */
    private fun updateMentionSuggestions(text: String, caret: Int) {
        val token = mentionTokenAt(text, caret)
        if (token == null) {
            mentionPopup?.dismiss()
            return
        }
        ensureMentionFiles {
            val query = token.removePrefix("@")
            val matches = ProjectFiles.filter(mentionFiles, query)
            if (matches.isEmpty()) {
                mentionPopup?.dismiss()
                return@ensureMentionFiles
            }
            activeMentionToken = token
            val labels = matches.map { "@$it" }
            val popup = mentionPopup ?: ListPopupWindow(this).apply {
                anchorView = input
                isModal = false
                setOnItemClickListener { _, _, position, _ ->
                    mentionPopup?.dismiss()
                    val relative = matches.getOrNull(position) ?: return@setOnItemClickListener
                    val current = activeMentionToken ?: return@setOnItemClickListener
                    val editable = input.text ?: return@setOnItemClickListener
                    val start = (editable.length - current.length).coerceAtLeast(0)
                    editable.replace(start, editable.length, "@$relative ")
                    input.setSelection(editable.length)
                    activeMentionToken = null
                }
                mentionPopup = this
            }
            popup.setContentWidth(input.width.coerceAtLeast(dp(220)))
            popup.setAdapter(android.widget.ArrayAdapter(this, android.R.layout.simple_list_item_1, labels))
            runCatching { popup.show() }
        }
    }

    /** The "@..." token ending at [caret], or null when none is being typed. */
    private fun mentionTokenAt(text: String, caret: Int): String? {
        if (text.isEmpty()) return null
        val end = if (caret in 1..text.length) caret else text.length
        val upToCaret = text.substring(0, end)
        val start = upToCaret.lastIndexOf('@')
        if (start < 0) return null
        if (start > 0) {
            val prev = upToCaret[start - 1]
            // emails and mid-word @ are not mentions
            if (prev.isLetterOrDigit() || prev == '_' || prev == '.' || prev == '/') return null
        }
        val token = upToCaret.substring(start)
        if (token.contains(' ') || token.contains('\n')) return null
        return if (token.length >= 1) token else null
    }

    private fun ensureMentionFiles(onReady: () -> Unit) {
        val root = engine.workspaceRoot()
        if (mentionFilesRoot == root && mentionFiles.isNotEmpty()) {
            onReady()
            return
        }
        if (mentionFilesLoading) return
        mentionFilesLoading = true
        scope.launch {
            val files = withContext(Dispatchers.IO) { ProjectFiles.list(root) }
            mentionFiles = files
            mentionFilesRoot = root
            mentionFilesLoading = false
            onReady()
        }
    }

    private fun addUserBubble(text: String) {
        val p = StudioUi.palette
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = AiUiKit.rounded(p.accent, dp(12))
            setPadding(dp(12), dp(9), dp(12), dp(9))
        }
        card.addView(TextView(this).apply {
            this.text = text
            setTextColor(p.onAccent)
            textSize = StudioUi.TypeBody
            setLineSpacing(dp(2).toFloat(), 1.05f)
            setTextIsSelectable(true)
        })
        // Speech bubble: WRAP_CONTENT card inside a MATCH_PARENT horizontal row
        // aligned to the end, so it hugs the right edge and is capped at ~85%
        // of the row. (A width-0 weighted child of the VERTICAL chatMessages
        // list measures 0 px wide - weight distributes height there.)
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }
        row.addView(
            card,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { setMargins(dp(8), dp(6), dp(8), dp(6)) },
        )
        chatMessages.addView(
            row,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT),
        )
        capUserBubbleWidthOnceLaidOut(row, card)
        scrollChatToBottom()
    }

    private fun assistantCard(): Pair<LinearLayout, TextView> {
        val p = StudioUi.palette
        val content = TextView(this).apply {
            setTextColor(p.text)
            textSize = StudioUi.TypeBody
            setLineSpacing(dp(3).toFloat(), 1.08f)
            setTextIsSelectable(true)
        }
        val card = AiUiKit.card(this, radiusDp = 12, color = p.surface, elevated = true).apply {
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                .apply { setMargins(dp(8), dp(6), dp(8), dp(6)) }
        }
        val header = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        header.addView(AiUiKit.circle(this, "AI", AiUiKit.withAlpha(p.accent, 0.22f), p.accent, 22))
        header.addView(TextView(this).apply {
            text = t("Agent", "에이전트")
            setTextColor(p.textSoft)
            textSize = 11f
            setTypeface(null, Typeface.BOLD)
            setPadding(dp(8), 0, 0, 0)
        })
        card.addView(header)
        card.addView(content)
        val footer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(6), 0, 0)
        }
        footer.addView(AiUiKit.actionChip(this, t("Copy", "복사")) { _ ->
            val cm = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
            cm.setPrimaryClip(android.content.ClipData.newPlainText("message", content.text))
            Toast.makeText(this@AiWorkspaceActivity, t("Copied", "복사됨"), Toast.LENGTH_SHORT).show()
        })
        footer.addView(AiUiKit.actionChip(this, t("Regenerate", "다시 생성")) { _ ->
            engine.sendMessage(t("Continue and improve the previous answer.", "이전 답변을 이어서 더 개선해줘."))
        })
        card.addView(footer)
        chatMessages.addView(card)
        return card to content
    }

    private fun addToolTimeline(name: String, running: Boolean) {
        val p = StudioUi.palette
        chatMessages.addView(TextView(this).apply {
            text = if (running) "⚙ $name …" else "✓ $name"
            setTextColor(p.muted)
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setPadding(dp(16), dp(2), dp(16), dp(2))
        })
        scrollChatToBottom()
    }

    private fun addSystemBubble(text: String) {
        val p = StudioUi.palette
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(4), dp(8), dp(4))
        }
        row.addView(TextView(this).apply {
            this.text = text
            setTextColor(p.warning)
            textSize = StudioUi.TypeSmall
            background = AiUiKit.rounded(AiUiKit.withAlpha(p.warning, 0.12f), dp(999))
            setPadding(dp(12), dp(6), dp(12), dp(6))
        })
        chatMessages.addView(row)
        scrollChatToBottom()
    }

    private fun renderSessionHistory() {
        chatMessages.removeAllViews()
        streamingView = null
        streamingCard = null
        changedFiles.clear()
        restoreChangedFilesFromSession()
        val sessionId = engine.currentSessionId() ?: return
        val messages = engine.sessionMessages(sessionId)
        for (msg in messages) {
            val role = msg.optString("role", "assistant")
            val parts = OcMessage.parts(msg)
            val sb = StringBuilder()
            for (i in 0 until parts.length()) {
                val part = parts.optJSONObject(i) ?: continue
                when (part.optString("type")) {
                    "text" -> sb.append(part.optString("text"))
                    "tool" -> sb.append("\n▸ tool: ").append(part.optString("tool")).append("\n")
                }
            }
            if (sb.isNotBlank()) {
                if (role == "user") addUserBubble(sb.toString().trim()) else addAssistantBubble(sb.toString().trim())
            }
        }
        // Reopening the activity loses the ScrollView state; land the user on
        // the newest message instead of the top of the restored transcript.
        scrollChatToBottom()
    }

    /**
     * The in-memory [changedFiles] set dies with the activity, which made the
     * Changes tab empty after leaving and returning. Rebuild it from the
     * persisted write/edit/patch tool calls of the restored session so the
     * files the agent touched stay openable (diff / open in editor).
     */
    private fun restoreChangedFilesFromSession() {
        val sessionId = engine.currentSessionId() ?: return
        for (msg in engine.sessionMessages(sessionId)) {
            val parts = OcMessage.parts(msg)
            for (i in 0 until parts.length()) {
                val part = parts.optJSONObject(i) ?: continue
                if (part.optString("type") != "tool") continue
                if (part.optString("tool") !in FILE_CHANGING_TOOLS) continue
                val input = part.optJSONObject("state")?.optJSONObject("input") ?: continue
                val path = input.optString("filePath", "").ifEmpty { input.optString("path", "") }
                if (path.isNotEmpty()) changedFiles.add(path)
            }
        }
    }

    private fun addAssistantBubble(text: String) {
        val (_, content) = assistantCard()
        content.text = AiMarkdown.render(this, text)
        scrollChatToBottom()
    }

    private fun scrollChatToBottom() {
        chatScroll.post { chatScroll.fullScroll(View.FOCUS_DOWN) }
    }

    // ------------------------------------------------------------- engine glue

    private fun setupEngine() {
        engine.listener = object : OcEngine.Listener {
            override fun onMessage(text: String, isComplete: Boolean) = runOnUiThread {
                if (text.isBlank()) return@runOnUiThread
                if (streamingView == null) {
                    val (card, tv) = assistantCard()
                    streamingCard = card
                    streamingView = tv
                }
                // Plain text while streaming (cheap); markdown once the turn ends.
                streamingView?.text = if (isComplete) AiMarkdown.render(this@AiWorkspaceActivity, text) else text
                scrollChatToBottom()
                // onMessage hands over the whole reply so far, not a delta, so
                // the character count is just its length. The clock starts on
                // the first streamed chunk, not at the GENERATING state: the
                // local runtime prefills the prompt inside that state and a
                // clock started there would report prefill+decode, not t/s.
                rateChars = text.length.toLong()
                if (rateStartMs == 0L) {
                    rateStartMs = System.currentTimeMillis()
                    lastRatePaintMs = 0L
                }
                if (isComplete) {
                    streamingView = null
                    streamingCard = null
                    updateTokenCount()
                } else if (rateStartMs > 0) {
                    val now = System.currentTimeMillis()
                    if (now - lastRatePaintMs >= ratePaintIntervalMs) {
                        lastRatePaintMs = now
                        updateTokenCount()
                    }
                }
            }

            override fun onThinking(status: String) = runOnUiThread {
                thinkBase = status.removeSuffix("...").ifBlank { "Working" }
                thinkingView.visibility = View.VISIBLE
            }

            override fun onReasoning(text: String) = runOnUiThread {
                thinkingView.visibility = View.VISIBLE
                thinkingView.text = "🧠 " + text.takeLast(240).replace("\n", " ")
            }

            override fun onReasoningDone() = runOnUiThread {
                thinkingView.visibility = View.GONE
            }

            override fun onToolCall(
                name: String, args: JSONObject, risk: OcEngine.RiskLevel,
                onApproval: (Boolean, Boolean) -> Unit,
            ) = runOnUiThread {
                showApprovalDialog(name, args, risk, onApproval)
            }

            override fun onToolStart(name: String, args: JSONObject) = runOnUiThread {
                thinkingView.visibility = View.VISIBLE
                thinkingView.text = "⚙ " + t("Running", "실행 중") + ": $name"
                addToolTimeline(name, running = true)
            }

            override fun onToolEnd(name: String, result: String) = runOnUiThread {
                thinkingView.text = "⚙ " + t("Done", "완료") + ": $name"
                addToolTimeline(name, running = false)
            }

            override fun onPlanUpdate(rawPlan: String) = runOnUiThread {
                if (activeTab == Tab.PLAN) showTab(Tab.PLAN)
            }

            override fun onStateChange(state: OcEngine.AgentState) = runOnUiThread {
                engineState = state
                // The bottom bar's token count is derived from the stored
                // messages, so it has to be recomputed as the turn moves.
                // Without this it kept the value from the initial build - the
                // bar read "0 tok" for the whole session.
                updateStatusBar()
                // Stop the rate clock the moment the engine leaves GENERATING,
                // freezing the finished turn's average on screen instead of
                // blanking it. (It was started by the first streamed chunk.)
                if (state == OcEngine.AgentState.GENERATING && rateStartMs == 0L) {
                    // The local runtime prefills inside GENERATING; until the
                    // first chunk the previous turn's frozen value would be
                    // misleading, so show that a measurement is in progress.
                    if (::tokenCounter.isInitialized) tokenCounter.text = "…"
                } else if (state != OcEngine.AgentState.GENERATING && rateStartMs > 0L) {
                    val elapsedSec = (System.currentTimeMillis() - rateStartMs).coerceAtLeast(1L) / 1000.0
                    lastTokensPerSec = (rateChars / tokensPerChar) / elapsedSec
                    rateStartMs = 0L
                    updateTokenCount()
                }
                sendButton.text = if (state == OcEngine.AgentState.GENERATING || state == OcEngine.AgentState.THINKING)
                    t("Stop", "중지") else t("Send", "전송")
                val p = StudioUi.palette
                val (label, color) = when (state) {
                    OcEngine.AgentState.THINKING -> t("Thinking…", "생각 중…") to p.info
                    OcEngine.AgentState.GENERATING -> t("Writing…", "응답 생성 중…") to p.accent
                    OcEngine.AgentState.EXECUTING_TOOL -> t("Running tool…", "도구 실행 중…") to p.info
                    OcEngine.AgentState.WAITING_TOOL_APPROVAL -> t("Waiting for approval", "승인 대기") to p.warning
                    OcEngine.AgentState.PROCESSING_TOOL_RESULT -> t("Processing result…", "결과 처리 중…") to p.info
                    OcEngine.AgentState.COMPLETED -> t("Completed", "완료") to p.success
                    OcEngine.AgentState.ERROR -> t("Error", "오류") to p.error
                    OcEngine.AgentState.CANCELLED -> t("Cancelled", "취소됨") to p.muted
                    OcEngine.AgentState.IDLE -> t("Ready", "대기") to p.textSoft
                }
                updateStatusPill(label, color)
                if (state == OcEngine.AgentState.IDLE || state == OcEngine.AgentState.COMPLETED) {
                    thinkingView.visibility = View.GONE
                }
            }

            override fun onError(message: String) = runOnUiThread {
                thinkingView.visibility = View.GONE
                addSystemBubble("⚠ $message")
            }

            override fun onTodos(todos: JSONArray) = runOnUiThread {
                if (activeTab == Tab.PLAN) showTab(Tab.PLAN)
            }

            override fun onAskQuestion(questions: JSONArray, onAnswer: (JSONArray?) -> Unit) = runOnUiThread {
                showQuestionDialog(questions, onAnswer)
            }

            override fun onFileChanged(path: String) = runOnUiThread {
                changedFiles.add(path)
                scheduleValidation()
            }

            override fun onNotice(text: String) = runOnUiThread {
                addSystemBubble(text)
            }
        }
    }

    private fun showApprovalDialog(
        name: String, args: JSONObject, risk: OcEngine.RiskLevel,
        onApproval: (Boolean, Boolean) -> Unit,
    ) {
        val detail = args.toString(2).take(1500)
        AlertDialog.Builder(this)
            .setTitle(t("Approval required", "승인 필요") + " · $name")
            .setMessage("${t("Risk", "위험도")}: ${risk.name}\n\n$detail")
            .setPositiveButton(t("Allow once", "1회 허용")) { _, _ -> onApproval(true, false) }
            .setNeutralButton(t("Always", "항상 허용")) { _, _ -> onApproval(true, true) }
            .setNegativeButton(t("Deny", "거부")) { _, _ -> onApproval(false, false) }
            .setOnCancelListener { onApproval(false, false) }
            .show()
    }

    private fun showQuestionDialog(questions: JSONArray, onAnswer: (JSONArray?) -> Unit) {
        var answered = false
        val answerOnce: (JSONArray?) -> Unit = { r -> if (!answered) { answered = true; onAnswer(r) } }
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        val scroll = ScrollView(this).apply { addView(column) }
        val providers = mutableListOf<() -> JSONObject?>()
        for (i in 0 until questions.length()) {
            val q = questions.optJSONObject(i) ?: continue
            val header = q.optString("header", "Question").ifBlank { "Question" }
            val text = q.optString("question", "")
            val options = q.optJSONArray("options")
            column.addView(TextView(this).apply {
                this.text = header
                setTypeface(null, Typeface.BOLD)
                setTextColor(StudioUi.palette.accent)
                textSize = StudioUi.TypeSmall
                setPadding(0, dp(12), 0, dp(4))
            })
            if (text.isNotEmpty()) column.addView(TextView(this).apply {
                this.text = text
                setTextColor(StudioUi.palette.text)
                textSize = StudioUi.TypeBody
            })
            if (options == null || options.length() == 0) {
                val et = EditText(this).apply { hint = t("Your answer", "답변 입력") }
                column.addView(et)
                providers.add { et.text.toString().trim().takeIf { s -> s.isNotEmpty() }?.let { JSONObject().put("answer", it) } }
            } else {
                val selected = mutableListOf<Pair<CheckBox, String>>()
                for (j in 0 until options.length()) {
                    val opt = options.optJSONObject(j) ?: continue
                    val label = opt.optString("label").ifBlank { "Option ${j + 1}" }
                    val cb = CheckBox(this).apply {
                        this.text = label
                        setTextColor(StudioUi.palette.text)
                    }
                    column.addView(cb)
                    selected.add(cb to label)
                }
                providers.add {
                    val chosen = selected.filter { it.first.isChecked }.map { it.second }
                    if (chosen.isEmpty()) null
                    else JSONArray().apply { chosen.forEach { put(it) } }.let { JSONObject().put("selected", it) }
                }
            }
        }
        AlertDialog.Builder(this)
            .setTitle(t("Agent question", "에이전트 질문"))
            .setView(scroll)
            .setPositiveButton(t("Answer", "답변")) { _, _ ->
                val answers = JSONArray()
                for (provider in providers) provider()?.let { answers.put(it) }
                answerOnce(if (answers.length() == 0) null else answers)
            }
            .setNegativeButton(t("Dismiss", "취소")) { _, _ -> answerOnce(null) }
            .setOnCancelListener { answerOnce(null) }
            .show()
    }

    // ------------------------------------------------------------- Plan tab

    private fun buildPlanTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        column.addView(sectionTitle(t("Plan / Tasks", "계획 / 작업")))
        val todos = engine.currentTodos()
        if (todos.length() == 0) {
            column.addView(AiUiKit.emptyState(this, "🗒️", t("No plan yet. Ask the agent to plan a change.", "아직 계획이 없습니다. 에이전트에게 계획을 요청하세요.")))
        } else {
            for (i in 0 until todos.length()) {
                val todo = todos.optJSONObject(i) ?: continue
                val status = todo.optString("status", "pending")
                val mark = when (status) {
                    "completed" -> "✔"; "in_progress" -> "▶"; "cancelled" -> "✖"; else -> "○"
                }
                val priority = todo.optString("priority", "medium")
                val content = todo.optString("content")
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, dp(4), 0, dp(4))
                }
                row.addView(TextView(this).apply {
                    text = "$mark  $content  ($priority)"
                    setTextColor(if (status == "completed") p.muted else p.text)
                    textSize = StudioUi.TypeBody
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(AiUiKit.actionChip(this, t("Retry", "재시도")) { _ ->
                    engine.sendMessage(t("Retry this step: $content", "다음 단계를 다시 시도해줘: $content"))
                    showTab(Tab.CHAT)
                })
                row.setOnLongClickListener { showTodoActions(i, content); true }
                column.addView(row)
            }
            column.addView(AiUiKit.ghostButton(this, t("+ Add step", "+ 단계 추가")) { _ -> promptAddTodo() })
            column.addView(AiUiKit.ghostButton(this, t("Edit plan", "계획 편집")) { _ ->
                engine.sendMessage(
                    t(
                        "Revise the plan: add, remove, or reorder steps as needed, then show the updated todo list.",
                        "계획을 수정해줘: 필요하면 단계를 추가/삭제/순서 변경하고, 갱신된 todo 목록을 보여줘.",
                    ),
                )
                showTab(Tab.CHAT)
            })
        }
        column.addView(Button(this).apply {
            text = t("Ask the agent to plan", "에이전트에게 계획 요청")
            isAllCaps = false
            setOnClickListener {
                engine.sendMessage(
                    t(
                        "Break this task into a short, ordered plan with a todo list, then wait for approval.",
                        "이 작업을 짧고 순서가 있는 계획과 todo 목록으로 나눠서 작성하고 승인을 기다려줘.",
                    ),
                )
                showTab(Tab.CHAT)
            }
        })
        return ScrollView(this).apply { addView(column) }
    }

    private fun planTodos(): MutableList<JSONObject> {
        val arr = engine.currentTodos()
        return (0 until arr.length())
            .mapNotNull { arr.optJSONObject(it)?.let { o -> JSONObject(o.toString()) } }
            .toMutableList()
    }

    private fun saveTodos(list: List<JSONObject>) {
        val arr = JSONArray()
        list.forEach { arr.put(it) }
        engine.updateTodos(arr)
        showTab(Tab.PLAN)
    }

    private fun promptAddTodo() {
        val field = EditText(this).apply { hint = t("Step description", "단계 설명") }
        AlertDialog.Builder(this)
            .setTitle(t("Add step", "단계 추가"))
            .setView(field)
            .setPositiveButton(t("Add", "추가")) { _, _ ->
                val content = field.text.toString().trim()
                if (content.isNotEmpty()) {
                    val list = planTodos()
                    list.add(JSONObject().apply {
                        put("content", content)
                        put("status", "pending")
                        put("priority", "medium")
                    })
                    saveTodos(list)
                }
            }
            .setNegativeButton(t("Cancel", "취소"), null)
            .show()
    }

    private fun showTodoActions(index: Int, content: String) {
        val actions = arrayOf(
            t("Retry", "재시도"),
            t("Move up", "위로 이동"),
            t("Move down", "아래로 이동"),
            t("Toggle done", "완료 토글"),
            t("Delete", "삭제"),
        )
        AlertDialog.Builder(this)
            .setTitle(content.take(60))
            .setItems(actions) { _, which ->
                when (which) {
                    0 -> {
                        engine.sendMessage(t("Retry this step: $content", "다음 단계를 다시 시도해줘: $content"))
                        showTab(Tab.CHAT)
                    }
                    1 -> {
                        val list = planTodos()
                        if (index > 0) {
                            val item = list.removeAt(index)
                            list.add(index - 1, item)
                            saveTodos(list)
                        }
                    }
                    2 -> {
                        val list = planTodos()
                        if (index < list.size - 1) {
                            val item = list.removeAt(index)
                            list.add(index + 1, item)
                            saveTodos(list)
                        }
                    }
                    3 -> {
                        val list = planTodos()
                        list.getOrNull(index)?.let { o ->
                            o.put("status", if (o.optString("status") == "completed") "pending" else "completed")
                        }
                        saveTodos(list)
                    }
                    4 -> {
                        val list = planTodos()
                        if (index in list.indices) {
                            list.removeAt(index)
                            saveTodos(list)
                        }
                    }
                }
            }
            .show()
    }

    // ------------------------------------------------------------ Changes tab

    private fun buildChangesTab(): View {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        column.addView(sectionTitle(t("Changed files", "변경된 파일")))
        if (changedFiles.isEmpty()) {
            column.addView(AiUiKit.emptyState(this, "📝", t("No files changed in this session yet.", "이 세션에서 아직 변경된 파일이 없습니다.")))
        } else {
            for (path in changedFiles) {
                // Tap the file name to preview the diff; the chip hands the
                // file to the editor (finishing this activity).
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, dp(8), 0, dp(8))
                }
                row.addView(TextView(this).apply {
                    text = relativePath(path)
                    setTextColor(StudioUi.palette.info)
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    setOnClickListener { showDiff(path) }
                })
                row.addView(AiUiKit.actionChip(this, t("Open in editor", "편집기에서 열기")) { _ ->
                    openInEditor(path)
                })
                column.addView(row)
            }
            column.addView(Button(this).apply {
                text = t("Revert all changes", "모든 변경 되돌리기")
                isAllCaps = false
                setOnClickListener {
                    AlertDialog.Builder(this@AiWorkspaceActivity)
                        .setMessage(t("Revert all files changed in this session?", "이 세션에서 변경한 모든 파일을 되돌릴까요?"))
                        .setPositiveButton(t("Revert", "되돌리기")) { _, _ ->
                            val restored = engine.revert()
                            addSystemBubble(t("Reverted ${restored.size} file(s).", "${restored.size}개 파일을 되돌렸습니다."))
                            changedFiles.clear()
                            showTab(Tab.CHANGES)
                        }
                        .setNegativeButton(t("Cancel", "취소"), null)
                        .show()
                }
            })
        }
        return ScrollView(this).apply { addView(column) }
    }

    private fun refreshChangedFiles() {
        // no-op placeholder retained for clarity; the Changes tab is rebuilt on show.
    }

    private fun showDiff(path: String) {
        val file = File(path)
        val after = runCatching { file.readText() }.getOrDefault("")
        val before = engine.latestBackupForCurrentSession(path)?.let { runCatching { it.readText() }.getOrNull() }
        if (before == null) {
            AlertDialog.Builder(this)
                .setTitle(relativePath(path))
                .setMessage(t("No pre-edit snapshot available for this file.", "이 파일의 변경 전 스냅샷이 없습니다."))
                .setPositiveButton("OK", null)
                .show()
            return
        }
        DiffViewerActivity.start(this, relativePath(path) + " (before)", before, relativePath(path) + " (after)", after)
    }

    /**
     * Public entry point for editing an agent-changed file: hand it back to
     * the editor and leave the workspace. MainActivity already consumes the
     * [EXTRA_OPEN_IN_EDITOR] extra ("open_file") in its onCreate, so this
     * needs no MainActivity change. CLEAR_TOP folds the stack back to a single
     * editor instance; because MainActivity has the default (standard) launch
     * mode, it is recreated with the new intent and its onCreate opens the
     * file (its onNewIntent deliberately ignores this extra).
     */
    fun openInEditor(path: String) {
        val intent = Intent(this, com.aide.mobilestudio.MainActivity::class.java).apply {
            putExtra(EXTRA_OPEN_IN_EDITOR, path)
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
        }
        startActivity(intent)
        finish()
    }

    // ------------------------------------------------------------ Search tab

    private lateinit var searchInput: EditText
    private lateinit var searchResults: LinearLayout

    private data class SearchHit(val file: String, val line: Int, val text: String)

    private fun buildSearchTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(10), dp(10), dp(6))
        }
        searchInput = EditText(this).apply {
            hint = t("Search file names and contents…", "파일명·내용 검색…")
            setHintTextColor(p.muted)
            setTextColor(p.text)
            textSize = StudioUi.TypeBody
            background = AiUiKit.rounded(p.surface2, dp(12), strokeColor = p.border, strokePx = dp(1))
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEARCH) { runSearch(); true } else false
            }
        }
        row.addView(searchInput)
        row.addView(Button(this).apply {
            text = t("Search", "검색")
            isAllCaps = false
            setTextColor(p.onAccent)
            background = AiUiKit.rippleContent(this@AiWorkspaceActivity, p.accent, dp(12))
            setOnClickListener { runSearch() }
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(8) }
        })
        column.addView(row)

        searchResults = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(10), dp(6), dp(10), dp(10))
        }
        column.addView(
            ScrollView(this).apply { addView(searchResults) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f),
        )
        return column
    }

    private fun runSearch() {
        val query = searchInput.text?.toString()?.trim().orEmpty()
        val root = projectDir ?: return
        if (query.isEmpty()) return
        val p = StudioUi.palette
        searchResults.removeAllViews()
        searchResults.addView(body(t("Searching…", "검색 중…")))
        scope.launch {
            val results = withContext(Dispatchers.IO) { performSearch(root, query) }
            searchResults.removeAllViews()
            if (results.isEmpty()) {
                searchResults.addView(AiUiKit.emptyState(this@AiWorkspaceActivity, "🔎", t("No matches.", "결과 없음.")))
                return@launch
            }
            for (r in results) {
                searchResults.addView(TextView(this@AiWorkspaceActivity).apply {
                    text = "${r.file}:${r.line}  ${r.text.take(160)}"
                    setTextColor(p.info)
                    textSize = StudioUi.TypeSmall
                    typeface = Typeface.MONOSPACE
                    setPadding(0, dp(6), 0, dp(6))
                    setOnClickListener {
                        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        cm.setPrimaryClip(android.content.ClipData.newPlainText("result", r.file + ":" + r.line))
                        Toast.makeText(this@AiWorkspaceActivity, t("Copied path", "경로 복사됨"), Toast.LENGTH_SHORT).show()
                    }
                    setOnLongClickListener {
                        engine.sendMessage(t("Open and explain ${r.file}:${r.line}:\n${r.text}", "${r.file}:${r.line} 를 열어서 설명해줘:\n${r.text}"))
                        showTab(Tab.CHAT)
                        true
                    }
                })
            }
        }
    }

    private fun performSearch(root: File, query: String): List<SearchHit> {
        val regex = runCatching { Regex(query, RegexOption.IGNORE_CASE) }.getOrNull()
        val nameQuery = query.lowercase()
        val hits = mutableListOf<SearchHit>()
        val excluded = setOf(".git", "build", ".gradle", "node_modules", ".idea", ".cxx", ".kotlin", "out")
        root.walkTopDown().onEnter { it.name !in excluded }.forEach { f ->
            if (hits.size >= 200 || !f.isFile || f.length() > 2L * 1024 * 1024) return@forEach
            val rel = runCatching { f.relativeTo(root).path }.getOrDefault(f.name)
            if (rel.lowercase().contains(nameQuery)) hits.add(SearchHit(rel, 0, ""))
            if (regex == null) return@forEach
            val text = runCatching { f.readText() }.getOrNull() ?: return@forEach
            var n = 1
            for (ln in text.lineSequence()) {
                if (regex.containsMatchIn(ln)) {
                    hits.add(SearchHit(rel, n, ln.trim()))
                    if (hits.size >= 200) break
                }
                n++
            }
        }
        return hits
    }

    // -------------------------------------------------------- Diagnostics tab

    private fun buildDiagnosticsTab(): View {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        column.addView(sectionTitle(t("Diagnostics", "진단")))
        if (projectDir == null) {
            column.addView(AiUiKit.emptyState(this, "🧭", t("No project is open, so live validation is unavailable.", "열린 프로젝트가 없어 실시간 검증을 사용할 수 없습니다.")))
            return ScrollView(this).apply { addView(column) }
        }
        val summary = validation.lastSummary()
        val summaryText = when {
            summary == null -> t("Not validated yet.", "아직 검증하지 않았습니다.")
            else -> summary.report()
        }
        column.addView(body(summaryText))
        column.addView(Button(this).apply {
            text = t("Re-run validation", "다시 검사")
            isAllCaps = false
            setOnClickListener { scheduleValidation(immediate = true) }
        })

        for (d in validation.diagnostics()) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(0, dp(8), 0, dp(8))
                setOnClickListener { showDiagnostic(d) }
            }
            row.addView(TextView(this).apply {
                text = (if (d.severity.name == "ERROR") "✖ " else "⚠ ") + "${d.file}:${d.line}:${d.column}"
                setTextColor(if (d.severity.name == "ERROR") StudioUi.palette.error else StudioUi.palette.warning)
                textSize = StudioUi.TypeSmall
            })
            row.addView(TextView(this).apply {
                text = d.message
                setTextColor(StudioUi.palette.text)
                textSize = 13f
            })
            column.addView(row)
        }
        val resolved = validation.resolved()
        if (resolved.isNotEmpty()) {
            column.addView(sectionTitle(t("Resolved", "해결됨")))
            for (d in resolved) {
                column.addView(TextView(this).apply {
                    text = "✔ ${d.file}:${d.line} ${d.message}"
                    setTextColor(StudioUi.palette.success)
                    textSize = StudioUi.TypeSmall
                })
            }
        }
        return ScrollView(this).apply { addView(column) }
    }

    private fun showDiagnostic(d: com.aide.mobilestudio.ai.validation.LiveDiagnostic) {
        AlertDialog.Builder(this)
            .setTitle(t("Diagnostic", "진단"))
            .setMessage(
                "${d.file}:${d.line}:${d.column}\n${d.severity}\n\n${d.message}" +
                    (d.snippet?.let { "\n\n$it" } ?: ""),
            )
            .setPositiveButton(t("Ask AI to fix", "AI에게 수정 요청")) { _, _ ->
                engine.sendMessage(
                    t(
                        "Fix this problem:\n${d.file}:${d.line}:${d.column}: ${d.message}",
                        "이 문제를 수정해줘:\n${d.file}:${d.line}:${d.column}: ${d.message}",
                    ),
                )
                showTab(Tab.CHAT)
            }
            .setNeutralButton(t("Copy", "복사")) { _, _ ->
                val cm = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                cm.setPrimaryClip(android.content.ClipData.newPlainText("diagnostic", "${d.file}:${d.line}:${d.column}: ${d.message}"))
            }
            .setNegativeButton(t("Close", "닫기"), null)
            .show()
    }

    private fun scheduleValidation(immediate: Boolean = false) {
        if (projectDir == null) return
        handler.removeCallbacks(validateRunnable)
        handler.postDelayed(validateRunnable, if (immediate) 0 else 1200)
    }

    private val validateRunnable = Runnable {
        val dir = projectDir ?: return@Runnable
        scope.launch {
            withContext(Dispatchers.IO) {
                validation.recordChanges(changedFiles.map { File(it) })
                validation.validate(dir)
            }
            if (activeTab == Tab.DIAGNOSTICS) showTab(Tab.DIAGNOSTICS)
        }
    }

    // --------------------------------------------------- Files rail destination

    private fun buildFilesTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        if (projectDir == null) {
            column.addView(AiUiKit.emptyState(this, "🗂", t("No project is open.", "열린 프로젝트가 없습니다.")))
            return column
        }
        if (browseDir == null || !browseDir!!.isDirectory) browseDir = projectDir
        val crumbs = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(10), dp(12), dp(6))
        }
        if (browseDir != projectDir) {
            crumbs.addView(
                AiUiKit.actionChip(this, "‹ ${t("Up", "위로")}") { _ ->
                    browseDir = browseDir?.parentFile ?: projectDir
                    renderFiles()
                },
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { marginEnd = dp(8) },
            )
        }
        crumbs.addView(TextView(this).apply {
            text = relativePath(browseDir!!.absolutePath).ifEmpty { "." }
            setTextColor(p.textSoft)
            textSize = 12f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        column.addView(crumbs)
        filesList = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(8), dp(0), dp(8), dp(12))
        }
        column.addView(
            ScrollView(this).apply { addView(filesList) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f),
        )
        renderFiles()
        return column
    }

    private fun renderFiles() {
        if (!::filesList.isInitialized) return
        filesList.removeAllViews()
        val dir = browseDir ?: return
        val entries = dir.listFiles()
            ?.sortedWith(compareByDescending<File> { it.isDirectory }.thenBy { it.name.lowercase() })
            ?.take(300)
            .orEmpty()
        if (entries.isEmpty()) {
            filesList.addView(AiUiKit.emptyState(this, "📭", t("Empty folder.", "빈 폴더입니다.")))
            return
        }
        for (entry in entries) {
            val isDir = entry.isDirectory
            val icon = when {
                isDir -> "📁"
                entry.extension.lowercase() in setOf("kt", "kts") -> "🄺"
                entry.extension.lowercase() == "java" -> "☕"
                entry.extension.lowercase() in setOf("xml") -> "📄"
                entry.extension.lowercase() in setOf("png", "jpg", "jpeg", "webp", "gif") -> "🖼"
                else -> "📄"
            }
            filesList.addView(
                LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(8), dp(9), dp(8), dp(9))
                    background = AiUiKit.rippleContent(this@AiWorkspaceActivity, StudioUi.palette.surface, dp(8))
                    setOnClickListener {
                        if (isDir) {
                            browseDir = entry
                            renderFiles()
                        } else {
                            showFileActions(entry)
                        }
                    }
                    addView(TextView(this@AiWorkspaceActivity).apply {
                        text = icon
                        textSize = 13f
                    })
                    addView(TextView(this@AiWorkspaceActivity).apply {
                        text = entry.name + if (isDir) "/" else ""
                        setTextColor(if (isDir) StudioUi.palette.text else StudioUi.palette.textSoft)
                        textSize = 13f
                        setPadding(dp(8), 0, 0, 0)
                        layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    })
                    addView(TextView(this@AiWorkspaceActivity).apply {
                        text = if (isDir) "" else formatBytes(entry.length())
                        setTextColor(StudioUi.palette.muted)
                        textSize = 10f
                    })
                },
            )
        }
    }

    private fun formatBytes(bytes: Long): String = when {
        bytes >= 1L shl 20 -> "%.1f MB".format(Locale.US, bytes.toDouble() / (1L shl 20))
        bytes >= 1L shl 10 -> "%.0f KB".format(Locale.US, bytes.toDouble() / (1L shl 10))
        else -> "$bytes B"
    }

    private fun showFileActions(file: File) {
        val actions = arrayOf(
            t("Preview", "미리보기"),
            t("Ask AI to explain", "AI에게 설명 요청"),
            t("Ask AI to change", "AI에게 수정 요청"),
            t("Open in editor", "편집기에서 열기"),
        )
        AlertDialog.Builder(this)
            .setTitle(file.name)
            .setItems(actions) { _, which ->
                when (which) {
                    0 -> previewFile(file)
                    1 -> {
                        showTab(Tab.CHAT)
                        submit(
                            t(
                                "Read ${relativePath(file.absolutePath)} and explain what it does.",
                                "${relativePath(file.absolutePath)} 파일을 읽고 어떤 역할을 하는지 설명해줘.",
                            ),
                        )
                    }
                    2 -> {
                        showTab(Tab.CHAT)
                        // Prefill instead of sending: the user still needs to
                        // describe what to change.
                        input.setText(
                            t(
                                "Change ${relativePath(file.absolutePath)} as follows: ",
                                "${relativePath(file.absolutePath)} 파일을 다음과 같이 수정해줘: ",
                            ),
                        )
                        input.requestFocus()
                    }
                    3 -> openInEditor(file.absolutePath)
                }
            }
            .show()
    }

    private fun previewFile(file: File) {
        val content = runCatching { file.readText() }.getOrNull()
        if (content == null) {
            Toast.makeText(this, t("Cannot read file (binary?)", "파일을 읽을 수 없습니다(바이너리?)"), Toast.LENGTH_SHORT).show()
            return
        }
        val lines = content.lineSequence().take(200).toList()
        val body = lines.joinToString("\n") + if (content.lines().size > 200) "\n… (+more)" else ""
        AlertDialog.Builder(this)
            .setTitle(file.name)
            .setMessage(body)
            .setPositiveButton("OK", null)
            .show()
    }

    // ----------------------------------------------------- Git rail destination

    private fun buildGitTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
        }
        if (projectDir == null) {
            column.addView(AiUiKit.emptyState(this, "⑂", t("No project is open.", "열린 프로젝트가 없습니다.")))
            return column
        }
        val actions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(10), dp(10), dp(10), dp(4))
        }
        fun gitChip(label: String, args: String) {
            actions.addView(
                AiUiKit.actionChip(this, label) { _ -> runGit(args) },
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { marginEnd = dp(6) },
            )
        }
        gitChip(t("Status", "상태"), "status --short --branch")
        gitChip(t("Log", "로그"), "log --oneline -15")
        gitChip(t("Diff", "변경"), "diff --stat")
        gitChip(t("Branch", "브랜치"), "branch --show-current")
        column.addView(actions)
        gitOutput = TextView(this).apply {
            setTextColor(p.text)
            textSize = 12f
            typeface = Typeface.MONOSPACE
            setPadding(dp(12), dp(8), dp(12), dp(12))
            setTextIsSelectable(true)
            text = t("Run a git command to see output.", "git 명령을 실행하면 결과가 표시됩니다.")
        }
        column.addView(
            ScrollView(this).apply { addView(gitOutput) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f),
        )
        return column
    }

    private fun runGit(args: String) {
        val dir = projectDir ?: return
        if (!::gitOutput.isInitialized) return
        gitOutput.text = "git $args\n" + t("Running…", "실행 중…")
        scope.launch {
            val result = withContext(Dispatchers.IO) {
                val env = runCatching { ShellEnvironment.buildToolEnvironment(this@AiWorkspaceActivity, dir) }.getOrDefault(emptyMap())
                val lines = mutableListOf<String>()
                val run = runCatching {
                    ShellRunner.runSh(this@AiWorkspaceActivity, "git $args", dir, env, 60_000) { line -> lines.add(line) }
                }.getOrNull()
                (lines.joinToString("\n")) to (run?.exitCode ?: -1)
            }
            val (output, exit) = result
            gitOutput.text = "git $args\n" +
                (output.ifBlank { t("(no output)", "(출력 없음)") }) +
                "\n\n[${t("exit", "종료")} $exit]"
        }
    }

    // ----------------------------------------------------- build (실행 / 정리)

    private fun runGradleBuild() {
        val dir = projectDir
        if (dir == null) {
            Toast.makeText(this, t("No project is open.", "열린 프로젝트가 없습니다."), Toast.LENGTH_SHORT).show()
            return
        }
        val gradlew = File(dir, "gradlew").isFile
        runBuildCommand(if (gradlew) "./gradlew assembleDebug" else "gradle assembleDebug")
    }

    private fun runGradleClean() {
        val dir = projectDir
        if (dir == null) {
            Toast.makeText(this, t("No project is open.", "열린 프로젝트가 없습니다."), Toast.LENGTH_SHORT).show()
            return
        }
        val gradlew = File(dir, "gradlew").isFile
        runBuildCommand(if (gradlew) "./gradlew clean" else "gradle clean")
    }

    private fun runBuildCommand(command: String) {
        showTab(Tab.TERMINAL)
        if (!::terminalInput.isInitialized) return
        terminalInput.setText(command)
        runTerminalCommand()
    }

    // ------------------------------------------------------------- Tools tab

    private fun buildToolsTab(): View {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        column.addView(sectionTitle(t("Available tools", "사용 가능한 도구")))
        val names = runCatching { engine.toolNames() }.getOrDefault(emptyList()).sorted()
        if (names.isEmpty()) {
            column.addView(body(t("No tools registered.", "등록된 도구가 없습니다.")))
        } else {
            for (name in names) {
                val readOnly = name in READ_ONLY_HINT
                column.addView(TextView(this).apply {
                    text = (if (readOnly) "👁 " else "⚙ ") + name + if (readOnly) t("  (read-only)", "  (읽기 전용)") else ""
                    setTextColor(if (readOnly) StudioUi.palette.info else StudioUi.palette.text)
                    textSize = 13f
                    setPadding(0, dp(5), 0, dp(5))
                })
            }
        }
        column.addView(body(
            t(
                "Destructive actions require approval. Read-only tools may run in parallel when already permitted.",
                "파괴적 작업은 승인이 필요합니다. 이미 허용된 읽기 전용 도구는 병렬 실행될 수 있습니다."
            ),
        ))
        return ScrollView(this).apply { addView(column) }
    }

    // --------------------------------------------------------- Resources tab

    private fun buildResourcesTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        column.addView(sectionTitle(t("Resource monitor", "자원 모니터")))
        val mem = AndroidMemoryProbe.snapshot(this)
        val rm = ResourceManager()
        val decision = rm.decide(mem, pendingTasks = 4, loadedLocalModels = 1, buildRunning = false, userMax = 4)

        fun row(label: String, value: String, color: Int = p.text) {
            column.addView(TextView(this).apply {
                text = "$label: $value"
                setTextColor(color)
                textSize = 13f
                setPadding(0, dp(5), 0, dp(5))
            })
        }
        row(t("Current model", "현재 모델"), engine.currentModelLabel(), p.accent)
        row(t("Free memory", "여유 메모리"), "${mem.availableMb} MB")
        row(t("Heap used", "힙 사용"), "${(mem.heapUsedFraction * 100).toInt()}%")
        row(t("Low memory", "저메모리"), mem.lowMemory.toString(), if (mem.lowMemory) p.error else p.success)
        row(t("Parallel tasks (estimate)", "병렬 작업(추정)"), "${decision.count} — ${decision.reason}")
        val canLoad = rm.canLoadModel(mem, alreadyLoaded = 0)
        row(
            t("Can load a model", "모델 로드 가능"),
            (if (canLoad.first) "yes" else "no") + " (" + canLoad.second + ")",
            if (canLoad.first) p.success else p.warning,
        )
        row(t("Build with inference", "빌드+추론 동시"), rm.canRunBuildWithInference(mem, inferenceActive = true).toString())
        column.addView(body(
            t(
                "Memory values are measured from Android; model memory and parallelism are estimates.",
                "메모리 값은 Android에서 측정되며, 모델 메모리와 병렬성은 추정치입니다."
            ),
        ))
        return ScrollView(this).apply { addView(column) }
    }

    // ----------------------------------------------------------- Sessions tab

    private fun buildSessionsTab(): View {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        column.addView(sectionTitle(t("Sessions", "세션")))
        column.addView(AiUiKit.primaryButton(this, t("+ New session", "+ 새 세션")) {
            newChat()
            if (projectDir != null) scheduleValidation(immediate = true)
        })
        val search = EditText(this).apply {
            hint = t("Search sessions…", "세션 검색…")
            setHintTextColor(StudioUi.palette.muted)
            setTextColor(StudioUi.palette.text)
            textSize = 13f
            maxLines = 1
            setText(sessionQuery)
            background = AiUiKit.rounded(StudioUi.palette.surface2, dp(10), strokeColor = StudioUi.palette.border, strokePx = dp(1))
            setPadding(dp(12), dp(8), dp(12), dp(8))
            addTextChangedListener(object : android.text.TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
                override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                    sessionQuery = s?.toString().orEmpty()
                    populateSessions()
                }
                override fun afterTextChanged(s: android.text.Editable?) {}
            })
        }
        column.addView(search, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { setMargins(0, dp(8), 0, dp(6)) })
        sessionListContainer = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        column.addView(
            ScrollView(this).apply { addView(sessionListContainer) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f),
        )
        populateSessions()
        return column
    }

    private lateinit var sessionListContainer: LinearLayout
    private var sessionQuery: String = ""

    private fun populateSessions() {
        sessionListContainer.removeAllViews()
        val all = engine.listSessions(includeArchived = true)
        val sessions = if (sessionQuery.isBlank()) all
        else all.filter { it.title.contains(sessionQuery, ignoreCase = true) }
        if (sessions.isEmpty()) {
            sessionListContainer.addView(AiUiKit.emptyState(this, "💬", t("No sessions.", "세션이 없습니다.")))
            return
        }
        val fmt = SimpleDateFormat("MM-dd HH:mm", Locale.getDefault())
        for (s in sessions) {
            val current = s.id == engine.currentSessionId()
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(0, dp(8), 0, dp(8))
                setOnClickListener { openSession(s) }
            }
            row.addView(TextView(this).apply {
                text = (if (current) "● " else "") + s.title
                setTextColor(if (current) StudioUi.palette.accent else StudioUi.palette.text)
                textSize = StudioUi.TypeBody
            })
            row.addView(TextView(this).apply {
                text = "${s.agent} · ${fmt.format(Date(s.timeUpdated))}" + if (s.archived) " · archived" else ""
                setTextColor(StudioUi.palette.textSoft)
                textSize = 11f
            })
            row.setOnLongClickListener { showSessionActions(s); true }
            sessionListContainer.addView(row)
        }
    }

    private fun openSession(s: OcSessionStore.Session) {
        engine.loadSession(s.id)
        changedFiles.clear()
        validation.reset()
        if (projectDir != null) scheduleValidation(immediate = true)
        renderSessionHistory()
        showTab(Tab.CHAT)
    }

    private fun showSessionActions(s: OcSessionStore.Session) {
        val actions = arrayOf(
            t("Rename", "이름 변경"),
            t("Fork", "분기"),
            if (s.archived) t("Unarchive", "보관 해제") else t("Archive", "보관"),
            t("Delete", "삭제"),
        )
        AlertDialog.Builder(this)
            .setTitle(s.title)
            .setItems(actions) { _, which ->
                when (which) {
                    0 -> promptRename(s)
                    1 -> { engine.forkSession(s.id)?.let { engine.loadSession(it) }; renderSessionHistory(); showTab(Tab.CHAT) }
                    2 -> { engine.archiveSession(s.id, !s.archived); showTab(Tab.SESSIONS) }
                    3 -> AlertDialog.Builder(this)
                        .setMessage(t("Delete this session?", "이 세션을 삭제할까요?"))
                        .setPositiveButton(t("Delete", "삭제")) { _, _ -> engine.deleteSession(s.id); showTab(Tab.SESSIONS) }
                        .setNegativeButton(t("Cancel", "취소"), null)
                        .show()
                }
            }
            .show()
    }

    private fun promptRename(s: OcSessionStore.Session) {
        val et = EditText(this).apply { setText(s.title) }
        AlertDialog.Builder(this)
            .setTitle(t("Rename session", "세션 이름 변경"))
            .setView(et)
            .setPositiveButton("OK") { _, _ ->
                engine.renameSession(s.id, et.text.toString().ifBlank { s.title })
                showTab(Tab.SESSIONS)
            }
            .setNegativeButton(t("Cancel", "취소"), null)
            .show()
    }

    // ------------------------------------------------------------- Models tab

    private fun buildModelsTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        column.addView(sectionTitle(t("Providers & models", "공급자 & 모델")))
        column.addView(TextView(this).apply {
            text = t("Current", "현재") + ": " + engine.currentModelLabel()
            setTextColor(p.accent)
            textSize = 13f
            setPadding(0, 0, 0, dp(8))
        })
        for ((id, label) in engine.listProviders()) {
            val headerRow = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, dp(12), 0, dp(4))
            }
            headerRow.addView(TextView(this).apply {
                text = label
                setTextColor(p.textSoft)
                textSize = 13f
                setTypeface(null, Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            })
            endpointFor(id)?.let { ep ->
                val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
                actions.addView(AiUiKit.actionChip(this, t("Test", "연결 테스트")) { _ -> testConnection(ep) },
                    LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(6) })
                actions.addView(AiUiKit.actionChip(this, t("Models", "모델 목록")) { _ -> fetchModels(ep, id) })
                headerRow.addView(actions)
            }
            column.addView(headerRow)
            val models: List<OcModelInfo> = runCatching { engine.listProviderModels(id) }.getOrDefault(emptyList())
            if (models.isEmpty()) {
                column.addView(TextView(this).apply {
                    text = t("No models / API key missing", "모델 없음 / API 키 필요")
                    setTextColor(p.muted); textSize = StudioUi.TypeSmall
                })
            } else {
                for (m in models) {
                    column.addView(TextView(this).apply {
                        text = "  ${m.name}  (${m.ref})  ctx=${m.contextLimit}"
                        setTextColor(p.info)
                        textSize = 13f
                        setPadding(0, dp(4), 0, dp(4))
                        setOnClickListener {
                            engine.setModel(m.ref.toString())
                            updateModelLabel()
                            Toast.makeText(this@AiWorkspaceActivity, m.ref.toString(), Toast.LENGTH_SHORT).show()
                        }
                    })
                }
            }
        }
        column.addView(body(
            t(
                "Local models require an existing GGUF file. API providers require a key set in Settings.",
                "로컬 모델은 GGUF 파일이 있어야 하고, API 공급자는 설정에서 키를 입력해야 합니다."
            ),
        ))
        return ScrollView(this).apply { addView(column) }
    }

    private fun updateModelLabel() {
        modelText.text = engine.currentModelLabel()
    }

    private fun fetchModels(endpoint: ProviderEndpoint, providerId: String) {
        Toast.makeText(this, t("Fetching models…", "모델 목록 가져오는 중…"), Toast.LENGTH_SHORT).show()
        scope.launch {
            val models = withContext(Dispatchers.IO) {
                runCatching {
                    val response = AndroidHttpTransport().send(ConnectionTester.buildRequest(endpoint))
                    if (response.status in 200..299) ProviderModelList.parse(providerId, response.body) else emptyList()
                }.getOrDefault(emptyList())
            }
            val message = if (models.isEmpty()) {
                t("No models returned (check the key and endpoint).", "모델을 가져오지 못했습니다 (키/엔드포인트 확인).")
            } else {
                models.joinToString("\n")
            }
            AlertDialog.Builder(this@AiWorkspaceActivity)
                .setTitle(t("Available models", "사용 가능한 모델"))
                .setMessage(message)
                .setPositiveButton("OK", null)
                .show()
        }
    }

    private fun endpointFor(providerId: String): ProviderEndpoint? = when (providerId) {
        "gemini" -> settings.apiKey?.let {
            ProviderEndpoint("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta", it, settings.geminiModel, AuthStyle.QUERY_KEY)
        }
        "openai" -> settings.openaiApiKey?.let {
            ProviderEndpoint("openai", "OpenAI", settings.openaiBaseUrl, it, settings.openaiModel, AuthStyle.BEARER)
        }
        "anthropic" -> settings.anthropicApiKey?.let {
            ProviderEndpoint("anthropic", "Anthropic", "https://api.anthropic.com/v1", it, settings.anthropicModel, AuthStyle.X_API_KEY)
        }
        else -> null
    }

    private fun testConnection(endpoint: ProviderEndpoint) {
        Toast.makeText(this, t("Testing ${endpoint.displayName}…", "${endpoint.displayName} 테스트 중…"), Toast.LENGTH_SHORT).show()
        scope.launch {
            val result = withContext(Dispatchers.IO) { ConnectionTester.test(endpoint, AndroidHttpTransport()) }
            AlertDialog.Builder(this@AiWorkspaceActivity)
                .setTitle(t("Connection test", "연결 테스트"))
                .setMessage("${endpoint.displayName} (${ConnectionTester.mask(endpoint.apiKey)})\n\n${result.message}")
                .setPositiveButton("OK", null)
                .show()
        }
    }

    // ----------------------------------------------------------- Settings tab

    private fun buildSettingsTab(): View {
        val p = StudioUi.palette
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
        }
        column.addView(sectionTitle(t("AI settings", "AI 설정")))

        val geminiKey = field("Gemini API key", settings.apiKey.orEmpty(), secret = true)
        val geminiModel = field("Gemini model", settings.geminiModel)
        val openaiKey = field("OpenAI API key", settings.openaiApiKey.orEmpty(), secret = true)
        val openaiBase = field("OpenAI base URL", settings.openaiBaseUrl)
        val openaiModel = field("OpenAI model", settings.openaiModel)
        val anthropicKey = field("Anthropic API key", settings.anthropicApiKey.orEmpty(), secret = true)
        val anthropicModel = field("Anthropic model", settings.anthropicModel)
        val threads = field("CPU threads", settings.nThreads.toString())
        val context = field("Context length", settings.contextLength.toString())
        // GPU offload: layers llama.cpp places on the Vulkan GPU. 99 (or any
        // negative value) means every layer the model has - llama.h clamps it to
        // the real count - and 0 keeps inference on the CPU.
        val gpuLayers = field("GPU layers (99 = all, 0 = CPU)", settings.nGpuLayers.toString())

        for (v in listOf(geminiKey, geminiModel, openaiKey, openaiBase, openaiModel, anthropicKey, anthropicModel, threads, context, gpuLayers)) {
            column.addView(label(v.first))
            column.addView(v.second)
        }
        val autoApply = CheckBox(this).apply { text = t("Auto-apply edits", "자동 편집 적용"); isChecked = settings.autoApplyChanges }
        val webSearch = CheckBox(this).apply { text = t("Web search tool", "웹 검색 도구"); isChecked = settings.webSearchEnabled }
        column.addView(autoApply)
        column.addView(webSearch)

        column.addView(Button(this).apply {
            text = t("Save", "저장")
            isAllCaps = false
            setOnClickListener {
                settings.apiKey = geminiKey.second.text.toString().trim().ifBlank { null }
                settings.geminiModel = geminiModel.second.text.toString().trim().ifBlank { settings.geminiModel }
                settings.openaiApiKey = openaiKey.second.text.toString().trim().ifBlank { null }
                settings.openaiBaseUrl = openaiBase.second.text.toString().trim().ifBlank { settings.openaiBaseUrl }
                settings.openaiModel = openaiModel.second.text.toString().trim().ifBlank { settings.openaiModel }
                settings.anthropicApiKey = anthropicKey.second.text.toString().trim().ifBlank { null }
                settings.anthropicModel = anthropicModel.second.text.toString().trim().ifBlank { settings.anthropicModel }
                settings.nThreads = threads.second.text.toString().toIntOrNull() ?: settings.nThreads
                settings.contextLength = context.second.text.toString().toIntOrNull() ?: settings.contextLength
                settings.nGpuLayers = gpuLayers.second.text.toString().toIntOrNull() ?: settings.nGpuLayers
                settings.autoApplyChanges = autoApply.isChecked
                settings.webSearchEnabled = webSearch.isChecked
                engine.reloadConfig()
                updateModelLabel()
                Toast.makeText(this@AiWorkspaceActivity, t("Saved", "저장됨"), Toast.LENGTH_SHORT).show()
            }
        })
        column.addView(body(
            t(
                "API keys are stored in the app's encrypted preferences and are never written to logs.",
                "API 키는 앱의 암호화된 저장소에 보관되며 로그에 남기지 않습니다."
            ),
        ))
        column.addView(body(
            t(
                "GPU layers: 99 (or -1) puts every layer the model has on the Vulkan GPU, 0 keeps it on the CPU. " +
                    "A device that cannot hold the model lowers the offload on its own; the layers actually " +
                    "offloaded are reported by llama.cpp in the log.",
                "GPU 레이어: 99(또는 -1)는 모델의 모든 레이어를 Vulkan GPU에 올리고, 0은 CPU만 사용합니다. " +
                    "메모리가 부족한 기기에서는 오프로드 수준을 스스로 낮춥니다. 실제로 올라간 레이어 수는 " +
                    "llama.cpp가 로그에 남깁니다.",
            ),
        ))

        val mem = com.aide.mobilestudio.ai.resource.AndroidMemoryProbe.snapshot(this)
        val decision = com.aide.mobilestudio.ai.resource.ResourceManager().decide(
            snapshot = mem,
            pendingTasks = 4,
            loadedLocalModels = 1,
            buildRunning = false,
            userMax = 4,
        )
        column.addView(sectionTitle(t("Resources", "자원")))
        column.addView(body(
            t(
                "Free ${mem.availableMb}MB · heap ${(mem.heapUsedFraction * 100).toInt()}% · " +
                    "parallel tasks ≤ ${decision.count} (${decision.reason})",
                "여유 ${mem.availableMb}MB · 힙 ${(mem.heapUsedFraction * 100).toInt()}% · " +
                    "병렬 작업 ≤ ${decision.count} (${decision.reason})",
            ),
        ))

        column.addView(sectionTitle(t("Project languages", "프로젝트 언어")))
        if (projectDir == null) {
            column.addView(body(t("No project open.", "열린 프로젝트가 없습니다.")))
        } else {
            val markers = projectDir!!.listFiles()?.map { it.name } ?: emptyList()
            val detected = com.aide.mobilestudio.ai.lang.LanguageRegistry.byProjectFiles(markers)
            if (detected.isEmpty()) {
                column.addView(body(t("No build/config markers detected.", "빌드·설정 마커를 찾지 못했습니다.")))
            } else {
                for (spec in detected) {
                    val caps = com.aide.mobilestudio.ai.lang.LanguageRegistry.capabilities(spec, com.aide.mobilestudio.ai.lang.PathToolProbe)
                    column.addView(TextView(this).apply {
                        text = "${spec.displayName} — " + if (caps.compile.available) "compile: ${caps.compile.tool}" else "compile: unavailable"
                        setTextColor(if (caps.compile.available) StudioUi.palette.success else StudioUi.palette.textSoft)
                        textSize = StudioUi.TypeSmall
                        setPadding(0, dp(3), 0, dp(3))
                    })
                }
            }
        }

        column.addView(sectionTitle(t("Language servers (LSP)", "언어 서버 (LSP)")))
        val lsp = com.aide.mobilestudio.ai.lsp.LspServerRegistry.availability(com.aide.mobilestudio.ai.lang.PathToolProbe)
        val availableCount = lsp.count { it.second }
        for ((spec, available) in lsp) {
            column.addView(TextView(this).apply {
                text = (if (available) "✔ " else "· ") + spec.command + "  (" + spec.languageIds.joinToString(", ") + ")"
                setTextColor(if (available) StudioUi.palette.success else StudioUi.palette.muted)
                textSize = StudioUi.TypeSmall
                setPadding(0, dp(3), 0, dp(3))
            })
        }
        column.addView(body(
            t(
                "$availableCount of ${lsp.size} language servers are present on PATH. Others are unavailable.",
                "${lsp.size}개 중 ${availableCount}개 언어 서버가 PATH에 있습니다. 나머지는 사용할 수 없습니다."
            ),
        ))

        column.addView(sectionTitle(t("MCP servers", "MCP 서버")))
        val mcp = runCatching { engine.mcpServerNames() }.getOrDefault(emptyList())
        if (mcp.isEmpty()) {
            column.addView(body(t("No MCP servers configured.", "설정된 MCP 서버가 없습니다.")))
        } else {
            for (name in mcp) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, dp(4), 0, dp(4))
                }
                row.addView(TextView(this).apply {
                    text = name
                    setTextColor(StudioUi.palette.text)
                    textSize = 13f
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                })
                row.addView(AiUiKit.actionChip(this, t("Test", "테스트")) { _ ->
                    scope.launch {
                        val status = engine.testMcpServer(name)
                        AlertDialog.Builder(this@AiWorkspaceActivity)
                            .setTitle(name)
                            .setMessage(status)
                            .setPositiveButton("OK", null)
                            .show()
                    }
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(6) })
                row.addView(AiUiKit.actionChip(this, t("Delete", "삭제")) { _ ->
                    engine.removeMcpServer(name)
                    Toast.makeText(this@AiWorkspaceActivity, t("Removed $name", "$name 삭제됨"), Toast.LENGTH_SHORT).show()
                    showTab(Tab.SETTINGS)
                })
                column.addView(row)
            }
        }
        column.addView(AiUiKit.ghostButton(this, t("+ Add MCP server", "+ MCP 서버 추가")) { _ -> promptAddMcp() })

        return ScrollView(this).apply { addView(column) }
    }

    private fun promptAddMcp() {
        val column = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(8))
        }
        val nameField = EditText(this).apply { hint = t("Name (e.g. filesystem)", "이름 (예: filesystem)") }
        val commandField = EditText(this).apply {
            hint = t("Command (e.g. npx -y @modelcontextprotocol/server-filesystem /sdcard)", "명령 (예: npx -y @modelcontextprotocol/server-filesystem /sdcard)")
            typeface = Typeface.MONOSPACE
            textSize = StudioUi.TypeSmall
        }
        column.addView(label(t("Name", "이름")))
        column.addView(nameField)
        column.addView(label(t("Command", "명령")))
        column.addView(commandField)
        AlertDialog.Builder(this)
            .setTitle(t("Add MCP server", "MCP 서버 추가"))
            .setView(column)
            .setPositiveButton(t("Add", "추가")) { _, _ ->
                val name = nameField.text.toString().trim()
                val command = commandField.text.toString().trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
                if (name.isNotEmpty() && command.isNotEmpty()) {
                    engine.addMcpServer(name, command)
                    Toast.makeText(this, t("Added $name", "$name 추가됨"), Toast.LENGTH_SHORT).show()
                    showTab(Tab.SETTINGS)
                }
            }
            .setNegativeButton(t("Cancel", "취소"), null)
            .show()
    }

    private fun field(hintText: String, value: String, secret: Boolean = false): Pair<String, EditText> {
        val et = EditText(this).apply {
            hint = hintText
            setText(value)
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.muted)
            textSize = StudioUi.TypeBody
            if (secret) inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        return hintText to et
    }

    private fun label(text: String): TextView = TextView(this).apply {
        this.text = text
        setTextColor(StudioUi.palette.textSoft)
        textSize = 11f
        setPadding(0, dp(10), 0, dp(2))
    }

    // ----------------------------------------------------------- Terminal tab

    private lateinit var terminalOutput: TextView
    private lateinit var terminalInput: EditText

    private fun buildTerminalTab(): View {
        val p = StudioUi.palette
        val wrapper = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        terminalOutput = TextView(this).apply {
            setTextColor(p.text)
            textSize = StudioUi.TypeSmall
            typeface = Typeface.MONOSPACE
            setPadding(dp(10), dp(10), dp(10), dp(10))
        }
        val scroll = ScrollView(this).apply { addView(terminalOutput) }
        wrapper.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(p.surface)
            setPadding(dp(8), dp(6), dp(8), dp(6))
        }
        terminalInput = EditText(this).apply {
            hint = t("Shell command…", "셸 명령…")
            setHintTextColor(p.muted)
            setTextColor(p.text)
            textSize = 13f
            typeface = Typeface.MONOSPACE
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        row.addView(terminalInput)
        row.addView(Button(this).apply {
            text = t("Run", "실행")
            isAllCaps = false
            setOnClickListener { runTerminalCommand() }
        })
        wrapper.addView(row)
        return wrapper
    }

    private fun runTerminalCommand() {
        val cmd = terminalInput.text?.toString()?.trim().orEmpty()
        if (cmd.isEmpty()) return
        terminalInput.setText("")
        val cwd = projectDir ?: File(filesDir, "workspace").apply { mkdirs() }
        terminalOutput.append("$ $cmd\n")
        Thread {
            val env = runCatching { ShellEnvironment.buildToolEnvironment(this, cwd) }.getOrDefault(emptyMap())
            val result = runCatching {
                ShellRunner.runSh(this, cmd, cwd, env, 100_000) { line ->
                    runOnUiThread { terminalOutput.append(line + "\n") }
                }
            }.getOrNull()
            runOnUiThread {
                terminalOutput.append("[exit ${result?.exitCode ?: -1}]\n\n")
            }
        }.start()
    }

    // ------------------------------------------------------------- small utils

    private fun t(en: String, ko: String): String = AiWorkspaceStrings.t(this, en, ko)

    private fun sectionTitle(text: String): View = AiUiKit.sectionHeader(this, text)

    private fun body(text: String): TextView = AiUiKit.body(this, text)

    private fun relativePath(path: String): String {
        val root = projectDir ?: return path
        return runCatching { File(path).relativeTo(root).path }.getOrDefault(path)
    }
}
