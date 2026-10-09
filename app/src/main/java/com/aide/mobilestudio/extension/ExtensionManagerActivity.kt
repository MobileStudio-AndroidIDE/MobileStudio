package com.aide.mobilestudio.extension

import android.app.Activity
import android.app.AlertDialog
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import com.aide.mobilestudio.StudioUi
import java.io.File

/**
 * Extension Manager UI — Installed / Available / Updates tabs, search,
 * permission display, install/uninstall/enable/disable/update actions.
 * Settings → Extensions entry point.
 */
class ExtensionManagerActivity : Activity() {

    private lateinit var root: LinearLayout
    private lateinit var tabRow: LinearLayout
    private lateinit var listHost: LinearLayout
    private lateinit var searchBox: EditText
    private var currentTab = 0 // 0=Installed 1=Available 2=Updates
    private var query = ""
    private var loginBtn: TextView? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ExtensionHost.init(this)
        StudioUi.applyTheme(com.aide.mobilestudio.UiTheme.resolve(this))
        root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(StudioUi.palette.bg)
        }
        setContentView(root, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        buildHeader()
        buildSearch()
        buildTabs()
        listHost = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        // Scrollable list (cards overflow the screen otherwise)
        val scroller = ScrollView(this).apply { isVerticalScrollBarEnabled = true; addView(listHost) }
        root.addView(scroller, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        // Focus the extension requested via intent (banner "Click Here to download")
        val focusId = intent?.getStringExtra("extension_id")
        if (focusId != null) {
            currentTab = 1
            refreshTabs()
        }
        refresh()
        // Warm the registry cache; empty/failed registry -> empty Available list (never a fake catalog)
        ExtensionManager.get(this).refreshRepo(force = false) { _, err ->
            runOnUiThread {
                if (err != null) Toast.makeText(this, "registry: $err", Toast.LENGTH_LONG).show()
                refresh()
            }
        }
    }

    private fun headerText(text: String, size: Float, color: Int, bold: Boolean = false) =
        TextView(this).apply {
            this.text = text
            textSize = size
            setTextColor(color)
            setTypeface(Typeface.MONOSPACE, if (bold) Typeface.BOLD else Typeface.NORMAL)
            setPadding(StudioUi.SpacingLg, dp(10), StudioUi.SpacingLg, dp(4))
        }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildHeader() {
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setBackgroundColor(StudioUi.palette.surface)
            setPadding(StudioUi.SpacingLg, dp(12), StudioUi.SpacingSm, dp(12))
        }
        header.addView(headerText("Extensions", 17f, StudioUi.palette.text, true).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        })
        header.addView(textButton("Refresh") {
            Toast.makeText(this, "registry 새로고침 중…", Toast.LENGTH_SHORT).show()
            ExtensionManager.get(this).refreshRepo(force = true) { _, err ->
                runOnUiThread {
                    if (err != null) Toast.makeText(this, "registry: $err", Toast.LENGTH_LONG).show()
                    refreshTabs(); refresh()
                }
            }
        })
        val login = textButton(if (ExtensionAccount.get(this).isLoggedIn()) "My" else "Login") {
            if (ExtensionAccount.get(this).isLoggedIn()) {
                startActivity(android.content.Intent(this, MyExtensionsActivity::class.java))
            } else {
                showLoginDialog()
            }
        }
        loginBtn = login
        header.addView(login)
        refreshLoginButton()
        header.addView(textButton("Developer") { showDeveloperDialog() })
        header.addView(textButton("Error Log") { showErrorLog() })
        header.addView(textButton("Close") { finish() })
        root.addView(header, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(StudioUi.thinDivider(this))
    }

    /** Header Login/My label reflects Gmail or server (browser OAuth) session. */
    private fun refreshLoginButton() {
        val btn = loginBtn ?: return
        btn.text = when {
            ExtensionAccount.get(this).isLoggedIn() -> "My"
            ExtensionServerClient.hasSession(this) ->
                "@" + ExtensionServerClient.loginName(this).ifBlank { "GitHub" }
            else -> "Login"
        }
    }

    override fun onResume() {
        super.onResume()
        refreshLoginButton()
    }

    private fun buildSearch() {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(StudioUi.SpacingLg, dp(8), StudioUi.SpacingLg, dp(8))
        }
        searchBox = EditText(this).apply {
            hint = "Search Extensions"
            setSingleLine(true)
            textSize = 13f
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.textSoft)
            setBackgroundResource(0)
            background = StudioUi.inputBg(this@ExtensionManagerActivity)
            setPadding(dp(10), dp(8), dp(10), dp(8))
            addTextChangedListener(object : TextWatcher {
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun afterTextChanged(s: Editable?) { query = s?.toString()?.trim()?.lowercase() ?: ""; refresh() }
            })
        }
        row.addView(searchBox, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(row)
    }

    private fun buildTabs() {
        tabRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(StudioUi.SpacingSm, 0, StudioUi.SpacingSm, 0) }
        val scroller = HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false; addView(tabRow) }
        root.addView(scroller, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(StudioUi.thinDivider(this))
        refreshTabs()
    }

    private fun refreshTabs() {
        val names = listOf("Installed", "Available", "Updates")
        val updates = ExtensionManager.get(this).updates().size
        tabRow.removeAllViews()
        names.forEachIndexed { i, name ->
            val label = if (i == 2 && updates > 0) "$name ($updates)" else name
            val tab = TextView(this).apply {
                text = label
                textSize = 13f
                setTypeface(Typeface.MONOSPACE, if (currentTab == i) Typeface.BOLD else Typeface.NORMAL)
                setTextColor(if (currentTab == i) StudioUi.palette.accent else StudioUi.palette.textSoft)
                background = if (currentTab == i) StudioUi.roundedInt(this@ExtensionManagerActivity, fill = StudioUi.palette.surface2, radiusDp = StudioUi.RadiusMd) else null
                setPadding(dp(14), dp(8), dp(14), dp(8))
                setOnClickListener { currentTab = i; refreshTabs(); refresh() }
            }
            tabRow.addView(tab)
        }
    }

    private fun refresh() {
        listHost.removeAllViews()
        val manager = ExtensionManager.get(this)
        when (currentTab) {
            0 -> {
                val items = manager.installed().filter { query.isEmpty() || it.manifest.name.lowercase().contains(query) || it.manifest.id.contains(query) }
                if (items.isEmpty()) listHost.addView(emptyView("설치된 Extension이 없습니다.\nAvailable 탭에서 설치하세요."))
                items.forEach { listHost.addView(extensionCardInstalled(it)) }
            }
            1 -> {
                val installedIds = manager.installed().map { it.manifest.id }.toSet()
                // Only entries that actually exist in the registry (never a hardcoded catalog)
                val items = manager.cachedRepo()
                    .filter { query.isEmpty() || it.name.lowercase().contains(query) || it.id.contains(query) || it.description.lowercase().contains(query) }
                if (items.isEmpty()) {
                    listHost.addView(emptyView("registry에서 확장을 가져올 수 없거나 등록된 확장이 없습니다.\nRefresh 버튼으로 다시 시도하세요."))
                }
                items.forEach { entry ->
                    val inst = manager.installed().firstOrNull { it.manifest.id == entry.id }
                    listHost.addView(extensionCardAvailable(entry, inst, installedIds))
                }
            }
            2 -> {
                val updates = manager.updates()
                if (updates.isEmpty()) listHost.addView(emptyView("업데이트가 없습니다."))
                updates.forEach { (inst, entry) -> listHost.addView(extensionCardUpdate(inst, entry)) }
            }
        }
    }

    private fun emptyView(text: String) = TextView(this).apply {
        this.text = text
        textSize = 13f
        setTextColor(StudioUi.palette.textSoft)
        setPadding(StudioUi.SpacingLg, dp(24), StudioUi.SpacingLg, dp(24))
        gravity = Gravity.CENTER
    }

    private fun extensionCardInstalled(it: ExtensionManager.InstalledExtension): View {
        val card = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(StudioUi.SpacingLg, dp(12), StudioUi.SpacingLg, dp(12)) }
        card.addView(titleRow(it.manifest.name, it.manifest.version, if (it.enabled) "Enabled" else "Disabled", if (it.enabled) StudioUi.palette.success else StudioUi.palette.textSoft))
        if (it.manifest.description.isNotBlank()) card.addView(desc(it.manifest.description))
        card.addView(permissionRow(it.manifest.permissions))
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(6), 0, 0) }
        actions.addView(textButton(if (it.enabled) "Disable" else "Enable") {
            ExtensionManager.get(this).setEnabled(it.manifest.id, !it.enabled)
            ExtensionBridge.notifyBannersChanged()
            refresh()
        })
        actions.addView(textButton("Uninstall", danger = true) {
            val ok = ExtensionManager.get(this).uninstall(it.manifest.id)
            Toast.makeText(this, if (ok) "제거됨" else "제거 실패", Toast.LENGTH_SHORT).show()
            ExtensionBridge.notifyBannersChanged()
            refresh()
        })
        card.addView(actions)
        if (it.enabled) {
            actions.addView(textButton("Publish") { publishInstalled(it) })
        }
        return LinearLayout(this).apply { setPadding(dp(6), dp(6), dp(6), dp(6)); setBackgroundColor(Color.TRANSPARENT); addView(card); background = StudioUi.cardBg(this@ExtensionManagerActivity) }
    }

    /** Packs the installed extension's project folder into .msext and publishes it.
     *  Preferred path: server API with a browser-OAuth session (no token in the app).
     *  Legacy fallback: direct GitHub upload with a PAT, if one is still configured. */
    private fun publishInstalled(inst: ExtensionManager.InstalledExtension) {
        val useServer = ExtensionServerClient.hasSession(this)
        if (!useServer && !GitHubPublisher.hasToken(this)) {
            Toast.makeText(this, "GitHub 로그인이 필요합니다 — Login에서 브라우저로 로그인하면 바로 업로드됩니다", Toast.LENGTH_LONG).show()
            showLoginDialog()
            return
        }
        Toast.makeText(this, "패키징 후 게시합니다…", Toast.LENGTH_SHORT).show()
        Thread {
            try {
                val owner = "MobileStudio-AndroidIDE"
                val repo = "MobileStudio-Extensions"
                val packed = ExtensionManager.get(this).packMsext(inst.dir, File(cacheDir, "publish")) { msg ->
                    runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
                } ?: run {
                    runOnUiThread { Toast.makeText(this, "패키징 실패", Toast.LENGTH_LONG).show() }
                    return@Thread
                }
                if (useServer) {
                    ExtensionServerClient.upload(this, packed.first, inst.manifest.id,
                        onProgress = { p -> runOnUiThread { Toast.makeText(this, p, Toast.LENGTH_SHORT).show() } },
                        onDone = { r -> runOnUiThread { Toast.makeText(this, r.message, Toast.LENGTH_LONG).show() } })
                } else {
                    GitHubPublisher.publishExtension(this, owner, repo, packed.first, packed.second,
                        { msg -> runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() } }) { result ->
                        runOnUiThread {
                            Toast.makeText(this, result.message, Toast.LENGTH_LONG).show()
                        }
                    }
                }
            } catch (e: Exception) {
                runOnUiThread { Toast.makeText(this, "게시 실패: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }.start()
    }

    /** Developer dialog: GitHub PAT + developer name/email registration + repo creation. */
    private fun showDeveloperDialog() {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(12), dp(20), dp(8))
        }
        container.addView(TextView(this).apply {
            text = "개발자 등록 — 확장 생성/업로드 시 사용되며 이후에도 재사용됩니다"
            textSize = 13f
            setTextColor(StudioUi.palette.textSoft)
        })

        fun field(hint: String, initial: String = "") = EditText(this).apply {
            this.hint = hint
            setSingleLine(true)
            textSize = 13f
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.textSoft)
            background = StudioUi.inputBg(this@ExtensionManagerActivity)
            setPadding(dp(10), dp(8), dp(10), dp(8))
            setText(initial)
        }

        val profile = DeveloperProfile.get(this)
        val nameField = field("개발자 이름 (20자 이내, 영어+()._ )", profile?.name ?: "")
        val emailField = field("이메일 (gmail)", profile?.email ?: "")
        val tokenField = field("GitHub Personal Access Token (repo scope)")
        container.addView(nameField)
        container.addView(emailField)
        container.addView(tokenField)

        AlertDialog.Builder(this)
            .setTitle("Developer")
            .setView(container)
            .setPositiveButton("등록") { _, _ ->
                val token = tokenField.text.toString().trim()
                if (token.isNotBlank()) GitHubPublisher.setToken(this, token)
                DeveloperProfile.register(this, nameField.text.toString().trim(), emailField.text.toString().trim(), null) { ok, msg ->
                    runOnUiThread {
                        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
                        if (ok) refresh()
                    }
                }
            }
            .setNeutralButton("저장소 생성") { _, _ ->
                val token = tokenField.text.toString().trim()
                if (token.isNotBlank()) GitHubPublisher.setToken(this, token)
                if (!GitHubPublisher.hasToken(this)) {
                    Toast.makeText(this, "GitHub 토큰 필요", Toast.LENGTH_LONG).show()
                    return@setNeutralButton
                }
                Toast.makeText(this, "저장소 생성 중…", Toast.LENGTH_SHORT).show()
                Thread {
                    val result = GitHubPublisher.createRepo(this, "MobileStudio-Extensions", "MobileStudio Extension Repository")
                    runOnUiThread { Toast.makeText(this, result.message, Toast.LENGTH_LONG).show() }
                }.start()
            }
            .setNegativeButton("닫기", null)
            .show()
    }

    private fun extensionCardAvailable(entry: ExtensionManager.RepoEntry, inst: ExtensionManager.InstalledExtension?, installedIds: Set<String>): View {
        val marketplace = ExtensionMarketplace.get(this)
        val card = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(StudioUi.SpacingLg, dp(12), StudioUi.SpacingLg, dp(12)) }
        card.addView(titleRow(entry.name, entry.version, if (inst != null) (if (inst.enabled) "Enabled" else "Disabled") else "", if (inst != null) (if (inst.enabled) StudioUi.palette.success else StudioUi.palette.textSoft) else StudioUi.palette.textSoft, entry.author.ifBlank { DeveloperProfile.get(this)?.name ?: "" }, DeveloperProfile.get(this)?.email ?: ""))
        if (entry.description.isNotBlank()) card.addView(desc(entry.description))
        card.addView(metaRow(entry))
        card.addView(permissionRow(entry.permissions))
        // 신고 누적/관리자 차단 → Extension Hub에 "Under review" 표시
        if (marketplace.isDownloadSuspended(entry.id)) {
            card.addView(TextView(this).apply {
                text = "⚠ Under review — ${marketplace.suspensionReason(entry.id)}"
                textSize = 11f
                setTextColor(StudioUi.palette.warning)
                setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            })
        }
        // 좋아요 수 표시
        card.addView(TextView(this).apply {
            text = "♥ ${marketplace.likeCount(entry.id)}  ·  ${marketplace.reviews(entry.id).size} reviews  ·  ${marketplace.reportCount(entry.id)} reports"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            setTypeface(Typeface.MONOSPACE, Typeface.NORMAL)
        })
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(6), 0, 0) }
        if (entry.download.isNotBlank()) {
            actions.addView(textButton(if (inst != null) "Reinstall" else "Install") {
                installEntry(entry)
            })
        } else {
            actions.addView(textButton("Install") { installOfficial(entry) })
        }
        // 좋아요 (계정당 1회 토글, 로그인 필요)
        actions.addView(textButton(if (ExtensionAccount.get(this).likedExtensions().contains(entry.id)) "♥ Liked" else "♥ Like") {
            val err = marketplace.toggleLike(entry.id)
            if (err != null) Toast.makeText(this, err, Toast.LENGTH_SHORT).show() else refresh()
        })
        // 신고 (하루 1회/확장, 로그인 필요)
        actions.addView(textButton("Report") { showReportDialog(entry) })
        // 상세 정보
        actions.addView(textButton("Details") { showExtensionDetail(entry) })
        card.addView(actions)
        return LinearLayout(this).apply { setPadding(dp(6), dp(6), dp(6), dp(6)); setBackgroundColor(Color.TRANSPARENT); addView(card); background = StudioUi.cardBg(this@ExtensionManagerActivity) }
    }

    /** GitHub 로그인 — 브라우저 OAuth(권장, 토큰 미보관) + 선택적 직접 토큰(고급). */
    private fun showLoginDialog() {
        val account = ExtensionAccount.get(this)
        val hasGoogle = ExtensionAccount.isGoogleSignInAvailable(this) && ExtensionAccount.clientId(this).isNotBlank()
        var dialog: AlertDialog? = null
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(12), dp(20), dp(8))
        }
        val profile = DeveloperProfile.get(this)
        val serverSession = ExtensionServerClient.hasSession(this)
        val status = buildString {
            if (serverSession) {
                val login = ExtensionServerClient.loginName(this@ExtensionManagerActivity).ifBlank { "로그인됨" }
                append("GitHub (서버 세션): $login\n")
            }
            if (GitHubPublisher.hasToken(this@ExtensionManagerActivity)) append("GitHub (직접 토큰): 로그인됨\n")
            if (account.isLoggedIn()) append("Gmail: ${account.current()?.displayName ?: "로그인됨"}\n")
            if (profile != null) append("개발자: ${profile.name}")
        }
        container.addView(TextView(this).apply {
            text = if (status.isBlank()) "로그인 상태 없음 — GitHub 로그인 버튼으로 브라우저 승인을 마치면 업로드가 바로 가능합니다" else status
            textSize = 13f
            setTextColor(StudioUi.palette.textSoft)
        })

        if (serverSession) {
            container.addView(TextView(this).apply {
                text = "▸ 세션 로그아웃"
                textSize = 12f
                setTextColor(StudioUi.palette.error)
                setPadding(0, dp(8), 0, 0)
                setOnClickListener {
                    ExtensionServerClient.logout(this@ExtensionManagerActivity)
                    Toast.makeText(this@ExtensionManagerActivity, "로그아웃되었습니다", Toast.LENGTH_SHORT).show()
                    refreshLoginButton()
                    dialog?.dismiss()
                }
            })
        }

        fun field(hint: String, initial: String = "") = EditText(this).apply {
            this.hint = hint
            setSingleLine(true)
            textSize = 13f
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.textSoft)
            background = StudioUi.inputBg(this@ExtensionManagerActivity)
            setPadding(dp(10), dp(8), dp(10), dp(8))
            setText(initial)
        }

        val tokenField = field(if (GitHubPublisher.hasToken(this)) "고급: 새 GitHub 직접 토큰 (변경 시에만 입력)" else "고급: GitHub 직접 토큰 (선택 — 미사용 권장)")
        if (hasGoogle) {
            container.addView(TextView(this).apply {
                text = "Gmail 계정으로도 로그인할 수 있습니다"
                textSize = 11f
                setTextColor(StudioUi.palette.textSoft)
                setPadding(0, dp(6), 0, 0)
            })
        }
        container.addView(tokenField)

        val builder = AlertDialog.Builder(this)
            .setTitle("Login")
            .setView(container)
            .setPositiveButton("GitHub 로그인") { _, _ ->
                val token = tokenField.text.toString().trim()
                if (token.isNotBlank()) GitHubPublisher.setToken(this, token)
                Toast.makeText(this, "브라우저에서 GitHub 승인을 진행하세요", Toast.LENGTH_SHORT).show()
                ExtensionServerClient.startLogin(this) { msg ->
                    runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
                }
            }
            .setNegativeButton("닫기", null)
        if (hasGoogle) {
            builder.setNeutralButton("Gmail 로그인") { _, _ ->
                try {
                    val intent = account.googleSignInClient().signInIntent
                    startActivityForResult(intent, 0x5101)
                } catch (e: Exception) {
                    Toast.makeText(this, "로그인 실패: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
        dialog = builder.show()
        refreshLoginButton()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: android.content.Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 0x5101) {
            val profile = ExtensionAccount.get(this).handleSignInResult(data)
            if (profile != null) {
                Toast.makeText(this, "로그인됨: ${profile.displayName}", Toast.LENGTH_SHORT).show()
                refresh()
            } else {
                Toast.makeText(this, "로그인 취소됨", Toast.LENGTH_SHORT).show()
            }
        }
    }

    /** 신고 다이얼로그 — 사유 선택 + 메시지. */
    private fun showReportDialog(entry: ExtensionManager.RepoEntry) {
        ExtensionAccount.get(this).requireLogin("신고")?.let {
            Toast.makeText(this, it, Toast.LENGTH_LONG).show()
            return
        }
        val reasons = ExtensionMarketplace.REPORT_REASONS.toTypedArray()
        android.app.AlertDialog.Builder(this)
            .setTitle("Report: ${entry.name}")
            .setItems(reasons) { _, which ->
                val reason = reasons[which]
                val ok = ExtensionMarketplace.get(this).addReport(entry.id, reason, "")
                if (ok != null) Toast.makeText(this, ok, Toast.LENGTH_LONG).show()
                else Toast.makeText(this, "신고가 접수되었습니다: $reason", Toast.LENGTH_SHORT).show()
                refresh()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** 확장 상세 페이지 — 상세 정보 + 후기 + 보안 검사 상태. */
    private fun showExtensionDetail(entry: ExtensionManager.RepoEntry) {
        val marketplace = ExtensionMarketplace.get(this)
        val container = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), dp(16), dp(20), dp(8)) }

        fun row(label: String, value: String) {
            container.addView(TextView(this).apply {
                text = "$label: $value"
                textSize = 12f
                setTextColor(StudioUi.palette.text)
                setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            })
        }
            row("Name", entry.name)
        row("ID", entry.id)
        row("Version", entry.version)
        row("Author", entry.author)
        row("Size", if (entry.size > 0) "${entry.size / 1024} KB" else "-")
        row("Downloads", "${marketplace.downloadCount(entry.id)}")
        row("Likes", "${marketplace.likeCount(entry.id)}")
        row("Reports", "${marketplace.reportCount(entry.id)}")
        row("Required MobileStudio", entry.minStudioVersion)
        row("Permissions", entry.permissions.joinToString(", ").ifBlank { "-" })
        row("Security scan", if (marketplace.isDownloadSuspended(entry.id)) "Under review" else "Passed (automated)")
        container.addView(TextView(this).apply {
            text = "자동 검사 통과가 악성코드 부재를 보장하지는 않습니다"
            textSize = 10f
            setTextColor(StudioUi.palette.textSoft)
        })

        container.addView(TextView(this).apply {
            text = "Reviews"
            textSize = 13f
            setTextColor(StudioUi.palette.text)
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
            setPadding(0, dp(10), 0, dp(4))
        })
        val reviews = marketplace.reviews(entry.id)
        if (reviews.isEmpty()) {
            container.addView(TextView(this).apply { text = "아직 후기가 없습니다"; textSize = 11f; setTextColor(StudioUi.palette.textSoft) })
        } else {
            for (r in reviews.take(10)) {
                container.addView(TextView(this).apply {
                    text = "${r.author} · ${java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date(r.createdAt))}${if (r.edited) " (edited)" else ""}\n${r.content}"
                    textSize = 11f
                    setTextColor(StudioUi.palette.text)
                    setPadding(0, dp(4), 0, 0)
                })
            }
        }

        android.app.AlertDialog.Builder(this)
            .setTitle("Extension Detail")
            .setView(container)
            .setPositiveButton("Install") { _, _ -> if (entry.download.isNotBlank()) installEntry(entry) else installOfficial(entry) }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun extensionCardUpdate(inst: ExtensionManager.InstalledExtension, entry: ExtensionManager.RepoEntry): View {
        val card = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(StudioUi.SpacingLg, dp(12), StudioUi.SpacingLg, dp(12)) }
        card.addView(titleRow(entry.name, "${inst.manifest.version} → ${entry.version}", "Update", StudioUi.palette.accent))
        if (entry.description.isNotBlank()) card.addView(desc(entry.description))
        card.addView(permissionRow(entry.permissions))
        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(6), 0, 0) }
        actions.addView(textButton("Update") {
            if (entry.download.isNotBlank()) installEntry(entry) else installOfficial(entry)
        })
        card.addView(actions)
        return LinearLayout(this).apply { setPadding(dp(6), dp(6), dp(6), dp(6)); setBackgroundColor(Color.TRANSPARENT); addView(card); background = StudioUi.cardBg(this@ExtensionManagerActivity) }
    }

    private fun titleRow(name: String, version: String, status: String, statusColor: Int, author: String = "", email: String = ""): View {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(TextView(this).apply {
            text = name
            textSize = 17f
            setTextColor(StudioUi.palette.text)
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
        })
        if (author.isNotBlank()) {
            col.addView(TextView(this).apply {
                text = "developed by $author" + (if (email.isNotBlank()) "  $email" else "")
                textSize = 11f
                setTextColor(StudioUi.palette.textSoft)
                setTypeface(Typeface.MONOSPACE, Typeface.NORMAL)
            })
        }
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        row.addView(col, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        if (status.isNotBlank()) row.addView(TextView(this).apply {
            text = status
            textSize = 11f
            setTextColor(statusColor)
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
        })
        if (version.isNotBlank()) row.addView(TextView(this).apply {
            text = if (status.isNotBlank()) "  v$version" else "v$version"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            setTypeface(Typeface.MONOSPACE, Typeface.NORMAL)
        })
        return row
    }

    /** Description in small text; over 100 chars truncated with a "...show more" toggle. */
    private fun desc(text: String): View {
        if (text.length <= 100) {
            return TextView(this).apply {
                this.text = text
                textSize = 12f
                setTextColor(StudioUi.palette.textSoft)
                setPadding(0, dp(4), 0, 0)
            }
        }
        val shortText = text.take(100) + "…"
        val tv = TextView(this).apply {
            textSize = 12f
            setTextColor(StudioUi.palette.textSoft)
            setPadding(0, dp(4), 0, 0)
        }
        var expanded = false
        fun apply() {
            tv.text = if (expanded) text + "\n...show less" else shortText + "...show more"
        }
        apply()
        tv.setOnClickListener {
            expanded = !expanded
            apply()
        }
        return tv
    }

    private fun metaRow(entry: ExtensionManager.RepoEntry): View = TextView(this).apply {
        val size = if (entry.size > 0) "Size ${(entry.size / 1_000_000.0).let { String.format("%.1f MB", it)}}" else ""
        this.text = listOfNotNull("Version ${entry.version}", size.takeIf { it.isNotBlank() }, "by ${entry.author}".takeIf { entry.author.isNotBlank() }).joinToString("  ·  ")
        textSize = 11f
        setTextColor(StudioUi.palette.textSoft)
        setPadding(0, dp(3), 0, 0)
    }

    private fun permissionRow(perms: List<String>): View {
        val row = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(6), 0, 0) }
        if (perms.isEmpty()) return TextView(this).apply {
            text = "권한: 없음"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
        }
        row.addView(TextView(this).apply {
            text = "권한:"
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
        })
        perms.forEach { id ->
            row.addView(TextView(this).apply {
                val sensitive = ExtensionApi.isSensitive(id)
                text = "${if (sensitive) "⚠" else "✓"} ${ExtensionApi.permissionLabel(id)}${idLabelSuffix(id)}"
                textSize = 11f
                setTextColor(if (sensitive) StudioUi.palette.warning else StudioUi.palette.textSoft)
                setPadding(dp(10), dp(1), 0, 0)
            })
        }
        return row
    }

    private fun idLabelSuffix(id: String) = " (${id})".takeIf { ExtensionApi.permissionLabel(id) != id } ?: ""

    /** Install only from a real download URL — never fabricate release URLs (404 source). */
    private fun installOfficial(entry: ExtensionManager.RepoEntry) {
        if (entry.download.isBlank()) {
            Toast.makeText(this,
                "이 확장은 아직 배포 파일이 등록되지 않았습니다 (download URL 없음)",
                Toast.LENGTH_LONG).show()
            return
        }
        installEntry(entry)
    }

    private fun installEntry(entry: ExtensionManager.RepoEntry) {
        val manager = ExtensionManager.get(this)
        manager.install(entry, { msg -> runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() } }) { ok, msg ->
            runOnUiThread {
                Toast.makeText(this, msg, if (ok) Toast.LENGTH_SHORT else Toast.LENGTH_LONG).show()
                if (ok) ExtensionBridge.notifyBannersChanged()
                refreshTabs(); refresh()
            }
        }
    }

    private fun showErrorLog() {
        val log = CrashReporter.readLog(this)
        val scroller = ScrollView(this)
        val text = TextView(this).apply {
            this.text = "Error Log\n\n[[[ERROR ERROR ERROR ERROR ERROR]]]\n\n$log"
            textSize = 12f
            setTextColor(StudioUi.palette.text)
            setTypeface(Typeface.MONOSPACE, Typeface.NORMAL)
            setPadding(StudioUi.SpacingLg, dp(16), StudioUi.SpacingLg, dp(16))
            setTextIsSelectable(true)
        }
        scroller.addView(text)
        root.removeAllViews()
        val back = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(StudioUi.SpacingLg, dp(12), StudioUi.SpacingSm, dp(12)); setBackgroundColor(StudioUi.palette.surface) }
        back.addView(textButton("← Extensions") { onCreate(null) })
        back.addView(textButton("Clear Log", danger = true) { CrashReporter.clearLog(this); showErrorLog() })
        root.addView(back)
        root.addView(StudioUi.thinDivider(this))
        root.addView(scroller, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    }

    private fun textButton(label: String, danger: Boolean = false, onClick: () -> Unit) = TextView(this).apply {
        text = label
        textSize = 12f
        setTypeface(Typeface.MONOSPACE, Typeface.BOLD)
        setTextColor(if (danger) StudioUi.palette.error else StudioUi.palette.accent)
        background = StudioUi.ripple(this@ExtensionManagerActivity, StudioUi.roundedInt(this@ExtensionManagerActivity, fill = StudioUi.palette.surface2, radiusDp = 4))
        setPadding(dp(12), dp(8), dp(12), dp(8))
        StudioUi.touchLift(this)
        setOnClickListener { onClick() }
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(6) }
    }
}
