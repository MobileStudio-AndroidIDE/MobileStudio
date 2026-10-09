package com.aide.mobilestudio

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File
import com.aide.mobilestudio.terminal.TerminalPtySelfTest

class BuildBootActivity : AppCompatActivity() {
    private enum class BootAction { NONE, OPEN, RETRY }

    private val mainHandler = Handler(Looper.getMainLooper())
    private lateinit var jreGradleInstaller: JreGradleInstaller
    private lateinit var sdkInstaller: BootstrapSdkInstaller
    private lateinit var toolchain: ToolchainManager

    private val bootLines = mutableStateListOf<BootLine>()
    private var statusText by mutableStateOf("preparing")
    private var bootProgress by mutableIntStateOf(0)
    private var actionLabel by mutableStateOf("BOOTING...")
    private var actionEnabled by mutableStateOf(false)
    private var bootAction by mutableStateOf(BootAction.NONE)
    private var bootStarted = false
    private var launchedMain = false

    // Setup wizard state
    private var setupStep by mutableIntStateOf(0)
    private var selectedLanguage by mutableStateOf("ko")
    private var selectedDevType by mutableIntStateOf(0) // 0=Android, 1=Python, 2=Both
    private var selectedJdkVersion by mutableStateOf("jdk-21")
    private var selectedGradleVersion by mutableStateOf(ToolchainManager.DEFAULT_GRADLE)
    private var selectedSdkVersion by mutableStateOf("android-34")
    private var wizardComplete by mutableStateOf(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NotificationPermission.ensure(this)

        // adb (BuildBootActivity is the exported launcher entry point):
        //   adb shell am start -n com.aide.mobilestudio/.BuildBootActivity --ez terminal_self_test true
        if (intent.getBooleanExtra("terminal_self_test", false)) {
            intent.removeExtra("terminal_self_test")
            // The self-test exercises the runtime-tools wrappers (mkdir/ls
            // applets) — install first, then run the tests.
            Thread {
                runCatching { installRuntimeTools() }
                TerminalPtySelfTest.runAsync(this)
            }.start()
        } else {
            // Boot-time runtime-tools install.  Nothing called this before —
            // fresh installs were left without busybox/pcre2/wrappers until
            // the user opened ToolsActivity manually.  Background: installAll
            // does network work (GitHub index fetch).
            Thread {
                runCatching { installRuntimeTools() }
            }.start()
        }

        jreGradleInstaller = JreGradleInstaller(this, { appendShell(it) }, { updateProgress(it) })
        sdkInstaller = BootstrapSdkInstaller(this, { appendShell(it) }, { updateProgress(it) })
        toolchain = ToolchainManager(this)

        val prefs = getSharedPreferences("setup_wizard", MODE_PRIVATE)
        val wizardDonePref = prefs.getBoolean("completed", false)

        // Skip setup wizard if already completed or JDK/Gradle already exist
        val hasJdk = toolchain.installedVersions(ToolchainManager.Kind.JDK).isNotEmpty() ||
            jreGradleInstaller.isJdkInstalled
        val hasGradle = toolchain.installedVersions(ToolchainManager.Kind.GRADLE).isNotEmpty() ||
            jreGradleInstaller.isGradleInstalled
        val bothMissing = !hasJdk && !hasGradle

        if (wizardDonePref || !bothMissing) {
            wizardComplete = true
        }

        setContent {
            if (!wizardComplete) {
                SetupWizardScreen(
                    step = setupStep,
                    selectedLanguage = selectedLanguage,
                    onLanguageChange = { selectedLanguage = it; AppLanguage.set(this@BuildBootActivity, it) },
                    selectedDevType = selectedDevType,
                    onDevTypeChange = { selectedDevType = it },
                    selectedJdkVersion = selectedJdkVersion,
                    onJdkVersionChange = { selectedJdkVersion = it },
                    selectedGradleVersion = selectedGradleVersion,
                    onGradleVersionChange = { selectedGradleVersion = it },
                    selectedSdkVersion = selectedSdkVersion,
                    onSdkVersionChange = { selectedSdkVersion = it },
                    onNext = { step ->
                        if (step >= 5) {
                            prefs.edit().putBoolean("completed", true).apply()
                            wizardComplete = true
                            appendShell("wizard complete, starting boot")
                            mainHandler.post { startBootIfNeeded() }
                        } else {
                            setupStep = step
                        }
                    },
                    onSkip = {
                        prefs.edit().putBoolean("completed", true).apply()
                        wizardComplete = true
                        appendShell("wizard skipped, starting boot")
                        mainHandler.post { startBootIfNeeded() }
                    },
                    onPrev = { setupStep = it }
                )
            } else {
                BuildBootScreen(
                    status = statusText,
                    progress = bootProgress,
                    lines = bootLines,
                    actionLabel = actionLabel,
                    actionEnabled = actionEnabled,
                    onAction = { handleActionClick() }
                )
            }
        }

        // If both missing, boot will start after wizard. Otherwise start boot immediately.
        if (bothMissing) {
            appendShell("JDK and Gradle both missing → showing setup wizard")
        } else {
            mainHandler.post { startBootIfNeeded() }
        }
    }

    @Deprecated("Deprecated in Java")
    @android.annotation.SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        MobileStudioCrashGuard.safe(this, "BuildBootActivity.onBackPressed") {
            if (!wizardComplete && setupStep > 0) {
                setupStep--
            } else if (launchedMain) {
                MobileStudioCrashGuard.finishSafely(this)
            } else {
                appendShell("boot continues in background")
                moveTaskToBack(true)
            }
        }
    }

    private fun handleActionClick() {
        when (bootAction) {
            BootAction.OPEN -> MobileStudioCrashGuard.safe(this, "BuildBootActivity.openIde") { launchMain() }
            BootAction.RETRY -> MobileStudioCrashGuard.safe(this, "BuildBootActivity.retry") {
                bootStarted = false
                bootProgress = 0
                actionEnabled = false
                actionLabel = "BOOTING..."
                bootAction = BootAction.NONE
                bootLines.clear()
                appendShell("retry")
                startBootIfNeeded()
            }
            BootAction.NONE -> Unit
        }
    }

    private fun startBootIfNeeded() {
        if (bootStarted) return
        bootStarted = true
        actionEnabled = false
        actionLabel = "BOOTING..."
        bootAction = BootAction.NONE

        // Minimal startup header
        appendShell("StudioShell v1.0.0 \"Core\"")
        appendShell("MobileStudio Command Environment")
        appendShell("Android ARM64 Development Environment")
        appendShell("")

        Thread {
            runCatching { NativeLibSetup.ensureVersionedLibs(this@BuildBootActivity) }
            runCatching { toolchain.ensureAssetOverridesInstalled() }
            runCatching { fixJdkPermissions() }
            runCatching { BuildToolchain.migrateLegacyGradleUserHome(this@BuildBootActivity) }
            mainHandler.post { checkAndInstallJdkGradle() }
        }.start()
    }

    private fun checkAndInstallJdkGradle() {
        val needJdk = toolchain.installedVersions(ToolchainManager.Kind.JDK).isEmpty() &&
            !jreGradleInstaller.isJdkInstalled
        val needGradle = toolchain.installedVersions(ToolchainManager.Kind.GRADLE).isEmpty() &&
            !jreGradleInstaller.isGradleInstalled

        if (!needJdk) {
            val installedVers = toolchain.installedVersions(ToolchainManager.Kind.JDK)
            if (selectedJdkVersion !in installedVers) {
                appendShell("jdk $selectedJdkVersion not installed — using available JDK (no auto-download)")
            } else {
                appendShell("jdk $selectedJdkVersion already installed")
            }
        } else {
            appendShell("jdk: missing — install from Toolchain Manager (no auto-download)")
        }

        if (!needGradle) {
            val installedVers = toolchain.installedVersions(ToolchainManager.Kind.GRADLE)
            if (selectedGradleVersion !in installedVers) {
                appendShell("downloading selected Gradle $selectedGradleVersion")
                downloadVersion(ToolchainManager.Kind.GRADLE, selectedGradleVersion)
            } else {
                appendShell("gradle $selectedGradleVersion already installed")
            }
        } else {
            appendShell("gradle: missing")
        }

        if (needGradle) {
            statusText = "downloading gradle"
            appendShell("install gradle $selectedGradleVersion (JDK auto-download disabled)")
            jreGradleInstaller.installGradleOnly(selectedGradleVersion) { success ->
                if (success) {
                    appendShell("gradle ready")
                    checkAndInstallSdk()
                } else {
                    showFailure("gradle install failed")
                }
            }
            return
        }
        if (!needJdk) {
            checkAndInstallSdk()
            return
        }
        showFailure("jdk/gradle missing — install from Toolchain Manager")
    }

    private fun downloadVersion(kind: ToolchainManager.Kind, version: String) {
        val url = toolchain.downloadUrl(kind, version) ?: return
        val target = File(filesDir, "toolchain/${kind.dirName}/$version")
        DownloadTask.start(
            context = this,
            label = "$kind $version",
            url = url,
            targetDir = target,
            mode = ToolchainDownloadService.MODE_TOOLCHAIN,
            kind = kind,
            version = version
        )
    }

    private fun checkAndInstallSdk() {
        val needSdk = toolchain.installedVersions(ToolchainManager.Kind.SDK).isEmpty() &&
            !sdkInstaller.isInstalled

        if (!needSdk) {
            appendShell("sdk: ok")
            finishBoot()
            return
        } else {
            appendShell("sdk: missing")
        }

        if (selectedDevType != 1) { // Android or Both
            val sdkVers = toolchain.installedVersions(ToolchainManager.Kind.SDK)
            if (sdkVers.isNotEmpty() && selectedSdkVersion !in sdkVers) {
                appendShell("downloading SDK $selectedSdkVersion")
                downloadVersion(ToolchainManager.Kind.SDK, selectedSdkVersion)
            }
        }

        if (!needSdk) {
            finishBoot()
            return
        }
        statusText = "downloading sdk $selectedSdkVersion"
        appendShell("install sdk $selectedSdkVersion")
        sdkInstaller.install { success ->
            if (success) finishBoot() else showFailure("sdk install failed")
        }
    }

    private fun finishBoot() {
        updateProgress(100)
        statusText = "ready"
        appendShell("environment ready")
        actionLabel = "OPEN IDE"
        actionEnabled = true
        bootAction = BootAction.OPEN
        mainHandler.postDelayed({ launchMain() }, 650)
    }

    private fun showFailure(message: String) {
        statusText = "failed"
        appendShell(message)
        actionLabel = "RETRY"
        actionEnabled = true
        bootAction = BootAction.RETRY
    }

    private fun appendShell(message: String) {
        runOnUiThread {
            val kind = when {
                message.contains("failed", ignoreCase = true) || message.contains("error", ignoreCase = true) -> BootLineKind.Error
                message.contains("ready", ignoreCase = true) || message.endsWith(": ok") -> BootLineKind.Success
                message.contains("install", ignoreCase = true) || message.contains("download", ignoreCase = true) -> BootLineKind.Warning
                else -> BootLineKind.Output
            }
            bootLines += BootLine(kind, "> $message")
            while (bootLines.size > 260) bootLines.removeAt(0)
            statusText = message.take(40)
        }
    }

    private fun updateProgress(percent: Int) {
        runOnUiThread { bootProgress = percent.coerceIn(0, 100) }
    }

    private fun copyAssetTree(assetPath: String, targetDir: File) {
        val children = assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            targetDir.parentFile?.mkdirs()
            assets.open(assetPath).use { input ->
                targetDir.outputStream().use { output -> input.copyTo(output) }
            }
            targetDir.setReadable(true, false)
            if (targetDir.name in setOf("cmake", "ninja") || targetDir.extension == "so") {
                targetDir.setExecutable(true, false)
            }
            return
        }
        targetDir.mkdirs()
        children.forEach { child ->
            copyAssetTree("$assetPath/$child", File(targetDir, child))
        }
    }

    private fun installRuntimeTools() {
        if (RuntimeToolsInstaller.areAllToolsInstalled(this)) {
            appendShell("runtime-tools: ok")
            return
        }
        appendShell("runtime-tools: installing busybox, curl, rg, git, tar, unzip...")
        RuntimeToolsInstaller.installAll(this) { msg ->
            appendShell("runtime-tools: $msg")
        }
        appendShell("runtime-tools: done")
    }

    private fun fixJdkPermissions() {
        for (ver in toolchain.installedVersions(ToolchainManager.Kind.JDK)) {
            val dir = File(toolchain.kindDir(ToolchainManager.Kind.JDK), ver).takeIf { it.exists() } ?: continue
            val home = toolchain.resolveJdkHome(dir) ?: continue

            // Regenerate linker wrappers if format version changed (e.g. LD_PRELOAD fix)
            runCatching {
                JdkDownloader.regenerateLinkerWrappers(home, { msg -> appendShell("jdk: $msg") })
            }

            val javaBin = File(home, "bin/java")
            if (javaBin.isFile && !javaBin.canExecute()) {
                javaBin.setExecutable(true, false)
                appendShell("jdk: fixed permissions for $ver")
            }
            File(home, "bin").listFiles()?.forEach { f ->
                if (f.isFile && !f.canExecute()) f.setExecutable(true, false)
            }
        }
    }

    private fun launchMain() {
        if (isFinishing || launchedMain) return
        launchedMain = true
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}

private data class BootLine(val kind: BootLineKind, val text: String)
private enum class BootLineKind { Output, Success, Warning, Error }

@Composable
private fun SetupWizardScreen(
    step: Int,
    selectedLanguage: String,
    onLanguageChange: (String) -> Unit,
    selectedDevType: Int,
    onDevTypeChange: (Int) -> Unit,
    selectedJdkVersion: String,
    onJdkVersionChange: (String) -> Unit,
    selectedGradleVersion: String,
    onGradleVersionChange: (String) -> Unit,
    selectedSdkVersion: String,
    onSdkVersionChange: (String) -> Unit,
    onNext: (Int) -> Unit,
    onSkip: () -> Unit,
    onPrev: ((Int) -> Unit)? = null
) {
    MaterialTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFF0B0E14),
                            Color(0xFF0B0E14),
                            Color(0xFF0B0E14)
                        )
                    )
                )
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp, vertical = 18.dp)
            ) {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = "MobileStudio Setup",
                    color = Color(0xFFDCE3EE),
                    fontSize = 26.sp,
                    fontWeight = FontWeight.Black
                )
                Text(
                    text = if (selectedLanguage == "ko") "환영합니다! 개발 환경을 설정해 주세요."
                    else "Welcome! Configure your development environment.",
                    color = Color(0xFF8E99AC),
                    fontSize = 13.sp
                )

                Spacer(Modifier.height(8.dp))

                // Scrollable content area ??takes all remaining space
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    when (step) {
                        0 -> LanguageStep(selectedLanguage, onLanguageChange)
                        1 -> DevTypeStep(selectedDevType, onDevTypeChange)
                        2 -> if (selectedDevType != 1) AndroidSdkStep(selectedSdkVersion, onSdkVersionChange) else Unit
                        3 -> if (selectedDevType != 1) JdkStep(selectedJdkVersion, onJdkVersionChange) else Unit
                        4 -> if (selectedDevType != 1) GradleStep(selectedGradleVersion, onGradleVersionChange) else Unit
                    }
                }

                // Fixed bottom buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(
                            onClick = onSkip,
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = Color(0xFF8E99AC)
                            )
                        ) {
                            Text(if (selectedLanguage == "ko") "건너뛰기" else "Skip",
                                fontSize = 13.sp)
                        }

                        if (step > 0) {
                            OutlinedButton(
                                onClick = { onPrev?.invoke(step - 1) },
                                shape = RoundedCornerShape(12.dp),
                                colors = ButtonDefaults.outlinedButtonColors(
                                    contentColor = Color(0xFF4F8EF7)
                                )
                            ) {
                                Text(if (selectedLanguage == "ko") "이전" else "Back",
                                    fontSize = 13.sp)
                            }
                        }
                    }

                    val maxStep = if (selectedDevType == 1) 2 else 5
                    val label = if (step >= maxStep - 1)
                        (if (selectedLanguage == "ko") "시작하기" else "Start")
                    else (if (selectedLanguage == "ko") "다음" else "Next")

                    Button(
                        onClick = { onNext(step + 1) },
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF2E8549),
                            contentColor = Color.White
                        )
                    ) {
                        Text(label, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun LanguageStep(selected: String, onChange: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        OptionCard(
            title = "한국어",
            subtitle = "Korean language interface",
            selected = selected == "ko",
            onClick = { onChange("ko") }
        )
        OptionCard(
            title = "English",
            subtitle = "영어 인터페이스",
            selected = selected == "en",
            onClick = { onChange("en") }
        )
    }
}

@Composable
private fun DevTypeStep(selected: Int, onChange: (Int) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        OptionCard(
            title = "Android App Development",
            subtitle = "Build APK apps with Gradle, JDK, SDK, NDK",
            selected = selected == 0,
            onClick = { onChange(0) }
        )
        OptionCard(
            title = "Python Development",
            subtitle = "Python scripting with pip, terminal, AI agent",
            selected = selected == 1,
            onClick = { onChange(1) }
        )
        OptionCard(
            title = "Both (Android + Python)",
            subtitle = "Full environment: Android toolchain + Python",
            selected = selected == 2,
            onClick = { onChange(2) }
        )
    }
}

@Composable
private fun AndroidSdkStep(selected: String, onChange: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Android SDK Version", color = Color(0xFF4F8EF7),
            fontWeight = FontWeight.Bold, fontSize = 15.sp)
        val versions = listOf("android-29", "android-30", "android-31", "android-32",
            "android-33", "android-34", "android-35", "android-36")
        versions.forEach { v ->
            OptionCard(
                title = "API ${v.removePrefix("android-")}",
                subtitle = if (v == "android-35") "Recommended" else "",
                selected = selected == v,
                onClick = { onChange(v) }
            )
        }
    }
}

@Composable
private fun JdkStep(selected: String, onChange: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("JDK Version", color = Color(0xFF4F8EF7),
            fontWeight = FontWeight.Bold, fontSize = 15.sp)
        OptionCard(
            title = "JDK 21 (Android ARM64 Bionic)",
            subtitle = "Custom OpenJDK 21 built for Android with Bionic libc, Gradle Daemon ready",
            selected = selected == "jdk-21",
            onClick = { onChange("jdk-21") }
        )
        OptionCard(
            title = "JDK 17 (Android ARM64 Bionic)",
            subtitle = "Custom OpenJDK 17 built for Android with Bionic libc, LTS support",
            selected = selected == "jdk-17",
            onClick = { onChange("jdk-17") }
        )
    }
}

@Composable
private fun GradleStep(selected: String, onChange: (String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Gradle Version", color = Color(0xFF4F8EF7),
            fontWeight = FontWeight.Bold, fontSize = 15.sp)
        val versions = buildList {
            ToolchainManager.AVAILABLE_GRADLE.forEach { v ->
                add(v to when (v) {
                    "gradle-9.7.0" -> "Latest"
                    "gradle-9.6.1" -> "Recommended (stable)"
                    "gradle-9.5.1" -> "Stable"
                    "gradle-9.4.1" -> "Stable"
                    else -> "Gradle 9.x"
                })
            }
            ToolchainManager.OLD_GRADLE.forEach { v ->
                add(v to "Gradle 8.x (old)")
            }
        }
        versions.forEach { (v, desc) ->
            OptionCard(
                title = v.removePrefix("gradle-"),
                subtitle = desc,
                selected = selected == v,
                onClick = { onChange(v) }
            )
        }
    }
}

@Composable
private fun OptionCard(
    title: String,
    subtitle: String,
    selected: Boolean,
    onClick: () -> Unit
) {
    val bgColor = if (selected) Color(0x33238636) else Color(0xFF222830)
    val borderColor = if (selected) Color(0xFF2E8549) else Color(0xFF3B4252)

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        shape = RoundedCornerShape(14.dp),
        color = bgColor,
        border = androidx.compose.foundation.BorderStroke(
            width = if (selected) 2.dp else 1.dp,
            color = borderColor
        )
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(22.dp)
                    .clip(CircleShape)
                    .background(
                        if (selected) Color(0xFF2E8549) else Color(0xFF303842)
                    ),
                contentAlignment = Alignment.Center
            ) {
                if (selected) Text("\u2714", color = Color.White, fontSize = 12.sp,
                    fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.width(14.dp))
            Column {
                Text(title, color = Color(0xFFDCE3EE),
                    fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                if (subtitle.isNotBlank()) {
                    Text(subtitle, color = Color(0xFF8E99AC), fontSize = 11.sp)
                }
            }
        }
    }
}

// ?????? Original Boot Screen (unchanged) ????????????????????????????????????????????????????????????????????????????

@Composable
private fun BuildBootScreen(
    status: String,
    progress: Int,
    lines: List<BootLine>,
    actionLabel: String,
    actionEnabled: Boolean,
    onAction: () -> Unit
) {
    val animatedProgress by animateFloatAsState(
        targetValue = progress / 100f,
        animationSpec = tween(420),
        label = "boot-progress"
    )
    val pulse by rememberInfiniteTransition(label = "boot-pulse").animateFloat(
        initialValue = 0.65f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(1300),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )
    val cursorAlpha by rememberInfiniteTransition(label = "cursor").animateFloat(
        initialValue = 0.15f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(520),
            repeatMode = RepeatMode.Reverse
        ),
        label = "cursor-alpha"
    )
    val bootStage = bootStageFor(progress)
    val listState = rememberLazyListState()

    LaunchedEffect(lines.size) {
        if (lines.isNotEmpty()) listState.animateScrollToItem(lines.lastIndex)
    }

    MaterialTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFF0B0E14),
                            Color(0xFF0B0E14),
                            Color(0xFF0B0E14)
                        )
                    )
                )
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(260.dp)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                Color(0xFF4F8EF7).copy(alpha = 0.34f * pulse),
                                Color.Transparent
                            )
                        )
                    )
            )

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp, vertical = 18.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                BootHero(status = status, cursorAlpha = cursorAlpha)

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .border(1.dp, Color(0xFF3B4252), RoundedCornerShape(22.dp)),
                    shape = RoundedCornerShape(22.dp),
                    color = Color(0xDD0B0E14),
                    shadowElevation = 12.dp
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(14.dp)
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            StatusDot(color = Color(0xFF46C07D))
                            StatusDot(color = Color(0xFFE3B963))
                            StatusDot(color = Color(0xFFF05C5C))
                            Spacer(Modifier.weight(1f))
                            Text(
                                text = "root@mobilestudio:~",
                                color = Color(0xFF4F8EF7),
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                maxLines = 1
                            )
                        }
                        Spacer(Modifier.height(10.dp))
                        LazyColumn(
                            state = listState,
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            items(lines.size) { index ->
                                BootTerminalLine(lines[index])
                            }
                        }
                    }
                }

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "stage ${bootStage.first}/5·${bootStage.second}",
                            color = Color(0xFF8E99AC),
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.sp,
                            letterSpacing = 0.6.sp,
                            maxLines = 1
                        )
                        Spacer(Modifier.weight(1f))
                        if (progress in 1..99) {
                            Text(
                                text = "$progress%",
                                color = Color(0xFF46C07D),
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                    LinearProgressIndicator(
                        progress = animatedProgress,
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(9.dp)
                            .clip(CircleShape),
                        color = Color(0xFF46C07D),
                        trackColor = Color(0xFF303842)
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        AnimatedVisibility(visible = progress in 1..99) {
                            Text(
                                text = "$progress%",
                                color = Color(0xFF8E99AC),
                                fontFamily = FontFamily.Monospace,
                                fontSize = 12.sp
                            )
                        }
                        Button(
                            onClick = onAction,
                            enabled = actionEnabled,
                            modifier = Modifier
                                .weight(1f)
                                .height(48.dp)
                                .scale(if (actionEnabled) 1f else 0.985f),
                            shape = RoundedCornerShape(15.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF46C07D),
                                contentColor = Color(0xFF0B0E14),
                                disabledContainerColor = Color(0xFF323B44),
                                disabledContentColor = Color(0xFF8E99AC)
                            )
                        ) {
                            Text(actionLabel, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                        }
                    }
                }
            }
        }
    }
}

private fun bootStageFor(progress: Int): Pair<Int, String> {
    val stages = listOf("JDK", "Android SDK", "Gradle", "Build tools", "Startup")
    val idx = (progress.coerceIn(0, 99) / 20).coerceIn(0, stages.size - 1)
    return (idx + 1) to stages[idx]
}

@Composable
private fun BootHero(status: String, cursorAlpha: Float) {
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(42.dp)
                    .background(Color(0xFF4F8EF7), RoundedCornerShape(13.dp)),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "M",
                    color = Color.White,
                    fontWeight = FontWeight.Black,
                    fontSize = 21.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
            Spacer(Modifier.size(10.dp))
            Column {
                Text(
                    text = "MobileStudio",
                    color = Color(0xFFDCE3EE),
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Black,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
                Text(
                    text = "on-device Android IDE bootstrap",
                    color = Color(0xFF46C07D),
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .fillMaxWidth()
                .background(Color(0xFF222830), RoundedCornerShape(14.dp))
                .padding(horizontal = 12.dp, vertical = 9.dp)
        ) {
            Text(
                text = "$ ",
                color = Color(0xFF46C07D),
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = status,
                color = Color(0xFFDCE3EE),
                fontFamily = FontFamily.Monospace,
                fontSize = 13.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "_",
                color = Color(0xFF46C07D).copy(alpha = cursorAlpha),
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun StatusDot(color: Color) {
    Box(
        modifier = Modifier
            .padding(end = 6.dp)
            .size(9.dp)
            .background(color, CircleShape)
    )
}

@Composable
private fun BootTerminalLine(line: BootLine) {
    val color = when (line.kind) {
        BootLineKind.Output -> Color(0xFFBDC7D4)
        BootLineKind.Success -> Color(0xFF46C07D)
        BootLineKind.Warning -> Color(0xFFE3B963)
        BootLineKind.Error -> Color(0xFFF05C5C)
    }
    Text(
        text = line.text,
        color = color,
        fontFamily = FontFamily.Monospace,
        fontSize = 12.sp,
        lineHeight = 16.sp
    )
}
