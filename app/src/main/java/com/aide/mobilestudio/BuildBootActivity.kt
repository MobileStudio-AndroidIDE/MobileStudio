package com.aide.mobilestudio

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.background
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File
import com.aide.mobilestudio.terminal.TerminalPtySelfTest

/**
 * Launcher entry point.
 *
 * First launch shows ONLY a language selection (ko/en) and then goes straight
 * to the IDE — no build-tool downloads, no "check what tools are installed"
 * step. Every tool (JDK / Gradle / NDK / CMake / Ninja / LLVM) is provisioned
 * lazily by [BuildProvisioner] the first time a build actually needs it.
 */
class BuildBootActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "BuildBootActivity"
    }

    private var launchedMain = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NotificationPermission.ensure(this)

        // adb (BuildBootActivity is the exported launcher entry point):
        //   adb shell am start -n com.aide.mobilestudio/.BuildBootActivity --ez terminal_self_test true
        // Explicit diagnostic only — never runs on a normal launch.
        if (intent.getBooleanExtra("terminal_self_test", false)) {
            intent.removeExtra("terminal_self_test")
            // The self-test exercises the runtime-tools wrappers (mkdir/ls
            // applets) — install first, then run the tests.
            Thread {
                runCatching { installRuntimeTools() }
                TerminalPtySelfTest.runAsync(this)
            }.start()
        }

        val prefs = getSharedPreferences("setup_wizard", MODE_PRIVATE)
        val wizardDone = prefs.getBoolean("completed", false)

        if (wizardDone) {
            // Existing user (old wizard already completed, or returning after
            // the language step): land in the IDE directly. Only cheap local
            // maintenance runs here — never a download.
            runLocalMaintenance()
            launchMain()
            return
        }

        setContent {
            LanguageSetupScreen(
                initial = initialLanguage(),
                onLanguageChange = { AppLanguage.set(this@BuildBootActivity, it) },
                onStart = {
                    // Mark the wizard completed so it never reappears — the
                    // old multi-step wizard (dev type / SDK / JDK / Gradle)
                    // is retired; tools are provisioned at build time.
                    prefs.edit().putBoolean("completed", true).apply()
                    runLocalMaintenance()
                    MobileStudioCrashGuard.safe(this@BuildBootActivity, "BuildBootActivity.openIde") {
                        launchMain()
                    }
                }
            )
        }
    }

    @Deprecated("Deprecated in Java")
    @android.annotation.SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        MobileStudioCrashGuard.safe(this, "BuildBootActivity.onBackPressed") {
            if (launchedMain) {
                MobileStudioCrashGuard.finishSafely(this)
            } else {
                moveTaskToBack(true)
            }
        }
    }

    private fun initialLanguage(): String {
        val current = AppLanguage.current(this).code
        return if (current == "en") "en" else "ko"
    }

    /**
     * Cheap, LOCAL-only hygiene (no network): versioned native libs, legacy
     * Gradle-home migration, JDK exec permissions + linker wrappers.
     * All network provisioning moved to build time (BuildProvisioner).
     */
    private fun runLocalMaintenance() {
        Thread {
            runCatching { NativeLibSetup.ensureVersionedLibs(this@BuildBootActivity) }
                .onFailure { Log.w(TAG, "native lib setup failed: ${it.message}") }
            runCatching { BuildToolchain.migrateLegacyGradleUserHome(this@BuildBootActivity) }
                .onFailure { Log.w(TAG, "gradle home migration failed: ${it.message}") }
            runCatching { fixJdkPermissions() }
                .onFailure { Log.w(TAG, "jdk permission fix failed: ${it.message}") }
        }.start()
    }

    private fun fixJdkPermissions() {
        val toolchain = ToolchainManager(this)
        for (ver in toolchain.installedVersions(ToolchainManager.Kind.JDK)) {
            val dir = File(toolchain.kindDir(ToolchainManager.Kind.JDK), ver).takeIf { it.exists() } ?: continue
            val home = toolchain.resolveJdkHome(dir) ?: continue

            // Regenerate linker wrappers if format version changed (e.g. LD_PRELOAD fix)
            runCatching {
                JdkDownloader.regenerateLinkerWrappers(home, { msg -> Log.i(TAG, "jdk: $msg") })
            }

            val javaBin = File(home, "bin/java")
            if (javaBin.isFile && !javaBin.canExecute()) {
                javaBin.setExecutable(true, false)
                Log.i(TAG, "jdk: fixed permissions for $ver")
            }
            File(home, "bin").listFiles()?.forEach { f ->
                if (f.isFile && !f.canExecute()) f.setExecutable(true, false)
            }
        }
    }

    /** Only for the explicit terminal_self_test adb entry point. */
    private fun installRuntimeTools() {
        if (RuntimeToolsInstaller.areAllToolsInstalled(this)) {
            Log.i(TAG, "runtime-tools: ok")
            return
        }
        Log.i(TAG, "runtime-tools: installing busybox, curl, rg, git, tar, unzip...")
        RuntimeToolsInstaller.installAll(this) { msg -> Log.i(TAG, "runtime-tools: $msg") }
        Log.i(TAG, "runtime-tools: done")
    }

    private fun launchMain() {
        if (isFinishing || launchedMain) return
        launchedMain = true
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}

@Composable
private fun LanguageSetupScreen(
    initial: String,
    onLanguageChange: (String) -> Unit,
    onStart: () -> Unit
) {
    var selected by remember { mutableStateOf(initial) }
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
                Spacer(Modifier.height(24.dp))
                Text(
                    text = "MobileStudio",
                    color = Color(0xFFDCE3EE),
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Black
                )
                Text(
                    text = if (selected == "ko") "언어를 선택해 주세요." else "Choose your language.",
                    color = Color(0xFF8E99AC),
                    fontSize = 13.sp
                )

                Spacer(Modifier.height(24.dp))

                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    OptionCard(
                        title = "한국어",
                        subtitle = "Korean language interface",
                        selected = selected == "ko",
                        onClick = {
                            selected = "ko"
                            onLanguageChange("ko")
                        }
                    )
                    OptionCard(
                        title = "English",
                        subtitle = "영어 인터페이스",
                        selected = selected == "en",
                        onClick = {
                            selected = "en"
                            onLanguageChange("en")
                        }
                    )
                }

                Spacer(Modifier.weight(1f))

                Button(
                    onClick = onStart,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF2E8549),
                        contentColor = Color.White
                    )
                ) {
                    Text(
                        text = if (selected == "ko") "시작하기" else "Start",
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }
            }
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
