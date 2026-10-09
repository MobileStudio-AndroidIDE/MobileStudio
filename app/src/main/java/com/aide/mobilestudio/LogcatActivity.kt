package com.aide.mobilestudio

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.BufferedReader
import java.io.File
import com.aide.mobilestudio.LogcatManager
import com.aide.mobilestudio.ExecHelper
import androidx.compose.ui.platform.LocalContext

class LogcatActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val packageName = intent.getStringExtra(EXTRA_PACKAGE) ?: packageName
        val pid = intent.getIntExtra(EXTRA_PID, 0)
        setContent {
            ThemeManager.ProvideTheme {
                LogcatScreen(packageName = packageName, pid = pid, onBack = { finish() })
            }
        }
    }

    companion object {
        const val EXTRA_PACKAGE = "package_name"
        const val EXTRA_PID = "pid"
        fun start(context: android.content.Context, packageName: String, pid: Int = 0) {
            context.startActivity(
                android.content.Intent(context, LogcatActivity::class.java)
                    .putExtra(EXTRA_PACKAGE, packageName)
                    .putExtra(EXTRA_PID, pid)
            )
        }
    }
}

data class LogEntry(
    val raw: String,
    val tag: String = "",
    val priority: String = "I",
    val message: String = "",
    @Transient var isCrash: Boolean = false
) {
    companion object {
        fun parse(line: String): LogEntry {
            return try {
                val m = Regex("""^(\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+)\s+(\d+)\s+(\d+)\s+([VDIWEF])\s+(\S+)\s*:\s*(.*)""").find(line)
                if (m != null) {
                    val priority = m.groupValues[4]
                    val tag = m.groupValues[5]
                    val msg = m.groupValues[6]
                    LogEntry(line, tag, priority, msg)
                } else {
                    LogEntry(line, priority = "I", message = line)
                }
            } catch (_: Exception) {
                LogEntry(line, priority = "I", message = line)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LogcatScreen(packageName: String, pid: Int = 0, onBack: () -> Unit) {
    val context = LocalContext.current
    val logs = remember { mutableStateListOf<LogEntry>() }
    var isCapturing by remember { mutableStateOf(true) }
    var filterText by remember { mutableStateOf("") }
    var priorityFilter by remember { mutableStateOf(setOf("V", "D", "I", "W", "E", "F")) }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    var process by remember { mutableStateOf<Process?>(null) }
    var modeText by remember { mutableStateOf("Initializing...") }

    LaunchedEffect(isCapturing, packageName, pid) {
        if (!isCapturing) return@LaunchedEffect
        launch(Dispatchers.IO) {
            try {
                val logcatManager = LogcatManager(context, { _, _ -> }, {})
                val adb = logcatManager.resolveAdbBinary()
                
// Clear logs first
                 if (adb != null) {
                     ExecHelper.createProcess(context, listOf(adb, "logcat", "-c"), cwd = null, env = System.getenv())
                         .start().waitFor()
                 } else {
                     Runtime.getRuntime().exec("logcat -c").waitFor()
                 }

                val cmd = if (adb != null) {
                    modeText = "Using ADB logcat"
                    val args = mutableListOf(adb, "logcat", "-v", "time")
                    if (pid > 0) args.add("--pid=$pid") else args.add("*:V")
                    args
                } else {
                    modeText = "Using direct logcat (ADB missing)"
                    if (pid > 0) listOf("logcat", "--pid=$pid")
                    else listOf("logcat", "-v", "time", "*:V")
                }

                val pb = ExecHelper.createProcess(context, cmd, cwd = null, env = System.getenv())
                pb.redirectErrorStream(true)
                val proc = pb.start()
                process = proc
                val reader: BufferedReader = proc.inputStream.bufferedReader()
                while (isActive) {
                    val line = reader.readLine() ?: break
                    // (PID filter logic remains same or is handled by command)
                    val entry = LogEntry.parse(line)
                    synchronized(logs) {
                        logs.add(entry)
                        // Detect crashes in real-time
                        if (entry.message.contains("FATAL EXCEPTION") || entry.priority == "E") {
                            if (entry.message.contains("FATAL EXCEPTION") ||
                                entry.message.contains("AndroidRuntime") ||
                                entry.message.contains("java.lang.") ||
                                entry.message.contains("kotlin.")) {
                                entry.isCrash = true
                            }
                        }
                        if (logs.size > 2000) {
                            val removeCount = logs.size - 1500
                            repeat(removeCount) { logs.removeAt(0) }
                        }
                    }
                }
            } catch (_: Exception) {}
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            process?.destroy()
        }
    }

    val filteredLogs = remember(logs.size, filterText, priorityFilter) {
        logs.filter { entry ->
            entry.priority in priorityFilter &&
                (filterText.isBlank() || entry.message.contains(filterText, true) || entry.tag.contains(filterText, true))
        }
    }

    LaunchedEffect(filteredLogs.size) {
        if (filteredLogs.isNotEmpty()) {
            listState.animateScrollToItem(filteredLogs.size - 1)
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { 
                    Column {
                        Text("Logcat: $packageName", fontSize = 14.sp)
                        Text(modeText, fontSize = 10.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                },
                navigationIcon = {
                    TextButton(onClick = onBack) { Text("Back") }
                },
                actions = {
                    TextButton(onClick = {
                        val activity = context as? android.app.Activity
                        if (activity != null) MobileStudioCrashGuard.showCrashAnalysis(activity, null)
                    }) {
                        Text("Crash", color = Color(StudioUi.color(StudioUi.Red)))
                    }
                    TextButton(onClick = { logs.clear() }) { Text("Clear") }
                    TextButton(onClick = { isCapturing = !isCapturing }) {
                        Text(if (isCapturing) "Stop" else "Start")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.surface
                )
            )
        }
    ) { padding ->
        Column(Modifier.padding(padding)) {
            OutlinedTextField(
                value = filterText,
                onValueChange = { filterText = it },
                modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                placeholder = { Text("Filter logs...", fontSize = 12.sp) },
                singleLine = true,
                textStyle = MaterialTheme.typography.bodySmall
            )

            Row(
                Modifier.padding(horizontal = 8.dp, vertical = 2.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                val levels = listOf("V" to Color(StudioUi.color(StudioUi.TextSoft)), "D" to Color(StudioUi.color(StudioUi.Blue)), "I" to Color(StudioUi.color(StudioUi.Green)), "W" to Color(StudioUi.color(StudioUi.Amber)), "E" to Color(StudioUi.color(StudioUi.Red)), "F" to Color(StudioUi.color(StudioUi.Purple)))
                levels.forEach { (level, color) ->
                    FilterChip(
                        selected = level in priorityFilter,
                        onClick = {
                            priorityFilter = if (level in priorityFilter) priorityFilter - level else priorityFilter + level
                        },
                        label = { Text(level, fontSize = 10.sp) },
                        colors = FilterChipDefaults.filterChipColors(
                            selectedContainerColor = color.copy(alpha = 0.2f)
                        )
                    )
                }
            }

            Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.surface)) {
                SelectionContainer {
                    LazyColumn(state = listState) {
                        items(filteredLogs, key = { it.raw.hashCode() }) { entry ->
val color = when (entry.priority) {
        "V" -> Color(StudioUi.color(StudioUi.TextSoft))
        "D" -> Color(StudioUi.color(StudioUi.Blue))
        "I" -> Color(StudioUi.color(StudioUi.Green))
        "W" -> Color(StudioUi.color(StudioUi.Amber))
        "E" -> Color(StudioUi.color(StudioUi.Red))
        "F" -> Color(StudioUi.color(StudioUi.Purple))
        else -> MaterialTheme.colorScheme.onSurface
    }
    Text(
        text = entry.raw,
        color = color,
                                fontSize = 11.sp,
                                fontFamily = FontFamily.Monospace,
                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 1.dp)
                            )
                        }
                    }
                }

                Text(
                    text = "${filteredLogs.size} entries",
                    fontSize = 10.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.align(Alignment.BottomEnd).padding(4.dp)
                )
            }
        }
    }
}
