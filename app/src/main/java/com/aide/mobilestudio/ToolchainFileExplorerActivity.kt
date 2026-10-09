package com.aide.mobilestudio

import android.content.Context
import android.content.Intent
import android.graphics.Color as AndroidColor
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.os.Environment
import android.provider.Settings
import android.system.Os
import android.widget.EditText
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.compose.setContent
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import com.aide.mobilestudio.ZstdInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.tukaani.xz.XZInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.GZIPInputStream
import java.util.zip.ZipInputStream

class ToolchainFileExplorerActivity : AppCompatActivity(), ContextMenuUtils.FileContextMenuHandler {
    companion object {
        private const val EXTRA_KIND = "kind"
        private const val EXTRA_PATH = "path"
        private const val EXTRA_TITLE = "title"
        private const val EXTRA_SELECTABLE = "selectable"
        private const val EXTRA_PICK_EXE = "pick_exe"
        const val RESULT_EXE_PATH = "exe_path"
        private val COLOR_BG = Color(0xFF14181E)
        private val COLOR_PANEL = Color(0xFF1D222B)
        private val COLOR_CARD = Color(0xFF242A35)
        private val COLOR_TEXT = Color(0xFFD9E0EB)
        private val COLOR_DIM = Color(0xFF8791A4)
        private val COLOR_ACCENT = Color(0xFF64A4F5)
        private val COLOR_PURPLE = Color(0xFFB57CE7)
        private val COLOR_GREEN = Color(0xFF46C07D)
        private val COLOR_RED = Color(0xFFF05C5C)

        fun start(context: Context, kind: ToolchainManager.Kind) {
            context.startActivity(Intent(context, ToolchainFileExplorerActivity::class.java).apply {
                putExtra(EXTRA_KIND, kind.name)
                putExtra(EXTRA_SELECTABLE, true)
            })
        }

        fun startExePicker(activity: androidx.appcompat.app.AppCompatActivity, requestCode: Int, startDir: File) {
            activity.startActivityForResult(Intent(activity, ToolchainFileExplorerActivity::class.java).apply {
                putExtra(EXTRA_TITLE, "Select .exe to run")
                putExtra(EXTRA_PATH, startDir.absolutePath)
                putExtra(EXTRA_SELECTABLE, true)
                putExtra(EXTRA_PICK_EXE, true)
            }, requestCode)
        }

        fun startAt(context: Context, title: String, path: File) {
            context.startActivity(Intent(context, ToolchainFileExplorerActivity::class.java).apply {
                putExtra(EXTRA_TITLE, title)
                putExtra(EXTRA_PATH, path.absolutePath)
                putExtra(EXTRA_SELECTABLE, false)
            })
        }
    }

    private lateinit var toolchain: ToolchainManager
    private lateinit var kind: ToolchainManager.Kind
    private var currentDir by mutableStateOf(File("/"))
    private var titleText: String? = null
    private var selectionAllowed: Boolean = true
    private var pickExeMode: Boolean = false
    private var visibleFiles by mutableStateOf<List<File>>(emptyList())
    private var currentPath by mutableStateOf("/")
    private var statusText by mutableStateOf("")
    private var statusOk by mutableStateOf(true)
    private var isBusy by mutableStateOf(false)
    private var clipboardFile: File? = null
    private var clipboardMove: Boolean = false
    private var clipboardText by mutableStateOf<String?>(null)
    private var sortMode by mutableStateOf(0) // 0=name, 1=size, 2=modified
    private var showHidden by mutableStateOf(false)
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        toolchain = ToolchainManager(this)
        titleText = intent.getStringExtra(EXTRA_TITLE)
        selectionAllowed = intent.getBooleanExtra(EXTRA_SELECTABLE, true)
        pickExeMode = intent.getBooleanExtra(EXTRA_PICK_EXE, false)
        kind = runCatching {
            ToolchainManager.Kind.valueOf(intent.getStringExtra(EXTRA_KIND) ?: "")
        }.getOrDefault(ToolchainManager.Kind.SDK)
        currentDir = intent.getStringExtra(EXTRA_PATH)?.let { File(it) }
            ?: toolchain.activeDir(kind)
            ?: File(applicationInfo.dataDir)
        refreshDirectory()
        setContent {
            FileExplorerTheme {
                FileExplorerScreen()
            }
        }
    }

    @Deprecated("Deprecated in Java")
    @android.annotation.SuppressLint("MissingSuperCall")
    override fun onBackPressed() {
        MobileStudioCrashGuard.safe(this, "ToolchainFileExplorerActivity.onBackPressed") {
            val parent = currentDir.parentFile
            if (parent != null && currentDir.absolutePath != "/") {
                currentDir = parent
                refreshDirectory()
            } else {
                MobileStudioCrashGuard.finishSafely(this)
            }
        }
    }

    @Composable
    private fun FileExplorerTheme(content: @Composable () -> Unit) {
        MaterialTheme(
            colorScheme = darkColorScheme(
                primary = COLOR_ACCENT,
                secondary = COLOR_PURPLE,
                background = COLOR_BG,
                surface = COLOR_PANEL,
                onSurface = COLOR_TEXT
            ),
            content = content
        )
    }

    @Composable
    private fun FileExplorerScreen() {
        val pulseTransition = rememberInfiniteTransition(label = "explorer-pulse")
        val pulse by pulseTransition.animateFloat(
            initialValue = 0.35f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(tween(1600), RepeatMode.Reverse),
            label = "pulse-alpha"
        )
        Surface(
            modifier = Modifier.fillMaxSize(),
            color = COLOR_BG
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .background(
                        Brush.verticalGradient(
                            listOf(Color(0xFF141A24), Color(0xFF151922), Color(0xFF0F1218))
                        )
                    )
            ) {
                ExplorerHeader(pulse = pulse)
                PathCard()
                QuickPathBar()
                AnimatedVisibility(visible = clipboardText != null) {
                    ClipboardBanner()
                }
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(
                        start = 10.dp,
                        end = 10.dp,
                        top = 8.dp,
                        bottom = 14.dp
                    )
                ) {
                    item {
                        StatusCard()
                    }
                    if (visibleFiles.isEmpty()) {
                        item {
                            EmptyCard()
                        }
                    } else {
                        items(
                            items = visibleFiles,
                            key = { file -> file.absolutePath }
                        ) { file ->
                            FileRow(file = file)
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun ExplorerHeader(pulse: Float) {
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(10.dp),
            shape = RoundedCornerShape(24.dp),
            colors = CardDefaults.elevatedCardColors(containerColor = COLOR_PANEL)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(Brush.linearGradient(listOf(COLOR_ACCENT, COLOR_PURPLE))),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = ">_",
                        color = Color.Black,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Black,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = titleText ?: "Select ${kind.displayName} Path",
                        color = COLOR_TEXT,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(7.dp)
                                .clip(CircleShape)
                                .background(if (statusOk) COLOR_GREEN.copy(alpha = pulse) else COLOR_RED.copy(alpha = pulse))
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = if (isBusy) "working..." else "${visibleFiles.size} items",
                            color = COLOR_DIM,
                            fontSize = 11.sp,
                            maxLines = 1
                        )
                    }
                }
                if (selectionAllowed && !pickExeMode) {
                    CompactButton("Use") { useCurrentDirectory() }
                    Spacer(modifier = Modifier.width(6.dp))
                }
                CompactButton("More") { showDirectoryMenu() }
                Spacer(modifier = Modifier.width(6.dp))
                CompactButton("Paste") { pasteIntoCurrentDirectory() }
                Spacer(modifier = Modifier.width(6.dp))
                CompactButton("Close") { MobileStudioCrashGuard.finishSafely(this@ToolchainFileExplorerActivity) }
            }
        }
    }

    @Composable
    private fun PathCard() {
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp),
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.elevatedCardColors(containerColor = Color(0xFF161B23))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "cwd",
                    color = COLOR_ACCENT,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(COLOR_ACCENT.copy(alpha = 0.12f))
                        .padding(horizontal = 8.dp, vertical = 5.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = currentPath,
                    color = COLOR_TEXT,
                    fontSize = 11.sp,
                    fontFamily = FontFamily.Monospace,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.horizontalScroll(rememberScrollState())
                )
            }
        }
    }

    @Composable
    private fun QuickPathBar() {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 10.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            QuickChip("App") {
                currentDir = File(applicationInfo.dataDir)
                refreshDirectory()
            }
            QuickChip("/sdcard") {
                if (requestExternalStorageAccess()) {
                    currentDir = Environment.getExternalStorageDirectory()
                    refreshDirectory()
                }
            }
            QuickChip("Tools") {
                currentDir = File(filesDir, "toolchain")
                refreshDirectory()
            }
            QuickChip("Downloads") {
                if (requestExternalStorageAccess()) {
                    currentDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                    refreshDirectory()
                }
            }
            QuickChip("Up") {
                currentDir.parentFile?.let { parent ->
                    currentDir = parent
                    refreshDirectory()
                }
            }
        }
    }

    @Composable
    private fun ClipboardBanner() {
        val text = clipboardText.orEmpty()
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 10.dp),
            shape = RoundedCornerShape(18.dp),
            colors = CardDefaults.elevatedCardColors(containerColor = COLOR_PURPLE.copy(alpha = 0.20f))
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = text,
                    color = COLOR_TEXT,
                    fontSize = 12.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.weight(1f)
                )
                Spacer(modifier = Modifier.width(8.dp))
                CompactButton("Paste") { pasteIntoCurrentDirectory() }
            }
        }
    }

    @Composable
    private fun StatusCard() {
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 8.dp),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.elevatedCardColors(containerColor = COLOR_CARD)
        ) {
            Text(
                text = statusText,
                color = if (statusOk) COLOR_GREEN else COLOR_RED,
                fontSize = 12.sp,
                lineHeight = 16.sp,
                modifier = Modifier.padding(12.dp)
            )
        }
    }

    @Composable
    private fun EmptyCard() {
        ElevatedCard(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(22.dp),
            colors = CardDefaults.elevatedCardColors(containerColor = COLOR_CARD.copy(alpha = 0.75f))
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(22.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("\uD83D\uDCC1", color = COLOR_DIM, fontSize = 28.sp, fontWeight = FontWeight.Bold)
                Text("Empty or permission denied", color = COLOR_DIM, fontSize = 12.sp)
            }
        }
    }

    @OptIn(ExperimentalFoundationApi::class)
    @Composable
    private fun FileRow(file: File) {
        val rowScale by animateFloatAsState(
            targetValue = if (file.isDirectory) 1f else 0.995f,
            animationSpec = tween(180),
            label = "file-row-scale"
        )
        ElevatedCard(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 8.dp)
                .scale(rowScale)
                .combinedClickable(
                    onClick = { openFile(file) },
                    onLongClick = { showFileActions(file) }
                ),
            shape = RoundedCornerShape(20.dp),
            colors = CardDefaults.elevatedCardColors(
                containerColor = if (file.isDirectory) Color(0xFF262C36) else Color(0xFF161C25)
            )
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 58.dp)
                    .padding(horizontal = 10.dp, vertical = 9.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(RoundedCornerShape(14.dp))
                        .background(fileColor(file).copy(alpha = 0.14f))
                        .border(1.dp, fileColor(file).copy(alpha = 0.42f), RoundedCornerShape(14.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = fileIcon(file),
                        color = fileColor(file),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Black,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = file.name.ifBlank { file.absolutePath },
                        color = if (file.isDirectory) COLOR_TEXT else COLOR_DIM,
                        fontSize = 14.sp,
                        fontWeight = if (file.isDirectory) FontWeight.Bold else FontWeight.Medium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Spacer(modifier = Modifier.height(3.dp))
                    Text(
                        text = fileSubtitle(file),
                        color = COLOR_DIM,
                        fontSize = 11.sp,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                OutlinedButton(
                    onClick = { showFileActions(file) },
                    modifier = Modifier.height(36.dp),
                    shape = RoundedCornerShape(12.dp),
                    contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 0.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = COLOR_ACCENT)
                ) {
                    Text("+", fontSize = 16.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }

    @Composable
    private fun CompactButton(text: String, onClick: () -> Unit) {
        Button(
            onClick = {
                MobileStudioCrashGuard.safe(this@ToolchainFileExplorerActivity, "ToolchainFileExplorerActivity.button:$text") {
                    onClick()
                }
            },
            modifier = Modifier.height(36.dp),
            shape = RoundedCornerShape(12.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 10.dp, vertical = 0.dp),
            colors = ButtonDefaults.buttonColors(containerColor = COLOR_ACCENT, contentColor = Color.Black)
        ) {
            Text(text, fontSize = 11.sp, fontWeight = FontWeight.Bold, maxLines = 1)
        }
    }

    @Composable
    private fun QuickChip(text: String, onClick: () -> Unit) {
        OutlinedButton(
            onClick = onClick,
            modifier = Modifier.height(34.dp),
            shape = RoundedCornerShape(999.dp),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 12.dp, vertical = 0.dp),
            colors = ButtonDefaults.outlinedButtonColors(contentColor = COLOR_TEXT)
        ) {
            Text(text, fontSize = 11.sp, maxLines = 1)
        }
    }

    private fun refreshDirectory() {
        currentDir = runCatching { currentDir.canonicalFile }.getOrDefault(currentDir)
        currentPath = currentDir.absolutePath
        val validation = if (selectionAllowed) validatePath(kind, currentDir) else null
        statusOk = validation == null
        statusText = when {
            pickExeMode -> "Tap any .exe file to run it. Long-press for copy, move, rename, delete, or permissions."
            !selectionAllowed -> "Tap a folder to open it. Long-press or tap the menu button for copy, move, delete, rename, permissions, extract, APK install, or JAR run."
            validation == null -> "OK: This folder can be used as ${kind.displayName}."
            else -> "WARN: $validation"
        }
        visibleFiles = currentDir.listFiles()
            ?.filter { showHidden || !it.name.startsWith(".") }
            ?.sortedWith(
                when (sortMode) {
                    1 -> compareBy<File>({ !it.isDirectory }, { -it.length() })
                    2 -> compareBy<File>({ !it.isDirectory }, { -it.lastModified() })
                    else -> compareBy({ !it.isDirectory }, { file -> file.name.lowercase(Locale.US) })
                }
            )
            ?: emptyList()
    }

    private fun requestExternalStorageAccess(): Boolean {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (Environment.isExternalStorageManager()) return true
            AlertDialog.Builder(this)
                .setTitle("Storage Access Required")
                .setMessage("This feature requires access to manage all files. Grant permission in Settings.")
                .setPositiveButton("Open Settings") { _, _ ->
                    startActivity(Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    })
                }
                .setNegativeButton("Cancel", null)
                .show()
            return false
        }
        return true
    }

    private fun fileColor(file: File): Color = when {
        file.isDirectory -> COLOR_ACCENT
        isArchive(file) -> COLOR_PURPLE
        file.canExecute() -> COLOR_GREEN
        else -> COLOR_DIM
    }

    private fun fileIcon(file: File): String = when {
        file.isDirectory -> "DIR"
        isArchive(file) -> "ZIP"
        file.canExecute() -> "EXE"
        else -> "TXT"
    }

    private fun fileSubtitle(file: File): String {
        val size = if (file.isDirectory) {
            "${file.listFiles()?.size ?: 0} items"
        } else {
            formatSize(file.length())
        }
        return "$size | ${permissions(file)} | ${dateFormat.format(Date(file.lastModified()))}"
    }

    private fun showDirectoryMenu() {
        val sortLabel = when (sortMode) { 0 -> "Name"; 1 -> "Size"; else -> "Modified" }
        val items = listOf(
            "New File",
            "New Folder",
            "Compress folder to .zip",
            "Sort: $sortLabel",
            if (showHidden) "Hide hidden files" else "Show hidden files"
        )
        AlertDialog.Builder(this)
            .setTitle(currentDir.name.ifEmpty { "/" })
            .setItems(items.toTypedArray()) { _, which ->
                when (which) {
                    0 -> promptCreateEntry(isFile = true)
                    1 -> promptCreateEntry(isFile = false)
                    2 -> zipCurrentDirectory()
                    3 -> {
                        sortMode = (sortMode + 1) % 3
                        refreshDirectory()
                        toast("Sort: " + when (sortMode) { 0 -> "Name"; 1 -> "Size"; else -> "Modified" })
                    }
                    4 -> {
                        showHidden = !showHidden
                        refreshDirectory()
                    }
                }
            }
            .show()
    }

    private fun promptCreateEntry(isFile: Boolean) {
        val density = resources.displayMetrics.density
        val pad = (16 * density).toInt()
        val input = EditText(this).apply {
            hint = if (isFile) "file name (e.g. Main.kt)" else "folder name"
            isSingleLine = true
        }
        val wrapper = android.widget.FrameLayout(this).apply {
            setPadding(pad, pad / 2, pad, 0)
            addView(
                input,
                android.view.ViewGroup.LayoutParams(
                    android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                    android.view.ViewGroup.LayoutParams.WRAP_CONTENT
                )
            )
        }
        AlertDialog.Builder(this)
            .setTitle(if (isFile) "New File" else "New Folder")
            .setView(wrapper)
            .setPositiveButton("Create") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isEmpty() || name.contains('/')) {
                    toast("Invalid name")
                    return@setPositiveButton
                }
                val target = File(currentDir, name)
                if (target.exists()) {
                    toast("Already exists: $name")
                    return@setPositiveButton
                }
                val ok = if (isFile) runCatching { target.createNewFile() }.getOrDefault(false)
                else runCatching { target.mkdirs() }.getOrDefault(false)
                if (ok) {
                    toast("${if (isFile) "Created " else "Created folder "}$name")
                    refreshDirectory()
                } else {
                    toast("Could not create $name")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun zipCurrentDirectory() = zipEntry(currentDir)

    private fun zipEntry(target: File) {
        val isDir = target.isDirectory
        val outFile = File(
            target.parentFile ?: target,
            (if (isDir) target.name.ifEmpty { "folder" } else target.nameWithoutExtension.ifEmpty { "file" }) + ".zip"
        )
        if (outFile.absolutePath == target.absolutePath) return
        toast("Compressing to ${outFile.name} ...")
        Thread {
            try {
                java.util.zip.ZipOutputStream(java.io.BufferedOutputStream(outFile.outputStream())).use { zos ->
                    if (isDir) {
                        target.walkTopDown().filter { it.isFile && it != outFile }.forEach { f ->
                            val rel = target.name + "/" + f.relativeTo(target).path.replace('\\', '/')
                            zos.putNextEntry(java.util.zip.ZipEntry(rel))
                            f.inputStream().use { inp -> inp.copyTo(zos) }
                            zos.closeEntry()
                        }
                    } else {
                        zos.putNextEntry(java.util.zip.ZipEntry(target.name))
                        target.inputStream().use { inp -> inp.copyTo(zos) }
                        zos.closeEntry()
                    }
                }
                runOnUiThread {
                    toast("Created ${outFile.name} (${formatSize(outFile.length())})")
                    refreshDirectory()
                }
            } catch (e: Exception) {
                runOnUiThread { toast("Zip failed: ${e.message}") }
            }
        }.start()
    }

    private fun showFileActions(file: File) {
        val actions = buildList {
            add("Open")
            if (file.extension.equals("exe", true)) {
                add(if (AppLanguage.isKorean(this@ToolchainFileExplorerActivity)) "box64로 실행" else "Run with box64")
                val hasSource = listOf("main.c", "main.cpp", "main.cc").any { File(file.parentFile, it).isFile }
                if (hasSource) add(if (AppLanguage.isKorean(this@ToolchainFileExplorerActivity)) "box64 빌드 & 실행" else "Build & Run (box64)")
            }
            if (isTextLike(file)) add("Open Text")
            if (file.extension.equals("apk", true)) add("Install APK")
            if (file.extension.equals("jar", true)) add("Run JAR")
            if (selectionAllowed && file.isDirectory) add("Use as ${kind.displayName}")
            if (isArchive(file)) add("Extract here")
            add("Compress to .zip")
            add("Copy Path")
            add("Copy")
            add("Move")
            add("Rename")
            add("Delete")
            add("Permissions")
            add("Set executable")
        }
        AlertDialog.Builder(this)
            .setTitle(file.name)
            .setItems(actions.toTypedArray()) { _, which ->
                when (actions[which]) {
                    "Open" -> openFile(file)
                    "Open Text" -> openTextFile(file)
                    "Run with box64" -> runExeWithBox64(file)
                    "Build & Run (box64)" -> {
                        val src = listOf(File(file.parentFile, "main.c"), File(file.parentFile, "main.cpp"), File(file.parentFile, "main.cc"))
                            .firstOrNull { it.isFile }
                        if (src != null) {
                            toast("Building ${src.name}...")
                            Thread {
                                try {
                                    val env = WindowsExeEnvironment(this@ToolchainFileExplorerActivity)
                                    if (!env.isInstalled) {
                                        runOnUiThread { toast("Wine/box64 not installed") }
                                        return@Thread
                                    }
                                    val built = env.buildWindowsExe(src, onLine = { })
                                    if (built != null) runOnUiThread { runExeWithBox64(built) }
                                    else runOnUiThread { toast("Build failed") }
                                } catch (e: Exception) {
                                    runOnUiThread { toast("Build error: ${e.message}") }
                                }
                            }.start()
                        } else runExeWithBox64(file)
                    }
                    "Install APK" -> installApk(file)
                    "Run JAR" -> runJar(file)
                    "Use as ${kind.displayName}" -> {
                        currentDir = file
                        useCurrentDirectory()
                    }
                    "Extract here" -> extractArchive(file)
                    "Compress to .zip" -> zipEntry(file)
                    "Copy Path" -> {
                        val path = file.absolutePath
                        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as android.content.ClipboardManager
                        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("FilePath", path))
                        toast("Path copied: $path")
                    }
                    "Copy" -> {
                        clipboardFile = file
                        clipboardMove = false
                        clipboardText = "Copy ready: ${file.name}"
                        toast("Copied: ${file.name}. Open a folder and tap Paste.")
                    }
                    "Move" -> {
                        clipboardFile = file
                        clipboardMove = true
                        clipboardText = "Move ready: ${file.name}"
                        toast("Move selected: ${file.name}. Open a folder and tap Paste.")
                    }
                    "Rename" -> renameFile(file)
                    "Delete" -> confirmDelete(file)
                    "Permissions" -> showPermissions(file)
                    "Set executable" -> setExecutable(file)
                }
            }
            .show()
    }

    override fun showFileContextMenu(file: File) {
        showFileActions(file)
    }

    private fun openFile(file: File) {
        if (pickExeMode) {
            if (file.isDirectory) {
                currentDir = file
                refreshDirectory()
            } else if (file.isFile) {
                setResult(RESULT_OK, Intent().putExtra(RESULT_EXE_PATH, file.absolutePath))
                MobileStudioCrashGuard.finishSafely(this)
            }
            return
        }
        when {
            file.isDirectory -> {
                currentDir = file
                refreshDirectory()
            }
            file.extension.equals("apk", true) -> installApk(file)
            file.extension.equals("jar", true) -> runJar(file)
            file.extension.equals("exe", true) -> showFileActions(file)
            isTextLike(file) -> openTextFile(file)
            else -> toast("No built-in opener for .${file.extension.ifBlank { "file" }}")
        }
    }

    private fun isTextLike(file: File): Boolean {
        if (file.isDirectory) return false
        return file.extension.lowercase(Locale.US) in setOf(
            "kt", "kts", "java", "xml", "gradle", "properties", "txt", "md", "json",
            "h", "hpp", "c", "cc", "cpp", "cmake", "pro", "sh", "log"
        )
    }

    private fun openTextFile(file: File) {
        Thread {
            val result = runCatching {
                val bytes = file.inputStream().use { input ->
                    val output = ByteArrayOutputStream()
                    val buffer = ByteArray(8192)
                    var total = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read <= 0) break
                        total += read
                        if (total > 20L * 1024L * 1024L) break
                        output.write(buffer, 0, read)
                    }
                    output.toByteArray()
                }
                String(bytes, Charsets.UTF_8)
            }.getOrElse { "Failed to open: ${it.message}" }
            runOnUiThread {
                val textView = TextView(this).apply {
                    text = result
                    typeface = Typeface.MONOSPACE
                    textSize = 11f
                    setTextColor(AndroidColor.parseColor("#E6EAF2"))
                    setTextIsSelectable(true)
                    setPadding(dp(12), dp(12), dp(12), dp(12))
                }
                AlertDialog.Builder(this)
                    .setTitle(file.name)
                    .setView(ScrollView(this).apply { addView(textView) })
                    .setPositiveButton("Close", null)
                    .show()
            }
        }.start()
    }

    private fun installApk(file: File) {
        val uri = fileProviderUri(file)
        val intent = Intent(Intent.ACTION_VIEW).apply {
            setDataAndType(uri, "application/vnd.android.package-archive")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        runCatching { startActivity(intent) }
            .onFailure { toast("APK install failed: ${it.message}") }
    }

    private fun runExeWithBox64(exeFile: File) {
        val output = StringBuilder("> box64 + wine64: ${exeFile.name}\n")
        val textView = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setTextColor(AndroidColor.parseColor("#E6EAF2"))
            setTextIsSelectable(true)
            setHorizontallyScrolling(true)
            text = output.toString()
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("box64 Console")
            .setView(ScrollView(this).apply { addView(textView) })
            .setNegativeButton("Close", null)
            .setCancelable(false)
            .show()
        Thread {
            try {
                val env = WindowsExeEnvironment(this@ToolchainFileExplorerActivity)
                if (!env.isInstalled) {
                    runOnUiThread {
                        output.append("\n> Wine/box64 environment not installed.\n")
                        textView.text = output.toString()
                        toast("Wine/box64 not installed")
                    }
                    return@Thread
                }
                if (!env.isPrefixBooted) {
                    runOnUiThread { output.append("\n> Bootstrapping Wine prefix (first run)...\n"); textView.text = output.toString() }
                    env.bootstrapPrefix { msg -> runOnUiThread { output.append("$msg\n"); textView.text = output.toString() } }
                }
                val code = env.runWindowsExe(exeFile) { line -> runOnUiThread { output.append("$line\n"); textView.text = output.toString() } }
                runOnUiThread { output.append("\n> exit=$code\n"); textView.text = output.toString() }
            } catch (e: Exception) {
                runOnUiThread { output.append("\n> Run error: ${e.message}\n"); textView.text = output.toString() }
            }
        }.start()
    }

    private fun runJar(file: File) {
        val javaBin = listOf(
            JreGradleInstaller(this, {}, {}).effectiveJavaBin,
            File(filesDir, "toolchain/jdk/lib/jvm/java-17-openjdk/bin/java"),
            File(filesDir, "toolchain/jdk/jdk-21/bin/java"),
            File(filesDir, "toolchain/jdk/shim/java-17-openjdk/bin/java")
        ).firstOrNull { java -> java.exists() } ?: run {
            toast("JDK not found")
            return
        }
        val output = StringBuilder("> ${javaBin.absolutePath} -jar ${file.absolutePath}\n")
        val dialog = AlertDialog.Builder(this)
            .setTitle("JAR Console")
            .setView(ScrollView(this).apply {
                addView(TextView(this@ToolchainFileExplorerActivity).apply {
                    text = output.toString()
                    typeface = Typeface.MONOSPACE
                    textSize = 11f
                    setTextColor(AndroidColor.parseColor("#E6EAF2"))
                    setTextIsSelectable(true)
                    tag = "jar_output"
                    setPadding(dp(12), dp(12), dp(12), dp(12))
                })
            })
            .setPositiveButton("Close", null)
            .show()
        Thread {
            val env = mapOf(
                "JAVA_HOME" to javaBin.parentFile?.parentFile?.absolutePath.orEmpty(),
                "LD_LIBRARY_PATH" to buildString {
                    val home = javaBin.parentFile?.parentFile
                    if (home != null) {
                        append(File(home, "lib/server").absolutePath)
                        append(":")
                        append(File(home, "lib").absolutePath)
                        append(":")
                        append(File(home, "lib/jli").absolutePath)
                        append(":")
                    }
                    append(applicationInfo.nativeLibraryDir)
                }
            )
            val result = ShellRunner.runSh(
                context = this,
                command = "\"${javaBin.absolutePath}\" -jar \"${file.absolutePath}\"",
                cwd = file.parentFile ?: currentDir,
                env = env
            ) { line ->
                output.append(line).append('\n')
                runOnUiThread {
                    findJarOutputView(dialog)?.text = output.toString().takeLast(80_000)
                }
            }
            output.append("\nexit=").append(result.exitCode)
            runOnUiThread {
                findJarOutputView(dialog)?.text = output.toString().takeLast(80_000)
            }
        }.start()
    }

    private fun findJarOutputView(dialog: AlertDialog): TextView? {
        return dialog.window?.decorView?.findViewWithTag("jar_output")
    }

    private fun fileProviderUri(file: File): Uri {
        return FileProvider.getUriForFile(this, "${packageName}.provider", file)
    }

    private fun pasteIntoCurrentDirectory() {
        val source = clipboardFile ?: run {
            toast("Nothing to paste")
            return
        }
        if (!source.exists()) {
            toast("Source no longer exists")
            clipboardFile = null
            clipboardText = null
            return
        }
        val target = uniqueTarget(File(currentDir, source.name))
        isBusy = true
        Thread {
            val error = runCatching {
                if (clipboardMove) {
                    if (!source.renameTo(target)) {
                        copyPath(source, target)
                        source.deleteRecursively()
                    }
                } else {
                    copyPath(source, target)
                }
            }.exceptionOrNull()
            runOnUiThread {
                isBusy = false
                if (error == null) {
                    if (clipboardMove) {
                        clipboardFile = null
                        clipboardText = null
                    }
                    toast(if (clipboardMove) "Moved to ${target.name}" else "Copied to ${target.name}")
                    refreshDirectory()
                } else {
                    toast("Paste failed: ${error.message}")
                }
            }
        }.start()
    }

    private fun renameFile(file: File) {
        val input = EditText(this).apply {
            setText(file.name)
            setSelection(text.length)
        }
        AlertDialog.Builder(this)
            .setTitle("Rename")
            .setView(input)
            .setPositiveButton("Rename") { _, _ ->
                val name = input.text.toString().trim()
                if (name.isBlank() || "/" in name) {
                    toast("Invalid name")
                    return@setPositiveButton
                }
                val target = File(file.parentFile, name)
                if (target.exists()) {
                    toast("Already exists")
                } else if (file.renameTo(target)) {
                    toast("Renamed")
                    refreshDirectory()
                } else {
                    toast("Rename failed")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun confirmDelete(file: File) {
        AlertDialog.Builder(this)
            .setTitle("Delete ${file.name}?")
            .setMessage("This cannot be undone.")
            .setPositiveButton("Delete") { _, _ ->
                isBusy = true
                Thread {
                    val ok = if (file.isDirectory) file.deleteRecursively() else file.delete()
                    runOnUiThread {
                        isBusy = false
                        toast(if (ok) "Deleted" else "Delete failed")
                        refreshDirectory()
                    }
                }.start()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun showPermissions(file: File) {
        val detail = buildString {
            appendLine(file.absolutePath)
            appendLine("Readable: ${file.canRead()}")
            appendLine("Writable: ${file.canWrite()}")
            appendLine("Executable: ${file.canExecute()}")
            appendLine("Unix: ${permissions(file)}")
            appendLine("Size: ${if (file.isDirectory) "${file.listFiles()?.size ?: 0} items" else formatSize(file.length())}")
        }
        AlertDialog.Builder(this)
            .setTitle("Permissions")
            .setMessage(detail)
            .setPositiveButton("OK", null)
            .show()
    }

    private fun setExecutable(file: File) {
        val ok = runCatching {
            file.setReadable(true, false)
            file.setWritable(true, true)
            file.setExecutable(true, false)
            Os.chmod(file.absolutePath, if (file.isDirectory) 0b111_101_101 else 0b111_101_101)
        }.isSuccess
        toast(if (ok) "Executable permission granted" else "Permission change failed")
        refreshDirectory()
    }

    private fun extractArchive(file: File) {
        val dest = File(file.parentFile, file.nameWithoutArchiveExtension()).also { it.mkdirs() }
        isBusy = true
        Thread {
            val error = runCatching {
                when {
                    file.name.endsWith(".zip", ignoreCase = true) -> extractZip(file, dest)
                    file.name.endsWith(".tar.gz", ignoreCase = true) || file.name.endsWith(".tgz", ignoreCase = true) ->
                        GZIPInputStream(file.inputStream().buffered()).use { input -> extractTarStream(input, dest) }
                    file.name.endsWith(".tar.xz", ignoreCase = true) || file.name.endsWith(".txz", ignoreCase = true) ->
                        XZInputStream(file.inputStream().buffered()).use { input -> extractTarStream(input, dest) }
                    file.name.endsWith(".tar.zst", ignoreCase = true) ->
                        java.io.FileInputStream(file).use { fileIn ->
                            ZstdInputStream(fileIn.buffered()).use { input -> extractTarStream(input, dest) }
                        }
                    file.name.endsWith(".tar", ignoreCase = true) ->
                        file.inputStream().buffered().use { input -> extractTarStream(input, dest) }
                    else -> error("Unsupported archive")
                }
            }.exceptionOrNull()
            runOnUiThread {
                isBusy = false
                if (error == null) {
                    toast("Extracted to ${dest.name}")
                    refreshDirectory()
                } else {
                    toast("Extract failed: ${error.message}")
                }
            }
        }.start()
    }

    private fun extractZip(zipFile: File, dest: File) {
        ZipInputStream(zipFile.inputStream().buffered()).use { zip ->
            var entry = zip.nextEntry
            while (entry != null) {
                val out = safeExtractTarget(dest, entry.name)
                if (entry.isDirectory) {
                    out.mkdirs()
                } else {
                    out.parentFile?.mkdirs()
                    out.outputStream().use { zip.copyTo(it) }
                    if (entry.name.substringAfterLast('/').contains('.').not()) out.setExecutable(true, false)
                }
                zip.closeEntry()
                entry = zip.nextEntry
            }
        }
    }

    private fun extractTarStream(input: InputStream, dest: File) {
        TarArchiveInputStream(input).use { tar ->
            var entry = tar.nextTarEntry
            while (entry != null) {
                val out = safeExtractTarget(dest, entry.name)
                if (entry.isDirectory) {
                    out.mkdirs()
                } else if (entry.isSymbolicLink) {
                    // Skip links for safety inside the app explorer.
                } else {
                    out.parentFile?.mkdirs()
                    out.outputStream().use { tar.copyTo(it) }
                    if ((entry.mode and 0b001_001_001) != 0) out.setExecutable(true, false)
                }
                entry = tar.nextTarEntry
            }
        }
    }

    private fun safeExtractTarget(root: File, name: String): File {
        val target = File(root, name).canonicalFile
        val canonicalRoot = root.canonicalFile.absolutePath
        check(target.absolutePath.startsWith(canonicalRoot)) { "Unsafe archive entry: $name" }
        return target
    }

    private fun copyPath(source: File, target: File) {
        if (source.isDirectory) {
            target.mkdirs()
            source.listFiles()?.forEach { child -> copyPath(child, File(target, child.name)) }
        } else {
            target.parentFile?.mkdirs()
            source.copyTo(target, overwrite = false)
            target.setReadable(source.canRead(), false)
            target.setWritable(source.canWrite(), true)
            target.setExecutable(source.canExecute(), false)
        }
    }

    private fun uniqueTarget(file: File): File {
        if (!file.exists()) return file
        val parent = file.parentFile ?: currentDir
        val base = file.nameWithoutExtension
        val ext = file.extension.takeIf { it.isNotBlank() }?.let { ".$it" }.orEmpty()
        var index = 1
        while (true) {
            val candidate = File(parent, "$base copy $index$ext")
            if (!candidate.exists()) return candidate
            index++
        }
    }

    private fun useCurrentDirectory() {
        val warning = validatePath(kind, currentDir)
        if (warning != null) {
            AlertDialog.Builder(this)
                .setTitle("Use anyway?")
                .setMessage("$warning\n\nUse this path anyway?")
                .setPositiveButton("Use") { _, _ -> savePath() }
                .setNegativeButton("Cancel", null)
                .show()
            return
        }
        savePath()
    }

    private fun savePath() {
        if (toolchain.setActivePath(kind, currentDir)) {
            toast("Active ${kind.displayName}: ${currentDir.absolutePath}")
            MobileStudioCrashGuard.finishSafely(this)
        } else {
            toast("Cannot use this path")
        }
    }

    private fun validatePath(kind: ToolchainManager.Kind, dir: File): String? {
        if (!dir.isDirectory) return "Selected path is not a directory."
        return when (kind) {
            ToolchainManager.Kind.JDK -> if (
                File(dir, "bin/java").exists() &&
                (File(dir, "lib/libjli.so").exists() || File(dir, "lib/jli/libjli.so").exists())
            ) null else "JDK folder should contain bin/java and libjli.so."
            ToolchainManager.Kind.GRADLE -> if (File(dir, "lib").listFiles()
                    ?.any { file -> (file.name.startsWith("gradle-launcher") || file.name.startsWith("gradle-gradle-cli-main")) && file.extension == "jar" } == true
            ) null else "Gradle folder should contain lib/gradle-launcher*.jar."
            ToolchainManager.Kind.SDK -> if (
                File(dir, "platforms").isDirectory || File(dir, "android.jar").exists()
            ) null else "SDK folder should contain platforms/android-XX/android.jar."
            ToolchainManager.Kind.NDK -> if (
                File(dir, "toolchains/llvm/prebuilt").isDirectory
            ) null else "NDK folder should contain toolchains/llvm/prebuilt."
            ToolchainManager.Kind.CMAKE -> if (
                File(dir, "bin/cmake").exists() || File(dir, "bin/cmake.exe").exists()
            ) null else "CMake folder should contain bin/cmake."
            ToolchainManager.Kind.PYTHON -> if (
                File(dir, "bin/python3").exists() || File(dir, "bin/python").exists()
            ) null else "Python folder should contain bin/python3 or bin/python."
        }
    }

    private fun isArchive(file: File): Boolean {
        val name = file.name.lowercase(Locale.US)
        return file.isFile && (name.endsWith(".zip") || name.endsWith(".tar") ||
            name.endsWith(".tar.gz") || name.endsWith(".tgz") ||
            name.endsWith(".tar.xz") || name.endsWith(".txz") ||
            name.endsWith(".tar.zst"))
    }

    private fun File.nameWithoutArchiveExtension(): String {
        val lower = name.lowercase(Locale.US)
        val suffix = listOf(".tar.gz", ".tar.xz", ".tar.zst", ".tgz", ".txz", ".zip", ".tar")
            .firstOrNull { value -> lower.endsWith(value) }
            ?: return nameWithoutExtension
        return name.dropLast(suffix.length).ifBlank { nameWithoutExtension }
    }

    private fun permissions(file: File): String {
        return buildString {
            append(if (file.isDirectory) "d" else "-")
            append(if (file.canRead()) "r" else "-")
            append(if (file.canWrite()) "w" else "-")
            append(if (file.canExecute()) "x" else "-")
        }
    }

    private fun formatSize(bytes: Long): String = when {
        bytes < 1024 -> "$bytes B"
        bytes < 1024L * 1024L -> "${bytes / 1024} KB"
        bytes < 1024L * 1024L * 1024L -> "%.1f MB".format(Locale.US, bytes / 1024.0 / 1024.0)
        else -> "%.2f GB".format(Locale.US, bytes / 1024.0 / 1024.0 / 1024.0)
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}
