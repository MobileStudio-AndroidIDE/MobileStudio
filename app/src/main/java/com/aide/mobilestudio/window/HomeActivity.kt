package com.aide.mobilestudio.ui

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.aide.mobilestudio.R
import com.aide.mobilestudio.ui.ScreenRegistry

class HomeActivity : AppCompatActivity() {

    private lateinit var screenContainer: FrameLayout

    companion object {
        fun start(context: Context) {
            context.startActivity(Intent(context, HomeActivity::class.java))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        screenContainer = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        ScreenRegistry.init(screenContainer)

        val root = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(Color.parseColor("#0B0F15"))
        }

        root.addView(screenContainer)

        val grid = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setPadding(24, 24, 24, 24)
        }

        val logo = TextView(this).apply {
            text = "MobileStudio"
            textSize = 32f
            setTextColor(Color.WHITE)
            setPadding(0, 60, 0, 20)
        }
        grid.addView(logo)

        val version = TextView(this).apply {
            text = "v. 26.10.24 Rose"
            textSize = 14f
            setTextColor(Color.parseColor("#94A3B8"))
            setPadding(0, 0, 0, 40)
        }
        grid.addView(version)

        // 타일 행 1
        grid.addView(makeTileRow(
            makeTile("Settings", "\u2699", Color.parseColor("#94A3B8")) {
                ScreenRegistry.show("settings")
            },
            makeTile("AI", "AI", Color.parseColor("#3B82F6")) {
                ScreenRegistry.show("ai")
            },
            makeTile("Finder", "\uD83D\uDD0E", Color.parseColor("#FFFFFF")) {
                ScreenRegistry.show("finder")
            }
        ))

        // 타일 행 2
        grid.addView(makeTileRow(
            makeTile("Shell", ">_", Color.parseColor("#22C55E")) {
                ScreenRegistry.show("shell")
            },
            makeTile("Browser", "\uD83C\uDF10", Color.parseColor("#3B82F6")) {
                ScreenRegistry.show("browser")
            },
            makeTile("Files", "\uD83D\uDCC1", Color.parseColor("#FFFFFF")) {
                ScreenRegistry.show("files")
            }
        ))

        // 타일 행 3
        grid.addView(makeTileRow(
            makeTile("Editor", "\uD83D\uDD8E", Color.parseColor("#569CD6")) {
                ScreenRegistry.show("editor")
            },
            makeTile("Tools", "\uD83D\uDCBB", Color.parseColor("#94A3B8")) {
                ScreenRegistry.show("tools")
            },
            makeTile("Downloader", "\u2B07", Color.parseColor("#F59E0B")) {
                ScreenRegistry.show("downloader")
            }
        ))

        root.addView(grid)
        setContentView(root)

        // 기본 화면 등록
        ScreenRegistry.register(HomeScreen::class.java)
        ScreenRegistry.register("settings") { ctx ->
            object : Screen(ctx) {
                override val id: String = "settings"
                override fun build(): View = TextView(ctx).apply { text = "Settings"; textSize = 18f; setTextColor(Color.WHITE) }
            }
        }
        ScreenRegistry.register("finder") { ctx ->
            object : Screen(ctx) {
                override val id: String = "finder"
                override fun build(): View = TextView(ctx).apply { text = "Finder"; textSize = 18f; setTextColor(Color.WHITE) }
            }
        }
        ScreenRegistry.register("shell") { ctx ->
            ShellScreen(ctx)
        }
        ScreenRegistry.register("browser") { ctx ->
            BrowserScreen(ctx)
        }
        ScreenRegistry.register("files") { ctx ->
            FileExplorerScreen(ctx)
        }
        ScreenRegistry.register("editor") { ctx ->
            EditorScreen(ctx)
        }
        ScreenRegistry.register("tools") { ctx ->
            ToolManagerScreen(ctx)
        }
        ScreenRegistry.register("downloader") { ctx ->
            DownloaderScreen(ctx)
        }

        // 홈 화면 표시
        ScreenRegistry.show("home")
    }

    private fun makeTileRow(vararg tiles: View): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setPadding(0, 12, 0, 12)
            tiles.forEach { addView(it) }
        }
    }

    private fun makeTile(label: String, icon: String, iconColor: Int, onClick: () -> Unit): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { weight = 1f }
            setPadding(12, 12, 12, 12)
            setBackgroundResource(R.drawable.home_tile_bg)
            setOnClickListener { onClick() }

            val iconView = TextView(this@HomeActivity).apply {
                text = icon
                textSize = 32f
                setTextColor(iconColor)
                gravity = Gravity.CENTER
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
            }
            addView(iconView)

            val labelView = TextView(this@HomeActivity).apply {
                text = label
                textSize = 12f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
                )
                setPadding(0, 8, 0, 0)
            }
            addView(labelView)
        }
    }

    // ---- 개별 화면 빌더 (화면 추가/수정은 여기서만 하면 됨) ----

    private fun ShellScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "shell"
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(12, 12, 12, 12)
            }
            c.addView(TextView(ctx).apply {
                text = "Shell"
                textSize = 18f
                setTextColor(Color.WHITE)
                setPadding(0, 0, 0, 12)
            })
            val output = TextView(ctx).apply {
                text = "$ ready\n"
                textSize = 12f
                setTextColor(Color.parseColor("#22C55E"))
                setBackgroundColor(Color.parseColor("#0B0F15"))
                setPadding(12, 12, 12, 12)
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0
                ).apply { weight = 1f }
            }
            c.addView(output)
            val input = EditText(ctx)
            input.hint = "Type a command..."
            input.textSize = 14f
            input.setTextColor(Color.WHITE)
            input.setPadding(12, 10, 12, 10)
            input.setBackgroundResource(R.drawable.home_tile_bg)
            input.layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            input.setImeOptions(android.view.inputmethod.EditorInfo.IME_ACTION_DONE)
            input.setOnEditorActionListener { _, actionId, _ ->
                if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_DONE) {
                    val cmd = input.text.toString().trim()
                    input.setText("")
                    if (cmd.isNotEmpty()) {
                        output.append("$ $cmd\n")
                        execShellCmd(cmd, output)
                    }
                    true
                } else {
                    false
                }
            }
            c.addView(input)
            return c
        }
    }

    private fun execShellCmd(cmd: String, output: TextView) {
        val ctx = this@HomeActivity
        java.util.concurrent.Executors.newSingleThreadExecutor().execute {
            try {
                val pb = if (cmd.equals("clear", ignoreCase = true)) {
                    ProcessBuilder("sh", "-c", "clear")
                } else {
                    ProcessBuilder("sh", "-c", cmd)
                }
                pb.directory(ctx.filesDir)
                pb.redirectErrorStream(true)
                val p = pb.start()
                val result = p.inputStream.bufferedReader().readText()
                p.waitFor(5, java.util.concurrent.TimeUnit.SECONDS)
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    output.append(result.ifEmpty { "(no output)" })
                    output.append("\n")
                }
            } catch (e: Exception) {
                android.os.Handler(android.os.Looper.getMainLooper()).post {
                    output.append("Error: ${e.message}\n")
                }
            }
        }
    }

    private fun BrowserScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "browser"
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(12, 12, 12, 12)
            }
            c.addView(TextView(ctx).apply {
                text = "Web Browser"
                textSize = 18f
                setTextColor(Color.WHITE)
                setPadding(0, 0, 0, 12)
            })
            val urlBar = EditText(ctx).apply {
                hint = "https://example.com"
                textSize = 14f
                setTextColor(Color.WHITE)
                setPadding(12, 10, 12, 10)
                setBackgroundResource(R.drawable.home_tile_bg)
            }
            c.addView(urlBar)
            val webView = android.webkit.WebView(ctx).apply {
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0
                ).apply { weight = 1f }
            }
            c.addView(webView)
            val btnGo = Button(ctx).apply {
                text = "Go"
                textSize = 12f
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.parseColor("#2E8549"))
                setPadding(12, 6, 12, 6)
            }
            btnGo.setOnClickListener {
                val url = urlBar.text.toString().trim()
                if (url.isNotEmpty()) {
                    val finalUrl = if (url.startsWith("http://") || url.startsWith("https://")) url else "https://$url"
                    webView.loadUrl(finalUrl)
                }
            }
            c.addView(btnGo)
            webView.loadUrl("https://github.com/MobileStudio-AndroidIDE/MobileStudio")
            return c
        }
    }

    private fun FileExplorerScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "files"
        private var currentDir = ctx.filesDir
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(12, 12, 12, 12)
            }
            c.addView(TextView(ctx).apply {
                text = "File Explorer"
                textSize = 18f
                setTextColor(Color.WHITE)
                typeface = android.graphics.Typeface.DEFAULT_BOLD
                setPadding(0, 0, 0, 12)
            })
            val pathBar = TextView(ctx).apply {
                text = currentDir.absolutePath
                textSize = 12f
                setTextColor(Color.parseColor("#94A3B8"))
                setPadding(0, 0, 0, 8)
            }
            c.addView(pathBar)
            val fileList = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0
                ).apply { weight = 1f }
            }
            c.addView(fileList)
            val btnBack = Button(ctx).apply {
                text = "Back"
                textSize = 12f
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.parseColor("#2E8549"))
                setPadding(12, 6, 12, 6)
            }
            btnBack.setOnClickListener {
                val parent = currentDir.parentFile
                if (parent != null) {
                    currentDir = parent
                    pathBar.text = currentDir.absolutePath
                    refreshList(ctx, fileList, pathBar)
                }
            }
            c.addView(btnBack)
            val btnRefresh = Button(ctx).apply {
                text = "Refresh"
                textSize = 12f
                setTextColor(Color.WHITE)
                setBackgroundColor(Color.parseColor("#2E8549"))
                setPadding(12, 6, 12, 6)
            }
            btnRefresh.setOnClickListener {
                refreshList(ctx, fileList, pathBar)
            }
            c.addView(btnRefresh)
            refreshList(ctx, fileList, pathBar)
            return c
        }

        private fun refreshList(ctx: Context, container: LinearLayout, pathBar: TextView) {
            container.removeAllViews()
            val dir = currentDir
            if (!dir.exists() || !dir.isDirectory) {
                container.addView(TextView(ctx).apply {
                    text = "Directory not found: $currentDir"
                    textSize = 14f
                    setTextColor(Color.parseColor("#94A3B8"))
                    setPadding(8, 4, 8, 4)
                })
                return
            }
            pathBar.text = dir.absolutePath
            val files = dir.listFiles()
            if (files == null || files.isEmpty()) {
                container.addView(TextView(ctx).apply {
                    text = "(empty)"
                    textSize = 14f
                    setTextColor(Color.parseColor("#94A3B8"))
                    setPadding(8, 4, 8, 4)
                })
                return
            }
            val sorted = files.sortedWith(compareBy({ !it.isDirectory }, { it.name.lowercase() }))
            for (f in sorted) {
                val row = LinearLayout(ctx).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(4, 4, 4, 4)
                }
                val icon = TextView(ctx).apply {
                    text = if (f.isDirectory) "\uD83D\uDCC2" else "\uD83D\uDCC4"
                    textSize = 16f
                    setPadding(4, 0, 8, 0)
                    layoutParams = LinearLayout.LayoutParams(
                        ViewGroup.LayoutParams.WRAP_CONTENT,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    )
                }
                row.addView(icon)
                val name = TextView(ctx).apply {
                    text = f.name
                    textSize = 14f
                    setTextColor(Color.WHITE)
                    layoutParams = LinearLayout.LayoutParams(
                        0,
                        ViewGroup.LayoutParams.WRAP_CONTENT
                    ).apply { weight = 1f }
                }
                row.addView(name)
                val size = TextView(ctx).apply {
                    text = if (f.isDirectory) "" else "${f.length()} bytes"
                    textSize = 12f
                    setTextColor(Color.parseColor("#94A3B8"))
                    setPadding(0, 0, 8, 0)
                }
                row.addView(size)
                row.setOnClickListener {
                    if (f.isDirectory) {
                        currentDir = f
                        refreshList(ctx, container, pathBar)
                    } else {
                        android.widget.Toast.makeText(ctx, "Opened: ${f.name}", android.widget.Toast.LENGTH_SHORT).show()
                    }
                }
                container.addView(row)
            }
        }
    }

    private fun EditorScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "editor"
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(12, 12, 12, 12)
            }
            c.addView(TextView(ctx).apply {
                text = "Editor"
                textSize = 18f
                setTextColor(Color.WHITE)
                setPadding(0, 0, 0, 12)
            })
            val editor = EditText(ctx).apply {
                hint = "fun main() {\n    println(\"Hello\")\n}"
                textSize = 13f
                setTextColor(Color.WHITE)
                setPadding(8, 8, 8, 8)
                setBackgroundColor(Color.parseColor("#0B0F15"))
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    0
                ).apply { weight = 1f }
            }
            c.addView(editor)
            return c
        }
    }

    private fun ToolManagerScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "tools"
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(20, 20, 20, 20)
            }
            c.addView(TextView(ctx).apply {
                text = "Tool Manager"
                textSize = 18f
                setTextColor(Color.WHITE)
                setPadding(0, 0, 0, 16)
            })
            c.addView(TextView(ctx).apply {
                text = "JDK / NDK versions"
                textSize = 14f
                setTextColor(Color.parseColor("#94A3B8"))
                setPadding(0, 0, 0, 8)
            })
            return c
        }
    }

    private fun DownloaderScreen(ctx: Context): Screen = object : Screen(ctx) {
        override val id: String = "downloader"
        override fun build(): View {
            val c = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(20, 20, 20, 20)
            }
            c.addView(TextView(ctx).apply {
                text = "Downloader"
                textSize = 18f
                setTextColor(Color.WHITE)
                setPadding(0, 0, 0, 16)
            })
            c.addView(TextView(ctx).apply {
                text = "Downloads will appear here."
                textSize = 14f
                setTextColor(Color.parseColor("#94A3B8"))
            })
            return c
        }
    }
}
