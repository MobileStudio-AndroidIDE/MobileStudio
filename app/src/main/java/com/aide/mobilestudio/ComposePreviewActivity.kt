package com.aide.mobilestudio

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.*
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.*
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import java.io.File

class ComposePreviewActivity : AppCompatActivity() {
    companion object {
        private const val EXTRA_FILE = "extra_file_path"
        fun start(context: Context, file: File) { context.startActivity(Intent(context, ComposePreviewActivity::class.java).putExtra(EXTRA_FILE, file.absolutePath)) }
    }

    private var sourceFile: File? = null
    private var previews = mutableListOf<PreviewFunc>()

    data class PreviewFunc(val name: String, val startLine: Int, val endLine: Int, val widthDp: Int = 320, val heightDp: Int = 640)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val filePath = intent.getStringExtra(EXTRA_FILE)
        val file = filePath?.let { File(it) }
        if (file == null || !file.exists()) { finish(); return }
        sourceFile = file
        setContentView(buildLayout())
        parsePreviews()
        renderPreviews()
    }

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(Color.parseColor("#1E2227")) }
        root.addView(LinearLayout(this).apply {
            setBackgroundColor(Color.parseColor("#191D25")); setPadding(dp(12), dp(8), dp(12), dp(8)); gravity = Gravity.CENTER_VERTICAL
            addView(ImageButton(this@ComposePreviewActivity).apply { setImageResource(android.R.drawable.ic_menu_close_clear_cancel); setBackgroundColor(Color.TRANSPARENT); setOnClickListener { finish() } })
            addView(TextView(this@ComposePreviewActivity).apply { text = "Compose Preview - ${sourceFile?.name ?: ""}"; setTextColor(Color.WHITE); textSize = 14f; typeface = Typeface.DEFAULT_BOLD; setPadding(dp(8), 0, 0, 0) })
        })
        val scroll = ScrollView(this).apply { layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f); setBackgroundColor(Color.parseColor("#0F1218")) }
        val container = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(12), dp(12), dp(12), dp(12)); tag = "preview_container" }
        scroll.addView(container); root.addView(scroll)
        root.addView(TextView(this).apply { text = "Detecting @Preview functions..."; setTextColor(Color.parseColor("#8B949E")); textSize = 11f; setPadding(dp(12), dp(6), dp(12), dp(6)); setBackgroundColor(Color.parseColor("#191D25")); tag = "status_bar" })
        return root
    }

    @SuppressLint("SetTextI18n")
    private fun parsePreviews() {
        previews.clear()
        val source = runCatching { sourceFile?.readText() }.getOrDefault("") ?: ""
        val pat = java.util.regex.Pattern.compile("""@Preview\s*\(([^)]*)\)\s*\n\s*(?:private\s+)?fun\s+(\w+)""")
        val m = pat.matcher(source)
        while (m.find()) {
            val name = m.group(2) ?: continue
            val startLine = source.substring(0, m.start()).count { it == '\n' } + 1
            previews.add(PreviewFunc(name = name, startLine = startLine, endLine = startLine + 20))
        }
        val pat2 = java.util.regex.Pattern.compile("""@Preview\s*\n\s*(?:private\s+)?fun\s+(\w+)""")
        val m2 = pat2.matcher(source)
        while (m2.find()) {
            val name = m2.group(1) ?: continue
            if (previews.any { it.name == name }) continue
            val startLine = source.substring(0, m2.start()).count { it == '\n' } + 1
            previews.add(PreviewFunc(name = name, startLine = startLine, endLine = startLine + 20))
        }
    }

    @SuppressLint("SetTextI18n")
    private fun renderPreviews() {
        val container = findViewWithTagHelper("preview_container") as? LinearLayout ?: return
        val status = findViewWithTagHelper("status_bar") as? TextView
        container.removeAllViews()
        if (previews.isEmpty()) {
            container.addView(TextView(this).apply { text = "No @Preview functions found.\n\nAdd @Preview annotation to a Composable:\n\n@Composable\n@Preview\nfun MyPreview() {\n    Text(\"Hello\")\n}"; setTextColor(Color.parseColor("#8B949E")); textSize = 13f; setPadding(dp(8), dp(16), dp(8), dp(16)) })
            status?.text = "No @Preview functions found"; return
        }
        status?.text = "Found ${previews.size} @Preview function(s)"
        previews.forEach { preview ->
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                background = GradientDrawable().apply { setColor(Color.parseColor("#191D25")); cornerRadius = dp(12f); setStroke(1, Color.parseColor("#333B47")) }
                setPadding(dp(8), dp(8), dp(8), dp(8))
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT).also { it.setMargins(0, 0, 0, dp(12)) }
            }
            card.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL; setPadding(dp(4), dp(4), dp(4), dp(8))
                addView(TextView(this@ComposePreviewActivity).apply { text = preview.name; setTextColor(Color.WHITE); textSize = 13f; typeface = Typeface.DEFAULT_BOLD })
                addView(Space(this@ComposePreviewActivity).apply { layoutParams = LinearLayout.LayoutParams(0, 1, 1f) })
                addView(TextView(this@ComposePreviewActivity).apply { text = "Line ${preview.startLine}"; setTextColor(Color.parseColor("#8B949E")); textSize = 10f })
            })
            val phoneFrame = FrameLayout(this).apply {
                val w = (preview.widthDp * resources.displayMetrics.density).toInt()
                val h = (preview.heightDp * resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(w.coerceAtMost(resources.displayMetrics.widthPixels - dp(48)), h.coerceAtMost((resources.displayMetrics.heightPixels * 0.6).toInt())).also { it.gravity = Gravity.CENTER_HORIZONTAL }
                background = GradientDrawable().apply { setColor(Color.WHITE); cornerRadius = dp(16f) }
                elevation = dp(4f)
            }
            phoneFrame.addView(ComposePreviewCanvas(this, preview), FrameLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT))
            card.addView(phoneFrame, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).also { it.gravity = Gravity.CENTER_HORIZONTAL })
            val source = runCatching { sourceFile?.readText() }.getOrDefault("") ?: ""
            val lines = source.lines(); val start = (preview.startLine - 1).coerceAtLeast(0); val end = preview.endLine.coerceAtMost(lines.size)
            card.addView(TextView(this).apply { text = lines.subList(start, end).joinToString("\n"); setTextColor(Color.parseColor("#C9D1D9")); typeface = Typeface.MONOSPACE; textSize = 10f; setPadding(dp(8), dp(8), dp(8), dp(8)); setBackgroundColor(Color.parseColor("#0F1218")) })
            container.addView(card)
        }
    }

    inner class ComposePreviewCanvas(context: Context, private val p: PreviewFunc) : View(context) {
        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas); canvas.drawColor(Color.WHITE); val d = resources.displayMetrics.density; val cx = width / 2f; val cy = height / 2f
            canvas.drawText(p.name, cx, cy - 40 * d, Paint().apply { color = Color.parseColor("#1A1A1A"); textSize = 18f * d; isAntiAlias = true; textAlign = Paint.Align.CENTER; typeface = Typeface.DEFAULT_BOLD })
            canvas.drawText("Preview Function", cx, cy, Paint().apply { color = Color.parseColor("#666666"); textSize = 12f * d; isAntiAlias = true; textAlign = Paint.Align.CENTER })
            val btnRect = RectF(cx - 60 * d, cy + 20 * d, cx + 60 * d, cy + 50 * d)
            canvas.drawRoundRect(btnRect, 8f * d, 8f * d, Paint().apply { color = Color.parseColor("#6200EE"); isAntiAlias = true })
            canvas.drawText("Button", btnRect.centerX(), btnRect.centerY() + 4 * d, Paint().apply { color = Color.WHITE; textSize = 12f * d; isAntiAlias = true; textAlign = Paint.Align.CENTER })
            val etRect = RectF(cx - 80 * d, cy + 70 * d, cx + 80 * d, cy + 100 * d)
            canvas.drawRoundRect(etRect, 4f * d, 4f * d, Paint().apply { color = Color.parseColor("#F5F5F5"); isAntiAlias = true })
            canvas.drawRoundRect(etRect, 4f * d, 4f * d, Paint().apply { color = Color.parseColor("#CCCCCC"); style = Paint.Style.STROKE; strokeWidth = 1f })
            canvas.drawText("Text field", etRect.left + 8 * d, etRect.centerY() + 4 * d, Paint().apply { color = Color.parseColor("#999999"); textSize = 11f * d; isAntiAlias = true })
        }
    }

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()
    private fun dp(v: Float): Float = v * resources.displayMetrics.density

    private fun findViewWithTagHelper(tag: String): View? {
        val contentView = findViewById<View>(android.R.id.content) ?: return null
        return contentView.findViewWithTag(tag)
    }
}
