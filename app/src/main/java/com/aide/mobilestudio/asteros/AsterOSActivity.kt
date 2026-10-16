package com.aide.mobilestudio.asteros

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import com.aide.mobilestudio.R
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AsterOS 기본 화면 (Phase 1-A).
 *
 * 규칙:
 * - 바이너리 선택은 Android 안전한 문서 선택 API(ACTION_OPEN_DOCUMENT)만 사용한다.
 * - 선택한 파일은 캐시로 복사한 뒤 분석하며, 어떤 경우에도 실행하지 않는다.
 * - Mach-O 분석은 백그라운드 스레드에서만 수행한다 (UI 스레드 차단 금지).
 * - 분석 결과는 읽기 전용 정보만 표시한다.
 */
class AsterOSActivity : Activity() {

    private var pickedPath: String? = null
    @Volatile private var analyzing = false

    private lateinit var statusView: TextView
    private lateinit var pathView: TextView
    private lateinit var analyzeButton: Button
    private lateinit var resultView: TextView
    private lateinit var logView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.asteros_activity)

        statusView = findViewById(R.id.asteros_status)
        pathView = findViewById(R.id.asteros_file_path)
        analyzeButton = findViewById(R.id.asteros_analyze_button)
        resultView = findViewById(R.id.asteros_result)
        logView = findViewById(R.id.asteros_log)

        analyzeButton.text = "파일 선택 및 분석"
        analyzeButton.setOnClickListener { onAnalyzeClick() }

        appendLog("AsterOS 준비 완료 (읽기 전용 Mach-O 분석)")
    }

    private fun onAnalyzeClick() {
        if (analyzing) {
            appendLog("분석 진행 중입니다")
            return
        }
        openFilePicker()
    }

    /** 안전한 문서 선택기만 사용한다. 외부 실행기 호출은 하지 않는다. */
    private fun openFilePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }
        try {
            startActivityForResult(intent, REQUEST_PICK)
            appendLog("파일 선택기를 엽니다")
        } catch (e: Exception) {
            statusView.text = "파일 선택기를 열 수 없습니다"
            appendLog("파일 선택기 실패: ${e.javaClass.simpleName}")
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_PICK) return
        if (resultCode != RESULT_OK) {
            appendLog("파일 선택이 취소되었습니다")
            return
        }
        val uri = data?.data
        if (uri == null) {
            appendLog("선택된 파일이 없습니다")
            return
        }
        copyToCache(uri)
    }

    /** content:// URI는 분석기에 경로로 못 넘기므로 캐시로 복사한다. */
    private fun copyToCache(uri: Uri) {
        val target = File(cacheDir, "asteros_picked.bin")
        val stream = try {
            contentResolver.openInputStream(uri)
        } catch (e: Exception) {
            null
        }
        if (stream == null) {
            statusView.text = "파일을 열 수 없습니다"
            appendLog("입력 스트림 없음: $uri")
            return
        }
        val copied = try {
            stream.use { input -> target.outputStream().use { output -> input.copyTo(output) } }
            true
        } catch (e: Exception) {
            statusView.text = "파일 복사 실패"
            appendLog("복사 실패: ${e.javaClass.simpleName} ${e.message}")
            false
        }
        if (!copied) return
        pickedPath = target.absolutePath
        pathView.text = pickedPath
        appendLog("캐시 복사 완료: ${target.length()}byte")
        analyze(target.absolutePath)
    }

    private fun analyze(path: String) {
        analyzing = true
        analyzeButton.isEnabled = false
        statusView.text = "분석 중..."
        resultView.text = "분석 중..."
        appendLog("분석 시작: $path")
        Thread {
            val started = System.currentTimeMillis()
            val result = MachOAnalyzer.analyze(path)
            val elapsed = System.currentTimeMillis() - started
            runOnUiThread {
                analyzing = false
                analyzeButton.isEnabled = true
                render(result, elapsed)
            }
        }.start()
    }

    private fun render(result: MachOResult, elapsedMs: Long) {
        when (result) {
            is MachOResult.Success -> {
                statusView.text = "분석 완료 (${elapsedMs}ms)"
                resultView.text = buildString {
                    appendLine("magic: ${result.magic}")
                    appendLine("cpu: ${result.cputype} (subtype ${result.cpusubtype})")
                    appendLine("filetype: ${result.filetype}")
                    appendLine("ncmds: ${result.ncmds}")
                    appendLine("file size: ${result.fileSize}byte")
                    if (result.architectures.size > 1) {
                        appendLine("fat architectures: ${result.architectures.joinToString()}")
                    }
                    result.uuid?.let { appendLine("uuid: $it") }
                    appendLine("segments (${result.segments.size}):")
                    for (s in result.segments) {
                        appendLine(
                            " - ${s.name} vmaddr=0x${s.vmaddr.toString(16)} vmsize=${s.vmsize}" +
                                " fileoff=${s.fileoff} filesize=${s.filesize} sections=${s.sectionCount}"
                        )
                    }
                    if (result.dylibs.isNotEmpty()) {
                        appendLine("dylibs (${result.dylibs.size}):")
                        for (lib in result.dylibs) appendLine(" - $lib")
                    }
                    if (result.warnings.isNotEmpty()) {
                        appendLine("warnings (${result.warnings.size}):")
                        for (w in result.warnings) appendLine(" ! $w")
                    }
                }
                appendLog(
                    "분석 완료: 세그먼트 ${result.segments.size}, dylib ${result.dylibs.size}, " +
                        "경고 ${result.warnings.size}"
                )
            }
            is MachOResult.Fail -> {
                statusView.text = "분석 실패"
                resultView.text = "분석 실패: ${result.reason}"
                appendLog("실패: ${result.reason}")
            }
            is MachOResult.NotSupported -> {
                statusView.text = "지원하지 않는 파일"
                resultView.text = "지원하지 않는 형식: ${result.reason}"
                appendLog("미지원: ${result.reason}")
            }
        }
    }

    private fun appendLog(line: String) {
        val stamp = SimpleDateFormat("HH:mm:ss", Locale.KOREA).format(Date())
        logView.text = "$stamp $line\n" + logView.text
    }

    companion object {
        private const val REQUEST_PICK = 1001
    }
}
