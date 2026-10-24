package com.aide.mobilestudio.codebrain

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.io.PrintWriter
import java.util.concurrent.*
import java.io.File
import java.util.concurrent.atomic.AtomicInteger

/**
 * CodeBrainService — code-brain (RENA-BR) 프로세스 관리 및 NDJSON 통신.
 *
 * 앱 assets에 번들된 code-brain Python 소스를 내부 저장소에 추출한 뒤,
 * agent_bridge.py를 서브프로세스로 띄우고 줄 단위 JSON(NDJSON)으로 통신한다.
 *
 * 프로토콜:
 *   요청:  {"id": 1, "method": "chat", "params": {"text": "..."}}
 *   응답:  {"id": 1, "ok": true, "result": {...}, "protocol": "1.0"}
 */
class CodeBrainService(private val context: Context) {

    private companion object {
        private const val TAG = "CodeBrainService"
        private const val ASSET_CODE_BRAIN = "code-brain"
        private const val DIR_NAME = "code-brain"
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()

    private var process: Process? = null
    private var writer: PrintWriter? = null
    private var reader: BufferedReader? = null
    private var readerThread: Thread? = null

    private val pendingCalls = ConcurrentHashMap<Int, CompletableFuture<String>>()
    private val nextId = AtomicInteger(1)

    /** code-brain이 내부 저장소에 추출된 경로 */
    private var codeBrainDir: File? = null

    /** 서비스 준비 여부 */
    var isReady: Boolean = false
        private set

    /** 마지막 에러 메시지 */
    var lastError: String? = null
        private set

    /**
     * code-brain을 초기화하고 서브프로세스를 시작한다.
     * 앱 최초 실행 시 assets에서 내부 저장소로 추출한다.
     */
    fun init(callback: (ok: Boolean, error: String?) -> Unit) {
        executor.execute {
            try {
                // 1) assets에서 내부 저장소로 추출
                codeBrainDir = extractCodeBrainFromAssets()
                if (codeBrainDir == null) {
                    mainHandler.post { callback(false, "code-brain 추출 실패") }
                    return@execute
                }

                // 2) Python 실행 파일 찾기
                val pythonPath = findPython()
                if (pythonPath == null) {
                    mainHandler.post { callback(false, "Python을 찾을 수 없습니다. 기기에 Python이 설치되어 있어야 합니다.") }
                    return@execute
                }

                // 3) agent_bridge.py 프로세스 시작
                val bridgeScript = File(codeBrainDir!!, "agent_bridge.py")
                if (!bridgeScript.exists()) {
                    mainHandler.post { callback(false, "agent_bridge.py를 찾을 수 없습니다.") }
                    return@execute
                }

                val pb = ProcessBuilder(
                    pythonPath,
                    bridgeScript.absolutePath
                )
                pb.directory(codeBrainDir!!)
                pb.redirectErrorStream(false)

                process = pb.start()

                writer = PrintWriter(OutputStreamWriter(process!!.outputStream, "UTF-8"), true)
                reader = BufferedReader(java.io.InputStreamReader(process!!.inputStream, "UTF-8"))

                // 4) 응답 읽기 스레드 시작
                readerThread = Thread {
                    try {
                        while (true) {
                            val line = reader!!.readLine()
                            if (line == null) break
                            val trimmed = line.trim()
                            if (trimmed.isEmpty()) continue
                            try {
                                val response = org.json.JSONObject(trimmed)
                                val id = response.optInt("id", -1)
                                val future = pendingCalls.remove(id)
                                if (future != null) {
                                    mainHandler.post {
                                        future.complete(response.toString())
                                    }
                                }
                            } catch (e: Exception) {
                                Log.e(TAG, "JSON 파싱 오류: $trimmed", e)
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "리더 스레드 종료", e)
                    } finally {
                        isReady = false
                    }
                }
                readerThread!!.isDaemon = true
                readerThread!!.start()

                // 5) 핑으로 연결 확인
                val pingResponse = sendRequest("ping", emptyMap(), timeoutMs = 5000)
                if (pingResponse != null) {
                    try {
                        val json = org.json.JSONObject(pingResponse)
                        if (json.optBoolean("ok", false)) {
                            isReady = true
                            lastError = null
                            mainHandler.post { callback(true, null) }
                            return@execute
                        }
                    } catch (e: Exception) { }
                }

                mainHandler.post { callback(false, "code-brain 준비 실패: 핑 응답 없음") }

            } catch (e: Exception) {
                Log.e(TAG, "초기화 오류", e)
                lastError = e.message
                mainHandler.post { callback(false, e.message ?: "알 수 없는 오류") }
            }
        }
    }

    /**
     * assets에서 code-brain을 내부 저장소로 추출한다.
     */
    private fun extractCodeBrainFromAssets(): File? {
        return try {
            val targetDir = File(context.filesDir, DIR_NAME)
            if (!targetDir.exists()) {
                targetDir.mkdirs()
                copyAssetsRecursive(context.assets, ASSET_CODE_BRAIN, targetDir)
            }
            // agent_bridge.py 실행 권한
            val bridgeScript = File(targetDir, "agent_bridge.py")
            bridgeScript.setReadable(true, false)
            bridgeScript.setExecutable(true, false)
            targetDir
        } catch (e: Exception) {
            Log.e(TAG, "추출 오류", e)
            null
        }
    }

    private fun copyAssetsRecursive(assetManager: android.content.res.AssetManager,
                                     assetPath: String,
                                     destDir: File) {
        val files = assetManager.list(assetPath)
        if (files == null || files.isEmpty()) {
            // 파일인 경우
            val fileName = assetPath.substringAfterLast("/")
            val destFile = File(destDir, fileName)
            destFile.parentFile?.mkdirs()
            assetManager.open(assetPath).use { input ->
                java.io.FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            destFile.setReadable(true, false)
        } else {
            // 디렉터리인 경우
            destDir.mkdirs()
            for (file in files) {
                val childAssetPath = if (assetPath.isEmpty()) file else "$assetPath/$file"
                copyAssetsRecursive(assetManager, childAssetPath, destDir)
            }
        }
    }

    /**
     * 기기에서 Python 실행 파일을 찾는다.
     */
    private fun findPython(): String? {
        val candidates = listOf(
            "python3",
            "python",
            "/system/bin/python3",
            "/system/bin/python",
            "/system/xbin/python3",
            "/system/xbin/python",
            "/data/local/tmp/python-android/bin/python3.13",
            "/data/local/tmp/python-android/bin/python3",
        )
        for (c in candidates) {
            try {
                val pb = ProcessBuilder(c, "--version")
                pb.redirectErrorStream(true)
                val p = pb.start()
                val result = p.inputStream.bufferedReader().readText()
                p.waitFor(3, TimeUnit.SECONDS)
                if (result.contains("Python")) {
                    return c
                }
            } catch (e: Exception) {
                // 무시하고 다음 후보 시도
            }
        }
        return null
    }

    /**
     * NDJSON 요청을 전송하고 응답을 기다린다.
     */
    private fun sendRequest(method: String, params: Map<String, Any?>, timeoutMs: Long): String? {
        if (!isReady || writer == null) return null

        val id = nextId.getAndIncrement()
        val future = CompletableFuture<String>()
        pendingCalls[id] = future

        val request = org.json.JSONObject().apply {
            put("id", id)
            put("method", method)
            put("params", org.json.JSONObject(params))
        }

        writer!!.println(request.toString())
        writer!!.flush()

        return try {
            future.get(timeoutMs, TimeUnit.MILLISECONDS)
        } catch (e: Exception) {
            pendingCalls.remove(id)
            null
        }
    }

    /**
     * 채팅 응답 결과
     */
    data class ChatResult(
        val output: String?,
        val error: String?,
        val needsClarification: Boolean = false,
        val missing: List<String> = emptyList(),
        val intent: String? = null,
        val slots: Map<String, Any> = emptyMap(),
    )

    /**
     * 자연어 채팅 (코드 생성, 설명, 질문 등).
     */
    fun chat(text: String, callback: (result: ChatResult) -> Unit) {
        executor.execute {
            val response = sendRequest("chat", mapOf("text" to text), timeoutMs = 30000)
            if (response == null) {
                mainHandler.post { callback(ChatResult(null, "응답 없음 (타임아웃)")) }
                return@execute
            }
            try {
                val json = org.json.JSONObject(response)
                if (json.optBoolean("ok", false)) {
                    val result = json.optJSONObject("result")
                    val needsClarification = result.optBoolean("needs_clarification", false)
                    val missing = if (needsClarification) {
                        val arr = result.optJSONArray("missing")
                        if (arr != null) {
                            (0 until arr.length()).map { arr.getString(it) }.toList()
                        } else emptyList()
                    } else emptyList()
                    val intent = if (needsClarification) result.optString("intent", null) else null
                    val slots = if (needsClarification) result.optJSONObject("slots") else null
                    val slotsMap = if (slots != null) {
                        val keys = slots.keys()
                        mutableMapOf<String, Any>().apply {
                            while (keys.hasNext()) {
                                val key = keys.next()
                                put(key, slots.get(key))
                            }
                        }
                    } else emptyMap()
                    val output = result?.optString("output") ?: result?.toString() ?: response
                    mainHandler.post { callback(ChatResult(output, null, needsClarification, missing, intent, slotsMap)) }
                } else {
                    mainHandler.post { callback(ChatResult(null, json.optString("error", "알 수 없는 오류"))) }
                }
            } catch (e: Exception) {
                mainHandler.post { callback(ChatResult(null, e.message)) }
            }
        }
    }

    /**
     * 코드 생성.
     */
    fun generateCode(prompt: String, language: String = "python", callback: (result: String?, error: String?) -> Unit) {
        executor.execute {
            val response = sendRequest("tool", mapOf(
                "name" to "code.generate",
                "args" to mapOf("prompt" to prompt, "language" to language)
            ), timeoutMs = 30000)
            if (response == null) {
                mainHandler.post { callback(null, "응답 없음") }
                return@execute
            }
            try {
                val json = org.json.JSONObject(response)
                if (json.optBoolean("ok", false)) {
                    val result = json.optJSONObject("result")
                    val output = result?.optString("output") ?: result?.toString() ?: response
                    mainHandler.post { callback(output, null) }
                } else {
                    mainHandler.post { callback(null, json.optString("error", "코드 생성 실패")) }
                }
            } catch (e: Exception) {
                mainHandler.post { callback(null, e.message) }
            }
        }
    }

    /**
     * 코드 검증.
     */
    fun verifyCode(code: String, function: String = "", callback: (result: String?, error: String?) -> Unit) {
        executor.execute {
            val response = sendRequest("tool", mapOf(
                "name" to "code.verify",
                "args" to mapOf("code" to code, "function" to function, "cases" to emptyList<Any>())
            ), timeoutMs = 30000)
            if (response == null) {
                mainHandler.post { callback(null, "응답 없음") }
                return@execute
            }
            try {
                val json = org.json.JSONObject(response)
                if (json.optBoolean("ok", false)) {
                    val result = json.optJSONObject("result")
                    mainHandler.post { callback(result?.toString() ?: response, null) }
                } else {
                    mainHandler.post { callback(null, json.optString("error", "검증 실패")) }
                }
            } catch (e: Exception) {
                mainHandler.post { callback(null, e.message) }
            }
        }
    }

    /**
     * 뇌 상태 / 학습 현황 조회.
     */
    fun status(callback: (result: String?, error: String?) -> Unit) {
        executor.execute {
            val response = sendRequest("status", emptyMap(), timeoutMs = 10000)
            if (response == null) {
                mainHandler.post { callback(null, "응답 없음") }
                return@execute
            }
            try {
                val json = org.json.JSONObject(response)
                if (json.optBoolean("ok", false)) {
                    val result = json.optJSONObject("result")
                    mainHandler.post { callback(result?.toString() ?: response, null) }
                } else {
                    mainHandler.post { callback(null, json.optString("error", "상태 조회 실패")) }
                }
            } catch (e: Exception) {
                mainHandler.post { callback(null, e.message) }
            }
        }
    }

    /**
     * 도구 목록 (manifest) 조회.
     */
    fun manifest(callback: (result: String?, error: String?) -> Unit) {
        executor.execute {
            val response = sendRequest("manifest", emptyMap(), timeoutMs = 10000)
            if (response == null) {
                mainHandler.post { callback(null, "응답 없음") }
                return@execute
            }
            try {
                val json = org.json.JSONObject(response)
                if (json.optBoolean("ok", false)) {
                    val result = json.optJSONObject("result")
                    mainHandler.post { callback(result?.toString() ?: response, null) }
                } else {
                    mainHandler.post { callback(null, json.optString("error", "매니페스트 조회 실패")) }
                }
            } catch (e: Exception) {
                mainHandler.post { callback(null, e.message) }
            }
        }
    }

    /**
     * 서비스 종료 (프로세스Kill).
     */
    fun shutdown() {
        isReady = false
        try {
            sendRequest("shutdown", emptyMap(), timeoutMs = 2000)
        } catch (e: Exception) { }
        try { writer?.close() } catch (e: Exception) { }
        try { reader?.close() } catch (e: Exception) { }
        try { readerThread?.interrupt() } catch (e: Exception) { }
        try { process?.destroyForcibly() } catch (e: Exception) { }
        process = null
        writer = null
        reader = null
        readerThread = null
        pendingCalls.clear()
    }
}
