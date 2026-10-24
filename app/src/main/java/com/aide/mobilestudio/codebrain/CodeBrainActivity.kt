package com.aide.mobilestudio.codebrain

import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.aide.mobilestudio.R
import java.util.concurrent.atomic.AtomicBoolean

/**
 * CodeBrainActivity — 코드 브레인 (RENA-BR) 채팅 UI.
 *
 * Python 기반 로컬 코딩 AI와 자연어로 대화한다.
 * - 코드 생성: "두 숫자를 더하는 함수 만들어줘"
 * - 코드 설명: "이 코드를 설명해줘"
 * - 코드 검증: 생성된 코드를 테스트 케이스로 검증
 * - 학습: 대화 내용이 brain_data에 자동 저장되어 이후 대화에 반영
 *
 * LLM/외부 API를 사용하지 않으며, 모든 연산은 기기 내부에서 CPU로 수행된다.
 */
class CodeBrainActivity : AppCompatActivity() {

    private lateinit var service: CodeBrainService
    private lateinit var scrollView: ScrollView
    private lateinit var chatContainer: LinearLayout
    private lateinit var input: EditText
    private lateinit var sendButton: Button
    private lateinit var statusText: TextView
    private val initializing = AtomicBoolean(false)
    private val mainHandler = android.os.Handler(android.os.Looper.getMainLooper())

    companion object {
        fun start(context: Context) {
            context.startActivity(Intent(context, CodeBrainActivity::class.java))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#0B0F15"))
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        // 헤더
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.parseColor("#12161F"))
            setPadding(20, 20, 20, 16)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val title = TextView(this).apply {
            text = "CodeBrain (RENA-BR)"
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            setTextColor(Color.WHITE)
        }
        header.addView(title)

        val subtitle = TextView(this).apply {
            text = "로컬 코딩 AI — LLM 없음, CPU 전용, 학습 가능"
            textSize = 12f
            setTextColor(Color.parseColor("#8B949E"))
            setPadding(0, 4, 0, 0)
        }
        header.addView(subtitle)

        root.addView(header)

        // 상태 바
        statusText = TextView(this).apply {
            text = "초기화 중..."
            textSize = 11f
            setTextColor(Color.parseColor("#58A6FF"))
            setPadding(20, 4, 20, 8)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }
        root.addView(statusText)

        // 채팅 영역
        scrollView = ScrollView(this).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0
            ).apply { weight = 1f }
        }
        scrollView.isFillViewport = true

        chatContainer = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 16, 20, 16)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }
        scrollView.addView(chatContainer)
        root.addView(scrollView)

        // 입력 영역
        val inputContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#12161F"))
            setPadding(12, 12, 12, 12)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        input = EditText(this).apply {
            hint = "코드를 요청하세요 (예: 두 숫자를 더하는 함수 만들어줘)"
            textSize = 14f
            setTextColor(Color.WHITE)
            setHintTextColor(Color.parseColor("#8B949E"))
            setBackgroundResource(R.drawable.codebrain_input_bg)
            setPadding(16, 14, 16, 14)
            layoutParams = LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { weight = 1f }
            maxLines = 5
            minLines = 1
            isVerticalScrollBarEnabled = true
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) {
                    onSend()
                    true
                } else {
                    false
                }
            }
        }
        inputContainer.addView(input)

        sendButton = Button(this).apply {
            text = "전송"
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            setBackgroundColor(Color.parseColor("#2E8549"))
            setPadding(20, 14, 20, 14)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setOnClickListener { onSend() }
        }
        inputContainer.addView(sendButton)

        root.addView(inputContainer)

        // 하단 버튼들
        val bottomBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setBackgroundColor(Color.parseColor("#12161F"))
            setPadding(16, 8, 16, 8)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            gravity = Gravity.CENTER_HORIZONTAL
        }

        val btnStatus = makeBottomButton("상태") {
            service.status { result, error ->
                if (error != null) {
                    Toast.makeText(this@CodeBrainActivity, error, Toast.LENGTH_SHORT).show()
                } else {
                    showInfoDialog("뇌 상태", result ?: "정보 없음")
                }
            }
        }
        bottomBar.addView(btnStatus)

        val btnManifest = makeBottomButton("도구 목록") {
            service.manifest { result, error ->
                if (error != null) {
                    Toast.makeText(this@CodeBrainActivity, error, Toast.LENGTH_SHORT).show()
                } else {
                    showInfoDialog("사용 가능 도구", result ?: "정보 없음")
                }
            }
        }
        bottomBar.addView(btnManifest)

        val btnClear = makeBottomButton("대화 초기화") {
            chatContainer.removeAllViews()
            addBotMessage("대화가 초기화되었습니다. 새로운 대화를 시작할 수 있습니다.")
        }
        bottomBar.addView(btnClear)

        root.addView(bottomBar)

        setContentView(root)

        // 입력 텍스트 변경 시 버튼 상태 업데이트
        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                sendButton.isEnabled = s?.isNotEmpty() == true
            }
            override fun afterTextChanged(s: Editable?) {}
        })

        sendButton.isEnabled = false

        // CodeBrainService 초기화
        service = CodeBrainService(this)
        initializing.set(true)
        statusText.text = "CodeBrain 초기화 중 (Python 프로세스 시작)..."
        service.init { ok, error ->
            initializing.set(false)
            if (ok) {
                statusText.text = "준비 완료 — 코드를 요청하세요"
                statusText.setTextColor(Color.parseColor("#3FB950"))
                addBotMessage("안녕하세요! CodeBrain입니다.\n\n저는 LLM이나 외부 API를 사용하지 않는 로컬 코딩 AI입니다.\n아래에서 코드를 요청해 보세요:\n\n• \"두 숫자를 더하는 함수 만들어줘\"\n• \"팩토리얼 함수를 작성해줘\"\n• \"이 코드의 버그를 찾아줘\"\n• \"코드를 설명해줘\"\n\n학습 기능이 활성화되어 있어, 대화 내용이 자동으로 저장됩니다.")
            } else {
                statusText.text = "초기화 실패: $error"
                statusText.setTextColor(Color.parseColor("#F85149"))
                addBotMessage("초기화 실패: $error\n\n해결 방법:\n1. 기기에 Python 3이 설치되어 있어야 합니다.\n2. code-brain Python 소스가 앱 assets에 번들되어 있어야 합니다.\n\n설치 후 다시 시도해 주세요.")
            }
        }
    }

    private fun onSend() {
        val text = input.text.toString().trim()
        if (text.isEmpty() || initializing.get()) return

        input.setText("")
        sendButton.isEnabled = false

        addUserMessage(text)

        statusText.text = "처리 중..."
        service.chat(text) { result, error ->
            mainHandler.post {
                if (error != null) {
                    addBotMessage("오류: $error")
                    statusText.text = "오류 발생"
                    statusText.setTextColor(Color.parseColor("#F85149"))
                } else {
                    addBotMessage(result ?: "응답 없음")
                    statusText.text = "준비 완료"
                    statusText.setTextColor(Color.parseColor("#3FB950"))
                }
                scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
            }
        }
    }

    private fun addUserMessage(text: String) {
        val bubble = createBubble(text, isUser = true)
        chatContainer.addView(bubble)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun addBotMessage(text: String) {
        val bubble = createBubble(text, isUser = false)
        chatContainer.addView(bubble)
        scrollView.post { scrollView.fullScroll(View.FOCUS_DOWN) }
    }

    private fun createBubble(text: String, isUser: Boolean): View {
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setPadding(0, 0, 0, 16)
            gravity = if (isUser) Gravity.END else Gravity.START
        }

        val bubble = TextView(this).apply {
            this.text = text
            textSize = 14f
            setTextColor(if (isUser) Color.WHITE else Color.parseColor("#C9D1D9"))
            setBackgroundResource(
                if (isUser) R.drawable.codebrain_bubble_user else R.drawable.codebrain_bubble_bot
            )
            setPadding(16, 12, 16, 12)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            maxWidth = (resources.displayMetrics.widthPixels * 0.85).toInt()
        }
        container.addView(bubble)
        return container
    }

    private fun makeBottomButton(text: String, onClick: () -> Unit): Button {
        return Button(this).apply {
            this.text = text
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setBackgroundColor(Color.parseColor("#262B33"))
            setPadding(16, 8, 16, 8)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
            setOnClickListener { onClick() }
        }
    }

    private fun showInfoDialog(title: String, message: String) {
        AlertDialog.Builder(this).apply {
            setTitle(title)
            setMessage(message)
            setPositiveButton("확인") { _, _ -> }
            setCancelable(true)
        }.create().show()
    }

    override fun onDestroy() {
        service.shutdown()
        super.onDestroy()
    }
}
