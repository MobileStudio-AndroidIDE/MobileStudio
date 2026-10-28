package com.aide.mobilestudio.ui

import android.content.Context
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout

/**
 * 화면 추상 클래스. 새로운 화면을 추가하려면 이 클래스를 상속받아 [build]를 구현하면 됩니다.
 *
 * 예:
 * class MyScreen(ctx: Context) : Screen(ctx) {
 *     override fun build(): View {
 *         return TextView(ctx).apply { text = "My Screen" }
 *     }
 * }
 */
abstract class Screen(context: Context) : FrameLayout(context) {

    /** 화면 고유 ID (화면 전환 시 사용) */
    abstract val id: String

    /** 화면 제목 */
    open val title: String get() = id

    /** 화면 레이아웃 빌드 */
    abstract fun build(): View

    init {
        addView(build())
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
    }
}

/**
 * 화면 레지스트리. 모든 화면을 여기서 등록하고,
 * [show]로 화면을 전환합니다.
 *
 * 사용 예:
 * ScreenRegistry.get(container).register(MyScreen::class)
 * ScreenRegistry.get(container).show("my-screen")
 */
object ScreenRegistry {

    private val screens = mutableMapOf<String, ScreenFactory>()
    private val instances = mutableMapOf<String, Screen>()
    private var container: ViewGroup? = null
    private var currentScreen: String? = null

    private data class ScreenFactory(
        val create: (Context) -> Screen
    )

    /** 루트 컨테이너 설정 (앱 시작 시 1회 호출) */
    fun init(container: ViewGroup) {
        this.container = container
    }

    /** 화면 클래스 등록 */
    fun <T : Screen> register(clazz: Class<T>) {
        val id = clazz.simpleName.lowercase()
        screens[id] = ScreenFactory { ctx ->
            clazz.getDeclaredConstructor(Context::class.java).newInstance(ctx)
        }
    }

    /** 문자열 ID로 직접 등록 (람다로 생성 로직 지정 가능) */
    fun register(id: String, factory: (Context) -> Screen) {
        screens[id] = ScreenFactory(factory)
    }

    /** 등록된 화면 표시 */
    fun show(id: String) {
        val ctx = container?.context ?: return

        // 기존 화면 숨기기
        currentScreen?.let { instances[it]?.visibility = View.GONE }

        // 새 화면 생성 또는 재사용
        val screen = instances.getOrPut(id) {
            screens[id]?.create(ctx) ?: return
        }

        // 컨테이너에 추가 (이미 있으면 순서만 변경)
        if (screen.parent == null) {
            container?.addView(screen, 0)
        } else {
            container?.removeView(screen)
            container?.addView(screen, 0)
        }

        screen.visibility = View.VISIBLE
        screen.requestFocus()
        currentScreen = id
    }

    /** 현재 표시 중인 화면 ID 반환 */
    fun current(): String? = currentScreen

    /** 현재 화면 숨기기 */
    fun hide() {
        currentScreen?.let { instances[it]?.visibility = View.GONE }
        currentScreen = null
    }

    /** 모든 화면 제거 */
    fun clear() {
        instances.values.forEach { it.visibility = View.GONE }
        instances.clear()
        screens.clear()
        currentScreen = null
    }
}
