package com.aide.mobilestudio

import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.graphics.toColorInt
import java.io.File

class BuildTaskDialog(private val activity: AppCompatActivity) {

    private val handler = Handler(Looper.getMainLooper())

    private var selectedVariant: String = "debug"
    private var selectedFlavor: String = ""

    fun show(
        currentTask: String? = null,
        currentFlag: String? = null,
        isBuildRunning: Boolean = false,
        gradleSyncNeeded: Boolean = false,
        variants: List<Pair<String, String>> = emptyList(), // (variantName, flavor) pairs
        onRun: (task: String, flag: String) -> Unit,
        onRunBackground: (task: String, flag: String) -> Unit,
        onStop: () -> Unit
    ) {
        val isKorean = AppLanguage.isKorean(activity)
        selectedVariant = currentTask?.let {
            when { it.contains("Release") -> "release"; else -> "debug" }
        } ?: "debug"

        val root = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 24, 48, 16)
            setBackgroundColor(StudioUi.color(StudioUi.Card))
        }

        // ???? Quick Actions ????????????????????????????????????????????????????????????????????????????????
        root.addView(sectionLabel(if (isKorean) "빠른 빌드" else "Quick Build"))

        val quickRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val quickTasks = listOf(
            "\uD83D\uDE80 Debug" to "assembleDebug",
            "\uD83D\uDD27 Release" to "assembleRelease",
            "\uD83D\uDCE6 Bundle" to "bundleDebug",
            "\u2699\uFE0F Build" to "build"
        )
        quickTasks.forEachIndexed { index, (label, task) ->
            val btn = Button(activity).apply {
                text = label
                textSize = 11f
                setTextColor(StudioUi.color(if (task == currentTask) StudioUi.Text else StudioUi.TextSoft))
                background = StudioUi.rounded(
                    activity,
                    fill = if (task == currentTask) StudioUi.Surface2 else StudioUi.ButtonNormal,
                    stroke = if (task == currentTask) StudioUi.Blue else StudioUi.Border,
                    radiusDp = 6
                )
                setPadding(dp(12), dp(8), dp(12), dp(8))
                minimumHeight = 0
                layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f).apply {
                    marginEnd = if (index < quickTasks.size - 1) dp(4) else 0
                }
                setOnClickListener {
                    val flag = getSelectedFlag()
                    onRun(task, flag)
                }
            }
            quickRow.addView(btn)
        }
        root.addView(quickRow)

        // ???? Build Variant ??????????????????????????????????????????????????????????????????????????????????
        if (variants.isNotEmpty()) {
            root.addView(sectionLabel(if (isKorean) "빌드 변형" else "Build Variant"))
            val variantGroup = LinearLayout(activity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
            }
            variants.forEach { (variant, flavor) ->
                val btn = Button(activity).apply {
                    text = variant
                    textSize = 11f
                    setTextColor(StudioUi.color(if (variant == selectedVariant && flavor == selectedFlavor) StudioUi.Text else StudioUi.TextSoft))
                    val isSelected = variant == selectedVariant && flavor == selectedFlavor
                    background = StudioUi.rounded(
                        activity,
                        fill = if (isSelected) StudioUi.Surface2 else StudioUi.ButtonNormal,
                        stroke = if (isSelected) StudioUi.Blue else StudioUi.Border,
                        radiusDp = 6
                    )
                    setPadding(dp(10), dp(6), dp(10), dp(6))
                    minimumHeight = 0
                    layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply {
                        marginEnd = dp(4)
                    }
                    setOnClickListener {
                        selectedVariant = variant
                        selectedFlavor = flavor
                        updateVariantSelection(this@apply, variants)
                    }
                }
                variantGroup.addView(btn)
            }
            root.addView(variantGroup)
        }

        // ???? Build Type ??????????????????????????????????????????????????????????????????????????????????????
        root.addView(sectionLabel(if (isKorean) "빌드 타입" else "Build Type"))

        val buildTypeGroup = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val buildTypes = listOf("Debug", "Release", "Clean")
        val buildTypeValues = listOf("debug", "release", "clean")
        val radioGroup = RadioGroup(activity).apply {
            orientation = RadioGroup.HORIZONTAL
            gravity = Gravity.CENTER
        }
        var selectedBuildType = buildTypeValues.indexOf(
            when {
                currentTask?.contains("Release") == true -> "release"
                currentTask?.contains("Clean") == true -> "clean"
                else -> "debug"
            }
        ).coerceAtLeast(0)

        buildTypes.forEachIndexed { index, label ->
            val rb = RadioButton(activity).apply {
                text = label
                id = index
                isChecked = index == selectedBuildType
                setTextColor(StudioUi.color(StudioUi.Text))
                textSize = 13f
                setPadding(dp(8), dp(4), dp(16), dp(4))
                setOnCheckedChangeListener { _, isChecked ->
                    if (isChecked) selectedBuildType = index
                }
            }
            radioGroup.addView(rb)
        }
        root.addView(radioGroup)

        // ???? Task Type ????????????????????????????????????????????????????????????????????????????????????????
        root.addView(sectionLabel(if (isKorean) "작업 유형" else "Task Type"))

        val taskGroup = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val taskTypes = listOf("APK", "AAB (Bundle)", "Full Build", "JAR")
        val taskValues = listOf("apk", "bundle", "build", "jar")
        val taskRadioGroup = RadioGroup(activity).apply {
            orientation = RadioGroup.HORIZONTAL
            gravity = Gravity.CENTER
        }
        var selectedTaskType = taskValues.indexOf(
            when {
                currentTask?.contains("bundle") == true -> "bundle"
                currentTask == "build" -> "build"
                currentTask == "jar" -> "jar"
                else -> "apk"
            }
        ).coerceAtLeast(0)

        taskTypes.forEachIndexed { index, label ->
            val rb = RadioButton(activity).apply {
                text = label
                id = index + 100
                isChecked = index == selectedTaskType
                setTextColor(StudioUi.color(StudioUi.Text))
                textSize = 13f
                setPadding(dp(8), dp(4), dp(16), dp(4))
                setOnCheckedChangeListener { _, isChecked ->
                    if (isChecked) selectedTaskType = index
                }
            }
            taskRadioGroup.addView(rb)
        }
        root.addView(taskRadioGroup)

        // ???? Gradle Flags ??????????????????????????????????????????????????????????????????????????????????
        root.addView(sectionLabel("Gradle Flags"))

        val flagRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        val flags = listOf("none", "--info", "--stacktrace", "--debug")
        val flagLabels = listOf("None", "Info", "Stack", "Debug")
        val flagRadioGroup = RadioGroup(activity).apply {
            orientation = RadioGroup.HORIZONTAL
            gravity = Gravity.CENTER
        }
        var selectedFlagIndex = flags.indexOf(currentFlag).coerceAtLeast(0)

        flagLabels.forEachIndexed { index, label ->
            val rb = RadioButton(activity).apply {
                text = label
                id = index + 200
                isChecked = index == selectedFlagIndex
                setTextColor(StudioUi.color(StudioUi.Text))
                textSize = 13f
                setPadding(dp(8), dp(4), dp(16), dp(4))
                setOnCheckedChangeListener { _, isChecked ->
                    if (isChecked) selectedFlagIndex = index
                }
            }
            flagRadioGroup.addView(rb)
        }
        root.addView(flagRadioGroup)

        // ???? Preview Command ????????????????????????????????????????????????????????????????????????????
        val previewText = TextView(activity).apply {
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setTextColor(StudioUi.color(StudioUi.TextSoft))
            background = StudioUi.rounded(activity, fill = StudioUi.Ink, stroke = StudioUi.Border, radiusDp = 6)
            setPadding(dp(10), dp(6), dp(10), dp(6))
            text = ""
        }
        root.addView(previewText)

        // ???? Background Status ????????????????????????????????????????????????????????????????????????
        val bgStatus = GradleBuildService.status(activity)
        if (bgStatus.isNotBlank()) {
            root.addView(TextView(activity).apply {
                text = "\u23F3 Background: $bgStatus"
                textSize = 12f
                setTextColor("#46C07D".toColorInt())
                setPadding(0, dp(8), 0, 0)
            })
        }

        // ???? Build Status ??????????????????????????????????????????????????????????????????????????????????
        if (isBuildRunning) {
            root.addView(TextView(activity).apply {
                text = "\u23F1 Build running..."
                textSize = 12f
                setTextColor("#E3B963".toColorInt())
                setPadding(0, dp(8), 0, 0)
            })
        }

        val dialog = AlertDialog.Builder(activity)
            .setTitle("\uD83D\uDEE0\uFE0F " + (if (isKorean) "빌드" else "Build"))
            .setView(ScrollView(activity).apply {
                addView(root)
                setBackgroundColor(Color.parseColor("#252C35"))
            })
            .setPositiveButton(if (isKorean) "빌드 시작" else "Build", null)
            .setNegativeButton(if (isKorean) "백그라운드" else "Background", null)
            .setNeutralButton(if (isKorean) "중지" else "Stop", null)
            .create()

        dialog.setOnShowListener {
            dialog.window?.setBackgroundDrawable(
                GradientDrawable().apply {
                    setColor(Color.parseColor("#252C35"))
                    cornerRadius = 16f
                }
            )
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setTextColor("#4F8EF7".toColorInt())
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE)?.setTextColor("#46C07D".toColorInt())
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL)?.setTextColor("#F05C5C".toColorInt())

            // Update preview on radio change
            val updatePreview = {
                val task = composeTask(selectedBuildType, selectedTaskType)
                val flag = flags[selectedFlagIndex]
                val cmd = "./gradlew $task${if (flag != "none") " $flag" else ""}"
                previewText.text = cmd
            }
            updatePreview()

            radioGroup.setOnCheckedChangeListener { _, _ -> updatePreview() }
            taskRadioGroup.setOnCheckedChangeListener { _, _ -> updatePreview() }
            flagRadioGroup.setOnCheckedChangeListener { _, _ -> updatePreview() }

            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setOnClickListener {
                if (isBuildRunning) {
                    Toast.makeText(activity,
                        if (isKorean) "빌드가 이미 실행 중입니다" else "Build already running",
                        Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                val task = composeTask(selectedBuildType, selectedTaskType)
                val flag = flags[selectedFlagIndex]
                dialog.dismiss()
                onRun(task, flag)
            }

            dialog.getButton(AlertDialog.BUTTON_NEGATIVE)?.setOnClickListener {
                val task = composeTask(selectedBuildType, selectedTaskType)
                val flag = flags[selectedFlagIndex]
                dialog.dismiss()
                onRunBackground(task, flag)
            }

            dialog.getButton(AlertDialog.BUTTON_NEUTRAL)?.setOnClickListener {
                dialog.dismiss()
                onStop()
            }
        }

        dialog.show()
    }

    private fun composeTask(buildTypeIndex: Int, taskTypeIndex: Int): String {
        val buildType = if (buildTypeIndex == 0) "Debug" else if (buildTypeIndex == 1) "Release" else ""
        val flavorPrefix = selectedFlavor.replaceFirstChar { it.uppercase() }

        return when (taskTypeIndex) {
            0 -> "assemble${flavorPrefix}${buildType}"
            1 -> "bundle${flavorPrefix}${buildType}"
            2 -> "build"
            3 -> "jar"
            else -> "build"
        }
    }

    private fun getSelectedFlag(): String = "none"

    private fun updateVariantSelection(selectedBtn: Button, allVariants: List<Pair<String, String>>) {
        val parent = selectedBtn.parent as? LinearLayout ?: return
        for (i in 0 until parent.childCount) {
            val child = parent.getChildAt(i) as? Button ?: continue
            child.setBackgroundColor(
                Color.parseColor("#3B4252")
            )
        }
        selectedBtn.setBackgroundColor(Color.parseColor("#61AFEF"))
    }

    private fun sectionLabel(text: String): TextView {
        return TextView(activity).apply {
            this.text = text
            textSize = 11f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor("#C3CCD8".toColorInt())
            setPadding(0, dp(14), 0, dp(6))
        }
    }

    private fun dp(v: Int): Int = (v * activity.resources.displayMetrics.density).toInt()
}
