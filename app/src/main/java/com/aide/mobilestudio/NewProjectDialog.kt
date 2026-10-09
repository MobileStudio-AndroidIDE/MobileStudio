package com.aide.mobilestudio

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.TextWatcher
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AlertDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File
import java.net.URL

object NewProjectDialog {

    fun show(
        context: Context,
        generator: ProjectTemplateGenerator,
        onCreated: (projectDir: File, cfg: ProjectTemplateGenerator.ProjectConfig) -> Unit
    ) {
        val root = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(context.dp(24), context.dp(12), context.dp(24), context.dp(8))
            setBackgroundColor(StudioUi.palette.card)
        }

        fun label(text: String) = TextView(context).apply {
            this.text = text
            textSize = 11f
            setTextColor(StudioUi.palette.textSoft)
            setPadding(0, 16, 0, 4)
        }

        fun field(hint: String, default: String = "") = EditText(context).apply {
            this.hint = hint
            setText(default)
            setTextColor(StudioUi.palette.text)
            setHintTextColor(StudioUi.palette.muted)
            textSize = 14f
            background = GradientDrawable().apply {
                setColor(StudioUi.palette.surface2)
                cornerRadius = 10f
                setStroke(1, StudioUi.palette.border)
            }
            setPadding(24, 18, 24, 18)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 4 }
        }

        val isKorean = AppLanguage.isKorean(context)

        root.addView(label(if (isKorean) "앱 이름" else "App name"))
        val etAppName = field(if (isKorean) "예시: MyApp" else "e.g. MyApp")
        root.addView(etAppName)

        val pkgLabel = label(if (isKorean) "패키지 이름" else "Package name")
        root.addView(pkgLabel)
        val etPackage = field("com.example.myapp", "com.example.")
        root.addView(etPackage)

        root.addView(label(if (isKorean) "프로젝트 폴더" else "Project folder"))
        val etProjectRoot = field(
                if (isKorean) "예: /sdcard/MobileStudio" else "e.g. /sdcard/MobileStudio",
            generator.projectsRoot.absolutePath
        )
        root.addView(etProjectRoot)

        val tvSavePath = TextView(context).apply {
            text = "${generator.projectsRoot.absolutePath}/<ProjectName>"
            textSize = 11f
            setTextColor(StudioUi.palette.muted)
            typeface = Typeface.MONOSPACE
            setPadding(0, 4, 0, 12)
        }
        root.addView(tvSavePath)

        etAppName.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                val derived = ProjectTemplateGenerator.ProjectConfig.derivePackage(etAppName.text.toString())
                etPackage.setText(derived)
                tvSavePath.text = "${etProjectRoot.text.toString().trim().ifEmpty { generator.projectsRoot.absolutePath }}/${etAppName.text.toString().trim().ifEmpty { "<ProjectName>" }}"
            }
        })
        etProjectRoot.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, st: Int, c: Int, a: Int) {}
            override fun onTextChanged(s: CharSequence?, st: Int, b: Int, c: Int) {}
            override fun afterTextChanged(s: Editable?) {
                tvSavePath.text = "${s.toString().trim().ifEmpty { generator.projectsRoot.absolutePath }}/${etAppName.text.toString().trim().ifEmpty { "<ProjectName>" }}"
            }
        })

        root.addView(label(if (isKorean) "프로젝트 타입" else "Project type"))

        // Android section (shown by default)
        val androidSection = LinearLayout(context).apply { orientation = LinearLayout.VERTICAL }
        androidSection.addView(label(if (isKorean) "템플릿" else "Template"))
        val androidTemplates = listOf(
            "Empty Activity (Kotlin + XML)",
            "Empty Compose Activity",
            "Empty Activity (Java + XML)",
            "Basic Activity (Material + FAB)",
            "Navigation Drawer Activity",
            "Bottom Navigation Activity",
            "Settings Activity",
            "Native C/C++ (Kotlin + NDK)",
            "JAR App (Java)",
            "Game: Pong (OpenGL ES 3.0)",
            "Game: Snake (OpenGL ES 3.0)",
            "Game: 3D Scene (OpenGL ES 3.0)",
            "Game: Voxel World (OpenGL ES 3.0)",
            "Game: Lua 2D (LuaJ + Canvas)",
            "Game: Vulkan 3D (Vulkan API)",
            "Game: JavaScript 2D (Rhino)",
            "Game: Java 2D (Canvas)",
            "Game: Python 2D (Chaquopy)",
            "Game: GDScript (Canvas)",
            "Game: C# (MonoGame-style)"
        )
        val spinnerTemplate = Spinner(context).apply {
            adapter = ArrayAdapter(context, android.R.layout.simple_spinner_item, androidTemplates)
                .also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 8 }
        }
        androidSection.addView(spinnerTemplate)

        // Build Script Format (KTS vs Groovy)
        androidSection.addView(label(if (isKorean) "빌드 스크립트" else "Build Script"))
        val radioGroupBuildDsl = RadioGroup(context).apply {
            orientation = RadioGroup.HORIZONTAL
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = 8 }
        }
        val rbKts = RadioButton(context).apply {
            text = "build.gradle.kts"
            id = View.generateViewId()
            isChecked = true
            setTextColor(StudioUi.palette.textSoft)
        }
        val rbGroovy = RadioButton(context).apply {
            text = "build.gradle (Groovy)"
            id = View.generateViewId()
            setTextColor(StudioUi.palette.textSoft)
        }
        radioGroupBuildDsl.addView(rbKts)
        radioGroupBuildDsl.addView(rbGroovy)
        androidSection.addView(radioGroupBuildDsl)

        androidSection.addView(label(if (isKorean) "최소 SDK" else "Minimum SDK"))
        val minSdkLabels = listOf(
            "21 (Android 5.0)", "24 (Android 7.0)", "26 (Android 8.0)",
            "28 (Android 9.0)", "29 (Android 10)", "31 (Android 12)", "33 (Android 13)"
        )
        val minSdkValues = listOf(21, 24, 26, 28, 29, 31, 33)
        val spinnerMinSdk = Spinner(context).apply {
            adapter = ArrayAdapter(context, android.R.layout.simple_spinner_item, minSdkLabels)
                .also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(1)
        }
        androidSection.addView(spinnerMinSdk)
        androidSection.addView(label(if (isKorean) "옵션" else "Options"))
        val cbViewBinding = CheckBox(context).apply {
            text = if (isKorean) "View Binding 활성화" else "Enable View Binding"
            isChecked = true
            setTextColor(StudioUi.palette.textSoft)
        }
        androidSection.addView(cbViewBinding)
        root.addView(androidSection)

        // Python section (hidden by default)
        val pythonSection = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }
        pythonSection.addView(label(if (isKorean) "템플릿" else "Template"))
        val tvPythonOnly = TextView(context).apply {
            text = if (isKorean) "Python 스크립트 (터미널 + pip)" else "Python Script (terminal + pip)"
            textSize = 14f
            setTextColor(StudioUi.palette.success)
            setPadding(24, 14, 24, 14)
            background = GradientDrawable().apply {
                setColor(StudioUi.palette.surface2)
                cornerRadius = 10f
                setStroke(1, StudioUi.palette.border)
            }
        }
        pythonSection.addView(tvPythonOnly)
        root.addView(pythonSection)

        // Windows .exe section (hidden by default)
        val windowsSection = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        }
        windowsSection.addView(label(if (isKorean) "템플릿" else "Template"))
        val tvWindowsOnly = TextView(context).apply {
            text = "Windows .exe (C - mingw-w64)"
            textSize = 14f
            setTextColor(StudioUi.palette.warning)
            setPadding(24, 14, 24, 14)
            background = GradientDrawable().apply {
                setColor(StudioUi.palette.surface2)
                cornerRadius = 10f
                setStroke(1, StudioUi.palette.border)
            }
        }
        windowsSection.addView(tvWindowsOnly)
        val tvWindowsNote = TextView(context).apply {
            text = if (isKorean)
                "Tools > Wine & Box64 > Build .exe 로 컴파일, Run .exe 로 실행"
            else "Compile with Tools > Wine & Box64 > Build .exe, run with Run .exe"
            textSize = 11f
            setTextColor(StudioUi.palette.muted)
            setPadding(24, 4, 24, 8)
        }
        windowsSection.addView(tvWindowsNote)
        root.addView(windowsSection)

        val radioGroup = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        var selectedType = 0
        fun applyType(t: Int) {
            selectedType = t
            androidSection.visibility = if (t == 0) View.VISIBLE else View.GONE
            pythonSection.visibility = if (t == 1) View.VISIBLE else View.GONE
            windowsSection.visibility = if (t == 2) View.VISIBLE else View.GONE
            etPackage.visibility = if (t == 0) View.VISIBLE else View.GONE
            for (i in 0 until root.childCount) {
                val child = root.getChildAt(i)
                if (child === pkgLabel) {
                    child.visibility = if (t == 0) View.VISIBLE else View.GONE
                }
            }
        }
        val rbAndroid = RadioButton(context).apply {
            text = if (isKorean) "안드로이드 앱" else "Android App"
            isChecked = true
            setTextColor(StudioUi.palette.text)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { applyType(0) }
        }
        val rbPython = RadioButton(context).apply {
            text = if (isKorean) "Python (Shell)" else "Python (Shell)"
            setTextColor(StudioUi.palette.text)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { applyType(1) }
        }
        radioGroup.addView(rbAndroid)
        radioGroup.addView(rbPython)
        val rbWindows = RadioButton(context).apply {
            text = "Windows .exe"
            setTextColor(StudioUi.palette.text)
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            setOnClickListener { applyType(2) }
        }
        radioGroup.addView(rbWindows)
        root.addView(radioGroup)

        val tvError = TextView(context).apply {
            setTextColor(StudioUi.palette.error)
            textSize = 11f
            setPadding(0, 8, 0, 0)
            visibility = View.GONE
        }
        root.addView(tvError)

        val dialog = AlertDialog.Builder(context)
            .setTitle(if (isKorean) "새 프로젝트" else "New Project")
            .setView(ScrollView(context).apply {
                addView(root)
                setBackgroundColor(StudioUi.palette.card)
            })
            .setPositiveButton(if (isKorean) "만들기" else "Create", null)
            .setNegativeButton(if (isKorean) "취소" else "Cancel", null)
            .create()

        dialog.setOnShowListener {
            dialog.window?.setBackgroundDrawable(
                GradientDrawable().apply {
                    setColor(StudioUi.palette.card)
                    cornerRadius = 16f
                }
            )
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE)?.setTextColor(StudioUi.palette.muted)
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setTextColor(StudioUi.palette.accent)

            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setOnClickListener {
                val appName = etAppName.text.toString().trim()
                val pkgName = etPackage.text.toString().trim()
                val projectRoot = File(etProjectRoot.text.toString().trim().ifEmpty { generator.projectsRoot.absolutePath })

                when {
                    appName.isEmpty() -> {
                        tvError.text = if (isKorean) "앱 이름은 비워둘 수 없습니다." else "App name cannot be empty."
                        tvError.visibility = View.VISIBLE
                        return@setOnClickListener
                    }
                    appName.contains(Regex("[^a-zA-Z0-9_ ]")) -> {
                        tvError.text = if (isKorean) "앱 이름에 잘못된 문자가 포함되어 있습니다." else "App name contains invalid characters."
                        tvError.visibility = View.VISIBLE
                        return@setOnClickListener
                    }
                    selectedType == 0 && !pkgName.matches(Regex("[a-z][a-z0-9]*(\\.[a-z][a-z0-9]*)+")) -> {
                        tvError.text = if (isKorean) "유효하지 않은 패키지 이름." else "Invalid package name."
                        tvError.visibility = View.VISIBLE
                        return@setOnClickListener
                    }
                    File(projectRoot, appName).exists() -> {
                        tvError.text = if (isKorean) "\"$appName\" 은(는) 이미 있습니다. 다른 프로젝트 이름으로 시도해주세요." else "A project named \"$appName\" already exists."
                        tvError.visibility = View.VISIBLE
                        return@setOnClickListener
                    }
                }

                tvError.visibility = View.GONE

                val template = when (selectedType) {
                0 -> {
                    when (spinnerTemplate.selectedItemPosition) {
                        0 -> ProjectTemplateGenerator.Template.EMPTY_ACTIVITY
                        1 -> ProjectTemplateGenerator.Template.EMPTY_COMPOSE_ACTIVITY
                        2 -> ProjectTemplateGenerator.Template.EMPTY_JAVA_ACTIVITY
                        3 -> ProjectTemplateGenerator.Template.BASIC_ACTIVITY
                        4 -> ProjectTemplateGenerator.Template.NAVIGATION_DRAWER
                        5 -> ProjectTemplateGenerator.Template.BOTTOM_NAVIGATION
                        6 -> ProjectTemplateGenerator.Template.SETTINGS_ACTIVITY
                        7 -> ProjectTemplateGenerator.Template.NATIVE_CPP
                        8 -> ProjectTemplateGenerator.Template.JAR_APP
                        9 -> ProjectTemplateGenerator.Template.GAME_2D_PONG
                        10 -> ProjectTemplateGenerator.Template.GAME_2D_SNAKE
                        11 -> ProjectTemplateGenerator.Template.GAME_3D_SCENE
                        12 -> ProjectTemplateGenerator.Template.GAME_VOXEL_WORLD
                        13 -> ProjectTemplateGenerator.Template.GAME_LUA_2D
                        14 -> ProjectTemplateGenerator.Template.GAME_VULKAN
                        15 -> ProjectTemplateGenerator.Template.GAME_JS_2D
                        16 -> ProjectTemplateGenerator.Template.GAME_JAVA_2D
                        17 -> ProjectTemplateGenerator.Template.GAME_PYTHON_2D
                        18 -> ProjectTemplateGenerator.Template.GAME_GDSCRIPT
                        19 -> ProjectTemplateGenerator.Template.GAME_CSHARP_2D
                        else -> ProjectTemplateGenerator.Template.EMPTY_ACTIVITY
                    }
                }
                1 -> ProjectTemplateGenerator.Template.PYTHON_SCRIPT
                else -> ProjectTemplateGenerator.Template.WINDOWS_EXE
            }

                val toolchain = ToolchainManager(context)
                val ndkVersion = toolchain.activeDir(ToolchainManager.Kind.NDK)?.let { ndkDir ->
                    val props = File(ndkDir, "source.properties")
                    if (props.exists()) {
                        props.readText().lines()
                            .firstOrNull { it.startsWith("Pkg.Revision") }
                            ?.substringAfter("=")?.trim()
                    } else null
                } ?: ""
                val cfg = ProjectTemplateGenerator.ProjectConfig(
                    appName        = appName,
                    packageName    = when (selectedType) {
                    0 -> pkgName
                    1 -> "python.$appName"
                    else -> "win.$appName"
                },
                    minSdk         = if (selectedType == 0) minSdkValues[spinnerMinSdk.selectedItemPosition] else 21,
                    template       = template,
                    useViewBinding = selectedType == 0 && cbViewBinding.isChecked,
                    projectRoot    = projectRoot,
                    buildDsl       = if (rbKts.isChecked) ProjectTemplateGenerator.BuildDsl.KTS else ProjectTemplateGenerator.BuildDsl.GROOVY,
                    ndkVersion     = ndkVersion
                )

                if (template == ProjectTemplateGenerator.Template.NATIVE_CPP) {
                    val toolchain = ToolchainManager(context)
                    val hasNdk = toolchain.activeDir(ToolchainManager.Kind.NDK) != null
                    val hasLlvm = AndroidHostedLlvmInstaller.hasUsableCompiler(context)
                    val sdkDir = toolchain.activeDir(ToolchainManager.Kind.SDK)
                    val hasCmake = sdkDir != null && File(sdkDir, "cmake/3.22.1/bin/cmake").exists()
                    val hasNinja = Arm64ToolsDownloader.linkerRunnableToolPath(context, "ninja") != null
                        || (sdkDir != null && File(sdkDir, "cmake/3.22.1/bin/ninja").exists())
                    val missingTools = mutableListOf<String>().apply {
                        if (!hasNdk) add("NDK")
                        if (!hasLlvm) add("Android LLVM (Clang)")
                        if (!hasCmake) add("CMake 3.22.1")
                        if (!hasNinja) add("Ninja")
                    }
                    if (missingTools.isNotEmpty()) {
                        val msg = if (isKorean)
                            "이 프로젝트는 다음 도구가 필요합니다:\n${missingTools.joinToString("\n")}\n\n지금 다운로드 하시겠습니까?"
                        else
                            "This project needs:\n${missingTools.joinToString("\n")}\n\nDownload now?"
                        
                        MaterialAlertDialogBuilder(context)
                            .setTitle(if (isKorean) "빌드 도구가 필요합니다." else "Build Tools Required")
                            .setMessage(msg)
                            .setPositiveButton(if (isKorean) "다운로드" else "Download") { _, _ ->
                                dialog.dismiss()
                                installNdkAndCreate(context, generator, cfg, onCreated, isKorean, hasNdk, hasLlvm, hasCmake, hasNinja)
                            }
                            .setNegativeButton(if (isKorean) "취소" else "Cancel") { _, _ ->
                                Toast.makeText(context,
                                    if (isKorean) "프로젝트 생성이 취소되었습니다."
                                    else "Project creation canceled.",
                                    Toast.LENGTH_LONG).show()
                            }
                            .setCancelable(false)
                            .show()
                        return@setOnClickListener
                    }
                }

                dialog.dismiss()

                Thread {
                    try {
                        val dir = generator.create(cfg)
                        Handler(Looper.getMainLooper()).post {
                            onCreated(dir, cfg)
                            Toast.makeText(
                                context,
                                if (isKorean) "프로젝트가 생성되었습니다: ${StorageHelper.displayPath(context, cfg.appName)}"
                                else "Project created: ${StorageHelper.displayPath(context, cfg.appName)}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    } catch (e: Exception) {
                        Handler(Looper.getMainLooper()).post {
                            val sw = java.io.StringWriter()
                            java.io.PrintWriter(sw).use { pw -> e.printStackTrace(pw) }
                            val stack = sw.toString()
                            val scrollView = android.widget.ScrollView(context)
                            val errorText = android.widget.TextView(context).apply {
                                text = stack
                                textSize = 10f
                                typeface = android.graphics.Typeface.MONOSPACE
                                setTextColor(android.graphics.Color.parseColor("#F05C5C"))
                                setPadding(30, 20, 30, 20)
                            }
                            scrollView.addView(errorText)
                            MaterialAlertDialogBuilder(context)
                                .setTitle(if (isKorean) "프로젝트 생성이 실패하였습니다." else "Project Creation Failed")
                                .setView(scrollView)
                                .setPositiveButton(if (isKorean) "확인" else "OK", null)
                                .show()
                        }
                    }
                }.start()
            }
        }

        dialog.show()
        dialog.window?.setLayout(
            (context.resources.displayMetrics.widthPixels * 0.92).toInt(),
            ViewGroup.LayoutParams.WRAP_CONTENT
        )
    }

    private fun installNdkAndCreate(
        context: Context,
        generator: ProjectTemplateGenerator,
        cfg: ProjectTemplateGenerator.ProjectConfig,
        onCreated: (File, ProjectTemplateGenerator.ProjectConfig) -> Unit,
        isKorean: Boolean,
        hasNdk: Boolean,
        hasLlvm: Boolean,
        hasCmake: Boolean = false,
        hasNinja: Boolean = false
    ) {
        val progress = MaterialAlertDialogBuilder(context)
            .setTitle(if (isKorean) "빌드 도구 다운로드 중..." else "Installing Build Tools...")
            .setMessage(if (isKorean) "준비중..." else "Preparing...")
            .setCancelable(false)
            .create()
        progress.show()

        Thread {
            try {
                val toolchain = ToolchainManager(context)
                val sdkVersion = toolchain.activeVersion(ToolchainManager.Kind.SDK)
                if (sdkVersion == null) {
                    throw RuntimeException(if (isKorean) "SDK가 설치되지 않았습니다. SDK를 먼저 설치해주세요."
                        else "SDK not installed. Install SDK first.")
                }
                val sdkDir = toolchain.activeDir(ToolchainManager.Kind.SDK)!!

                // 1. Download NDK if missing
                if (!hasNdk) {
                    val ndkVersion = ToolchainManager.AVAILABLE_NDK.first()
                    Handler(Looper.getMainLooper()).post {
                        progress.setMessage(if (isKorean) "NDK 다운로드 중..." else "Downloading NDK...")
                    }
                    try {
                        val ndkDir = com.aide.mobilestudio.NdkToolchain.installFromGithub(
                            context, ndkVersion
                        ) { msg ->
                            Handler(Looper.getMainLooper()).post { progress.setMessage(msg) }
                        }
                        toolchain.setActiveVersion(ToolchainManager.Kind.NDK, ndkVersion)
                    } catch (_: Exception) {
                    }
                }

                // 2. Download CMake 3.22.1 if missing
                if (!hasCmake) {
                    Handler(Looper.getMainLooper()).post {
                        progress.setMessage(if (isKorean) "CMake 3.22.1 다운로드 중..." else "Downloading CMake 3.22.1...")
                    }
                    // Primary: user-ported CMake runtime from the MobileStudio GitHub release
                    var cmakeInstalled = false
                    try {
                        CmakeRuntimeManager.initAppContext(context)
                        val latch = java.util.concurrent.CountDownLatch(1)
                        var ok = false
                        var cmakeErr: String? = null
                        CmakeRuntimeManager.get(context).ensureInstalled(
                            onProgress = { read, total ->
                                Handler(Looper.getMainLooper()).post {
                                    val percent = if (total > 0) (read * 100 / total).toInt() else 0
                                    progress.setMessage(if (isKorean) "CMake 다운로드 중... $percent%" else "Downloading CMake... $percent%")
                                }
                            },
                            onLog = { msg -> android.util.Log.d("NDKSetup", msg) },
                            onDone = { success, error -> ok = success; cmakeErr = error; latch.countDown() }
                        )
                        latch.await()
                        cmakeInstalled = ok
                        if (!ok) android.util.Log.w("NDKSetup", "CMake runtime install failed: $cmakeErr")
                    } catch (e: Exception) {
                        android.util.Log.w("NDKSetup", "CMake runtime install error: ${e.message}")
                    }
                    if (cmakeInstalled) {
                        toolchain.setActiveVersion(ToolchainManager.Kind.CMAKE, "3.22.1")
                    } else
                    try {
                        val repositoryXml = URL("https://dl.google.com/android/repository/repository2-1.xml")
                            .openConnection().let { conn ->
                                conn as java.net.HttpURLConnection
                                conn.connectTimeout = 15000; conn.readTimeout = 30000
                                conn.inputStream.bufferedReader().readText()
                            }
                        val cmakePkg = "cmake;3.22.1"
                        val cmakeUrl = resolveSdkPackageUrl(repositoryXml, cmakePkg)
                        val cmakeZip = File(context.cacheDir, "cmake-3.22.1.zip")
                        val cmakeTemp = File(context.cacheDir, "cmake-extract").apply { deleteRecursively(); mkdirs() }
                        try {
                            downloadFileWithProgress(cmakeUrl, cmakeZip, isKorean) { msg ->
                                Handler(Looper.getMainLooper()).post { progress.setMessage(msg) }
                            }
                            Handler(Looper.getMainLooper()).post {
                                progress.setMessage(if (isKorean) "CMake 압축 푸는 중..." else "Extracting CMake...")
                            }
                            java.util.zip.ZipFile(cmakeZip).use { z ->
                                val entries = z.entries()
                                while (entries.hasMoreElements()) {
                                    val entry = entries.nextElement()
                                    val target = File(cmakeTemp, entry.name)
                                    if (entry.isDirectory) target.mkdirs()
                                    else {
                                        target.parentFile?.mkdirs()
                                        z.getInputStream(entry).use { inp -> target.outputStream().use { out -> inp.copyTo(out) } }
                                    }
                                }
                            }
                            val cmakeDest = File(sdkDir, "cmake/3.22.1")
                            val cmakeRoot = cmakeTemp.walkTopDown().firstOrNull {
                                it.isFile && it.name == "cmake" && it.parentFile?.name == "bin"
                            }?.parentFile?.parentFile
                                ?: cmakeTemp.walkTopDown().firstOrNull { it.isFile && it.name == "source.properties" }?.parentFile
                                ?: cmakeTemp
                            cmakeRoot.copyRecursively(cmakeDest, overwrite = true)
                            cmakeDest.walkTopDown().filter { it.parentFile?.name == "bin" || it.name in setOf("cmake", "ninja", "ctest", "cpack") }
                                .forEach { it.setExecutable(true, false) }
                            toolchain.setActiveVersion(ToolchainManager.Kind.CMAKE, "3.22.1")
                        } finally {
                            cmakeZip.delete(); cmakeTemp.deleteRecursively()
                        }
                    } catch (e: Exception) {
                        Handler(Looper.getMainLooper()).post {
                            progress.setMessage(if (isKorean) "CMake 다운로드 실패, 대체 소스 시도 중..." else "CMake download failed, trying alternate source...")
                        }
                        // Fallback: download cmake binary from Arm64ToolsDownloader
                        if (!Arm64ToolsDownloader.allInstalled(context)) {
                            Arm64ToolsDownloader.downloadAll(
                                context,
                                onProgress = { name, cur, total ->
                                    Handler(Looper.getMainLooper()).post {
                                        progress.setMessage(if (isKorean) "ARM64 도구 다운로드 중... ($name $cur/$total)" else "Downloading ARM64 tools... ($name $cur/$total)")
                                    }
                                },
                                onLog = { msg -> android.util.Log.d("NDKSetup", msg) }
                            )
                        }
                        Arm64ToolsDownloader.ensureCmakeLayout(context)
                    }
                }

                // 3. Download Ninja if missing
                if (!hasNinja) {
                    Handler(Looper.getMainLooper()).post {
                        progress.setMessage(if (isKorean) "Ninja 다운로드 중..." else "Downloading Ninja...")
                    }
                    if (!Arm64ToolsDownloader.allInstalled(context)) {
                        Arm64ToolsDownloader.downloadAll(
                            context,
                            onProgress = { name, cur, total ->
                                Handler(Looper.getMainLooper()).post {
                                    progress.setMessage(if (isKorean) "Ninja 다운로드 중... ($name $cur/$total)" else "Downloading Ninja... ($name $cur/$total)")
                                }
                            },
                            onLog = { msg -> android.util.Log.d("NDKSetup", msg) }
                        )
                    }
                    Arm64ToolsDownloader.ensureCmakeLayout(context)
                }

                // 4. Download Android LLVM if missing
                if (!hasLlvm) {
                    Handler(Looper.getMainLooper()).post {
                        progress.setMessage(if (isKorean) "Android LLVM(Clang) 다운로드 중... (~1100MB)"
                            else "Downloading Android LLVM (Clang)... (~1100MB)")
                    }
                    AndroidHostedLlvmInstaller.ensureInstalled(context, emptyList()) { msg, _ ->
                        Handler(Looper.getMainLooper()).post { progress.setMessage(msg) }
                    }
                }

                Handler(Looper.getMainLooper()).post { progress.dismiss() }

                val dir = generator.create(cfg)
                Handler(Looper.getMainLooper()).post {
                    onCreated(dir, cfg)
                    Toast.makeText(
                        context,
                        if (isKorean) "프로젝트 생성됨: ${StorageHelper.displayPath(context, cfg.appName)}"
                        else "Project created: ${StorageHelper.displayPath(context, cfg.appName)}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            } catch (e: Exception) {
                val sw = java.io.StringWriter()
                java.io.PrintWriter(sw).use { pw -> e.printStackTrace(pw) }
                val stack = sw.toString()
                Handler(Looper.getMainLooper()).post {
                    progress.dismiss()
                    val scrollView = android.widget.ScrollView(context)
                    val errorText = android.widget.TextView(context).apply {
                        text = stack
                        textSize = 10f
                        typeface = android.graphics.Typeface.MONOSPACE
                        setTextColor(android.graphics.Color.parseColor("#F05C5C"))
                        setPadding(30, 20, 30, 20)
                    }
                    scrollView.addView(errorText)
                    AlertDialog.Builder(context)
                        .setTitle(if (isKorean) "다운로드 실패" else "Install Failed")
                        .setView(scrollView)
                        .setPositiveButton(if (isKorean) "확인" else "OK", null)
                        .show()
                }
            }
        }.start()
    }

    private fun resolveSdkPackageUrl(repositoryXml: String, packagePath: String): String {
        val blocks = Regex(
            "<remotePackage\\b[^>]*\\bpath=\"${Regex.escape(packagePath)}\"[^>]*>[\\s\\S]*?</remotePackage>"
        ).findAll(repositoryXml).map { it.value }.toList()
        val block = blocks.firstOrNull { "obsolete=\"true\"" !in it } ?: blocks.firstOrNull()
            ?: throw RuntimeException("SDK package not found: $packagePath")
        val archive = Regex("<archive>[\\s\\S]*?</archive>")
            .findAll(block)
            .map { it.value }
            .firstOrNull { "<host-os>linux</host-os>" in it || "<host-os>" !in it }
            ?: throw RuntimeException("No Linux archive for SDK package: $packagePath")
        val url = Regex("<url>([^<]+)</url>").find(archive)?.groupValues?.get(1)
            ?: throw RuntimeException("No download URL for SDK package: $packagePath")
        return "https://dl.google.com/android/repository/$url"
    }

    private fun resolveNdkUrl(repositoryXml: String, packagePath: String): String {
        val block = Regex(
            """<remotePackage\b[^>]*\bpath="${Regex.escape(packagePath)}"[^>]*>[\s\S]*?</remotePackage>"""
        ).findAll(repositoryXml).map { it.value }.firstOrNull { "obsolete=\"true\"" !in it }
            ?: throw RuntimeException("NDK package not found: $packagePath")
        val url = Regex("""<url>([^<]+)</url>""").find(block)?.groupValues?.get(1)
            ?: throw RuntimeException("No download URL for $packagePath")
        return "https://dl.google.com/android/repository/$url"
    }

    private fun downloadFileWithProgress(url: String, dest: File, isKorean: Boolean = false, onProgress: (String) -> Unit) {
        val conn = java.net.URL(url).openConnection() as java.net.HttpURLConnection
        conn.connectTimeout = 30000; conn.readTimeout = 120000
        conn.instanceFollowRedirects = true; conn.connect()
        val total = conn.contentLengthLong
        dest.parentFile?.mkdirs()
        conn.inputStream.use { input ->
            dest.outputStream().use { output ->
                val buf = ByteArray(64 * 1024)
                var done = 0L; var lastEmit = 0L
                while (true) {
                    val read = input.read(buf); if (read < 0) break
                    output.write(buf, 0, read)
                    done += read
                    val now = System.currentTimeMillis()
                    if (now - lastEmit > 1000) {
                        lastEmit = now
                        val pct = if (total > 0) (done * 100 / total).coerceIn(0, 100) else -1
                        val size = if (total > 0) "${done / 1048576}MB / ${total / 1048576}MB"
                            else "${done / 1048576}MB"
                        val prefix = if (isKorean) "NDK 다운로드 중... " else "Downloading NDK... "
                        onProgress(if (pct >= 0) "$prefix$pct% ($size)" else "$prefix$size")
                    }
                }
            }
        }
    }
}
