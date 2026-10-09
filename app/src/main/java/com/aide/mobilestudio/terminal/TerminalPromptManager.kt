package com.aide.mobilestudio.terminal

import java.io.File

/**
 * Manages dynamic terminal prompt with working directory display.
 * Format: ~/MobileStudio/KotlinApp >
 */
class TerminalPromptManager {

    private var homeDir: String = "/data/data/com.aide.mobilestudio/files"
    private var cwd: String = homeDir

    fun setHomeDir(path: String) {
        homeDir = path
    }

    fun setCwd(path: String) {
        cwd = path
    }

    fun getCwd(): String = cwd

    /**
     * Format the prompt string with path abbreviation.
     * Returns: "~/MobileStudio/KotlinApp >"
     */
    fun formatPrompt(): String {
        val displayPath = abbreviatePath(cwd, homeDir)
        return "$displayPath >"
    }

    /**
     * Abbreviate a path for display.
     * - Replaces homeDir with ~
     * - If the path is very long, keeps the first component (~) and the last two/three components.
     */
    fun abbreviatePath(path: String, home: String = homeDir): String {
        // Normalize path
        val normalizedPath = if (path.endsWith("/") && path.length > 1) path.substring(0, path.length - 1) else path
        val normalizedHome = if (home.endsWith("/") && home.length > 1) home.substring(0, home.length - 1) else home

        // Replace home with ~
        val displayPath = if (normalizedPath == normalizedHome) {
            "~"
        } else if (normalizedPath.startsWith("$normalizedHome/")) {
            "~" + normalizedPath.substring(normalizedHome.length)
        } else {
            normalizedPath
        }

        // If short enough (e.g. < 40 chars), return as-is
        if (displayPath.length <= 40) return displayPath

        // Tokenize
        val parts = displayPath.split("/").filter { it.isNotEmpty() }
        if (parts.size <= 4) return displayPath

        // Abbreviate middle: e.g. ~/A/B/C/D -> ~/A/.../C/D
        val first = parts[0]
        val lastTwo = parts.takeLast(2).joinToString("/")
        
        return "$first/.../$lastTwo"
    }

    /**
     * Try to detect and update cwd from a command.
     * Returns true if cwd was updated.
     */
    fun updateFromCommand(command: String): Boolean {
        val trimmed = command.trim()
        val parts = trimmed.split("\\s+".toRegex())
        if (parts.isEmpty()) return false

        val cmd = parts[0]
        return when (cmd) {
            "cd" -> {
                val target = parts.getOrElse(1) { "~" }
                val newCwd = resolveCwd(target)
                if (newCwd != null) {
                    cwd = newCwd
                    true
                } else false
            }
            "pushd" -> {
                if (parts.size > 1) {
                    val newCwd = resolveCwd(parts[1])
                    if (newCwd != null) {
                        cwd = newCwd
                        true
                    } else false
                } else false
            }
            "popd" -> {
                // popd changes to previous dir in stack — we can't track the stack,
                // but at least try to go up one level
                val parent = File(cwd).parent
                if (parent != null) {
                    cwd = parent
                    true
                } else false
            }
            else -> false
        }
    }

    private fun resolveCwd(target: String): String? {
        val resolved = when {
            target == "~" || target.isEmpty() -> homeDir
            target.startsWith("~/") -> "$homeDir${target.removePrefix("~")}"
            target.startsWith("/") -> target
            target == ".." -> File(cwd).parent ?: cwd
            target == "." -> cwd
            else -> "$cwd/$target"
        }
        return resolved
    }
}
