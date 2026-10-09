package com.aide.mobilestudio

import android.content.Context
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

object ThemeManager {
    const val PREF_NAME = "ide"
    const val KEY_THEME = "app_theme"
    const val KEY_DARK_MODE = "dark_mode"

    enum class AppTheme(val label: String) {
        DARK("Dark"),
        LIGHT("Light"),
        DRACULA("Dracula"),
        MONOKAI("Monokai"),
        NORD("Nord")
    }

    enum class DarkMode(val label: String) {
        SYSTEM("System default"),
        LIGHT("Light"),
        DARK("Dark")
    }

    fun getTheme(context: Context): AppTheme {
        val name = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .getString(KEY_THEME, AppTheme.DARK.name) ?: AppTheme.DARK.name
        return runCatching { AppTheme.valueOf(name) }.getOrDefault(AppTheme.DARK)
    }

    fun setTheme(context: Context, theme: AppTheme) {
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_THEME, theme.name).apply()
    }

    fun getDarkMode(context: Context): DarkMode {
        val name = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .getString(KEY_DARK_MODE, DarkMode.SYSTEM.name) ?: DarkMode.SYSTEM.name
        return runCatching { DarkMode.valueOf(name) }.getOrDefault(DarkMode.SYSTEM)
    }

    fun setDarkMode(context: Context, mode: DarkMode) {
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_DARK_MODE, mode.name).apply()
    }

    @Composable
    fun ProvideTheme(content: @Composable () -> Unit) {
        val appTheme = getTheme(androidx.compose.ui.platform.LocalContext.current)
        val darkMode = getDarkMode(androidx.compose.ui.platform.LocalContext.current)
        val isDark = when (darkMode) {
            DarkMode.LIGHT -> false
            DarkMode.DARK -> true
            DarkMode.SYSTEM -> isSystemInDarkTheme()
        }
        val colorScheme = when (appTheme) {
            AppTheme.DARK -> if (isDark) DarkDark else DarkLight
            AppTheme.LIGHT -> if (isDark) DarkDark else DarkLight
            AppTheme.DRACULA -> if (isDark) DraculaDark else DraculaLight
            AppTheme.MONOKAI -> if (isDark) MonokaiDark else MonokaiLight
            AppTheme.NORD -> if (isDark) NordDark else NordLight
        }
        MaterialTheme(colorScheme = colorScheme, content = content)
    }
}

// Dark theme
private val DarkDark = darkColorScheme(
    primary = Color(0xFF58A6FF), onPrimary = Color(0xFF0D1117),
    primaryContainer = Color(0xFF1F6FEB), onPrimaryContainer = Color(0xFFC9D1D9),
    secondary = Color(0xFF8B949E), onSecondary = Color(0xFF0D1117),
    secondaryContainer = Color(0xFF21262D), onSecondaryContainer = Color(0xFFC9D1D9),
    surface = Color(0xFF161B22), onSurface = Color(0xFFC9D1D9),
    surfaceVariant = Color(0xFF21262D), onSurfaceVariant = Color(0xFF8B949E),
    background = Color(0xFF0D1117), onBackground = Color(0xFFC9D1D9)
)
private val DarkLight = lightColorScheme(
    primary = Color(0xFF1F6FEB), onPrimary = Color.White,
    primaryContainer = Color(0xFFC9D1FF), onPrimaryContainer = Color(0xFF0D1117),
    secondary = Color(0xFF57606A), onSecondary = Color.White,
    secondaryContainer = Color(0xFFECEEF1), onSecondaryContainer = Color(0xFF0D1117),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF1A1A1A),
    surfaceVariant = Color(0xFFF5F7FA), onSurfaceVariant = Color(0xFF57606A),
    background = Color(0xFFF5F7FA), onBackground = Color(0xFF1A1A1A)
)

// Dracula theme
private val DraculaDark = darkColorScheme(
    primary = Color(0xFFFF79C6), onPrimary = Color(0xFF282A36),
    primaryContainer = Color(0xFFBD93F9), onPrimaryContainer = Color(0xFF282A36),
    secondary = Color(0xFF8BE9FD), onSecondary = Color(0xFF282A36),
    secondaryContainer = Color(0xFF44475A), onSecondaryContainer = Color(0xFFF8F8F2),
    surface = Color(0xFF21222C), onSurface = Color(0xFFF8F8F2),
    surfaceVariant = Color(0xFF44475A), onSurfaceVariant = Color(0xFFF8F8F2),
    background = Color(0xFF282A36), onBackground = Color(0xFFF8F8F2)
)
private val DraculaLight = lightColorScheme(
    primary = Color(0xFFBD93F9), onPrimary = Color.White,
    primaryContainer = Color(0xFFE8DEFF), onPrimaryContainer = Color(0xFF282A36),
    secondary = Color(0xFF50FA7B), onSecondary = Color(0xFF282A36),
    secondaryContainer = Color(0xFFD6E6FF), onSecondaryContainer = Color(0xFF282A36),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF282A36),
    surfaceVariant = Color(0xFFF1F1F4), onSurfaceVariant = Color(0xFF57577A),
    background = Color(0xFFF8F8FF), onBackground = Color(0xFF282A36)
)

// Monokai theme
private val MonokaiDark = darkColorScheme(
    primary = Color(0xFFA9DC76), onPrimary = Color(0xFF2D2A2E),
    primaryContainer = Color(0xFFFC9867), onPrimaryContainer = Color(0xFF2D2A2E),
    secondary = Color(0xFF78DCE8), onSecondary = Color(0xFF2D2A2E),
    secondaryContainer = Color(0xFF403E41), onSecondaryContainer = Color(0xFFFDFDFE),
    surface = Color(0xFF403E41), onSurface = Color(0xFFFDFDFE),
    surfaceVariant = Color(0xFF5B595C), onSurfaceVariant = Color(0xFFFDFDFE),
    background = Color(0xFF2D2A2E), onBackground = Color(0xFFFDFDFE)
)
private val MonokaiLight = lightColorScheme(
    primary = Color(0xFFA9DC76), onPrimary = Color(0xFF2D2A2E),
    primaryContainer = Color(0xFFD8EDD0), onPrimaryContainer = Color(0xFF2D2A2E),
    secondary = Color(0xFF78DCE8), onSecondary = Color(0xFF2D2A2E),
    secondaryContainer = Color(0xFFD0ECF0), onSecondaryContainer = Color(0xFF2D2A2E),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF2D2A2E),
    surfaceVariant = Color(0xFFF5F5F5), onSurfaceVariant = Color(0xFF75717E),
    background = Color(0xFFF8F8F8), onBackground = Color(0xFF2D2A2E)
)

// Nord theme
private val NordDark = darkColorScheme(
    primary = Color(0xFF88C0D0), onPrimary = Color(0xFF2E3440),
    primaryContainer = Color(0xFF5E81AC), onPrimaryContainer = Color(0xFFECEFF4),
    secondary = Color(0xFF81A1C1), onSecondary = Color(0xFF2E3440),
    secondaryContainer = Color(0xFF3B4252), onSecondaryContainer = Color(0xFFECEFF4),
    surface = Color(0xFF3B4252), onSurface = Color(0xFFECEFF4),
    surfaceVariant = Color(0xFF434C5E), onSurfaceVariant = Color(0xFFD8DEE9),
    background = Color(0xFF2E3440), onBackground = Color(0xFFECEFF4)
)
private val NordLight = lightColorScheme(
    primary = Color(0xFF5E81AC), onPrimary = Color.White,
    primaryContainer = Color(0xFFD8DEEA), onPrimaryContainer = Color(0xFF2E3440),
    secondary = Color(0xFF81A1C1), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE5E9F0), onSecondaryContainer = Color(0xFF2E3440),
    surface = Color(0xFFFFFFFF), onSurface = Color(0xFF2E3440),
    surfaceVariant = Color(0xFFE5E9F0), onSurfaceVariant = Color(0xFF4C566A),
    background = Color(0xFFECEFF4), onBackground = Color(0xFF2E3440)
)
