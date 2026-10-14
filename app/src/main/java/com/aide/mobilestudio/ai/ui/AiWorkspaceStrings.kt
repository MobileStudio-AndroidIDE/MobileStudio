package com.aide.mobilestudio.ai.ui

import android.content.Context

/**
 * Minimal bilingual string helper for the AI workspace. English is the default;
 * Korean is used when the system locale is Korean. Code, paths and logs are never
 * translated — only UI labels go through here.
 */
object AiWorkspaceStrings {
    fun t(context: Context, en: String, ko: String): String {
        val lang = context.resources.configuration.locales.get(0).language
        return if (lang == "ko") ko else en
    }
}
