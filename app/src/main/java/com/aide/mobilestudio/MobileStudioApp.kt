package com.aide.mobilestudio

import android.app.Application

class MobileStudioApp : Application() {
    override fun onCreate() {
        super.onCreate()
        MobileStudioCrashGuard.install(this)
    }
}
