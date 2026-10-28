package com.aide.mobilestudio.ui

import android.content.Context
import android.graphics.Color
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView

class HomeScreen(context: Context) : Screen(context) {

    override val id: String = "home"

    override fun build(): View {
        val c = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#0B0F15"))
        }

        val title = TextView(context).apply {
            text = "MobileStudio"
            textSize = 32f
            setTextColor(Color.WHITE)
            setPadding(0, 60, 0, 20)
        }
        c.addView(title)

        return c
    }
}
