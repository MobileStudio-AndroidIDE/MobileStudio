package com.aide.mobilestudio

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.text.InputType
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.recyclerview.widget.RecyclerView
import com.aide.mobilestudio.AppLanguage
import java.io.File

private fun Context.buildToolsPanel(): View {
    val tabNames = listOf("Build", "APK", "HTML")
    val tabIds = listOf(R.id.tools_tab_build, R.id.tools_tab_apk, R.id.tools_tab_community)
    val panelIds = listOf(R.id.tools_panel_build, R.id.tools_panel_apk, R.id.tools_panel_community)
    return LinearLayout(this).apply {
        id = R.id.tools_panel_root
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(c(StudioUi.LayerBottomPanel))
        // Header
        addView(LinearLayout(this@buildToolsPanel).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(12), dp(8), dp(12), dp(8))
            setBackgroundColor(c(StudioUi.LayerPanelTabs))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(40))
            addView(TextView(this@buildToolsPanel).apply {
                text = "Tools"
                textSize = 13f
                typeface = Typeface.MONOSPACE
                setTextColor(c("#DCE3EE"))
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
            addView(TextView(this@buildToolsPanel).apply {
                id = R.id.btn_close_tools
                text = "\u2630"
                textSize = 16f
                setTextColor(c("#8E99AC"))
                setPadding(dp(6), dp(4), dp(6), dp(4))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
        })
        // Tab strip
        addView(LinearLayout(this@buildToolsPanel).apply {
            id = R.id.tools_tab_strip
            orientation = LinearLayout.HORIZONTAL
            setPadding(dp(4), 0, dp(4), 0)
            setBackgroundColor(c(StudioUi.LayerTabs))
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(36))
            tabNames.forEachIndexed { i, name ->
                addView(TextView(this@buildToolsPanel).apply {
                    id = tabIds[i]
                    text = name
                    textSize = 11f
                    typeface = Typeface.MONOSPACE
                    gravity = Gravity.CENTER
                    setTextColor(c(if (i == 0) "#4F8EF7" else "#8E99AC"))
                    if (i == 0) background = StudioUi.rounded(this@buildToolsPanel, fill = "#232D3B", stroke = "#4F8EF7", radiusDp = 12)
                    setPadding(dp(10), 0, dp(10), 0)
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
                })
            }
        })
        // Content area (FrameLayout for tab switching)
        addView(FrameLayout(this@buildToolsPanel).apply {
            id = R.id.tools_panel_content
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
            // Build tab
            addView(LinearLayout(this@buildToolsPanel).apply {
                id = R.id.tools_panel_build
                orientation = LinearLayout.VERTICAL
                setPadding(dp(10), dp(10), dp(10), dp(10))
                addView(TextView(this@buildToolsPanel).apply {
                    text = "Build"
                    textSize = 12f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c("#C3CCD8"))
                })
                addView(LinearLayout(this@buildToolsPanel).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(0, dp(8), 0, dp(4))
                    addView(TextView(this@buildToolsPanel).apply {
                        id = R.id.tools_build_option_label
                        text = "assembleDebug"
                        textSize = 11f
                        typeface = Typeface.MONOSPACE
                        setTextColor(c("#AFB9C8"))
                        setPadding(dp(4), 0, dp(4), 0)
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
                    addView(Button(this@buildToolsPanel).apply {
                        id = R.id.tools_btn_run
                        text = "🛠️ Build"
                        textSize = 11f
                        studioButton("#2E8549", "#46C07D")
                        layoutParams = LinearLayout.LayoutParams(dp(80), dp(36))
                    })
                })
                addView(LinearLayout(this@buildToolsPanel).apply {
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(0, dp(4), 0, dp(4))
                    addView(Button(this@buildToolsPanel).apply {
                        id = R.id.tools_btn_stop
                        text = "⏹️ Stop"
                        textSize = 10f
                        studioButton("#CE4A4E", "#F05C5C")
                        layoutParams = LinearLayout.LayoutParams(0, dp(32), 1f)
                    })
                    addView(Button(this@buildToolsPanel).apply {
                        id = R.id.tools_btn_clean
                        text = "\uD83E\uDDF9 Clean"
                        textSize = 10f
                        studioButton("#262B33", "#E3B963")
                        layoutParams = LinearLayout.LayoutParams(0, dp(32), 1f).apply { marginStart = dp(4) }
                    })
                    addView(Button(this@buildToolsPanel).apply {
                        id = R.id.tools_btn_apk
                        text = "\uD83D\uDCE6 APK"
                        textSize = 10f
                        studioButton("#262B33", "#4F8EF7")
                        layoutParams = LinearLayout.LayoutParams(0, dp(32), 1f).apply { marginStart = dp(4) }
                    })
                })
                addView(ProgressBar(this@buildToolsPanel).apply {
                    id = R.id.tools_build_progress
                    visibility = View.INVISIBLE
                    layoutParams = LinearLayout.LayoutParams(dp(20), dp(20)).apply { topMargin = dp(6) }
                })
                addView(TextView(this@buildToolsPanel).apply {
                    id = R.id.tools_build_output
                    text = ""
                    textSize = 10f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c("#8E99AC"))
                    setPadding(dp(2), dp(6), dp(2), dp(2))
                    minLines = 1
                    maxLines = 8
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            // APK tab
            addView(LinearLayout(this@buildToolsPanel).apply {
                id = R.id.tools_panel_apk
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                setPadding(dp(10), dp(10), dp(10), dp(10))
                addView(TextView(this@buildToolsPanel).apply {
                    text = "APK Download"
                    textSize = 12f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c("#C3CCD8"))
                })
                addView(ScrollView(this@buildToolsPanel).apply {
                    addView(LinearLayout(this@buildToolsPanel).apply {
                        id = R.id.tools_apk_list
                        orientation = LinearLayout.VERTICAL
                        setPadding(0, dp(8), 0, 0)
                    })
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            // HTML Viewer tab
            addView(LinearLayout(this@buildToolsPanel).apply {
                id = R.id.tools_panel_community
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                setPadding(dp(10), dp(10), dp(10), dp(10))
                addView(TextView(this@buildToolsPanel).apply {
                    text = "HTML Viewer"
                    textSize = 12f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c("#C3CCD8"))
                })
                addView(TextView(this@buildToolsPanel).apply {
                    text = "View and edit HTML files with full CSS/JS support."
                    textSize = 11f
                    setTextColor(c("#8E99AC"))
                    setPadding(0, dp(8), 0, dp(8))
                })
                addView(Button(this@buildToolsPanel).apply {
                    id = R.id.tools_community_chat
                    text = "\uD83C\uDF10 Open HTML Viewer"
                    textSize = 11f
                    studioButton("#B57CE7", "#FFFFFF")
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(40)).apply { topMargin = dp(4) }
                })
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        })
    }
}

internal fun MainActivity.buildMainLayout(): View {
    val root = DrawerLayout(this).apply {
        id = R.id.drawer_layout
        setBackgroundColor(c(StudioUi.Ink))
    }
    root.addView(buildMainSurface())
    val drawerWidth = minOf((resources.displayMetrics.widthPixels * 0.82f).toInt(), dp(340)).coerceAtLeast(dp(240))
    root.addView(buildDrawerSurface(), DrawerLayout.LayoutParams(drawerWidth, ViewGroup.LayoutParams.MATCH_PARENT).apply {
        gravity = GravityCompat.START
    })
        // Phase 4.2: Tools panel. DrawerLayout only supports LEFT/RIGHT/NO_GRAVITY
        // children — Gravity.BOTTOM throws "does not have a valid layout_gravity"
        // (IllegalStateException in DrawerLayout.onMeasure). Use an END drawer in
        // BOTH orientations; in landscape it's wider (60% of screen height was the
        // old broken bottom-sheet attempt).
        val toolsPanelWidth = minOf((resources.displayMetrics.widthPixels * 0.75f).toInt(), dp(360)).coerceAtLeast(dp(260))
        root.addView(buildToolsPanel(), DrawerLayout.LayoutParams(toolsPanelWidth, ViewGroup.LayoutParams.MATCH_PARENT).apply {
            gravity = GravityCompat.END
        })
    // Phase 4.3: Reduce drawer swipe edge from 24dp to 16dp to avoid back gesture conflict
    root.setDrawerLockMode(DrawerLayout.LOCK_MODE_UNLOCKED)
    root.setScrimColor(Color.TRANSPARENT)
    // Note: DrawerLayout edge size is not directly configurable in AndroidX,
    // but we can minimize the scrim and ensure gestures don't conflict
    return root
}

private fun Context.buildMainSurface(): View {
    val isLandscape = isLandscapeScreen()
    val isTablet = resources.displayMetrics.widthPixels >= 600 * resources.displayMetrics.density
    
    // Phase 6.1: Horizontal mode 3-pane layout (file tree | editor | bottom panel)
    if (isLandscape && isTablet) {
        return buildLandscapeMainSurface()
    }
    
    return FrameLayout(this).apply {
        val content = LinearLayout(this@buildMainSurface).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(c(StudioUi.Ink))
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            addView(buildChrome())
            addView(buildProjectTabStrip())
            addView(buildTabStrip())
            addView(buildWorkspaceArea(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            addView(buildStatusBar())
        }
        addView(content)
    }
}

private fun Context.isLandscapeScreen(): Boolean =
    resources.displayMetrics.widthPixels > resources.displayMetrics.heightPixels

/**
 * The menu row and the toolbar row.
 *
 * Portrait stacks them, which is fine because there is vertical room to spare.
 * A landscape phone has almost none - the chrome eats a fifth of the screen -
 * so the menu scrolls horizontally inside the toolbar row instead, giving the
 * editor the full 28dp back without dropping any menu: the items are found by
 * id, so moving them to another parent leaves every handler intact.
 */
private fun Context.buildChrome(): View {
    val menu = buildTopTextMenu()
    val toolbar = buildPrimaryToolbar()
    if (!isLandscapeScreen()) {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            addView(menu)
            addView(toolbar)
        }
    }
    return LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setBackgroundColor(c(StudioUi.LayerToolbar))
        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(40))
        addView(toolbar, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 3f))
        addView(View(this@buildChrome), LinearLayout.LayoutParams(dp(1), ViewGroup.LayoutParams.MATCH_PARENT).apply {
            // A hairline so the two groups read as separate, not as one row.
            setBackgroundColor(StudioUi.color("#3E4451"))
        })
        // The menu row keeps its own 28dp height so the parent's CENTER_VERTICAL
        // centres it instead of leaving it pinned to the top of the 40dp row.
        addView(menu, LinearLayout.LayoutParams(0, dp(28), 2f))
    }
}

private fun Context.buildLandscapeMainSurface(): View {
    val ctx = this
    return FrameLayout(ctx).apply {
        val root = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            background = StudioUi.gradient(ctx, "#0B0F15", "#12161F")
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            
            // Left pane: File tree (drawer-like, but always visible)
            val leftPane = FrameLayout(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(dp(320), ViewGroup.LayoutParams.MATCH_PARENT)
                background = StudioUi.gradient(ctx, "#0E1524", "#0A0F15")
                id = R.id.landscape_file_tree_container
            }
            addView(leftPane)
            
            // Vertical divider
            addView(View(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(dp(1), ViewGroup.LayoutParams.MATCH_PARENT)
                setBackgroundColor(StudioUi.color("#3E4451"))
            })
            
            // Center: Editor area (flexible)
            val centerPane = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
                background = StudioUi.gradient(ctx, "#0B0F15", "#12161F")
                
                addView(ctx.buildChrome())
                addView(ctx.buildProjectTabStrip())
                addView(ctx.buildTabStrip())
                addView(ctx.buildWorkspaceArea(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
                addView(ctx.buildStatusBar())
            }
            addView(centerPane)
            
            // Vertical divider
            addView(View(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(dp(1), ViewGroup.LayoutParams.MATCH_PARENT)
                setBackgroundColor(StudioUi.color("#3E4451"))
            })
            
            // Right pane: Bottom panel (fixed width)
            val rightPane = FrameLayout(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(dp(360), ViewGroup.LayoutParams.MATCH_PARENT)
                id = R.id.landscape_bottom_panel_container
            }
            addView(rightPane)
        }
        addView(root)
    }
}

private fun Context.buildProjectTabStrip(): View {
    return HorizontalScrollView(this).apply {
        id = R.id.project_tab_scroll
        isHorizontalScrollBarEnabled = false
        visibility = View.GONE
        background = StudioUi.gradient(this@buildProjectTabStrip, "#0E1524", "#0A0F15")
        addView(LinearLayout(this@buildProjectTabStrip).apply {
            id = R.id.project_tab_container
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), 0, dp(6), 0)
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(30))
        })
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(30))
        // Tag to track tab count for conditional visibility
        tag = 0
    }
}

/** Updates project tab strip visibility based on tab count */
fun MainActivity.updateProjectTabStripVisibility(tabCount: Int) {
    val tabScroll = findViewById<HorizontalScrollView>(R.id.project_tab_scroll)
    val container = findViewById<LinearLayout>(R.id.project_tab_container)
    if (tabCount <= 1) {
        tabScroll.visibility = View.GONE
    } else {
        tabScroll.visibility = View.VISIBLE
        container.tag = tabCount
    }
}

private fun Context.buildTopTextMenu(): View {
    return HorizontalScrollView(this).apply {
        isHorizontalScrollBarEnabled = false
        setBackgroundColor(c(StudioUi.Surface))
        addView(LinearLayout(this@buildTopTextMenu).apply {
            id = R.id.top_text_menu
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), 0, dp(6), 0)
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(28))
            listOf(
                R.id.menu_file       to AppLanguage.literal(this@buildTopTextMenu, "File"),
                R.id.menu_edit       to AppLanguage.literal(this@buildTopTextMenu, "Edit"),
                R.id.menu_view       to AppLanguage.literal(this@buildTopTextMenu, "View"),
                R.id.menu_navigate   to AppLanguage.literal(this@buildTopTextMenu, "Navigate"),
                R.id.menu_code       to AppLanguage.literal(this@buildTopTextMenu, "Code"),
                R.id.menu_refactor   to AppLanguage.literal(this@buildTopTextMenu, "Refactor"),
                R.id.menu_build      to AppLanguage.literal(this@buildTopTextMenu, "Build"),
                R.id.menu_run        to AppLanguage.literal(this@buildTopTextMenu, "Run"),
                R.id.menu_community  to AppLanguage.literal(this@buildTopTextMenu, "Community"),
                R.id.menu_extensions to AppLanguage.literal(this@buildTopTextMenu, "Extensions"),
                R.id.menu_ide        to "IDE",
                R.id.menu_tools      to AppLanguage.literal(this@buildTopTextMenu, "Tools")
            ).forEach { (viewId, label) ->
                addView(menuText(viewId, label))
            }
        })
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(28))
    }
}

private fun Context.buildPrimaryToolbar(): View {
    return LinearLayout(this).apply {
        // Row B is the ONLY remaining toolbar row — it keeps the legacy id so
        // applyTabletMode()/applyPhoneMode() in MainActivity keep working.
        id = R.id.build_icon_toolbar
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(4), 0, dp(4), 0)
        setBackgroundColor(c(StudioUi.LayerToolbar))
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(40))

        // Left: file history navigation (wired in MainActivity.setupToolbar)
        addView(iconButton(R.id.btn_back, android.R.drawable.ic_menu_revert, StudioUi.TextSoft, "Back", 34))
        addView(iconButton(R.id.btn_forward, android.R.drawable.ic_media_ff, StudioUi.TextSoft, "Forward", 34))

        // Spacer: pushes the project name group toward the center
        addView(View(this@buildPrimaryToolbar), LinearLayout.LayoutParams(0, 1, 1f))

        // Center: Project name (fills remaining space)
        addView(TextView(this@buildPrimaryToolbar).apply {
            id = R.id.tv_project_name
            text = "MobileStudio"
            textSize = 13f
            typeface = Typeface.MONOSPACE
            setTextColor(c(StudioUi.Text))
            setSingleLine(true)
            ellipsize = android.text.TextUtils.TruncateAt.END
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), 0, dp(4), 0)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))

        // Center-right: Error/Warning badges
        addView(TextView(this@buildPrimaryToolbar).apply {
            id = R.id.badge_warnings
            text = ""
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(c(StudioUi.ActiveAmber))
            visibility = View.GONE
            setPadding(dp(2), 0, dp(2), 0)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        addView(TextView(this@buildPrimaryToolbar).apply {
            id = R.id.badge_errors
            text = ""
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(c(StudioUi.ActiveRed))
            visibility = View.GONE
            setPadding(dp(2), 0, dp(4), 0)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        // Center-right: Build status text (hidden when not building)
        addView(TextView(this@buildPrimaryToolbar).apply {
            id = R.id.tv_build_status
            text = ""
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(c(StudioUi.ActiveGreen))
            setSingleLine(true)
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            setPadding(dp(4), 0, dp(4), 0)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))

        // Spacer: pushes sync/run to the right edge
        addView(View(this@buildPrimaryToolbar), LinearLayout.LayoutParams(0, 1, 1f))

        // Right: Sync button
        addView(iconButton(R.id.btn_sync_now, android.R.drawable.ic_menu_rotate, StudioUi.TextSoft, "Sync", 32))

        // Right: Run/Build button — 유일하게 색을 남긴다(주요 액션)
        addView(iconButton(R.id.btn_run, android.R.drawable.ic_media_play, StudioUi.ActiveGreen, "Build", 34))

        // Right: build progress indicator (moved up from the deleted build toolbar;
        // setBuildRunning() in MainActivity depends on this id being present)
        addView(ProgressBar(this@buildPrimaryToolbar).apply {
            id = R.id.build_progress
            visibility = View.INVISIBLE
            layoutParams = LinearLayout.LayoutParams(dp(20), dp(20))
        })

        // Hidden placeholder: MainActivity still drives btn_preview visibility
        // (updatePreviewButtonVisibility) and its click handler, so this one id
        // must keep a view in the tree. Everything else was dead code and went
        // away together with its handler.
        addView(hiddenIcon(R.id.btn_preview))
    }
}

private fun Context.buildTabStrip(): View {
    return LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(c(StudioUi.LayerTabs))
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(34))
        addView(HorizontalScrollView(this@buildTabStrip).apply {
            id = R.id.tab_scroll
            isHorizontalScrollBarEnabled = false
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(34))
            addView(LinearLayout(this@buildTabStrip).apply {
                id = R.id.tab_container
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(4), 0, dp(4), 0)
                layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(34))
            })
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        addView(View(this@buildTabStrip).apply {
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1))
            setBackgroundColor(c(StudioUi.Divider))
        })
    }
}

private fun Context.buildWorkspaceArea(): View {
    // The banner, welcome screen and editor form one stack in both
    // orientations; only what happens to the build-output panel differs.
    val editorStack = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(c(StudioUi.Ink))
        layoutParams = LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
            1f,
        )
        addView(LinearLayout(this@buildWorkspaceArea).apply {
            id = R.id.banner_container
            orientation = LinearLayout.VERTICAL
            visibility = View.GONE
        })
        addView(buildWelcomeScreen())
        addView(buildEditorContainer())
    }

    if (!isLandscapeScreen()) {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(c(StudioUi.Ink))
            addView(editorStack)
            addView(View(this@buildWorkspaceArea).apply {
                id = R.id.panel_divider
                setBackgroundColor(c(StudioUi.Divider))
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)))
            addView(buildPanelGrabber(landscape = false))
            addView(buildBottomPanel())
        }
    }

    // Landscape: the build-output panel moves beside the editor. A phone in
    // landscape has only ~360dp of height, and a panel across the bottom eats
    // a third of it; there is plenty of width instead. The divider stays the
    // same view, so MainActivity's drag-to-resize keeps working - it adjusts
    // width here and height in portrait.
    return LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        setBackgroundColor(c(StudioUi.Ink))
        addView(editorStack)
        addView(View(this@buildWorkspaceArea).apply {
            id = R.id.panel_divider
            setBackgroundColor(c(StudioUi.Divider))
        }, LinearLayout.LayoutParams(dp(1), ViewGroup.LayoutParams.MATCH_PARENT))
        addView(buildPanelGrabber(landscape = true))
        addView(buildBottomPanel(), LinearLayout.LayoutParams(
            landscapePanelWidth(),
            ViewGroup.LayoutParams.MATCH_PARENT,
        ))
    }
}

/**
 * The handle that pulls the bottom panel up and lets it back down.
 *
 * It sits directly against the panel's leading edge as a sibling rather than a
 * child of it, so it is still on screen while the panel is collapsed - the only
 * way a collapsed panel stays reachable - and it rides that edge as the edge
 * animates in. No separate animation is needed: the panel's own height
 * animation carries the handle with it.
 *
 * The glyph points the way the panel travels, and flips once the panel is out,
 * so the direction of the next press is always what is shown.
 */
private fun Context.buildPanelGrabber(landscape: Boolean): View {
    val icon = TextView(this).apply {
        id = R.id.shell_grabber_icon
        text = "^"
        textSize = 13f
        gravity = Gravity.CENTER
        setTextColor(c(StudioUi.TextSoft))
        rotation = if (landscape) -90f else 0f
    }
    return FrameLayout(this).apply {
        id = R.id.shell_grabber
        background = StudioUi.rounded(
            this@buildPanelGrabber,
            fill = StudioUi.Surface2,
            stroke = StudioUi.Border,
            radiusDp = 8,
        )
        addView(
            icon,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
        isClickable = true
        isFocusable = true
        contentDescription =
            if (landscape) "Show or hide the side panel" else "Show or hide the bottom panel"
        layoutParams = if (landscape) {
            LinearLayout.LayoutParams(dp(PANEL_GRABBER_SIZE_DP), ViewGroup.LayoutParams.MATCH_PARENT)
        } else {
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(PANEL_GRABBER_SIZE_DP))
        }
    }
}

/** Thickness of the panel grabber; also its touch target. */
private const val PANEL_GRABBER_SIZE_DP = 28

/** Share of the screen width the build-output panel takes in landscape. */
private fun Context.landscapePanelWidth(): Int {
    val widthDp = resources.displayMetrics.widthPixels / resources.displayMetrics.density
    return (widthDp * 0.34f).toInt().coerceIn(dp(280), dp(440))
}

private fun Context.buildWelcomeScreen(): View {
    return ScrollView(this).apply {
        id = R.id.welcome_screen
        setBackgroundColor(c(StudioUi.Ink))
        // 오른쪽 가장자리에 스크롤바 두께만큼 흰 줄이 보이던 것 제거
        isVerticalScrollBarEnabled = false
        isScrollbarFadingEnabled = false
        addView(LinearLayout(this@buildWelcomeScreen).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(26), dp(24), dp(24))
            addView(LinearLayout(this@buildWelcomeScreen).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(22), dp(20), dp(22), dp(20))
                background = StudioUi.glow(this@buildWelcomeScreen, "#0C1322", StudioUi.Blue, 24)
                addView(TextView(this@buildWelcomeScreen).apply {
                    text = "M"
                    textSize = 36f
                    gravity = Gravity.CENTER
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(c("#08111E"))
                    background = StudioUi.gradient(this@buildWelcomeScreen, "#4F8EF7", "#B57CE7", 24)
                    layoutParams = LinearLayout.LayoutParams(dp(86), dp(86)).apply { bottomMargin = dp(14) }
                })
                addView(TextView(this@buildWelcomeScreen).apply {
                    text = "MOBILESTUDIO"
                    textSize = 10f
                    gravity = Gravity.CENTER
                    typeface = Typeface.MONOSPACE
                    letterSpacing = 0.32f
                    setTextColor(c("#4F8EF7"))
                    setPadding(0, 0, 0, dp(4))
                })
                addView(TextView(this@buildWelcomeScreen).apply {
                    text = "MobileStudio"
                    textSize = 24f
                    gravity = Gravity.CENTER
                    typeface = Typeface.DEFAULT_BOLD
                    setTextColor(c(StudioUi.Text))
                })
                addView(TextView(this@buildWelcomeScreen).apply {
                    text = "on-device Android IDE bootstrap"
                    textSize = 12f
                    gravity = Gravity.CENTER
                    letterSpacing = 0.08f
                    setTextColor(c(StudioUi.Green))
                    setPadding(0, dp(4), 0, dp(16))
                })
                addView(TextView(this@buildWelcomeScreen).apply {
                    text = "Build, inspect, preview, chat, and ship projects from a phone-sized workspace."
                    textSize = 13f
                    gravity = Gravity.CENTER
                    setTextColor(c(StudioUi.TextSoft))
                })
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(18)
            })
            addView(TextView(this@buildWelcomeScreen).apply {
                text = "MS"
                textSize = 36f
                gravity = Gravity.CENTER
                typeface = Typeface.DEFAULT_BOLD
                setTextColor(c("#4F8EF7"))
                setPadding(0, 0, 0, dp(6))
                visibility = View.GONE
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
            addView(TextView(this@buildWelcomeScreen).apply {
                text = "MobileStudio"
                textSize = 22f
                gravity = Gravity.CENTER
                typeface = Typeface.MONOSPACE
                setTextColor(c("#E6EDF3"))
                visibility = View.GONE
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
            addView(TextView(this@buildWelcomeScreen).apply {
                text = "Android IDE on Device"
                textSize = 12f
                gravity = Gravity.CENTER
                setTextColor(c("#8E99AC"))
                setPadding(0, dp(3), 0, dp(22))
                visibility = View.GONE
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
            addView(LinearLayout(this@buildWelcomeScreen).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                setPadding(0, 0, 0, dp(22))
                // 하드코딩된 hex는 테마 리맵 대상이 아니어서 밝은 테마에서
                // "어두운 글자 + 어두운 배경"이 된다. 반드시 시맨틱 팔레트 상수를 쓴다.
                addView(actionButton(R.id.btn_new_project, "New Project", StudioUi.Blue, "#FFFFFF"))
                addView(actionButton(R.id.btn_open_project, "Open Project", StudioUi.Card2, StudioUi.Text))
                addView(actionButton(R.id.btn_ai_welcome, "AI", StudioUi.Purple, "#FFFFFF").apply {
                    setOnClickListener { (context as? MainActivity)?.launchAiAssistant() }
                })
            })
            addView(TextView(this@buildWelcomeScreen).apply {
                text = AppLanguage.literal(this@buildWelcomeScreen, "Recent Projects")
                textSize = 11f
                typeface = Typeface.MONOSPACE
                setTextColor(c("#8E99AC"))
                id = R.id.recent_label
                setPadding(0, 0, 0, dp(8))
            })
            addView(RecyclerView(this@buildWelcomeScreen).apply {
                id = R.id.rv_recent_projects
                visibility = View.GONE
                isNestedScrollingEnabled = false
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
            addView(TextView(this@buildWelcomeScreen).apply {
                text = AppLanguage.literal(this@buildWelcomeScreen, "Recent Files")
                textSize = 11f
                typeface = Typeface.MONOSPACE
                setTextColor(c("#8E99AC"))
                id = R.id.recent_files_label
                setPadding(0, dp(14), 0, dp(8))
                visibility = View.GONE
            })
            addView(RecyclerView(this@buildWelcomeScreen).apply {
                id = R.id.rv_recent_files
                visibility = View.GONE
                isNestedScrollingEnabled = false
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            })
        })
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
    }
}

private fun Context.buildEditorContainer(): View {
    return FrameLayout(this).apply {
        id = R.id.editor_container
        visibility = View.GONE
        addView(LinearLayout(this@buildEditorContainer).apply {
            orientation = LinearLayout.VERTICAL
            // Extension notice banner (small, non-blocking; e.g. Shell Script extension download)
            addView(LinearLayout(this@buildEditorContainer).apply {
                id = R.id.extension_banner
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                visibility = View.GONE
            })
            // Editor top bar
            addView(LinearLayout(this@buildEditorContainer).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                background = StudioUi.rounded(this@buildEditorContainer, fill = StudioUi.LayerToolbar, stroke = "#00000000", radiusDp = 0)
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(32))
                // 편집기 툴바도 상단과 같은 규칙: 아이콘은 단일 톤, 의미가 겹치지 않게 아이콘 분리.
                // (기존: Command Palette 와 AI Chat 이 둘 다 wrench 를 썼다)
                addView(ImageButton(this@buildEditorContainer).apply {
                    id = R.id.btn_shell_toggle
                    setImageResource(android.R.drawable.ic_menu_sort_by_size)
                    background = StudioUi.ripple(this@buildEditorContainer, StudioUi.rounded(this@buildEditorContainer, fill = "#00000000", stroke = "#00000000", radiusDp = StudioUi.RadiusXl))
                    setPadding(dp(6), dp(4), dp(6), dp(4))
                    setColorFilter(c(StudioUi.TextSoft))
                    setOnClickListener { (context as? MainActivity)?.toggleShellFromEditor() }
                    contentDescription = AppLanguage.literal(this@buildEditorContainer, "Shell")
                }, LinearLayout.LayoutParams(dp(32), dp(32)))
                addView(ImageButton(this@buildEditorContainer).apply {
                    id = R.id.btn_command_palette
                    setImageResource(R.drawable.ic_menu)
                    background = StudioUi.ripple(this@buildEditorContainer, StudioUi.rounded(this@buildEditorContainer, fill = "#00000000", stroke = "#00000000", radiusDp = StudioUi.RadiusXl))
                    setPadding(dp(6), dp(4), dp(6), dp(4))
                    setColorFilter(c(StudioUi.TextSoft))
                    setOnClickListener { (context as? MainActivity)?.showCommandPalette() }
                    contentDescription = AppLanguage.literal(this@buildEditorContainer, "Command Palette")
                }, LinearLayout.LayoutParams(dp(32), dp(32)))
                addView(ImageButton(this@buildEditorContainer).apply {
                    id = R.id.btn_ai_chat
                    setImageResource(R.drawable.ic_ai)
                    background = StudioUi.ripple(this@buildEditorContainer, StudioUi.rounded(this@buildEditorContainer, fill = "#00000000", stroke = "#00000000", radiusDp = StudioUi.RadiusXl))
                    setPadding(dp(6), dp(4), dp(6), dp(4))
                    setColorFilter(c(StudioUi.TextSoft))
                    setOnClickListener { (context as? MainActivity)?.openAiChatFromEditor() }
                    contentDescription = AppLanguage.literal(this@buildEditorContainer, "Open AI Chat")
                }, LinearLayout.LayoutParams(dp(32), dp(32)))
                addView(ImageButton(this@buildEditorContainer).apply {
                    id = R.id.btn_find_inline
                    setImageResource(android.R.drawable.ic_menu_search)
                    background = StudioUi.ripple(this@buildEditorContainer, StudioUi.rounded(this@buildEditorContainer, fill = "#00000000", stroke = "#00000000", radiusDp = StudioUi.RadiusXl))
                    setPadding(dp(6), dp(4), dp(6), dp(4))
                    setColorFilter(c(StudioUi.TextSoft))
                    setOnClickListener { (context as? MainActivity)?.toggleFindReplaceInline() }
                    contentDescription = AppLanguage.literal(this@buildEditorContainer, "Find/Replace")
                }, LinearLayout.LayoutParams(dp(32), dp(32)))
                // Breadcrumb
                addView(HorizontalScrollView(this@buildEditorContainer).apply {
                    isHorizontalScrollBarEnabled = false
                    addView(LinearLayout(this@buildEditorContainer).apply {
                        id = R.id.breadcrumb
                        orientation = LinearLayout.HORIZONTAL
                        gravity = Gravity.CENTER_VERTICAL
                        setPadding(dp(6), 0, dp(6), 0)
                        layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(28))
                    })
                }, LinearLayout.LayoutParams(0, dp(28), 1f))
                // Error/Warning navigation
                addView(LinearLayout(this@buildEditorContainer).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(6), 0, dp(8), 0)
                    addView(TextView(this@buildEditorContainer).apply {
                        id = R.id.editor_warning_count
                        text = ""
                        textSize = 10f
                        typeface = Typeface.MONOSPACE
                        setTextColor(c(StudioUi.ActiveAmber))
                        visibility = View.GONE
                        setPadding(dp(2), 0, dp(2), 0)
                    })
                    addView(TextView(this@buildEditorContainer).apply {
                        id = R.id.editor_error_count
                        text = ""
                        textSize = 10f
                        typeface = Typeface.MONOSPACE
                        setTextColor(c(StudioUi.ActiveRed))
                        visibility = View.GONE
                        setPadding(dp(2), 0, dp(2), 0)
                    })
                    addView(TextView(this@buildEditorContainer).apply {
                        id = R.id.btn_prev_error
                        text = "\u25B2"
                        textSize = 10f
                        setTextColor(c(StudioUi.InactiveIcon))
                        setPadding(dp(4), 0, dp(4), 0)
                        setOnClickListener { (context as? MainActivity)?.navigateToPrevError() }
                    })
                    addView(TextView(this@buildEditorContainer).apply {
                        id = R.id.btn_next_error
                        text = "\u25BC"
                        textSize = 10f
                        setTextColor(c(StudioUi.InactiveIcon))
                        setPadding(dp(4), 0, dp(4), 0)
                        setOnClickListener { (context as? MainActivity)?.navigateToNextError() }
                    })
                })
            })
            addView(buildFindReplacePanel())
            addView(buildQuickBar())
            addView(buildEditorOverlay(), LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
    }
}

private fun Context.buildFindReplacePanel(): View {
    return LinearLayout(this).apply {
        id = R.id.find_replace_panel
        orientation = LinearLayout.VERTICAL
        setPadding(dp(8), dp(6), dp(8), dp(6))
        background = StudioUi.rounded(this@buildFindReplacePanel, fill = StudioUi.LayerToolbar, stroke = "#00000000", radiusDp = 0)
        visibility = View.GONE
        // Row 1: Find
        addView(LinearLayout(this@buildFindReplacePanel).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(input(R.id.et_find, "Find...", imeOptions = EditorInfo.IME_ACTION_NEXT), LinearLayout.LayoutParams(0, dp(36), 1f))
            // Result count
            addView(TextView(this@buildFindReplacePanel).apply {
                id = R.id.tv_find_count
                text = "0 / 0"
                textSize = 11f
                typeface = Typeface.MONOSPACE
                gravity = Gravity.CENTER
                setTextColor(c(StudioUi.TextSoft))
                setPadding(dp(8), 0, dp(8), 0)
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            // Prev
            addView(iconButton(R.id.btn_find_prev, android.R.drawable.arrow_up_float, StudioUi.TextSoft, "Prev", 32))
            // Next
            addView(iconButton(R.id.btn_find_next, android.R.drawable.arrow_down_float, StudioUi.TextSoft, "Next", 32))
            // Case sensitive
            addView(checkBox(R.id.cb_case_sensitive, "Aa"))
            // Regex
            addView(checkBox(R.id.cb_regex, ".*"))
            // Close
            addView(iconButton(R.id.btn_find_close, android.R.drawable.ic_menu_close_clear_cancel, StudioUi.InactiveIcon, "Close", 32))
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(38)))
        // Row 2: Replace
        addView(LinearLayout(this@buildFindReplacePanel).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(input(R.id.et_replace, "Replace with..."), LinearLayout.LayoutParams(0, dp(36), 1f))
            addView(actionButton(R.id.btn_replace_one, "Replace", StudioUi.Card2, StudioUi.Text, 34))
            addView(actionButton(R.id.btn_replace_all, "All", StudioUi.Card2, StudioUi.Text, 34))
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(38)))
    }
}

private fun Context.buildEditorOverlay(): View {
    return FrameLayout(this).apply {
        id = R.id.editor_overlay
        setBackgroundColor(StudioUi.palette.editorBg)
        addView(HorizontalScrollView(this@buildEditorOverlay).apply {
            id = R.id.editor_host
            isFillViewport = true
            isHorizontalScrollBarEnabled = true
            setBackgroundColor(StudioUi.palette.editorBg)
            addView(ScrollView(this@buildEditorOverlay).apply {
                isFillViewport = true
                addView(LineNumberEditText(this@buildEditorOverlay).apply {
                    id = R.id.code_editor
                    this.textSize = 12.5f
                    this.setLineSpacing(0f, 1.2f)
                    this.letterSpacing = 0.02f
                    this.typeface = Typeface.create("monospace", Typeface.NORMAL)
                    this.setTextColor(StudioUi.palette.editorText)
                    setHintTextColor(StudioUi.palette.muted)
                    gravity = Gravity.TOP or Gravity.START
                    inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
                    imeOptions = EditorInfo.IME_FLAG_NO_FULLSCREEN
                    setPadding(dp(8), dp(8), dp(12), dp(120))
                    setBackgroundColor(StudioUi.palette.editorBg)
                    setHorizontallyScrolling(true)
                    isHorizontalScrollBarEnabled = true
                    minWidth = resources.displayMetrics.widthPixels
                    layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                })
            }, ViewGroup.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
    }
}

private fun Context.buildQuickBar(): View {
    return HorizontalScrollView(this).apply {
        id = R.id.quick_bar_scroll
        isHorizontalScrollBarEnabled = false
        overScrollMode = View.OVER_SCROLL_NEVER
        setBackgroundColor(c(StudioUi.LayerToolbar))
        visibility = View.GONE
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(38))
        addView(LinearLayout(this@buildQuickBar).apply {
            id = R.id.quick_bar_container
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(4), dp(2), dp(4), dp(2))
            // Symbol buttons
            val symbols = listOf(
                "Tab" to "\t",
                "{" to "{}",
                "(" to "()",
                "[" to "[]",
                "=" to "=",
                ":" to ":",
                "." to ".",
                "/" to "/",
                "->" to "\u2192",
                "_" to "_",
                ";" to ";",
                "!" to "!"
            )
            symbols.forEach { (label, insert) ->
                addView(TextView(this@buildQuickBar).apply {
                    text = label
                    textSize = 12f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c(StudioUi.Text))
                    gravity = Gravity.CENTER
                    setPadding(dp(10), dp(6), dp(10), dp(6))
                    minimumWidth = dp(36)
                    background = StudioUi.ripple(this@buildQuickBar, StudioUi.rounded(this@buildQuickBar, fill = StudioUi.Card, stroke = StudioUi.Divider, radiusDp = 6))
                    StudioUi.touchLift(this)
                    setOnClickListener {
                        val editor = (context as? MainActivity)?.codeEditor ?: return@setOnClickListener
                        val start = editor.selectionStart.coerceAtLeast(0)
                        editor.text?.insert(start, insert)
                    }
                })
            }
            // Separator
            addView(View(this@buildQuickBar).apply {
                setBackgroundColor(c(StudioUi.Divider))
                layoutParams = LinearLayout.LayoutParams(dp(1), dp(24)).apply { marginStart = dp(4); marginEnd = dp(4) }
            })
            // Action buttons (Undo, Redo, Format, Comment, Save, etc.)
            val actions = listOf(
                "\u21A9" to { (context as? MainActivity)?.codeEditor?.undo() },         // Undo
                "\u21AA" to { (context as? MainActivity)?.codeEditor?.redo() },         // Redo
                "\u2261" to { (context as? MainActivity)?.formatCurrentFile() },        // Format
                "\u2044" to { (context as? MainActivity)?.handleEditorToolbarAction("__TOGGLE_COMMENT__") }, // Comment
                "\u2398" to { (context as? MainActivity)?.handleEditorToolbarAction("__DUP_LINE__") },       // Duplicate line
                "\u2191" to { (context as? MainActivity)?.handleEditorToolbarAction("__MOVE_LINE_UP__") },   // Move up
                "\u2193" to { (context as? MainActivity)?.handleEditorToolbarAction("__MOVE_LINE_DOWN__") }, // Move down
                "\u2318" to { (context as? MainActivity)?.toggleFindReplace() },        // Find
                "\u2637" to { (context as? MainActivity)?.showGoToLineDialog() }        // Go to Line
            )
            actions.forEach { (label, action) ->
                addView(TextView(this@buildQuickBar).apply {
                    text = label
                    textSize = 12f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c(StudioUi.TextSoft))
                    gravity = Gravity.CENTER
                    setPadding(dp(10), dp(6), dp(10), dp(6))
                    minimumWidth = dp(36)
                    background = StudioUi.ripple(this@buildQuickBar, StudioUi.rounded(this@buildQuickBar, fill = StudioUi.Surface2, stroke = StudioUi.Divider, radiusDp = 6))
                    StudioUi.touchLift(this)
                    setOnClickListener { action() }
                })
            }
        })
    }
}

private fun Context.buildTerminalKeyBar(): View {
    return HorizontalScrollView(this).apply {
        id = R.id.terminal_key_bar
        isHorizontalScrollBarEnabled = false
        overScrollMode = View.OVER_SCROLL_NEVER
        setBackgroundColor(Color.parseColor("#141A24"))
        visibility = View.GONE
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
        addView(LinearLayout(this@buildTerminalKeyBar).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), dp(4), dp(6), dp(4))
            addView(terminalKeyButton(R.id.btn_terminal_ctrl, "Ctrl"))
            addView(terminalKeyButton(R.id.btn_terminal_esc, "Esc"))
            addView(terminalKeyButton(R.id.btn_terminal_tab, "Tab"))
            addView(terminalKeyButton(R.id.btn_terminal_up, "\u2191"))
            addView(terminalKeyButton(R.id.btn_terminal_down, "\u2193"))
            addView(terminalKeyButton(R.id.btn_terminal_left, "\u2190"))
            addView(terminalKeyButton(R.id.btn_terminal_right, "\u2192"))
            addView(terminalKeyButton(R.id.btn_terminal_enter, "Enter"))
            addView(terminalKeyButton(R.id.btn_terminal_backspace, "\u232B"))
        })
    }
}

private fun Context.terminalKeyButton(viewId: Int, label: String): TextView {
    return TextView(this).apply {
        id = viewId
        text = label
        textSize = 12f
        typeface = Typeface.MONOSPACE
        setTextColor(Color.parseColor("#8E99AC"))
        gravity = Gravity.CENTER
        setPadding(dp(10), dp(6), dp(10), dp(6))
        minWidth = dp(38)
        background = GradientDrawable().apply {
            setColor(Color.TRANSPARENT)
            cornerRadius = dp(6).toFloat()
            setStroke(1, Color.parseColor("#333333"))
        }
    }
}

private fun Context.buildBottomPanel(): View {
    val root = LinearLayout(this).apply {
        id = R.id.bottom_panel
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(c(StudioUi.LayerBottomPanel))
        visibility = View.GONE
        addView(buildPanelTabs())
        // Resize handle for dynamic panel height
        val resizeHandle = View(this@buildBottomPanel).apply {
            id = R.id.panel_resize_handle
            layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(9))
            setBackgroundColor(c(StudioUi.DividerLight))
        }
        addView(resizeHandle)
        resizeHandle.setOnTouchListener { v, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    val startY = event.rawY
                    val rootView = v.parent as LinearLayout
                    val startHeight = rootView.height
                    v.setOnTouchListener { _, e ->
                        when (e.actionMasked) {
                            MotionEvent.ACTION_MOVE -> {
                                val deltaY = (startY - e.rawY).toInt()
                                val newHeightDp = ((startHeight + deltaY) / resources.displayMetrics.density).toInt().coerceIn(300, 800)
                                rootView.layoutParams = rootView.layoutParams.apply { height = dp(newHeightDp) }
                                true
                            }
                            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                                v.setOnTouchListener(null)
                                true
                            }
                            else -> false
                        }
                    }
                    true
                }
                else -> false
            }
        }
    }
    val panelContent = FrameLayout(this)
    panelContent.apply {
            addView(LinearLayout(this@buildBottomPanel).apply {
                id = R.id.build_output_container
                orientation = LinearLayout.VERTICAL

                // Top: Build Monitor (task chips)
                addView(BuildMonitorView(this@buildBottomPanel).apply {
                    id = R.id.build_monitor
                    visibility = View.VISIBLE
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

                // Middle: Tree View
                addView(com.aide.mobilestudio.BuildOutputTreeView(this@buildBottomPanel).apply {
                    id = R.id.build_tree_view
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

                // Divider
                addView(View(this@buildBottomPanel).apply {
                    setBackgroundColor(c("#2F3B4D"))
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(1)))

                // Bottom: Raw Log Output
                addView(BuildOutputView(this@buildBottomPanel).apply {
                    id = R.id.build_output
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.5f))

            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            addView(LinearLayout(this@buildBottomPanel).apply {
                id = R.id.logcat_container
                visibility = View.GONE
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(c(StudioUi.LayerBottomPanel))
                addView(AutoScrollLogView(this@buildBottomPanel).apply {
                    id = R.id.logcat_text
                    textSize = 10.5f
                    typeface = Typeface.MONOSPACE
                    setTextColor(c("#8E99AC"))
                    setPadding(dp(8), dp(8), dp(8), dp(8))
                    setTextIsSelectable(false)
                    movementMethod = android.text.method.ScrollingMovementMethod()
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            addView(LinearLayout(this@buildBottomPanel).apply {
                id = R.id.problems_container
                visibility = View.GONE
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(c(StudioUi.LayerBottomPanel))
                
                // Problems header with tabs
                addView(LinearLayout(this@buildBottomPanel).apply {
                    id = R.id.problems_tabs
                    orientation = LinearLayout.HORIZONTAL
                    setPadding(dp(12), dp(8), dp(12), dp(8))
                    gravity = Gravity.CENTER_VERTICAL
                    
                    addView(TextView(this@buildBottomPanel).apply {
                        id = R.id.problems_tab_all
                        text = "Problems"
                        textSize = 12f
                        setTextColor(c(StudioUi.Text))
                        setPadding(dp(12), dp(4), dp(12), dp(4))
                        background = StudioUi.rounded(this@buildBottomPanel, fill = StudioUi.Surface2, radiusDp = StudioUi.RadiusMd)
                    })
                    
                    addView(TextView(this@buildBottomPanel).apply {
                        id = R.id.problems_tab_file
                        text = "File"
                        textSize = 12f
                        setTextColor(c(StudioUi.TextSoft))
                        setPadding(dp(12), dp(4), dp(12), dp(4))
                        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                            marginStart = dp(8)
                        }
                    })
                    
                    addView(TextView(this@buildBottomPanel).apply {
                        id = R.id.problems_tab_project
                        text = "Project Errors"
                        textSize = 12f
                        setTextColor(c(StudioUi.TextSoft))
                        setPadding(dp(12), dp(4), dp(12), dp(4))
                        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                            marginStart = dp(8)
                        }
                    })
                })
                
                addView(RecyclerView(this@buildBottomPanel).apply {
                    id = R.id.problems_recycler
                    setBackgroundColor(c(StudioUi.LayerBottomPanel))
                }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            addView(LinearLayout(this@buildBottomPanel).apply {
                id = R.id.search_panel
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                setBackgroundColor(c(StudioUi.LayerBottomPanel))
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
addView(LinearLayout(this@buildBottomPanel).apply {
                id = R.id.shell_panel
                orientation = LinearLayout.VERTICAL
                visibility = View.GONE
                setBackgroundColor(c(StudioUi.LayerBottomPanel))
                
                // Shell toolbar
                addView(LinearLayout(this@buildBottomPanel).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(8), dp(4), dp(8), dp(4))
                    setBackgroundColor(c(StudioUi.LayerTabs))
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(36))
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        text = "Terminal"
                        setTextColor(Color.parseColor("#CCCCCC"))
                        textSize = 12f
                        typeface = android.graphics.Typeface.DEFAULT_BOLD
                        layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    })
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        id = R.id.btn_terminal_copy
                        text = AppLanguage.literal(this@buildBottomPanel, "Copy")
                        setTextColor(Color.parseColor("#8E99AC"))
                        textSize = 11f
                        setPadding(dp(8), dp(2), dp(8), dp(2))
                        background = android.graphics.drawable.GradientDrawable().apply {
                            setColor(Color.TRANSPARENT)
                            cornerRadius = dp(4).toFloat()
                        }
                    })
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        id = R.id.btn_terminal_paste
                        text = AppLanguage.literal(this@buildBottomPanel, "Paste")
                        setTextColor(Color.parseColor("#8E99AC"))
                        textSize = 11f
                        setPadding(dp(8), dp(2), dp(8), dp(2))
                        background = android.graphics.drawable.GradientDrawable().apply {
                            setColor(Color.TRANSPARENT)
                            cornerRadius = dp(4).toFloat()
                        }
                    })
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        id = R.id.btn_shell_clear
                        text = "Clear"
                        setTextColor(Color.parseColor("#8E99AC"))
                        textSize = 11f
                        setPadding(dp(8), dp(2), dp(8), dp(2))
                        background = android.graphics.drawable.GradientDrawable().apply {
                            setColor(Color.TRANSPARENT)
                            cornerRadius = dp(4).toFloat()
                        }
                        setOnClickListener {
                            root.findViewById<TextView>(R.id.shell_output_text)?.text = ""
                        }
                    })
                })

                // Tabbed PTY terminal (created lazily on first switch to the Shell panel)
                addView(LinearLayout(this@buildBottomPanel).apply {
                    id = R.id.terminal_tab_bar
                    orientation = LinearLayout.HORIZONTAL
                    visibility = View.GONE
                    setBackgroundColor(c(StudioUi.LayerTabs))
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                })
                addView(android.widget.FrameLayout(this@buildBottomPanel).apply {
                    id = R.id.terminal_container
                    visibility = View.GONE
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
                })

                // Termux-style extra key row (Ctrl modifier + common keys)
                addView(buildTerminalKeyBar())

                // Output area (like Build log) - legacy fallback shell UI
                addView(android.widget.ScrollView(this@buildBottomPanel).apply {
                    id = R.id.shell_output_scroll
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f)
                    setBackgroundColor(Color.parseColor("#1A1A1A"))
                    setPadding(dp(8), dp(4), dp(8), dp(4))
                    isVerticalFadingEdgeEnabled = true
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        id = R.id.shell_output_text
                        typeface = android.graphics.Typeface.MONOSPACE
                        textSize = 13f
                        setTextColor(Color.parseColor("#D4D4D4"))
                        setTextIsSelectable(true)
                        setLineSpacing(0f, 1.1f)
                        layoutParams = android.widget.FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                    })
                })
                
                // Input area (Termux-style: prompt inline, Enter sends, no send button)
                addView(LinearLayout(this@buildBottomPanel).apply {
                    id = R.id.shell_input_row
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                    setPadding(dp(8), dp(4), dp(8), dp(4))
                    setBackgroundColor(Color.parseColor("#1A1A1A"))
                    layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                    addView(android.widget.TextView(this@buildBottomPanel).apply {
                        text = "$"
                        setTextColor(Color.parseColor("#50FA7B"))
                        textSize = 14f
                        typeface = android.graphics.Typeface.MONOSPACE
                        setPadding(dp(4), 0, dp(4), 0)
                    })
                    addView(android.widget.EditText(this@buildBottomPanel).apply {
                        id = R.id.shell_input
                        typeface = android.graphics.Typeface.MONOSPACE
                        textSize = 14f
                        setTextColor(Color.parseColor("#D4D4D4"))
                        setHintTextColor(Color.parseColor("#555555"))
                        hint = "명령어 입력..."
                        isSingleLine = true
                        setPadding(dp(2), dp(6), dp(8), dp(6))
                        background = null
                        layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                        imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_SEND
                        setOnEditorActionListener { v, actionId, _ ->
                            if (actionId == android.view.inputmethod.EditorInfo.IME_ACTION_SEND) {
                                performShellInput(v.rootView.findViewById(R.id.shell_panel))
                                true
                            } else false
                        }
                        setOnKeyListener { v, keyCode, event ->
                            if (event.action == android.view.KeyEvent.ACTION_DOWN && keyCode == android.view.KeyEvent.KEYCODE_ENTER) {
                                performShellInput(v.rootView.findViewById(R.id.shell_panel))
                                true
                            } else false
                        }
                    })
                })
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
            addView(com.aide.mobilestudio.ai.ui.AiAgentPanel(this@buildBottomPanel).apply {
                id = R.id.ai_agent_panel
                visibility = View.GONE
            }, FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
    root.addView(panelContent, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
    // Dynamic height: 50% of screen height, clamped between 300dp and 800dp
    val screenHeightDp = resources.displayMetrics.heightPixels / resources.displayMetrics.density
    val panelHeightDp = (screenHeightDp * 0.5f).toInt().coerceIn(dp(300), dp(800))
    root.layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(panelHeightDp))
    return root
}

private fun Context.performShellInput(root: View) {
    val input = root.findViewById<EditText>(R.id.shell_input) ?: return
    val outputText = root.findViewById<TextView>(R.id.shell_output_text) ?: return
    val scrollView = root.findViewById<ScrollView>(R.id.shell_output_scroll) ?: return
    val cmd = input.text.toString().trim()
    if (cmd.isEmpty()) return

    outputText.append("\n$ $cmd\n")
    input.text.clear()

    Thread {
        try {
            val cwd = com.aide.mobilestudio.StorageHelper.projectsRoot(this@performShellInput)

            // Native ELF tool auto-detection: clang/ninja/make 등 모든 네이티브 실행파일
            val tokens = cmd.split("\\s+".toRegex())
            val firstToken = tokens.firstOrNull()?.removeSuffix(".exe") ?: ""
            val nativeBinary = if (firstToken in NATIVE_TOOL_NAMES) resolveNativeToolBinary(firstToken) else null

            if (nativeBinary != null) {
                // Check ELF architecture before running
                val arch = readElfArch(File(nativeBinary))
                if (arch == "arm64-v8a") {
                    // Run Android ELF directly via /system/bin/linker64 (bypasses noexec)
                    val args = tokens.drop(1)
                    val pb = ProcessBuilder(listOf("/system/bin/linker64", nativeBinary) + args)
                    pb.directory(cwd)
                    val toolDir = File(nativeBinary).parentFile
                    val libDirs = buildList {
                        toolDir?.let {
                            add(it.absolutePath)
                            File(it.parentFile, "lib").takeIf { l -> l.isDirectory }?.let { l -> add(l.absolutePath) }
                            File(it.parentFile, "lib64").takeIf { l -> l.isDirectory }?.let { l -> add(l.absolutePath) }
                            File(it.parentFile, "usr/lib").takeIf { l -> l.isDirectory }?.let { l -> add(l.absolutePath) }
                            File(it.parentFile, "usr/lib64").takeIf { l -> l.isDirectory }?.let { l -> add(l.absolutePath) }
                        }
                        File(filesDir, "toolchain/runtime-tools/lib").takeIf { l -> l.isDirectory }?.let { add(it.absolutePath) }
                        add(applicationInfo.nativeLibraryDir)
                        add("/system/lib64")
                        add("/system/lib")
                    }.distinct().joinToString(":")
                    pb.environment().putAll(mapOf(
                        "LD_LIBRARY_PATH" to libDirs,
                        "HOME" to filesDir.absolutePath,
                        "TERM" to "xterm-256color",
                        "LANG" to "en_US.UTF-8",
                        "TMPDIR" to File(cacheDir, "shell_tmp").apply { mkdirs() }.absolutePath
                    ))
                    File(applicationInfo.nativeLibraryDir, "libexec_redirect.so").takeIf { it.exists() }?.let {
                        pb.environment()["LD_PRELOAD"] = it.absolutePath
                    }
                    pb.redirectErrorStream(true)
                    runShellProcess(pb, outputText, scrollView)
                } else {
                    android.os.Handler(android.os.Looper.getMainLooper()).post {
                        outputText.append("ERROR: architecture not supported: your cpu architecture is arm64(-v8a) but the file you are trying to run is $arch\n")
                        outputText.append("(${nativeBinary})\n")
                    }
                }
                return@Thread
            }

            // Regular shell command with full integrated environment
            val pb = ProcessBuilder("/system/bin/sh", "-c", cmd)
            pb.directory(cwd)
            pb.environment().putAll(buildFullShellEnv())
            pb.redirectErrorStream(true)
            runShellProcess(pb, outputText, scrollView)
        } catch (e: Exception) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                outputText.append("Error: ${e.message}\n")
                scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
            }
        }
    }.start()
}

private fun Context.runShellProcess(pb: ProcessBuilder, outputText: TextView, scrollView: ScrollView) {
    val proc = pb.start()
    val reader = proc.inputStream.bufferedReader()
    val lineBuf = StringBuilder()
    var overwrite = false
    var inEscape = false
    var escapeType = 0
    var ch: Int
    val handler = android.os.Handler(android.os.Looper.getMainLooper())

    fun emit(line: String, over: Boolean) {
        handler.post {
            if (over) {
                // Carriage return: replace the last output line in place (spinner updates)
                val t = outputText.text
                val lastNewline = t.lastIndexOf('\n')
                outputText.text = if (lastNewline >= 0)
                    t.subSequence(0, lastNewline + 1).toString() + line + "\n"
                else line + "\n"
            } else {
                outputText.append("$line\n")
            }
            scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
        }
    }

    while (reader.read().also { ch = it } != -1) {
        if (inEscape) {
            // Consume ANSI escape sequence characters (state machine)
            when (escapeType) {
                0 -> when (ch) {
                    '['.code -> escapeType = 1  // CSI: params + final byte
                    ']'.code -> escapeType = 2  // OSC: terminated by BEL or ST
                    else -> inEscape = false    // 2-char escape (ESC=, ESC>)
                }
                1 -> if (ch in 0x40..0x7E) inEscape = false  // CSI final byte
                2 -> when {
                    ch == 0x07 -> inEscape = false              // BEL terminator
                    ch == 0x1B -> escapeType = 3                // ST: ESC \
                    else -> {}
                }
                3 -> if (ch == '\\'.code) inEscape = false
            }
            continue
        }
        when (ch) {
            0x1B -> { inEscape = true; escapeType = 0 }
            '\r'.code -> {
                // CR: next text overwrites current line (TUI spinner behavior)
                if (lineBuf.isNotEmpty()) {
                    emit(lineBuf.toString(), true)
                    lineBuf.clear()
                } else {
                    overwrite = true
                }
            }
            '\n'.code -> {
                val line = lineBuf.toString()
                lineBuf.clear()
                if (line.isNotEmpty()) emit(line, overwrite)
                overwrite = false
            }
            8 -> {
                if (lineBuf.isNotEmpty()) lineBuf.deleteCharAt(lineBuf.length - 1)
            }
            else -> if (ch >= 32 || ch == '\t'.code) lineBuf.append(ch.toChar())
        }
    }
    if (lineBuf.isNotEmpty()) emit(lineBuf.toString(), overwrite)
    val exitCode = proc.waitFor()
    handler.post {
        outputText.append("[exit $exitCode]\n")
        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
    }
}

/** Native tools that are Android ELF executables (auto-detected + run via linker64). */
private val NATIVE_TOOL_NAMES = setOf(
    "clang", "clang++", "gcc", "g++", "cc", "c++",
    "ld", "ld.lld", "ld.gold", "lld", "lld-link", "llvm-ld",
    "ar", "llvm-ar", "nm", "llvm-nm", "objdump", "llvm-objdump",
    "readelf", "llvm-readelf", "strip", "llvm-strip", "ranlib", "llvm-ranlib",
    "make", "cmake", "ninja", "ninja-build", "pkg-config", "pkgconf",
    "addr2line", "llvm-addr2line", "objcopy", "llvm-objcopy", "size", "strings",
    "c++filt", "llvm-cxxfilt", "llvm-size", "llvm-cov", "llvm-profdata",
    "clang-format", "clang-tidy", "clangd", "lldb", "gdb"
)

/** Finds an Android-executable native tool binary by searching NDK/CMake/toolchain dirs. */
private fun Context.resolveNativeToolBinary(toolName: String): String? {
    val roots = mutableListOf<File>()
    // NDK via ToolchainManager
    runCatching {
        com.aide.mobilestudio.ToolchainManager(this@resolveNativeToolBinary)
            .activeDir(com.aide.mobilestudio.ToolchainManager.Kind.NDK)
    }?.getOrNull()?.let { roots.add(it) }
    // Common toolchain dirs
    File(filesDir, "toolchain/ndk").takeIf { it.isDirectory }?.let { roots.add(it) }
    File(filesDir, "toolchain/cmake").takeIf { it.isDirectory }?.let { roots.add(it) }
    File(filesDir, "toolchain/llvm").takeIf { it.isDirectory }?.let { roots.add(it) }
    File(filesDir, "toolchain/runtime-tools").takeIf { it.isDirectory }?.let { roots.add(it) }
    // Native lib dir (bundled tools)
    File(applicationInfo.nativeLibraryDir).takeIf { it.isDirectory }?.let { roots.add(it) }

    val searchSubdirs = listOf(
        "",
        "bin",
        "usr/bin",
        "toolchains/llvm/prebuilt/linux-x86_64/bin",
        "toolchains/llvm/prebuilt/android-aarch64/bin",
        "toolchains/llvm/prebuilt/linux-aarch64/bin"
    )
    for (rootDir in roots) {
        for (sub in searchSubdirs) {
            val f = File(rootDir, if (sub.isEmpty()) toolName else "$sub/$toolName")
            if (f.isFile) return f.absolutePath
        }
    }
    return null
}

/** Reads the ELF header and returns the architecture string (e.g. "arm64-v8a", "x86_64"). */
private fun readElfArch(file: File): String? {
    return try {
        java.io.RandomAccessFile(file, "r").use { raf ->
            if (raf.length() < 20) return null
            val header = ByteArray(20)
            raf.readFully(header)
            // ELF magic: 0x7f 'E' 'L' 'F'
            if (header[0] != 0x7f.toByte() || header[1] != 'E'.code.toByte() ||
                header[2] != 'L'.code.toByte() || header[3] != 'F'.code.toByte()) return null
            val machine = (header[18].toInt() and 0xFF) or ((header[19].toInt() and 0xFF) shl 8)
            when (machine) {
                0xB7 -> "arm64-v8a"   // EM_AARCH64
                0x3E -> "x86_64"      // EM_X86_64
                0x03 -> "x86"         // EM_386
                0x28 -> "arm32"       // EM_ARM
                0xF3 -> "riscv64"     // EM_RISCV
                0x8E -> "mips64"      // EM_MIPS64... (rare)
                else -> "unknown(machine=0x" + Integer.toHexString(machine) + ")"
            }
        }
    } catch (_: Exception) {
        null
    }
}

/** Full integrated shell environment (PATH with toolchains, LD_LIBRARY_PATH, LD_PRELOAD). */
private fun Context.buildFullShellEnv(): Map<String, String> {
    val tm = runCatching { com.aide.mobilestudio.ToolchainManager(this@buildFullShellEnv) }.getOrNull()
    val jdk = runCatching { tm?.activeJdkHome() }.getOrNull()
    val sdk = runCatching { tm?.activeDir(com.aide.mobilestudio.ToolchainManager.Kind.SDK) }.getOrNull()
    val ndk = runCatching { tm?.activeDir(com.aide.mobilestudio.ToolchainManager.Kind.NDK) }.getOrNull()
    val cmake = runCatching { tm?.activeDir(com.aide.mobilestudio.ToolchainManager.Kind.CMAKE) }.getOrNull()
    val gradle = runCatching { tm?.activeDir(com.aide.mobilestudio.ToolchainManager.Kind.GRADLE) }.getOrNull()
    val python = runCatching { tm?.activeDir(com.aide.mobilestudio.ToolchainManager.Kind.PYTHON) }.getOrNull()
    val nativeLibDir = applicationInfo.nativeLibraryDir
    val runtimeToolsBin = File(filesDir, "toolchain/runtime-tools/bin").takeIf { it.isDirectory }
    val runtimeToolsLib = File(filesDir, "toolchain/runtime-tools/lib").takeIf { it.isDirectory }
    val tmp = File(cacheDir, "shell_tmp").apply { mkdirs() }

    val pathParts = buildList {
        add("/system/bin")
        add("/system/xbin")
        runtimeToolsBin?.let { add(it.absolutePath) }
        jdk?.let { add(File(it, "bin").absolutePath) }
        gradle?.let { add(File(it, "bin").absolutePath) }
        sdk?.let {
            add(File(it, "platform-tools").absolutePath)
            add(File(it, "cmdline-tools/latest/bin").absolutePath)
        }
        cmake?.let { add(File(it, "bin").absolutePath) }
        ndk?.let { add(File(it, "toolchains/llvm/prebuilt/linux-x86_64/bin").absolutePath) }
        python?.let {
            add(File(it, "bin").absolutePath)
            add(File(it, "local/bin").absolutePath)
        }
    }.distinct().joinToString(":")

    val ldParts = buildList {
        jdk?.let {
            add(File(it, "lib/server").absolutePath)
            add(File(it, "lib").absolutePath)
            add(File(it, "lib/jli").absolutePath)
        }
        cmake?.let {
            add(File(it, "bin").absolutePath)
            add(File(it, "lib").absolutePath)
        }
        runtimeToolsLib?.let { add(it.absolutePath) }
        add(nativeLibDir)
        add("/system/lib64")
        add("/system/lib")
    }.distinct().joinToString(":")

    return buildMap {
        put("PATH", pathParts)
        put("LD_LIBRARY_PATH", ldParts)
        put("HOME", filesDir.absolutePath)
        put("TMPDIR", tmp.absolutePath)
        put("TERM", "xterm-256color")
        put("LANG", "en_US.UTF-8")
        put("MOBILESTUDIO_NATIVE_LIB_DIR", nativeLibDir)
        put("MOBILESTUDIO_APK_PATH", applicationInfo.sourceDir)
        put("MOBILESTUDIO_CODE_CACHE", codeCacheDir.absolutePath)
        File(nativeLibDir, "libexec_redirect.so").takeIf { it.exists() }?.let {
            put("LD_PRELOAD", it.absolutePath)
        }
        jdk?.let { put("JAVA_HOME", it.absolutePath) }
        sdk?.let {
            put("ANDROID_HOME", it.absolutePath)
            put("ANDROID_SDK_ROOT", it.absolutePath)
        }
        ndk?.let {
            put("ANDROID_NDK_HOME", it.absolutePath)
            put("ANDROID_NDK_ROOT", it.absolutePath)
        }
        gradle?.let { put("GRADLE_HOME", it.absolutePath) }
    }
}

private fun Context.buildPanelTabs(): View {
    return LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setBackgroundColor(c(StudioUi.LayerPanelTabs))
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(38))
        addView(HorizontalScrollView(this@buildPanelTabs).apply {
            isHorizontalScrollBarEnabled = false
            overScrollMode = View.OVER_SCROLL_NEVER
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f)
            addView(LinearLayout(this@buildPanelTabs).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(dp(4), 0, dp(4), 0)
                addView(panelTab(R.id.tab_build, "\u25B6 Build", selected = true, iconColor = StudioUi.ActiveGreen))
                addView(panelTab(R.id.tab_logcat, "\uD83D\uDC31 Logcat", iconColor = StudioUi.ActiveAmber))
                addView(panelTab(R.id.tab_problems, "\u2716 Problems", iconColor = StudioUi.ActiveRed))
                addView(panelTab(R.id.tab_search, "\uD83D\uDD0D Search", iconColor = StudioUi.ActiveBlue))
                addView(panelTab(R.id.tab_shell, ">_ Shell", iconColor = StudioUi.ActivePurple))
                addView(panelTab(R.id.tab_ai, "\u2728 AI", selected = false, iconColor = StudioUi.ActiveBlue))
            })
        })
        // Build progress indicator
        addView(ProgressBar(this@buildPanelTabs).apply {
            id = R.id.build_progress_mini
            visibility = View.GONE
            layoutParams = LinearLayout.LayoutParams(dp(18), dp(18))
        })
        // Clear
        addView(iconButton(R.id.btn_clear_panel, android.R.drawable.ic_menu_delete, StudioUi.InactiveIcon, "Clear", 30))
        // Hide
        addView(iconButton(R.id.btn_hide_panel, android.R.drawable.arrow_up_float, StudioUi.InactiveIcon, "Hide", 30))

        // Long-press drag to resize panel
        var dragStartY = 0f
        var dragStartHeight = 0
        setOnLongClickListener { true }
        setOnTouchListener { v, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    dragStartY = event.rawY
                    dragStartHeight = (v.parent as? LinearLayout)?.let { parent ->
                        // Find the bottom_panel in the hierarchy
                        var p: android.view.ViewParent? = v.parent
                        while (p != null && p !is android.widget.FrameLayout) p = p.parent
                        (p as? android.widget.FrameLayout)?.layoutParams?.height ?: 0
                    } ?: 0
                    false // let long-press propagate
                }
                MotionEvent.ACTION_MOVE -> {
                    val deltaY = (dragStartY - event.rawY).toInt()
                    val newHeight = (dragStartHeight + deltaY).coerceIn(dp(120), dp(600))
                    // Find and resize the bottom_panel
                    var p: android.view.ViewParent? = v.parent
                    while (p != null) {
                        if (p is android.widget.FrameLayout && p.id == R.id.bottom_panel) {
                            p.layoutParams = p.layoutParams.apply { height = newHeight }
                            break
                        }
                        p = p.parent
                    }
                    true
                }
                else -> false
            }
        }
    }
}

private fun Context.buildStatusBar(): View {
    return LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        setPadding(dp(8), 0, dp(8), 0)
        setBackgroundColor(c(StudioUi.LayerStatusBar))
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(28))

        // Left: status message
        addView(statusText(R.id.tv_status_left, "Ready"), LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))

        // Separator
        addView(dividerVertical())
        // Language
        addView(statusText(R.id.tv_status_lang, "Kotlin"))

        // Separator
        addView(dividerVertical())
        // Cursor position (clickable -> Go to Line)
        addView(TextView(this@buildStatusBar).apply {
            id = R.id.tv_status_cursor
            text = "Ln 1, Col 1"
            textSize = 10f
            typeface = Typeface.MONOSPACE
            setTextColor(c(StudioUi.TextSoft))
            setPadding(dp(8), 0, dp(8), 0)
            isClickable = true
            setOnClickListener { (context as? MainActivity)?.showGoToLineDialog() }
            StudioUi.touchLift(this)
        })

        // Separator
        addView(dividerVertical())
        // Encoding
        addView(statusText(R.id.tv_status_encoding, "UTF-8"))

        // Memory indicator
        addView(TextView(this@buildStatusBar).apply {
            id = R.id.tv_memory_indicator
            text = ""
            textSize = 9f
            setTextColor(c(StudioUi.InactiveText))
            setPadding(dp(6), 0, dp(4), 0)
            isClickable = true
        })
        // Background task
        addView(TextView(this@buildStatusBar).apply {
            id = R.id.tv_background_task
            text = ""
            textSize = 9f
            setTextColor(c(StudioUi.InactiveText))
            setPadding(dp(4), 0, dp(6), 0)
            isClickable = true
        })

        // Right: error/sync status
        addView(statusText(R.id.tv_status_right, ""))
    }
}

private fun Context.buildDrawerSurface(): View {
    return LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setBackgroundColor(c(StudioUi.Surface))
        addView(LinearLayout(this@buildDrawerSurface).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(8))
            setBackgroundColor(c(StudioUi.Surface))
            addView(TextView(this@buildDrawerSurface).apply {
                id = R.id.tv_drawer_title
                text = "EXPLORER"
                textSize = 10f
                letterSpacing = 0.12f
                typeface = Typeface.MONOSPACE
                setTextColor(c(StudioUi.Muted))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                    bottomMargin = dp(4)
                }
            })
            addView(LinearLayout(this@buildDrawerSurface).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                addView(iconButton(R.id.btn_drawer_new_project, android.R.drawable.ic_menu_add, String.format("#%06X", 0xFFFFFF and StudioUi.palette.accent), "New project", 28))
                addView(iconButton(R.id.btn_new_file, android.R.drawable.ic_menu_edit, String.format("#%06X", 0xFFFFFF and StudioUi.palette.success), "New file", 28))
                addView(iconButton(R.id.btn_new_folder, android.R.drawable.ic_menu_sort_by_size, String.format("#%06X", 0xFFFFFF and StudioUi.palette.warning), "New folder", 28))
                addView(View(this@buildDrawerSurface).apply {
                    layoutParams = LinearLayout.LayoutParams(0, 1, 1f)
                })
                addView(TextView(this@buildDrawerSurface).apply {
                    id = R.id.btn_tutorial
                    text = if (AppLanguage.isKorean(this@buildDrawerSurface)) "튜토리얼" else "Tutorial"
                    textSize = 11f
                    setTextColor(c(StudioUi.Muted))
                    setPadding(dp(8), dp(4), dp(8), dp(4))
                    gravity = Gravity.CENTER
                    background = StudioUi.ripple(this@buildDrawerSurface, StudioUi.rounded(this@buildDrawerSurface, fill = StudioUi.Card, radiusDp = StudioUi.RadiusSm))
                })
            })
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        addView(StudioUi.divider(this@buildDrawerSurface, StudioUi.Divider))
        // Windows Explorer style path bar: Up button + current directory breadcrumb
        addView(LinearLayout(this@buildDrawerSurface).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(4), dp(10), dp(4))
            setBackgroundColor(c(StudioUi.Surface))
            addView(ImageButton(this@buildDrawerSurface).apply {
                id = R.id.btn_drawer_up
                setImageResource(android.R.drawable.ic_menu_upload)
                background = null
                setColorFilter(c(StudioUi.Blue))
                layoutParams = LinearLayout.LayoutParams(dp(26), dp(24))
                setOnClickListener { (context as? MainActivity)?.navigateDrawerUp() }
                contentDescription = "Up"
            })
            addView(TextView(this@buildDrawerSurface).apply {
                id = R.id.tv_drawer_path
                textSize = 11f
                typeface = Typeface.MONOSPACE
                setTextColor(c(StudioUi.TextSoft))
                setSingleLine(true)
                ellipsize = android.text.TextUtils.TruncateAt.START
                setPadding(dp(6), 0, 0, 0)
                setOnClickListener { (context as? MainActivity)?.navigateDrawerUp() }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        })
        addView(StudioUi.thinDivider(this@buildDrawerSurface))
        addView(LinearLayout(this@buildDrawerSurface).apply {
            id = R.id.file_tree_search_container
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(10), dp(4), dp(10), dp(4))
            setBackgroundColor(c(StudioUi.Surface))
            addView(ImageButton(this@buildDrawerSurface).apply {
                setImageResource(android.R.drawable.ic_menu_search)
                background = null
                setColorFilter(c(StudioUi.Muted))
                layoutParams = LinearLayout.LayoutParams(dp(24), dp(24))
            })
            addView(EditText(this@buildDrawerSurface).apply {
                id = R.id.et_file_tree_search
                hint = if (AppLanguage.isKorean(this@buildDrawerSurface)) "파일 검색..." else "Search files..."
                setHintTextColor(c(StudioUi.Muted))
                setTextColor(c(StudioUi.Text))
                textSize = 12f
                background = null
                isSingleLine = true
                inputType = android.text.InputType.TYPE_CLASS_TEXT
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginStart = dp(6)
                }
            })
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(32)))
        addView(StudioUi.divider(this@buildDrawerSurface, StudioUi.Divider))
        addView(RecyclerView(this@buildDrawerSurface).apply {
            id = R.id.file_tree
            setBackgroundColor(c(StudioUi.Surface))
            clipToPadding = false
            overScrollMode = View.OVER_SCROLL_NEVER
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f).apply {
            topMargin = dp(2)
        })
        addView(StudioUi.divider(this@buildDrawerSurface, StudioUi.Divider))
        addView(LinearLayout(this@buildDrawerSurface).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(6), dp(8), dp(6))
            setBackgroundColor(c(StudioUi.Surface))
            addView(TextView(this@buildDrawerSurface).apply {
                id = R.id.btn_settings
                text = if (AppLanguage.isKorean(this@buildDrawerSurface)) "설정" else "Settings"
                textSize = 11f
                setTextColor(c(StudioUi.TextSoft))
                setPadding(dp(14), dp(6), dp(14), dp(6))
                gravity = Gravity.CENTER
                background = StudioUi.ripple(this@buildDrawerSurface, StudioUi.rounded(this@buildDrawerSurface, fill = StudioUi.Card, stroke = StudioUi.Border, radiusDp = StudioUi.RadiusSm))
                layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginEnd = dp(6) }
            })
            addView(TextView(this@buildDrawerSurface).apply {
                id = R.id.btn_theme
                text = if (AppLanguage.isKorean(this@buildDrawerSurface)) "테마" else "Theme"
                textSize = 11f
                setTextColor(c(StudioUi.TextSoft))
                setPadding(dp(14), dp(6), dp(14), dp(6))
                gravity = Gravity.CENTER
                background = StudioUi.ripple(this@buildDrawerSurface, StudioUi.rounded(this@buildDrawerSurface, fill = StudioUi.Card, stroke = StudioUi.Border, radiusDp = StudioUi.RadiusSm))
            })
        })
    }
}

private fun Context.menuText(viewId: Int, label: String): TextView {
    return TextView(this).apply {
        id = viewId
        text = label
        textSize = 12f
        typeface = Typeface.MONOSPACE
        gravity = Gravity.CENTER
        setTextColor(c(StudioUi.Text))
        setPadding(dp(10), 0, dp(10), 0)
        background = StudioUi.ripple(this@menuText, StudioUi.rounded(this@menuText, fill = StudioUi.Card, stroke = StudioUi.Border, radiusDp = StudioUi.RadiusXl))
        StudioUi.touchLift(this)
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT)
    }
}

private fun Context.panelTab(viewId: Int, label: String, selected: Boolean = false, iconColor: String = "#6B7280"): TextView {
    // Icon-only mode for narrow screens (< 400dp width)
    val widthDp = resources.displayMetrics.widthPixels / resources.displayMetrics.density
    val iconOnly = widthDp < 400
    
    val iconMap = mapOf(
        R.id.tab_build to "\u25B6",
        R.id.tab_logcat to "\uD83D\uDC31",
        R.id.tab_problems to "\u2716",
        R.id.tab_search to "\uD83D\uDD0D",
        R.id.tab_shell to ">_",
        R.id.tab_ai to "\u2728"
    )
    
    return TextView(this).apply {
        id = viewId
        text = if (iconOnly) iconMap[viewId] ?: label else AppLanguage.literal(this@panelTab, label)
        textSize = StudioUi.TypeSmall
        gravity = Gravity.CENTER
        setTextColor(c(if (selected) iconColor else StudioUi.InactiveIcon))
        if (selected) {
            background = GradientDrawable().apply {
                setColor(c(StudioUi.LayerPanelTabs))
                cornerRadius = dp(StudioUi.RadiusMd).toFloat()
                setStroke(dp(1), c(iconColor))
            }
        }
        StudioUi.touchLift(this)
        StudioUi.enforceMinTouchTarget(this)
        val pad = if (iconOnly) dp(8) else dp(10)
        setPadding(pad, 0, pad, 0)
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT)
    }
}

private fun Context.statusText(viewId: Int, label: String): TextView {
    return TextView(this).apply {
        id = viewId
        text = label
        textSize = 10f
        typeface = Typeface.MONOSPACE
        setTextColor(c(StudioUi.TextSoft))
        setSingleLine(true)
        ellipsize = android.text.TextUtils.TruncateAt.END
        setPadding(dp(8), 0, dp(8), 0)
    }
}

private fun Context.iconButton(viewId: Int, icon: Int, tint: String, description: String, size: Int): ImageButton {
    return ImageButton(this).apply {
        id = viewId
        setImageResource(icon)
        setColorFilter(c(tint))
        contentDescription = AppLanguage.literal(this@iconButton, description)
        background = StudioUi.ripple(this@iconButton, StudioUi.rounded(this@iconButton, fill = "#00000000", stroke = "#00000000", radiusDp = StudioUi.RadiusXl))
        setPadding(dp(8), dp(8), dp(8), dp(8))
        StudioUi.touchLift(this)
        StudioUi.enforceMinTouchTarget(this)
        layoutParams = LinearLayout.LayoutParams(dp(size), dp(size))
    }
}

private fun Context.hiddenIcon(viewId: Int): ImageButton {
    return ImageButton(this).apply {
        id = viewId
        visibility = View.GONE
        layoutParams = LinearLayout.LayoutParams(0, 0)
    }
}

private fun Context.actionButton(
    viewId: Int,
    label: String,
    bg: String,
    fg: String,
    height: Int = 44
): Button {
    return Button(this).apply {
        id = viewId
        text = AppLanguage.literal(this@actionButton, label)
        textSize = 11f
        studioButton(fill = bg, text = fg, accent = StudioUi.ButtonBorder, radiusDp = StudioUi.RadiusMd)
        StudioUi.enforceMinTouchTarget(this)
        layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(height)).apply {
            marginStart = dp(4)
            marginEnd = dp(4)
        }
    }
}

private fun Context.input(viewId: Int, hintText: String, imeOptions: Int = EditorInfo.IME_ACTION_NONE): EditText {
    return EditText(this).apply {
        id = viewId
        hint = AppLanguage.literal(this@input, hintText)
        textSize = 12f
        typeface = Typeface.MONOSPACE
        setTextColor(c(StudioUi.Text))
        setHintTextColor(c(StudioUi.Muted))
        setSingleLine(true)
        this.imeOptions = imeOptions
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        setPadding(dp(10), 0, dp(10), 0)
        background = StudioUi.rounded(this@input, fill = StudioUi.InputBg, stroke = StudioUi.InputBorder, radiusDp = StudioUi.RadiusMd)
    }
}

private fun Context.checkBox(viewId: Int, label: String): CheckBox {
    return CheckBox(this).apply {
        id = viewId
        text = label
        textSize = 10f
        setTextColor(c("#8E99AC"))
        buttonTintList = ColorStateList.valueOf(c("#4F8EF7"))
        minWidth = 0
        minHeight = 0
        StudioUi.enforceMinTouchTarget(this)
    }
}

private fun Context.dividerVertical(): View {
    return View(this).apply {
        setBackgroundColor(c(StudioUi.Divider))
        layoutParams = LinearLayout.LayoutParams(dp(1), dp(20)).apply {
            marginStart = dp(4)
            marginEnd = dp(4)
        }
    }
}

internal fun Context.dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

fun c(hex: String): Int = Color.parseColor(StudioUi.remapHex(hex))

internal class AutoFollowScrollView(context: Context) : ScrollView(context) {
    var followMode: Boolean = true
        private set

    var onFollowChanged: ((Boolean) -> Unit)? = null

    fun scrollToTopNow() {
        smoothScrollTo(0, 0)
    }

    fun scrollToBottomNow() {
        post { fullScroll(ScrollView.FOCUS_DOWN) }
    }

    fun setFollowMode(enabled: Boolean) {
        if (followMode == enabled) return
        followMode = enabled
        onFollowChanged?.invoke(enabled)
        if (enabled) scrollToBottomNow()
    }

    override fun onScrollChanged(l: Int, t: Int, oldl: Int, oldt: Int) {
        super.onScrollChanged(l, t, oldl, oldt)
        if (t > oldt) return
        val child = getChildAt(0)
        if (child == null) return
        val atBottom = (child.height - scrollY - height) <= 20
        if (followMode && !atBottom) {
            followMode = false
            onFollowChanged?.invoke(false)
        } else if (!followMode && atBottom) {
            followMode = true
            onFollowChanged?.invoke(true)
        }
    }
}