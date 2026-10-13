package com.aide.mobilestudio.terminal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regression tests for the reported input bug:
 *
 *   "clear을 실행하고 ls 치고 스페이스하면 clearls처럼 겹쳐져"
 *
 * Every scenario feeds the emulator exactly what a real PTY echoes, then
 * asserts that the line the cursor sits on holds only the freshly typed text.
 */
class TerminalClearRegressionTest {

    private val ESC = ""

    private fun fresh(cols: Int = 30, rows: Int = 6) = TerminalEmulator(cols, rows)

    private fun rowText(em: TerminalEmulator, row: Int): String =
        em.getLine(row).joinToString("") { it.char.toString() }

    private fun cursorLine(em: TerminalEmulator): String = rowText(em, em.getCursorRow()).trimEnd()

    // ------------------------------------------------------------- clear replay

    @Test
    fun `clear then ls then space renders only the new text`() {
        val em = fresh()
        em.processString("hello\r\n")
        em.processString("clear")
        em.processString("\r\n")
        em.processString(ESC + "[H" + ESC + "[2J")
        em.processString("ls")
        em.processString(" ")
        assertEquals("ls", cursorLine(em))
    }

    @Test
    fun `clear echo and its escape in one PTY chunk`() {
        val em = fresh()
        em.processString("hello\r\n")
        em.processString("clear\r\n" + ESC + "[H" + ESC + "[2J")
        em.processString("ls ")
        assertEquals("ls", cursorLine(em))
    }

    @Test
    fun `clear on a full screen does not resurrect the old command`() {
        val em = fresh()
        for (i in 0 until 10) em.processString("line$i\r\n")
        em.processString("clear\r\n" + ESC + "[H" + ESC + "[2J")
        em.processString("ls ")
        assertEquals("ls", cursorLine(em))
    }

    @Test
    fun `clear escape split across PTY chunks`() {
        val em = fresh()
        em.processString("hello\r\n")
        em.processString("clear\r\n" + ESC)
        em.processString("[H" + ESC)
        em.processString("[2J")
        em.processString("ls ")
        assertEquals("ls", cursorLine(em))
    }

    @Test
    fun `ESC 3J variant`() {
        val em = fresh()
        for (i in 0 until 10) em.processString("line$i\r\n")
        em.processString("clear" + ESC + "[3J")
        em.processString("ls ")
        assertEquals("ls", cursorLine(em))
    }

    // ------------------------------------------------- scrollback after a clear

    @Test
    fun `full screen erase drops the scrollback`() {
        val em = fresh()
        for (i in 0 until 10) em.processString("line$i\r\n")
        assertTrue(em.getScrollBackSize() > 0)
        em.processString("clear\r\n" + ESC + "[H" + ESC + "[2J")
        assertEquals(
            "`clear` must leave an empty screen, not a blank screen with the " +
                "previous command sitting right above the prompt",
            0,
            em.getScrollBackSize(),
        )
    }

    @Test
    fun `line erase does not touch the scrollback`() {
        val em = fresh()
        for (i in 0 until 10) em.processString("line$i\r\n")
        val before = em.getScrollBackSize()
        em.processString(ESC + "[2K")
        assertEquals(before, em.getScrollBackSize())
    }

    // ------------------------------------------------- IME delta clamping

    @Test
    fun `backspaces are clamped to what the shell line holds`() {
        // After Enter or clear the shell line is empty: no backspace can be
        // honoured, so a stale IME model must not emit any.
        assertEquals(0, boundedBackspaces(requested = 5, shellLineLength = 0))
        assertEquals(2, boundedBackspaces(requested = 5, shellLineLength = 2))
        assertEquals(5, boundedBackspaces(requested = 5, shellLineLength = 9))
        assertEquals(0, boundedBackspaces(requested = 0, shellLineLength = 0))
        // Defensive: a negative request erases nothing.
        assertEquals(0, boundedBackspaces(requested = -3, shellLineLength = 4))
    }
}
