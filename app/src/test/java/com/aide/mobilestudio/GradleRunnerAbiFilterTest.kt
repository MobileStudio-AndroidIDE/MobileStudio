package com.aide.mobilestudio

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class GradleRunnerAbiFilterTest {

    @Test
    fun arm64HostMapsToArm64V8a() {
        assertEquals("arm64-v8a", GradleRunner.abiFilterFor("aarch64"))
        assertEquals("arm64-v8a", GradleRunner.abiFilterFor("arm64"))
        assertEquals("arm64-v8a", GradleRunner.abiFilterFor("AARCH64"))
    }

    @Test
    fun x86HostsMapToX86Abis() {
        assertEquals("x86_64", GradleRunner.abiFilterFor("x86_64"))
        assertEquals("x86_64", GradleRunner.abiFilterFor("amd64"))
        assertEquals("x86", GradleRunner.abiFilterFor("x86"))
        assertEquals("x86", GradleRunner.abiFilterFor("i686"))
    }

    @Test
    fun unknownArchInjectsNothing() {
        assertNull(GradleRunner.abiFilterFor("riscv64"))
        assertNull(GradleRunner.abiFilterFor(null))
    }
}
