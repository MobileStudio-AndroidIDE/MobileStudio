package com.aide.mobilestudio.ai.agent

import android.content.Context
import android.util.Log
import java.io.File

class HardwareManager(private val context: Context) {
    enum class Backend(val label: String) { 
        CPU("CPU (Default)"), 
        GPU_VULKAN("GPU (Vulkan)"), 
        NPU_TPU("NPU/TPU (NNAPI)"),
        AUTO("Auto Optimize")
    }

    fun getAvailableBackends(): List<Backend> {
        val list = mutableListOf(Backend.CPU)
        
        // Vulkan is generally available on Android 7+
        if (checkVulkanSupport()) {
            list.add(Backend.GPU_VULKAN)
        }
        
        // NNAPI is for TPU/NPU acceleration (Pixel, Qualcomm, etc.)
        if (checkNnapiSupport()) {
            list.add(Backend.NPU_TPU)
        }
        
        list.add(Backend.AUTO)
        return list
    }

    private fun checkVulkanSupport(): Boolean {
        // Simple check for Vulkan loader
        return File("/system/lib64/libvulkan.so").exists() || File("/system/lib/libvulkan.so").exists()
    }

    private fun checkNnapiSupport(): Boolean {
        // NNAPI is available on Android 8.1+
        return android.os.Build.VERSION.SDK_INT >= 27
    }

    fun recommendBackend(): Backend {
        // Use GPU by default if available, as it's often more stable for LLMs than NNAPI
        if (checkVulkanSupport()) return Backend.GPU_VULKAN
        return Backend.CPU
    }
    
    fun getOptimalThreadCount(): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        // Use performance-oriented thread count: mobile ARM big.LITTLE benefits
        // from using more than 4 threads for llama.cpp inference (batch matmul
        // scales across cores). Cap at 8 to avoid oversubscription on small cores.
        return if (cores >= 8) 8 else if (cores >= 6) 6 else cores
    }
}
