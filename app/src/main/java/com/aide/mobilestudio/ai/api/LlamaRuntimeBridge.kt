package com.aide.mobilestudio.ai.api

import android.util.Log

/**
 * LlamaRuntimeBridge — dlopen-based JNI bridge to the downloaded llama.cpp
 * Android runtime (arm64-v8a, CPU + Vulkan backends).
 *
 * Unlike [LlamaAndroid] (which is statically linked to an app-built, CPU-only
 * llama.cpp), this bridge loads the runtime's own libllama.so/libggml*.so via
 * dlopen from the on-device install directory, so:
 *   - the prebuilt runtime is used (llama.cpp is never recompiled at app build)
 *   - the Vulkan GPU backend is really used (libggml-vulkan.so)
 *   - CPU fallback works on devices without Vulkan (libggml-cpu.so)
 *
 * GPU offload ([loadModelWithFallback]):
 *   CPU <====|=============|===========|===> GPU
 *   n_gpu_layers=0 (CPU 위주) ←→ 가능한 최대 n_gpu_layers (GPU 최대 offload)
 *   모델이 GPU 메모리에 안 들어가면 자동으로 offload를 낮춰 재시도한다.
 *   무한 재시도하지 않는다 (최대 [MAX_FALLBACK_RETRIES]회 후 CPU로 폴백).
 */
class LlamaRuntimeBridge {
    companion object {
        private const val TAG = "LlamaRuntimeBridge"
        private const val MAX_FALLBACK_RETRIES = 4

        /** True when libllama_runtime_jni.so (the bridge, not the runtime) loaded. */
        val bridgeAvailable: Boolean = try {
            System.loadLibrary("llama_runtime_jni")
            true
        } catch (e: Throwable) {
            Log.e(TAG, "libllama_runtime_jni.so load failed", e)
            false
        }

        fun isBridgeAvailable(): Boolean = bridgeAvailable
    }

    data class ModelStats(
        val loadTimeMs: Long,
        val sampleTimeMs: Long,
        val promptEvalTimeMs: Long,
        val evalTimeMs: Long,
        val totalTokens: Int
    )

    data class LoadResult(
        val handle: Long,
        val nGpuLayersUsed: Int,
        val backend: String
    )

    private var handle: Long = 0L
    private var gpuLayersUsed = 0
    private var backendUsed = "cpu"

    // ------------------------------------------------------------------
    // Runtime lifecycle (dlopen of the downloaded runtime)
    // ------------------------------------------------------------------

    /**
     * dlopen()s the runtime libs from [runtimeDir] and resolves the llama.cpp API.
     * [appNativeLibDir] (the app's native lib dir) is used to preload the app's
     * own libc++_shared.so first so DT_NEEDED lookups resolve.
     */
    fun loadRuntime(runtimeDir: String, appNativeLibDir: String? = null): Boolean {
        if (!bridgeAvailable) {
            Log.e(TAG, "bridge not available (libllama_runtime_jni.so)")
            return false
        }
        val ok = nativeLoadRuntime(runtimeDir, appNativeLibDir ?: "")
        if (!ok) {
            Log.e(TAG, "runtime load failed: ${nativeLastError()}")
        }
        return ok
    }

    fun isRuntimeLoaded(): Boolean = bridgeAvailable && nativeRuntimeLoaded()
    fun lastError(): String? = if (bridgeAvailable) nativeLastError().takeIf { it != "runtime not loaded" } else "bridge not available"
    fun isVulkanAvailable(): Boolean = bridgeAvailable && nativeIsVulkanAvailable()
    fun gpuName(): String = if (bridgeAvailable) nativeGetGpuName() else "unknown"

    // ------------------------------------------------------------------
    // Model management
    // ------------------------------------------------------------------

    /**
     * Loads a model with automatic GPU offload fallback:
     * load 실패 시 n_gpu_layers를 낮춰 재시도하고, 최대 [MAX_FALLBACK_RETRIES]회
     * 후 CPU(0 layers)로 폴백한다. 무한 재시도하지 않는다.
     */
    fun loadModelWithFallback(
        modelPath: String,
        nContext: Int,
        requestedGpuLayers: Int,
        nThreads: Int,
        vulkanAvailable: Boolean,
        maxLayers: Int = 0,
        onAttempt: ((attempt: Int, nGpuLayers: Int) -> Unit)? = null
    ): LoadResult {
        unload()

        // Vulkan unavailable → CPU only (honest configuration)
        var nGpu = if (vulkanAvailable && requestedGpuLayers != 0) {
            if (maxLayers in 1..Int.MAX_VALUE) {
                if (requestedGpuLayers < 0) maxLayers else requestedGpuLayers.coerceAtMost(maxLayers)
            } else requestedGpuLayers
        } else 0

        var attempt = 0
        while (true) {
            attempt++
            onAttempt?.invoke(attempt, nGpu)
            Log.i(TAG, "Load attempt #$attempt: n_gpu_layers=$nGpu (vulkan=$vulkanAvailable)")
            val h = try {
                nativeLoadModel(modelPath, nContext, nGpu, nThreads)
            } catch (e: Throwable) {
                Log.e(TAG, "nativeLoadModel crashed: ${e.message}", e)
                0L
            }
            if (h != 0L) {
                handle = h
                gpuLayersUsed = nGpu
                backendUsed = if (nGpu == 0) "CPU" else "Vulkan"
                return LoadResult(h, nGpu, backendUsed)
            }
            // Fallback: lower the offload level — never retry forever
            if (nGpu == 0 || attempt >= MAX_FALLBACK_RETRIES) {
                val err = nativeLastError() ?: "unknown"
                Log.e(TAG, "Model load failed after $attempt attempts: $err")
                throw RuntimeException("Model load failed: $err")
            }
            val step = if (maxLayers > 8) maxLayers / 4 else 2
            nGpu = if (nGpu < 0) 0 else (nGpu - step).coerceAtLeast(0)
            Log.w(TAG, "Retrying with n_gpu_layers=$nGpu")
        }
    }

    fun unload() {
        if (handle != 0L) {
            try { nativeUnloadModel(handle) } catch (e: Throwable) { Log.w(TAG, "unload failed: ${e.message}") }
            handle = 0L
        }
    }

    val currentHandle: Long get() = handle
    val currentGpuLayers: Int get() = gpuLayersUsed
    val currentBackend: String get() = backendUsed

    // ------------------------------------------------------------------
    // Generation
    // ------------------------------------------------------------------

    fun generate(
        modelHandle: Long,
        prompt: String,
        temperature: Float = 0.7f,
        topP: Float = 0.95f,
        topK: Int = 40,
        repeatPenalty: Float = 1.1f,
        seed: Int = -1,
        stopSequences: Array<String> = emptyArray(),
        onToken: (String) -> Boolean
    ): String {
        if (modelHandle == 0L) throw IllegalStateException("model is not loaded")
        return nativeGenerate(modelHandle, prompt, temperature, topP, topK, repeatPenalty, seed, stopSequences, onToken)
    }

    // ------------------------------------------------------------------
    // Native methods (implemented in llama_runtime_jni.cpp)
    // ------------------------------------------------------------------

    external fun nativeLoadRuntime(runtimeDir: String, appNativeLibDir: String): Boolean
    external fun nativeRuntimeLoaded(): Boolean
    external fun nativeLastError(): String
    external fun nativeIsVulkanAvailable(): Boolean
    external fun nativeGetGpuName(): String
    external fun nativeLoadModel(modelPath: String, nContext: Int, nGpuLayers: Int, nThreads: Int): Long
    external fun nativeUnloadModel(handle: Long)
    external fun nativeGenerate(
        modelHandle: Long,
        prompt: String,
        temperature: Float,
        topP: Float,
        topK: Int,
        repeatPenalty: Float,
        seed: Int,
        stopSequences: Array<String>,
        onToken: (String) -> Boolean
    ): String
    external fun getStats(modelHandle: Long): ModelStats?
}
