package com.aide.mobilestudio.ai

import java.io.DataInputStream
import java.io.File
import java.io.FileInputStream

/**
 * GgufInfo — minimal GGUF v3 metadata reader used for model status display:
 *   "GPU Layers: 28 / 32" (n_layer), model name, architecture.
 *
 * Reads only the header + metadata and skips tensor data. Values are never
 * fabricated: if a key is absent the field is null.
 */
object GgufInfo {
    private const val GGUF_STRING = 8

    data class Info(
        val generalName: String?,
        val architecture: String?,
        val nLayer: Int?,
        /**
         * The context the model itself was trained for, straight from the file.
         *
         * This is the model's real ceiling, as opposed to whatever window the
         * app decides to load, and the two used to be conflated: every local
         * model was loaded at one hard-coded size regardless of what the file
         * declared it could do.
         */
        val contextLength: Int? = null,
        val headCount: Int? = null,
        val headCountKv: Int? = null,
        val embeddingLength: Int? = null,
        val keyLength: Int? = null,
        val valueLength: Int? = null,
        /**
         * `general.file_type` - the ggml type ordinal the file declares for
         * itself. Used to confirm that the quantisation in the file name is
         * really what is inside; null when the file does not say.
         */
        val fileType: Int? = null,
    )

    fun read(file: File): Info? {
        if (!file.isFile) return null
        return try {
            DataInputStream(FileInputStream(file).buffered()).use { input ->
                val magic = ByteArray(4)
                input.readFully(magic)
                if (!magic.contentEquals(byteArrayOf(0x47, 0x47, 0x55, 0x46))) return null // "GGUF"
                val version = readIntLe(input)
                if (version !in 1..3) return null

                val tensorCount = readLongLe(input)
                val kvCount = readLongLe(input)
                if (tensorCount < 0 || kvCount < 0 || kvCount > 100_000) return null

                var generalName: String? = null
                var architecture: String? = null
                var nLayer: Int? = null
                var contextLength: Int? = null
                var headCount: Int? = null
                var headCountKv: Int? = null
                var embeddingLength: Int? = null
                var keyLength: Int? = null
                var valueLength: Int? = null
                var fileType: Int? = null

                fun positiveInt(value: Any?): Int? = (value as? Number)?.toInt()?.takeIf { it > 0 }

                for (i in 0 until kvCount) {
                    val key = readString(input) ?: return null
                    val type = readIntLe(input)
                    when {
                        key == "general.name" && type == GGUF_STRING -> generalName = readString(input)
                        // The string is consumed by this branch. Reading it a
                        // second time would desynchronise every key after it.
                        key == "general.architecture" && type == GGUF_STRING -> architecture = readString(input)
                        key == "general.file_type" -> fileType = (readValue(input, type) as? Number)?.toInt()
                        // Current llama.cpp writes block_count; older files
                        // used n_layer. Both are read, the value is consumed once.
                        //
                        // The match is on the suffix, not on "$architecture.": the
                        // architecture key is not guaranteed to have been read
                        // before the architecture's own keys, and when it had not,
                        // every one of these came back null - a model reported
                        // "layers: unknown" to the GPU-offload decision.
                        key == "$architecture.block_count" || key == "general.block_count" ||
                            key.endsWith(".block_count") -> {
                            val layers = positiveInt(readValue(input, type))
                            if (nLayer == null) nLayer = layers
                        }
                        key == "$architecture.n_layer" || key == "general.n_layer" ||
                            key.endsWith(".n_layer") ->
                            nLayer = positiveInt(readValue(input, type))
                        key == "$architecture.context_length" || key == "general.context_length" ||
                            key.endsWith(".context_length") ->
                            contextLength = positiveInt(readValue(input, type))
                        key == "$architecture.attention.head_count" || key.endsWith(".attention.head_count") ->
                            headCount = positiveInt(readValue(input, type))
                        key == "$architecture.attention.head_count_kv" || key.endsWith(".attention.head_count_kv") ->
                            headCountKv = positiveInt(readValue(input, type))
                        key == "$architecture.embedding_length" || key.endsWith(".embedding_length") ->
                            embeddingLength = positiveInt(readValue(input, type))
                        key == "$architecture.attention.key_length" || key.endsWith(".attention.key_length") ->
                            keyLength = positiveInt(readValue(input, type))
                        key == "$architecture.attention.value_length" || key.endsWith(".attention.value_length") ->
                            valueLength = positiveInt(readValue(input, type))
                        else -> readValue(input, type)
                    }
                    // Everything worth reading sits in the metadata block ahead
                    // of the tokenizer. Stopping there avoids stepping through a
                    // 150k-entry vocabulary one value at a time, which is what
                    // happens on any file that does not declare a context.
                    if (key.startsWith("tokenizer.") ||
                        (generalName != null && architecture != null && contextLength != null && nLayer != null)
                    ) {
                        break
                    }
                }
                Info(
                    generalName, architecture, nLayer, contextLength,
                    headCount, headCountKv, embeddingLength, keyLength, valueLength, fileType,
                )
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun readIntLe(input: DataInputStream): Int {
        val b = ByteArray(4)
        input.readFully(b)
        return (b[0].toInt() and 0xff) or
            ((b[1].toInt() and 0xff) shl 8) or
            ((b[2].toInt() and 0xff) shl 16) or
            ((b[3].toInt() and 0xff) shl 24)
    }

    private fun readLongLe(input: DataInputStream): Long {
        val lo = readIntLe(input).toLong() and 0xffffffffL
        val hi = readIntLe(input).toLong() and 0xffffffffL
        return lo or (hi shl 32)
    }

    private fun readString(input: DataInputStream): String? {
        val len = readLongLe(input)
        if (len < 0 || len > 10_000_000) return null
        val b = ByteArray(len.toInt())
        input.readFully(b)
        return String(b, Charsets.UTF_8)
    }

    /** Reads and discards one metadata value of the given type. */
    private fun readValue(input: DataInputStream, type: Int): Any? {
        return when (type) {
            0, 1 -> input.readByte().toLong()          // u8 / i8
            2, 3 -> {                                   // u16 / i16
                val b = ByteArray(2); input.readFully(b)
                ((b[0].toInt() and 0xff) or ((b[1].toInt() and 0xff) shl 8)).toShort().toLong()
            }
            4 -> readIntLe(input).toLong()              // u32 / i32
            5, 6 -> Float.fromBits(readIntLe(input)).toDouble() // f32
            7 -> input.readByte() != 0.toByte()         // bool
            8 -> readString(input)                      // string
            9 -> {                                      // array
                val elemType = readIntLe(input)
                val count = readLongLe(input)
                if (count < 0 || count > 10_000_000) return null
                for (i in 0 until count) readValue(input, elemType)
                null
            }
            10, 11 -> readLongLe(input)                 // u64 / i64
            12 -> java.lang.Double.longBitsToDouble(readLongLe(input)) // f64
            else -> null // unknown type — cannot skip reliably
        }
    }
}
