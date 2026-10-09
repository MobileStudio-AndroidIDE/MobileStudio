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
    data class Info(
        val generalName: String?,
        val architecture: String?,
        val nLayer: Int?
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

                for (i in 0 until kvCount) {
                    val key = readString(input) ?: return null
                    val type = readIntLe(input)
                    when {
                        key == "general.name" && type == 8 -> generalName = readString(input)
                        key == "general.architecture" && type == 8 -> {
                            architecture = readString(input)
                            val v = readValue(input, type) // consume
                        }
                        key == "$architecture.n_layer" || key == "general.n_layer" -> {
                            val v = readValue(input, type)
                            nLayer = (v as? Number)?.toInt()
                        }
                        else -> readValue(input, type)
                    }
                    if (nLayer != null && generalName != null) break
                }
                Info(generalName, architecture, nLayer)
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
