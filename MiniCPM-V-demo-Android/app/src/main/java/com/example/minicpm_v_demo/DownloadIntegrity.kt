package com.example.minicpm_v_demo

import java.io.File
import java.security.MessageDigest

internal object DownloadIntegrity {
    fun safePath(path: String): Boolean = path.isNotBlank() &&
        !path.startsWith("/") && !path.contains('\\') &&
        path.split('/').all { it.isNotBlank() && it != "." && it != ".." }

    fun matchesRange(header: String?, start: Long, size: Long): Boolean {
        val match = Regex("bytes (\\d+)-(\\d+)/(\\d+)").matchEntire(header ?: "") ?: return false
        val first = match.groupValues[1].toLongOrNull() ?: return false
        val last = match.groupValues[2].toLongOrNull() ?: return false
        val total = match.groupValues[3].toLongOrNull() ?: return false
        return first == start && last >= first && last < total && total == size
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(256 * 1024).use { input ->
            val buffer = ByteArray(256 * 1024)
            while (true) {
                val n = input.read(buffer)
                if (n < 0) break
                digest.update(buffer, 0, n)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
