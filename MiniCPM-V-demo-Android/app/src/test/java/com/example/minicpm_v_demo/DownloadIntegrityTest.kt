package com.example.minicpm_v_demo

import org.junit.Assert.*
import org.junit.Test
import java.io.File

class DownloadIntegrityTest {
    @Test fun resumeRequiresMatchingOffsetAndPinnedTotal() {
        assertTrue(DownloadIntegrity.matchesRange("bytes 100-999/1000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes 0-999/1000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes 100-999/2000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes 100-1000/1000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes 100-99/1000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes */1000", 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange(null, 100, 1000))
        assertFalse(DownloadIntegrity.matchesRange("bytes 999999999999999999999-999/1000", 100, 1000))
    }

    @Test fun catalogueCannotEscapeTheModelDirectory() {
        assertTrue(DownloadIntegrity.safePath("experts/layer_000.bin"))
        for (path in listOf("../other", "experts/../other", "/absolute", "experts//bad", "a\\b", "")) {
            assertFalse(path, DownloadIntegrity.safePath(path))
        }
    }

    @Test fun digestDetectsCorruptedOrTruncatedPayloads() {
        val file = File.createTempFile("model-integrity", ".bin")
        try {
            file.writeText("abc")
            assertEquals("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", DownloadIntegrity.sha256(file))
            val correct = DownloadIntegrity.sha256(file)
            file.writeText("abd")
            assertNotEquals(correct, DownloadIntegrity.sha256(file))
            file.writeText("ab")
            assertNotEquals(correct, DownloadIntegrity.sha256(file))
        } finally { file.delete() }
    }
}
