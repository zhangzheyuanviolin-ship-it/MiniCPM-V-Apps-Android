package com.example.minicpm_v_demo

import org.junit.Assert.*
import org.junit.Test

class InferenceSettingsTest {
    @Test fun recommendedLongReplyBudgetLeavesRoomForInputAndHistory() {
        val options = InferenceSettings()
        options.validate()
        assertEquals(16384, options.contextTokens)
        assertEquals(8192, options.maxOutputTokens)
        assertTrue(options.maxOutputTokens < options.contextTokens)
    }
    @Test fun acceptsDeclaredModelMaximumWithoutConfusingItWithNewTokens() {
        InferenceSettings(contextTokens = 262144, maxOutputTokens = 262016).validate()
    }
    @Test fun acceptsGreedyAndDisabledOptionalSamplers() {
        InferenceSettings(temperature = 0f, topP = 1f, topK = 0, minP = 0f,
            repeatPenalty = 1f, repeatLastN = 0, seed = 1234).validate()
    }
    @Test fun rejectsUnsafeAndNonfiniteValues() {
        val invalid = listOf(
            InferenceSettings(contextTokens = 4096, maxOutputTokens = 4096),
            InferenceSettings(maxOutputTokens = 0), InferenceSettings(contextTokens = 524288),
            InferenceSettings(temperature = Float.NaN), InferenceSettings(temperature = Float.POSITIVE_INFINITY),
            InferenceSettings(topP = 0f), InferenceSettings(topP = 1.1f), InferenceSettings(topK = -1),
            InferenceSettings(minP = Float.NaN), InferenceSettings(repeatPenalty = 0f),
            InferenceSettings(repeatLastN = 4097), InferenceSettings(seed = -2)
        )
        invalid.forEach { options ->
            try { options.validate(); fail("Accepted invalid settings: $options") }
            catch (_: IllegalArgumentException) { }
        }
    }
    @Test fun stoppedTurnsReportActualCountsAndCause() {
        val turn = TurnMetrics.fromNative(100, longArrayOf(20, 200, 8192, 1000, 1, 8212, 16384, 8192), 100)
        assertEquals(StopReason.OUTPUT_LIMIT, turn.stopReason)
        assertEquals(8192L, turn.generatedTokens)
        assertEquals(8192L, turn.outputLimit)
        assertEquals(8212L, turn.contextUsed)
        assertEquals(16384L, turn.contextCapacity)
    }
    @Test fun naturalEndAndContextFullHaveDifferentCauses() {
        assertEquals(StopReason.END_OF_SEQUENCE, StopReason.fromCode(3))
        assertEquals(StopReason.CONTEXT_LIMIT, StopReason.fromCode(2))
        assertNotEquals(StopReason.OUTPUT_LIMIT, StopReason.fromCode(3))
    }
    @Test(expected = IllegalArgumentException::class)
    fun rejectsPartiallyUpdatedNativeStatusSchema() {
        TurnMetrics.fromNative(null, longArrayOf(1, 2, 3, 4, 5, 6), 0)
    }
}
