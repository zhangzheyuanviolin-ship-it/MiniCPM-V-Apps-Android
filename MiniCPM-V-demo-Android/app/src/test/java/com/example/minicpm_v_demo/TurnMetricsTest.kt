package com.example.minicpm_v_demo

import org.junit.Assert.*
import org.junit.Test

class TurnMetricsTest {
    @Test fun ratesUseActualTokenCountsAndSeparateNativeDurations() {
        val turn = TurnMetrics.fromNative(2_500_000_000, longArrayOf(600, 2_000_000_000, 7, 500_000_000), 7_340_032)
        assertEquals(300.0, turn.prefillTokensPerSecond!!, 0.0001)
        assertEquals(14.0, turn.decodeTokensPerSecond!!, 0.0001)
        assertEquals(2_500_000_000L, turn.firstTokenNanos!!)
        assertEquals(7_340_032L, turn.peakPssKiB)
    }

    @Test fun cancelledBeforeFirstTokenDoesNotInventLatencyOrSpeed() {
        val turn = TurnMetrics.fromNative(null, longArrayOf(50, 1_000_000_000, 0, 0), 0)
        assertNull(turn.firstTokenNanos)
        assertNull(turn.decodeTokensPerSecond)
        assertEquals(50.0, turn.prefillTokensPerSecond!!, 0.0001)
    }

    @Test fun failedPrefillAndZeroDurationsStayUnavailable() {
        val turn = TurnMetrics.fromNative(null, longArrayOf(0, 0, 3, 0), 0)
        assertNull(turn.prefillTokensPerSecond)
        assertNull(turn.decodeTokensPerSecond)
    }

    @Test fun rateArithmeticDoesNotOverflowLongNanoseconds() {
        val turn = TurnMetrics.fromNative(0, longArrayOf(10_000_000_000, 2_000_000_000, 1, 1), 1)
        assertEquals(5_000_000_000.0, turn.prefillTokensPerSecond!!, 1.0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsIncompleteNativeCounters() {
        TurnMetrics.fromNative(null, longArrayOf(1, 2), 0)
    }

    @Test(expected = IllegalArgumentException::class)
    fun rejectsNegativeNativeTimings() {
        TurnMetrics.fromNative(null, longArrayOf(1, -1, 1, 1), 0)
    }
}
