package com.example.minicpm_v_demo

/** Counts are native token IDs, not UTF-8 chunks or displayed characters. */
data class TurnMetrics(
    val firstTokenNanos: Long?,
    val promptTokens: Long,
    val prefillNanos: Long,
    val generatedTokens: Long,
    val decodeNanos: Long,
    val peakPssKiB: Long
) {
    val prefillTokensPerSecond: Double? get() = rate(promptTokens, prefillNanos)
    val decodeTokensPerSecond: Double? get() = rate(generatedTokens, decodeNanos)

    private fun rate(tokens: Long, nanos: Long): Double? =
        if (tokens > 0 && nanos > 0) tokens.toDouble() * 1_000_000_000.0 / nanos else null

    companion object {
        fun fromNative(firstTokenNanos: Long?, values: LongArray, peakPssKiB: Long): TurnMetrics {
            require(values.size == 4) { "Invalid native turn statistics" }
            require(values.all { it >= 0 } && peakPssKiB >= 0)
            require(firstTokenNanos == null || firstTokenNanos >= 0)
            return TurnMetrics(firstTokenNanos, values[0], values[1], values[2], values[3], peakPssKiB)
        }
    }
}
