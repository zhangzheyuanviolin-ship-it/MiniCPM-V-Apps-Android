package com.example.minicpm_v_demo

enum class StopReason(val code: Int) {
    UNKNOWN(0), OUTPUT_LIMIT(1), CONTEXT_LIMIT(2), END_OF_SEQUENCE(3), ERROR(4), USER_STOP(5);
    companion object { fun fromCode(code: Long) = values().firstOrNull { it.code.toLong() == code } ?: UNKNOWN }
}

/** Counts are native token IDs, not UTF-8 chunks or displayed characters. */
data class TurnMetrics(
    val firstTokenNanos: Long?, val promptTokens: Long, val prefillNanos: Long,
    val generatedTokens: Long, val decodeNanos: Long, val peakPssKiB: Long,
    val stopReason: StopReason = StopReason.UNKNOWN,
    val contextUsed: Long = 0, val contextCapacity: Long = 0, val outputLimit: Long = 0
) {
    val prefillTokensPerSecond: Double? get() = rate(promptTokens, prefillNanos)
    val decodeTokensPerSecond: Double? get() = rate(generatedTokens, decodeNanos)
    private fun rate(tokens: Long, nanos: Long): Double? =
        if (tokens > 0 && nanos > 0) tokens.toDouble() * 1_000_000_000.0 / nanos else null
    companion object {
        fun fromNative(firstTokenNanos: Long?, values: LongArray, peakPssKiB: Long): TurnMetrics {
            require(values.size == 4 || values.size == 8) { "Invalid native turn statistics" }
            require(values.all { it >= 0 } && peakPssKiB >= 0)
            require(firstTokenNanos == null || firstTokenNanos >= 0)
            return TurnMetrics(firstTokenNanos, values[0], values[1], values[2], values[3], peakPssKiB,
                if (values.size == 8) StopReason.fromCode(values[4]) else StopReason.UNKNOWN,
                values.getOrElse(5) { 0 }, values.getOrElse(6) { 0 }, values.getOrElse(7) { 0 })
        }
    }
}
