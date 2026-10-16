package com.example.minicpm_v_demo

import android.content.Context
import android.os.Debug
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicLong

/** PSS includes native weights, expert buffers, mapped files and the Java heap. */
internal class TurnMemorySampler {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val peak = AtomicLong()
    private val worker = scope.launch {
        while (isActive) {
            sample()
            delay(1000)
        }
    }

    private fun sample() {
        try {
            val info = Debug.MemoryInfo()
            Debug.getMemoryInfo(info)
            val value = info.totalPss.toLong()
            peak.updateAndGet { previous -> maxOf(previous, value) }
        } catch (e: Exception) {
            Log.w("TurnMemorySampler", "Cannot sample process PSS", e)
        }
    }

    suspend fun finish(): Long {
        try {
            worker.cancelAndJoin()
            withContext(Dispatchers.IO) { sample() }
            return peak.get()
        } finally {
            scope.cancel()
        }
    }
}

internal fun TurnMetrics.displayText(context: Context): String {
    val unavailable = context.getString(R.string.perf_unavailable)
    fun number(value: Double?): String = value?.let { context.getString(R.string.perf_number, it) } ?: unavailable
    val performance = context.getString(
        R.string.perf_turn_summary,
        number(firstTokenNanos?.toDouble()?.div(1_000_000_000.0)),
        number(prefillTokensPerSecond),
        number(decodeTokensPerSecond),
        number(peakPssKiB.takeIf { it > 0 }?.toDouble()?.div(1024.0 * 1024.0))
    )
    val label = when (stopReason) {
        StopReason.UNKNOWN -> R.string.stop_unknown
        StopReason.OUTPUT_LIMIT -> R.string.stop_output_limit
        StopReason.CONTEXT_LIMIT -> R.string.stop_context_limit
        StopReason.END_OF_SEQUENCE -> R.string.stop_eos
        StopReason.ERROR -> R.string.stop_error
        StopReason.USER_STOP -> R.string.stop_user
    }
    return performance + "\n" + context.getString(R.string.perf_turn_budget,
        generatedTokens, outputLimit, contextUsed, contextCapacity, context.getString(label))
}
