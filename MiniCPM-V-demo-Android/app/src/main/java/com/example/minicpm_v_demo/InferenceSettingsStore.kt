package com.example.minicpm_v_demo

import android.content.Context

internal object InferenceSettingsStore {
    fun defaults(model: ModelInfo): InferenceSettings = if (model.isMoe) InferenceSettings() else
        InferenceSettings(contextTokens = if (model.id.contains("4_6") || model.id.contains("4.6")) 8192 else 4096,
            maxOutputTokens = 3072, temperature = 0.7f, topP = 1f, topK = 0)

    fun read(context: Context, model: ModelInfo): InferenceSettings {
        val p = context.getSharedPreferences("inference_settings", Context.MODE_PRIVATE)
        val d = defaults(model)
        val key = model.id + "."
        return try {
            InferenceSettings(
                p.getInt(key + "context", d.contextTokens), p.getInt(key + "output", d.maxOutputTokens),
                p.getFloat(key + "temperature", d.temperature), p.getFloat(key + "top_p", d.topP),
                p.getInt(key + "top_k", d.topK), p.getFloat(key + "min_p", d.minP),
                p.getFloat(key + "repeat_penalty", d.repeatPenalty), p.getInt(key + "repeat_last_n", d.repeatLastN),
                p.getInt(key + "seed", d.seed)
            ).also { it.validate() }
        } catch (_: RuntimeException) { d }
    }

    fun write(context: Context, model: ModelInfo, value: InferenceSettings) {
        value.validate()
        val key = model.id + "."
        context.getSharedPreferences("inference_settings", Context.MODE_PRIVATE).edit()
            .putInt(key + "context", value.contextTokens).putInt(key + "output", value.maxOutputTokens)
            .putFloat(key + "temperature", value.temperature).putFloat(key + "top_p", value.topP)
            .putInt(key + "top_k", value.topK).putFloat(key + "min_p", value.minP)
            .putFloat(key + "repeat_penalty", value.repeatPenalty).putInt(key + "repeat_last_n", value.repeatLastN)
            .putInt(key + "seed", value.seed).apply()
    }
}
