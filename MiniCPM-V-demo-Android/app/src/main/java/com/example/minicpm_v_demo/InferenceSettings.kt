package com.example.minicpm_v_demo

/** The context contains conversation history, input/visual tokens and generated output together. */
data class InferenceSettings(
    val contextTokens: Int = 16384,
    val maxOutputTokens: Int = 8192,
    val temperature: Float = 1.0f,
    val topP: Float = 0.95f,
    val topK: Int = 20,
    val minP: Float = 0.0f,
    val repeatPenalty: Float = 1.0f,
    val repeatLastN: Int = 64,
    val seed: Int = -1
) {
    fun validate() {
        require(contextTokens in CONTEXT_CHOICES) { "contextTokens" }
        require(maxOutputTokens in 1..(contextTokens - OUTPUT_RESERVE)) { "maxOutputTokens" }
        require(temperature.isFinite() && temperature in 0.0f..2.0f) { "temperature" }
        require(topP.isFinite() && topP > 0 && topP <= 1) { "topP" }
        require(topK in 0..1000) { "topK" }
        require(minP.isFinite() && minP in 0.0f..1.0f) { "minP" }
        require(repeatPenalty.isFinite() && repeatPenalty in 0.5f..2.0f) { "repeatPenalty" }
        require(repeatLastN in 0..4096) { "repeatLastN" }
        require(seed >= -1) { "seed" }
    }

    companion object {
        const val OUTPUT_RESERVE = 128
        val CONTEXT_CHOICES = listOf(2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
    }
}
