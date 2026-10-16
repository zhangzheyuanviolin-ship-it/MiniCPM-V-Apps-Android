package com.example.minicpm_v_demo

import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

class InferenceSettingsActivity : AppCompatActivity() {
    private lateinit var model: ModelInfo
    private lateinit var contextSize: Spinner
    private val fields = linkedMapOf<String, Pair<TextInputLayout, TextInputEditText>>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_inference_settings)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(android.R.id.content)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            v.updatePadding(left = bars.left, top = bars.top, right = bars.right, bottom = maxOf(bars.bottom, ime.bottom))
            insets
        }
        findViewById<Toolbar>(R.id.toolbar).setNavigationOnClickListener { finish() }
        model = LlamaEngine.getSelectedModel(this)
        if (model.isTts) { finish(); return }
        contextSize = findViewById(R.id.sp_context_tokens)
        contextSize.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item,
            InferenceSettings.CONTEXT_CHOICES.map { getString(R.string.settings_token_choice, it) }).also {
            it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        }
        val parent = findViewById<LinearLayout>(R.id.settings_fields)
        fun field(key: String, label: Int, help: Int, decimal: Boolean = false, signed: Boolean = false) {
            val layout = TextInputLayout(this).apply {
                hint = getString(label)
                helperText = getString(help)
                layoutParams = LinearLayout.LayoutParams(-1, -2).apply { topMargin = (12 * resources.displayMetrics.density).toInt() }
            }
            val edit = TextInputEditText(layout.context).apply {
                id = View.generateViewId()
                inputType = InputType.TYPE_CLASS_NUMBER or
                    (if (decimal) InputType.TYPE_NUMBER_FLAG_DECIMAL else 0) or
                    (if (signed) InputType.TYPE_NUMBER_FLAG_SIGNED else 0)
                setSingleLine(true)
            }
            layout.addView(edit)
            parent.addView(layout)
            fields[key] = layout to edit
        }
        field("maxOutputTokens", R.string.settings_output, R.string.settings_output_help)
        field("temperature", R.string.settings_temperature, R.string.settings_temperature_help, decimal = true)
        field("topP", R.string.settings_top_p, R.string.settings_top_p_help, decimal = true)
        field("topK", R.string.settings_top_k, R.string.settings_top_k_help)
        field("minP", R.string.settings_min_p, R.string.settings_min_p_help, decimal = true)
        field("repeatPenalty", R.string.settings_repeat, R.string.settings_repeat_help, decimal = true)
        field("repeatLastN", R.string.settings_repeat_window, R.string.settings_repeat_window_help)
        field("seed", R.string.settings_seed, R.string.settings_seed_help, signed = true)
        fill(InferenceSettingsStore.read(this, model))
        val save = findViewById<MaterialButton>(R.id.btn_save_settings)
        val reset = findViewById<MaterialButton>(R.id.btn_reset_settings)
        reset.setOnClickListener { fill(InferenceSettingsStore.defaults(model)) }
        save.setOnClickListener {
            val value = parse() ?: return@setOnClickListener
            save.isEnabled = false
            reset.isEnabled = false
            lifecycleScope.launch {
                try {
                    LlamaEngine.getInstance(applicationContext).updateInferenceSettings(model, value)
                    Toast.makeText(this@InferenceSettingsActivity, R.string.settings_saved, Toast.LENGTH_LONG).show()
                    finish()
                } catch (e: CancellationException) { throw e
                } catch (e: Exception) {
                    Toast.makeText(this@InferenceSettingsActivity, getString(R.string.settings_save_failed, e.message), Toast.LENGTH_LONG).show()
                    save.isEnabled = true
                    reset.isEnabled = true
                }
            }
        }
    }

    private fun fill(value: InferenceSettings) {
        contextSize.setSelection(InferenceSettings.CONTEXT_CHOICES.indexOf(value.contextTokens))
        val values = mapOf("maxOutputTokens" to value.maxOutputTokens, "temperature" to value.temperature,
            "topP" to value.topP, "topK" to value.topK, "minP" to value.minP, "repeatPenalty" to value.repeatPenalty,
            "repeatLastN" to value.repeatLastN, "seed" to value.seed)
        for ((key, number) in values) { fields.getValue(key).first.error = null; fields.getValue(key).second.setText(number.toString()) }
    }

    private fun parse(): InferenceSettings? {
        fields.values.forEach { it.first.error = null }
        fun text(key: String) = fields.getValue(key).second.text.toString().trim().replace(',', '.')
        fun integer(key: String) = text(key).toIntOrNull() ?: throw IllegalArgumentException(key)
        fun decimal(key: String) = text(key).toFloatOrNull() ?: throw IllegalArgumentException(key)
        return try {
            InferenceSettings(InferenceSettings.CONTEXT_CHOICES[contextSize.selectedItemPosition],
                integer("maxOutputTokens"), decimal("temperature"), decimal("topP"), integer("topK"),
                decimal("minP"), decimal("repeatPenalty"), integer("repeatLastN"), integer("seed")).also { it.validate() }
        } catch (e: IllegalArgumentException) {
            val key = e.message
            val field = fields[key]
            field?.first?.error = getString(R.string.settings_invalid_value)
            field?.second?.requestFocus()
            Toast.makeText(this, R.string.settings_invalid_value, Toast.LENGTH_LONG).show()
            null
        }
    }
}
