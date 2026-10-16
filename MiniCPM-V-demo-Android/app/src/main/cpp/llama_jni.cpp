#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <vector>
#include <cstdlib>
#include <exception>
#include <algorithm>
#include <array>
#include <chrono>
#include <deque>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "mtmd.h"
#include "mtmd-helper.h"

template<class T>
static std::string join(const std::vector<T> &values, const std::string &delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) { str << delim; }
    }
    return str.str();
}

// Inference parameters mirror the iOS demo defaults (MTMDParams.swift / mtmd-ios.cpp):
//   nThreads=4, nCtx=4096 (8192 for MiniCPM-V-4.6 to fit video frames),
//   nBatch=2048, temperature=0.7, top_k=0, top_p=1.0, penalty_repeat=1.0,
//   nPredict=100. Keeping Android in lockstep avoids per-platform
//   divergence in generation quality and prefill latency.
constexpr int   N_THREADS               = 4;

constexpr int   DEFAULT_CONTEXT_SIZE    = 4096;
// MiniCPM-V-4.6 video understanding feeds up to 64 frames * ~64 visual
// tokens each into the KV cache; 4096 is not enough.  iOS demo
// (MBHomeViewController+LoadModel.swift) bumps n_ctx to 8192 only on
// V-4.6 load so older / non-vision models keep the cheaper 4096 budget.
constexpr int   V46_CONTEXT_SIZE        = 8192;
constexpr int   OVERFLOW_HEADROOM       = 4;
constexpr int   BATCH_SIZE              = 2048;
// Aligned with the model's generation_config.json (do_sample=true,
// temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0). top_k=0 and
// top_p=1.0 effectively disable those filters, so sampling is pure
// temperature-only as the model card recommends.
constexpr float DEFAULT_SAMPLER_TEMP    = 0.7f;

static llama_model                      * g_model;
static llama_context                    * g_context;
static llama_batch                        g_batch;
static common_chat_templates_ptr          g_chat_templates;
static common_sampler                   * g_sampler;
static mtmd_context                     * g_ctx_vision;
// MiniCPM-V family version of the loaded mmproj. 0 = not minicpmv / no mmproj.
// 5  = V-4.0
// 6  = o-4.0
// 100045 = o-4.5
// 46/460/461 = V-4.6 (instruct/thinking)
static int                                g_minicpmv_version = 0;
static bool g_moe = false;
static mtmd_bitmap * g_pending_bitmap = nullptr;
static std::string g_experts_path;
struct InferenceOptions {
    int context = 16384;
    float temperature = 1.0f;
    float top_p = 0.95f;
    int top_k = 20;
    float min_p = 0.0f;
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    int seed = -1;
};
static InferenceOptions g_options;
static std::deque<llama_token> g_repetition_history;
static void remember_token(llama_token token, bool generated) {
    common_sampler_accept(g_sampler, token, generated);
    g_repetition_history.push_back(token);
    if (g_repetition_history.size() > 4096) g_repetition_history.pop_front();
}


// Per turn: prompt token count / prompt nanoseconds / generated token count / decode nanoseconds.
static std::array<jlong, 8> g_turn_stats{};
static int64_t turn_clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_getTurnStatsNative(JNIEnv *env, jobject) {
    auto result = env->NewLongArray(8);
    if (result) env->SetLongArrayRegion(result, 0, 8, g_turn_stats.data());
    return result;
}


// Most recent slice cap requested by the upper layer.  Persists across
// loadMmproj calls so a user-chosen value survives e.g. an unload/reload
// of the model.  Initial fallback = 9 (MiniCPM-V's built-in upper bound)
// so anything that calls into native before Kotlin has had a chance to
// thread the user-preference value through gets the "high quality"
// default rather than the legacy "no slicing" one.  In practice every
// real call site uses LlamaEngine.loadMmproj(path, n) which passes the
// persisted preference, so this only matters for the very first
// setImageMaxSliceNumsNative call before mmproj is loaded.
static int                                g_image_max_slice_nums = 9;

// Effective n_ctx used to create g_context.  Set by prepare(); reads by
// the few stay-under-context guards scattered through this file.  Kept
// as a global rather than a llama_n_ctx() call so we don't depend on
// llama-side defaults when llama_context_default_params changes.
static int                                g_n_ctx = DEFAULT_CONTEXT_SIZE;

// MiniCPM5 / V-4.6 thinking toggle. Default off so the first reply is
// not a long <think> dump.
static bool                               g_enable_thinking = false;

static void apply_minicpm5_thinking_suffix(std::string &prompt) {
    static const char kAss[] = "<|im_start|>assistant\n";
    static const char kOpen[] = "<think>\n";
    static const char kClosed[] = "<think>\n\n</think>\n\n";
    const size_t assLen = sizeof(kAss) - 1;
    const auto pos = prompt.rfind(kAss);
    if (pos != std::string::npos) {
        const std::string rest = prompt.substr(pos + assLen);
        if (rest.empty() || rest == kOpen || rest == kClosed) {
            prompt.resize(pos);
        }
    }
    prompt += kAss;
    prompt += g_enable_thinking ? kOpen : kClosed;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_init(JNIEnv *env, jobject /*unused*/, jstring nativeLibDir) {
    llama_log_set(minicpm_android_log_callback, nullptr);

    const auto *path_to_backend = env->GetStringUTFChars(nativeLibDir, 0);
    LOGi("Loading backends from %s", path_to_backend);
    ggml_backend_load_all_from_path(path_to_backend);
    env->ReleaseStringUTFChars(nativeLibDir, path_to_backend);

    if (ggml_backend_reg_count() == 0) {
        LOGw("%s: No backends loaded from path, trying ggml_backend_load_all()", __func__);
        ggml_backend_load_all();
    }

    LOGi("%s: Backend count: %zu", __func__, ggml_backend_reg_count());

    llama_backend_init();
    LOGi("Backend initiated; Log handler set.");
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_load(JNIEnv *env, jobject, jstring jmodel_path,
                                                  jstring jexperts_path, jint cache_window) {
    const char * model_utf = env->GetStringUTFChars(jmodel_path, nullptr);
    const std::string model_path(model_utf);
    env->ReleaseStringUTFChars(jmodel_path, model_utf);
    std::string experts;
    if (jexperts_path) {
        const char * utf = env->GetStringUTFChars(jexperts_path, nullptr);
        experts = utf;
        env->ReleaseStringUTFChars(jexperts_path, utf);
    }
    try {
        llama_model_params params = llama_model_default_params();
        g_moe = !experts.empty();
        if (g_moe) {
            g_experts_path = experts;
            params.moe_experts_path = g_experts_path.c_str();
            params.moe_slot_bank = 256;
            params.n_gpu_layers = 0;
            params.use_mmap = false;
            params.use_extra_bufts = false;
            const std::string window = std::to_string(std::clamp((int) cache_window, 8, 64));
            setenv("LLAMA_MOE_DECODE_WINDOW", window.c_str(), 1);
            setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1", 1);
            setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "4", 1);
            setenv("LLAMA_MOE_IO_THREADS", "4", 1);
            LOGi("LES CPU: bank=256 window=%s experts=%s", window.c_str(), experts.c_str());
        }
        g_model = llama_model_load_from_file(model_path.c_str(), params);
        if (!g_model) return 2;
        return 0;
    } catch (const std::exception & e) {
        LOGe("Model load failed: %s", e.what());
        return 3;
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_loadMmproj(JNIEnv *env, jobject,
                                                        jstring jmmproj_path,
                                                        jint jimage_max_slice_nums) {
    if (!g_model) {
        LOGe("%s: Base model must be loaded first!", __func__);
        return 1;
    }

    const auto *mmproj_path = env->GetStringUTFChars(jmmproj_path, 0);
    LOGd("%s: Loading mmproj from: \n%s\n", __func__, mmproj_path);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu   = false;
    mparams.print_timings = false;

    g_image_max_slice_nums       = (jint) jimage_max_slice_nums;
    mparams.image_max_slice_nums = g_image_max_slice_nums;

    mparams.n_threads = N_THREADS;

    g_ctx_vision = mtmd_init_from_file(mmproj_path, g_model, mparams);
    env->ReleaseStringUTFChars(jmmproj_path, mmproj_path);

    if (!g_ctx_vision) {
        LOGe("%s: Failed to load mmproj model!", __func__);
        return 2;
    }

    // mtmd_get_minicpmv_version() was a fork-only API that lived in the
    // demo-private mtmd-ios.h.  Upstream master mtmd does not expose the
    // mmproj's MiniCPM-V family version.  The Kotlin layer now sets
    // g_minicpmv_version explicitly via setMinicpmvVersionNative() based
    // on the selected ModelInfo id (4.6 -> 46).  See iOS opt-r1's
    // MBHomeViewController+LoadModel.swift for the symmetric move.

    LOGi("%s: mmproj model loaded successfully! Vision: %s, slice-cap: %d (applied)",
         __func__,
         mtmd_support_vision(g_ctx_vision) ? "yes" : "no",
         g_image_max_slice_nums);
    return 0;
}

// Returns the MiniCPM-V family version of the currently loaded mmproj
// (0 if no mmproj is loaded).  Backed by [setMinicpmvVersionNative]: the
// Kotlin layer infers the version from the selected ModelInfo id (4.6 ->
// 46) and pushes it down after loadMmproj.  Replaces the legacy
// mtmd_get_minicpmv_version() introspection that the fork mtmd-ios.h used
// to expose but upstream master mtmd removed.
extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_getMinicpmvVersionNative(JNIEnv * /*env*/, jobject) {
    return (jint) g_minicpmv_version;
}

// Kotlin-driven setter for the mmproj version (see comment on
// getMinicpmvVersionNative).  Affects prepare()'s n_ctx selection and
// the assistant-turn prefix logic; should be called immediately after
// loadMmproj() succeeds.  Passing 0 disables both branches.
extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_setMinicpmvVersionNative(JNIEnv * /*env*/,
                                                                      jobject,
                                                                      jint jversion) {
    g_minicpmv_version = (int) jversion;
    LOGi("%s: minicpmv_version set to %d", __func__, g_minicpmv_version);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_setImageMaxSliceNumsNative(JNIEnv * /*env*/,
                                                                        jobject,
                                                                        jint jn) {
    g_image_max_slice_nums = (int) jn;
    if (g_ctx_vision) {
        mtmd_set_image_max_slice_nums(g_ctx_vision, g_image_max_slice_nums);
        LOGi("%s: image_max_slice_nums set to %d (applied)", __func__, g_image_max_slice_nums);
    } else {
        LOGi("%s: image_max_slice_nums set to %d (stored)", __func__, g_image_max_slice_nums);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_setEnableThinkingNative(JNIEnv * /*env*/,
                                                                     jobject,
                                                                     jboolean jenable) {
    g_enable_thinking = (bool) jenable;
    LOGi("%s: enable_thinking=%d", __func__, (int) g_enable_thinking);
}

static llama_context *init_context(llama_model *model, const int n_ctx = DEFAULT_CONTEXT_SIZE) {
    if (!model) {
        LOGe("%s: model cannot be null", __func__);
        return nullptr;
    }

    LOGi("%s: Using %d threads (aligned with iOS demo MTMDParams)", __func__, N_THREADS);

    llama_context_params ctx_params = llama_context_default_params();
    const int trained_context_size = llama_model_n_ctx_train(model);
    if (n_ctx > trained_context_size) {
        LOGw("%s: Model was trained with only %d context size! Enforcing %d context size...",
             __func__, trained_context_size, n_ctx);
    }
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = g_moe ? 32 : BATCH_SIZE;
    ctx_params.n_threads = N_THREADS;
    ctx_params.n_threads_batch = N_THREADS;
    auto *context = llama_init_from_model(g_model, ctx_params);
    if (context == nullptr) {
        LOGe("%s: llama_new_context_with_model() returned null)", __func__);
    }
    return context;
}

static common_sampler *new_sampler(float temp) {
    common_params_sampling sparams;
    sparams.temp = temp;
    sparams.top_k = g_options.top_k;
    sparams.top_p = g_options.top_p;
    sparams.min_p = g_options.min_p;
    sparams.penalty_repeat = g_options.repeat_penalty;
    sparams.penalty_last_n = g_options.repeat_last_n;
    sparams.n_prev = std::max(64, g_options.repeat_last_n);
    sparams.seed = g_options.seed < 0 ? LLAMA_DEFAULT_SEED : (uint32_t) g_options.seed;
    return common_sampler_init(g_model, sparams);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_configureInferenceNative(
        JNIEnv *, jobject, jint context, jfloat temperature, jfloat top_p, jint top_k,
        jfloat min_p, jfloat repeat_penalty, jint repeat_last_n, jint seed) {
    if (context < 2048 || context > 262144 || !std::isfinite(temperature) || temperature < 0 || temperature > 2 ||
        !std::isfinite(top_p) || top_p <= 0 || top_p > 1 || top_k < 0 || top_k > 1000 ||
        !std::isfinite(min_p) || min_p < 0 || min_p > 1 || !std::isfinite(repeat_penalty) ||
        repeat_penalty < 0.5f || repeat_penalty > 2 || repeat_last_n < 0 || repeat_last_n > 4096 || seed < -1) return 1;
    if (g_model && context > (int) llama_model_n_ctx_train(g_model)) return 2;
    const auto previous = g_options;
    const bool sampling_changed = temperature != previous.temperature || top_p != previous.top_p ||
        top_k != previous.top_k || min_p != previous.min_p || repeat_penalty != previous.repeat_penalty ||
        repeat_last_n != previous.repeat_last_n || seed != previous.seed;
    g_options = {context, temperature, top_p, top_k, min_p, repeat_penalty, repeat_last_n, seed};
    if (g_sampler && sampling_changed) {
        try {
            auto * next = new_sampler(g_options.temperature);
            if (!next) { g_options = previous; return 1; }
            for (auto token : g_repetition_history) common_sampler_accept(next, token, false);
            common_sampler_free(g_sampler);
            g_sampler = next;
        } catch (const std::exception & e) {
            LOGe("Sampling configuration failed: %s", e.what());
            g_options = previous;
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_prepare(JNIEnv * /*env*/, jobject /*unused*/) {
    // Validated per-model settings; total context includes input, history and output.
    const int n_ctx = g_options.context;
    if (n_ctx > (int) llama_model_n_ctx_train(g_model)) return 2;
    LOGi("%s: minicpmv_version=%d -> n_ctx=%d", __func__, g_minicpmv_version, n_ctx);

    auto *context = init_context(g_model, n_ctx);
    if (!context) { return 1; }
    g_context = context;
    g_n_ctx = n_ctx;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(g_options.temperature);
    return g_sampler ? 0 : 1;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_systemInfo(JNIEnv *env, jobject /*unused*/) {
    return env->NewStringUTF(llama_print_system_info());
}

constexpr const char *ROLE_SYSTEM       = "system";
constexpr const char *ROLE_USER         = "user";
constexpr const char *ROLE_ASSISTANT    = "assistant";

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position;
static llama_pos current_position;
static llama_pos generation_start_position;
static bool g_image_prefilled = false;
static bool g_vision_mode = false;

static void reset_long_term_states(const bool clear_kv_cache = true) {
    chat_msgs.clear();
    g_repetition_history.clear();
    system_prompt_position = 0;
    current_position = 0;
    g_image_prefilled = false;
    g_vision_mode = false;

    if (g_pending_bitmap) { mtmd_bitmap_free(g_pending_bitmap); g_pending_bitmap = nullptr; }
    if (clear_kv_cache && g_context)
        llama_memory_clear(llama_get_memory(g_context), false);
}

// Mirror of iOS MBMtmd.mm prefill_text role="user":
// MiniCPM-V-4.6 (46/460/461) follows g_enable_thinking, same as MiniCPM5.
// v4.0 / v2.x use plain ChatML and have no thinking toggle.
static const char * assistant_turn_prefix() {
    switch (g_minicpmv_version) {
        case 47:
        case 46:
        case 460:
        case 461:
            return g_enable_thinking
                ? "<|im_start|>assistant\n<think>\n"
                : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        default:
            return "<|im_start|>assistant\n";
    }
}

static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    LOGi("%s: Discarding %d tokens", __func__, n_discard);
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
    LOGi("%s: Context shifting done! Current position: %d", __func__, current_position);
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, true);
    chat_msgs.push_back(new_msg);
    LOGi("%s: Formatted and added %s message: \n%s\n", __func__, role.c_str(), formatted.c_str());
    return formatted;
}

static llama_pos stop_generation_position;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

static void reset_short_term_states() {
    stop_generation_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
}

static int decode_tokens_in_batches(
        llama_context *context,
        llama_batch &batch,
        const llama_tokens &tokens,
        const llama_pos start_pos,
        const bool compute_last_logit = false) {
    LOGd("%s: Decode %d tokens starting at position %d", __func__, (int) tokens.size(), start_pos);
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(batch);
        LOGv("%s: Preparing a batch size of %d starting at: %d", __func__, cur_batch_size, i);

        if (start_pos + i + cur_batch_size >= g_n_ctx - OVERFLOW_HEADROOM) {
            if (g_moe) return 3; // Recurrent memory cannot shift an arbitrary prefix.
            LOGw("%s: Current batch won't fit into context! Shifting...", __func__);
            shift_context();
        }

        for (int j = 0; j < cur_batch_size; j++) {
            const llama_token token_id = tokens[i + j];
            const llama_pos position = start_pos + i + j;
            const bool want_logit = compute_last_logit && (i + j == tokens.size() - 1);
            common_batch_add(batch, token_id, position, {0}, want_logit);
        }

        const int decode_result = llama_decode(context, batch);
        if (decode_result) {
            LOGe("%s: llama_decode failed w/ %d", __func__, decode_result);
            return 1;
        }
    }
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_processSystemPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring jsystem_prompt
) {
    reset_short_term_states();

    const auto *system_prompt = env->GetStringUTFChars(jsystem_prompt, nullptr);
    LOGd("%s: System prompt received: \n%s", __func__, system_prompt);
    std::string formatted_system_prompt(system_prompt);

    const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
    if (has_chat_template) {
        formatted_system_prompt = chat_add_and_format(ROLE_SYSTEM, system_prompt);
    }
    env->ReleaseStringUTFChars(jsystem_prompt, system_prompt);

    if (g_ctx_vision) {
        mtmd_input_text input_text;
        input_text.text = formatted_system_prompt.c_str();
        input_text.add_special = current_position == 0;
        input_text.parse_special = true;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &input_text, nullptr, 0);
        if (res != 0) {
            LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                    current_position, 0, BATCH_SIZE,
                                    true, &new_n_past)) {
            LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        current_position = new_n_past;
        generation_start_position = current_position;
        mtmd_input_chunks_free(chunks);
    } else {
        const auto system_tokens = common_tokenize(g_context, formatted_system_prompt,
                                                   current_position == 0, true);
        for (auto id: system_tokens) {
            LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
        }

        const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
        if ((int) system_tokens.size() > max_batch_size) {
            LOGe("%s: System prompt too long for context! %d tokens, max: %d",
                 __func__, (int) system_tokens.size(), max_batch_size);
            return 1;
        }

        if (decode_tokens_in_batches(g_context, g_batch, system_tokens, current_position, true)) {
            LOGe("%s: llama_decode() failed!", __func__);
            return 2;
        }

        current_position += (int) system_tokens.size();
    }

    system_prompt_position = current_position;
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_prefillImage(
        JNIEnv *env,
        jobject /*unused*/,
        jbyteArray jimage_data,
        jint jimage_size
) {
    if (!g_ctx_vision) {
        LOGe("%s: mmproj not loaded!", __func__);
        return 1;
    }

    jbyte *data = env->GetByteArrayElements(jimage_data, nullptr);
    if (!data) {
        LOGe("%s: Failed to get image data from Java", __func__);
        return 2;
    }

    auto *bitmap = mtmd_helper_bitmap_init_from_buf(
        g_ctx_vision,
        reinterpret_cast<const unsigned char *>(data),
        static_cast<size_t>(jimage_size)
    );

    env->ReleaseByteArrayElements(jimage_data, data, JNI_ABORT);

    if (!bitmap) {
        LOGe("%s: Failed to create bitmap from image buffer", __func__);
        return 3;
    }

    if (g_moe) {
        // The image marker belongs inside the user turn, which arrives later.
        if (g_pending_bitmap) mtmd_bitmap_free(g_pending_bitmap);
        g_pending_bitmap = bitmap;
        g_image_prefilled = true;
        g_vision_mode = true;
        return 0;
    }

    mtmd_input_text text;
    text.text = mtmd_default_marker();
    text.add_special = current_position == 0;
    text.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps_cptr[] = { bitmap };
    int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text,
                                bitmaps_cptr, 1);

    mtmd_bitmap_free(bitmap);

    if (res != 0) {
        LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
        mtmd_input_chunks_free(chunks);
        return 4;
    }

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                current_position, 0, BATCH_SIZE,
                                false, &new_n_past)) {
        LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
        mtmd_input_chunks_free(chunks);
        return 5;
    }

    current_position = new_n_past;
    mtmd_input_chunks_free(chunks);
    g_image_prefilled = true;
    g_vision_mode = true;

    LOGi("%s: Image prefilled, current_position: %d", __func__, current_position);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_fullReset(JNIEnv *, jobject) {
    if (!g_model || g_options.context > (int) llama_model_n_ctx_train(g_model)) return 2;
    g_n_ctx = g_options.context;
    reset_long_term_states();
    reset_short_term_states();

    if (g_sampler) common_sampler_free(g_sampler);
    g_sampler = nullptr;
    g_chat_templates.reset();
    if (g_batch.token) llama_batch_free(g_batch);
    g_batch = {};
    if (g_context) llama_free(g_context);
    g_context = nullptr;

    auto *context = init_context(g_model, g_n_ctx);
    if (!context) {
        LOGe("%s: Failed to reinitialize context!", __func__);
        return 1;
    }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");
    g_sampler = new_sampler(g_options.temperature);
    if (!g_sampler) return 1;

    LOGi("%s: Full reset complete - context recreated, KV cache fresh, sampler reinitialized", __func__);
    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_nativeCancelGeneration(JNIEnv *, jobject) {
    if (!g_sampler) return;
    common_sampler_reset(g_sampler);
    for (auto token : g_repetition_history) common_sampler_accept(g_sampler, token, false);
    assistant_ss.str("");
    assistant_ss.clear();
    cached_token_chars.clear();
    LOGi("%s: Generation cancelled, sampler reset, KV cache preserved at position %d",
         __func__, current_position);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_processUserPrompt(
        JNIEnv *env,
        jobject /*unused*/,
        jstring juser_prompt,
        jint n_predict
) {
    g_turn_stats.fill(0);
    g_turn_stats[6] = g_n_ctx;
    g_turn_stats[7] = n_predict;
    const int64_t prefill_started = turn_clock_ns();
    reset_short_term_states();

    const auto *const user_prompt = env->GetStringUTFChars(juser_prompt, nullptr);
    LOGd("%s: User prompt received: \n%s", __func__, user_prompt);

    std::string content_for_format(user_prompt);
    if (content_for_format.empty()) {
        content_for_format = " ";
    }

    std::string formatted_user_prompt;
    if (g_ctx_vision) {
        // No default system prompt: aligned with iOS opt-r1's MBMtmd.mm.
        // An English-only "You are a helpful assistant." was observed to
        // bias MiniCPM-V into answering Chinese queries in English. The
        // reference Python pipeline (`AutoModel.chat(...)` /
        // `apply_chat_template`) does not inject one either when the
        // caller passes only a user message. If a system message is
        // required it must be added through processSystemPrompt()
        // *before* the first processUserPrompt() call.
        const std::string media = g_pending_bitmap ? std::string(mtmd_default_marker()) + "\n" : "";
        formatted_user_prompt += "<|im_start|>user\n" + media + content_for_format + "<|im_end|>\n";
        formatted_user_prompt += assistant_turn_prefix();

        common_chat_msg new_msg;
        new_msg.role = ROLE_USER;
        new_msg.content = content_for_format;
        chat_msgs.push_back(new_msg);

        LOGi("%s: Formatted user prompt (mtmd, image=%s, minicpmv=%d): \n%s\n",
             __func__,
             g_image_prefilled ? "yes" : "no",
             g_minicpmv_version,
             formatted_user_prompt.c_str());

        g_image_prefilled = false;
    } else {
        const bool has_chat_template = common_chat_templates_was_explicit(g_chat_templates.get());
        if (has_chat_template) {
            formatted_user_prompt = chat_add_and_format(ROLE_USER, content_for_format.c_str());
        } else {
            formatted_user_prompt = content_for_format;
        }
        apply_minicpm5_thinking_suffix(formatted_user_prompt);
    }
    env->ReleaseStringUTFChars(juser_prompt, user_prompt);

    if (formatted_user_prompt.size() >= 8 &&
        formatted_user_prompt.compare(formatted_user_prompt.size() - 8, 8, "<think>\n") == 0) {
        cached_token_chars = "<think>\n";
    }

    if (g_ctx_vision) {
        mtmd_input_text text;
        text.text          = formatted_user_prompt.c_str();
        text.add_special   = current_position == 0;
        text.parse_special = true;

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bitmaps[] = { g_pending_bitmap };
        int32_t res = mtmd_tokenize(g_ctx_vision, chunks, &text,
                                   g_pending_bitmap ? bitmaps : nullptr,
                                   g_pending_bitmap ? 1 : 0);
        if (g_pending_bitmap) { mtmd_bitmap_free(g_pending_bitmap); g_pending_bitmap = nullptr; }
        size_t prompt_tokens = 0;
        if (res == 0) {
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
                prompt_tokens += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i));
            }
            if (g_moe && (size_t) current_position + prompt_tokens + 32 >= (size_t) g_n_ctx) {
                g_turn_stats[4] = 2;
                g_turn_stats[5] = current_position;
                mtmd_input_chunks_free(chunks);
                return 3;
            }
        }
        if (res != 0) {
            LOGe("%s: mtmd_tokenize failed with code %d", __func__, res);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        llama_pos new_n_past;
        if (mtmd_helper_eval_chunks(g_ctx_vision, g_context, chunks,
                                    current_position, 0, BATCH_SIZE,
                                    true, &new_n_past)) {
            LOGe("%s: mtmd_helper_eval_chunks failed!", __func__);
            mtmd_input_chunks_free(chunks);
            return 2;
        }

        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto * chunk = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
            size_t count = 0;
            const auto * ids = mtmd_input_chunk_get_tokens_text(chunk, &count);
            if (ids) for (size_t j = 0; j < count; ++j) remember_token(ids[j], false);
        }
        g_turn_stats[0] = (jlong) prompt_tokens;
        current_position = new_n_past;
        generation_start_position = current_position;
        mtmd_input_chunks_free(chunks);
    } else {
        auto user_tokens = common_tokenize(g_context, formatted_user_prompt, current_position == 0, true);
        for (auto id: user_tokens) {
            LOGv("token: `%s`\t -> `%d`", common_token_to_piece(g_context, id).c_str(), id);
        }

        const int user_prompt_size = (int) user_tokens.size();
        const int max_batch_size = g_n_ctx - OVERFLOW_HEADROOM;
        if (user_prompt_size > max_batch_size) {
            const int skipped_tokens = user_prompt_size - max_batch_size;
            user_tokens.resize(max_batch_size);
            LOGw("%s: User prompt too long! Skipped %d tokens!", __func__, skipped_tokens);
        }

        if (decode_tokens_in_batches(g_context, g_batch, user_tokens, current_position, true)) {
            LOGe("%s: llama_decode() failed!", __func__);
            return 2;
        }

        for (auto token : user_tokens) remember_token(token, false);
        g_turn_stats[0] = (jlong) user_tokens.size();
        current_position += user_prompt_size;
        generation_start_position = current_position;
    }

    g_turn_stats[1] = turn_clock_ns() - prefill_started;
    g_turn_stats[5] = current_position;
    stop_generation_position = current_position + n_predict;
    return 0;
}

static bool is_valid_utf8(const char *string) {
    if (!string) { return true; }

    const auto *bytes = (const unsigned char *) string;
    int num;

    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_generateNextToken(
        JNIEnv *env,
        jobject /*unused*/
) {
    const int64_t decode_started = turn_clock_ns();
    if (current_position >= g_n_ctx - OVERFLOW_HEADROOM) {
        if (g_moe) { g_turn_stats[4] = 2; return nullptr; }
        LOGw("%s: Context full! Shifting...", __func__);
        shift_context();
    }

    if (g_turn_stats[2] >= g_turn_stats[7]) {
        LOGw("%s: STOP: hitting stop position: %d", __func__, stop_generation_position);
        g_turn_stats[4] = 1;
        return nullptr;
    }

    const auto new_token_id = common_sampler_sample(g_sampler, g_context, -1);
    remember_token(new_token_id, true);

    const bool is_eog = llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token_id);

    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token_id, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        g_turn_stats[4] = 4;
        LOGe("%s: llama_decode() failed for generated token", __func__);
        return nullptr;
    }

    current_position++;
    g_turn_stats[5] = current_position;

    if (is_eog) {
        g_turn_stats[4] = 3;
        LOGd("id: %d,\tIS EOG!\nSTOP.", new_token_id);
        std::string assistant_content = assistant_ss.str();
        auto think_end = assistant_content.find("</think>");
        if (think_end != std::string::npos) {
            assistant_content = assistant_content.substr(think_end + 8);
            while (!assistant_content.empty() && assistant_content[0] == '\n') {
                assistant_content.erase(0, 1);
            }
        }
        chat_add_and_format(ROLE_ASSISTANT, assistant_content);
        return nullptr;
    }

    ++g_turn_stats[2];
    g_turn_stats[3] += turn_clock_ns() - decode_started;
    auto new_token_chars = common_token_to_piece(g_context, new_token_id);
    cached_token_chars += new_token_chars;

    jstring result = nullptr;
    if (is_valid_utf8(cached_token_chars.c_str())) {
        result = env->NewStringUTF(cached_token_chars.c_str());
        LOGv("id: %d,\tcached: `%s`,\tnew: `%s`", new_token_id, cached_token_chars.c_str(), new_token_chars.c_str());

        assistant_ss << cached_token_chars;
        cached_token_chars.clear();
    } else {
        LOGv("id: %d,\tappend to cache", new_token_id);
        result = env->NewStringUTF("");
    }
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_unload(JNIEnv * /*env*/, jobject /*unused*/) {
    reset_long_term_states();
    reset_short_term_states();

    if (g_ctx_vision) {
        mtmd_free(g_ctx_vision);
        g_ctx_vision = nullptr;
    }

    if (g_sampler) common_sampler_free(g_sampler);
    g_sampler = nullptr;
    g_chat_templates.reset();
    if (g_batch.token) llama_batch_free(g_batch);
    g_batch = {};
    if (g_context) llama_free(g_context);
    g_context = nullptr;
    if (g_model) llama_model_free(g_model);
    g_model = nullptr;
    g_moe = false;
    g_minicpmv_version = 0;
    g_experts_path.clear();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_minicpm_1v_1demo_LlamaEngine_shutdown(JNIEnv *, jobject /*unused*/) {
    llama_backend_free();
}
