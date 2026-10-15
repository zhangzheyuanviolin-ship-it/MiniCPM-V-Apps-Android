#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<float> evaluate(const std::string & file, const std::string & experts) {
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = false;
    mp.use_extra_bufts = false;
    if (!experts.empty()) { mp.moe_experts_path = experts.c_str(); mp.moe_slot_bank = 16; }
    auto * model = llama_model_load_from_file(file.c_str(), mp);
    if (!model) throw std::runtime_error("fixture model failed to load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 8; cp.n_ubatch = 1;
    cp.n_threads = 2; cp.n_threads_batch = 2;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("fixture context failed to load");
    std::vector<float> logits;
    auto batch = llama_batch_init(1, 0, 1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int tokens[] = {3, 6, 9, 12, 15, 18, 21, 4};
    for (int i = 0; i < 8; ++i) {
        batch.n_tokens = 1;
        batch.token[0] = tokens[i]; batch.pos[0] = i;
        batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
        if (llama_decode(ctx, batch) != 0) throw std::runtime_error("fixture decode failed");
        const float * values = llama_get_logits_ith(ctx, -1);
        for (int j = 0; j < vocab; ++j) {
            if (!std::isfinite(values[j])) throw std::runtime_error("non-finite fixture logits");
            logits.push_back(values[j]);
        }
    }
    llama_batch_free(batch); llama_free(ctx); llama_model_free(model);
    return logits;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all(); llama_backend_init();
    const std::string root = argv[1];
    setenv("LLAMA_MOE_DECODE_WINDOW", "4", 1);
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH", "1", 1);
    setenv("LLAMA_MOE_LOOKAHEAD_PREFETCH_DEPTH", "2", 1);
    const auto full = evaluate(root + "/inline.gguf", "");
    const auto split = evaluate(root + "/split.gguf", root + "/experts");
    if (full.size() != split.size()) return 3;
    float delta = 0;
    for (size_t i = 0; i < full.size(); ++i) delta = std::max(delta, std::abs(full[i] - split[i]));
    std::printf("CPU LES comparison: %zu finite logits, max difference %.9g\n", full.size(), delta);
    llama_backend_free();
    return delta < 0.0001f ? 0 : 4;
}
