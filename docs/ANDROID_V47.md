# MiniCPM-V 4.7 Android build

This fork adds the 35B-A3B model to the Android app. The app runs locally after its files have been downloaded. It does not contain model weights or require a chat API key.

## Installation and download

The APK is `MiniCPM-V-demo-Android-v2.8.apk`, version code 20, application ID `com.zhang.minicpmv`. It uses a dedicated signing key so it can coexist with an official installation. Future updates of this fork must keep that signing key.

The first model in Model Management is MiniCPM-V 4.7 35B-A3B. Download its 43 files in the app, then load it. The complete download is 16,623,585,508 bytes: the LES main GGUF, vision projector, experts manifest and 40 expert shards. Other listed models are optional and are not downloaded automatically.

Downloads use HTTPS, fall back between HuggingFace and ModelScope, resume partial files, verify every file against a pinned SHA-256 and size, and rename verified files in place. No second archive or extracted copy is needed. Partial downloads survive interruption. Deleting this model also deletes its experts and partial files; the loaded model is released first.

## CPU backend changes

The shared backend is pinned at `96bbf61710d44421ea54e63f5194f9b17c20d877`. The patch in `patches/android-cpu-les.patch` enables writable ordinary CPU buffers for expert tensors when both mmap and weight repacking are disabled. The app applies those two settings only for the MoE model. The submodule's tracked revision and branch remain unchanged; CI applies the patch to its checkout before compilation.

The app uses a 256-slot prefill bank, a 32-token micro-batch, four computation threads, four I/O threads and lookahead prefetch depth four. Decode cache width is selected conservatively from available memory, between 16 and 64 experts per layer. The full trained top-8 routing is preserved. This build does not include DSpark speculative decoding or an Android GPU backend, and Apple benchmark speeds do not describe it.

For V4.7, images are inserted inside their ChatML user turn and their evaluation is deferred until the user prompt. Thinking is disabled by default. The recurrent context is 4096 tokens; it is not subjected to the old attention-only prefix-shifting routine. Oversized user turns return a message asking the user to clear the conversation, and generation stops safely at the context boundary.

## Cloud verification

The Android workflow installs SDK 36.1, NDK 27.0.12077973, CMake 4.1.2 and JDK 21, uses the repository's Gradle 9.3.1 wrapper checksum, runs unit tests and release lint, builds a signed ARM64 APK, checks its application ID/version and verifies APK v2/v3 signatures. It uploads the APK, SHA-256 and verification records as a short-lived Actions artifact. Signing credentials are encrypted Actions secrets and are never committed.

The separate CPU LES workflow generates tiny deterministic hybrid Qwen3.5 MoE weights and compares every finite output logit from an inline model against the same model with streamed experts, across multiple decode steps. It downloads no published model weights. This checks the CPU expert allocation and scheduling changes; it does not replace testing the complete 35B model on an Android device.

## Storage and handoff

All compilation takes place on GitHub-hosted runners. Local work contains only source text, small reference files, logs, and the requested APK. Keep task-specific temporary files until the user confirms installation and testing passed. Then remove only inventoried task files, preserving unrelated user data, the installed app and the chosen model files. Preserve access to the dedicated signing key for future upgrades.

## Android v2.7 foreground stability and turn statistics

The v2.6 build has been tested on the target Redmi K70 Pro: text answers and image descriptions work. Android exit history also records two OEM force stops with `used too many pss resource`, measured PSS around 6.5 GiB and threshold 6,291,456 KiB. This is an OEM policy stop, even though Android categorizes the exit as USER REQUESTED; it is not evidence that the user pressed force stop.

The v2.7 APK preserves the application ID, signing certificate, model catalogue, directories, model files, LES patch and all inference capacity/routing settings from that baseline. It adds the game application-category signal and an explicit local-model foreground service with an important application-context binding. Native model allocation waits for service binding, after `startForeground` has executed. The ongoing notification lasts while the model is loading, processing or resident between turns. Its unload action and task removal cancel generation, wait for native work and metrics completion, then unload and release the service. There is no sticky restart or automatic model reload after process death. The service uses Android's `specialUse` type for user-initiated offline inference, separately from the download service's `dataSync` type.

No main-thread priority is lowered. Closing the chat requests asynchronous unload instead of blocking the Android main thread in `runBlocking`. TTS and other model engines remain initialized only on their respective feature paths. Import/download checks do not load an inference model. The document's general mmap recommendation is not transplanted: writable CPU LES slots require the already-tested non-mmap allocation path. The classification and service changes require device testing; they are not a universal MIUI memory exemption.

Each finished, stopped or failed assistant message has a separate accessible plain-text statistics footer. Footers are view data, not generated content, and never enter the model context. Statistics use the following definitions:

- First token latency: monotonic elapsed time from accepting Send until the first nonempty UTF-8 output chunk is ready. Includes queueing and prompt evaluation. For V4.7, vision encoding is deferred into that evaluation and is included; selecting a picture earlier is excluded.
- Prefill speed: actual newly evaluated input token IDs, including chat-template and V4.7 visual tokens, divided by wall time inside the native prompt path. Includes tokenization and V4.7 vision encoding. Previously cached conversation tokens are not counted again.
- Decode speed: actual generated native token IDs divided by native sampling/forward-pass time. Includes thinking tokens and UTF-8 fragments, excludes EOS and UI rendering delays. It is not based on displayed character or Flow emission counts.
- Peak memory: this turn's maximum observed total process PSS, including native buffers, mapped/shared files and Java heap. PSS is sampled once per second on an I/O worker, with a final sample. It is a sampled per-turn high, not the process-lifetime VmHWM or an exact continuous-time peak. The model remains resident, so memory is absolute process usage, not an allocation delta.

Statistics reset for every turn and are finalized before the engine becomes ready/unloadable again. Cancellation still finalizes the footer; absent first-token/rate/memory measurements display unavailable rather than fake zeros. Unit tests cover rate units, large counters, zero/failed evaluation, cancellation before a first token and invalid native schemas. Cloud release checks precede expensive native compilation.

Android foreground-service type reference: https://developer.android.com/develop/background-work/services/fgs/service-types#special-use

## Android v2.8 generation settings and output limits

The tested v2.7 version had two distinct hard limits: 1,024 newly generated tokens per reply and a 4,096-token total context for V4.7. These are not character counts. Reaching either could previously produce a silent mid-sentence stop. User feedback confirms v2.7 no longer exhibited the earlier app termination on the target phone.

Official V4.7 35B-A3B `config.json` declares `text_config.max_position_embeddings = 262144`. The revision-pinned phone split's GGUF prefix independently declares `qwen35moe.context_length = 262144`. This is a shared total window (history, text/visual input and output), not a guaranteed 262144-token response on a phone. The official app README and MTMDParams default to a 4096-token context; the Swift wrapper defaults to 100 predicted tokens. No official maximum output validated for this Android phone was found. Example `max_new_tokens=2048` values are example generation budgets, not model context capacity.

V2.8 defaults V4.7 MoE to 16384 total context and 8192 maximum new tokens, retaining the tested foreground service/category policy, expert cache parameters, model paths and model files. Settings are per model under Model Management → Inference settings. The context selector offers 2048 through 262144; actual model trained-context limits are checked in native code. Output must be positive and leave 128 tokens of nominal input space; real history/visual input can further reduce remaining room. Large user-selected windows use more memory and have not been validated on every phone. The app does not assume that the model maximum is the mobile maximum.

The official model generation configuration specifies temperature 1.0, top_p 0.95 and top_k 20; these are the V4.7 defaults in v2.8. The settings page also supports Min-P, repetition penalty and last-N range, and random/fixed seeds. Min-P defaults off and repetition penalty defaults to 1 (off). Repetition penalties now receive real textual prompt IDs and generated IDs, skipping image embeddings. A bounded 4096-token history is replayed into a replacement sampler so changing sampling parameters does not erase the conversation's penalty history.

Sampling/output-budget changes preserve context and take effect next turn. Saving a changed context frees the old cache before allocating the new one while keeping the loaded weights and foreground protection. It clears native history and flags the chat UI to clear stale messages. Invalid settings are rejected; failed cache allocation unloads safely rather than leaving a null context marked ready.

Every assistant footer retains the four performance measurements and adds actual output/requested-limit and context-usage/capacity counts plus a distinct end reason: model EOS, requested output limit, context capacity, user stop or inference failure. Generated thinking tokens count toward the requested output budget. There is no automatic hidden continuation or unlimited generation.

References:
- https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/config.json
- https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B/blob/main/generation_config.json
- https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/README.md
- https://github.com/OpenBMB/MiniCPM-V-Apps/blob/main/MiniCPM-V-demo/MTMDWrapper/MTMDParams.swift
