#include "omni_duplex_perf.h"

#include "common.h"
#include "omni_perf.h"
#include "sampling.h"
#include "tts-condition-graph.h"
#include "token2wav/token2wav-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <vector>

bool omni_image_embed_make_chunks_with_filename(struct vision_ctx * ctx_vision,
                                                int n_threads,
                                                std::string image_path,
                                                std::vector<std::vector<float>> & vision_chunks);

llama_token sample_tts_token(struct common_sampler * smpl,
                             struct omni_context * ctx_omni,
                             common_params * params,
                             int * n_past_tts,
                             const std::vector<llama_token> * all_generated_tokens,
                             const std::vector<llama_token> * chunk_generated_tokens,
                             int token_index_in_chunk,
                             bool force_no_eos,
                             bool is_final_text_chunk);

namespace {

enum class PerfOmniTokenType {
    NORMAL,
    SPEAK,
    LISTEN,
    CHUNK_EOS,
    CHUNK_TTS_EOS,
    TURN_EOS,
    TTS_EOS,
    EOS,
};

double elapsed_ms(std::chrono::high_resolution_clock::time_point begin,
                  std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

PerfOmniTokenType get_token_type(struct omni_context * ctx, llama_token token) {
    if (token == ctx->special_token_speak) {
        return PerfOmniTokenType::SPEAK;
    }
    if (token == ctx->special_token_listen) {
        return PerfOmniTokenType::LISTEN;
    }
    if (token == ctx->special_token_chunk_eos) {
        return PerfOmniTokenType::CHUNK_EOS;
    }
    if (token == ctx->special_token_chunk_tts_eos) {
        return PerfOmniTokenType::CHUNK_TTS_EOS;
    }
    if (token == ctx->special_token_turn_eos) {
        return PerfOmniTokenType::TURN_EOS;
    }
    if (token == ctx->special_token_tts_eos) {
        return PerfOmniTokenType::TTS_EOS;
    }
    if (token == ctx->special_token_eos) {
        return PerfOmniTokenType::EOS;
    }
    return PerfOmniTokenType::NORMAL;
}

bool is_end_token(struct omni_context * ctx, llama_token token) {
    const PerfOmniTokenType type = get_token_type(ctx, token);
    return type == PerfOmniTokenType::LISTEN ||
           type == PerfOmniTokenType::CHUNK_EOS ||
           type == PerfOmniTokenType::CHUNK_TTS_EOS;
}

bool is_valid_tts_token(llama_token tid) {
    // Keep this copy intentionally conservative: the perf LLM path only needs
    // tokens that can feed TTS condition construction.
    return tid >= 0 && tid < 150000;
}

bool perf_eval_tokens(struct omni_context * ctx, common_params * params,
                      const std::vector<llama_token> & tokens,
                      int n_batch,
                      int * n_past,
                      bool get_emb = false) {
    (void) params;
    const int n_tokens = (int) tokens.size();
    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) {
            break;
        }
        if (get_emb) {
            llama_set_embeddings(ctx->ctx_llama, true);
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tokens[i]), n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past + j;
        }
        if (llama_decode(ctx->ctx_llama, batch)) {
            std::fprintf(stderr, "%s: failed to eval token batch\n", __func__);
            if (get_emb) {
                llama_set_embeddings(ctx->ctx_llama, false);
            }
            return false;
        }
        if (get_emb) {
            llama_set_embeddings(ctx->ctx_llama, false);
        }
        *n_past += n_eval;
    }
    return true;
}

bool perf_eval_tokens_with_hidden(struct omni_context * ctx, common_params * params,
                                  const std::vector<llama_token> & tokens,
                                  int n_batch,
                                  int * n_past,
                                  float *& hidden_states) {
    (void) params;
    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0) {
        hidden_states = nullptr;
        return true;
    }

    const int n_embd = llama_model_n_embd(llama_get_model(ctx->ctx_llama));
    hidden_states = (float *) malloc((size_t) n_tokens * n_embd * sizeof(float));
    if (hidden_states == nullptr) {
        return false;
    }

    int processed = 0;
    for (int i = 0; i < n_tokens; i += n_batch) {
        int n_eval = std::min(n_tokens - i, n_batch);
        if (n_eval == 0) {
            break;
        }

        llama_set_embeddings(ctx->ctx_llama, true);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&tokens[i]), n_eval);
        std::vector<llama_pos> pos_vec;
        if (batch.pos == nullptr) {
            pos_vec.resize(n_eval);
            batch.pos = pos_vec.data();
        }
        for (int j = 0; j < n_eval; ++j) {
            batch.pos[j] = *n_past + j;
        }
        if (llama_decode(ctx->ctx_llama, batch)) {
            llama_set_embeddings(ctx->ctx_llama, false);
            free(hidden_states);
            hidden_states = nullptr;
            return false;
        }
        float * emb = llama_get_embeddings(ctx->ctx_llama);
        if (emb != nullptr) {
            std::memcpy(hidden_states + (size_t) processed * n_embd, emb, (size_t) n_eval * n_embd * sizeof(float));
        }
        llama_set_embeddings(ctx->ctx_llama, false);
        *n_past += n_eval;
        processed += n_eval;
    }
    return true;
}

bool perf_eval_string(struct omni_context * ctx, common_params * params,
                      const char * text, int n_batch, int * n_past, bool add_bos) {
    std::vector<llama_token> tokens = common_tokenize(ctx->ctx_llama, text, add_bos, true);
    return perf_eval_tokens(ctx, params, tokens, n_batch, n_past);
}

bool perf_eval_id_with_hidden(struct omni_context * ctx, common_params * params,
                              llama_token id, int * n_past, float *& hidden_states) {
    std::vector<llama_token> tokens = {id};
    return perf_eval_tokens_with_hidden(ctx, params, tokens, 1, n_past, hidden_states);
}

const char * perf_sample_with_hidden_and_token(struct common_sampler * smpl,
                                               struct omni_context * ctx,
                                               common_params * params,
                                               int * n_past,
                                               float *& hidden_states,
                                               llama_token & token_id) {
    float * logits = llama_get_logits_ith(ctx->ctx_llama, -1);
    if (ctx->duplex_mode && logits != nullptr) {
        if (ctx->special_token_listen >= 0) {
            const float listen_bias = (ctx->listen_prob_scale - 1.0f) * 2.0f;
            logits[ctx->special_token_listen] += listen_bias;
        }
        if (ctx->special_token_tts_pad >= 0) {
            logits[ctx->special_token_tts_pad] = -INFINITY;
        }
        if (ctx->length_penalty != 1.0f && ctx->special_token_turn_eos >= 0) {
            float eos_logit = logits[ctx->special_token_turn_eos];
            logits[ctx->special_token_turn_eos] = eos_logit > 0 ? eos_logit / ctx->length_penalty
                                                                : eos_logit * ctx->length_penalty;
        }
    }

    const llama_token id = common_sampler_sample(smpl, ctx->ctx_llama, -1);
    token_id = id;
    common_sampler_accept(smpl, id, true);

    static std::string piece;
    if (llama_vocab_is_eog(llama_model_get_vocab(llama_get_model(ctx->ctx_llama)), id)) {
        piece = "</s>";
    } else {
        piece = common_token_to_piece(ctx->ctx_llama, id);
    }
    perf_eval_id_with_hidden(ctx, params, id, n_past, hidden_states);
    return piece.c_str();
}

std::string clean_response_text(std::string response) {
    static const std::vector<std::string> end_tokens = {
        "<|tts_eos|>", "</s>", "<|listen|>", "<|turn_eos|>",
        "<|chunk_eos|>", "<|chunk_tts_eos|>"
    };
    for (const auto & token : end_tokens) {
        const size_t pos = response.find(token);
        if (pos != std::string::npos) {
            response = response.substr(0, pos);
        }
    }
    size_t speak_pos = response.find("<|speak|>");
    while (speak_pos != std::string::npos) {
        response.erase(speak_pos, std::string("<|speak|>").length());
        speak_pos = response.find("<|speak|>");
    }
    return response;
}

std::string frame_detail(int64_t frame_id, int64_t user_seq) {
    std::ostringstream ss;
    ss << "frame_id=" << frame_id << ",user_seq=" << user_seq;
    return ss.str();
}

bool emb_text_lookup(struct omni_context * ctx, llama_token token, std::vector<float> & out, int tts_n_embd) {
    if (ctx->emb_text_weight == nullptr || token < 0 || token >= ctx->emb_text_vocab_size) {
        return false;
    }
    out.resize(tts_n_embd);
    const float * src = ctx->emb_text_weight + (size_t) token * tts_n_embd;
    std::copy(src, src + tts_n_embd, out.begin());
    return true;
}

void write_wav_file(const std::string & path, const std::vector<float> & wav, int sample_rate) {
    std::vector<int16_t> pcm(wav.size());
    for (size_t i = 0; i < wav.size(); ++i) {
        float x = wav[i];
        if (!std::isfinite(x)) {
            x = 0.0f;
        }
        x = std::max(-1.0f, std::min(1.0f, x));
        pcm[i] = (int16_t) (x * 32767.0f);
    }

    const int16_t num_channels = 1;
    const int16_t bits_per_sample = 16;
    const int16_t block_align = num_channels * (bits_per_sample / 8);
    const int32_t byte_rate = sample_rate * block_align;
    const uint32_t data_bytes = (uint32_t) (pcm.size() * sizeof(int16_t));
    const uint32_t riff_size = 36u + data_bytes;

    FILE * f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return;
    }
    std::fwrite("RIFF", 1, 4, f);
    std::fwrite(&riff_size, 4, 1, f);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;
    std::fwrite(&fmt_size, 4, 1, f);
    std::fwrite(&audio_format, 2, 1, f);
    std::fwrite(&num_channels, 2, 1, f);
    std::fwrite(&sample_rate, 4, 1, f);
    std::fwrite(&byte_rate, 4, 1, f);
    std::fwrite(&block_align, 2, 1, f);
    std::fwrite(&bits_per_sample, 2, 1, f);
    std::fwrite("data", 1, 4, f);
    std::fwrite(&data_bytes, 4, 1, f);
    std::fwrite(pcm.data(), 1, data_bytes, f);
    std::fclose(f);
}

} // namespace

OmniDuplexPerfSession::~OmniDuplexPerfSession() {
    end();
}

bool OmniDuplexPerfSession::begin(struct omni_context * ctx_omni,
                                  const std::string & voice_audio,
                                  const std::string & debug_dir_) {
    if (ctx_omni == nullptr) {
        return false;
    }
    ctx = ctx_omni;
    params = ctx_omni->params;
    debug_dir = debug_dir_.empty() ? std::string("./") : debug_dir_;
    running.store(true);

    omni_gpu_perf_sampler_start();
    // Reuse the existing system prompt initialization for now. Subsequent frame
    // prefill/decode goes through this perf pipeline instead of omni.cpp's
    // duplex_prefill / duplex_decode route.
    const bool old_async = ctx->async;
    ctx->async = false;
    if (!stream_prefill(ctx, voice_audio, "", 0)) {
        ctx->async = old_async;
        running.store(false);
        omni_gpu_perf_sampler_stop();
        return false;
    }
    ctx->async = old_async;

    encoder_thread = std::thread(&OmniDuplexPerfSession::encoder_thread_func, this);
    llm_thread = std::thread(&OmniDuplexPerfSession::llm_thread_func, this);
    if (ctx->use_tts) {
        tts_thread = std::thread(&OmniDuplexPerfSession::tts_thread_func, this);
        t2w_thread = std::thread(&OmniDuplexPerfSession::t2w_thread_func, this);
    }
    return true;
}

int64_t OmniDuplexPerfSession::push_frame(const OmniDuplexFrame & frame) {
    if (!running.load()) {
        return -1;
    }

    auto * req = new EncodeReq();
    req->frame = frame;
    req->frame_id = frame_id_counter.fetch_add(1) + 1;
    req->t_push = std::chrono::high_resolution_clock::now();

    {
        std::unique_lock<std::mutex> lock(encode_mtx);
        encode_cv.wait(lock, [&] {
            return encode_queue.size() < ENCODE_QUEUE_CAP || !running.load();
        });
        if (!running.load()) {
            delete req;
            return -1;
        }
        encode_queue.push(req);
    }

    in_flight.fetch_add(1);
    encode_cv.notify_all();
    return req->frame_id;
}

bool OmniDuplexPerfSession::wait_next_frame(OmniDuplexFrameResult * out, int timeout_ms) {
    if (out == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(done_mtx);
    if (timeout_ms < 0) {
        done_cv.wait(lock, [&] { return !done_results.empty() || !running.load(); });
    } else if (timeout_ms == 0) {
        if (done_results.empty()) {
            return false;
        }
    } else {
        bool got = done_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return !done_results.empty() || !running.load();
        });
        if (!got) {
            return false;
        }
    }

    if (done_results.empty()) {
        return false;
    }
    *out = std::move(done_results.front());
    done_results.pop();
    return true;
}

void OmniDuplexPerfSession::end() {
    if (!running.exchange(false)) {
        return;
    }
    while (in_flight.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    encode_cv.notify_all();
    llm_cv.notify_all();
    tts_cv.notify_all();
    t2w_cv.notify_all();
    done_cv.notify_all();
    if (encoder_thread.joinable()) {
        encoder_thread.join();
    }
    if (llm_thread.joinable()) {
        llm_thread.join();
    }
    if (tts_thread.joinable()) {
        tts_thread.join();
    }
    if (t2w_thread.joinable()) {
        t2w_thread.join();
    }
    omni_perf_print_token_stats(ctx);
    omni_gpu_perf_sampler_stop();
}

void OmniDuplexPerfSession::encoder_thread_func() {
    const int hidden_size = llama_model_n_embd(llama_get_model(ctx->ctx_llama));
    while (running.load()) {
        EncodeReq * req = nullptr;
        {
            std::unique_lock<std::mutex> lock(encode_mtx);
            encode_cv.wait(lock, [&] { return !encode_queue.empty() || !running.load(); });
            if (!running.load() && encode_queue.empty()) {
                break;
            }
            req = encode_queue.front();
            encode_queue.pop();
        }
        encode_cv.notify_all();

        auto * packet = new PrefillPacket();
        packet->frame_id = req->frame_id;
        packet->user_seq = req->frame.user_seq;
        packet->t_push = req->t_push;

        const bool has_img = !req->frame.img_fname.empty() && ctx->ctx_vision != nullptr;
        const bool has_aud = !req->frame.aud_fname.empty();
        const int perf_chunk_index = (int) req->frame_id;
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::string start_detail = frame_detail(req->frame_id, req->frame.user_seq) +
                                   ",has_img=" + std::to_string(has_img ? 1 : 0) +
                                   ",has_aud=" + std::to_string(has_aud ? 1 : 0);
        omni_perf_mark(ctx, "duplex.encode", "start", perf_chunk_index, -1.0, start_detail.c_str());

        if (has_img && req->frame.max_slice_nums >= 1) {
            vision_set_max_slice_nums(ctx->ctx_vision, req->frame.max_slice_nums);
        }
        if (has_img) {
            omni_image_embed_make_chunks_with_filename(ctx->ctx_vision,
                                                       params->cpuparams.n_threads,
                                                       req->frame.img_fname,
                                                       packet->vision_embed);
        }
        if (has_aud) {
            auto * audio = omni_audio_embed_make_with_filename(ctx->ctx_audio,
                                                               params->cpuparams.n_threads,
                                                               req->frame.aud_fname);
            if (audio != nullptr && audio->n_pos > 0) {
                packet->audio_embed.resize((size_t) audio->n_pos * hidden_size);
                std::memcpy(packet->audio_embed.data(), audio->embed, packet->audio_embed.size() * sizeof(float));
                omni_embed_free(audio);
            }
        }
        packet->t_encoded = std::chrono::high_resolution_clock::now();
        const double enc_ms = elapsed_ms(t0, packet->t_encoded);
        std::string end_detail = start_detail +
                                 ",vision_chunks=" + std::to_string(packet->vision_embed.size()) +
                                 ",audio_tokens=" + std::to_string(hidden_size > 0 ? (int) packet->audio_embed.size() / hidden_size : 0);
        omni_perf_mark(ctx, "duplex.encode", "end", perf_chunk_index, enc_ms, end_detail.c_str());

        delete req;
        {
            std::unique_lock<std::mutex> lock(llm_mtx);
            llm_cv.wait(lock, [&] { return prefill_queue.size() < PREFILL_QUEUE_CAP || !running.load(); });
            if (!running.load()) {
                delete packet;
                break;
            }
            prefill_queue.push(packet);
        }
        llm_cv.notify_all();
    }
}

bool OmniDuplexPerfSession::do_prefill(PrefillPacket * packet, double & prefill_ms, long long & prefill_tokens) {
    const int hidden_size = llama_model_n_embd(llama_get_model(ctx->ctx_llama));
    const int n_past_before = ctx->n_past;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int perf_chunk_index = (int) packet->frame_id;
    std::string detail = frame_detail(packet->frame_id, packet->user_seq);
    omni_perf_mark(ctx, "duplex.llm.prefill", "start", perf_chunk_index, -1.0, detail.c_str());

    if (ctx->sliding_window_config.mode != "off") {
        sliding_window_register_unit_start(ctx);
    }

    bool ok = true;
    const bool has_vision = !packet->vision_embed.empty();
    const int n_audio_tokens = hidden_size > 0 ? (int) packet->audio_embed.size() / hidden_size : 0;
    const bool has_audio = n_audio_tokens > 0;

    if (has_vision) {
        const int n_chunks = (int) packet->vision_embed.size();
        const int tokens_per_chunk = (int) packet->vision_embed[0].size() / hidden_size;
        ok &= perf_eval_string(ctx, params, "<unit><image>", params->n_batch, &ctx->n_past, false);
        ok &= prefill_with_emb(ctx, params, packet->vision_embed[0].data(), tokens_per_chunk, params->n_batch, &ctx->n_past);
        ok &= perf_eval_string(ctx, params, "</image>", params->n_batch, &ctx->n_past, false);
        for (int i = 1; ok && i < n_chunks; ++i) {
            ok &= perf_eval_string(ctx, params, "<slice>", params->n_batch, &ctx->n_past, false);
            ok &= prefill_with_emb(ctx, params, packet->vision_embed[i].data(), tokens_per_chunk, params->n_batch, &ctx->n_past);
            ok &= perf_eval_string(ctx, params, "</slice>", params->n_batch, &ctx->n_past, false);
        }
        if (ok && n_chunks > 1) {
            ok &= perf_eval_string(ctx, params, "\n", params->n_batch, &ctx->n_past, false);
        }
        if (ok && has_audio) {
            ok &= prefill_with_emb(ctx, params, packet->audio_embed.data(), n_audio_tokens, params->n_batch, &ctx->n_past);
        }
    } else {
        ok &= perf_eval_string(ctx, params, "<unit>", params->n_batch, &ctx->n_past, false);
        if (ok && has_audio) {
            ok &= prefill_with_emb(ctx, params, packet->audio_embed.data(), n_audio_tokens, params->n_batch, &ctx->n_past);
        }
    }

    if (ctx->sliding_window_config.mode != "off") {
        std::string unit_type = "audio";
        if (has_vision) {
            unit_type = has_audio ? "omni" : "video";
        }
        sliding_window_register_unit_end(ctx, unit_type);
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    prefill_ms = elapsed_ms(t0, t1);
    prefill_tokens = ctx->n_past - n_past_before;
    std::string end_detail = detail +
                             ",tokens=" + std::to_string(prefill_tokens) +
                             omni_perf_speed_detail(prefill_tokens, prefill_ms);
    omni_perf_record_tokens(ctx, "duplex.llm.prefill", prefill_tokens, prefill_ms);
    omni_perf_mark(ctx, "duplex.llm.prefill", ok ? "end" : "error", perf_chunk_index, prefill_ms, end_detail.c_str());
    return ok;
}

bool OmniDuplexPerfSession::do_decode(const DecodeReq & req, OmniDuplexFrameResult & result) {
    const int perf_chunk_index = (int) req.frame_id;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const int n_past_before = ctx->n_past;
    const int llm_n_embd = llama_model_n_embd(llama_get_model(ctx->ctx_llama));
    result.user_seq = req.user_seq;
    result.frame_id = req.frame_id;

    std::string start_detail = frame_detail(req.frame_id, req.user_seq);
    omni_perf_mark(ctx, "duplex.llm.decode", "start", perf_chunk_index, -1.0, start_detail.c_str());

    ctx->stream_decode_start_time = t0;
    ctx->ended_with_listen = false;
    if (ctx->break_event.load()) {
        ctx->break_event.store(false);
    }
    {
        std::lock_guard<std::mutex> lock(ctx->text_mtx);
        ctx->text_queue.clear();
        ctx->text_done_flag = false;
        ctx->text_streaming = true;
    }
    if (ctx->use_tts) {
        ctx->speek_done = false;
    }

    if (ctx->force_listen_used < ctx->force_listen_count) {
        ctx->force_listen_used++;
        ctx->ended_with_listen = true;
        ctx->slide_last_was_listen = true;
        ctx->current_turn_ended = false;
        if (ctx->use_tts) {
            ctx->speek_done = true;
        }
        {
            std::lock_guard<std::mutex> lock(ctx->text_mtx);
            ctx->text_queue.push_back("__IS_LISTEN__");
            ctx->text_done_flag = true;
            ctx->text_streaming = false;
            ctx->text_cv.notify_all();
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        result.ok = true;
        result.is_speak = false;
        result.n_past_after = ctx->n_past;
        result.ms_decode = elapsed_ms(t0, t1);
        omni_perf_mark(ctx, "duplex.llm.decode", "end", perf_chunk_index, result.ms_decode,
                       (start_detail + ",tokens=0,is_speak=0,force_listen=1").c_str());
        return true;
    }

    const int max_tgt_len = params->n_predict < 0 ? params->n_ctx : params->n_predict;
    const int step_size = 10;
    bool llm_finish = false;
    bool local_is_end_of_turn = false;
    int current_chunk_tokens = 0;
    const int max_chunk_tokens = ctx->max_new_speak_tokens_per_chunk;
    bool chunk_limit_reached = false;
    const int decode_start_cache_len = ctx->n_past;
    std::string response;

    for (int il = 0; il < max_tgt_len && !llm_finish; ) {
        if (ctx->break_event.load()) {
            llm_finish = true;
            break;
        }

        response.clear();
        int jl = 0;
        int total_tokens_generated = 0;
        std::vector<llama_token> chunk_token_ids;
        std::vector<float> chunk_hidden_states;
        local_is_end_of_turn = false;

        while (jl < step_size && !llm_finish && !ctx->break_event.load() && !chunk_limit_reached) {
            float * hidden_states = nullptr;
            llama_token sampled_token = 0;
            const char * piece = perf_sample_with_hidden_and_token(ctx->ctx_sampler, ctx, params,
                                                                   &ctx->n_past, hidden_states, sampled_token);
            total_tokens_generated++;
            if (piece == nullptr) {
                break;
            }
            if (hidden_states != nullptr && is_valid_tts_token(sampled_token)) {
                chunk_token_ids.push_back(sampled_token);
                chunk_hidden_states.insert(chunk_hidden_states.end(), hidden_states, hidden_states + llm_n_embd);
                jl++;
                current_chunk_tokens++;
                if (max_chunk_tokens > 0 && current_chunk_tokens >= max_chunk_tokens) {
                    chunk_limit_reached = true;
                }
            }

            const PerfOmniTokenType token_type = get_token_type(ctx, sampled_token);
            if (token_type == PerfOmniTokenType::TURN_EOS ||
                token_type == PerfOmniTokenType::TTS_EOS ||
                token_type == PerfOmniTokenType::EOS) {
                local_is_end_of_turn = true;
                ctx->current_turn_ended = true;
            } else if (token_type == PerfOmniTokenType::LISTEN && !ctx->slide_last_was_listen.load()) {
                local_is_end_of_turn = true;
            }
            if (is_end_token(ctx, sampled_token)) {
                llm_finish = true;
                if (token_type == PerfOmniTokenType::LISTEN) {
                    ctx->ended_with_listen = true;
                    ctx->slide_last_was_listen = true;
                    std::lock_guard<std::mutex> lock(ctx->text_mtx);
                    ctx->text_queue.push_back("__IS_LISTEN__");
                    ctx->text_cv.notify_all();
                } else {
                    ctx->slide_last_was_listen = false;
                }
                break;
            }
            response += piece;
        }

        if (chunk_limit_reached) {
            if (ctx->special_token_chunk_eos >= 0) {
                std::vector<llama_token> chunk_eos_tokens = {ctx->special_token_chunk_eos};
                perf_eval_tokens(ctx, params, chunk_eos_tokens, params->n_batch, &ctx->n_past);
            }
            llm_finish = true;
            current_chunk_tokens = 0;
        }

        if (ctx->special_token_unit_end >= 0) {
            std::vector<llama_token> unit_end = {ctx->special_token_unit_end};
            perf_eval_tokens(ctx, params, unit_end, params->n_batch, &ctx->n_past);
        }

        il += total_tokens_generated;
        response = clean_response_text(response);
        if (!response.empty()) {
            std::lock_guard<std::mutex> lock(ctx->text_mtx);
            ctx->text_queue.push_back(response);
            ctx->text_cv.notify_all();
            result.text += response;
        }

        if (ctx->use_tts && (!response.empty() || llm_finish)) {
            auto * llm_out = new PerfLLMOut();
            llm_out->text = response;
            llm_out->n_past = ctx->n_past;
            llm_out->llm_finish = llm_finish;
            llm_out->debug_dir = debug_dir;
            llm_out->token_ids = chunk_token_ids;
            llm_out->hidden_states = chunk_hidden_states;
            llm_out->n_embd = llm_n_embd;
            llm_out->is_end_of_turn = local_is_end_of_turn;
            llm_out->perf_chunk_index = perf_chunk_index;
            {
                std::unique_lock<std::mutex> lock(tts_mtx);
                tts_cv.wait(lock, [&] {
                    return tts_queue.size() < TTS_QUEUE_CAP || !running.load();
                });
                tts_queue.push(llm_out);
            }
            tts_cv.notify_all();
        }
    }

    {
        std::lock_guard<std::mutex> lock(ctx->text_mtx);
        if (!ctx->ended_with_listen) {
            ctx->text_queue.push_back("__END_OF_TURN__");
        }
        ctx->text_done_flag = true;
        ctx->text_streaming = false;
        ctx->text_cv.notify_all();
    }

    if (ctx->sliding_window_config.mode != "off") {
        int response_len = ctx->n_past - decode_start_cache_len;
        if (response_len > 0) {
            UnitEntry entry;
            entry.unit_id = ctx->next_unit_id++;
            entry.length = response_len;
            entry.type = "response";
            entry.is_listen = ctx->ended_with_listen.load();
            entry.turn_id = ctx->current_turn_id;
            ctx->unit_history.push_back(entry);
        }
    }

    if (ctx->ended_with_listen) {
        ctx->round_start_positions.push_back(ctx->n_past);
        ctx->current_turn_id++;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    result.ok = true;
    result.is_speak = !ctx->ended_with_listen.load();
    result.n_past_after = ctx->n_past;
    result.ms_decode = elapsed_ms(t0, t1);
    const long long decode_tokens = ctx->n_past - n_past_before;
    std::string end_detail = start_detail +
                             ",tokens=" + std::to_string(decode_tokens) +
                             ",is_speak=" + std::to_string(result.is_speak ? 1 : 0) +
                             omni_perf_speed_detail(decode_tokens, result.ms_decode);
    omni_perf_record_tokens(ctx, "duplex.llm.decode", decode_tokens, result.ms_decode);
    omni_perf_mark(ctx, "duplex.llm.decode", "end", perf_chunk_index, result.ms_decode, end_detail.c_str());
    return true;
}

bool OmniDuplexPerfSession::generate_audio_tokens(const std::vector<float> & merged_embeddings,
                                                  int n_tokens,
                                                  int tts_n_embd,
                                                  int tts_chunk_idx,
                                                  bool is_end_of_turn,
                                                  int perf_chunk_index) {
    const int audio_bos_token_id = 151687;
    const int text_eos_token_id = 151692;
    const int num_audio_tokens = 6562;
    std::vector<float> condition = merged_embeddings;
    if (is_end_of_turn) {
        std::vector<float> text_eos;
        if (emb_text_lookup(ctx, text_eos_token_id, text_eos, tts_n_embd)) {
            condition.insert(condition.end(), text_eos.begin(), text_eos.end());
            n_tokens += 1;
        }
    }
    std::vector<float> audio_bos;
    if (emb_text_lookup(ctx, audio_bos_token_id, audio_bos, tts_n_embd)) {
        condition.insert(condition.end(), audio_bos.begin(), audio_bos.end());
        n_tokens += 1;
    }

    if (n_tokens <= 0 || condition.empty()) {
        return false;
    }

    if (tts_chunk_idx == 0) {
        llama_memory_t mem = llama_get_memory(ctx->ctx_tts_llama);
        if (mem) {
            llama_memory_seq_rm(mem, 0, 0, -1);
        }
        ctx->tts_n_past_accumulated = 0;
        ctx->tts_all_generated_tokens.clear();
        ctx->tts_condition_saved = false;
    }

    ctx->tts_condition_embeddings = condition;
    ctx->tts_condition_length = n_tokens;
    ctx->tts_condition_n_embd = tts_n_embd;
    ctx->tts_condition_saved = true;

    int n_past_tts = tts_chunk_idx == 0 ? 0 : ctx->tts_n_past_accumulated;
    std::string prefill_detail = "tts_chunk=" + std::to_string(tts_chunk_idx) +
                                 ",tokens=" + std::to_string(n_tokens) +
                                 ",is_end_of_turn=" + std::to_string(is_end_of_turn ? 1 : 0);
    {
        OmniPerfScope scope(ctx, "tts.prefill", perf_chunk_index, prefill_detail);
        scope.set_tokens(n_tokens);
        if (!prefill_with_emb_tts(ctx, params, condition.data(), n_tokens, params->n_batch, &n_past_tts)) {
            return false;
        }
    }

    common_params_sampling tts_sampling = params->sampling;
    tts_sampling.temp = ctx->tts_temperature;
    tts_sampling.top_p = 0.85f;
    tts_sampling.top_k = 25;
    tts_sampling.penalty_repeat = 1.05f;
    tts_sampling.min_p = 0.01f;
    tts_sampling.penalty_last_n = 16;
    common_sampler * sampler = common_sampler_init(ctx->model_tts, tts_sampling);
    if (sampler == nullptr) {
        return false;
    }

    const int min_new_tokens = is_end_of_turn ? 0 : 26;
    const int max_audio_tokens = is_end_of_turn ? 100 : 26;
    std::vector<llama_token> chunk_generated_tokens;
    std::vector<int32_t> stream_buffer;
    bool first_chunk_pushed = false;
    bool produced = false;

    std::string decode_detail = "tts_chunk=" + std::to_string(tts_chunk_idx) +
                                ",max_audio_tokens=" + std::to_string(max_audio_tokens) +
                                ",is_end_of_turn=" + std::to_string(is_end_of_turn ? 1 : 0);
    {
        OmniPerfScope scope(ctx, "tts.decode", perf_chunk_index, decode_detail);
        for (int t = 0; t < max_audio_tokens; ++t) {
            const bool force_no_eos = t < min_new_tokens;
            llama_token sampled = sample_tts_token(sampler, ctx, params, &n_past_tts,
                                                   &ctx->tts_all_generated_tokens,
                                                   &chunk_generated_tokens,
                                                   t,
                                                   force_no_eos,
                                                   is_end_of_turn);
            if (sampled == 0) {
                break;
            }
            int relative_idx = sampled - audio_bos_token_id;
            if (relative_idx < 0 || relative_idx >= num_audio_tokens) {
                break;
            }
            const bool is_eos = relative_idx == num_audio_tokens - 1;
            if (!is_eos) {
                stream_buffer.push_back(relative_idx);
                ctx->tts_all_generated_tokens.push_back(sampled);
                chunk_generated_tokens.push_back(sampled);
                produced = true;
            }

            const int push_threshold = first_chunk_pushed ? 25 : 28;
            if ((int) stream_buffer.size() >= push_threshold && !is_end_of_turn) {
                first_chunk_pushed = true;
                auto * out = new PerfT2WOut();
                out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
                out->is_final = false;
                out->is_chunk_end = false;
                out->round_idx = ctx->simplex_round_idx;
                out->perf_chunk_index = perf_chunk_index;
                {
                    std::lock_guard<std::mutex> lock(t2w_mtx);
                    t2w_queue.push(out);
                }
                t2w_cv.notify_all();
                stream_buffer.clear();
            }
            if (is_eos) {
                break;
            }
        }
        scope.set_tokens((long long) chunk_generated_tokens.size());
    }

    {
        auto * out = new PerfT2WOut();
        out->audio_tokens.assign(stream_buffer.begin(), stream_buffer.end());
        out->is_final = is_end_of_turn;
        out->is_chunk_end = !is_end_of_turn;
        out->round_idx = ctx->simplex_round_idx;
        out->perf_chunk_index = perf_chunk_index;
        {
            std::lock_guard<std::mutex> lock(t2w_mtx);
            t2w_queue.push(out);
        }
        t2w_cv.notify_all();
    }

    ctx->tts_n_past_accumulated = n_past_tts;
    common_sampler_free(sampler);
    return produced || is_end_of_turn;
}

void OmniDuplexPerfSession::tts_thread_func() {
    int tts_chunk_idx = 0;
    while (running.load() || !tts_queue.empty()) {
        std::vector<PerfLLMOut *> items;
        {
            std::unique_lock<std::mutex> lock(tts_mtx);
            tts_cv.wait(lock, [&] { return !tts_queue.empty() || !running.load(); });
            while (!tts_queue.empty()) {
                PerfLLMOut * item = tts_queue.front();
                tts_queue.pop();
                items.push_back(item);
                if (item->is_end_of_turn) {
                    break;
                }
            }
        }
        if (items.empty()) {
            continue;
        }

        std::vector<llama_token> token_ids;
        std::vector<float> hidden_states;
        int n_embd = 0;
        bool is_end_of_turn = false;
        int perf_chunk_index = -1;
        for (PerfLLMOut * item : items) {
            if (perf_chunk_index < 0) {
                perf_chunk_index = item->perf_chunk_index;
            }
            is_end_of_turn = is_end_of_turn || item->is_end_of_turn;
            if (!item->token_ids.empty() && !item->hidden_states.empty()) {
                token_ids.insert(token_ids.end(), item->token_ids.begin(), item->token_ids.end());
                hidden_states.insert(hidden_states.end(), item->hidden_states.begin(), item->hidden_states.end());
                n_embd = item->n_embd;
            }
            delete item;
        }

        if (token_ids.empty() || hidden_states.empty() || n_embd <= 0) {
            if (is_end_of_turn) {
                auto * out = new PerfT2WOut();
                out->is_final = true;
                out->round_idx = ctx->simplex_round_idx;
                out->perf_chunk_index = perf_chunk_index;
                {
                    std::lock_guard<std::mutex> lock(t2w_mtx);
                    t2w_queue.push(out);
                }
                t2w_cv.notify_all();
            }
            continue;
        }

        std::vector<float> merged_embeddings;
        int tts_n_embd = 0;
        std::string condition_detail = "tts_chunk=" + std::to_string(tts_chunk_idx) +
                                       ",llm_tokens=" + std::to_string(token_ids.size()) +
                                       ",is_end_of_turn=" + std::to_string(is_end_of_turn ? 1 : 0);
        const auto t0 = std::chrono::steady_clock::now();
        omni_perf_mark(ctx, "tts.condition", "start", perf_chunk_index, -1.0, condition_detail.c_str());
        if (!ctx->tts_condition_graph.initialized) {
            tts_condition_graph_init(ctx);
        }
        bool ok = tts_condition_graph_forward(ctx, token_ids.data(), hidden_states.data(),
                                              (int) token_ids.size(), n_embd,
                                              merged_embeddings, tts_n_embd);
        const double condition_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        std::string end_detail = condition_detail +
                                 ",condition_tokens=" + std::to_string(ok ? (int) token_ids.size() : 0) +
                                 ",tokens=" + std::to_string(ok ? (int) token_ids.size() : 0) +
                                 ",merged_success=" + std::to_string(ok ? 1 : 0);
        omni_perf_mark(ctx, "tts.condition", "end", perf_chunk_index, condition_ms, end_detail.c_str());
        if (ok) {
            generate_audio_tokens(merged_embeddings, (int) token_ids.size(), tts_n_embd,
                                  tts_chunk_idx, is_end_of_turn, perf_chunk_index);
            ++tts_chunk_idx;
        }
    }
}

void OmniDuplexPerfSession::t2w_thread_func() {
    const int sample_rate = 24000;
    const int chunk_size = 25;
    const int pre_lookahead = 3;
    int wav_idx = 0;
    std::vector<int32_t> token_buffer = {4218, 4218, 4218};
    std::string wav_dir = ctx->base_output_dir + "/tts_wav";
    std::filesystem::create_directories(wav_dir);

    while (running.load() || !t2w_queue.empty()) {
        std::vector<int32_t> new_tokens;
        bool is_final = false;
        bool is_chunk_end = false;
        int perf_chunk_index = -1;
        {
            std::unique_lock<std::mutex> lock(t2w_mtx);
            t2w_cv.wait(lock, [&] { return !t2w_queue.empty() || !running.load(); });
            while (!t2w_queue.empty()) {
                PerfT2WOut * item = t2w_queue.front();
                t2w_queue.pop();
                new_tokens.insert(new_tokens.end(), item->audio_tokens.begin(), item->audio_tokens.end());
                is_final = is_final || item->is_final;
                is_chunk_end = is_chunk_end || item->is_chunk_end;
                if (perf_chunk_index < 0) {
                    perf_chunk_index = item->perf_chunk_index;
                }
                delete item;
            }
        }

        if (new_tokens.empty() && !is_chunk_end && !is_final) {
            continue;
        }

        token_buffer.insert(token_buffer.end(), new_tokens.begin(), new_tokens.end());
        const bool need_flush = is_final;
        const size_t window_size = (size_t) (chunk_size + pre_lookahead);
        while (token_buffer.size() >= window_size || (need_flush && !token_buffer.empty())) {
            const size_t process_size = std::min(token_buffer.size(), window_size);
            const bool is_last_window = is_final && token_buffer.size() <= window_size;
            std::vector<int32_t> window(token_buffer.begin(), token_buffer.begin() + process_size);

            std::string detail = "backend=cpp,window_tokens=" + std::to_string(window.size()) +
                                 ",is_last=" + std::to_string(is_last_window ? 1 : 0);
            auto t0 = std::chrono::steady_clock::now();
            omni_perf_mark(ctx, "t2w.infer", "start", perf_chunk_index, -1.0, detail.c_str());
            std::vector<float> wav;
            bool ok = ctx->token2wav_session && ctx->token2wav_session->feed_window(window, is_last_window, wav);
            const double infer_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            omni_perf_mark(ctx, "t2w.infer", "end", perf_chunk_index, infer_ms,
                           (detail + ",ok=" + std::to_string(ok ? 1 : 0) + ",wav_samples=" + std::to_string(wav.size())).c_str());

            if (ok && !wav.empty()) {
                const std::string wav_path = wav_dir + "/wav_" + std::to_string(ctx->wav_turn_base + wav_idx) + ".wav";
                std::string write_detail = "backend=cpp,wav_samples=" + std::to_string(wav.size()) +
                                           ",path=" + wav_path;
                auto tw = std::chrono::steady_clock::now();
                omni_perf_mark(ctx, "t2w.write", "start", perf_chunk_index, -1.0, write_detail.c_str());
                write_wav_file(wav_path, wav, sample_rate);
                const double write_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tw).count();
                omni_perf_mark(ctx, "t2w.write", "end", perf_chunk_index, write_ms, write_detail.c_str());
                ++wav_idx;
            }

            size_t slide = 0;
            if (is_last_window) {
                slide = token_buffer.size();
            } else if (token_buffer.size() > (size_t) chunk_size) {
                slide = (size_t) chunk_size;
            } else if (token_buffer.size() > (size_t) pre_lookahead) {
                slide = token_buffer.size() - (size_t) pre_lookahead;
            }
            if (slide > 0 && slide <= token_buffer.size()) {
                token_buffer.erase(token_buffer.begin(), token_buffer.begin() + slide);
            } else if (slide > token_buffer.size()) {
                token_buffer.clear();
            }
            if (is_last_window) {
                if (is_final) {
                    token_buffer = {4218, 4218, 4218};
                }
                break;
            }
        }
    }
}

void OmniDuplexPerfSession::llm_thread_func() {
    while (running.load()) {
        PrefillPacket * packet = nullptr;
        {
            std::unique_lock<std::mutex> lock(llm_mtx);
            llm_cv.wait(lock, [&] { return !prefill_queue.empty() || !running.load(); });
            if (!running.load() && prefill_queue.empty()) {
                break;
            }
            packet = prefill_queue.front();
            prefill_queue.pop();
        }

        double prefill_ms = 0.0;
        long long prefill_tokens = 0;
        bool ok = do_prefill(packet, prefill_ms, prefill_tokens);

        DecodeReq req;
        req.frame_id = packet->frame_id;
        req.user_seq = packet->user_seq;
        req.t_push = packet->t_push;
        req.t_prefilled = std::chrono::high_resolution_clock::now();
        delete packet;

        OmniDuplexFrameResult result;
        result.ms_prefill_submit = elapsed_ms(req.t_push, req.t_prefilled);
        if (ok) {
            ok = do_decode(req, result);
        } else {
            result.ok = false;
        }
        result.ms_total = elapsed_ms(req.t_push, std::chrono::high_resolution_clock::now());

        {
            std::lock_guard<std::mutex> lock(done_mtx);
            done_results.push(std::move(result));
        }
        done_cv.notify_all();
        in_flight.fetch_sub(1);
    }
}
