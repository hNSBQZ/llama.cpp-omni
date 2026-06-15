#pragma once

#include "omni.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class OmniDuplexPerfSession {
public:
    OmniDuplexPerfSession() = default;
    ~OmniDuplexPerfSession();

    OmniDuplexPerfSession(const OmniDuplexPerfSession &) = delete;
    OmniDuplexPerfSession & operator=(const OmniDuplexPerfSession &) = delete;

    bool begin(struct omni_context * ctx_omni,
               const std::string & voice_audio,
               const std::string & debug_dir = "./");
    int64_t push_frame(const OmniDuplexFrame & frame);
    bool wait_next_frame(OmniDuplexFrameResult * out, int timeout_ms = -1);
    void end();

private:
    struct EncodeReq {
        OmniDuplexFrame frame;
        int64_t frame_id = -1;
        std::chrono::high_resolution_clock::time_point t_push;
    };

    struct PrefillPacket {
        std::vector<std::vector<float>> vision_embed;
        std::vector<float> audio_embed;
        int64_t frame_id = -1;
        int64_t user_seq = 0;
        std::chrono::high_resolution_clock::time_point t_push;
        std::chrono::high_resolution_clock::time_point t_encoded;
    };

    struct DecodeReq {
        int64_t frame_id = -1;
        int64_t user_seq = 0;
        std::chrono::high_resolution_clock::time_point t_push;
        std::chrono::high_resolution_clock::time_point t_prefilled;
    };

    struct PerfLLMOut {
        std::string text;
        int n_past = 0;
        bool llm_finish = false;
        std::string debug_dir;
        std::vector<llama_token> token_ids;
        std::vector<float> hidden_states;
        int n_embd = 0;
        bool is_end_of_turn = false;
        int perf_chunk_index = -1;
    };

    struct PerfT2WOut {
        std::vector<int32_t> audio_tokens;
        bool is_final = false;
        bool is_chunk_end = false;
        int round_idx = -1;
        int perf_chunk_index = -1;
        std::chrono::steady_clock::time_point enqueue_time = std::chrono::steady_clock::now();
    };

    void encoder_thread_func();
    void llm_thread_func();
    void tts_thread_func();
    void t2w_thread_func();
    bool do_prefill(PrefillPacket * packet, double & prefill_ms, long long & prefill_tokens);
    bool do_decode(const DecodeReq & req, OmniDuplexFrameResult & result);
    bool generate_audio_tokens(const std::vector<float> & merged_embeddings,
                               int n_tokens,
                               int tts_n_embd,
                               int tts_chunk_idx,
                               bool is_end_of_turn,
                               int perf_chunk_index);

    struct omni_context * ctx = nullptr;
    common_params * params = nullptr;
    std::string debug_dir;
    std::atomic<bool> running{false};

    std::queue<EncodeReq *> encode_queue;
    std::mutex encode_mtx;
    std::condition_variable encode_cv;
    static constexpr size_t ENCODE_QUEUE_CAP = 16;

    std::queue<PrefillPacket *> prefill_queue;
    std::queue<DecodeReq *> decode_queue;
    std::mutex llm_mtx;
    std::condition_variable llm_cv;
    static constexpr size_t PREFILL_QUEUE_CAP = 32;

    std::queue<OmniDuplexFrameResult> done_results;
    std::mutex done_mtx;
    std::condition_variable done_cv;

    std::queue<PerfLLMOut *> tts_queue;
    std::mutex tts_mtx;
    std::condition_variable tts_cv;
    static constexpr size_t TTS_QUEUE_CAP = 8;

    std::queue<PerfT2WOut *> t2w_queue;
    std::mutex t2w_mtx;
    std::condition_variable t2w_cv;
    static constexpr size_t T2W_QUEUE_CAP = 32;

    std::atomic<int64_t> frame_id_counter{0};
    std::atomic<int> in_flight{0};

    std::thread encoder_thread;
    std::thread llm_thread;
    std::thread tts_thread;
    std::thread t2w_thread;
};
