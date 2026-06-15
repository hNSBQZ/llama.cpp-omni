#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>

struct omni_context;

struct OmniPerfTokenStats {
    long long calls = 0;
    long long tokens = 0;
    double duration_ms = 0.0;
};

struct OmniPerfState {
    std::atomic<int> current_chunk_index{-1};
    std::mutex token_stats_mtx;
    OmniPerfTokenStats llm_prefill;
    OmniPerfTokenStats llm_decode;
    OmniPerfTokenStats tts_prefill;
    OmniPerfTokenStats tts_decode;
};

bool omni_tts_debug_dump_enabled();

void omni_gpu_perf_sampler_start();
void omni_gpu_perf_sampler_stop();

void omni_perf_print_token_stats(struct omni_context * ctx_omni);
void omni_perf_record_tokens(struct omni_context * ctx_omni, const char * stage, long long tokens, double duration_ms);
std::string omni_perf_speed_detail(long long tokens, double duration_ms);

void omni_perf_mark(struct omni_context * ctx_omni,
                    const char * stage,
                    const char * event,
                    int chunk_index = -1,
                    double duration_ms = -1.0,
                    const char * detail = nullptr);

class OmniPerfScope {
public:
    OmniPerfScope(struct omni_context * ctx, const char * stage, int chunk_index, const std::string & detail);
    ~OmniPerfScope();

    void set_detail(const std::string & detail);
    void set_tokens(long long tokens);

private:
    struct omni_context * ctx;
    const char * stage;
    int chunk_index;
    std::string detail;
    long long tokens = 0;
    bool has_tokens = false;
    std::chrono::steady_clock::time_point start;
};
