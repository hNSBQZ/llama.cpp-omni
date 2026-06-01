#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

typedef enum ggml_cuda_marlin_type {
    GGML_CUDA_MARLIN_TYPE_S4 = 0,
    GGML_CUDA_MARLIN_TYPE_U4 = 1,
    GGML_CUDA_MARLIN_TYPE_U4B8 = 2,
    GGML_CUDA_MARLIN_TYPE_S8 = 3,
    GGML_CUDA_MARLIN_TYPE_U8 = 4,
    GGML_CUDA_MARLIN_TYPE_U8B128 = 5,
    GGML_CUDA_MARLIN_TYPE_FE2M1F = 6,
    GGML_CUDA_MARLIN_TYPE_FE4M3FN = 7,
    GGML_CUDA_MARLIN_TYPE_FE8M0FNU = 8,
    GGML_CUDA_MARLIN_TYPE_F16 = 9,
    GGML_CUDA_MARLIN_TYPE_BF16 = 10,
} ggml_cuda_marlin_type;

typedef struct ggml_cuda_marlin_gemm_params {
    const void * a;
    const void * b_q_weight;
    void * c;
    void * c_tmp;
    const void * b_bias;
    const float * a_scales;
    const void * b_scales;
    const void * global_scale;
    const void * b_zeros;
    const int * g_idx;
    const int * perm;
    void * a_tmp;
    int * workspace;

    int64_t size_m;
    int64_t size_n;
    int64_t size_k;
    int64_t lda;
    int64_t num_groups;
    int64_t group_size;

    int32_t device;
    void * stream;
    int32_t sms;
    int32_t thread_k;
    int32_t thread_n;

    ggml_cuda_marlin_type a_type;
    ggml_cuda_marlin_type b_type;
    ggml_cuda_marlin_type c_type;
    ggml_cuda_marlin_type s_type;

    bool has_bias;
    bool has_act_order;
    bool is_k_full;
    bool has_zp;
    bool use_atomic_add;
    bool use_fp32_reduce;
    bool is_zp_float;
} ggml_cuda_marlin_gemm_params;

GGML_BACKEND_API int ggml_cuda_marlin_min_workspace_elements(int device);

GGML_BACKEND_API bool ggml_cuda_marlin_awq_repack(
        const uint32_t * b_q_weight,
        uint32_t * out,
        int64_t size_k,
        int64_t size_n,
        int64_t num_bits,
        bool is_a_8bit,
        int device,
        void * stream);

GGML_BACKEND_API bool ggml_cuda_marlin_gptq_repack(
        const uint32_t * b_q_weight,
        const uint32_t * perm,
        uint32_t * out,
        int64_t size_k,
        int64_t size_n,
        int64_t num_bits,
        bool has_perm,
        bool is_a_8bit,
        int device,
        void * stream);

GGML_BACKEND_API bool ggml_cuda_marlin_gemm(const ggml_cuda_marlin_gemm_params * params);

GGML_BACKEND_API bool ggml_cuda_marlin_w4a16_gemm(
        const struct ggml_tensor * a,
        const struct ggml_tensor * b_q_weight,
        const struct ggml_tensor * b_scales,
        const struct ggml_tensor * b_zeros,
        struct ggml_tensor * c,
        struct ggml_tensor * workspace,
        int device,
        void * stream);

GGML_BACKEND_API bool ggml_cuda_marlin_gptq_w8_gemm(
        const struct ggml_tensor * a,
        const struct ggml_tensor * b_q_weight,
        const struct ggml_tensor * b_scales,
        struct ggml_tensor * c,
        struct ggml_tensor * workspace,
        int device,
        void * stream);

#ifdef  __cplusplus
}
#endif
