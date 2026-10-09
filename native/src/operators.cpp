#include <inferbridge/linux_model_execution_policy.h>
#include "operators.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d_pointwise_gemm_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv2d_tiled_spv.h"
#include "conv2d8_tiled_spv.h"
#include "conv2d8_stride2_tiled_spv.h"
#include "conv2d8_stride2_tiled_half_spv.h"
#include "conv2d8_tiled16x8_spv.h"
#include "conv2d8_tiled16x8_half_spv.h"
#include "conv2d_winograd_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_vec8_spv.h"
#include "linear_vec8_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "quantize_rows_int8_spv.h"
#include "inferbridge/native_harness_precision.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"
#include "group_norm_spv.h"
#include "group_norm_silu_spv.h"
#include "silu_spv.h"
#include "nearest_spv.h"
#include "concatenate_spv.h"
#include "add_channel_spv.h"
#include "nchw_tokens_spv.h"
#include "geglu_spv.h"
#include "attention_scores_spv.h"
#include "attention_values_spv.h"
#include "preprocess_rgb_spv.h"
#include "preprocess_texture_spv.h"
#include "seeded_noise_spv.h"
#include "posterior_sample_spv.h"
#include "scale_values_spv.h"
#include "depth_output_spv.h"
#include "depth_range_spv.h"
#include "normalize_depth_spv.h"
#include "depth_to_image_spv.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace iris_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

struct BmmParameters {
    std::uint32_t rows;
    std::uint32_t columns;
    std::uint32_t inner;
    std::uint32_t batches;
    std::uint32_t weight_transposed;
    std::uint32_t output_token_major;
    std::uint32_t qkv_embedding;
    std::uint32_t input_qkv_query;
    std::uint32_t weight_qkv_kind;
    std::uint32_t qkv_heads;
    std::uint32_t qkv_tokens;
    float input_scale;
};
static_assert(sizeof(BmmParameters) == 48);

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      linear_(context.create_pipeline(
          iris_linear_spv, iris_linear_spv_size, 4, 12)),
      linear16_(context.create_pipeline(
          iris_linear16_spv, iris_linear16_spv_size, 4, 12)),
      linear_half_(context.create_pipeline(
          iris_linear_half_spv, iris_linear_half_spv_size, 4, 12)),
      linear16_half_(context.create_pipeline(
          iris_linear16_half_spv,
          iris_linear16_half_spv_size,
          4,
          12)),
      linear_vec8_(context.create_pipeline(
          iris_linear_vec8_spv,
          iris_linear_vec8_spv_size,
          4,
          12)),
      linear_vec8_half_(context.create_pipeline(
          iris_linear_vec8_half_spv,
          iris_linear_vec8_half_spv_size,
          4,
          12)),
      quantize_rows_int8_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(iris_quantize_rows_int8_spv,
                iris_quantize_rows_int8_spv_size,
              {
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            },
                            {
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                            },
                            4)
          : VulkanPipeline{}),
      linear_int8_tiled_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(iris_linear_int8_tiled_spv,
                iris_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      gelu_(context.create_pipeline(
          iris_gelu_spv, iris_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          iris_layer_norm_spv, iris_layer_norm_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          iris_add_scaled_spv, iris_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          iris_bmm_spv, iris_bmm_spv_size, 3, 48)),
      bmm_score_half_(context.create_pipeline(
          iris_bmm_score_half_spv,
          iris_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          iris_bmm_value_half_spv,
          iris_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          iris_softmax_lastdim_spv,
          iris_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          iris_softmax_lastdim_half_spv,
          iris_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          iris_prepare_tokens_spv,
          iris_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          iris_position_bicubic_spv,
          iris_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          iris_add_position_spv,
          iris_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          iris_add_spv, iris_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          iris_project_tokens_spv,
          iris_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          iris_project_tokens_half_spv,
          iris_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          iris_conv2d_spv, iris_conv2d_spv_size, 4, 48)),
      conv2d_pointwise_gemm_(context.create_pipeline(
          iris_conv2d_pointwise_gemm_spv,
          iris_conv2d_pointwise_gemm_spv_size, 4, 48)),
      conv2d8_(context.create_pipeline(
          iris_conv2d8_spv, iris_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          iris_conv2d_half_spv, iris_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          iris_conv2d8_half_spv,
          iris_conv2d8_half_spv_size,
          4,
          48)),
      conv2d_tiled_(context.create_pipeline(
          iris_conv2d_tiled_spv,
          iris_conv2d_tiled_spv_size,
          4,
          48)),
      conv2d8_tiled_(context.create_pipeline(
          iris_conv2d8_tiled_spv,
          iris_conv2d8_tiled_spv_size,
          4,
          48)),
      conv2d8_stride2_tiled_(context.create_pipeline(
          iris_conv2d8_stride2_tiled_spv,
          iris_conv2d8_stride2_tiled_spv_size,
          4,
          48)),
      conv2d8_stride2_tiled_half_(context.create_pipeline(
          iris_conv2d8_stride2_tiled_half_spv,
          iris_conv2d8_stride2_tiled_half_spv_size,
          4,
          48)),
      conv2d8_tiled16x8_(context.create_pipeline(
          iris_conv2d8_tiled16x8_spv,
          iris_conv2d8_tiled16x8_spv_size,
          4,
          48)),
      conv2d8_tiled16x8_half_(context.create_pipeline(
          iris_conv2d8_tiled16x8_half_spv,
          iris_conv2d8_tiled16x8_half_spv_size,
          4,
          48)),
      conv2d_winograd_(context.create_pipeline(
          iris_conv2d_winograd_spv,
          iris_conv2d_winograd_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          iris_conv_transpose_nonoverlap_spv,
          iris_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          iris_conv_transpose_nonoverlap_half_spv,
          iris_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          iris_bilinear_align_true_spv,
          iris_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          iris_bilinear_align_true_image_spv,
          iris_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          iris_relu_spv, iris_relu_spv_size, 2, 4)),
      group_norm_(context.create_pipeline(
          iris_group_norm_spv, iris_group_norm_spv_size, 3, 16)),
      group_norm_silu_(context.create_pipeline(
          iris_group_norm_silu_spv,
          iris_group_norm_silu_spv_size, 3, 16)),
      silu_(context.create_pipeline(
          iris_silu_spv, iris_silu_spv_size, 1, 4)),
      nearest_(context.create_pipeline(
          iris_nearest_spv, iris_nearest_spv_size, 2, 20)),
      concatenate_(context.create_pipeline(
          iris_concatenate_spv, iris_concatenate_spv_size, 3, 8)),
      add_channel_(context.create_pipeline(
          iris_add_channel_spv, iris_add_channel_spv_size, 2, 8)),
      nchw_tokens_(context.create_pipeline(
          iris_nchw_tokens_spv, iris_nchw_tokens_spv_size, 2, 12)),
      geglu_(context.create_pipeline(
          iris_geglu_spv, iris_geglu_spv_size, 2, 8)),
      attention_scores_(context.create_pipeline(
          iris_attention_scores_spv,
          iris_attention_scores_spv_size, 3, 16)),
      attention_values_(context.create_pipeline(
          iris_attention_values_spv,
          iris_attention_values_spv_size, 3, 16)),
      preprocess_rgb_(context.create_pipeline(
          iris_preprocess_rgb_spv, iris_preprocess_rgb_spv_size, 2, 8)),
      preprocess_texture_(context.create_pipeline(
          iris_preprocess_texture_spv, iris_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 16)),
      seeded_noise_(context.create_pipeline(
          iris_seeded_noise_spv, iris_seeded_noise_spv_size, 2, 12)),
      posterior_sample_(context.create_pipeline(
          iris_posterior_sample_spv,
          iris_posterior_sample_spv_size, 3, 4)),
      scale_values_(context.create_pipeline(
          iris_scale_values_spv, iris_scale_values_spv_size, 1, 8)),
      depth_output_(context.create_pipeline(
          iris_depth_output_spv, iris_depth_output_spv_size, 2, 16)),
      depth_range_(context.create_pipeline(
          iris_depth_range_spv, iris_depth_range_spv_size, 2, 4)),
      normalize_depth_(context.create_pipeline(
          iris_normalize_depth_spv,
          iris_normalize_depth_spv_size, 2, 4)),
      depth_to_image_(context.create_pipeline(
          iris_depth_to_image_spv, iris_depth_to_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 8)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_vec8_half_.set_debug_name("linear_vec8_half");
    if (context.supports_packed_int8_dot() &&
        inferbridge::native::requested_precision() ==
            inferbridge::native::Precision::int8) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d_pointwise_gemm_.set_debug_name("conv2d_pointwise_gemm");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv2d_tiled_.set_debug_name("conv2d_tiled");
    conv2d8_tiled_.set_debug_name("conv2d8_tiled");
    conv2d8_stride2_tiled_.set_debug_name(
        "conv2d8_stride2_tiled");
    conv2d8_stride2_tiled_half_.set_debug_name(
        "conv2d8_stride2_tiled_half");
    conv2d8_tiled16x8_.set_debug_name("conv2d8_tiled16x8");
    conv2d8_tiled16x8_half_.set_debug_name(
        "conv2d8_tiled16x8_half");
    conv2d_winograd_.set_debug_name("conv2d_winograd");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
    group_norm_.set_debug_name("group_norm");
    group_norm_silu_.set_debug_name("group_norm_silu");
    silu_.set_debug_name("silu");
    nearest_.set_debug_name("nearest");
    concatenate_.set_debug_name("concatenate");
    add_channel_.set_debug_name("add_channel");
    nchw_tokens_.set_debug_name("nchw_tokens");
    geglu_.set_debug_name("geglu");
    attention_scores_.set_debug_name("attention_scores");
    attention_values_.set_debug_name("attention_values");
    preprocess_rgb_.set_debug_name("preprocess_rgb");
    preprocess_texture_.set_debug_name("preprocess_texture");
    seeded_noise_.set_debug_name("seeded_noise");
    posterior_sample_.set_debug_name("posterior_sample");
    scale_values_.set_debug_name("scale_values");
    depth_output_.set_debug_name("depth_output");
    depth_range_.set_debug_name("depth_range");
    normalize_depth_.set_debug_name("normalize_depth");
    depth_to_image_.set_debug_name("depth_to_image");
}

#include <inferbridge/vulkan/operators/vulkan_operators_linear_int8_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_linear_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_layer_norm_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_scaled.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_head64_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_prepare_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_project_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_conv2d_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_conv2d_asymmetric.h>

#include <inferbridge/vulkan/operators/vulkan_operators_conv_transpose_nonoverlap_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_image.h>

#include <inferbridge/vulkan/operators/vulkan_operators_relu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add.h>

#include <inferbridge/vulkan/operators/vulkan_operators_group_norm.h>

#include <inferbridge/vulkan/operators/vulkan_operators_silu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nearest.h>

#include <inferbridge/vulkan/operators/vulkan_operators_concatenate.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_channel.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nchw_tokens.h>

#include <inferbridge/vulkan/operators/vulkan_operators_geglu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_separate.h>

#include <inferbridge/vulkan/operators/vulkan_operators_preprocess_rgb.h>

#include <inferbridge/vulkan/operators/vulkan_operators_preprocess_texture.h>

#include <inferbridge/vulkan/operators/vulkan_operators_seeded_noise.h>

#include <inferbridge/vulkan/operators/vulkan_operators_posterior_sample.h>

#include <inferbridge/vulkan/operators/vulkan_operators_scale_values.h>

#include <inferbridge/vulkan/operators/vulkan_operators_depth_output.h>

#include <inferbridge/vulkan/operators/vulkan_operators_normalize_depth.h>

#include <inferbridge/vulkan/operators/vulkan_operators_depth_to_image.h>

}  // namespace iris_native
