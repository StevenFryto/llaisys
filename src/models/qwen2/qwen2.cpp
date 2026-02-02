#include "qwen2.hpp"

namespace llaisys::models {

Qwen2Model::Qwen2Model(const Qwen2Config& config, llaisysDeviceType_t device_type, int device_id)
    : config_(config), device_type_(device_type), device_id_(device_id), cache_len_(0) {
    
    const size_t nl = config_.num_layers;
    const size_t hs = config_.hidden_size;
    const size_t nh = config_.num_heads;
    const size_t nkvh = config_.num_kv_heads;
    const size_t dh = config_.head_dim;
    const size_t di = config_.intermediate_size;
    const size_t voc = config_.vocab_size;
    
    // Allocate embedding weights
    embed_tokens_ = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    lm_head_ = Tensor::create({voc, hs}, config_.dtype, device_type_, device_id_);
    norm_weight_ = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
    
    // Allocate per-layer weights
    input_layernorm_weight_.resize(nl);
    q_proj_weight_.resize(nl);
    q_proj_bias_.resize(nl);
    k_proj_weight_.resize(nl);
    k_proj_bias_.resize(nl);
    v_proj_weight_.resize(nl);
    v_proj_bias_.resize(nl);
    o_proj_weight_.resize(nl);
    post_attn_layernorm_weight_.resize(nl);
    gate_proj_weight_.resize(nl);
    up_proj_weight_.resize(nl);
    down_proj_weight_.resize(nl);
    
    for (size_t i = 0; i < nl; ++i) {
        input_layernorm_weight_[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        q_proj_weight_[i] = Tensor::create({nh * dh, hs}, config_.dtype, device_type_, device_id_);
        q_proj_bias_[i] = Tensor::create({nh * dh}, config_.dtype, device_type_, device_id_);
        k_proj_weight_[i] = Tensor::create({nkvh * dh, hs}, config_.dtype, device_type_, device_id_);
        k_proj_bias_[i] = Tensor::create({nkvh * dh}, config_.dtype, device_type_, device_id_);
        v_proj_weight_[i] = Tensor::create({nkvh * dh, hs}, config_.dtype, device_type_, device_id_);
        v_proj_bias_[i] = Tensor::create({nkvh * dh}, config_.dtype, device_type_, device_id_);
        o_proj_weight_[i] = Tensor::create({hs, nh * dh}, config_.dtype, device_type_, device_id_);
        post_attn_layernorm_weight_[i] = Tensor::create({hs}, config_.dtype, device_type_, device_id_);
        gate_proj_weight_[i] = Tensor::create({di, hs}, config_.dtype, device_type_, device_id_);
        up_proj_weight_[i] = Tensor::create({di, hs}, config_.dtype, device_type_, device_id_);
        down_proj_weight_[i] = Tensor::create({hs, di}, config_.dtype, device_type_, device_id_);
    }
    
    // Allocate KV cache
    allocateKVCache();
}

void Qwen2Model::allocateKVCache() {
    const size_t nl = config_.num_layers;
    const size_t nkvh = config_.num_kv_heads;
    const size_t dh = config_.head_dim;
    const size_t maxseq = config_.max_seq_len;
    
    k_cache_.resize(nl);
    v_cache_.resize(nl);
    
    for (size_t i = 0; i < nl; ++i) {
        k_cache_[i] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, device_type_, device_id_);
        v_cache_[i] = Tensor::create({maxseq, nkvh, dh}, config_.dtype, device_type_, device_id_);
    }
}

void Qwen2Model::allocateBuffers(size_t seq_len) {
    const size_t hs = config_.hidden_size;
    const size_t nh = config_.num_heads;
    const size_t nkvh = config_.num_kv_heads;
    const size_t dh = config_.head_dim;
    const size_t di = config_.intermediate_size;
    const size_t voc = config_.vocab_size;
    
    hidden_states_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    residual_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    normed_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    q_ = Tensor::create({seq_len, nh, dh}, config_.dtype, device_type_, device_id_);
    k_ = Tensor::create({seq_len, nkvh, dh}, config_.dtype, device_type_, device_id_);
    v_ = Tensor::create({seq_len, nkvh, dh}, config_.dtype, device_type_, device_id_);
    attn_out_ = Tensor::create({seq_len, nh, dh}, config_.dtype, device_type_, device_id_);
    attn_proj_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    gate_ = Tensor::create({seq_len, di}, config_.dtype, device_type_, device_id_);
    up_ = Tensor::create({seq_len, di}, config_.dtype, device_type_, device_id_);
    mlp_out_ = Tensor::create({seq_len, di}, config_.dtype, device_type_, device_id_);
    down_ = Tensor::create({seq_len, hs}, config_.dtype, device_type_, device_id_);
    logits_ = Tensor::create({1, voc}, config_.dtype, device_type_, device_id_);
    pos_ids_ = Tensor::create({seq_len}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_idx_ = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    max_val_ = Tensor::create({1}, config_.dtype, device_type_, device_id_);
}

int64_t Qwen2Model::infer(const int64_t* token_ids, size_t num_tokens) {
    const size_t nh = config_.num_heads;
    const size_t nkvh = config_.num_kv_heads;
    const size_t dh = config_.head_dim;
    const size_t nl = config_.num_layers;
    const float eps = config_.rms_norm_eps;
    const float theta = config_.rope_theta;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    
    // Allocate buffers for this sequence length
    allocateBuffers(num_tokens);
    
    // Create input token tensor
    auto input_ids = Tensor::create({num_tokens}, LLAISYS_DTYPE_I64, device_type_, device_id_);
    input_ids->load(token_ids);
    
    // Create position ids: [cache_len, cache_len+1, ..., cache_len+num_tokens-1]
    std::vector<int64_t> pos_ids_data(num_tokens);
    for (size_t i = 0; i < num_tokens; ++i) {
        pos_ids_data[i] = static_cast<int64_t>(cache_len_ + i);
    }
    pos_ids_->load(pos_ids_data.data());
    
    // Embedding lookup: hidden_states = embed_tokens[input_ids]
    ops::embedding(hidden_states_, input_ids, embed_tokens_);
    
    // Process each layer
    for (size_t layer = 0; layer < nl; ++layer) {
        // Save residual
        // residual = hidden_states (copy via add with zero? or just swap pointers)
        // For simplicity, we'll use add: residual = hidden_states + 0
        // Actually, let's just keep track: residual points to hidden_states before attention
        
        // Input layernorm
        ops::rms_norm(normed_, hidden_states_, input_layernorm_weight_[layer], eps);
        
        // Q, K, V projections
        // q = normed @ q_weight.T + q_bias -> [seq_len, nh * dh]
        auto q_flat = q_->view({num_tokens, nh * dh});
        auto k_flat = k_->view({num_tokens, nkvh * dh});
        auto v_flat = v_->view({num_tokens, nkvh * dh});
        
        ops::linear(q_flat, normed_, q_proj_weight_[layer], q_proj_bias_[layer]);
        ops::linear(k_flat, normed_, k_proj_weight_[layer], k_proj_bias_[layer]);
        ops::linear(v_flat, normed_, v_proj_weight_[layer], v_proj_bias_[layer]);
        
        // Apply RoPE to Q and K
        // q, k are [seq_len, num_heads, head_dim]
        ops::rope(q_, q_, pos_ids_, theta);
        ops::rope(k_, k_, pos_ids_, theta);
        
        // Update KV cache
        // Copy new k, v to cache at positions [cache_len : cache_len + num_tokens]
        auto k_cache_slice = k_cache_[layer]->slice(0, cache_len_, cache_len_ + num_tokens);
        auto v_cache_slice = v_cache_[layer]->slice(0, cache_len_, cache_len_ + num_tokens);
        ops::rearrange(k_cache_slice, k_);
        ops::rearrange(v_cache_slice, v_);
        
        // Get full KV cache for attention
        auto k_full = k_cache_[layer]->slice(0, 0, cache_len_ + num_tokens);
        auto v_full = v_cache_[layer]->slice(0, 0, cache_len_ + num_tokens);
        
        // Self attention
        // attn_out = self_attention(q, k_full, v_full, scale)
        ops::self_attention(attn_out_, q_, k_full, v_full, scale);
        
        // Output projection
        auto attn_out_flat = attn_out_->view({num_tokens, nh * dh});
        ops::linear(attn_proj_, attn_out_flat, o_proj_weight_[layer], nullptr);
        
        // Residual connection: hidden_states = hidden_states + attn_proj
        ops::add(hidden_states_, hidden_states_, attn_proj_);
        
        // Post attention layernorm
        ops::rms_norm(normed_, hidden_states_, post_attn_layernorm_weight_[layer], eps);
        
        // MLP: gate_proj, up_proj, swiglu, down_proj
        ops::linear(gate_, normed_, gate_proj_weight_[layer], nullptr);
        ops::linear(up_, normed_, up_proj_weight_[layer], nullptr);
        ops::swiglu(mlp_out_, gate_, up_);
        ops::linear(down_, mlp_out_, down_proj_weight_[layer], nullptr);
        
        // Residual connection: hidden_states = hidden_states + down
        ops::add(hidden_states_, hidden_states_, down_);
    }
    
    // Final layer norm
    ops::rms_norm(normed_, hidden_states_, norm_weight_, eps);
    
    // Get last token's hidden state for prediction
    auto last_hidden = normed_->slice(0, num_tokens - 1, num_tokens);  // [1, hs]
    
    // LM head: logits = last_hidden @ lm_head.T
    ops::linear(logits_, last_hidden, lm_head_, nullptr);
    
    // Argmax to get next token
    auto logits_1d = logits_->view({config_.vocab_size});
    ops::argmax(max_idx_, max_val_, logits_1d);
    
    // Read result
    int64_t next_token;
    // Copy from device to host
    auto& ctx = core::context();
    ctx.setDevice(device_type_, device_id_);
    auto& runtime = ctx.runtime();
    runtime.api()->memcpy_sync(&next_token, max_idx_->data(), sizeof(int64_t), LLAISYS_MEMCPY_D2H);
    
    // Update cache length
    cache_len_ += num_tokens;
    
    return next_token;
}

} // namespace llaisys::models
