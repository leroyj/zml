const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");
const inference = @import("inference.zig");
const q8_triton = @import("../../gguf/q8_triton.zig");
const q4k_triton = @import("../../gguf/q4k_triton.zig");
const q5k_triton = @import("../../gguf/q5k_triton.zig");
const q6k_triton = @import("../../gguf/q6k_triton.zig");
const iq4xs_triton = @import("../../gguf/iq4xs_triton.zig");
const iq4nl_triton = @import("../../gguf/iq4nl_triton.zig");

const log = std.log.scoped(.qwen3_5);

pub const Config = struct {
    text_config: TextConfig,
};

pub const TextConfig = struct {
    // General
    num_hidden_layers: i64,
    layer_types: []const LayerType,
    hidden_size: i64,
    max_position_embeddings: i64,
    rms_norm_eps: f32,
    // Self attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rope_parameters: RopeParameters,
    // Linear attention
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_num_value_heads: i64,
    linear_value_head_dim: i64,
};

// Each layer uses either: full attention (SelfAttn) or linear attention (GatedDeltaNet).
pub const LayerType = enum {
    linear_attention,
    full_attention,
};

pub const RopeParameters = struct {
    mrope_section: [3]i64,
    partial_rotary_factor: f32,
    rope_theta: f32,
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        const options: Model.GenOptions = .{
            .sampling_strategy = generation.sampling_strategy,
            .max_seq_len = parsed_config.value.text_config.max_position_embeddings,
        };

        return .{
            .inner = try .init(allocator, store, parsed_config.value, options),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer self.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        try loader.load(io, Model, &self.inner, &buffers, store, .{ .progress = progress });
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        TextModel.unloadBuffers(&buffers.text_model, allocator);
        LinearLike.unloadBuffers(&buffers.lm_head);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        meshes: common.Meshes,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), meshes);
        return inference.CompiledModel.init(allocator, io, platform, self, self.inner, params, progress);
    }
};

pub const Buffers = zml.Bufferized(Model);

fn sampleLogits(logits_: zml.Tensor, sampling: zml.nn.SamplingStrategy, rng: zml.Tensor.Rng) struct { zml.Tensor, zml.Tensor.Rng } {
    // Use the caller's strategy as-is: `topk = 1` (greedy) must stay greedy here.
    // The previous version treated `temperature < eps` as the greedy switch and
    // otherwise sampled over the whole vocabulary, which turned every speculative
    // correction into a temperature-1.0 random sample.
    const tokens, const updated_rng = zml.nn.sampleTokens(logits_.withPartialTags(.{.voc}), sampling, rng);
    return .{ tokens.convert(.u32), updated_rng };
}

fn verifyDraftTokens(
    draft_tokens_: zml.Tensor,
    target_logits_: zml.Tensor,
    sampling: zml.nn.SamplingStrategy,
    rng: zml.Tensor.Rng,
) struct { zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
    const draft_tokens = draft_tokens_.withPartialTags(.{.s});
    const target_logits = target_logits_.withPartialTags(.{ .s, .voc });
    const draft_len = draft_tokens.dim(.s);
    const verify_len = draft_len - 1;

    const sampled_tokens, const updated_rng = sampleLogits(target_logits, sampling, rng);
    const proposals = draft_tokens.slice(.s, .{ .start = 1, .end = draft_len });
    const posterior = sampled_tokens.slice(.s, .{ .start = 0, .end = verify_len });
    const matches = proposals.cmp(.EQ, posterior);
    const candidate_idx = zml.Tensor.iota(.init(.{ .b = 1, .s = verify_len }, .u32), .s).convert(.u32);
    const sentinel = zml.Tensor.scalar(@as(u32, @intCast(verify_len)), .u32).broad(candidate_idx.shape());
    const valid_draft_tokens = matches.select(sentinel, candidate_idx).min(.s).squeeze(.s);
    const correction_token = sampled_tokens.gather(.{ .s = valid_draft_tokens }, .{}).convert(.u32);
    return .{ valid_draft_tokens, correction_token, updated_rng };
}

pub const Model = struct {
    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_seq_len: i64 };

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
    };

    text_model: TextModel,
    lm_head: LinearLike,

    config: Config,
    gen_options: GenOptions,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Model {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        const lm_head_prefix = b: {
            if (store.hasKey("lm_head.weight")) break :b "lm_head";
            break :b "model.language_model.embed_tokens";
        };
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config),
            .lm_head = LinearLike.init(
                store.withPrefix(lm_head_prefix),
                .{ .dout, .d },
                .model,
                .{ .dout = .model, .d = .replicated },
                .{.dout},
                .model,
                .{ .dout = .replicated },
                .d,
            ),
            .config = config,
            .gen_options = gen_options,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
    }

    pub fn load(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        meshes: []const *const zml.Mesh,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);

        var buffers = try zml.mem.bufferize(allocator, Model, self);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();

        loader.load(io, Model, self, &buffers, store, meshes);
        try loader.await(io);

        const took = now.untilNow(io, .awake);
        const total_bytes: u64 = loader.bytes_loaded.raw;
        const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
        LinearLike.unloadBuffers(&self.lm_head);
    }
    pub fn forward(
        self: Model,
        tokens_: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const text_model_output, const updated_kv_cache = self.text_model.forward(tokens, token_index, active_length, kv_cache);
        const result = Sampler.sampleTokens(.{
            .sampler = self.sampler(),
            .hidden = text_model_output,
            .tokens = tokens,
            .active_length = active_length,
            .rng = rng,
            .token_index = token_index,
        });
        return .{ result.tokens, updated_kv_cache, result.rng };
    }

    pub fn prefillForward(
        self: Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, KvCache, zml.Tensor.Rng } {
        const tokens_ = tokens.withPartialTags(.{.s});
        const target_hidden, const hidden, const updated_kv_cache = self.text_model.forwardWithTaps(
            tokens_,
            token_index,
            active_length,
            kv_cache,
            target_layer_ids,
            false,
        );
        const last = hidden.slice(.s, .dyn(active_length.convert(.i32).addConstant(-1), 1));
        const logits = self.sampler().logitsAll(last);
        const next_token, const updated_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        return .{ target_hidden, next_token.convert(.u32), updated_kv_cache, updated_rng };
    }

    pub fn verifyForward(
        self: Model,
        tokens: zml.Tensor,
        draft_tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, zml.Tensor, zml.Tensor, KvCache, zml.Tensor, zml.Tensor, zml.Tensor.Rng } {
        const tokens_ = tokens.withPartialTags(.{.s});
        const target_hidden, const hidden, const updated_kv_cache, const stacked_states, const conv_feats = self.text_model.forwardWithTapsSpec(
            tokens_,
            token_index,
            active_length,
            kv_cache,
            target_layer_ids,
        );
        const logits = self.sampler().logitsAll(hidden);
        const valid_draft_tokens, const correction_token, const updated_rng = verifyDraftTokens(
            draft_tokens,
            logits,
            self.gen_options.sampling_strategy,
            rng,
        );
        return .{ target_hidden, valid_draft_tokens, correction_token, updated_kv_cache, stacked_states, conv_feats, updated_rng };
    }

    pub fn sampler(self: Model) Sampler {
        return .{
            .norm = self.text_model.norm,
            .lm_head = self.lm_head,
            .gen_options = self.gen_options,
        };
    }
};

pub const Sampler = struct {
    norm: RmsNorm,
    lm_head: LinearLike,
    gen_options: Model.GenOptions,

    pub const Input = struct {
        sampler: Sampler,
        hidden: zml.Tensor,
        tokens: zml.Tensor,
        active_length: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub const Output = struct {
        tokens: zml.Tensor,
        rng: zml.Tensor.Rng,
        token_index: zml.Tensor,
    };

    pub fn sampleTokens(input: Input) Output {
        const self = input.sampler;
        const last_index = input.active_length.convert(.i32).addConstant(-1);
        const x = self.norm.forward(input.hidden.slice(.s, .dyn(last_index, 1)));
        const logits = self.lm_head.forward(x.withPartialTags(.{.d}), x.dtype()).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, input.rng);
        const tokens = next_tokens.convert(.u32);
        return .{
            .tokens = if (input.tokens.dim(.s) == 1) tokens.reuseBuffer(input.tokens) else tokens,
            .rng = new_rng,
            .token_index = input.token_index.addConstant(1).reuseBuffer(input.token_index),
        };
    }

    /// Logits for every position, used by speculative verification.
    pub fn logitsAll(self: Sampler, hidden: zml.Tensor) zml.Tensor {
        const x = self.norm.forward(hidden);
        return self.lm_head.forward(x.withPartialTags(.{.d}), x.dtype()).rename(.{ .dout = .voc });
    }
};

pub const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,

    pub const Input = struct {
        embedding: EmbedTokens,
        tokens: zml.Tensor,
    };

    pub const Output = struct {
        hidden: zml.Tensor,
    };

    pub fn forward(input: Input) Output {
        const tokens = input.tokens.withPartialTags(.{.s});
        return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
            .withPartialTags(.{.d})
            .replicate() };
    }
};

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
    ) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, @intCast(config.text_config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .model, .{ .voc = .replicated, .d = .model }) },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        var hidden_states = EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = self.embed_tokens },
            .tokens = tokens,
        }).hidden;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, active_length, updated_kv_cache.atLayer(i));
        }

        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    /// Runs the text model and returns the concatenated hidden states at
    /// `target_layer_ids` alongside the final hidden state.
    pub fn forwardWithTaps(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
        use_prior_state: bool,
    ) struct { zml.Tensor, zml.Tensor, KvCache } {
        var hidden_states = EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = self.embed_tokens },
            .tokens = tokens,
        }).hidden;

        var updated_kv_cache = kv_cache;
        var selected: [16]zml.Tensor = undefined;
        var selected_len: usize = 0;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forwardWithCache(hidden_states, token_index, active_length, updated_kv_cache.atLayer(i), use_prior_state);
            for (target_layer_ids) |target| {
                if (target == i) {
                    selected[selected_len] = hidden_states;
                    selected_len += 1;
                    break;
                }
            }
        }

        const target_hidden = zml.Tensor.concatenate(selected[0..selected_len], .d).squeeze(.b);
        return .{ target_hidden, hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    /// Speculative variant of `forwardWithTaps`: the GatedDeltaNet layers do
    /// not update the cache; instead they return their per-step recurrent
    /// states and conv features so the caller can roll the cache back to the
    /// accepted prefix with `commitSpeculativeState`.
    pub fn forwardWithTapsSpec(
        self: TextModel,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache,
        target_layer_ids: []const u32,
    ) struct { zml.Tensor, zml.Tensor, KvCache, zml.Tensor, zml.Tensor } {
        var hidden_states = EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = self.embed_tokens },
            .tokens = tokens,
        }).hidden;

        var updated_kv_cache = kv_cache;
        var selected: [16]zml.Tensor = undefined;
        var selected_len: usize = 0;
        var stacked_list: [64]zml.Tensor = undefined;
        var conv_list: [64]zml.Tensor = undefined;
        var stacked_len: usize = 0;
        for (self.layers, 0..) |layer, i| {
            const result = layer.forwardSpec(hidden_states, token_index, active_length, updated_kv_cache.atLayer(i));
            hidden_states = result.hidden;
            updated_kv_cache = result.cache;
            if (result.stacked) |stacked| {
                stacked_list[stacked_len] = stacked;
                conv_list[stacked_len] = result.conv_feats.?;
                stacked_len += 1;
            }
            for (target_layer_ids) |target| {
                if (target == i) {
                    selected[selected_len] = hidden_states;
                    selected_len += 1;
                    break;
                }
            }
        }

        const target_hidden = zml.Tensor.concatenate(selected[0..selected_len], .d).squeeze(.b);
        const stacked = zml.Tensor.stack(stacked_list[0..stacked_len], 0, .layer);
        const conv_feats = zml.Tensor.stack(conv_list[0..stacked_len], 1, .layer);
        return .{ target_hidden, hidden_states, updated_kv_cache.reuseBuffer(kv_cache), stacked, conv_feats };
    }
};

pub const TransformerLayer = struct {
    const Attn = union(LayerType) {
        linear_attention: GatedDeltaNet,
        full_attention: SelfAttn,
    };

    input_layernorm: RmsNorm,
    attn: Attn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,

    pub const SelfAttnInput = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        token_index: zml.Tensor,
        cache: KvCache.SelfAttnCache,
    };

    pub const SelfAttnOutput = struct {
        hidden: zml.Tensor,
        cache: KvCache.SelfAttnCache,
    };

    pub const LinearAttnInput = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        active_length: zml.Tensor,
        cache: KvCache.GatedDeltaNetCache,
    };

    pub const LinearAttnOutput = struct {
        hidden: zml.Tensor,
        cache: KvCache.GatedDeltaNetCache,
    };

    pub fn init(store: zml.io.TensorStore.View, config: Config, layer_index: usize) !TransformerLayer {
        const is_full_attention = config.text_config.layer_types[layer_index] == .full_attention;
        const attn: Attn = switch (is_full_attention) {
            true => .{ .full_attention = try .init(store.withPrefix("self_attn"), config) },
            false => .{ .linear_attention = .init(store.withPrefix("linear_attn"), config) },
        };
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.text_config.rms_norm_eps),
            .attn = attn,
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn schemeKey(self: TransformerLayer) LayerKey {
        var key: LayerKey = .{
            .kind = std.meta.activeTag(self.attn),
            .schemes = @splat(.bf16),
        };
        switch (self.attn) {
            .full_attention => |attn| attn.writeSchemeKey(&key.schemes),
            .linear_attention => |attn| attn.writeSchemeKey(&key.schemes),
        }
        self.mlp.writeSchemeKey(&key.schemes, 5);
        return key;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        switch (self.attn) {
            .full_attention => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attention => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        }
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache.LayerView,
    ) struct { zml.Tensor, KvCache } {
        return self.forwardWithCache(x0, token_index, active_length, kv_cache, x0.dim(.s) == 1);
    }

    pub fn forwardWithCache(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache.LayerView,
        use_prior_state: bool,
    ) struct { zml.Tensor, KvCache } {
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

        var attention_output: zml.Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        switch (self.attn) {
            .full_attention => |*self_attn| {
                const cache = switch (kv_cache.cache) {
                    .self_attn => |cache| cache,
                    .linear_attn => unreachable,
                };
                const result = self_attn.forward(normalized_x0, token_index, cache);
                attention_output = result[0];
                updated_kv_cache.self_attn = result[1].reuseBuffer(updated_kv_cache.self_attn);
            },
            .linear_attention => |*linear_attn| {
                const cache = switch (kv_cache.cache) {
                    .linear_attn => |cache| cache,
                    .self_attn => unreachable,
                };
                const result = linear_attn.forwardWithCache(normalized_x0, cache, active_length, use_prior_state);
                attention_output = result[0];
                updated_kv_cache.gated_delta_net = result[1].reuseBuffer(updated_kv_cache.gated_delta_net);
            },
        }

        const x1 = attention_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.model, .{ .d = .replicated });

        return .{ mlp_output.add(x1).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0), updated_kv_cache };
    }

    pub const SpecOutput = struct {
        hidden: zml.Tensor,
        cache: KvCache,
        stacked: ?zml.Tensor = null,
        conv_feats: ?zml.Tensor = null,
    };

    /// Speculative variant: full-attention layers update the KV cache as
    /// usual (later blocks overwrite any rejected entries), while
    /// GatedDeltaNet layers leave the cache untouched and report their
    /// per-step states for a later rollback.
    pub fn forwardSpec(
        self: TransformerLayer,
        x0: zml.Tensor,
        token_index: zml.Tensor,
        active_length: zml.Tensor,
        kv_cache: KvCache.LayerView,
    ) SpecOutput {
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

        var attention_output: zml.Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        var stacked: ?zml.Tensor = null;
        var conv_feats: ?zml.Tensor = null;
        switch (self.attn) {
            .full_attention => |*self_attn| {
                const cache = switch (kv_cache.cache) {
                    .self_attn => |cache| cache,
                    .linear_attn => unreachable,
                };
                const result = self_attn.forward(normalized_x0, token_index, cache);
                attention_output = result[0];
                updated_kv_cache.self_attn = result[1].reuseBuffer(updated_kv_cache.self_attn);
            },
            .linear_attention => |*linear_attn| {
                const cache = switch (kv_cache.cache) {
                    .linear_attn => |cache| cache,
                    .self_attn => unreachable,
                };
                const result = linear_attn.forwardSpeculative(normalized_x0, cache, active_length);
                attention_output = result[0];
                stacked = result[1];
                conv_feats = result[2];
            },
        }

        const x1 = attention_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.model, .{ .d = .replicated });

        return .{
            .hidden = mlp_output.add(x1).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
            .cache = updated_kv_cache,
            .stacked = stacked,
            .conv_feats = conv_feats,
        };
    }

    pub fn forwardSelfAttn(input: SelfAttnInput) SelfAttnOutput {
        const self = input.layer;
        const x0 = input.hidden;
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

        const self_attn = switch (self.attn) {
            .full_attention => |self_attn| self_attn,
            .linear_attention => unreachable,
        };
        const attention_output, const updated_kv_cache = self_attn.forward(normalized_x0, input.token_index, input.cache);

        const x1 = attention_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);
        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.model, .{ .d = .replicated });

        return .{
            .hidden = mlp_output.add(x1).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
            .cache = updated_kv_cache,
        };
    }

    pub fn forwardLinearAttn(input: LinearAttnInput) LinearAttnOutput {
        const self = input.layer;
        const x0 = input.hidden;
        const x0_replicated = x0.withPartitioning(.model, .{ .d = .replicated });
        const normalized_x0 = self.input_layernorm.forward(x0_replicated);

        const linear_attn = switch (self.attn) {
            .linear_attention => |linear_attn| linear_attn,
            .full_attention => unreachable,
        };
        const attention_output, const updated_kv_cache = linear_attn.forward(normalized_x0, input.cache, input.active_length);

        const x1 = attention_output.add(x0_replicated).withPartitioning(.model, .{ .d = .replicated });
        const normalized_hidden = self.post_attention_layernorm.forward(x1);
        const mlp_output = self.mlp.forward(normalized_hidden).withPartitioning(.model, .{ .d = .replicated });

        return .{
            .hidden = mlp_output.add(x1).withPartitioning(.model, .{ .d = .replicated }).reuseBuffer(x0),
            .cache = updated_kv_cache,
        };
    }
};

pub const QuantizedLinearQ8 = struct {
    quants: zml.Tensor,
    scales: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearQ8 {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;
        const contraction_partition = if (is_d) partitioning.d else partitioning.dout;

        const quants = store.createTensor("weight.q8", .{ .n, .k }, mesh_, .{ .n = output_partition, .k = contraction_partition });
        const dout = quants.dim(.n);
        return .{
            .quants = quants,
            .scales = store.createTensor("weight.s", .{ .n, .k32 }, mesh_, .{ .n = output_partition, .k32 = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearQ8, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.quants.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = q8GraphDequant(self.quants, self.scales, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk q8Matvec1(x_row, self.quants, self.scales, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk q8Matvec(x_padded, self.quants, self.scales, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const QuantizedLinearQ4 = struct {
    qs: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearQ4 {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;

        const qs = store.createTensor("weight.q4.qs", .{ .n, .k2 }, mesh_, .{ .n = output_partition, .k2 = .replicated });
        const dout = qs.dim(.n);
        return .{
            .qs = qs,
            .sc = store.createTensor("weight.q4.sc", .{ .n, .kb, .eight }, mesh_, .{ .n = output_partition, .kb = .replicated, .eight = .replicated }),
            .mn = store.createTensor("weight.q4.mn", .{ .n, .kb, .eight }, mesh_, .{ .n = output_partition, .kb = .replicated, .eight = .replicated }),
            .d = store.createTensor("weight.q4.d", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .dmin = store.createTensor("weight.q4.dmin", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearQ4, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.qs.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = q4GraphDequant(self.qs, self.sc, self.mn, self.d, self.dmin, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk q4Matvec1(x_row, self.qs, self.sc, self.mn, self.d, self.dmin, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk q4Matvec(x_padded, self.qs, self.sc, self.mn, self.d, self.dmin, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const QuantizedLinearQ5 = struct {
    qs: zml.Tensor,
    qh: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearQ5 {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;

        const qs = store.createTensor("weight.q5.qs", .{ .n, .k2 }, mesh_, .{ .n = output_partition, .k2 = .replicated });
        const dout = qs.dim(.n);
        return .{
            .qs = qs,
            .qh = store.createTensor("weight.q5.qh", .{ .n, .k8 }, mesh_, .{ .n = output_partition, .k8 = .replicated }),
            .sc = store.createTensor("weight.q5.sc", .{ .n, .kb, .eight }, mesh_, .{ .n = output_partition, .kb = .replicated, .eight = .replicated }),
            .mn = store.createTensor("weight.q5.mn", .{ .n, .kb, .eight }, mesh_, .{ .n = output_partition, .kb = .replicated, .eight = .replicated }),
            .d = store.createTensor("weight.q5.d", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .dmin = store.createTensor("weight.q5.dmin", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearQ5, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.qs.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = q5GraphDequant(self.qs, self.qh, self.sc, self.mn, self.d, self.dmin, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk q5Matvec1(x_row, self.qs, self.qh, self.sc, self.mn, self.d, self.dmin, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk q5Matvec(x_padded, self.qs, self.qh, self.sc, self.mn, self.d, self.dmin, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const QuantizedLinearQ6 = struct {
    ql0: zml.Tensor,
    ql1: zml.Tensor,
    qh: zml.Tensor,
    sc1: zml.Tensor,
    sc2: zml.Tensor,
    sc3: zml.Tensor,
    sc4: zml.Tensor,
    d: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearQ6 {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;
        const part = .{ .n = output_partition, .kb = .replicated, .two = .replicated, .e32 = .replicated };
        const part_sc = .{ .n = output_partition, .kb = .replicated, .two = .replicated, .e2 = .replicated };

        const ql0 = store.createTensor("weight.q6.ql0", .{ .n, .kb, .two, .e32 }, mesh_, part);
        const dout = ql0.dim(.n);
        return .{
            .ql0 = ql0,
            .ql1 = store.createTensor("weight.q6.ql1", .{ .n, .kb, .two, .e32 }, mesh_, part),
            .qh = store.createTensor("weight.q6.qh", .{ .n, .kb, .two, .e32 }, mesh_, part),
            .sc1 = store.createTensor("weight.q6.sc1", .{ .n, .kb, .two, .e2 }, mesh_, part_sc),
            .sc2 = store.createTensor("weight.q6.sc2", .{ .n, .kb, .two, .e2 }, mesh_, part_sc),
            .sc3 = store.createTensor("weight.q6.sc3", .{ .n, .kb, .two, .e2 }, mesh_, part_sc),
            .sc4 = store.createTensor("weight.q6.sc4", .{ .n, .kb, .two, .e2 }, mesh_, part_sc),
            .d = store.createTensor("weight.q6.d", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearQ6, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.ql0.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = q6GraphDequant(self.ql0, self.ql1, self.qh, self.sc1, self.sc2, self.sc3, self.sc4, self.d, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk q6Matvec1(x_row, self.ql0, self.ql1, self.qh, self.sc1, self.sc2, self.sc3, self.sc4, self.d, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk q6Matvec(x_padded, self.ql0, self.ql1, self.qh, self.sc1, self.sc2, self.sc3, self.sc4, self.d, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const QuantizedLinearIq4Xs = struct {
    qs: zml.Tensor,
    scl: zml.Tensor,
    sch: zml.Tensor,
    d: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearIq4Xs {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;

        const qs = store.createTensor("weight.iq4.qs", .{ .n, .k2 }, mesh_, .{ .n = output_partition, .k2 = .replicated });
        const dout = qs.dim(.n);
        return .{
            .qs = qs,
            .scl = store.createTensor("weight.iq4.scl", .{ .n, .k64 }, mesh_, .{ .n = output_partition, .k64 = .replicated }),
            .sch = store.createTensor("weight.iq4.sch", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .d = store.createTensor("weight.iq4.d", .{ .n, .kb }, mesh_, .{ .n = output_partition, .kb = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearIq4Xs, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.qs.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = iq4GraphDequant(self.qs, self.scl, self.sch, self.d, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk iq4Matvec1(x_row, self.qs, self.scl, self.sch, self.d, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk iq4Matvec(x_padded, self.qs, self.scl, self.sch, self.d, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

pub const QuantizedLinearIq4Nl = struct {
    qs: zml.Tensor,
    d: zml.Tensor,
    bias: ?zml.Tensor,
    contraction_tag: zml.Shape.Tag,
    output_tag: zml.Shape.Tag,
    block_n: i64,

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) QuantizedLinearIq4Nl {
        _ = tags;
        const contraction: zml.Shape.Tag = zml.Shape.toTag(contraction_tag);
        const is_d = comptime std.mem.eql(u8, @tagName(contraction_tag), "d");
        const output: zml.Shape.Tag = if (is_d) zml.Shape.toTag(.dout) else zml.Shape.toTag(.d);
        const output_partition = if (is_d) partitioning.dout else partitioning.d;

        const qs = store.createTensor("weight.iq4nl.qs", .{ .n, .k2 }, mesh_, .{ .n = output_partition, .k2 = .replicated });
        const dout = qs.dim(.n);
        return .{
            .qs = qs,
            .d = store.createTensor("weight.iq4nl.d", .{ .n, .k32 }, mesh_, .{ .n = output_partition, .k32 = .replicated }),
            .bias = store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            .contraction_tag = contraction,
            .output_tag = output,
            .block_n = if (dout >= 64) 64 else 16,
        };
    }

    pub fn forward(self: QuantizedLinearIq4Nl, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        _ = dtype;
        const x_shape = x.shape();
        const d = x.dim(self.contraction_tag);
        const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
        const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
        const n = self.qs.dim(.n);

        const use_kernel = switch (zml.Compiler.current().platform.target) {
            .rocm, .cuda => true,
            else => false,
        };

        const out_padded = if (!use_kernel) blk: {
            const w = iq4nlGraphDequant(self.qs, self.d, n, d);
            break :blk x_flat.dot(w, .k);
        } else if (m == 1) blk: {
            const x_row = x_flat.reshape(zml.Shape.init(.{ .k = d }, x.dtype()));
            break :blk iq4nlMatvec1(x_row, self.qs, self.d, self.block_n).reshape(zml.Shape.init(.{ .m = 1, .n = n }, .bf16));
        } else blk: {
            const m_padded = std.mem.alignForward(i64, m, 16);
            const x_padded = if (m_padded == m) x_flat else blk2: {
                const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
                break :blk2 zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
            };
            break :blk iq4nlMatvec(x_padded, self.qs, self.d, self.block_n);
        };

        const out_flat = if (out_padded.dim(.m) == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
        const out_shape = x_shape.setDim(self.contraction_tag, n);
        var out = out_flat.reshape(out_shape).renameTag(self.contraction_tag, self.output_tag);
        if (self.bias) |bias| out = out.add(bias.broad(out.shape()));
        return out;
    }
};

fn q8GraphDequant(quants: zml.Tensor, scales: zml.Tensor, n: i64, k: i64) zml.Tensor {
    const nb = @divExact(k, 32);
    const q = quants.reshape(zml.Shape.init(.{ .n = n, .nb = nb, .block = 32 }, .i8)).convert(.f32);
    const s = scales.reshape(zml.Shape.init(.{ .n = n, .nb = nb, .one = 1 }, .f16)).convert(.f32);
    return q.mul(s.broad(q.shape())).reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn q4GraphDequant(
    qs: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    n: i64,
    k: i64,
) zml.Tensor {
    const kb = @divExact(k, 256);
    const q = qs.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 32 }, .u8));
    const lo = q.logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(q.shape())).convert(.f32);
    const hi = q.div(zml.Tensor.scalar(16, .u8).broad(q.shape())).convert(.f32);

    const sc4 = sc.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 2 }, .u8)).convert(.f32);
    const mn4 = mn.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 2 }, .u8)).convert(.f32);
    const sc_lo, const sc_hi = sc4.chunkExact(.e, 2);
    const mn_lo, const mn_hi = mn4.chunkExact(.e, 2);

    const d4 = d.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 1, .e = 1 }, .f32)).broad(q.shape());
    const dmin4 = dmin.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 1, .e = 1 }, .f32)).broad(q.shape());

    const w_lo = d4.mul(sc_lo.broad(q.shape())).mul(lo).sub(dmin4.mul(mn_lo.broad(q.shape())));
    const w_hi = d4.mul(sc_hi.broad(q.shape())).mul(hi).sub(dmin4.mul(mn_hi.broad(q.shape())));
    const w = zml.Tensor.concatenate(&.{ w_lo, w_hi }, .e);
    return w.reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn q5GraphDequant(
    qs: zml.Tensor,
    qh: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    n: i64,
    k: i64,
) zml.Tensor {
    const kb = @divExact(k, 256);
    const qshape = zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 32 }, .u8);
    const q = qs.reshape(qshape);
    const lo = q.logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(q.shape())).convert(.f32);
    const hi = q.div(zml.Tensor.scalar(16, .u8).broad(q.shape())).convert(.f32);

    const qh3 = qh.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .one = 1, .e = 32 }, .u8)).broad(q.shape());
    const bit_lo = zml.Tensor.iota(zml.Shape.init(.{ .g = 4 }, .u8), .g).scale(2).convert(.u8);
    const bit_hi = bit_lo.addConstant(1);
    const bit_lo_b = bit_lo.reshape(zml.Shape.init(.{ .n = 1, .kb = 1, .g = 4, .e = 1 }, .u8)).broad(q.shape());
    const bit_hi_b = bit_hi.reshape(zml.Shape.init(.{ .n = 1, .kb = 1, .g = 4, .e = 1 }, .u8)).broad(q.shape());
    const lo_hi = qh3.shiftRightLogical(bit_lo_b).logical(.AND, zml.Tensor.scalar(1, .u8).broad(q.shape())).convert(.f32).scale(16);
    const hi_hi = qh3.shiftRightLogical(bit_hi_b).logical(.AND, zml.Tensor.scalar(1, .u8).broad(q.shape())).convert(.f32).scale(16);

    const sc4 = sc.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 2 }, .u8)).convert(.f32);
    const mn4 = mn.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 4, .e = 2 }, .u8)).convert(.f32);
    const sc_lo, const sc_hi = sc4.chunkExact(.e, 2);
    const mn_lo, const mn_hi = mn4.chunkExact(.e, 2);

    const d4 = d.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 1, .e = 1 }, .f32)).broad(q.shape());
    const dmin4 = dmin.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .g = 1, .e = 1 }, .f32)).broad(q.shape());

    const w_lo = d4.mul(sc_lo.broad(q.shape())).mul(lo.add(lo_hi)).sub(dmin4.mul(mn_lo.broad(q.shape())));
    const w_hi = d4.mul(sc_hi.broad(q.shape())).mul(hi.add(hi_hi)).sub(dmin4.mul(mn_hi.broad(q.shape())));
    const w = zml.Tensor.concatenate(&.{ w_lo, w_hi }, .e);
    return w.reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn q6GraphDequant(
    ql0: zml.Tensor,
    ql1: zml.Tensor,
    qh: zml.Tensor,
    sc1: zml.Tensor,
    sc2: zml.Tensor,
    sc3: zml.Tensor,
    sc4: zml.Tensor,
    d: zml.Tensor,
    n: i64,
    k: i64,
) zml.Tensor {
    const kb = @divExact(k, 256);
    const qshape = zml.Shape.init(.{ .n = n, .kb = kb, .h = 2, .e = 32 }, .u8);
    const ql0_3 = ql0.reshape(qshape);
    const ql1_3 = ql1.reshape(qshape);
    const qh_3 = qh.reshape(qshape);

    const mask4 = zml.Tensor.scalar(0x0F, .u8).broad(qshape);
    const lo0 = ql0_3.logical(.AND, mask4);
    const lo1 = ql1_3.logical(.AND, mask4);
    const hi0 = ql0_3.shiftRightLogical(zml.Tensor.scalar(4, .u8).broad(qshape)).logical(.AND, mask4);
    const hi1 = ql1_3.shiftRightLogical(zml.Tensor.scalar(4, .u8).broad(qshape)).logical(.AND, mask4);

    const q1 = sixBitGraph(qh_3, lo0, 0);
    const q2 = sixBitGraph(qh_3, lo1, 2);
    const q3 = sixBitGraph(qh_3, hi0, 4);
    const q4 = sixBitGraph(qh_3, hi1, 6);

    const scshape = zml.Shape.init(.{ .n = n, .kb = kb, .h = 2, .i = 2 }, .i8);
    const w1 = scaleChunkGraph(q1, sc1.reshape(scshape), n, kb);
    const w2 = scaleChunkGraph(q2, sc2.reshape(scshape), n, kb);
    const w3 = scaleChunkGraph(q3, sc3.reshape(scshape), n, kb);
    const w4 = scaleChunkGraph(q4, sc4.reshape(scshape), n, kb);

    const w12 = zml.Tensor.concatenate(&.{ w1, w2 }, -1);
    const w123 = zml.Tensor.concatenate(&.{ w12, w3 }, -1);
    const w1234 = zml.Tensor.concatenate(&.{ w123, w4 }, -1);
    const w = w1234.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .k = 256 }, .f32));
    const d4 = d.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .one = 1 }, .f32));
    return w.mul(d4).reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn sixBitGraph(qh: zml.Tensor, lo: zml.Tensor, comptime shift: u8) zml.Tensor {
    const bits = qh.shiftRightLogical(zml.Tensor.scalar(shift, .u8).broad(qh.shape()))
        .logical(.AND, zml.Tensor.scalar(3, .u8).broad(qh.shape()));
    const value = lo.logical(.OR, bits.shiftLeft(zml.Tensor.scalar(4, .u8).broad(qh.shape())));
    return value.convert(.f32).subConstant(32);
}

fn scaleChunkGraph(q: zml.Tensor, sc: zml.Tensor, n: i64, kb: i64) zml.Tensor {
    const q4 = q.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .h = 2, .i = 2, .e = 16 }, .f32));
    const s = sc.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .h = 2, .i = 2, .one = 1 }, .f32));
    return q4.mul(s).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .h = 2, .e = 32 }, .f32));
}

const kvalues_iq4nl = [16]i8{ -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

fn iq4Lut(idx: zml.Tensor) zml.Tensor {
    var result = zml.Tensor.scalar(kvalues_iq4nl[0], .f32).broad(idx.shape());
    inline for (1..16) |i| {
        const cond = idx.cmp(.EQ, zml.Tensor.scalar(i, .u8).broad(idx.shape()));
        result = zml.Tensor.select(cond, zml.Tensor.scalar(kvalues_iq4nl[i], .f32).broad(idx.shape()), result);
    }
    return result;
}

fn iq4GraphDequant(
    qs: zml.Tensor,
    scl: zml.Tensor,
    sch: zml.Tensor,
    d: zml.Tensor,
    n: i64,
    k: i64,
) zml.Tensor {
    const kb = @divExact(k, 256);
    const qshape = zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8, .e = 16 }, .u8);
    const q = qs.reshape(qshape);
    const lo = q.logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(qshape));
    const hi = q.shiftRightLogical(zml.Tensor.scalar(4, .u8).broad(qshape)).logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(qshape));
    const val_lo = iq4Lut(lo);
    const val_hi = iq4Lut(hi);

    const ib = zml.Tensor.iota(zml.Shape.init(.{ .ib = 8 }, .i32), .ib);
    const scl_exp = scl.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .four = 4, .one = 1 }, .u8))
        .broad(zml.Shape.init(.{ .n = n, .kb = kb, .four = 4, .two = 2 }, .u8))
        .reshape(zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8 }, .u8))
        .convert(.i32);
    const scl_shift = ib.remainderConst(2).scale(4).broad(zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8 }, .i32));
    const scl_part = scl_exp.shiftRightLogical(scl_shift).logical(.AND, zml.Tensor.scalar(0x0F, .i32).broad(scl_shift.shape()));

    const sch_exp = sch.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .one = 1 }, .u16))
        .broad(zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8 }, .u16))
        .convert(.i32);
    const sch_shift = ib.scale(2).broad(zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8 }, .i32));
    const sch_part = sch_exp.shiftRightLogical(sch_shift)
        .logical(.AND, zml.Tensor.scalar(3, .i32).broad(sch_shift.shape()))
        .shiftLeft(zml.Tensor.scalar(4, .i32).broad(sch_shift.shape()));

    const ls = scl_part.logical(.OR, sch_part).convert(.f32).subConstant(32);
    const dl = d.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .kb = kb, .one = 1 }, .f32)).broad(ls.shape()).mul(ls);

    const dl4 = dl.reshape(zml.Shape.init(.{ .n = n, .kb = kb, .ib = 8, .one = 1 }, .f32));
    const w_lo = val_lo.mul(dl4.broad(val_lo.shape()));
    const w_hi = val_hi.mul(dl4.broad(val_hi.shape()));
    const w = zml.Tensor.concatenate(&.{ w_lo, w_hi }, -1);
    return w.reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn iq4nlGraphDequant(qs: zml.Tensor, d: zml.Tensor, n: i64, k: i64) zml.Tensor {
    const nb = @divExact(k, 32);
    const qshape = zml.Shape.init(.{ .n = n, .nb = nb, .e = 16 }, .u8);
    const q = qs.reshape(qshape);
    const lo = q.logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(qshape));
    const hi = q.shiftRightLogical(zml.Tensor.scalar(4, .u8).broad(qshape)).logical(.AND, zml.Tensor.scalar(0x0F, .u8).broad(qshape));
    const val_lo = iq4Lut(lo);
    const val_hi = iq4Lut(hi);
    const d3 = d.convert(.f32).reshape(zml.Shape.init(.{ .n = n, .nb = nb, .one = 1 }, .f32)).broad(val_lo.shape());
    const w_lo = val_lo.mul(d3);
    const w_hi = val_hi.mul(d3);
    const w = zml.Tensor.concatenate(&.{ w_lo, w_hi }, -1);
    return w.reshape(zml.Shape.init(.{ .n = n, .k = k }, .f32)).convert(.bf16);
}

fn q8Matvec(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = quants.dim(.n);
    const cfg: q8_triton.Config = .{
        .m = m,
        .n = n,
        .k = k,
        .block_m = 16,
        .block_n = block_n,
        .block_k = 256,
        .stride_xm = k,
        .stride_xk = 1,
        .stride_qn = k,
        .stride_qk = 1,
        .stride_sn = @divExact(k, 32),
        .stride_sk = 1,
        .stride_om = n,
        .stride_on = 1,
    };
    return q8_triton.Q8Matvec.call(
        .{ .x_ptr = x, .q_ptr = quants, .s_ptr = scales },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q8Matvec1(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = quants.dim(.n);
    const cfg: q8_triton.Config = .{
        .m = 1,
        .n = n,
        .k = k,
        .block_m = 16,
        .block_n = block_n,
        .block_k = 256,
        .stride_xm = k,
        .stride_xk = 1,
        .stride_qn = k,
        .stride_qk = 1,
        .stride_sn = @divExact(k, 32),
        .stride_sk = 1,
        .stride_om = n,
        .stride_on = 1,
    };
    return q8_triton.Q8Matvec1.call(
        .{ .x_ptr = x, .q_ptr = quants, .s_ptr = scales },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q4Matvec(x: zml.Tensor, qs: zml.Tensor, sc: zml.Tensor, mn: zml.Tensor, d: zml.Tensor, dmin: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q4k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q4k_triton.Q4KMatvec.call(
        .{ .x_ptr = x, .qs_ptr = qs, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q4Matvec1(x: zml.Tensor, qs: zml.Tensor, sc: zml.Tensor, mn: zml.Tensor, d: zml.Tensor, dmin: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q4k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q4k_triton.Q4KMatvec1.call(
        .{ .x_ptr = x, .qs_ptr = qs, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q5Matvec(x: zml.Tensor, qs: zml.Tensor, qh: zml.Tensor, sc: zml.Tensor, mn: zml.Tensor, d: zml.Tensor, dmin: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q5k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_qh_n = @divExact(k, 8),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q5k_triton.Q5KMatvec.call(
        .{ .x_ptr = x, .qs_ptr = qs, .qh_ptr = qh, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q5Matvec1(x: zml.Tensor, qs: zml.Tensor, qh: zml.Tensor, sc: zml.Tensor, mn: zml.Tensor, d: zml.Tensor, dmin: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q5k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_qh_n = @divExact(k, 8),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q5k_triton.Q5KMatvec1.call(
        .{ .x_ptr = x, .qs_ptr = qs, .qh_ptr = qh, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q6Matvec(x: zml.Tensor, ql0: zml.Tensor, ql1: zml.Tensor, qh: zml.Tensor, sc1: zml.Tensor, sc2: zml.Tensor, sc3: zml.Tensor, sc4: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = ql0.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q6k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_ql_n = @divExact(k, 4),
        .stride_qh_n = @divExact(k, 4),
        .stride_sc_n = @divExact(k, 64),
        .stride_d_n = kb,
    };
    return q6k_triton.Q6KMatvec.call(
        .{ .x_ptr = x, .ql0_ptr = ql0, .ql1_ptr = ql1, .qh_ptr = qh, .sc1_ptr = sc1, .sc2_ptr = sc2, .sc3_ptr = sc3, .sc4_ptr = sc4, .d_ptr = d },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q6Matvec1(x: zml.Tensor, ql0: zml.Tensor, ql1: zml.Tensor, qh: zml.Tensor, sc1: zml.Tensor, sc2: zml.Tensor, sc3: zml.Tensor, sc4: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = ql0.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: q6k_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_ql_n = @divExact(k, 4),
        .stride_qh_n = @divExact(k, 4),
        .stride_sc_n = @divExact(k, 64),
        .stride_d_n = kb,
    };
    return q6k_triton.Q6KMatvec1.call(
        .{ .x_ptr = x, .ql0_ptr = ql0, .ql1_ptr = ql1, .qh_ptr = qh, .sc1_ptr = sc1, .sc2_ptr = sc2, .sc3_ptr = sc3, .sc4_ptr = sc4, .d_ptr = d },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn iq4Matvec(x: zml.Tensor, qs: zml.Tensor, scl: zml.Tensor, sch: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: iq4xs_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_scl_n = @divExact(k, 64),
        .stride_sch_n = kb,
        .stride_d_n = kb,
    };
    const lut = zml.Tensor.constantTensor(zml.Shape.init(.{ .e = 16 }, .i8), std.mem.sliceAsBytes(&kvalues_iq4nl));
    return iq4xs_triton.Iq4XsMatvec.call(
        .{ .x_ptr = x, .qs_ptr = qs, .scl_ptr = scl, .sch_ptr = sch, .d_ptr = d, .lut_ptr = lut },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn iq4Matvec1(x: zml.Tensor, qs: zml.Tensor, scl: zml.Tensor, sch: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const kb = @divExact(k, 256);
    const cfg: iq4xs_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_scl_n = @divExact(k, 64),
        .stride_sch_n = kb,
        .stride_d_n = kb,
    };
    const lut = zml.Tensor.constantTensor(zml.Shape.init(.{ .e = 16 }, .i8), std.mem.sliceAsBytes(&kvalues_iq4nl));
    return iq4xs_triton.Iq4XsMatvec1.call(
        .{ .x_ptr = x, .qs_ptr = qs, .scl_ptr = scl, .sch_ptr = sch, .d_ptr = d, .lut_ptr = lut },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn iq4nlMatvec(x: zml.Tensor, qs: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const cfg: iq4nl_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_d_n = @divExact(k, 32),
    };
    const lut = zml.Tensor.constantTensor(zml.Shape.init(.{ .e = 16 }, .i8), std.mem.sliceAsBytes(&kvalues_iq4nl));
    return iq4nl_triton.Iq4NlMatvec.call(
        .{ .x_ptr = x, .qs_ptr = qs, .d_ptr = d, .lut_ptr = lut },
        .{ .out = zml.Shape.init(.{ .m = m, .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, 16)), @intCast(@divExact(n, block_n)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn iq4nlMatvec1(x: zml.Tensor, qs: zml.Tensor, d: zml.Tensor, block_n: i64) zml.Tensor {
    const k = x.dim(.k);
    const n = qs.dim(.n);
    const cfg: iq4nl_triton.Config = .{
        .n = n,
        .k = k,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(k, 2),
        .stride_d_n = @divExact(k, 32),
    };
    const lut = zml.Tensor.constantTensor(zml.Shape.init(.{ .e = 16 }, .i8), std.mem.sliceAsBytes(&kvalues_iq4nl));
    return iq4nl_triton.Iq4NlMatvec1.call(
        .{ .x_ptr = x, .qs_ptr = qs, .d_ptr = d, .lut_ptr = lut },
        .{ .out = zml.Shape.init(.{ .n = n }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(n, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

pub const LinearLike = union(enum) {
    bf16: zml.nn.Linear,
    q8: QuantizedLinearQ8,
    q4: QuantizedLinearQ4,
    q5: QuantizedLinearQ5,
    q6: QuantizedLinearQ6,
    iq4: QuantizedLinearIq4Xs,
    iq4nl: QuantizedLinearIq4Nl,

    pub const Scheme = enum { bf16, q8, q4, q5, q6, iq4, iq4nl };

    pub fn scheme(self: LinearLike) Scheme {
        return switch (self) {
            .bf16 => .bf16,
            .q8 => .q8,
            .q4 => .q4,
            .q5 => .q5,
            .q6 => .q6,
            .iq4 => .iq4,
            .iq4nl => .iq4nl,
        };
    }

    pub fn init(
        store: zml.io.TensorStore.View,
        tags: anytype,
        mesh_: @EnumLiteral(),
        partitioning: anytype,
        bias_tags: anytype,
        bias_mesh: @EnumLiteral(),
        bias_partitioning: anytype,
        comptime contraction_tag: anytype,
    ) LinearLike {
        if (store.getShape("weight.iq4nl.qs") != null) {
            return .{ .iq4nl = QuantizedLinearIq4Nl.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        if (store.getShape("weight.iq4.qs") != null) {
            return .{ .iq4 = QuantizedLinearIq4Xs.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        if (store.getShape("weight.q6.ql0") != null) {
            return .{ .q6 = QuantizedLinearQ6.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        if (store.getShape("weight.q5.qs") != null) {
            return .{ .q5 = QuantizedLinearQ5.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        if (store.getShape("weight.q4.qs") != null) {
            return .{ .q4 = QuantizedLinearQ4.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        if (store.getShape("weight.q8") != null) {
            return .{ .q8 = QuantizedLinearQ8.init(store, tags, mesh_, partitioning, bias_tags, bias_mesh, bias_partitioning, contraction_tag) };
        }
        return .{ .bf16 = zml.nn.Linear.init(
            store.createTensor("weight", tags, mesh_, partitioning),
            store.maybeCreateTensor("bias", bias_tags, bias_mesh, bias_partitioning),
            contraction_tag,
        ) };
    }

    pub fn forward(self: LinearLike, x: zml.Tensor, dtype: zml.DataType) zml.Tensor {
        return switch (self) {
            .bf16 => |linear| linear.forward(x, dtype),
            .q8 => |linear| linear.forward(x, dtype),
            .q4 => |linear| linear.forward(x, dtype),
            .q5 => |linear| linear.forward(x, dtype),
            .q6 => |linear| linear.forward(x, dtype),
            .iq4 => |linear| linear.forward(x, dtype),
            .iq4nl => |linear| linear.forward(x, dtype),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LinearLike)) void {
        switch (self.*) {
            .bf16 => |*linear| zml.nn.Linear.unloadBuffers(linear),
            .q8 => |*linear| zml.Buffer.deinitAll(QuantizedLinearQ8, linear),
            .q4 => |*linear| zml.Buffer.deinitAll(QuantizedLinearQ4, linear),
            .q5 => |*linear| zml.Buffer.deinitAll(QuantizedLinearQ5, linear),
            .q6 => |*linear| zml.Buffer.deinitAll(QuantizedLinearQ6, linear),
            .iq4 => |*linear| zml.Buffer.deinitAll(QuantizedLinearIq4Xs, linear),
            .iq4nl => |*linear| zml.Buffer.deinitAll(QuantizedLinearIq4Nl, linear),
        }
    }
};

pub const LayerKey = struct {
    kind: LayerType,
    schemes: [8]LinearLike.Scheme,

    pub fn eql(a: LayerKey, b: LayerKey) bool {
        return a.kind == b.kind and std.mem.eql(LinearLike.Scheme, &a.schemes, &b.schemes);
    }
};

pub const Mlp = struct {
    up_proj: LinearLike,
    gate_proj: LinearLike,
    down_proj: LinearLike,

    fn writeSchemeKey(self: Mlp, schemes: *[8]LinearLike.Scheme, offset: usize) void {
        schemes[offset + 0] = self.up_proj.scheme();
        schemes[offset + 1] = self.gate_proj.scheme();
        schemes[offset + 2] = self.down_proj.scheme();
    }

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = LinearLike.init(
                store.withPrefix("up_proj"),
                .{ .dout, .d },
                .model,
                .{ .dout = .model, .d = .replicated },
                .{.dout},
                .model,
                .{ .dout = .model },
                .d,
            ),
            .gate_proj = LinearLike.init(
                store.withPrefix("gate_proj"),
                .{ .dout, .d },
                .model,
                .{ .dout = .model, .d = .replicated },
                .{.dout},
                .model,
                .{ .dout = .model },
                .d,
            ),
            .down_proj = LinearLike.init(
                store.withPrefix("down_proj"),
                .{ .d, .dout },
                .model,
                .{ .d = .replicated, .dout = .model },
                .{.d},
                .model,
                .{ .d = .replicated },
                .dout,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        LinearLike.unloadBuffers(&self.up_proj);
        LinearLike.unloadBuffers(&self.gate_proj);
        LinearLike.unloadBuffers(&self.down_proj);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const up_projed = self.up_proj.forward(x, x.dtype());
        const gate = self.gate_proj.forward(x, x.dtype());
        const hidden = gate.silu().mul(up_projed);

        const output = self.down_proj.forward(hidden, hidden.dtype());
        return output;
    }
};

pub const SelfAttn = struct {
    q_proj: LinearLike,
    k_proj: LinearLike,
    v_proj: LinearLike,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rotary_embed: TextRotaryEmbedding,
    o_proj: LinearLike,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype, bias_partitions: anytype) LinearLike {
        return LinearLike.init(
            store,
            .{ .dout, .d },
            .model,
            partitions,
            .{.dout},
            .model,
            bias_partitions,
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) !SelfAttn {
        const rotary_dim: i64 = @intFromFloat(@as(f32, @floatFromInt(config.text_config.head_dim)) *
            config.text_config.rope_parameters.partial_rotary_factor);
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }, .{ .dout = .model }),
            .o_proj = initProj(store.withPrefix("o_proj"), .{ .dout = .replicated, .d = .model }, .{ .dout = .replicated }),
            .q_norm = RmsNorm.init(store.withPrefix("q_norm"), config.text_config.rms_norm_eps),
            .k_norm = RmsNorm.init(store.withPrefix("k_norm"), config.text_config.rms_norm_eps),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            .head_dim = config.text_config.head_dim,
            .rotary_dim = rotary_dim,
            .rotary_embed = .init(rotary_dim, config.text_config.rope_parameters.rope_theta, config.text_config.rope_parameters.mrope_section),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        LinearLike.unloadBuffers(&self.q_proj);
        LinearLike.unloadBuffers(&self.k_proj);
        LinearLike.unloadBuffers(&self.v_proj);
        LinearLike.unloadBuffers(&self.o_proj);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn writeSchemeKey(self: SelfAttn, schemes: *[8]LinearLike.Scheme) void {
        schemes[0] = self.q_proj.scheme();
        schemes[1] = self.k_proj.scheme();
        schemes[2] = self.v_proj.scheme();
        schemes[3] = self.o_proj.scheme();
    }

    fn projectQAndGate(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const q_proj = self.q_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        const q, var gate = q_proj.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });
        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const num_kv_heads = b: {
            if (self.num_kv_heads > 0) break :b self.num_kv_heads;
            break :b self.num_heads;
        };
        const k = self.k_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x, x.dtype()).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    fn partitionProjectedKv(kv: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
        return switch (kv_head_sharding) {
            .sharded => |heads| blk: {
                const sharded_kv = if (heads.factor != 1) kv.stutter1d(kv.axis(.h), heads.factor) else kv;
                break :blk sharded_kv.withPartitioning(.model, .{ .s = .replicated, .h = .model, .hd = .replicated });
            },
            .replicated => kv.withPartitioning(.model, .{ .s = .replicated, .h = .replicated, .hd = .replicated }),
        };
    }

    fn partitionCachedKv(tensor: zml.Tensor, kv_head_sharding: zml.Sharding.DimSharding) zml.Tensor {
        var cache_tensor = tensor.rename(.{ .s = .k });
        return switch (kv_head_sharding) {
            .sharded => |heads| blk: {
                if (heads.factor != 1) {
                    cache_tensor = cache_tensor.stutter1d(cache_tensor.axis(.h), heads.factor);
                }
                break :blk cache_tensor.withPartitioning(.model, .{ .k = .replicated, .h = .model, .hd = .replicated });
            },
            .replicated => cache_tensor.withPartitioning(.model, .{ .k = .replicated, .h = .replicated, .hd = .replicated }),
        };
    }

    pub fn forward(
        self: SelfAttn,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache.SelfAttnCache,
    ) struct { zml.Tensor, KvCache.SelfAttnCache } {
        const tp = zml.Compiler.current().mesh(.model);
        const x_qkv = x.withPartitioning(tp, .{ .d = .replicated });

        var q, var gate = self.projectQAndGate(x_qkv);
        var k, var v = self.projectKV(x_qkv);
        const kv_head_sharding = tp.shardableDim(k.dim(.h), .model, q.dim(.h));

        k = partitionProjectedKv(k, kv_head_sharding);
        v = partitionProjectedKv(v, kv_head_sharding);
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const position_ids = zml.Tensor.arange(.{ .end = x.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64))
            .add(token_index.convert(.i64).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64)));

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);
        k = partitionProjectedKv(k, kv_head_sharding);
        v = partitionProjectedKv(v, kv_head_sharding);

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        q = q.rename(.{ .s = .q });
        k = partitionCachedKv(k, kv_head_sharding);
        v = partitionCachedKv(v, kv_head_sharding);

        const head_partitioning: zml.attention.Partitioning = switch (kv_head_sharding) {
            .sharded => .head(.model),
            .replicated => .head(.replicated),
        };
        const attn_output = zml.attention.attention(
            q,
            k,
            v,
            token_index,
            zml.attention.Metadata.init(.fromBackend(.stablehlo)),
            zml.attention.Parameters.init(.fromBackend(.stablehlo)),
            zml.Compiler.current().mesh(.model),
            head_partitioning,
        ).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const projected_output = self.o_proj
            .forward(gated_output.rename(.{ .d_out_proj = .d }), gated_output.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(tp, .{ .d = .replicated });

        return .{ projected_output, new_kv_cache };
    }
};

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64,
    mrope_section: [3]i64,

    pub fn init(rotary_dim: i64, theta: f32, mrope_section: [3]i64) TextRotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = theta } },
            },
            .rotary_dim = rotary_dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);

        const emb = zml.Tensor.concatenate(&.{ freqs_t, freqs_t }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    pub fn getCosAndSinInterleaved(
        self: TextRotaryEmbedding,
        position_ids: zml.Tensor,
        dtype: zml.DataType,
    ) struct { zml.Tensor, zml.Tensor } {
        // To be used later in image extension.
        const stacked_position_ids = zml.Tensor.stack(&.{ position_ids, position_ids, position_ids }, 0, .g).convert(.f32);
        const inv_freq = zml.nn.InvFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        var freqs = stacked_position_ids.outer(inv_freq);
        var freqs_t, var freqs_h, var freqs_w = freqs.chunkExact(.g, 3);
        freqs_t = freqs_t.squeeze(.g);
        freqs_h = freqs_h.squeeze(.g);
        freqs_w = freqs_w.squeeze(.g);

        const h_indices = zml.Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[1] }, .i32), .h).scale(3).addConstant(1);
        const w_indices = zml.Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[2] }, .i32), .h).scale(3).addConstant(2);

        const h_input = freqs_h.gather(.{ .dh = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .dh = w_indices }, .{ .indices_are_sorted = true });
        freqs_t = freqs_t.scatterSlices(.{ .dh = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs_t.scatterSlices(.{ .dh = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });

        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyRope(self: TextRotaryEmbedding, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_rot = x.slice(-1, .{ .start = 0, .end = self.rotary_dim });
        const x_pass = x.slice(-1, .{ .start = self.rotary_dim, .end = x.dim(-1) });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));

        return zml.Tensor.concatenate(&.{ rotated, x_pass }, -1);
    }
};

/// Gated-delta recurrence that additionally snapshots the recurrent state at
/// every step of the block, so a speculative verification can roll the state
/// back to the accepted prefix.
const GdnCapture = struct {
    queries: zml.Tensor,
    keys: zml.Tensor,
    values: zml.Tensor,
    alphas: zml.Tensor,
    betas: zml.Tensor,
    active_length: zml.Tensor,

    pub const State = struct {
        step: zml.Tensor,
        s: zml.Tensor,
        outputs: zml.Tensor,
        stacked: zml.Tensor,
    };

    pub fn cond(gdn: GdnCapture, state: State) zml.Tensor {
        const active_length = gdn.active_length
            .convert(.i32)
            .maximum(zml.Tensor.scalar(0, .i32))
            .minimum(zml.Tensor.scalar(gdn.queries.dim(.s), .i32));
        return state.step.cmp(.LT, active_length);
    }

    pub fn sliceStep(input: zml.Tensor, step_: zml.Tensor) zml.Tensor {
        return input.slice(.s, .dynSingle(step_));
    }

    pub fn body(gdn: GdnCapture, state: State) State {
        const q = sliceStep(gdn.queries, state.step);
        const k = sliceStep(gdn.keys, state.step);
        const v = sliceStep(gdn.values, state.step);
        const alpha = sliceStep(gdn.alphas, state.step);
        const beta = sliceStep(gdn.betas, state.step);

        const hvk = state.s.shape();
        const hv = hvk.drop(.k);

        const v_hat = state.s.dot(k, .k).mul(alpha.broad(hv));
        const delta = v.sub(v_hat).mul(beta.broad(hv));
        const delta_k = delta.broad(hvk).mul(k.broad(hvk));
        const s_new = state.s.mul(alpha.broad(hvk)).add(delta_k);
        const y = s_new.dot(q, .k);

        return .{
            .s = s_new,
            .outputs = state.outputs.dynamicUpdateSlice(.{ .s = state.step }, y),
            .stacked = state.stacked.dynamicUpdateSlice(.{ .s = state.step }, s_new),
            .step = state.step.addConstant(1),
        };
    }
};

pub const GatedDeltaNet = struct {
    in_proj_qkv: LinearLike,
    in_proj_z: LinearLike,
    in_proj_b: LinearLike,
    in_proj_a: LinearLike,
    out_proj: LinearLike,
    conv1d_weight: zml.Tensor,
    dt_bias: zml.Tensor,
    aLog: zml.Tensor,
    norm: RmsNormGated,

    num_k_heads: i64,
    num_v_heads: i64,
    qk_head_repetition: u32,
    head_k_dim: i64,
    head_v_dim: i64,
    conv_kernel_size: i64,

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype) LinearLike {
        return LinearLike.init(
            store,
            .{ .dout, .d },
            .model,
            partitions,
            .{.dout},
            .model,
            .{ .dout = .replicated },
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, config: Config) GatedDeltaNet {
        const qk_head_repetition: u32 =
            @intCast(@divExact(config.text_config.linear_num_value_heads, config.text_config.linear_num_key_heads));
        return .{
            .in_proj_qkv = initProj(store.withPrefix("in_proj_qkv"), .{ .dout = .model, .d = .replicated }),
            .in_proj_z = initProj(store.withPrefix("in_proj_z"), .{ .dout = .model, .d = .replicated }),
            .in_proj_b = initProj(store.withPrefix("in_proj_b"), .{ .dout = .model, .d = .replicated }),
            .in_proj_a = initProj(store.withPrefix("in_proj_a"), .{ .dout = .model, .d = .replicated }),
            .out_proj = initProj(store.withPrefix("out_proj"), .{ .dout = .replicated, .d = .model }),
            .conv1d_weight = store.withPrefix("conv1d").createTensor("weight", .{ .out, .in, .kernel_size }, .model, .{ .out = .model, .in = .replicated, .kernel_size = .replicated }),
            .dt_bias = store.createTensor("dt_bias", .{.vh}, .model, .{ .vh = .model }),
            .aLog = store.createTensor("A_log", .{.vh}, .model, .{ .vh = .model }),
            .norm = RmsNormGated.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .num_k_heads = config.text_config.linear_num_key_heads,
            .num_v_heads = config.text_config.linear_num_value_heads,
            .qk_head_repetition = qk_head_repetition,
            .head_k_dim = config.text_config.linear_key_head_dim,
            .head_v_dim = config.text_config.linear_value_head_dim,
            .conv_kernel_size = config.text_config.linear_conv_kernel_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedDeltaNet)) void {
        LinearLike.unloadBuffers(&self.in_proj_qkv);
        LinearLike.unloadBuffers(&self.in_proj_z);
        LinearLike.unloadBuffers(&self.in_proj_b);
        LinearLike.unloadBuffers(&self.in_proj_a);
        LinearLike.unloadBuffers(&self.out_proj);
        self.conv1d_weight.deinit();
        self.dt_bias.deinit();
        self.aLog.deinit();
        RmsNormGated.unloadBuffers(&self.norm);
    }

    fn writeSchemeKey(self: GatedDeltaNet, schemes: *[8]LinearLike.Scheme) void {
        schemes[0] = self.in_proj_qkv.scheme();
        schemes[1] = self.in_proj_z.scheme();
        schemes[2] = self.in_proj_b.scheme();
        schemes[3] = self.in_proj_a.scheme();
        schemes[4] = self.out_proj.scheme();
    }

    fn recurrentGatedDeltaRule(
        query: zml.Tensor,
        key: zml.Tensor,
        value: zml.Tensor,
        g: zml.Tensor,
        beta: zml.Tensor,
        initial_state: ?zml.Tensor,
        active_length: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32, const key_f32, const value_f32, const alpha_f32, const beta_f32 = .{
            query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k }),
            key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k }),
            value.convert(.f32).rename(.{ .vh = .h, .vhd = .v }),
            g.convert(.f32).exp().rename(.{ .vh = .h }),
            beta.convert(.f32).rename(.{ .vh = .h }),
        };

        const initial_recurrent_state = b: {
            if (initial_state) |state| {
                break :b state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k });
            }

            break :b zml.Tensor.constant(zml.DataType.zero(.f32)).broad(zml.Shape.init(.{
                .b = value.dim(.b),
                .h = value.dim(.vh),
                .v = value.dim(.vhd),
                .k = query.dim(.khd),
            }, .f32));
        };

        const result = zml.nn.GatedDeltaNet.forward(
            query_f32,
            key_f32,
            value_f32,
            alpha_f32,
            beta_f32,
            .{ .s = initial_recurrent_state },
            active_length,
        );

        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            result.state.s.transpose(.{ .b, .h, .k, .v }).rename(.{ .h = .vh, .k = .khd, .v = .vhd }),
        };
    }

    /// Same as `recurrentGatedDeltaRule`, but also returns the recurrent state
    /// after every block position: `stacked[j]` is the state that has consumed
    /// block positions 0..j.
    fn recurrentGatedDeltaRuleStacked(
        query: zml.Tensor,
        key: zml.Tensor,
        value: zml.Tensor,
        g: zml.Tensor,
        beta: zml.Tensor,
        initial_state: zml.Tensor,
        active_length: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32, const key_f32, const value_f32, const alpha_f32, const beta_f32 = .{
            query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k }),
            key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k }),
            value.convert(.f32).rename(.{ .vh = .h, .vhd = .v }),
            g.convert(.f32).exp().rename(.{ .vh = .h }),
            beta.convert(.f32).rename(.{ .vh = .h }),
        };

        const initial_recurrent_state = initial_state
            .convert(.f32)
            .transpose(.{ .b, .vh, .vhd, .khd })
            .rename(.{ .vh = .h, .vhd = .v, .khd = .k });

        const gdn: GdnCapture = .{
            .queries = query_f32,
            .keys = key_f32,
            .values = value_f32,
            .alphas = alpha_f32,
            .betas = beta_f32,
            .active_length = active_length,
        };
        const state: GdnCapture.State = .{
            .step = zml.Tensor.scalar(0, .i32),
            .s = initial_recurrent_state,
            .outputs = zml.Tensor.zeroes(value_f32.shape()),
            .stacked = zml.Tensor.zeroes(zml.Shape.init(.{
                .s = query_f32.dim(.s),
                .b = value_f32.dim(.b),
                .h = value_f32.dim(.h),
                .v = value_f32.dim(.v),
                .k = query_f32.dim(.k),
            }, .f32)),
        };
        const final_state = zml.ops.@"while"(GdnCapture, gdn, state);
        return .{
            final_state.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            final_state.stacked.rename(.{ .h = .vh, .v = .vhd, .k = .khd }),
        };
    }

    fn buildUpdatedConvState(input: zml.Tensor, left_pad: i64) zml.Tensor {
        const copy_len = @min(input.dim(.s), left_pad);
        const tail = input.slice(.s, .{ .start = input.dim(.s) - copy_len, .end = input.dim(.s) });
        if (copy_len == left_pad) return tail;

        const padding_shape = zml.Shape.init(.{ .b = input.dim(.b), .s = left_pad - copy_len, .mix = input.dim(.mix) }, input.dtype());
        const padding = zml.Tensor.constant(input.dtype().zero()).broad(padding_shape);
        return zml.Tensor.concatenate(&.{ padding, tail }, .s);
    }

    fn buildUpdatedConvStateFromPrefix(input: zml.Tensor, left_pad: i64, active_length: zml.Tensor) zml.Tensor {
        const padding = zml.Tensor.zeroes(input.shape().setDim(.s, left_pad));
        const padded = zml.Tensor.concatenate(&.{ padding, input }, .s);
        return padded.slice(.s, .dyn(active_length.convert(.i64), left_pad));
    }

    pub fn forward(self: GatedDeltaNet, x: zml.Tensor, cache: KvCache.GatedDeltaNetCache, active_length: zml.Tensor) struct { zml.Tensor, KvCache.GatedDeltaNetCache } {
        return self.forwardWithCache(x, cache, active_length, x.dim(.s) == 1);
    }

    pub fn forwardWithCache(self: GatedDeltaNet, x: zml.Tensor, cache: KvCache.GatedDeltaNetCache, active_length: zml.Tensor, use_prior_state: bool) struct { zml.Tensor, KvCache.GatedDeltaNetCache } {
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const left_pad = self.conv_kernel_size - 1;

        const x_in = x.withPartitioning(.model, .{ .d = .replicated });
        const projected_qkv = self.in_proj_qkv.forward(x_in, x_in.dtype())
            .rename(.{ .dout = .mix })
            .withPartitioning(.model, .{ .s = .replicated, .mix = .model });
        const use_cached_state = use_prior_state and left_pad > 0;
        const conv_input = b: {
            if (use_cached_state) break :b zml.Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s);
            break :b projected_qkv;
        };

        const kernel = self.conv1d_weight;
        var mixed_qkv = zml.Tensor.conv1d(
            conv_input,
            kernel,
            .{
                .padding = &.{ left_pad, 0 },
                .input_batch_dimension = 0,
                .input_feature_dimension = 2,
                .input_spatial_dimensions = 1,
                .kernel_output_feature_dimension = 0,
                .kernel_input_feature_dimension = 1,
                .kernel_spatial_dimensions = 2,
                .output_batch_dimension = 0,
                .output_feature_dimension = 2,
                .output_spatial_dimensions = 1,
                .feature_group_count = conv_dim,
            },
        )
            .silu();

        if (use_cached_state) {
            mixed_qkv = mixed_qkv.slice(.s, .{ .start = mixed_qkv.dim(.s) - x.dim(.s), .end = mixed_qkv.dim(.s) });
        }
        mixed_qkv = mixed_qkv.withPartitioning(.model, .{ .s = .replicated, .mix = .model });

        const z = self.in_proj_z.forward(x_in, x_in.dtype())
            .splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });
        const b = self.in_proj_b.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });
        const a = self.in_proj_a.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });

        const query = mixed_qkv
            .slice(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const key = mixed_qkv
            .slice(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const value = mixed_qkv
            .slice(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });

        const beta = b.sigmoid();
        const aLog_type = self.aLog.dtype();
        const g = self.aLog.broad(a.shape()).exp().mul(zml.nn.softplus(a.convert(aLog_type).add(self.dt_bias.convert(aLog_type).broad(a.shape())), .{})).negate();

        const query_for_rule = b: {
            if (self.qk_head_repetition == 1) break :b query;
            break :b query.stutter1d(query.axis(.kh), self.qk_head_repetition);
        };
        const key_for_rule = b: {
            if (self.qk_head_repetition == 1) break :b key;
            break :b key.stutter1d(key.axis(.kh), self.qk_head_repetition);
        };

        const core_attn_out, const last_recurrent_state = recurrentGatedDeltaRule(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            b: {
                if (use_cached_state) break :b cache.recurrentState();
                break :b null;
            },
            active_length,
        );

        const core_attn_out_normed = self.norm
            .forward(
                core_attn_out.rename(.{ .vhd = .d }),
                z.rename(.{ .vhd = .d }),
            )
            .rename(.{ .d = .vhd });

        const output = self.out_proj
            .forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } }), core_attn_out_normed.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.model, .{ .d = .replicated });
        const updated_conv_state = if (use_cached_state)
            conv_input.slice(.s, .{ .start = conv_input.dim(.s) - left_pad, .end = conv_input.dim(.s) })
        else
            buildUpdatedConvStateFromPrefix(projected_qkv, left_pad, active_length);
        const updated_cache = cache.update(updated_conv_state, last_recurrent_state);
        return .{ output, updated_cache };
    }

    /// Speculative variant of `forwardWithCache`: always continues from the
    /// cached conv/recurrent state, runs the recurrence with per-step state
    /// snapshots, and leaves the cache untouched. Returns the block output,
    /// the stacked per-step recurrent states and the conv input features
    /// (`[cached conv state, block]`), for `commitSpeculativeState`.
    pub fn forwardSpeculative(
        self: GatedDeltaNet,
        x: zml.Tensor,
        cache: KvCache.GatedDeltaNetCache,
        active_length: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const left_pad = self.conv_kernel_size - 1;

        const x_in = x.withPartitioning(.model, .{ .d = .replicated });
        const projected_qkv = self.in_proj_qkv.forward(x_in, x_in.dtype())
            .rename(.{ .dout = .mix })
            .withPartitioning(.model, .{ .s = .replicated, .mix = .model });

        const conv_input = zml.Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s);

        const kernel = self.conv1d_weight;
        var mixed_qkv = zml.Tensor.conv1d(
            conv_input,
            kernel,
            .{
                .padding = &.{ left_pad, 0 },
                .input_batch_dimension = 0,
                .input_feature_dimension = 2,
                .input_spatial_dimensions = 1,
                .kernel_output_feature_dimension = 0,
                .kernel_input_feature_dimension = 1,
                .kernel_spatial_dimensions = 2,
                .output_batch_dimension = 0,
                .output_feature_dimension = 2,
                .output_spatial_dimensions = 1,
                .feature_group_count = conv_dim,
            },
        )
            .silu();

        mixed_qkv = mixed_qkv.slice(.s, .{ .start = mixed_qkv.dim(.s) - x.dim(.s), .end = mixed_qkv.dim(.s) });
        mixed_qkv = mixed_qkv.withPartitioning(.model, .{ .s = .replicated, .mix = .model });

        const z = self.in_proj_z.forward(x_in, x_in.dtype())
            .splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });
        const b = self.in_proj_b.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });
        const a = self.in_proj_a.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });

        const query = mixed_qkv
            .slice(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const key = mixed_qkv
            .slice(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const value = mixed_qkv
            .slice(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });

        const beta = b.sigmoid();
        const aLog_type = self.aLog.dtype();
        const g = self.aLog.broad(a.shape()).exp().mul(zml.nn.softplus(a.convert(aLog_type).add(self.dt_bias.convert(aLog_type).broad(a.shape())), .{})).negate();

        const query_for_rule = b: {
            if (self.qk_head_repetition == 1) break :b query;
            break :b query.stutter1d(query.axis(.kh), self.qk_head_repetition);
        };
        const key_for_rule = b: {
            if (self.qk_head_repetition == 1) break :b key;
            break :b key.stutter1d(key.axis(.kh), self.qk_head_repetition);
        };

        const core_attn_out, const stacked = recurrentGatedDeltaRuleStacked(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            cache.recurrentState(),
            active_length,
        );

        const core_attn_out_normed = self.norm
            .forward(
                core_attn_out.rename(.{ .vhd = .d }),
                z.rename(.{ .vhd = .d }),
            )
            .rename(.{ .d = .vhd });

        const output = self.out_proj
            .forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } }), core_attn_out_normed.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.model, .{ .d = .replicated });
        return .{ output, stacked, conv_input };
    }
};

/// Rolls the GatedDeltaNet cache forward to the accepted prefix of a
/// speculative block. `stacked_states[j]` is the recurrent state after block
/// position j, so accepting `valid` draft tokens keeps `stacked_states[valid]`;
/// the conv state becomes the features of the last committed tokens.
pub fn commitSpeculativeState(
    cache: KvCache.GatedDeltaNetCache,
    stacked_states: zml.Tensor,
    conv_feats: zml.Tensor,
    valid_draft_tokens: zml.Tensor,
) KvCache.GatedDeltaNetCache {
    const valid = valid_draft_tokens.convert(.i64);
    const selected = stacked_states.slice(.s, .dyn(valid, 1)).squeeze(.s);
    const new_recurrent = selected
        .transpose(.{ .b, .layer, .vh, .khd, .vhd })
        .convert(cache.recurrent_state.dtype())
        .reuseBuffer(cache.recurrent_state);
    const conv_start = valid.addConstant(1);
    const new_conv = conv_feats
        .slice(.s, .dyn(conv_start, cache.conv_state.dim(.s)))
        .convert(cache.conv_state.dtype())
        .reuseBuffer(cache.conv_state);
    return .{
        .conv_state = new_conv,
        .recurrent_state = new_recurrent,
        .layer_index = cache.layer_index,
    };
}

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, .model, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        return normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
    }
};

pub const RmsNormGated = struct {
    weight: zml.Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNormGated {
        return .{ .weight = store.createTensor("weight", .{.d}, .model, .{ .d = .replicated }), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNormGated)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNormGated, x: zml.Tensor, gate: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const gate_f32 = gate.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        const output = normalized.mul(weight_f32.broad(x.shape()));

        const gated_output = output.mul(gate_f32.silu());
        return gated_output.convert(x.dtype());
    }
};

pub const KvCache = struct {
    layer_types: []const LayerType,
    self_attn: SelfAttnCache,
    gated_delta_net: GatedDeltaNetCache,

    pub const SelfAttnCache = struct {
        k: zml.Tensor,
        v: zml.Tensor,
        layer_index: zml.Tensor,

        pub const Buffers = zml.Bufferized(SelfAttnCache);

        pub fn init(config: Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType, tp: *const zml.Mesh) SelfAttnCache {
            const num_self_attn_layers = countLayers(config.text_config.layer_types, .full_attention);
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_self_attn_layers,
                .s = max_seq_len,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, dtype);
            const kv_head_sharding = tp.shardableDim(kv_shape.dim(.h), .model, config.text_config.num_attention_heads);
            const sharded_kv_shape = switch (kv_head_sharding) {
                .sharded => |heads| kv_shape.setDim(.h, heads.dim).withPartitioning(tp, .{ .h = .model }),
                .replicated => kv_shape.withPartitioning(tp, .{ .h = .replicated }),
            };
            return .{
                .k = .fromShape(sharded_kv_shape),
                .v = .fromShape(sharded_kv_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(kv: SelfAttnCache, io: std.Io, platform: *const zml.Platform) !SelfAttnCache.Buffers {
            return .{
                .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), .{}),
                .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
            };
        }

        pub fn deinitBuffer(kv: *SelfAttnCache.Buffers) void {
            kv.k.deinit();
            kv.v.deinit();
            kv.layer_index.deinit();
        }

        pub fn keys(kv: SelfAttnCache) zml.Tensor {
            return kv.k.slice(.layer, .dynSingle(kv.layer_index));
        }

        pub fn values(kv: SelfAttnCache) zml.Tensor {
            return kv.v.slice(.layer, .dynSingle(kv.layer_index));
        }

        pub fn update(kv: SelfAttnCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) SelfAttnCache {
            const k_shape = kv.k.shape().drop(.layer);
            var layer = kv.layer_index;
            if (token_index) |idx| {
                layer = layer.broad(idx.shape());
                return .{
                    .k = kv.k.scatterSlices(
                        .{ .layer = layer, .s = idx },
                        new_k.convert(kv.k.dtype()).transpose(k_shape),
                        .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                    ).reuseBuffer(kv.k),
                    .v = kv.v.scatterSlices(
                        .{ .layer = layer, .s = idx },
                        new_v.convert(kv.v.dtype()).transpose(k_shape),
                        .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                    ).reuseBuffer(kv.v),
                    .layer_index = kv.layer_index,
                };
            }

            return .{
                .k = kv.k.scatterSlices(
                    .{ .layer = layer },
                    new_k.convert(kv.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(kv.k),
                .v = kv.v.scatterSlices(
                    .{ .layer = layer },
                    new_v.convert(kv.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(kv.v),
                .layer_index = kv.layer_index,
            };
        }
        pub fn atLayer(kv: SelfAttnCache, layer_index: usize) SelfAttnCache {
            return .{
                .k = kv.k,
                .v = kv.v,
                .layer_index = zml.Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(kv: SelfAttnCache, other: SelfAttnCache) SelfAttnCache {
            return .{
                .k = kv.k.reuseBuffer(other.k),
                .v = kv.v.reuseBuffer(other.v),
                .layer_index = kv.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub const GatedDeltaNetCache = struct {
        conv_state: zml.Tensor,
        recurrent_state: zml.Tensor,
        layer_index: zml.Tensor,

        pub const Buffers = zml.Bufferized(GatedDeltaNetCache);

        pub fn init(config: Config, batch_dim: i64, conv_dtype: zml.DataType, recurrent_dtype: zml.DataType, tp: *const zml.Mesh) GatedDeltaNetCache {
            const num_linear_attn_layers = countLayers(config.text_config.layer_types, .linear_attention);
            const conv_dim =
                2 * config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim +
                config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim;
            const conv_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .s = config.text_config.linear_conv_kernel_dim - 1,
                .mix = conv_dim,
            }, conv_dtype);
            const recurrent_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .vh = config.text_config.linear_num_value_heads,
                .khd = config.text_config.linear_key_head_dim,
                .vhd = config.text_config.linear_value_head_dim,
            }, recurrent_dtype);
            const sharded_conv_state_shape = conv_state_shape.withPartitioning(tp, .{ .mix = .model });
            const sharded_recurrent_state_shape = recurrent_state_shape.withPartitioning(tp, .{ .vh = .model });
            return .{
                .conv_state = zml.Tensor.fromShape(sharded_conv_state_shape),
                .recurrent_state = zml.Tensor.fromShape(sharded_recurrent_state_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: GatedDeltaNetCache, io: std.Io, platform: *const zml.Platform) !GatedDeltaNetCache.Buffers {
            return .{
                .conv_state = try zml.Buffer.uninitialized(io, platform, self.conv_state.shape(), .{}),
                .recurrent_state = try zml.Buffer.uninitialized(io, platform, self.recurrent_state.shape(), .{}),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
            };
        }

        pub fn deinitBuffer(self: *GatedDeltaNetCache.Buffers) void {
            self.conv_state.deinit();
            self.recurrent_state.deinit();
            self.layer_index.deinit();
        }

        pub fn convState(self: GatedDeltaNetCache) zml.Tensor {
            return self.conv_state.slice(.layer, .dynSingle(self.layer_index));
        }

        pub fn recurrentState(self: GatedDeltaNetCache) zml.Tensor {
            return self.recurrent_state.slice(.layer, .dynSingle(self.layer_index));
        }

        pub fn update(self: GatedDeltaNetCache, new_conv_state: ?zml.Tensor, new_recurrent_state: ?zml.Tensor) GatedDeltaNetCache {
            const conv_state = b: {
                if (new_conv_state) |state| {
                    break :b self.conv_state.scatterSlices(
                        .{ .layer = self.layer_index },
                        state.convert(self.conv_state.dtype()).transpose(self.conv_state.shape().drop(.layer)),
                        .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                    ).reuseBuffer(self.conv_state);
                }

                break :b self.conv_state;
            };

            const recurrent_state = b: {
                if (new_recurrent_state) |state| {
                    break :b self.recurrent_state.scatterSlices(
                        .{ .layer = self.layer_index },
                        state.convert(self.recurrent_state.dtype()).transpose(self.recurrent_state.shape().drop(.layer)),
                        .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                    ).reuseBuffer(self.recurrent_state);
                }

                break :b self.recurrent_state;
            };

            return .{
                .conv_state = conv_state,
                .recurrent_state = recurrent_state,
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: GatedDeltaNetCache, layer_index: usize) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state,
                .recurrent_state = self.recurrent_state,
                .layer_index = zml.Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: GatedDeltaNetCache, other: GatedDeltaNetCache) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state.reuseBuffer(other.conv_state),
                .recurrent_state = self.recurrent_state.reuseBuffer(other.recurrent_state),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub fn init(
        config: Config,
        batch_dim: i64,
        max_seq_len: i64,
        cache_dtype: zml.DataType,
        recurrent_dtype: zml.DataType,
        tp: *const zml.Mesh,
    ) KvCache {
        return .{
            .layer_types = config.text_config.layer_types,
            .self_attn = .init(config, batch_dim, max_seq_len, cache_dtype, tp),
            .gated_delta_net = .init(config, batch_dim, cache_dtype, recurrent_dtype, tp),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .self_attn = try self.self_attn.initBuffer(io, platform),
            .gated_delta_net = try self.gated_delta_net.initBuffer(io, platform),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        SelfAttnCache.deinitBuffer(&self.self_attn);
        GatedDeltaNetCache.deinitBuffer(&self.gated_delta_net);
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .layer_types = self.layer_types,
            .self_attn = self.self_attn.reuseBuffer(other.self_attn),
            .gated_delta_net = self.gated_delta_net.reuseBuffer(other.gated_delta_net),
        };
    }

    pub const LayerView = struct {
        parent: KvCache,
        cache: union(enum) {
            self_attn: SelfAttnCache,
            linear_attn: GatedDeltaNetCache,
        },
    };

    pub fn atLayer(self: KvCache, layer_index: usize) LayerView {
        return switch (denseIndex(self.layer_types, layer_index)) {
            .full_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .self_attn = self.self_attn.atLayer(dense_index.layer_dense_index) },
            },
            .linear_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .linear_attn = self.gated_delta_net.atLayer(dense_index.layer_dense_index) },
            },
        };
    }

    fn countLayers(layer_types: []const LayerType, layer_type: LayerType) i64 {
        return @intCast(std.mem.countScalar(LayerType, layer_types, layer_type));
    }

    fn denseIndex(layer_types: []const LayerType, layer_index: usize) union(enum) {
        full_attention: struct { layer_dense_index: usize },
        linear_attention: struct { layer_dense_index: usize },
    } {
        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types[0..layer_index]) |layer_type| {
            switch (layer_type) {
                .full_attention => self_attn_layer_index += 1,
                .linear_attention => linear_attn_layer_index += 1,
            }
        }
        return switch (layer_types[layer_index]) {
            .full_attention => .{ .full_attention = .{ .layer_dense_index = self_attn_layer_index } },
            .linear_attention => .{ .linear_attention = .{ .layer_dense_index = linear_attn_layer_index } },
        };
    }
};
