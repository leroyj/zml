const std = @import("std");

const zml = @import("zml");

pub const KvCache = struct {
    kv: zml.Tensor,
    layer_index: ?u32,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape, mesh: *const zml.Mesh) KvCache {
        const sharded_shape = kv_shape.insert(.k, .{ .kv = 2 }).withPartitioning(mesh, .{ .h = .model });
        return .{
            .kv = .fromShape(sharded_shape),
            .layer_index = null,
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.kv.deinit();
    }

    pub fn replaceBuffers(dst: *Buffer, src: *Buffer) void {
        replaceBuffer(&dst.kv, &src.kv);
    }

    pub fn initZeroBuffer(
        kv: KvCache,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
    ) !Buffer {
        return .{
            .kv = try zeroBuffer(allocator, io, platform, kv.kv.shape()),
        };
    }

    pub fn keys(kv: KvCache) zml.Tensor {
        const layer: i64 = @intCast(kv.layer_index orelse @panic("forgot to call atLayer"));
        return kv.kv
            .slice(.layer, .{ .start = layer, .end = layer + 1 })
            .squeeze(.layer)
            .slice(.kv, .{ .start = 0, .end = 1 })
            .squeeze(.kv);
    }

    pub fn values(kv: KvCache) zml.Tensor {
        const layer: i64 = @intCast(kv.layer_index orelse @panic("forgot to call atLayer"));
        return kv.kv
            .slice(.layer, .{ .start = layer, .end = layer + 1 })
            .squeeze(.layer)
            .slice(.kv, .{ .start = 1, .end = 2 })
            .squeeze(.kv);
    }

    pub fn update(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor) KvCache {
        const update_shape = kv.kv.shape().drop(.layer);
        const layer: zml.Tensor = .scalar(kv.layer_index orelse @panic("forgot to call atLayer"), .u32);
        const new_kv = zml.Tensor.stack(&.{
            new_k.convert(kv.kv.dtype()),
            new_v.convert(kv.kv.dtype()),
        }, .k, .kv);

        return .{
            .kv = kv.kv.scatterSlices(.{ .layer = layer, .k = token_index }, new_kv.transpose(update_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.kv),
            .layer_index = kv.layer_index,
        };
    }

    pub fn atLayer(kv: KvCache, layer_index: usize) KvCache {
        return .{
            .kv = kv.kv,
            .layer_index = @intCast(layer_index),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .kv = kv.kv.reuseBuffer(other.kv),
            .layer_index = null,
        };
    }
};

pub const Config = struct {
    vocab_size: u32 = 0,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: ?u32 = null,
    attention_bias: bool = false,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    block_size: u32,
    conv_kernel_size: u32 = 2,
    conv_group_size: u32 = 16,
    selector_rank: u32 = 256,
    selector_top_k: u32 = 16,
    dflash_config: DFlashConfig,
};

pub const DFlashConfig = struct {
    target_layer_ids: []const u32,
    mask_token_id: ?u32 = null,
};

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    layers: []DecoderLayer,
    norm: RmsNorm,
    fc: zml.nn.Linear,
    hidden_norm: RmsNorm,
    target_layer_ids: []u32,
    selector_hidden: zml.nn.Linear,
    selector_predecessor: zml.Tensor,
    selector_successor: zml.Tensor,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        const layers = try allocator.alloc(DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = try .init(store.withPrefix("layers").withLayer(i), config);
        }

        const target_layer_ids = try allocator.alloc(u32, config.dflash_config.target_layer_ids.len);
        @memcpy(target_layer_ids, config.dflash_config.target_layer_ids);
        errdefer allocator.free(target_layer_ids);

        return .{
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps),
            .fc = .init(
                store.createTensor("fc.weight", .{ .dout, .d }, .model, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .hidden_norm = .init(store.withPrefix("hidden_norm"), config.rms_norm_eps),
            .target_layer_ids = target_layer_ids,
            .selector_hidden = .init(
                store.createTensor("selector_hidden.weight", .{ .rank, .d }, .model, .{ .rank = .replicated, .d = .replicated }),
                null,
                .d,
            ),
            .selector_predecessor = store.createTensor("selector_predecessor.weight", .{ .voc, .rank }, .model, .{ .voc = .replicated, .rank = .replicated }),
            .selector_successor = store.createTensor("selector_successor.weight", .{ .voc, .rank }, .model, .{ .voc = .replicated, .rank = .replicated }),
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
        allocator.free(self.target_layer_ids);
    }

    pub fn unloadBuffers(self: *Buffers, allocator: std.mem.Allocator) void {
        for (self.layers) |*layer| DecoderLayer.unloadBuffers(layer);
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
        self.fc.weight.deinit();
        if (self.fc.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.hidden_norm);
        self.selector_hidden.weight.deinit();
        if (self.selector_hidden.bias) |*bias| bias.deinit();
        self.selector_predecessor.deinit();
        self.selector_successor.deinit();
    }

    pub fn initKvCache(self: Model, config: Config, cache_seq_len: u32, mesh: *const zml.Mesh) KvCache {
        _ = self;
        return .init(.init(.{
            .layer = config.num_hidden_layers,
            .k = cache_seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse config.hidden_size / config.num_attention_heads,
        }, .f32), mesh);
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
    ) !Buffers {
        var buffers = try zml.mem.bufferize(allocator, Model, self);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .parallelism = 4,
            .dma_chunks = 4,
            .dma_chunk_size = 16 * zml.MiB,
        });
        defer loader.deinit();

        try loader.load(io, Model, self, &buffers, store, .{ .progress = progress });
        try loader.await(io);
        return buffers;
    }

    pub fn forward(
        self: Model,
        target_hidden_: zml.Tensor,
        noise_embedding_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        var hidden = noise_embedding_.withPartialTags(.{ .s, .d }).convert(self.norm.weight.dtype());
        const target_hidden = self.hidden_norm.forward(
            self.fc.forward(target_hidden_.withPartialTags(.{ .s, .d }).convert(self.fc.weight.dtype()), .bf16)
                .rename(.{ .dout = .d }),
        );
        const position_ids = position_ids_.withPartialTags(.{.s});
        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(
                hidden,
                target_hidden,
                position_ids,
                updated_kv_cache.atLayer(i),
                cache_index,
                active_context_len,
            );
        }

        return .{ self.norm.forward(hidden), updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const DecoderLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: DFlashAttention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    attn_conv_proj: zml.nn.Linear,
    attn_conv_base: zml.Tensor,
    ffn_conv_proj: zml.nn.Linear,
    ffn_conv_base: zml.Tensor,
    conv_kernel_size: i64,
    conv_group_size: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !DecoderLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp")),
            .attn_conv_proj = .init(
                store.createTensor("self_attn.attn_conv_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .attn_conv_base = store.createTensor("self_attn.attn_conv_base", .{ .d, .kernel, .side }, .model, .{ .d = .replicated, .kernel = .replicated, .side = .replicated }),
            .ffn_conv_proj = .init(
                store.createTensor("ffn_conv_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .ffn_conv_base = store.createTensor("ffn_conv_base", .{ .d, .kernel, .side }, .model, .{ .d = .replicated, .kernel = .replicated, .side = .replicated }),
            .conv_kernel_size = @intCast(config.conv_kernel_size),
            .conv_group_size = @intCast(config.conv_group_size),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DecoderLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        DFlashAttention.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        Mlp.unloadBuffers(&self.mlp);
        self.attn_conv_proj.weight.deinit();
        if (self.attn_conv_proj.bias) |*bias| bias.deinit();
        self.attn_conv_base.deinit();
        self.ffn_conv_proj.weight.deinit();
        if (self.ffn_conv_proj.bias) |*bias| bias.deinit();
        self.ffn_conv_base.deinit();
    }

    pub fn forward(
        self: DecoderLayer,
        hidden_states: zml.Tensor,
        target_hidden: zml.Tensor,
        position_ids: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const residual = hidden_states.withPartitioning(.model, .{ .d = .replicated });

        const attn_in = self.input_layernorm.forward(residual);
        const attn_dynamic = self.attn_conv_proj.forward(attn_in, attn_in.dtype());
        const attn_conv_in = conv(attn_in, attn_dynamic, self.attn_conv_base, 0, self.conv_kernel_size, self.conv_group_size);
        const attn_out_, const updated_kv_cache = self.self_attn.forward(
            attn_conv_in,
            target_hidden,
            position_ids,
            kv_cache,
            cache_index,
            active_context_len,
        );
        const attn_out = conv(attn_out_, attn_dynamic, self.attn_conv_base, 1, self.conv_kernel_size, self.conv_group_size);

        const post_attn = residual.add(attn_out).withPartitioning(.model, .{ .d = .replicated });
        const ffn_in = self.post_attention_layernorm.forward(post_attn);
        const ffn_dynamic = self.ffn_conv_proj.forward(ffn_in, ffn_in.dtype());
        const ffn_conv_in = conv(ffn_in, ffn_dynamic, self.ffn_conv_base, 0, self.conv_kernel_size, self.conv_group_size);
        const mlp_out_ = self.mlp.forward(ffn_conv_in)
            .rename(.{ .dout = .d })
            .withPartitioning(.model, .{ .d = .replicated });
        const mlp_out = conv(mlp_out_, ffn_dynamic, self.ffn_conv_base, 1, self.conv_kernel_size, self.conv_group_size);

        return .{ post_attn.add(mlp_out).withPartitioning(.model, .{ .d = .replicated }), updated_kv_cache };
    }
};

pub const DFlashAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: Config) !DFlashAttention {
        return .{
            .q_proj = linear(store, "q_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .k_proj = linear(store, "k_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .v_proj = linear(store, "v_proj", config.attention_bias, .{ .dout = .model, .d = .replicated }),
            .o_proj = linear(store, "o_proj", config.attention_bias, .{ .d = .model }),
            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = config.rope_theta } },
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DFlashAttention)) void {
        unloadLinear(&self.q_proj);
        unloadLinear(&self.k_proj);
        unloadLinear(&self.v_proj);
        unloadLinear(&self.o_proj);
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    pub fn forward(
        self: DFlashAttention,
        hidden_states_: zml.Tensor,
        target_hidden_: zml.Tensor,
        position_ids_: zml.Tensor,
        kv_cache: KvCache,
        cache_index: zml.Tensor,
        active_context_len: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const hidden_states = hidden_states_.withPartialTags(.{ .s, .d }).withPartitioning(.model, .{ .d = .replicated });
        const target_hidden = target_hidden_.withPartialTags(.{ .s, .d }).withPartitioning(.model, .{ .d = .replicated });
        const position_ids = position_ids_.withPartialTags(.{.s});

        var q = self.q_proj.forward(hidden_states, hidden_states.dtype())
            .splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });

        var new_k = zml.Tensor.concatenate(&.{
            self.k_proj.forward(target_hidden, target_hidden.dtype()),
            self.k_proj.forward(hidden_states, hidden_states.dtype()),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        var new_v = zml.Tensor.concatenate(&.{
            linearForwardF32(self.v_proj, target_hidden),
            linearForwardF32(self.v_proj, hidden_states),
        }, .s).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = .auto });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        new_k = self.k_norm.forward(new_k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const q_pos = position_ids.slice(.s, .{
            .start = position_ids.dim(.s) - hidden_states.dim(.s),
            .end = position_ids.dim(.s),
        });
        q = zml.nn.rope(q, q_pos, self.rope_opts).rename(.{ .s = .q });
        new_k = zml.nn.rope(new_k, position_ids, self.rope_opts).rename(.{ .s = .k });
        new_v = new_v.rename(.{ .s = .k });

        const updated_kv_cache = updateBucketed(kv_cache, new_k, new_v, cache_index, active_context_len, target_hidden.dim(.s));
        var k = updated_kv_cache.keys().convert(q.dtype());
        var v = updated_kv_cache.values().convert(.f32);
        k = k.withPartitioning(.model, .{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.model, .{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_shape = zml.Shape.init(.{ .q = q.dim(.q), .k = k.dim(.k) }, .bool);
        const k_idx = zml.Tensor.iota(attn_shape, .k).convert(cache_index.dtype());
        const active_context_end = cache_index.add(active_context_len.convert(cache_index.dtype()));
        const valid_k = active_context_end.add(zml.Tensor.scalar(hidden_states.dim(.s), cache_index.dtype()));
        const valid_mask = k_idx.cmp(.LT, valid_k.broad(k_idx.shape()));
        const zeros = zml.Tensor.scalar(@as(f32, 0.0), .f32).broad(attn_shape.withDtype(.f32));
        const minus_inf = zml.Tensor.scalar(-std.math.inf(f32), .f32).broad(attn_shape.withDtype(.f32));
        const attn_mask = valid_mask.select(zeros, minus_inf).convert(q.dtype());

        const attn = zml.nn.sdpa(q.convert(.f32), k.convert(.f32), v, .{ .attn_mask = attn_mask.convert(.f32) })
            .withPartitioning(.model, .{ .q = .replicated, .h = .model, .hd = .replicated })
            .merge(.{ .d = .{ .h, .hd } })
            .rename(.{ .q = .s })
            .convert(self.o_proj.weight.dtype());

        return .{
            self.o_proj.forward(attn, attn.dtype())
                .rename(.{ .dout = .d })
                .withPartitioning(.model, .{ .d = .replicated }),
            updated_kv_cache,
        };
    }
};

pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .model, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        return zml.nn.rmsNorm(x, .d, self.eps)
            .mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub const Mlp = struct {
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .model, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .model, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        unloadLinear(&self.gate_proj);
        unloadLinear(&self.up_proj);
        unloadLinear(&self.down_proj);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return self.down_proj.forward(
            self.gate_proj.forward(x, x.dtype())
                .silu()
                .mul(self.up_proj.forward(x, x.dtype()))
                .rename(.{ .dout = .d }),
            .bf16,
        );
    }
};

fn conv(
    hidden_: zml.Tensor,
    dynamic_: zml.Tensor,
    base_: zml.Tensor,
    side: i64,
    kernel_size: i64,
    group_size: i64,
) zml.Tensor {
    const hidden = hidden_.withPartialTags(.{ .s, .d });
    const s = hidden.dim(.s);
    const d = hidden.dim(.d);
    const groups = @divExact(d, group_size);

    // The projection is laid out [groups, kernel, side] with groups fastest.
    const dyn = dynamic_.reshape(zml.Shape.init(.{ .s = s, .side = 2, .kernel = kernel_size, .groups = groups }, dynamic_.dtype()))
        .transpose(.{ 0, 3, 2, 1 });
    const coeff = dyn.slice(.side, .{ .start = side, .end = side + 1 }).squeeze(.side);

    const coeff_exp = coeff.reshape(zml.Shape.init(.{ .s = s, .groups = groups, .one = 1, .kernel = kernel_size }, coeff.dtype()))
        .broad(zml.Shape.init(.{ .s = s, .groups = groups, .within = group_size, .kernel = kernel_size }, coeff.dtype()))
        .reshape(zml.Shape.init(.{ .s = s, .d = d, .kernel = kernel_size }, coeff.dtype()));

    const base = base_.slice(.side, .{ .start = side, .end = side + 1 }).squeeze(.side);
    const weight = coeff_exp.add(base.reshape(zml.Shape.init(.{ .one = 1, .d = d, .kernel = kernel_size }, base.dtype())).broad(coeff_exp.shape()));

    var result: zml.Tensor = undefined;
    for (0..@intCast(kernel_size)) |tap| {
        const values = if (tap == 0) hidden else shiftDown(hidden, @intCast(tap));
        const w = weight.slice(.kernel, .{ .start = @intCast(tap), .end = @intCast(tap + 1) }).squeeze(.kernel);
        const term = w.mul(values);
        result = if (tap == 0) term else result.add(term);
    }
    return result;
}

fn shiftDown(x: zml.Tensor, tap: i64) zml.Tensor {
    const s = x.dim(.s);
    const zeros = zml.Tensor.constant(x.dtype().zero()).broad(zml.Shape.init(.{ .s = tap, .d = x.dim(.d) }, x.dtype()));
    const prev = x.slice(.s, .{ .start = 0, .end = s - tap });
    return zml.Tensor.concatenate(&.{ zeros, prev }, .s);
}

pub fn selectorLattice(
    selector_hidden: zml.nn.Linear,
    selector_predecessor: zml.Tensor,
    selector_successor: zml.Tensor,
    logits_: zml.Tensor,
    hidden_: zml.Tensor,
    anchor: zml.Tensor,
    n_embd: i64,
    top_k: i64,
    rank: i64,
) zml.Tensor {
    const logits = logits_.withPartialTags(.{ .s, .voc });
    const s = logits.dim(.s);
    const gate = selector_hidden.forward(hidden_.withPartialTags(.{.d}), .bf16);

    const top = logits.topK(.{ .cand = .voc }, @intCast(top_k), .{ .descending = true });
    const cand = top.indices;
    const unary = top.values;

    const succ = selector_successor.gather(.{ .voc = cand }, .{});
    const anchor_b = anchor.convert(.i32).broad(zml.Shape.init(.{ .s = 1, .cand = top_k }, .i32));
    const prev_cand = cand.slice(.s, .{ .start = 0, .end = s - 1 });
    const pred_ids = zml.Tensor.concatenate(&.{ anchor_b, prev_cand }, .s);
    const pred = selector_predecessor.gather(.{ .voc = pred_ids }, .{});

    const gate_b = gate.reshape(zml.Shape.init(.{ .s = s, .one = 1, .rank = rank }, gate.dtype()));
    const cond = pred.mul(gate_b.broad(pred.shape()));

    const cond4 = cond.reshape(zml.Shape.init(.{ .s = s, .p = top_k, .one = 1, .rank = rank }, cond.dtype()));
    const succ4 = succ.reshape(zml.Shape.init(.{ .s = s, .one = 1, .c = top_k, .rank = rank }, succ.dtype()));
    const prod_shape = zml.Shape.init(.{ .s = s, .p = top_k, .c = top_k, .rank = rank }, cond.dtype());
    const prod = cond4.broad(prod_shape).mul(succ4.broad(prod_shape));
    const score = prod.sum(.rank).squeeze(.rank);
    const score_u = score.add(unary.reshape(zml.Shape.init(.{ .s = s, .one = 1, .c = top_k }, unary.dtype())).broad(score.shape()));

    const cand_f = cand.convert(.f32);
    const score_f = score_u.convert(.f32).reshape(zml.Shape.init(.{ .s = s, .flat = top_k * top_k }, score_u.dtype()));
    const row = zml.Tensor.concatenate(&.{ cand_f, score_f }, -1);
    const pad = zml.Tensor.scalar(0, .f32).broad(zml.Shape.init(.{ .s = s, .pad = n_embd - (top_k + top_k * top_k) }, .f32));
    return zml.Tensor.concatenate(&.{ row, pad }, -1);
}

fn linear(
    store: zml.io.TensorStore.View,
    comptime prefix: []const u8,
    with_bias: bool,
    partitioning: anytype,
) zml.nn.Linear {
    const prefixed = store.withPrefix(prefix);
    return .init(
        prefixed.createTensor("weight", .{ .dout, .d }, .model, partitioning),
        if (with_bias) prefixed.createTensor("bias", .{.dout}, .model, .{ .dout = .replicated }) else null,
        .d,
    );
}

fn unloadLinear(linear_: anytype) void {
    if (linear_.bias) |*bias| bias.deinit();
    linear_.weight.deinit();
}

fn linearForwardF32(linear_: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var y = x.convert(.f32).dot(linear_.weight.convert(.f32), linear_.tag);
    return if (linear_.bias) |bias| y.add(bias.convert(.f32).broad(y.shape())) else y;
}

fn updateBucketed(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, active_context_len: zml.Tensor, context_len: i64) KvCache {
    _ = active_context_len;
    _ = context_len;
    const layer_idx: i64 = @intCast(kv.layer_index orelse @panic("forgot to call atLayer"));
    const layer_slice = kv.kv.slice(.layer, .{ .start = layer_idx, .end = layer_idx + 1 }).squeeze(.layer);
    const new_kv = zml.Tensor.stack(&.{
        new_k.convert(kv.kv.dtype()),
        new_v.convert(kv.kv.dtype()),
    }, .k, .kv);
    const updated = layer_slice.dynamicUpdateSlice(.{ .k = token_index, .kv = zml.Tensor.scalar(0, .u32), .h = zml.Tensor.scalar(0, .u32), .hd = zml.Tensor.scalar(0, .u32) }, new_kv);
    return .{
        .kv = kv.kv.scatterSlices(
            .{ .layer = zml.Tensor.scalar(layer_idx, .u32) },
            updated.transpose(kv.kv.shape().drop(.layer)),
            .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
        ).reuseBuffer(kv.kv),
        .layer_index = kv.layer_index,
    };
}

fn zeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromSlice(io, platform, zml.Slice.init(shape, bytes));
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
