const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const gguf = @import("gguf.zig");
const dflash2 = @import("gguf/dflash2_model.zig");
const common = @import("models/common.zig");
const qwen3_5 = @import("models/qwen3_5/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};
const log = std.log.scoped(.dflash2);

const Args = struct {
    model: []const u8,
    draft: []const u8,
    prompt: []const u8 = "What is the capital of France? Answer in one sentence.",
    max_seq_len: u32 = 64,
    zero_context: bool = false,

    pub const help =
        \\Options:
        \\  --model=<path>       Path to the Qwen3.8 target GGUF
        \\  --draft=<path>       Path to the DFlash2 draft GGUF
        \\  --prompt=<text>      Prompt
        \\  --max-seq-len=<n>    Maximum new tokens
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    const meshes: common.Meshes = try .init(platform);

    // Target model.
    var target_ctx = try gguf.Context.init(allocator, io, args.model);
    defer target_ctx.deinit();
    var target_registry = try target_ctx.registry(allocator);
    defer target_registry.deinit();
    var target_store = zml.io.TensorStore.fromRegistry(allocator, &target_registry, &meshes.all());
    defer target_store.deinit();

    const cache_dir = try target_ctx.ensureCacheDir(allocator, io, init.environ_map);
    defer allocator.free(cache_dir);
    const repo = try std.Io.Dir.openDirAbsolute(io, cache_dir, .{});
    defer repo.close(io);

    const generation: common.GenerationOptions = .{ .sampling_strategy = .{ .topk = 1 } };
    var target_loaded = try qwen3_5.LoadedModel.init(allocator, io, repo, target_store.view(), generation);
    defer target_loaded.deinit(allocator);
    const target_model = target_loaded.inner;
    const target_config = target_loaded.parsed_config.value;

    // Draft model.
    var draft_ctx = try gguf.Context.init(allocator, io, args.draft);
    defer draft_ctx.deinit();
    var draft_registry = try draft_ctx.registryDflash(allocator);
    defer draft_registry.deinit();
    var draft_store = zml.io.TensorStore.fromRegistry(allocator, &draft_registry, &meshes.all());
    defer draft_store.deinit();

    const draft_vocab: u32 = @intCast((try draft_ctx.getArray("tokenizer.ggml.tokens")).len);
    const target_layer_ids_i64 = try draft_ctx.getIntArray("dflash.target_layers");
    const target_layer_ids = try allocator.alloc(u32, target_layer_ids_i64.len);
    defer allocator.free(target_layer_ids);
    for (target_layer_ids_i64, 0..) |id, i| target_layer_ids[i] = @intCast(id);

    const draft_config: dflash2.Config = .{
        .vocab_size = draft_vocab,
        .hidden_size = @intCast(try draft_ctx.getInt("dflash.embedding_length")),
        .intermediate_size = @intCast(try draft_ctx.getInt("dflash.feed_forward_length")),
        .num_hidden_layers = @intCast(try draft_ctx.getInt("dflash.block_count")),
        .num_attention_heads = @intCast(try draft_ctx.getInt("dflash.attention.head_count")),
        .num_key_value_heads = @intCast(try draft_ctx.getInt("dflash.attention.head_count_kv")),
        .head_dim = @intCast(try draft_ctx.getInt("dflash.attention.key_length")),
        .rms_norm_eps = @floatCast(try draft_ctx.getFloat("dflash.attention.layer_norm_rms_epsilon")),
        .rope_theta = @floatCast(try draft_ctx.getFloat("dflash.rope.freq_base")),
        .block_size = @intCast(try draft_ctx.getInt("dflash.block_size")),
        .conv_kernel_size = @intCast(try draft_ctx.getInt("dflash.conv_kernel_size")),
        .conv_group_size = @intCast(try draft_ctx.getInt("dflash.conv_group_size")),
        .selector_rank = @intCast(try draft_ctx.getInt("dflash.selector_rank")),
        .selector_top_k = @intCast(try draft_ctx.getInt("dflash.selector_top_k")),
        .dflash_config = .{ .target_layer_ids = target_layer_ids },
    };
    var draft_model = try dflash2.Model.init(allocator, draft_store.view(), draft_config);
    defer draft_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = "dflash2" });
    defer progress.end();

    log.info("Loading target weights...", .{});
    var target_buffers = try target_loaded.loadBuffers(allocator, io, platform, &target_store, &progress);
    defer qwen3_5.Model.unloadBuffers(&target_buffers, allocator);

    log.info("Loading draft weights...", .{});
    var draft_buffers = try draft_model.loadBuffers(allocator, io, platform, &draft_store, &progress);
    defer dflash2.Model.unloadBuffers(&draft_buffers, allocator);

    const seqlen: i64 = @max(@as(i64, args.max_seq_len) + 64, 512);
    const cache_dtype = target_model.text_model.embed_tokens.weight.dtype();
    const target_kv_cache = qwen3_5.KvCache.init(target_config, 1, seqlen, cache_dtype, .f32, meshes.model);
    var target_kv_buffers = try target_kv_cache.initBuffer(io, platform);
    defer qwen3_5.KvCache.deinitBuffer(&target_kv_buffers);

    const draft_kv_cache = draft_model.initKvCache(draft_config, @intCast(seqlen), meshes.model);
    var draft_kv_buffers = try draft_kv_cache.initZeroBuffer(allocator, io, platform);
    defer dflash2.KvCache.deinitBuffer(&draft_kv_buffers);

    const block = draft_config.block_size;
    const hidden = target_config.text_config.hidden_size;
    const rng = zml.Tensor.Rng.init();

    log.info("Compiling target prefill...", .{});
    const prefill_len: i64 = 64;
    var prefill_exe = try platform.compileFn(allocator, io, qwen3_5.Model.prefillForward, .{
        target_model,
        zml.Tensor.init(.{ .b = 1, .s = prefill_len }, .u32),
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
        target_kv_cache,
        target_layer_ids,
        rng,
    }, .{ .meshes = &meshes.all() });
    defer prefill_exe.deinit();

    log.info("Compiling target verify...", .{});
    var verify_exe = try platform.compileFn(allocator, io, qwen3_5.Model.verifyForward, .{
        target_model,
        zml.Tensor.init(.{ .b = 1, .s = block }, .u32),
        zml.Tensor.init(.{ .b = 1, .s = block }, .u32),
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
        target_kv_cache,
        target_layer_ids,
        rng,
    }, .{ .meshes = &meshes.all() });
    defer verify_exe.deinit();

    log.info("Compiling state commit...", .{});
    const text_cfg = target_config.text_config;
    var num_linear: i64 = 0;
    for (text_cfg.layer_types) |layer_type| {
        if (layer_type == .linear_attention) num_linear += 1;
    }
    const conv_dim: i64 = 2 * @as(i64, @intCast(text_cfg.linear_num_key_heads)) * @as(i64, @intCast(text_cfg.linear_key_head_dim)) +
        @as(i64, @intCast(text_cfg.linear_num_value_heads)) * @as(i64, @intCast(text_cfg.linear_value_head_dim));
    const stacked_shape = zml.Shape.init(.{
        .layer = num_linear,
        .s = @as(i64, block),
        .b = 1,
        .vh = @as(i64, @intCast(text_cfg.linear_num_value_heads)),
        .vhd = @as(i64, @intCast(text_cfg.linear_value_head_dim)),
        .khd = @as(i64, @intCast(text_cfg.linear_key_head_dim)),
    }, .f32);
    const conv_feats_shape = zml.Shape.init(.{
        .b = 1,
        .layer = num_linear,
        .s = @as(i64, block) + @as(i64, @intCast(text_cfg.linear_conv_kernel_dim)) - 1,
        .mix = conv_dim,
    }, cache_dtype);
    var commit_exe = try platform.compileFn(allocator, io, qwen3_5.commitSpeculativeState, .{
        target_kv_cache.gated_delta_net,
        zml.Tensor.fromShape(stacked_shape),
        zml.Tensor.fromShape(conv_feats_shape),
        zml.Tensor.init(.{}, .u32),
    }, .{ .meshes = &meshes.all() });
    defer commit_exe.deinit();

    log.info("Compiling draft prefill...", .{});
    var draft_prefill_exe = try platform.compileFn(allocator, io, draftStep, .{
        draft_model,
        target_model,
        zml.Tensor.init(.{ .s = prefill_len, .d = @as(i64, @intCast(target_layer_ids.len)) * hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = block, .d = hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = prefill_len + block }, .u32),
        draft_kv_cache,
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
    }, .{ .meshes = &meshes.all() });
    defer draft_prefill_exe.deinit();

    log.info("Compiling draft steady...", .{});
    var draft_exe = try platform.compileFn(allocator, io, draftStep, .{
        draft_model,
        target_model,
        zml.Tensor.init(.{ .s = block, .d = @as(i64, @intCast(target_layer_ids.len)) * hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = block, .d = hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = block + block }, .u32),
        draft_kv_cache,
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
    }, .{ .meshes = &meshes.all() });
    defer draft_exe.deinit();

    log.info("Compiling embed...", .{});
    var embed_exe = try platform.compileFn(allocator, io, qwen3_5.EmbedTokens.forward, .{.{
        .embedding = .{ .embed_tokens = target_model.text_model.embed_tokens },
        .tokens = zml.Tensor.init(.{ .s = block }, .u32),
    }}, .{ .meshes = &meshes.all() });
    defer embed_exe.deinit();

    // Tokenizer.
    const tokenizer_file = try repo.openFile(io, "tokenizer.json", .{});
    defer tokenizer_file.close(io);
    var tokenizer_reader = tokenizer_file.reader(io, &.{});
    const tokenizer_bytes = try tokenizer_reader.interface.readAlloc(allocator, try tokenizer_file.length(io));
    defer allocator.free(tokenizer_bytes);
    var tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, tokenizer_bytes);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const im_start = tokenizer.tokenId("<|im_start|>") orelse target_model.special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse target_model.special_tokens.im_end_token_id;
    const newline = tokenizer.tokenId("\\n") orelse 0;
    var chat_tokens: std.ArrayList(u32) = try .initCapacity(allocator, 64);
    defer chat_tokens.deinit(allocator);
    try chat_tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try chat_tokens.appendSlice(allocator, user_tokens);
    const encoded = try encoder.encodeAlloc(allocator, args.prompt);
    defer allocator.free(encoded);
    try chat_tokens.appendSlice(allocator, encoded);
    try chat_tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try chat_tokens.appendSlice(allocator, assistant_tokens);

    const prompt_len: u32 = @intCast(@min(chat_tokens.items.len, prefill_len));

    var prompt_tokens = try allocator.alloc(u32, @intCast(prefill_len));
    defer allocator.free(prompt_tokens);
    @memset(prompt_tokens, 0);
    @memcpy(prompt_tokens[0..prompt_len], chat_tokens.items[0..prompt_len]);

    const mask_token_id: u32 = @intCast(try draft_ctx.getInt("tokenizer.ggml.mask_token_id"));

    // Prefill.
    var prefill_args = try prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    var prompt_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .b = 1, .s = prefill_len }, .u32), std.mem.sliceAsBytes(prompt_tokens));
    defer prompt_buf.deinit();
    var zero_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
    defer zero_index.deinit();
    var active_length_buf = try zml.Buffer.scalar(io, platform, prompt_len, .u32);
    defer active_length_buf.deinit();
    var rng_buf = try rng.initBuffer(io, platform, @intCast(std.Io.Clock.now(.real, io).toNanoseconds()));
    defer zml.Tensor.Rng.deinitBuffer(&rng_buf);

    prefill_args.set(.{ target_buffers, prompt_buf, zero_index, active_length_buf, target_kv_buffers, rng_buf });
    prefill_exe.call(io, prefill_args, &prefill_results, .{ .wait = true });

    var context_hidden, var next_token_buf, var updated_target_kv, var updated_rng = prefill_results.get(struct {
        zml.Buffer,
        zml.Buffer,
        zml.Bufferized(qwen3_5.KvCache),
        zml.Tensor.Rng.Buffer,
    });
    replaceKvBuffers(&target_kv_buffers, &updated_target_kv);
    replaceBuffer(&rng_buf._state, &updated_rng._state);

    var next_token_slice = try next_token_buf.toSliceAlloc(allocator, io);
    defer next_token_slice.free(allocator);
    var last_token = next_token_slice.constItems(u32)[0];
    next_token_buf.deinit();
    log.info("prefill last_token={d} prompt_len={d}", .{ last_token, prompt_len });

    var generated: std.ArrayList(u32) = try .initCapacity(allocator, args.max_seq_len);
    defer generated.deinit(allocator);
    try generated.appendSlice(allocator, chat_tokens.items[0..prompt_len]);
    try generated.append(allocator, last_token);

    var start: u32 = prompt_len;
    var context_len: u32 = prompt_len;
    var draft_cache_index: u32 = 0;
    var owns_context = false;
    var step: usize = 0;
    const decode_start = std.Io.Clock.now(.awake, io);

    var block_tokens = try allocator.alloc(u32, @intCast(block));
    defer allocator.free(block_tokens);
    var pos0_hits: usize = 0;

    while (start < args.max_seq_len and step < 64) : (step += 1) {
        @memset(block_tokens, mask_token_id);
        block_tokens[0] = last_token;

        var block_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = block }, .u32), std.mem.sliceAsBytes(block_tokens));
        defer block_buf.deinit();

        // Embed block tokens.
        var embed_args = try embed_exe.args(allocator);
        defer embed_args.deinit(allocator);
        var embed_results = try embed_exe.results(allocator);
        defer embed_results.deinit(allocator);
        embed_args.set(.{.{ .embedding = target_buffers.text_model.embed_tokens, .tokens = block_buf }});
        embed_exe.call(io, embed_args, &embed_results, .{ .wait = true });
        var noise_buf = embed_results.get(zml.Buffer);
        defer noise_buf.deinit();

        const is_prefill_draft = step == 0;
        const ctx_len: u32 = if (is_prefill_draft) @intCast(prefill_len) else block;
        const pos_len: u32 = ctx_len + block;
        var pos_data = try allocator.alloc(u32, @intCast(pos_len));
        defer allocator.free(pos_data);
        for (pos_data[0..ctx_len], 0..) |*p, i| p.* = draft_cache_index + @as(u32, @intCast(i));
        for (pos_data[ctx_len..], 0..) |*p, i| p.* = draft_cache_index + context_len + @as(u32, @intCast(i));
        var pos_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = pos_len }, .u32), std.mem.sliceAsBytes(pos_data));
        defer pos_buf.deinit();

        var cache_index_buf = try zml.Buffer.scalar(io, platform, draft_cache_index, .u32);
        defer cache_index_buf.deinit();
        var draft_active_len_buf = try zml.Buffer.scalar(io, platform, context_len, .u32);
        defer draft_active_len_buf.deinit();
        var anchor_buf = try zml.Buffer.scalar(io, platform, last_token, .u32);
        defer anchor_buf.deinit();

        var zeroed_context: zml.Buffer = undefined;
        if (args.zero_context) {
            const bytes = try allocator.alloc(u8, context_hidden.shape().byteSize());
            defer allocator.free(bytes);
            @memset(bytes, 0);
            zeroed_context = try zml.Buffer.fromBytes(io, platform, context_hidden.shape(), bytes);
        }
        defer if (args.zero_context) zeroed_context.deinit();

        const draft_exe_ = if (is_prefill_draft) &draft_prefill_exe else &draft_exe;
        var draft_args = try draft_exe_.args(allocator);
        defer draft_args.deinit(allocator);
        var draft_results = try draft_exe_.results(allocator);
        defer draft_results.deinit(allocator);
        draft_args.set(.{
            draft_buffers,
            target_buffers,
            if (args.zero_context) zeroed_context else context_hidden,
            noise_buf,
            pos_buf,
            draft_kv_buffers,
            cache_index_buf,
            draft_active_len_buf,
            anchor_buf,
        });
        draft_exe_.call(io, draft_args, &draft_results, .{ .wait = true });

        var draft_hidden, var lattice_buf, var updated_draft_kv = draft_results.get(struct {
            zml.Buffer,
            zml.Buffer,
            dflash2.KvCache.Buffer,
        });
        defer draft_hidden.deinit();
        defer lattice_buf.deinit();
        dflash2.KvCache.replaceBuffers(&draft_kv_buffers, &updated_draft_kv);

        // Walk the selector lattice.
        var lattice_slice = try lattice_buf.toSliceAlloc(allocator, io);
        defer lattice_slice.free(allocator);
        const lattice = lattice_slice.constItems(f32);
        var predecessor: usize = 0;
        for (1..block) |i| {
            const row = lattice[i * 5120 ..][0..5120];
            const scores = row[16 + predecessor * 16 ..][0..16];
            var best: usize = 0;
            for (1..16) |j| {
                if (scores[j] > scores[best]) best = j;
            }
            predecessor = best;
            block_tokens[i] = @intFromFloat(row[best]);
        }

        var drafted_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .b = 1, .s = block }, .u32), std.mem.sliceAsBytes(block_tokens));
        defer drafted_buf.deinit();

        // Verify.
        var verify_args = try verify_exe.args(allocator);
        defer verify_args.deinit(allocator);
        var verify_results = try verify_exe.results(allocator);
        defer verify_results.deinit(allocator);
        var token_index_buf = try zml.Buffer.scalar(io, platform, start, .u32);
        defer token_index_buf.deinit();
        var verify_active_len_buf = try zml.Buffer.scalar(io, platform, start + block, .u32);
        defer verify_active_len_buf.deinit();
        verify_args.set(.{ target_buffers, drafted_buf, drafted_buf, token_index_buf, verify_active_len_buf, target_kv_buffers, rng_buf });
        verify_exe.call(io, verify_args, &verify_results, .{ .wait = true });

        const block_hidden, var valid_buf, var correction_buf, const kv_after, var stacked_buf, var conv_feats_buf, var rng_after = verify_results.get(struct {
            zml.Buffer,
            zml.Buffer,
            zml.Buffer,
            zml.Bufferized(qwen3_5.KvCache),
            zml.Buffer,
            zml.Buffer,
            zml.Tensor.Rng.Buffer,
        });
        defer stacked_buf.deinit();
        defer conv_feats_buf.deinit();

        var valid_slice = try valid_buf.toSliceAlloc(allocator, io);
        defer valid_slice.free(allocator);
        var correction_slice = try correction_buf.toSliceAlloc(allocator, io);
        defer correction_slice.free(allocator);
        const valid = valid_slice.constItems(u32)[0];
        const correction = correction_slice.constItems(u32)[0];
        valid_buf.deinit();
        correction_buf.deinit();
        const draft_pos0: u32 = @as(u32, @intFromFloat(lattice[0]));
        if (draft_pos0 == correction) pos0_hits += 1;

        // Roll the recurrent/conv state forward to the accepted prefix.
        var valid_scalar_buf = try zml.Buffer.scalar(io, platform, valid, .u32);
        defer valid_scalar_buf.deinit();
        var commit_args = try commit_exe.args(allocator);
        defer commit_args.deinit(allocator);
        var commit_results = try commit_exe.results(allocator);
        defer commit_results.deinit(allocator);
        commit_args.set(.{ kv_after.gated_delta_net, stacked_buf, conv_feats_buf, valid_scalar_buf });
        commit_exe.call(io, commit_args, &commit_results, .{ .wait = true });
        const committed_gdn = commit_results.get(zml.Bufferized(qwen3_5.KvCache.GatedDeltaNetCache));

        var next_kv = kv_after;
        next_kv.gated_delta_net = committed_gdn;
        replaceKvBuffers(&target_kv_buffers, &next_kv);
        replaceBuffer(&rng_buf._state, &rng_after._state);

        const step_start = start;
        for (0..valid) |i| {
            if (start >= args.max_seq_len) break;
            try generated.append(allocator, block_tokens[i + 1]);
            start += 1;
        }
        if (start < args.max_seq_len) {
            try generated.append(allocator, correction);
            start += 1;
        }
        last_token = correction;

        if (owns_context) context_hidden.deinit();
        context_hidden = block_hidden;
        owns_context = true;
        context_len = valid + 1;
        draft_cache_index = step_start;

        log.info("step {d}: valid={d} start={d}", .{ step, valid, start });
    }
    if (owns_context) context_hidden.deinit();

    log.info("draft position-0 accuracy: {d}/{d} = {d:.1}%", .{ pos0_hits, step, 100.0 * @as(f64, @floatFromInt(pos0_hits)) / @as(f64, @floatFromInt(@max(step, 1))) });
    const elapsed = decode_start.untilNow(io, .awake);
    const seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
    const decoded = start - prompt_len;
    log.info("decoded {d} tokens in {d:.2}s ({d:.2} tok/s)", .{ decoded, seconds, @as(f64, @floatFromInt(decoded)) / seconds });

    var out_encoder = try tokenizer.encoder();
    defer out_encoder.deinit();
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();
    const out_buf = try allocator.alloc(u8, 256);
    defer allocator.free(out_buf);
    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    for (generated.items[prompt_len..]) |token| {
        try stdout.interface.writeAll(try decoder.feedOne(token, out_buf));
    }
    try stdout.interface.writeAll(try decoder.finalize(out_buf));
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn replaceKvBuffers(dst: *zml.Bufferized(qwen3_5.KvCache), src: *zml.Bufferized(qwen3_5.KvCache)) void {
    replaceBuffer(&dst.self_attn.k, &src.self_attn.k);
    replaceBuffer(&dst.self_attn.v, &src.self_attn.v);
    replaceBuffer(&dst.self_attn.layer_index, &src.self_attn.layer_index);
    replaceBuffer(&dst.gated_delta_net.conv_state, &src.gated_delta_net.conv_state);
    replaceBuffer(&dst.gated_delta_net.recurrent_state, &src.gated_delta_net.recurrent_state);
    replaceBuffer(&dst.gated_delta_net.layer_index, &src.gated_delta_net.layer_index);
}

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (dst._shards.len != src._shards.len) {
        dst.deinit();
    } else {
        var same = true;
        for (dst._shards.constSlice(), src._shards.constSlice()) |a, b| {
            if (a != b) same = false;
        }
        if (!same) dst.deinit();
    }
    dst.* = src.*;
}

fn draftStep(
    draft_model: dflash2.Model,
    target_model: qwen3_5.Model,
    target_hidden: zml.Tensor,
    noise: zml.Tensor,
    position_ids: zml.Tensor,
    draft_kv_cache: dflash2.KvCache,
    cache_index: zml.Tensor,
    active_context_len: zml.Tensor,
    anchor: zml.Tensor,
) struct { zml.Tensor, zml.Tensor, dflash2.KvCache } {
    const hidden, const updated_kv_cache = draft_model.forward(
        target_hidden,
        noise,
        position_ids,
        draft_kv_cache,
        cache_index,
        active_context_len,
    );
    const logits = target_model.sampler().logitsAll(hidden);
    const lattice = dflash2.selectorLattice(
        draft_model.selector_hidden,
        draft_model.selector_predecessor,
        draft_model.selector_successor,
        logits,
        hidden,
        anchor,
        5120,
        16,
        256,
    );
    return .{ hidden, lattice, updated_kv_cache };
}
