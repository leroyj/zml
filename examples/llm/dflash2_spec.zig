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
    prompt_file: []const u8 = "",
    max_new_tokens: u32 = 256,
    warmup_tokens: u32 = 32,
    prefill_len: u32 = 0,
    draft_tokens: u32 = 0,
    zero_context: bool = false,

    pub const help =
        \\Options:
        \\  --model=<path>          Path to the Qwen3.8 target GGUF
        \\  --draft=<path>          Path to the DFlash2 draft GGUF
        \\  --prompt=<text>         Prompt
        \\  --prompt-file=<path>    Run every prompt of the file in one process; one prompt
        \\                          per line, optionally `name<TAB>prompt`, `#` comments
        \\  --warmup-tokens=<n>     Untimed warm-up generation before the measured prompts (default 32)
        \\  --max-new-tokens=<n>    Maximum generated tokens (default 256)
        \\  --prefill-len=<n>       Prefill bucket; 0 rounds the prompt up to a multiple of 64
        \\  --draft-tokens=<n>      Tokens proposed per block; 0 uses the draft's block_size - 1
        \\  --zero-context          Blank the target features fed to the drafter (diagnostic)
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

    var prompts: std.ArrayList(Prompt) = .empty;
    defer {
        for (prompts.items) |prompt| allocator.free(prompt.tokens);
        prompts.deinit(allocator);
    }
    var prompt_file_bytes: []u8 = &.{};
    defer allocator.free(prompt_file_bytes);
    if (args.prompt_file.len > 0) {
        prompt_file_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args.prompt_file, allocator, .limited(16 * zml.MiB));
        var lines = std.mem.splitScalar(u8, prompt_file_bytes, '\n');
        while (lines.next()) |raw_line| {
            const line = std.mem.trim(u8, raw_line, " \r");
            if (line.len == 0 or line[0] == '#') continue;
            const tab = std.mem.indexOfScalar(u8, line, '\t');
            const name = if (tab) |t| line[0..t] else "";
            const text = if (tab) |t| line[t + 1 ..] else line;
            try prompts.append(allocator, .{ .name = name, .tokens = try chatTokens(allocator, &encoder, text, im_start, im_end, newline) });
        }
    } else {
        try prompts.append(allocator, .{ .name = "prompt", .tokens = try chatTokens(allocator, &encoder, args.prompt, im_start, im_end, newline) });
    }
    if (prompts.items.len == 0) return error.NoPrompts;

    var longest_prompt: u32 = 0;
    for (prompts.items) |prompt| longest_prompt = @max(longest_prompt, @as(u32, @intCast(prompt.tokens.len)));
    const prefill_len: u32 = if (args.prefill_len != 0) args.prefill_len else std.mem.alignForward(u32, longest_prompt, 64);
    if (longest_prompt > prefill_len) {
        log.err("longest prompt has {d} tokens but --prefill-len={d}", .{ longest_prompt, prefill_len });
        return error.PromptTooLong;
    }

    const max_block = draft_config.block_size;
    const draft_tokens: u32 = if (args.draft_tokens != 0) @min(args.draft_tokens, max_block - 1) else max_block - 1;
    const block: u32 = draft_tokens + 1;
    // The last verify writes a full block past the final committed position.
    const seqlen: i64 = std.mem.alignForward(i64, @as(i64, prefill_len) + args.max_new_tokens + 2 * block, 256);
    log.info("prompts={d} longest_prompt={d} prefill_len={d} draft_tokens={d} kv_len={d}", .{ prompts.items.len, longest_prompt, prefill_len, draft_tokens, seqlen });
    const cache_dtype = target_model.text_model.embed_tokens.weight.dtype();
    const target_kv_cache = qwen3_5.KvCache.init(target_config, 1, seqlen, cache_dtype, .f32, meshes.model);

    const draft_kv_cache = draft_model.initKvCache(draft_config, @intCast(seqlen), meshes.model);

    const hidden = target_config.text_config.hidden_size;
    const rng = zml.Tensor.Rng.init();

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
    const selector_dims: SelectorDims = .{
        .lattice_width = draft_config.hidden_size,
        .top_k = draft_config.selector_top_k,
        .rank = draft_config.selector_rank,
    };
    var draft_prefill_exe = try platform.compileFn(allocator, io, draftStep, .{
        selector_dims,
        draft_model,
        target_model,
        zml.Tensor.init(.{ .s = prefill_len, .d = @as(i64, @intCast(target_layer_ids.len)) * hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = block, .d = hidden }, cache_dtype),
        zml.Tensor.init(.{ .s = @as(i64, prefill_len) + block }, .u32),
        draft_kv_cache,
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
        zml.Tensor.init(.{}, .u32),
    }, .{ .meshes = &meshes.all() });
    defer draft_prefill_exe.deinit();

    log.info("Compiling draft steady...", .{});
    var draft_exe = try platform.compileFn(allocator, io, draftStep, .{
        selector_dims,
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

    const mask_token_id: u32 = @intCast(try draft_ctx.getInt("tokenizer.ggml.mask_token_id"));

    var total: Stats = .{ .accepted_hist = try allocator.alloc(usize, block) };
    defer allocator.free(total.accepted_hist);
    @memset(total.accepted_hist, 0);

    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(allocator);
    if (args.warmup_tokens > 0) try runs.append(allocator, .{ .prompt = prompts.items[0], .max_new_tokens = args.warmup_tokens, .warmup = true });
    for (prompts.items) |prompt| try runs.append(allocator, .{ .prompt = prompt, .max_new_tokens = args.max_new_tokens, .warmup = false });

    // The args/results objects own the buffers that `get` hands back, so they
    // must outlive every prompt: the KV caches adopted from them are reused.
    var embed_args = try embed_exe.args(allocator);
    defer embed_args.deinit(allocator);
    var embed_results = try embed_exe.results(allocator);
    defer embed_results.deinit(allocator);
    var draft_prefill_args = try draft_prefill_exe.args(allocator);
    defer draft_prefill_args.deinit(allocator);
    var draft_prefill_results = try draft_prefill_exe.results(allocator);
    defer draft_prefill_results.deinit(allocator);
    var draft_args_ = try draft_exe.args(allocator);
    defer draft_args_.deinit(allocator);
    var draft_results_ = try draft_exe.results(allocator);
    defer draft_results_.deinit(allocator);
    var verify_args = try verify_exe.args(allocator);
    defer verify_args.deinit(allocator);
    var verify_results = try verify_exe.results(allocator);
    defer verify_results.deinit(allocator);
    var commit_args = try commit_exe.args(allocator);
    defer commit_args.deinit(allocator);
    var commit_results = try commit_exe.results(allocator);
    defer commit_results.deinit(allocator);

    for (runs.items) |run| {
        // Prefill.
        const chat = run.prompt.tokens;
        const prompt_len: u32 = @intCast(chat.len);
        const max_len: u32 = prompt_len + run.max_new_tokens;
        var prompt_tokens = try allocator.alloc(u32, prefill_len);
        defer allocator.free(prompt_tokens);
        @memset(prompt_tokens, 0);
        @memcpy(prompt_tokens[0..prompt_len], chat);

        // Fresh caches per prompt: nothing carries over between prompts.
        var target_kv_buffers = try target_kv_cache.initBuffer(io, platform);
        defer qwen3_5.KvCache.deinitBuffer(&target_kv_buffers);
        var draft_kv_buffers = try draft_kv_cache.initZeroBuffer(allocator, io, platform);
        defer dflash2.KvCache.deinitBuffer(&draft_kv_buffers);

        var prompt_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .b = 1, .s = prefill_len }, .u32), std.mem.sliceAsBytes(prompt_tokens));
        defer prompt_buf.deinit();
        var zero_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
        defer zero_index.deinit();
        var active_length_buf = try zml.Buffer.scalar(io, platform, prompt_len, .u32);
        defer active_length_buf.deinit();
        var rng_buf = try rng.initBuffer(io, platform, @intCast(std.Io.Clock.now(.real, io).toNanoseconds()));
        defer zml.Tensor.Rng.deinitBuffer(&rng_buf);

        // A fresh executable per prompt: the prefill returns the KV caches as
        // `reuseBuffer` aliases of its inputs, and reusing one compiled
        // executable across prompts makes PJRT's donation bookkeeping hand back
        // a released handle on the second call. Compilation is cached, so this
        // costs a lookup rather than a rebuild.
        var prefill_exe_run = try platform.compileFn(allocator, io, qwen3_5.Model.prefillForward, .{
            target_model,
            zml.Tensor.init(.{ .b = 1, .s = prefill_len }, .u32),
            zml.Tensor.init(.{}, .u32),
            zml.Tensor.init(.{}, .u32),
            target_kv_cache,
            target_layer_ids,
            rng,
        }, .{ .meshes = &meshes.all() });
        defer prefill_exe_run.deinit();
        var prefill_results = try prefill_exe_run.results(allocator);
        defer prefill_results.deinit(allocator);
        var prefill_args = try prefill_exe_run.args(allocator);
        defer prefill_args.deinit(allocator);

        prefill_args.set(.{ target_buffers, prompt_buf, zero_index, active_length_buf, target_kv_buffers, rng_buf });
        prefill_exe_run.call(io, prefill_args, &prefill_results, .{ .wait = true });

        var context_hidden, var next_token_buf, var updated_target_kv, var updated_rng = prefill_results.get(struct {
            zml.Buffer,
            zml.Buffer,
            zml.Bufferized(qwen3_5.KvCache),
            zml.Tensor.Rng.Buffer,
        });
        releaseResults(&prefill_results);
        replaceKvBuffers(&target_kv_buffers, &updated_target_kv);
        replaceBuffer(&rng_buf._state, &updated_rng._state);

        var next_token_slice = try next_token_buf.toSliceAlloc(allocator, io);
        defer next_token_slice.free(allocator);
        var last_token = next_token_slice.constItems(u32)[0];
        next_token_buf.deinit();
        log.info("prefill last_token={d} prompt_len={d}", .{ last_token, prompt_len });

        var generated: std.ArrayList(u32) = try .initCapacity(allocator, max_len + block);
        defer generated.deinit(allocator);
        try generated.appendSlice(allocator, chat);
        try generated.append(allocator, last_token);

        const eos_token: u32 = if (target_ctx.getInt("tokenizer.ggml.eos_token_id")) |id| @intCast(id) else |_| im_end;
        const stop_tokens = [_]u32{ eos_token, im_end, target_model.special_tokens.end_of_text_token_id };
        var finished = std.mem.indexOfScalar(u32, &stop_tokens, last_token) != null;

        var start: u32 = prompt_len;
        var context_len: u32 = prompt_len;
        var draft_cache_index: u32 = 0;
        var owns_context = false;
        var step: usize = 0;
        const decode_start = std.Io.Clock.now(.awake, io);

        const top_k: usize = draft_config.selector_top_k;
        const lattice_width: usize = draft_config.hidden_size;
        var block_tokens = try allocator.alloc(u32, block);
        defer allocator.free(block_tokens);
        var stats: Stats = .{ .accepted_hist = try allocator.alloc(usize, block) };
        defer allocator.free(stats.accepted_hist);
        @memset(stats.accepted_hist, 0);

        while (!finished and start < max_len) : (step += 1) {
            @memset(block_tokens, mask_token_id);
            block_tokens[0] = last_token;

            var block_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .s = block }, .u32), std.mem.sliceAsBytes(block_tokens));
            defer block_buf.deinit();

            var phase_start = std.Io.Clock.now(.awake, io);
            // Embed block tokens.
            embed_args.set(.{.{ .embedding = target_buffers.text_model.embed_tokens, .tokens = block_buf }});
            embed_exe.call(io, embed_args, &embed_results, .{ .wait = true });
            var noise_buf = embed_results.get(zml.Buffer);
            releaseResults(&embed_results);
            defer noise_buf.deinit();

            stats.embed_ns += lap(io, &phase_start);
            const is_prefill_draft = step == 0;
            const ctx_len: u32 = if (is_prefill_draft) prefill_len else block;
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
            const draft_args = if (is_prefill_draft) &draft_prefill_args else &draft_args_;
            const draft_results = if (is_prefill_draft) &draft_prefill_results else &draft_results_;
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
            draft_exe_.call(io, draft_args.*, draft_results, .{ .wait = true });

            var draft_hidden, var lattice_buf, var updated_draft_kv = draft_results.get(struct {
                zml.Buffer,
                zml.Buffer,
                dflash2.KvCache.Buffer,
            });
            releaseResults(draft_results);
            defer draft_hidden.deinit();
            defer lattice_buf.deinit();
            dflash2.KvCache.replaceBuffers(&draft_kv_buffers, &updated_draft_kv);

            // Walk the selector lattice.
            var lattice_slice = try lattice_buf.toSliceAlloc(allocator, io);
            defer lattice_slice.free(allocator);
            const lattice = lattice_slice.constItems(f32);
            var predecessor: usize = 0;
            for (1..block) |i| {
                const row = lattice[i * lattice_width ..][0..lattice_width];
                const scores = row[top_k + predecessor * top_k ..][0..top_k];
                var best: usize = 0;
                for (1..top_k) |j| {
                    if (scores[j] > scores[best]) best = j;
                }
                predecessor = best;
                block_tokens[i] = @intFromFloat(row[best]);
            }

            stats.draft_ns += lap(io, &phase_start);
            var drafted_buf = try zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{ .b = 1, .s = block }, .u32), std.mem.sliceAsBytes(block_tokens));
            defer drafted_buf.deinit();

            // Verify.
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
            releaseResults(&verify_results);
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
            stats.verify_ns += lap(io, &phase_start);
            stats.accepted_hist[valid] += 1;

            // Roll the recurrent/conv state forward to the accepted prefix.
            var valid_scalar_buf = try zml.Buffer.scalar(io, platform, valid, .u32);
            defer valid_scalar_buf.deinit();
            commit_args.set(.{ kv_after.gated_delta_net, stacked_buf, conv_feats_buf, valid_scalar_buf });
            commit_exe.call(io, commit_args, &commit_results, .{ .wait = true });
            const committed_gdn = commit_results.get(zml.Bufferized(qwen3_5.KvCache.GatedDeltaNetCache));
            releaseResults(&commit_results);

            var next_kv = kv_after;
            next_kv.gated_delta_net = committed_gdn;
            replaceKvBuffers(&target_kv_buffers, &next_kv);
            replaceBuffer(&rng_buf._state, &rng_after._state);
            stats.commit_ns += lap(io, &phase_start);

            const step_start = start;
            for (0..valid) |i| {
                if (start >= max_len) break;
                const token = block_tokens[i + 1];
                try generated.append(allocator, token);
                start += 1;
                if (std.mem.indexOfScalar(u32, &stop_tokens, token) != null) {
                    finished = true;
                    break;
                }
            }
            if (!finished and start < max_len) {
                try generated.append(allocator, correction);
                start += 1;
                if (std.mem.indexOfScalar(u32, &stop_tokens, correction) != null) finished = true;
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

        const elapsed = decode_start.untilNow(io, .awake);
        const seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
        const decoded = start - prompt_len;
        if (run.warmup) {
            log.info("warm-up: decoded {d} tokens in {d:.2}s (not counted)", .{ decoded, seconds });
            continue;
        }
        log.info("=== [{s}] prompt_len={d} ===", .{ run.prompt.name, prompt_len });
        stats.report(step, decoded, seconds);
        log.info("decoded {d} tokens in {d:.2}s ({d:.2} tok/s)", .{ decoded, seconds, @as(f64, @floatFromInt(decoded)) / seconds });
        total.add(stats, step, decoded, seconds);

        var out_encoder = try tokenizer.encoder();
        defer out_encoder.deinit();
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        const out_buf = try allocator.alloc(u8, 256);
        defer allocator.free(out_buf);
        var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
        try stdout.interface.print("=== [{s}] ===\n", .{run.prompt.name});
        for (generated.items[prompt_len..]) |token| {
            if (std.mem.indexOfScalar(u32, &stop_tokens, token) != null) break;
            try stdout.interface.writeAll(try decoder.feedOne(token, out_buf));
        }
        try stdout.interface.writeAll(try decoder.finalize(out_buf));
        try stdout.interface.writeByte('\n');
        try stdout.interface.flush();
    }

    if (prompts.items.len > 1) {
        log.info("=== aggregate over {d} prompts ===", .{prompts.items.len});
        total.report(total.steps, total.decoded, total.seconds);
        log.info("TOTAL decoded {d} tokens in {d:.2}s ({d:.2} tok/s)", .{ total.decoded, total.seconds, @as(f64, @floatFromInt(total.decoded)) / total.seconds });
    }
}

const Stats = struct {
    accepted_hist: []usize,
    embed_ns: u64 = 0,
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    commit_ns: u64 = 0,
    steps: usize = 0,
    decoded: u32 = 0,
    seconds: f64 = 0,

    fn add(self: *Stats, other: Stats, steps: usize, decoded: u32, seconds: f64) void {
        for (self.accepted_hist, other.accepted_hist) |*a, b| a.* += b;
        self.embed_ns += other.embed_ns;
        self.draft_ns += other.draft_ns;
        self.verify_ns += other.verify_ns;
        self.commit_ns += other.commit_ns;
        self.steps += steps;
        self.decoded += decoded;
        self.seconds += seconds;
    }

    fn report(self: Stats, steps: usize, decoded: u32, seconds: f64) void {
        var accepted: usize = 0;
        for (self.accepted_hist, 0..) |count, n| accepted += count * n;
        const steps_f: f64 = @floatFromInt(@max(steps, 1));
        const first_hits = steps - self.accepted_hist[0];
        log.info("acceptance: {d} steps, {d:.2} drafts accepted/step, {d:.2} tokens/step, first-draft hit {d:.1}%, histogram {any}", .{
            steps,
            @as(f64, @floatFromInt(accepted)) / steps_f,
            @as(f64, @floatFromInt(decoded)) / steps_f,
            100.0 * @as(f64, @floatFromInt(first_hits)) / steps_f,
            self.accepted_hist,
        });
        const ms = struct {
            fn f(ns: u64, n: f64) f64 {
                return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms / n;
            }
        }.f;
        const total_ms = seconds * std.time.ms_per_s / steps_f;
        log.info("per step: embed {d:.1} ms, draft {d:.1} ms, verify {d:.1} ms, commit {d:.1} ms, total {d:.1} ms", .{
            ms(self.embed_ns, steps_f), ms(self.draft_ns, steps_f), ms(self.verify_ns, steps_f), ms(self.commit_ns, steps_f), total_ms,
        });
    }
};

fn lap(io: std.Io, start: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.now(.awake, io);
    const ns: u64 = @intCast(start.durationTo(now).nanoseconds);
    start.* = now;
    return ns;
}

const Prompt = struct { name: []const u8, tokens: []u32 };
const Run = struct { prompt: Prompt, max_new_tokens: u32, warmup: bool };

fn chatTokens(allocator: std.mem.Allocator, encoder: anytype, prompt: []const u8, im_start: u32, im_end: u32, newline: u32) ![]u32 {
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(allocator);
    try tokens.append(allocator, im_start);
    const user_tokens = try encoder.encodeAlloc(allocator, "user\n");
    defer allocator.free(user_tokens);
    try tokens.appendSlice(allocator, user_tokens);
    const encoded = try encoder.encodeAlloc(allocator, prompt);
    defer allocator.free(encoded);
    try tokens.appendSlice(allocator, encoded);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start });
    const assistant_tokens = try encoder.encodeAlloc(allocator, "assistant\n");
    defer allocator.free(assistant_tokens);
    try tokens.appendSlice(allocator, assistant_tokens);
    return tokens.toOwnedSlice(allocator);
}

/// `Results.get` hands the buffer handles to the caller but leaves them in the
/// result slots. PJRT treats a non-null slot as a preallocated destination for
/// outputs that alias their inputs (`reuseBuffer`, as the KV caches do) and
/// would then copy into an already-freed handle. Clear the slots after every
/// `get` so the next execution allocates fresh outputs.
fn releaseResults(results: anytype) void {
    const raw: [*]usize = @ptrCast(@constCast(results.flat_buffers.raw_buffers.ptr));
    @memset(raw[0..results.flat_buffers.raw_buffers.len], 0);
}

const SelectorDims = struct { lattice_width: i64, top_k: i64, rank: i64 };

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
    selector: SelectorDims,
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
        selector.lattice_width,
        selector.top_k,
        selector.rank,
    );
    return .{ hidden, lattice, updated_kv_cache };
}
