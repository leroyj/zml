const std = @import("std");

const zml = @import("zml");

const gguf = @import("gguf.zig");
const dflash2 = @import("dflash2_model.zig");
const common = @import("common.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};
const log = std.log.scoped(.dflash2);

const Args = struct {
    model: []const u8,

    pub const help =
        \\Options:
        \\  --model=<path>   Path to the DFlash2 GGUF draft model
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const shardings = try common.Shardings.init(platform);

    var ctx = try gguf.Context.init(allocator, io, args.model);
    defer ctx.deinit();

    var registry = try ctx.registry(allocator);
    defer registry.deinit();

    var store = zml.io.TensorStore.fromRegistry(allocator, &registry);
    defer store.deinit();

    const vocab_size: u32 = @intCast((try ctx.getArray("tokenizer.ggml.tokens")).len);
    const target_layer_ids_i64 = try ctx.getIntArray("dflash.target_layers");
    const target_layer_ids = try allocator.alloc(u32, target_layer_ids_i64.len);
    defer allocator.free(target_layer_ids);
    for (target_layer_ids_i64, 0..) |id, i| target_layer_ids[i] = @intCast(id);

    const config: dflash2.Config = .{
        .vocab_size = vocab_size,
        .hidden_size = @intCast(try ctx.getInt("dflash.embedding_length")),
        .intermediate_size = @intCast(try ctx.getInt("dflash.feed_forward_length")),
        .num_hidden_layers = @intCast(try ctx.getInt("dflash.block_count")),
        .num_attention_heads = @intCast(try ctx.getInt("dflash.attention.head_count")),
        .num_key_value_heads = @intCast(try ctx.getInt("dflash.attention.head_count_kv")),
        .head_dim = @intCast(try ctx.getInt("dflash.attention.key_length")),
        .rms_norm_eps = @floatCast(try ctx.getFloat("dflash.attention.layer_norm_rms_epsilon")),
        .rope_theta = @floatCast(try ctx.getFloat("dflash.rope.freq_base")),
        .block_size = @intCast(try ctx.getInt("dflash.block_size")),
        .conv_kernel_size = @intCast(try ctx.getInt("dflash.conv_kernel_size")),
        .conv_group_size = @intCast(try ctx.getInt("dflash.conv_group_size")),
        .selector_rank = @intCast(try ctx.getInt("dflash.selector_rank")),
        .selector_top_k = @intCast(try ctx.getInt("dflash.selector_top_k")),
        .dflash_config = .{ .target_layer_ids = target_layer_ids },
    };
    log.info("DFlash2 config: {}", .{config});

    var model = try dflash2.Model.init(allocator, store.view(), config);
    defer model.deinit(allocator);

    log.info("Loading draft weights...", .{});
    var buffers = try model.loadBuffers(allocator, io, platform, &store, &.{shardings.model});
    defer dflash2.Model.unloadBuffers(&buffers, allocator);

    const block = config.block_size;
    const hidden = config.hidden_size;
    const target_hidden = zml.Tensor.init(.{ .s = block, .d = target_layer_ids.len * hidden }, .bf16);
    const noise = zml.Tensor.init(.{ .s = block, .d = hidden }, .bf16);
    const pos = zml.Tensor.init(.{ .s = block * 2 }, .u32);
    const kv_cache = model.initKvCache(config, block * 2);
    const cache_index = zml.Tensor.init(.{}, .u32);
    const active_len = zml.Tensor.init(.{}, .u32);

    log.info("Compiling draft forward...", .{});
    var exe = try platform.compileFn(allocator, io, forwardTest, .{
        model,
        target_hidden,
        noise,
        pos,
        kv_cache,
        cache_index,
        active_len,
    }, .{ .shardings = &.{shardings.model} });
    defer exe.deinit();

    var args_exe = try exe.args(allocator);
    defer args_exe.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    const target_hidden_data = try allocator.alloc(u16, @intCast(block * target_layer_ids.len * hidden));
    defer allocator.free(target_hidden_data);
    @memset(target_hidden_data, 0);
    const noise_data = try allocator.alloc(u16, @intCast(block * hidden));
    defer allocator.free(noise_data);
    @memset(noise_data, 0);
    const pos_data = try allocator.alloc(u32, @intCast(block * 2));
    defer allocator.free(pos_data);
    for (pos_data, 0..) |*p, i| p.* = @intCast(i);

    var target_hidden_buf = try zml.Buffer.fromBytes(io, platform, target_hidden.shape(), .replicated, std.mem.sliceAsBytes(target_hidden_data));
    defer target_hidden_buf.deinit();
    var noise_buf = try zml.Buffer.fromBytes(io, platform, noise.shape(), .replicated, std.mem.sliceAsBytes(noise_data));
    defer noise_buf.deinit();
    var pos_buf = try zml.Buffer.fromBytes(io, platform, pos.shape(), .replicated, std.mem.sliceAsBytes(pos_data));
    defer pos_buf.deinit();
    var kv_buf = try kv_cache.initZeroBuffer(allocator, io, platform, shardings.model);
    defer dflash2.KvCache.deinitBuffer(&kv_buf);
    var cache_index_buf = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
    defer cache_index_buf.deinit();
    var active_len_buf = try zml.Buffer.scalar(io, platform, @as(u32, block), .u32);
    defer active_len_buf.deinit();

    args_exe.set(.{
        buffers,
        target_hidden_buf,
        noise_buf,
        pos_buf,
        kv_buf,
        cache_index_buf,
        active_len_buf,
    });
    exe.call(args_exe, &results);

    var out_hidden, var out_kv = results.get(struct { zml.Buffer, dflash2.KvCache.Buffer });
    defer out_hidden.deinit();
    defer dflash2.KvCache.deinitBuffer(&out_kv);

    var slice = try out_hidden.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    const values = slice.constItems(u16);
    log.info("forward ok: hidden shape {f}, first values {any}", .{ out_hidden.shape(), values[0..8] });

    // Selector lattice smoke test with zero logits.
    const logits_shape = zml.Shape.init(.{ .s = block, .voc = vocab_size }, .bf16);
    const logits = zml.Tensor.fromShape(logits_shape);
    const hidden_tensor = zml.Tensor.fromShape(out_hidden.shape());
    const anchor = zml.Tensor.init(.{}, .u32);
    var selector_exe = try platform.compileFn(allocator, io, selectorTest, .{
        model,
        logits,
        hidden_tensor,
        anchor,
    }, .{ .shardings = &.{shardings.model} });
    defer selector_exe.deinit();

    var selector_args = try selector_exe.args(allocator);
    defer selector_args.deinit(allocator);
    var selector_results = try selector_exe.results(allocator);
    defer selector_results.deinit(allocator);

    const logits_data = try allocator.alloc(u16, @intCast(block * vocab_size));
    defer allocator.free(logits_data);
    @memset(logits_data, 0);
    var logits_buf = try zml.Buffer.fromBytes(io, platform, logits_shape, .replicated, std.mem.sliceAsBytes(logits_data));
    defer logits_buf.deinit();
    var anchor_buf = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
    defer anchor_buf.deinit();

    selector_args.set(.{ buffers, logits_buf, out_hidden, anchor_buf });
    selector_exe.call(selector_args, &selector_results);

    var lattice = selector_results.get(zml.Buffer);
    defer lattice.deinit();
    var lattice_slice = try lattice.toSliceAlloc(allocator, io);
    defer lattice_slice.free(allocator);
    const lattice_values = lattice_slice.constItems(u16);
    log.info("selector ok: lattice shape {f}, first values {any}", .{ lattice.shape(), lattice_values[0..8] });
}

fn selectorTest(
    model: dflash2.Model,
    logits: zml.Tensor,
    hidden: zml.Tensor,
    anchor: zml.Tensor,
) zml.Tensor {
    return dflash2.selectorLattice(
        model.selector_hidden,
        model.selector_predecessor,
        model.selector_successor,
        logits,
        hidden,
        anchor,
        5120,
        16,
        256,
    );
}

fn forwardTest(
    model: dflash2.Model,
    target_hidden: zml.Tensor,
    noise: zml.Tensor,
    position_ids: zml.Tensor,
    kv_cache: dflash2.KvCache,
    cache_index: zml.Tensor,
    active_context_len: zml.Tensor,
) struct { zml.Tensor, dflash2.KvCache } {
    const hidden, const updated_kv_cache = model.forward(
        target_hidden,
        noise,
        position_ids,
        kv_cache,
        cache_index,
        active_context_len,
    );
    return .{ hidden, updated_kv_cache };
}
