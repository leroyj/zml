const std = @import("std");
const zml = @import("zml");
const q8_triton = @import("q8_triton.zig");

const K: i64 = 5120;
const N: i64 = 17408;
const BLOCK: i64 = 32;
const NB: i64 = K / BLOCK;
const ITERS: usize = 2;

fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x8000) >> 16);
}

fn q8Triton(x: zml.Tensor, q: zml.Tensor, s: zml.Tensor) zml.Tensor {
    const cfg: q8_triton.Config = .{
        .m = 16,
        .n = N,
        .k = K,
        .block_m = 16,
        .block_n = 64,
        .block_k = 256,
        .stride_xm = K,
        .stride_xk = 1,
        .stride_qn = K,
        .stride_qk = 1,
        .stride_sn = NB,
        .stride_sk = 1,
        .stride_om = N,
        .stride_on = 1,
    };
    return q8_triton.Q8Matvec.call(
        .{ .x_ptr = x, .q_ptr = q, .s_ptr = s },
        .{ .out = zml.Shape.init(.{ .m = 16, .n = N }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ 1, @intCast(@divExact(N, 64)), 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn q8MatvecLocal(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor) zml.Tensor {
    const m = x.dim(.m);
    const k = x.dim(.k);
    const n = quants.dim(.n);
    const block_n: i64 = if (n >= 64) 64 else 16;
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

fn q8Linear(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor) zml.Tensor {
    const x_shape = x.shape();
    const d = x.dim(.k);
    const m: i64 = @divExact(@as(i64, @intCast(x_shape.count())), d);
    const x_flat = x.reshape(zml.Shape.init(.{ .m = m, .k = d }, x.dtype()));
    const m_padded = std.mem.alignForward(i64, m, 16);
    const x_padded = if (m_padded == m) x_flat else blk: {
        const zeros = zml.Tensor.scalar(0, x.dtype()).broad(zml.Shape.init(.{ .m = m_padded - m, .k = d }, x.dtype()));
        break :blk zml.Tensor.concatenate(&.{ x_flat, zeros }, .m);
    };
    const out_padded = q8MatvecLocal(x_padded, quants, scales);
    const out_flat = if (m_padded == m) out_padded else out_padded.slice(.m, .{ .start = 0, .end = m });
    const out_shape = x_shape.setDim(.k, quants.dim(.n));
    return out_flat.reshape(out_shape).renameTag(.k, .n);
}

fn q8DotFlat(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor) zml.Tensor {
    const q3 = quants.reshape(zml.Shape.init(.{ .n = N, .nb = NB, .block = BLOCK }, .i8));
    const s3 = scales.reshape(zml.Shape.init(.{ .n = N, .nb = NB, .one = 1 }, .f16));
    const w3 = q3.convert(.f32).mul(s3.convert(.f32).broad(q3.shape()));
    const w = w3.reshape(zml.Shape.init(.{ .n = N, .k = K }, .f32)).convert(.bf16);
    return x.dot(w, .k);
}

const Functions = struct {
    fn q8Dot(x: zml.Tensor, quants: zml.Tensor, scales: zml.Tensor) zml.Tensor {
        const q = quants.convert(.f32);
        const s = scales.convert(.f32);
        const w = q.mul(s.broad(q.shape()))
            .reshape(zml.Shape.init(.{ .n = N, .k = K }, .f32))
            .convert(.bf16);
        return x.dot(w, .k);
    }

    fn bf16Dot(x: zml.Tensor, w: zml.Tensor) zml.Tensor {
        return x.dot(w, .k);
    }
};

fn bench(comptime label: []const u8, exe: *zml.Exe, args: anytype, results: anytype, io: std.Io) !void {
    for (0..10) |_| {
        exe.call(io, args.*, results, .{ .wait = true });
        var out = results.get(zml.Buffer);
        out.deinit();
    }

    const start = std.Io.Clock.now(.awake, io);
    for (0..ITERS) |_| {
        exe.call(io, args.*, results, .{ .wait = true });
        var out = results.get(zml.Buffer);
        out.deinit();
    }
    const elapsed = start.untilNow(io, .awake);
    const seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
    const flops = 2.0 * @as(f64, @floatFromInt(N * K)) * @as(f64, @floatFromInt(ITERS));
    std.debug.print("{s}: {d:.3} ms/iter, {d:.1} GFLOP/s\n", .{
        label,
        seconds * 1000.0 / @as(f64, @floatFromInt(ITERS)),
        flops / seconds / 1e9,
    });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const x_shape = zml.Shape.init(.{ .m = 1, .k = K }, .bf16);
    const q_shape = zml.Shape.init(.{ .n = N, .nb = NB, .block = BLOCK }, .i8);
    const s_shape = zml.Shape.init(.{ .n = N, .nb = NB, .one = 1 }, .f16);
    const w_shape = zml.Shape.init(.{ .n = N, .k = K }, .bf16);

    var exe_q8 = try platform.compileFn(allocator, io, Functions.q8Dot, .{
        zml.Tensor.fromShape(x_shape),
        zml.Tensor.fromShape(q_shape),
        zml.Tensor.fromShape(s_shape),
    }, .{});
    defer exe_q8.deinit();

    var exe_bf = try platform.compileFn(allocator, io, Functions.bf16Dot, .{
        zml.Tensor.fromShape(x_shape),
        zml.Tensor.fromShape(w_shape),
    }, .{});
    defer exe_bf.deinit();

    const x_data = try allocator.alloc(u16, @intCast(K));
    defer allocator.free(x_data);
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    for (x_data) |*v| v.* = bf16Bits(rand.float(f32) * 0.02 - 0.01);

    const q_data = try allocator.alloc(i8, @intCast(N * NB * BLOCK));
    defer allocator.free(q_data);
    for (q_data, 0..) |*v, i| v.* = @intCast(@as(i32, @intCast(i % 255)) - 127);

    const s_data = try allocator.alloc(f16, @intCast(N * NB));
    defer allocator.free(s_data);
    @memset(s_data, 0.01);

    const w_data = try allocator.alloc(u16, @intCast(N * K));
    defer allocator.free(w_data);
    @memset(w_data, bf16Bits(0.01));

    var x_buf = try zml.Buffer.fromBytes(io, platform, x_shape, std.mem.sliceAsBytes(x_data));
    defer x_buf.deinit();
    var q_buf = try zml.Buffer.fromBytes(io, platform, q_shape, std.mem.sliceAsBytes(q_data));
    defer q_buf.deinit();
    var s_buf = try zml.Buffer.fromBytes(io, platform, s_shape, std.mem.sliceAsBytes(s_data));
    defer s_buf.deinit();
    var w_buf = try zml.Buffer.fromBytes(io, platform, w_shape, std.mem.sliceAsBytes(w_data));
    defer w_buf.deinit();

    var args_q8 = try exe_q8.args(allocator);
    defer args_q8.deinit(allocator);
    var results_q8 = try exe_q8.results(allocator);
    defer results_q8.deinit(allocator);
    args_q8.set(.{ x_buf, q_buf, s_buf });

    var args_bf = try exe_bf.args(allocator);
    defer args_bf.deinit(allocator);
    var results_bf = try exe_bf.results(allocator);
    defer results_bf.deinit(allocator);
    args_bf.set(.{ x_buf, w_buf });

    try bench("q8_0_graph", &exe_q8, &args_q8, &results_q8, io);
    try bench("bf16", &exe_bf, &args_bf, &results_bf, io);

    // Correctness: graph dequant vs fused Triton kernel.
    exe_q8.call(io, args_q8, &results_q8, .{ .wait = true });
    var out_graph_buf = results_q8.get(zml.Buffer);
    defer out_graph_buf.deinit();
    const graph_shape = zml.Shape.init(.{ .m = 1, .n = N }, .bf16);
    var graph_slice = try zml.Slice.alloc(allocator, graph_shape);
    defer graph_slice.free(allocator);
    try out_graph_buf.toSlice(io, graph_slice);
    const graph_vals = graph_slice.items(u16);

    const x16_shape = zml.Shape.init(.{ .m = 16, .k = K }, .bf16);
    const q_flat_shape = zml.Shape.init(.{ .n = N, .k = K }, .i8);
    const s_flat_shape = zml.Shape.init(.{ .n = N, .nb = NB }, .f16);
    var exe_tri = try platform.compileFn(allocator, io, q8Triton, .{
        zml.Tensor.fromShape(x16_shape),
        zml.Tensor.fromShape(q_flat_shape),
        zml.Tensor.fromShape(s_flat_shape),
    }, .{});
    defer exe_tri.deinit();

    const x16_data = try allocator.alloc(u16, @intCast(16 * K));
    defer allocator.free(x16_data);
    for (x16_data, 0..) |*v, i| {
        v.* = if (i < K) x_data[i] else bf16Bits(rand.float(f32) * 0.02 - 0.01);
    }

    var x16_buf = try zml.Buffer.fromBytes(io, platform, x16_shape, std.mem.sliceAsBytes(x16_data));
    defer x16_buf.deinit();
    var q_flat_buf = try zml.Buffer.fromBytes(io, platform, q_flat_shape, std.mem.sliceAsBytes(q_data));
    defer q_flat_buf.deinit();
    var s_flat_buf = try zml.Buffer.fromBytes(io, platform, s_flat_shape, std.mem.sliceAsBytes(s_data));
    defer s_flat_buf.deinit();

    var args_tri = try exe_tri.args(allocator);
    defer args_tri.deinit(allocator);
    var results_tri = try exe_tri.results(allocator);
    defer results_tri.deinit(allocator);
    args_tri.set(.{ x16_buf, q_flat_buf, s_flat_buf });

    try bench("q8_0_triton", &exe_tri, &args_tri, &results_tri, io);

    exe_tri.call(io, args_tri, &results_tri, .{ .wait = true });
    var out_tri_buf = results_tri.get(zml.Buffer);
    defer out_tri_buf.deinit();
    const tri_shape = zml.Shape.init(.{ .m = 16, .n = N }, .bf16);
    var tri_slice = try zml.Slice.alloc(allocator, tri_shape);
    defer tri_slice.free(allocator);
    try out_tri_buf.toSlice(io, tri_slice);
    const tri_vals = tri_slice.items(u16);
    var mismatch: usize = 0;
    var max_diff: u16 = 0;
    for (0..@intCast(N)) |i| {
        if (graph_vals[i] != tri_vals[i]) {
            mismatch += 1;
            const diff = if (graph_vals[i] > tri_vals[i]) graph_vals[i] - tri_vals[i] else tri_vals[i] - graph_vals[i];
            max_diff = @max(max_diff, diff);
        }
    }
    std.debug.print("correctness: {d}/{d} mismatches, max bf16 diff {d}; graph[0..4]={any} tri[0..4]={any}\n", .{
        mismatch, N, max_diff, graph_vals[0..4], tri_vals[0..4],
    });

    // Compare the QuantizedLinear-style reshape/tag path for prefill [1, 32, K].
    const x3_shape = zml.Shape.init(.{ .b = 1, .s = 32, .k = K }, .bf16);
    const x3_data = try allocator.alloc(u16, @intCast(32 * K));
    defer allocator.free(x3_data);
    for (x3_data, 0..) |*v, i| {
        const row: f32 = @floatFromInt(i / @as(usize, @intCast(K)));
        v.* = bf16Bits(0.01 + row * 0.001);
    }
    var x3_buf = try zml.Buffer.fromBytes(io, platform, x3_shape, std.mem.sliceAsBytes(x3_data));
    defer x3_buf.deinit();

    var exe_lin = try platform.compileFn(allocator, io, q8Linear, .{
        zml.Tensor.fromShape(x3_shape),
        zml.Tensor.fromShape(q_flat_shape),
        zml.Tensor.fromShape(s_flat_shape),
    }, .{});
    defer exe_lin.deinit();
    var exe_ref = try platform.compileFn(allocator, io, q8DotFlat, .{
        zml.Tensor.fromShape(x3_shape),
        zml.Tensor.fromShape(q_flat_shape),
        zml.Tensor.fromShape(s_flat_shape),
    }, .{});
    defer exe_ref.deinit();

    var args_lin = try exe_lin.args(allocator);
    defer args_lin.deinit(allocator);
    var results_lin = try exe_lin.results(allocator);
    defer results_lin.deinit(allocator);
    args_lin.set(.{ x3_buf, q_flat_buf, s_flat_buf });
    exe_lin.call(io, args_lin, &results_lin, .{ .wait = true });
    var out_lin = results_lin.get(zml.Buffer);
    defer out_lin.deinit();

    var args_ref = try exe_ref.args(allocator);
    defer args_ref.deinit(allocator);
    var results_ref = try exe_ref.results(allocator);
    defer results_ref.deinit(allocator);
    args_ref.set(.{ x3_buf, q_flat_buf, s_flat_buf });
    exe_ref.call(io, args_ref, &results_ref, .{ .wait = true });
    var out_ref = results_ref.get(zml.Buffer);
    defer out_ref.deinit();

    var lin_slice = try zml.Slice.alloc(allocator, x3_shape.withDtype(.bf16).setDim(.k, N));
    defer lin_slice.free(allocator);
    try out_lin.toSlice(io, lin_slice);
    var ref_slice = try zml.Slice.alloc(allocator, x3_shape.withDtype(.bf16).setDim(.k, N));
    defer ref_slice.free(allocator);
    try out_ref.toSlice(io, ref_slice);
    const lin_vals = lin_slice.items(u16);
    const ref_vals = ref_slice.items(u16);
    var lin_mismatch: usize = 0;
    for (0..@intCast(32 * N)) |i| if (lin_vals[i] != ref_vals[i]) {
        lin_mismatch += 1;
    };
    std.debug.print("linear prefill: {d}/{d} mismatches; lin[0..4]={any} ref[0..4]={any} lin[last4]={any} ref[last4]={any}\n", .{
        lin_mismatch, 32 * N, lin_vals[0..4], ref_vals[0..4], lin_vals[32 * @as(usize, @intCast(N)) - 4 ..][0..4], ref_vals[32 * @as(usize, @intCast(N)) - 4 ..][0..4],
    });

    // Decode shape: [1, K] -> [1, N] with the m=1 pad/slice path.
    const x1_shape = zml.Shape.init(.{ .m = 1, .k = K }, .bf16);
    var exe_lin1 = try platform.compileFn(allocator, io, q8Linear, .{
        zml.Tensor.fromShape(x1_shape),
        zml.Tensor.fromShape(q_flat_shape),
        zml.Tensor.fromShape(s_flat_shape),
    }, .{});
    defer exe_lin1.deinit();
    var exe_ref1 = try platform.compileFn(allocator, io, q8DotFlat, .{
        zml.Tensor.fromShape(x1_shape),
        zml.Tensor.fromShape(q_flat_shape),
        zml.Tensor.fromShape(s_flat_shape),
    }, .{});
    defer exe_ref1.deinit();

    var args_lin1 = try exe_lin1.args(allocator);
    defer args_lin1.deinit(allocator);
    var results_lin1 = try exe_lin1.results(allocator);
    defer results_lin1.deinit(allocator);
    args_lin1.set(.{ x_buf, q_flat_buf, s_flat_buf });
    exe_lin1.call(io, args_lin1, &results_lin1, .{ .wait = true });
    var out_lin1 = results_lin1.get(zml.Buffer);
    defer out_lin1.deinit();

    var args_ref1 = try exe_ref1.args(allocator);
    defer args_ref1.deinit(allocator);
    var results_ref1 = try exe_ref1.results(allocator);
    defer results_ref1.deinit(allocator);
    args_ref1.set(.{ x_buf, q_flat_buf, s_flat_buf });
    exe_ref1.call(io, args_ref1, &results_ref1, .{ .wait = true });
    var out_ref1 = results_ref1.get(zml.Buffer);
    defer out_ref1.deinit();

    const out1_shape = zml.Shape.init(.{ .m = 1, .n = N }, .bf16);
    var lin1_slice = try zml.Slice.alloc(allocator, out1_shape);
    defer lin1_slice.free(allocator);
    try out_lin1.toSlice(io, lin1_slice);
    var ref1_slice = try zml.Slice.alloc(allocator, out1_shape);
    defer ref1_slice.free(allocator);
    try out_ref1.toSlice(io, ref1_slice);
    const lin1_vals = lin1_slice.items(u16);
    const ref1_vals = ref1_slice.items(u16);
    var lin1_mismatch: usize = 0;
    for (0..@intCast(N)) |i| if (lin1_vals[i] != ref1_vals[i]) {
        lin1_mismatch += 1;
    };
    std.debug.print("linear decode: {d}/{d} mismatches; lin[0..4]={any} ref[0..4]={any}\n", .{
        lin1_mismatch, N, lin1_vals[0..4], ref1_vals[0..4],
    });
}
