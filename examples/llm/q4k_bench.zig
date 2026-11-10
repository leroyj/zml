const std = @import("std");

const zml = @import("zml");
const q4k = @import("gguf/q4k_triton.zig");

/// Microbenchmark for the Q4_K few-row dot kernel used by speculative
/// verification. It isolates one projection so launch configurations can be
/// compared in seconds instead of through a full 19 GiB model run.
///
/// The synthetic weights have the real layout and byte width, so the kernel
/// streams exactly the bytes it would stream in the model; only the values are
/// synthetic, which does not affect timing.
const n: i64 = 5120;
const k: i64 = 5120;
const iters: usize = 20;
const warmup: usize = 5;

fn matvec(
    x: zml.Tensor,
    qs: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    block_n: i64,
    block_m: i64,
    num_warps: i64,
    num_stages: i64,
) zml.Tensor {
    const m = x.dim(.m);
    const kk = x.dim(.k);
    const nn = qs.dim(.n);
    const kb = @divExact(kk, 256);
    const cfg: q4k.Config = .{
        .n = nn,
        .k = kk,
        .block_n = block_n,
        .block_m = block_m,
        .stride_qs_n = @divExact(kk, 2),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q4k.Q4KMatvec.call(
        .{ .x_ptr = x, .qs_ptr = qs, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .m = m, .n = nn }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(m, block_m)), @intCast(@divExact(nn, block_n)), 1 },
            .num_stages = @intCast(num_stages),
            .num_warps = @intCast(num_warps),
        },
    ).out;
}

fn matvec1(
    x: zml.Tensor,
    qs: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    block_n: i64,
) zml.Tensor {
    const kk = x.dim(.k);
    const nn = qs.dim(.n);
    const kb = @divExact(kk, 256);
    const cfg: q4k.Config = .{
        .n = nn,
        .k = kk,
        .block_n = block_n,
        .block_m = 16,
        .stride_qs_n = @divExact(kk, 2),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q4k.Q4KMatvec1.call(
        .{ .x_ptr = x, .qs_ptr = qs, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .m = 1, .n = nn }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(nn, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

/// `Results.get` leaves the handle in the result slot; PJRT treats a non-null
/// slot as a preallocated destination and would reuse a freed handle.
fn releaseResults(results: anytype) void {
    const raw: [*]usize = @ptrCast(@constCast(results.flat_buffers.raw_buffers.ptr));
    @memset(raw[0..results.flat_buffers.raw_buffers.len], 0);
}

/// `compileFn` needs a plain function, so the comptime row count is bound here.
const Rows8 = struct {
    fn f(x: zml.Tensor, qs: zml.Tensor, sc: zml.Tensor, mn: zml.Tensor, d: zml.Tensor, dmin: zml.Tensor, block_n: i64) zml.Tensor {
        return matvecRows(8, x, qs, sc, mn, d, dmin, block_n);
    }
};

fn matvecRows(
    comptime rows_: i64,
    x: zml.Tensor,
    qs: zml.Tensor,
    sc: zml.Tensor,
    mn: zml.Tensor,
    d: zml.Tensor,
    dmin: zml.Tensor,
    block_n: i64,
) zml.Tensor {
    const m = x.dim(.m);
    const kk = x.dim(.k);
    const nn = qs.dim(.n);
    const kb = @divExact(kk, 256);
    const rows: i64 = rows_;
    const cfg: q4k.Config = .{
        .n = nn,
        .k = kk,
        .block_n = block_n,
        .block_m = rows,
        .stride_qs_n = @divExact(kk, 2),
        .stride_sc_n = kb * 8,
        .stride_mn_n = kb * 8,
        .stride_d_n = kb,
        .stride_dmin_n = kb,
    };
    return q4k.Q4KMatvecRows(rows_).call(
        .{ .x_ptr = x, .qs_ptr = qs, .sc_ptr = sc, .mn_ptr = mn, .d_ptr = d, .dmin_ptr = dmin },
        .{ .out = zml.Shape.init(.{ .m = m, .n = nn }, .bf16) },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(@divExact(nn, block_n)), 1, 1 },
            .num_stages = 3,
            .num_warps = 4,
        },
    ).out;
}

fn time(label: []const u8, exe: *zml.Exe, args: anytype, results: anytype, io: std.Io, rows: i64) !void {
    for (0..warmup) |_| {
        exe.call(io, args.*, results, .{ .wait = true });
        var out = results.get(zml.Buffer);
        out.deinit();
        releaseResults(results);
    }
    const start = std.Io.Clock.now(.awake, io);
    for (0..iters) |_| {
        exe.call(io, args.*, results, .{ .wait = true });
        var out = results.get(zml.Buffer);
        out.deinit();
        releaseResults(results);
    }
    const elapsed = start.untilNow(io, .awake);
    const seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
    const ms = seconds * 1000.0 / @as(f64, @floatFromInt(iters));
    // One full weight sweep is the floor: 4.5 bits/weight for Q4_K.
    const bytes = @as(f64, @floatFromInt(n * k)) * 4.5 / 8.0;
    const gbs = bytes / (ms / 1000.0) / 1e9;
    std.debug.print("{s:<28} {d:>7.2} ms  {d:>6.1} GB/s  ({d:.1} ms/row)\n", .{ label, ms, gbs, ms / @as(f64, @floatFromInt(rows)) });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const kb = @divExact(k, 256);
    const qs_shape = zml.Shape.init(.{ .n = n, .k2 = @divExact(k, 2) }, .i8);
    const sc_shape = zml.Shape.init(.{ .n = n, .kb = kb, .eight = 8 }, .i8);
    const d_shape = zml.Shape.init(.{ .n = n, .kb = kb }, .f16);

    const qs_data = try allocator.alloc(i8, @intCast(n * k / 2));
    defer allocator.free(qs_data);
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(std.mem.sliceAsBytes(qs_data));

    const sc_data = try allocator.alloc(i8, @intCast(n * kb * 8));
    defer allocator.free(sc_data);
    prng.random().bytes(std.mem.sliceAsBytes(sc_data));

    const d_data = try allocator.alloc(f16, @intCast(n * kb));
    defer allocator.free(d_data);
    @memset(d_data, 0.01);

    var qs_buf = try zml.Buffer.fromBytes(io, platform, qs_shape, std.mem.sliceAsBytes(qs_data));
    defer qs_buf.deinit();
    var sc_buf = try zml.Buffer.fromBytes(io, platform, sc_shape, std.mem.sliceAsBytes(sc_data));
    defer sc_buf.deinit();
    var d_buf = try zml.Buffer.fromBytes(io, platform, d_shape, std.mem.sliceAsBytes(d_data));
    defer d_buf.deinit();

    // The model pads the 8-row verification block to 16 before the dot, so the
    // bench uses the same padded shape.
    const rows = 16;
    const x_shape = zml.Shape.init(.{ .m = rows, .k = k }, .bf16);
    const x_data = try allocator.alloc(u16, @intCast(rows * k));
    defer allocator.free(x_data);
    for (x_data) |*v| v.* = @bitCast(@as(f16, 0.01));
    var x_buf = try zml.Buffer.fromBytes(io, platform, x_shape, std.mem.sliceAsBytes(x_data));
    defer x_buf.deinit();

    const x1_shape = zml.Shape.init(.{ .m = 1, .k = k }, .bf16);
    const x1_data = try allocator.alloc(u16, @intCast(k));
    defer allocator.free(x1_data);
    for (x1_data) |*v| v.* = @bitCast(@as(f16, 0.01));
    var x1_buf = try zml.Buffer.fromBytes(io, platform, x1_shape, std.mem.sliceAsBytes(x1_data));
    defer x1_buf.deinit();

    std.debug.print("Q4_K {d}x{d}, {d} rows (8 real, padded), {d} iterations\n", .{ n, k, rows, iters });

    // Single-row reference: the fastest way to stream these weights today.
    {
        var exe = try platform.compileFn(allocator, io, matvec1, .{ zml.Tensor.fromShape(x1_shape), zml.Tensor.fromShape(qs_shape), zml.Tensor.fromShape(sc_shape), zml.Tensor.fromShape(sc_shape), zml.Tensor.fromShape(d_shape), zml.Tensor.fromShape(d_shape), 64 }, .{});
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ x1_buf, qs_buf, sc_buf, sc_buf, d_buf, d_buf });
        try time("matvec1 m=1 bn=64", &exe, &args, &results, io, 1);
    }

    const Config = struct { rows: i64, bn: i64, warps: i64, stages: i64 };
    const configs = [_]Config{
        .{ .rows = 16, .bn = 16, .warps = 4, .stages = 3 },
        .{ .rows = 16, .bn = 16, .warps = 4, .stages = 4 },
        .{ .rows = 16, .bn = 16, .warps = 1, .stages = 3 },
        .{ .rows = 16, .bn = 8, .warps = 4, .stages = 3 },
        .{ .rows = 8, .bn = 16, .warps = 4, .stages = 3 },
        .{ .rows = 8, .bn = 16, .warps = 2, .stages = 3 },
        .{ .rows = 8, .bn = 8, .warps = 4, .stages = 3 },
        .{ .rows = 8, .bn = 32, .warps = 4, .stages = 3 },
        .{ .rows = 8, .bn = 64, .warps = 8, .stages = 2 },
    };
    for (configs) |cfg| {
        const xs = zml.Shape.init(.{ .m = cfg.rows, .k = k }, .bf16);
        const xd = try allocator.alloc(u16, @intCast(cfg.rows * k));
        defer allocator.free(xd);
        for (xd) |*v| v.* = @bitCast(@as(f16, 0.01));
        var xb = try zml.Buffer.fromBytes(io, platform, xs, std.mem.sliceAsBytes(xd));
        defer xb.deinit();
        var exe = try platform.compileFn(allocator, io, matvec, .{
            zml.Tensor.fromShape(xs),
            zml.Tensor.fromShape(qs_shape),
            zml.Tensor.fromShape(sc_shape),
            zml.Tensor.fromShape(sc_shape),
            zml.Tensor.fromShape(d_shape),
            zml.Tensor.fromShape(d_shape),
            cfg.bn,
            cfg.rows,
            cfg.warps,
            cfg.stages,
        }, .{});
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ xb, qs_buf, sc_buf, sc_buf, d_buf, d_buf });
        const label = try std.fmt.allocPrint(allocator, "dot m={d} bn={d} w={d} s={d}", .{ cfg.rows, cfg.bn, cfg.warps, cfg.stages });
        defer allocator.free(label);
        try time(label, &exe, &args, &results, io, cfg.rows);
    }

    const RowConfig = struct { bn: i64, warps: i64 };
    const row_configs = [_]RowConfig{
        .{ .bn = 16, .warps = 4 },
        .{ .bn = 32, .warps = 4 },
        .{ .bn = 64, .warps = 4 },
    };
    for (row_configs) |rc| {
        const xs = zml.Shape.init(.{ .m = 8, .k = k }, .bf16);
        const xd = try allocator.alloc(u16, @intCast(8 * k));
        defer allocator.free(xd);
        for (xd) |*v| v.* = @bitCast(@as(f16, 0.01));
        var xb = try zml.Buffer.fromBytes(io, platform, xs, std.mem.sliceAsBytes(xd));
        defer xb.deinit();
        var exe = try platform.compileFn(allocator, io, Rows8.f, .{
            zml.Tensor.fromShape(xs),
            zml.Tensor.fromShape(qs_shape),
            zml.Tensor.fromShape(sc_shape),
            zml.Tensor.fromShape(sc_shape),
            zml.Tensor.fromShape(d_shape),
            zml.Tensor.fromShape(d_shape),
            rc.bn,
        }, .{});
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ xb, qs_buf, sc_buf, sc_buf, d_buf, d_buf });
        const label = try std.fmt.allocPrint(allocator, "rows m=8 bn={d} w={d}", .{ rc.bn, rc.warps });
        defer allocator.free(label);
        try time(label, &exe, &args, &results, io, 8);
    }
}
