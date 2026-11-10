const std = @import("std");
const zml = @import("zml");
const tri = zml.kernel.triton;
const Builder = tri.Builder;
const Value = tri.Value;

pub const Config = struct {
    n: i64,
    k: i64,
    block_n: i64,
    block_m: i64,
    stride_qs_n: i64,
    stride_sc_n: i64,
    stride_mn_n: i64,
    stride_d_n: i64,
    stride_dmin_n: i64,
};

pub const Q4KMatvec1 = tri.Kernel(Config, .{
    .name = "q4k_matvec1",
    .inputs = &.{ "x_ptr", "qs_ptr", "sc_ptr", "mn_ptr", "d_ptr", "dmin_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

pub const Q4KMatvec = tri.Kernel(Config, .{
    .name = "q4k_matvec",
    .inputs = &.{ "x_ptr", "qs_ptr", "sc_ptr", "mn_ptr", "d_ptr", "dmin_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec,
});

/// Few-row variant: each program owns one block of output columns and keeps one
/// accumulator per row in registers. The weight block is dequantized once and
/// reused by every row, so a verification block pays the dequant cost once and
/// needs no `tl.dot` operand layout.
///
/// Measured against `Q4KMatvec` with `examples/llm/q4k_bench`: both reach the
/// same 58 GB/s at `block_n = 16`, so the model uses the dot. Kept because it
/// is the natural alternative to re-measure against.
pub fn Q4KMatvecRows(comptime rows: i64) type {
    return tri.Kernel(Config, .{
        .name = "q4k_matvec_rows",
        .inputs = &.{ "x_ptr", "qs_ptr", "sc_ptr", "mn_ptr", "d_ptr", "dmin_ptr" },
        .outputs = &.{"out"},
        .run = struct {
            fn run(b: *Builder, cfg: Config) tri.FinishError!void {
                return runMatvecRows(b, cfg, rows);
            }
        }.run,
    });
}

const Args = struct {
    x_ptr: Value,
    qs_ptr: Value,
    sc_ptr: Value,
    mn_ptr: Value,
    d_ptr: Value,
    dmin_ptr: Value,
    out_ptr: Value,
};

fn declare(b: *Builder) !Args {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .qs_ptr = .{ .ptr = .i8 },
        .sc_ptr = .{ .ptr = .i8 },
        .mn_ptr = .{ .ptr = .i8 },
        .d_ptr = .{ .ptr = .f16 },
        .dmin_ptr = .{ .ptr = .f16 },
        .out_ptr = .{ .ptr = .bf16 },
    });
    return .{
        .x_ptr = a.x_ptr,
        .qs_ptr = a.qs_ptr,
        .sc_ptr = a.sc_ptr,
        .mn_ptr = a.mn_ptr,
        .d_ptr = a.d_ptr,
        .dmin_ptr = a.dmin_ptr,
        .out_ptr = a.out_ptr,
    };
}

/// Dequantizes one 256-element Q4_K block per output row into `[block_n, 256]` f32.
fn dequantQ4K(
    b: *Builder,
    qs: Value,
    sc_lo: Value,
    sc_hi: Value,
    mn_lo: Value,
    mn_hi: Value,
    d: Value,
    dmin: Value,
    block_n: i64,
    comptime bf16_out: bool,
) Value {
    const qs3 = b.reshape(qs, &.{ block_n, 4, 32 }).to(.i32);
    const lo = qs3.bitAnd(b.splat(@as(i32, 0x0F), &.{ block_n, 4, 32 })).to(.f32);
    const hi = b.shrsi(qs3, b.splat(@as(i32, 4), &.{ block_n, 4, 32 })).bitAnd(b.splat(@as(i32, 0x0F), &.{ block_n, 4, 32 })).to(.f32);

    const sc_lo_f = sc_lo.to(.f32).expandDims(2);
    const sc_hi_f = sc_hi.to(.f32).expandDims(2);
    const mn_lo_f = mn_lo.to(.f32).expandDims(2);
    const mn_hi_f = mn_hi.to(.f32).expandDims(2);
    const d_f = d.to(.f32).expandDims(1).expandDims(2);
    const dmin_f = dmin.to(.f32).expandDims(1).expandDims(2);

    const w_lo = d_f.mul(sc_lo_f).mul(lo).sub(dmin_f.mul(mn_lo_f));
    const w_hi = d_f.mul(sc_hi_f).mul(hi).sub(dmin_f.mul(mn_hi_f));
    // Round to the dot operand type before the concat so the interleave and the
    // reshape move half as many bytes.
    const w = if (bf16_out) b.catOpts(w_lo.to(.bf16), w_hi.to(.bf16), .{ .dim = 2 }) else b.catOpts(w_lo, w_hi, .{ .dim = 2 });
    return b.reshape(w, &.{ block_n, 256 });
}

fn loadBlock(
    b: *Builder,
    a: Args,
    cfg: Config,
    offs_n: Value,
    k_iter: Value,
    block_n: i64,
) struct { Value, Value, Value, Value, Value, Value } {
    const qs_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qs_n), &.{ block_n, 128 })
        .add(b.broadcastTo(k_iter.mul(128).add(b.arange(0, 128, .i64)).expandDims(0), &.{ block_n, 128 }));
    const qs = b.load(a.qs_ptr.addPtr(qs_off));

    const sc_base = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_sc_n), &.{ block_n, 4 })
        .add(b.broadcastTo(k_iter.mul(8).add(b.arange(0, 4, .i64).mul(2)).expandDims(0), &.{ block_n, 4 }));
    const sc_lo = b.load(a.sc_ptr.addPtr(sc_base));
    const sc_hi = b.load(a.sc_ptr.addPtr(sc_base.add(b.splat(@as(i64, 1), &.{ block_n, 4 }))));

    const mn_base = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_mn_n), &.{ block_n, 4 })
        .add(b.broadcastTo(k_iter.mul(8).add(b.arange(0, 4, .i64).mul(2)).expandDims(0), &.{ block_n, 4 }));
    const mn_lo = b.load(a.mn_ptr.addPtr(mn_base));
    const mn_hi = b.load(a.mn_ptr.addPtr(mn_base.add(b.splat(@as(i64, 1), &.{ block_n, 4 }))));

    const d_off = offs_n.to(.i64).mul(cfg.stride_d_n).add(k_iter);
    const d = b.load(a.d_ptr.addPtr(d_off));
    return .{ qs, sc_lo, sc_hi, mn_lo, mn_hi, d };
}

fn runMatvec1(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try declare(b);
    const pid_n = b.programId(.x);
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));
    const block_n = cfg.block_n;

    const acc = b.zeros(&.{block_n}, .f32);
    const num_k = @divExact(cfg.k, 256);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const x_off = k_iter.mul(256).add(b.arange(0, 256, .i64));
        const x_val = b.load(a.x_ptr.addPtr(x_off));
        const x_bcast = b.broadcastTo(x_val.expandDims(0), &.{ block_n, 256 });

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const dmin_off = offs_n.to(.i64).mul(cfg.stride_dmin_n).add(k_iter);
        const dmin = b.load(a.dmin_ptr.addPtr(dmin_off));
        const w = dequantQ4K(b, block[0], block[1], block[2], block[3], block[4], block[5], dmin, block_n, false);

        const prod = w.mul(x_bcast.to(.f32));
        const new_acc = acc_.add(b.sumOpts(prod, .{ .axis = 1 }));
        loop.yield(.{new_acc});
    }

    const out_off = offs_n.to(.i64);
    b.store(a.out_ptr.addPtr(out_off), loop.results[0].to(.bf16));
}

fn runMatvec(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try declare(b);
    const pid_m = b.programId(.x);
    const pid_n = b.programId(.y);
    const offs_m = pid_m.mul(cfg.block_m).add(b.arange(0, cfg.block_m, .i32));
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));
    const block_n = cfg.block_n;

    const acc = b.zeros(&.{ cfg.block_m, block_n }, .f32);
    const num_k = @divExact(cfg.k, 256);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const offs_k = k_iter.mul(256).add(b.arange(0, 256, .i64));
        const x_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.k), &.{ cfg.block_m, 256 });
        const x_cols = b.broadcastTo(offs_k.expandDims(0), &.{ cfg.block_m, 256 });
        const x_val = b.load(a.x_ptr.addPtr(x_rows.add(x_cols)));

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const dmin_off = offs_n.to(.i64).mul(cfg.stride_dmin_n).add(k_iter);
        const dmin = b.load(a.dmin_ptr.addPtr(dmin_off));
        const w = dequantQ4K(b, block[0], block[1], block[2], block[3], block[4], block[5], dmin, block_n, true);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.n), &.{ cfg.block_m, block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64), &.{ cfg.block_m, block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}

fn runMatvecRows(b: *Builder, cfg: Config, comptime rows: i64) tri.FinishError!void {
    const a = try declare(b);
    const pid_n = b.programId(.x);
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));
    const block_n = cfg.block_n;

    var accs: [rows]Value = undefined;
    inline for (0..rows) |r| accs[r] = b.zeros(&.{block_n}, .f32);

    const num_k = @divExact(cfg.k, 256);
    var loop = b.openFor(0, num_k, 1, accs);
    {
        const k_iter = loop.iv;

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const dmin_off = offs_n.to(.i64).mul(cfg.stride_dmin_n).add(k_iter);
        const dmin = b.load(a.dmin_ptr.addPtr(dmin_off));
        const w = dequantQ4K(b, block[0], block[1], block[2], block[3], block[4], block[5], dmin, block_n, false);

        inline for (0..rows) |r| {
            const x_off = k_iter.mul(256)
                .add(b.arange(0, 256, .i64))
                .add(b.splat(@as(i64, @intCast(r)) * cfg.k, &.{256}));
            const x_val = b.load(a.x_ptr.addPtr(x_off));
            const x_bcast = b.broadcastTo(x_val.expandDims(0), &.{ block_n, 256 });
            loop.carried[r] = loop.carried[r].add(b.sumOpts(w.mul(x_bcast.to(.f32)), .{ .axis = 1 }));
        }
        loop.yield(loop.carried);
    }

    // `out` is [rows, n] row-major: one contiguous store per row.
    inline for (0..rows) |r| {
        const out_off = offs_n.to(.i64).add(b.splat(@as(i64, @intCast(r)) * cfg.n, &.{block_n}));
        b.store(a.out_ptr.addPtr(out_off), loop.results[r].to(.bf16));
    }
}
