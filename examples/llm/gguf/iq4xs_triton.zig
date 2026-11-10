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
    stride_scl_n: i64,
    stride_sch_n: i64,
    stride_d_n: i64,
};

pub const Iq4XsMatvec1 = tri.Kernel(Config, .{
    .name = "iq4xs_matvec1",
    .inputs = &.{ "x_ptr", "qs_ptr", "scl_ptr", "sch_ptr", "d_ptr", "lut_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

pub const Iq4XsMatvec = tri.Kernel(Config, .{
    .name = "iq4xs_matvec",
    .inputs = &.{ "x_ptr", "qs_ptr", "scl_ptr", "sch_ptr", "d_ptr", "lut_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec,
});

const Args = struct {
    x_ptr: Value,
    qs_ptr: Value,
    scl_ptr: Value,
    sch_ptr: Value,
    d_ptr: Value,
    lut_ptr: Value,
    out_ptr: Value,
};

fn declare(b: *Builder) !Args {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .qs_ptr = .{ .ptr = .i8 },
        .scl_ptr = .{ .ptr = .i8 },
        .sch_ptr = .{ .ptr = .i16 },
        .d_ptr = .{ .ptr = .f16 },
        .lut_ptr = .{ .ptr = .i8 },
        .out_ptr = .{ .ptr = .bf16 },
    });
    return .{
        .x_ptr = a.x_ptr,
        .qs_ptr = a.qs_ptr,
        .scl_ptr = a.scl_ptr,
        .sch_ptr = a.sch_ptr,
        .d_ptr = a.d_ptr,
        .lut_ptr = a.lut_ptr,
        .out_ptr = a.out_ptr,
    };
}

/// Dequantizes one 256-element IQ4_XS block per output row into `[block_n, 256]` f32.
fn dequantIq4Xs(
    b: *Builder,
    qs: Value,
    scl: Value,
    sch: Value,
    d: Value,
    lut: Value,
    block_n: i64,
    comptime bf16_out: bool,
) Value {
    const shape = [_]i64{ block_n, 8, 16 };
    const qs3 = b.reshape(qs, &shape).to(.i32);
    const lo = qs3.bitAnd(b.splat(@as(i32, 0x0F), &shape));
    const hi = b.shrsi(qs3, b.splat(@as(i32, 4), &shape)).bitAnd(b.splat(@as(i32, 0x0F), &shape));

    const lut3 = b.broadcastTo(b.reshape(lut, &.{ 1, 1, 16 }), &shape);
    const val_lo = b.gather(lut3, lo, 2);
    const val_hi = b.gather(lut3, hi, 2);

    const scl3 = b.reshape(scl, &.{ block_n, 4 }).to(.i32);
    const sch3 = b.reshape(sch, &.{block_n}).to(.i32);

    const ib = b.arange(0, 8, .i32);
    const scl_idx = b.broadcastTo(b.shrsi(ib, b.splat(@as(i32, 2), &.{8})).expandDims(0), &.{ block_n, 8 });
    const scl_shift = b.broadcastTo(ib.bitAnd(b.splat(@as(i32, 1), &.{8})).mul(b.splat(@as(i32, 4), &.{8})).expandDims(0), &.{ block_n, 8 });
    const scl_g = b.gather(scl3, scl_idx, 1);
    const scl_part = b.shrui(scl_g, scl_shift).bitAnd(b.splat(@as(i32, 0x0F), &.{ block_n, 8 }));

    const sch_shift = b.broadcastTo(ib.mul(b.splat(@as(i32, 2), &.{8})).expandDims(0), &.{ block_n, 8 });
    const sch_g = b.broadcastTo(sch3.expandDims(1), &.{ block_n, 8 });
    const sch_part = b.shli(b.shrui(sch_g, sch_shift).bitAnd(b.splat(@as(i32, 3), &.{ block_n, 8 })), b.splat(@as(i32, 4), &.{ block_n, 8 }));

    const ls = scl_part.bitOr(sch_part);
    const dl = d.to(.f32).expandDims(1).mul(ls.to(.f32).sub(b.splat(@as(f32, 32), &.{ block_n, 8 })));

    const w_lo = val_lo.to(.f32).mul(dl.expandDims(2));
    const w_hi = val_hi.to(.f32).mul(dl.expandDims(2));
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
) struct { Value, Value, Value, Value, Value } {
    const qs_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qs_n), &.{ block_n, 128 })
        .add(b.broadcastTo(k_iter.mul(128).add(b.arange(0, 128, .i64)).expandDims(0), &.{ block_n, 128 }));
    const qs = b.load(a.qs_ptr.addPtr(qs_off));

    const scl_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_scl_n), &.{ block_n, 4 })
        .add(b.broadcastTo(k_iter.mul(4).add(b.arange(0, 4, .i64)).expandDims(0), &.{ block_n, 4 }));
    const scl = b.load(a.scl_ptr.addPtr(scl_off));

    const sch_off = offs_n.to(.i64).mul(cfg.stride_sch_n).add(k_iter);
    const sch = b.load(a.sch_ptr.addPtr(sch_off));

    const d_off = offs_n.to(.i64).mul(cfg.stride_d_n).add(k_iter);
    const d = b.load(a.d_ptr.addPtr(d_off));

    const lut = b.load(a.lut_ptr.addPtr(b.arange(0, 16, .i64)));
    return .{ qs, scl, sch, d, lut };
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
        const w = dequantIq4Xs(b, block[0], block[1], block[2], block[3], block[4], block_n, false);

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
        const w = dequantIq4Xs(b, block[0], block[1], block[2], block[3], block[4], block_n, true);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.n), &.{ cfg.block_m, block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64), &.{ cfg.block_m, block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}
