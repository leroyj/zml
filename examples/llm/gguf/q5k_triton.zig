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
    stride_qh_n: i64,
    stride_sc_n: i64,
    stride_mn_n: i64,
    stride_d_n: i64,
    stride_dmin_n: i64,
};

pub const Q5KMatvec1 = tri.Kernel(Config, .{
    .name = "q5k_matvec1",
    .inputs = &.{ "x_ptr", "qs_ptr", "qh_ptr", "sc_ptr", "mn_ptr", "d_ptr", "dmin_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

pub const Q5KMatvec = tri.Kernel(Config, .{
    .name = "q5k_matvec",
    .inputs = &.{ "x_ptr", "qs_ptr", "qh_ptr", "sc_ptr", "mn_ptr", "d_ptr", "dmin_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec,
});

const Args = struct {
    x_ptr: Value,
    qs_ptr: Value,
    qh_ptr: Value,
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
        .qh_ptr = .{ .ptr = .i8 },
        .sc_ptr = .{ .ptr = .i8 },
        .mn_ptr = .{ .ptr = .i8 },
        .d_ptr = .{ .ptr = .f16 },
        .dmin_ptr = .{ .ptr = .f16 },
        .out_ptr = .{ .ptr = .bf16 },
    });
    return .{
        .x_ptr = a.x_ptr,
        .qs_ptr = a.qs_ptr,
        .qh_ptr = a.qh_ptr,
        .sc_ptr = a.sc_ptr,
        .mn_ptr = a.mn_ptr,
        .d_ptr = a.d_ptr,
        .dmin_ptr = a.dmin_ptr,
        .out_ptr = a.out_ptr,
    };
}

/// Dequantizes one 256-element Q5_K block per output row into `[block_n, 256]` f32.
fn dequantQ5K(
    b: *Builder,
    qs: Value,
    qh: Value,
    sc_lo: Value,
    sc_hi: Value,
    mn_lo: Value,
    mn_hi: Value,
    d: Value,
    dmin: Value,
    block_n: i64,
) Value {
    const shape = [_]i64{ block_n, 4, 32 };
    const qs3 = b.reshape(qs, &shape).to(.i32);
    const qh3 = b.broadcastTo(b.reshape(qh, &.{ block_n, 1, 32 }).to(.i32), &shape);

    const chunk = b.arange(0, 4, .i32);
    const shift_lo = b.reshape(chunk.mul(b.splat(@as(i32, 2), &.{4})), &.{ 1, 4, 1 });
    const shift_hi = b.reshape(chunk.mul(b.splat(@as(i32, 2), &.{4})).add(b.splat(@as(i32, 1), &.{4})), &.{ 1, 4, 1 });
    const shift_lo_b = b.broadcastTo(shift_lo, &shape);
    const shift_hi_b = b.broadcastTo(shift_hi, &shape);

    const lo = qs3.bitAnd(b.splat(@as(i32, 0x0F), &shape));
    const hi = b.shrsi(qs3, b.splat(@as(i32, 4), &shape)).bitAnd(b.splat(@as(i32, 0x0F), &shape));

    const lo5 = lo.add(b.shrui(qh3, shift_lo_b).bitAnd(b.splat(@as(i32, 1), &shape)).mul(b.splat(@as(i32, 16), &shape)));
    const hi5 = hi.add(b.shrui(qh3, shift_hi_b).bitAnd(b.splat(@as(i32, 1), &shape)).mul(b.splat(@as(i32, 16), &shape)));

    const sc_lo_f = sc_lo.to(.f32).expandDims(2);
    const sc_hi_f = sc_hi.to(.f32).expandDims(2);
    const mn_lo_f = mn_lo.to(.f32).expandDims(2);
    const mn_hi_f = mn_hi.to(.f32).expandDims(2);
    const d_f = d.to(.f32).expandDims(1).expandDims(2);
    const dmin_f = dmin.to(.f32).expandDims(1).expandDims(2);

    const w_lo = d_f.mul(sc_lo_f).mul(lo5.to(.f32)).sub(dmin_f.mul(mn_lo_f));
    const w_hi = d_f.mul(sc_hi_f).mul(hi5.to(.f32)).sub(dmin_f.mul(mn_hi_f));
    const w = b.catOpts(w_lo, w_hi, .{ .dim = 2 });
    return b.reshape(w, &.{ block_n, 256 });
}

fn loadBlock(
    b: *Builder,
    a: Args,
    cfg: Config,
    offs_n: Value,
    k_iter: Value,
    block_n: i64,
) struct { Value, Value, Value, Value, Value, Value, Value } {
    const qs_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qs_n), &.{ block_n, 128 })
        .add(b.broadcastTo(k_iter.mul(128).add(b.arange(0, 128, .i64)).expandDims(0), &.{ block_n, 128 }));
    const qs = b.load(a.qs_ptr.addPtr(qs_off));

    const qh_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qh_n), &.{ block_n, 32 })
        .add(b.broadcastTo(k_iter.mul(32).add(b.arange(0, 32, .i64)).expandDims(0), &.{ block_n, 32 }));
    const qh = b.load(a.qh_ptr.addPtr(qh_off));

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
    return .{ qs, qh, sc_lo, sc_hi, mn_lo, mn_hi, d };
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
        const w = dequantQ5K(b, block[0], block[1], block[2], block[3], block[4], block[5], block[6], dmin, block_n);

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
        const w = dequantQ5K(b, block[0], block[1], block[2], block[3], block[4], block[5], block[6], dmin, block_n).to(.bf16);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.n), &.{ cfg.block_m, block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64), &.{ cfg.block_m, block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}
