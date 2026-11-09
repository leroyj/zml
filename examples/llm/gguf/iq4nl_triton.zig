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
    stride_d_n: i64,
};

pub const Iq4NlMatvec1 = tri.Kernel(Config, .{
    .name = "iq4nl_matvec1",
    .inputs = &.{ "x_ptr", "qs_ptr", "d_ptr", "lut_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

pub const Iq4NlMatvec = tri.Kernel(Config, .{
    .name = "iq4nl_matvec",
    .inputs = &.{ "x_ptr", "qs_ptr", "d_ptr", "lut_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec,
});

const Args = struct {
    x_ptr: Value,
    qs_ptr: Value,
    d_ptr: Value,
    lut_ptr: Value,
    out_ptr: Value,
};

fn declare(b: *Builder) !Args {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .qs_ptr = .{ .ptr = .i8 },
        .d_ptr = .{ .ptr = .f16 },
        .lut_ptr = .{ .ptr = .i8 },
        .out_ptr = .{ .ptr = .bf16 },
    });
    return .{
        .x_ptr = a.x_ptr,
        .qs_ptr = a.qs_ptr,
        .d_ptr = a.d_ptr,
        .lut_ptr = a.lut_ptr,
        .out_ptr = a.out_ptr,
    };
}

/// Dequantizes one 32-element IQ4_NL block per output row into `[block_n, 32]` f32.
fn dequantIq4Nl(b: *Builder, qs: Value, d: Value, lut: Value, block_n: i64) Value {
    const shape = [_]i64{ block_n, 16 };
    const qs3 = b.reshape(qs, &shape).to(.i32);
    const lo = qs3.bitAnd(b.splat(@as(i32, 0x0F), &shape));
    const hi = b.shrsi(qs3, b.splat(@as(i32, 4), &shape)).bitAnd(b.splat(@as(i32, 0x0F), &shape));

    const lut3 = b.broadcastTo(b.reshape(lut, &.{ 1, 16 }), &shape);
    const val_lo = b.gather(lut3, lo, 1);
    const val_hi = b.gather(lut3, hi, 1);

    const d_f = d.to(.f32).expandDims(1);
    const w_lo = val_lo.to(.f32).mul(d_f);
    const w_hi = val_hi.to(.f32).mul(d_f);
    return b.catOpts(w_lo, w_hi, .{ .dim = 1 });
}

fn loadBlock(b: *Builder, a: Args, cfg: Config, offs_n: Value, k_iter: Value, block_n: i64) struct { Value, Value, Value } {
    const qs_off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qs_n), &.{ block_n, 16 })
        .add(b.broadcastTo(k_iter.mul(16).add(b.arange(0, 16, .i64)).expandDims(0), &.{ block_n, 16 }));
    const qs = b.load(a.qs_ptr.addPtr(qs_off));

    const d_off = offs_n.to(.i64).mul(cfg.stride_d_n).add(k_iter);
    const d = b.load(a.d_ptr.addPtr(d_off));

    const lut = b.load(a.lut_ptr.addPtr(b.arange(0, 16, .i64)));
    return .{ qs, d, lut };
}

fn runMatvec1(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try declare(b);
    const pid_n = b.programId(.x);
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));
    const block_n = cfg.block_n;

    const acc = b.zeros(&.{block_n}, .f32);
    const num_k = @divExact(cfg.k, 32);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const x_off = k_iter.mul(32).add(b.arange(0, 32, .i64));
        const x_val = b.load(a.x_ptr.addPtr(x_off));
        const x_bcast = b.broadcastTo(x_val.expandDims(0), &.{ block_n, 32 });

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const w = dequantIq4Nl(b, block[0], block[1], block[2], block_n);

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
    const num_k = @divExact(cfg.k, 32);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const offs_k = k_iter.mul(32).add(b.arange(0, 32, .i64));
        const x_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.k), &.{ cfg.block_m, 32 });
        const x_cols = b.broadcastTo(offs_k.expandDims(0), &.{ cfg.block_m, 32 });
        const x_val = b.load(a.x_ptr.addPtr(x_rows.add(x_cols)));

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const w = dequantIq4Nl(b, block[0], block[1], block[2], block_n).to(.bf16);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.n), &.{ cfg.block_m, block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64), &.{ cfg.block_m, block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}
