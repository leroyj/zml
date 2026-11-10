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
    stride_ql_n: i64,
    stride_qh_n: i64,
    stride_sc_n: i64,
    stride_d_n: i64,
};

pub const Q6KMatvec1 = tri.Kernel(Config, .{
    .name = "q6k_matvec1",
    .inputs = &.{ "x_ptr", "ql0_ptr", "ql1_ptr", "qh_ptr", "sc1_ptr", "sc2_ptr", "sc3_ptr", "sc4_ptr", "d_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

pub const Q6KMatvec = tri.Kernel(Config, .{
    .name = "q6k_matvec",
    .inputs = &.{ "x_ptr", "ql0_ptr", "ql1_ptr", "qh_ptr", "sc1_ptr", "sc2_ptr", "sc3_ptr", "sc4_ptr", "d_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec,
});

const Args = struct {
    x_ptr: Value,
    ql0_ptr: Value,
    ql1_ptr: Value,
    qh_ptr: Value,
    sc1_ptr: Value,
    sc2_ptr: Value,
    sc3_ptr: Value,
    sc4_ptr: Value,
    d_ptr: Value,
    out_ptr: Value,
};

fn declare(b: *Builder) !Args {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .ql0_ptr = .{ .ptr = .i8 },
        .ql1_ptr = .{ .ptr = .i8 },
        .qh_ptr = .{ .ptr = .i8 },
        .sc1_ptr = .{ .ptr = .i8 },
        .sc2_ptr = .{ .ptr = .i8 },
        .sc3_ptr = .{ .ptr = .i8 },
        .sc4_ptr = .{ .ptr = .i8 },
        .d_ptr = .{ .ptr = .f16 },
        .out_ptr = .{ .ptr = .bf16 },
    });
    return .{
        .x_ptr = a.x_ptr,
        .ql0_ptr = a.ql0_ptr,
        .ql1_ptr = a.ql1_ptr,
        .qh_ptr = a.qh_ptr,
        .sc1_ptr = a.sc1_ptr,
        .sc2_ptr = a.sc2_ptr,
        .sc3_ptr = a.sc3_ptr,
        .sc4_ptr = a.sc4_ptr,
        .d_ptr = a.d_ptr,
        .out_ptr = a.out_ptr,
    };
}

fn sixBit(b: *Builder, lo: Value, qh: Value, shift: i32, shape: []const i64) Value {
    const bits = b.shrsi(qh, b.splat(@as(i32, shift), shape)).bitAnd(b.splat(@as(i32, 3), shape));
    const value = lo.bitOr(b.shli(bits, b.splat(@as(i32, 4), shape)));
    return value.sub(b.splat(@as(i32, 32), shape));
}

fn scaleChunk(b: *Builder, q: Value, sc: Value, block_n: i64) Value {
    const q4 = b.reshape(q, &.{ block_n, 2, 2, 16 }).to(.f32);
    const s = b.reshape(sc, &.{ block_n, 2, 2 }).to(.f32).expandDims(3);
    return q4.mul(s);
}

/// Dequantizes one 256-element Q6_K block per output row into `[block_n, 256]` f32.
fn dequantQ6K(
    b: *Builder,
    ql0: Value,
    ql1: Value,
    qh: Value,
    sc1: Value,
    sc2: Value,
    sc3: Value,
    sc4: Value,
    d: Value,
    block_n: i64,
    comptime bf16_out: bool,
) Value {
    const shape = [_]i64{ block_n, 2, 32 };
    const ql0_3 = b.reshape(ql0, &shape).to(.i32);
    const ql1_3 = b.reshape(ql1, &shape).to(.i32);
    const qh_3 = b.reshape(qh, &shape).to(.i32);

    const mask4 = b.splat(@as(i32, 0x0F), &shape);
    const lo0 = ql0_3.bitAnd(mask4);
    const lo1 = ql1_3.bitAnd(mask4);
    const hi0 = b.shrsi(ql0_3, b.splat(@as(i32, 4), &shape)).bitAnd(mask4);
    const hi1 = b.shrsi(ql1_3, b.splat(@as(i32, 4), &shape)).bitAnd(mask4);

    const q1 = sixBit(b, lo0, qh_3, 0, &shape);
    const q2 = sixBit(b, lo1, qh_3, 2, &shape);
    const q3 = sixBit(b, hi0, qh_3, 4, &shape);
    const q4 = sixBit(b, hi1, qh_3, 6, &shape);

    const w1 = b.reshape(scaleChunk(b, q1, sc1, block_n), &shape);
    const w2 = b.reshape(scaleChunk(b, q2, sc2, block_n), &shape);
    const w3 = b.reshape(scaleChunk(b, q3, sc3, block_n), &shape);
    const w4 = b.reshape(scaleChunk(b, q4, sc4, block_n), &shape);

    const w13 = b.join(w1, w3);
    const w24 = b.join(w2, w4);
    const stacked = b.join(w13, w24);
    const w = b.reshape(stacked, &.{ block_n, 256 });
    const scaled = w.mul(d.to(.f32).expandDims(1));
    // Round to the dot operand type before the reshape moves the tile.
    return if (bf16_out) scaled.to(.bf16) else scaled;
}

/// Maps the kernel's `[half, l, q]` element order back to the GGML `[half, q, l]`
/// order, so the weight can be built with plain joins and the activation row is
/// gathered into the matching order.
fn xIndex(b: *Builder) Value {
    const shape = [_]i64{256};
    const j = b.arange(0, 256, .i32);
    const h = b.shrsi(j, b.splat(@as(i32, 7), &shape));
    const rem = j.bitAnd(b.splat(@as(i32, 127), &shape));
    const l = b.shrsi(rem, b.splat(@as(i32, 2), &shape));
    const q = rem.bitAnd(b.splat(@as(i32, 3), &shape));
    return h.mul(b.splat(@as(i32, 128), &shape))
        .add(q.mul(b.splat(@as(i32, 32), &shape)))
        .add(l)
        .to(.i64);
}

const Block = struct {
    ql0: Value,
    ql1: Value,
    qh: Value,
    sc1: Value,
    sc2: Value,
    sc3: Value,
    sc4: Value,
    d: Value,
};

fn loadPointer(
    b: *Builder,
    ptr: Value,
    stride_n: i64,
    offs_n: Value,
    k_iter: Value,
    block_bytes: i64,
    block_n: i64,
) Value {
    const off = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(stride_n), &.{ block_n, block_bytes })
        .add(b.broadcastTo(k_iter.mul(block_bytes).add(b.arange(0, block_bytes, .i64)).expandDims(0), &.{ block_n, block_bytes }));
    return b.load(ptr.addPtr(off));
}

fn loadBlock(b: *Builder, a: Args, cfg: Config, offs_n: Value, k_iter: Value, block_n: i64) Block {
    const ql0 = b.reshape(loadPointer(b, a.ql0_ptr, cfg.stride_ql_n, offs_n, k_iter, 64, block_n), &.{ block_n, 2, 32 });
    const ql1 = b.reshape(loadPointer(b, a.ql1_ptr, cfg.stride_ql_n, offs_n, k_iter, 64, block_n), &.{ block_n, 2, 32 });
    const qh = b.reshape(loadPointer(b, a.qh_ptr, cfg.stride_qh_n, offs_n, k_iter, 64, block_n), &.{ block_n, 2, 32 });
    const sc1 = b.reshape(loadPointer(b, a.sc1_ptr, cfg.stride_sc_n, offs_n, k_iter, 4, block_n), &.{ block_n, 2, 2 });
    const sc2 = b.reshape(loadPointer(b, a.sc2_ptr, cfg.stride_sc_n, offs_n, k_iter, 4, block_n), &.{ block_n, 2, 2 });
    const sc3 = b.reshape(loadPointer(b, a.sc3_ptr, cfg.stride_sc_n, offs_n, k_iter, 4, block_n), &.{ block_n, 2, 2 });
    const sc4 = b.reshape(loadPointer(b, a.sc4_ptr, cfg.stride_sc_n, offs_n, k_iter, 4, block_n), &.{ block_n, 2, 2 });
    const d_off = offs_n.to(.i64).mul(cfg.stride_d_n).add(k_iter);
    const d = b.load(a.d_ptr.addPtr(d_off));
    return .{ .ql0 = ql0, .ql1 = ql1, .qh = qh, .sc1 = sc1, .sc2 = sc2, .sc3 = sc3, .sc4 = sc4, .d = d };
}

fn runMatvec1(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try declare(b);
    const pid_n = b.programId(.x);
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));
    const block_n = cfg.block_n;

    const perm = xIndex(b);
    const acc = b.zeros(&.{block_n}, .f32);
    const num_k = @divExact(cfg.k, 256);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const x_off = k_iter.mul(256).add(perm);
        const x_val = b.load(a.x_ptr.addPtr(x_off));
        const x_bcast = b.broadcastTo(x_val.expandDims(0), &.{ block_n, 256 });

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const w = dequantQ6K(b, block.ql0, block.ql1, block.qh, block.sc1, block.sc2, block.sc3, block.sc4, block.d, block_n, false);

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

    const perm = xIndex(b);
    const acc = b.zeros(&.{ cfg.block_m, block_n }, .f32);
    const num_k = @divExact(cfg.k, 256);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];

        const offs_k = k_iter.mul(256).add(perm);
        const x_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.k), &.{ cfg.block_m, 256 });
        const x_cols = b.broadcastTo(offs_k.expandDims(0), &.{ cfg.block_m, 256 });
        const x_val = b.load(a.x_ptr.addPtr(x_rows.add(x_cols)));

        const block = loadBlock(b, a, cfg, offs_n, k_iter, block_n);
        const w = dequantQ6K(b, block.ql0, block.ql1, block.qh, block.sc1, block.sc2, block.sc3, block.sc4, block.d, block_n, true);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.n), &.{ cfg.block_m, block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64), &.{ cfg.block_m, block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}
