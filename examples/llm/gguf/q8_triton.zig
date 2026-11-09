const std = @import("std");
const zml = @import("zml");
const tri = zml.kernel.triton;
const Builder = tri.Builder;

pub const Config = struct {
    m: i64,
    n: i64,
    k: i64,
    block_m: i64,
    block_n: i64,
    block_k: i64,
    stride_xm: i64,
    stride_xk: i64,
    stride_qn: i64,
    stride_qk: i64,
    stride_sn: i64,
    stride_sk: i64,
    stride_om: i64,
    stride_on: i64,
};

pub const Q8Matvec = tri.Kernel(Config, .{
    .name = "q8_matvec",
    .inputs = &.{ "x_ptr", "q_ptr", "s_ptr" },
    .outputs = &.{"out"},
    .run = run,
});

fn run(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .q_ptr = .{ .ptr = .i8 },
        .s_ptr = .{ .ptr = .f16 },
        .out_ptr = .{ .ptr = .bf16 },
    });

    const pid_m = b.programId(.x);
    const pid_n = b.programId(.y);

    const offs_m = pid_m.mul(cfg.block_m).add(b.arange(0, cfg.block_m, .i32));
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));

    const acc = b.zeros(&.{ cfg.block_m, cfg.block_n }, .f32);

    const num_k = @divExact(cfg.k, cfg.block_k);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];
        const offs_k = k_iter.mul(cfg.block_k).add(b.arange(0, cfg.block_k, .i64));

        const x_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.stride_xm), &.{ cfg.block_m, cfg.block_k });
        const x_cols = b.broadcastTo(offs_k.expandDims(0).to(.i64).mul(cfg.stride_xk), &.{ cfg.block_m, cfg.block_k });
        const x_val = b.load(a.x_ptr.addPtr(x_rows.add(x_cols)));

        const q_rows = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qn), &.{ cfg.block_n, cfg.block_k });
        const q_cols = b.broadcastTo(offs_k.expandDims(0).to(.i64).mul(cfg.stride_qk), &.{ cfg.block_n, cfg.block_k });
        const q_val = b.load(a.q_ptr.addPtr(q_rows.add(q_cols)));

        const scale_k = @divExact(cfg.block_k, 32);
        const s_offs_k = k_iter.mul(scale_k).add(b.arange(0, scale_k, .i64));
        const s_rows = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_sn), &.{ cfg.block_n, scale_k });
        const s_cols = b.broadcastTo(s_offs_k.expandDims(0).to(.i64).mul(cfg.stride_sk), &.{ cfg.block_n, scale_k });
        const s_val = b.load(a.s_ptr.addPtr(s_rows.add(s_cols)));

        const s_bcast = b.reshape(
            b.broadcastTo(s_val.expandDims(2), &.{ cfg.block_n, scale_k, 32 }),
            &.{ cfg.block_n, cfg.block_k },
        );
        const w = q_val.to(.f32).mul(s_bcast.to(.f32)).to(.bf16);
        const w_t = b.trans(w, &.{ 1, 0 });

        const new_acc = b.dot(x_val, w_t, acc_);
        loop.yield(.{new_acc});
    }

    const out_rows = b.broadcastTo(offs_m.expandDims(1).to(.i64).mul(cfg.stride_om), &.{ cfg.block_m, cfg.block_n });
    const out_cols = b.broadcastTo(offs_n.expandDims(0).to(.i64).mul(cfg.stride_on), &.{ cfg.block_m, cfg.block_n });
    b.store(a.out_ptr.addPtr(out_rows.add(out_cols)), loop.results[0].to(.bf16));
}

/// Batch-1 decode kernel: one program per output tile, reduction over K.
pub const Q8Matvec1 = tri.Kernel(Config, .{
    .name = "q8_matvec1",
    .inputs = &.{ "x_ptr", "q_ptr", "s_ptr" },
    .outputs = &.{"out"},
    .run = runMatvec1,
});

fn runMatvec1(b: *Builder, cfg: Config) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .x_ptr = .{ .ptr = .bf16 },
        .q_ptr = .{ .ptr = .i8 },
        .s_ptr = .{ .ptr = .f16 },
        .out_ptr = .{ .ptr = .bf16 },
    });

    const pid_n = b.programId(.x);
    const offs_n = pid_n.mul(cfg.block_n).add(b.arange(0, cfg.block_n, .i32));

    const acc = b.zeros(&.{cfg.block_n}, .f32);
    const num_k = @divExact(cfg.k, cfg.block_k);
    var loop = b.openFor(0, num_k, 1, .{acc});
    {
        const k_iter = loop.iv;
        const acc_ = loop.carried[0];
        const offs_k = k_iter.mul(cfg.block_k).add(b.arange(0, cfg.block_k, .i64));

        const x_val = b.load(a.x_ptr.addPtr(offs_k));
        const x_bcast = b.broadcastTo(x_val.expandDims(0), &.{ cfg.block_n, cfg.block_k });

        const q_rows = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_qn), &.{ cfg.block_n, cfg.block_k });
        const q_cols = b.broadcastTo(offs_k.expandDims(0).to(.i64).mul(cfg.stride_qk), &.{ cfg.block_n, cfg.block_k });
        const q_val = b.load(a.q_ptr.addPtr(q_rows.add(q_cols)));

        const scale_k = @divExact(cfg.block_k, 32);
        const s_offs_k = k_iter.mul(scale_k).add(b.arange(0, scale_k, .i64));
        const s_rows = b.broadcastTo(offs_n.expandDims(1).to(.i64).mul(cfg.stride_sn), &.{ cfg.block_n, scale_k });
        const s_cols = b.broadcastTo(s_offs_k.expandDims(0).to(.i64).mul(cfg.stride_sk), &.{ cfg.block_n, scale_k });
        const s_val = b.load(a.s_ptr.addPtr(s_rows.add(s_cols)));

        const s_bcast = b.reshape(
            b.broadcastTo(s_val.expandDims(2), &.{ cfg.block_n, scale_k, 32 }),
            &.{ cfg.block_n, cfg.block_k },
        );
        const w = q_val.to(.f32).mul(s_bcast.to(.f32));
        const prod = w.mul(x_bcast.to(.f32));
        const new_acc = acc_.add(b.sumOpts(prod, .{ .axis = 1 }));
        loop.yield(.{new_acc});
    }

    const out_off = offs_n.to(.i64).mul(cfg.stride_on);
    b.store(a.out_ptr.addPtr(out_off), loop.results[0].to(.bf16));
}
