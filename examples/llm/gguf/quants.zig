//! Dequantization of the GGML/GGUF block formats used by the Qwen3.8 GGUF.
//!
//! The implementations mirror `ggml-quants.c` and write float32 values in the
//! same order as the GGML reference implementations. The caller is responsible
//! for converting the result to the model dtype.

const std = @import("std");

pub const Quant = enum(u32) {
    f32 = 0,
    f16 = 1,
    q8_0 = 8,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq4_nl = 20,
    iq3_s = 21,
    iq4_xs = 23,
    bf16 = 30,

    pub fn fromId(id: u32) !Quant {
        return std.enums.fromInt(Quant, id) orelse error.UnsupportedQuant;
    }

    pub fn blockSize(self: Quant) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q8_0 => 34,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .iq4_nl => 18,
            .iq3_s => 110,
            .iq4_xs => 136,
        };
    }

    pub fn elementsPerBlock(self: Quant) usize {
        return switch (self) {
            .f32, .f16, .bf16 => 1,
            .q8_0, .iq4_nl => 32,
            .q3_k, .q4_k, .q5_k, .q6_k, .iq3_s, .iq4_xs => 256,
        };
    }
};

inline fn readU16(b: []const u8, off: usize) u16 {
    return @as(u16, b[off]) | (@as(u16, b[off + 1]) << 8);
}

inline fn readU32(b: []const u8, off: usize) u32 {
    return @as(u32, b[off]) |
        (@as(u32, b[off + 1]) << 8) |
        (@as(u32, b[off + 2]) << 16) |
        (@as(u32, b[off + 3]) << 24);
}

inline fn f16ToF32(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

inline fn bf16ToF32(bits: u16) f32 {
    const bits32: u32 = @as(u32, bits) << 16;
    return @bitCast(bits32);
}

/// Dequantizes exactly one block into `out`.
/// `out.len` must be `quant.elementsPerBlock()`.
pub fn dequantBlock(quant: Quant, block: []const u8, out: []f32) void {
    switch (quant) {
        inline else => |q| dequantBlockKind(q, block, out),
    }
}

/// Comptime-specialized version of `dequantBlock`. Use this in hot loops so
/// the compiler can inline and vectorize the selected format.
pub fn dequantBlockKind(comptime quant: Quant, block: []const u8, out: []f32) void {
    std.debug.assert(out.len == quant.elementsPerBlock());
    std.debug.assert(block.len == quant.blockSize());

    switch (quant) {
        .f32 => out[0] = @bitCast(readU32(block, 0)),
        .f16 => out[0] = f16ToF32(readU16(block, 0)),
        .bf16 => out[0] = bf16ToF32(readU16(block, 0)),
        .q8_0 => dequantQ8_0(block, out),
        .q3_k => dequantQ3K(block, out),
        .q4_k => dequantQ4K(block, out),
        .q5_k => dequantQ5K(block, out),
        .q6_k => dequantQ6K(block, out),
        .iq4_nl => dequantIq4Nl(block, out),
        .iq3_s => dequantIq3S(block, out),
        .iq4_xs => dequantIq4Xs(block, out),
    }
}

fn dequantQ8_0(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    for (0..32) |j| {
        const q: i8 = @bitCast(block[2 + j]);
        out[j] = d * @as(f32, @floatFromInt(q));
    }
}

fn dequantQ3K(block: []const u8, out: []f32) void {
    const d_all = f16ToF32(readU16(block, 108));

    const hm = block[0..32];
    const qs = block[32..96];

    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;

    var aux: [4]u32 = .{ 0, 0, 0, 0 };
    for (0..3) |i| aux[i] = readU32(block, 96 + i * 4);

    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    var scales: [16]i8 = undefined;
    for (0..4) |word| {
        const v = aux[word];
        inline for (0..4) |byte| {
            scales[word * 4 + byte] = @bitCast(@as(u8, @truncate(v >> (8 * byte))));
        }
    }

    var y: usize = 0;
    var is: usize = 0;
    var m: u8 = 1;
    var q_off: usize = 0;

    var n: usize = 0;
    while (n < 256) : (n += 128) {
        var shift: usize = 0;
        for (0..4) |_| {
            const dl1 = d_all * (@as(f32, @floatFromInt(scales[is])) - 32.0);
            is += 1;
            for (0..16) |l| {
                const qv: i32 = @intCast((qs[q_off + l] >> @intCast(shift)) & 3);
                const hb: i32 = if ((hm[l] & m) != 0) 0 else 4;
                out[y] = dl1 * @as(f32, @floatFromInt(qv - hb));
                y += 1;
            }

            const dl2 = d_all * (@as(f32, @floatFromInt(scales[is])) - 32.0);
            is += 1;
            for (0..16) |l| {
                const qv: i32 = @intCast((qs[q_off + l + 16] >> @intCast(shift)) & 3);
                const hb: i32 = if ((hm[l + 16] & m) != 0) 0 else 4;
                out[y] = dl2 * @as(f32, @floatFromInt(qv - hb));
                y += 1;
            }

            shift += 2;
            m *%= 2;
        }
        q_off += 32;
    }
}

inline fn getScaleMinK4(j: usize, q: []const u8, sc: *u8, m: *u8) void {
    if (j < 4) {
        sc.* = q[j] & 63;
        m.* = q[j + 4] & 63;
    } else {
        sc.* = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        m.* = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

fn dequantQ4K(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    const dmin = f16ToF32(readU16(block, 2));
    const scales = block[4..16];
    const qs = block[16..144];

    var y: usize = 0;
    var is: usize = 0;
    var q_off: usize = 0;

    var j: usize = 0;
    while (j < 256) : (j += 64) {
        var sc: u8 = undefined;
        var m: u8 = undefined;

        getScaleMinK4(is + 0, scales, &sc, &m);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(m));

        getScaleMinK4(is + 1, scales, &sc, &m);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(m));

        for (0..32) |l| {
            out[y] = d1 * @as(f32, @floatFromInt(qs[q_off + l] & 0x0f)) - m1;
            y += 1;
        }
        for (0..32) |l| {
            out[y] = d2 * @as(f32, @floatFromInt(qs[q_off + l] >> 4)) - m2;
            y += 1;
        }

        q_off += 32;
        is += 2;
    }
}

fn dequantQ5K(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    const dmin = f16ToF32(readU16(block, 2));
    const scales = block[4..16];
    const qh = block[16..48];
    const ql = block[48..176];

    var y: usize = 0;
    var is: usize = 0;
    var q_off: usize = 0;
    var bit1: u8 = 1;
    var bit2: u8 = 2;

    var j: usize = 0;
    while (j < 256) : (j += 64) {
        var sc: u8 = undefined;
        var m: u8 = undefined;

        getScaleMinK4(is + 0, scales, &sc, &m);
        const d1 = d * @as(f32, @floatFromInt(sc));
        const m1 = dmin * @as(f32, @floatFromInt(m));

        getScaleMinK4(is + 1, scales, &sc, &m);
        const d2 = d * @as(f32, @floatFromInt(sc));
        const m2 = dmin * @as(f32, @floatFromInt(m));

        for (0..32) |l| {
            const hi: u8 = if ((qh[l] & bit1) != 0) 16 else 0;
            out[y] = d1 * @as(f32, @floatFromInt((ql[q_off + l] & 0x0f) + hi)) - m1;
            y += 1;
        }
        for (0..32) |l| {
            const hi: u8 = if ((qh[l] & bit2) != 0) 16 else 0;
            out[y] = d2 * @as(f32, @floatFromInt((ql[q_off + l] >> 4) + hi)) - m2;
            y += 1;
        }

        q_off += 32;
        is += 2;
        bit1 *%= 4;
        bit2 *%= 4;
    }
}

fn dequantQ6K(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 208));

    const ql = block[0..128];
    const qh = block[128..192];
    const sc = block[192..208];

    var y: usize = 0;
    var ql_off: usize = 0;
    var qh_off: usize = 0;
    var sc_off: usize = 0;

    var n: usize = 0;
    while (n < 256) : (n += 128) {
        for (0..32) |l| {
            const is: usize = l / 16;

            const q1: i32 = @as(i32, @intCast((ql[ql_off + l] & 0x0f) | (((qh[qh_off + l] >> 0) & 3) << 4))) - 32;
            const q2: i32 = @as(i32, @intCast((ql[ql_off + l + 32] & 0x0f) | (((qh[qh_off + l] >> 2) & 3) << 4))) - 32;
            const q3: i32 = @as(i32, @intCast((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4))) - 32;
            const q4: i32 = @as(i32, @intCast((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4))) - 32;

            out[y + l + 0] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 0])))) * @as(f32, @floatFromInt(q1));
            out[y + l + 32] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 2])))) * @as(f32, @floatFromInt(q2));
            out[y + l + 64] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 4])))) * @as(f32, @floatFromInt(q3));
            out[y + l + 96] = d * @as(f32, @floatFromInt(@as(i8, @bitCast(sc[sc_off + is + 6])))) * @as(f32, @floatFromInt(q4));
        }

        y += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}

fn dequantIq4Nl(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    const qs = block[2..18];

    for (0..16) |j| {
        out[j + 0] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] & 0x0f]));
        out[j + 16] = d * @as(f32, @floatFromInt(kvalues_iq4nl[qs[j] >> 4]));
    }
}

fn dequantIq4Xs(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    const scales_h = readU16(block, 2);
    const scales_l = block[4..8];
    const qs = block[8..136];

    var y: usize = 0;
    var q_off: usize = 0;

    for (0..8) |ib| {
        const ls: u32 = (@as(u32, scales_l[ib / 2] >> @intCast(4 * (ib % 2))) & 0x0f) |
            (((@as(u32, scales_h) >> @intCast(2 * ib)) & 3) << 4);
        const dl = d * (@as(f32, @floatFromInt(ls)) - 32.0);

        for (0..16) |j| {
            out[y + j + 0] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[q_off + j] & 0x0f]));
            out[y + j + 16] = dl * @as(f32, @floatFromInt(kvalues_iq4nl[qs[q_off + j] >> 4]));
        }

        y += 32;
        q_off += 16;
    }
}

fn dequantIq3S(block: []const u8, out: []f32) void {
    const d = f16ToF32(readU16(block, 0));
    const qs = block[2..66];
    const qh = block[66..74];
    const signs = block[74..106];
    const scales = block[106..110];

    var y: usize = 0;
    var q_off: usize = 0;
    var qh_off: usize = 0;
    var sign_off: usize = 0;

    var ib: usize = 0;
    while (ib < 8) : (ib += 2) {
        const db1 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(scales[ib / 2] & 0x0f)));
        const db2 = d * (1.0 + 2.0 * @as(f32, @floatFromInt(scales[ib / 2] >> 4)));

        for (0..4) |l| {
            const idx1: u32 = @as(u32, qs[q_off + 2 * l + 0]) | ((@as(u32, qh[qh_off + 0]) << @intCast(8 - 2 * l)) & 256);
            const idx2: u32 = @as(u32, qs[q_off + 2 * l + 1]) | ((@as(u32, qh[qh_off + 0]) << @intCast(7 - 2 * l)) & 256);
            const grid1 = iq3s_grid[idx1];
            const grid2 = iq3s_grid[idx2];

            for (0..4) |j| {
                const g1: u8 = @truncate(grid1 >> @intCast(8 * j));
                const g2: u8 = @truncate(grid2 >> @intCast(8 * j));
                const s1: f32 = if ((signs[sign_off + l] & kmask_iq2xs[j + 0]) != 0) -1.0 else 1.0;
                const s2: f32 = if ((signs[sign_off + l] & kmask_iq2xs[j + 4]) != 0) -1.0 else 1.0;
                out[y + j + 0] = db1 * @as(f32, @floatFromInt(g1)) * s1;
                out[y + j + 4] = db1 * @as(f32, @floatFromInt(g2)) * s2;
            }
            y += 8;
        }

        q_off += 8;
        sign_off += 4;

        for (0..4) |l| {
            const idx1: u32 = @as(u32, qs[q_off + 2 * l + 0]) | ((@as(u32, qh[qh_off + 1]) << @intCast(8 - 2 * l)) & 256);
            const idx2: u32 = @as(u32, qs[q_off + 2 * l + 1]) | ((@as(u32, qh[qh_off + 1]) << @intCast(7 - 2 * l)) & 256);
            const grid1 = iq3s_grid[idx1];
            const grid2 = iq3s_grid[idx2];

            for (0..4) |j| {
                const g1: u8 = @truncate(grid1 >> @intCast(8 * j));
                const g2: u8 = @truncate(grid2 >> @intCast(8 * j));
                const s1: f32 = if ((signs[sign_off + l] & kmask_iq2xs[j + 0]) != 0) -1.0 else 1.0;
                const s2: f32 = if ((signs[sign_off + l] & kmask_iq2xs[j + 4]) != 0) -1.0 else 1.0;
                out[y + j + 0] = db2 * @as(f32, @floatFromInt(g1)) * s1;
                out[y + j + 4] = db2 * @as(f32, @floatFromInt(g2)) * s2;
            }
            y += 8;
        }

        qh_off += 2;
        q_off += 8;
        sign_off += 4;
    }
}

const kvalues_iq4nl = [16]i8{
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

const kmask_iq2xs = [8]u8{ 1, 2, 4, 8, 16, 32, 64, 128 };

const iq3s_grid = [512]u32{
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};
