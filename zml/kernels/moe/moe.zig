const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
pub const cute = @import("cute/cute.zig");
const ExpertsLayout = @import("options.zig").ExpertsLayout;
pub const flashinfer_cutlass = @import("flashinfer_cutlass/flashinfer_cutlass.zig");
pub const fly = @import("fly/fly.zig");
pub const metal = @import("metal/metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu/mosaic_tpu.zig");
const Options = @import("options.zig").Options;
pub const stablehlo = @import("stablehlo/stablehlo.zig");
pub const tcl = @import("tcl/tcl.zig");
pub const triton = @import("triton/triton.zig");

const log = std.log.scoped(.@"zml/moe");

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}

pub const Backend = enum {
    stablehlo,
    cute,
    flashinfer_cutlass,
    triton,
    fly,
    mosaic_tpu,
    metal,
    tcl,

    pub fn auto(platform: *const zml.Platform, scheme: ?zml.Quantization.Scheme, dtype: zml.DataType) !Backend {
        // Keep the dtype as an argument because non scheme-specific backends may depend on it later
        _ = dtype;
        return switch (platform.target) {
            .cuda => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (cute.isAvailable(platform))
                        .cute
                    else
                        .triton,
                    .nvfp4 => if (cute.isNvfp4Supported(platform))
                        .cute
                    else if (flashinfer_cutlass.isNvfp4Supported(platform))
                        .flashinfer_cutlass
                    else
                        error.UnsupportedQuantization,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                };
            },
            .rocm => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (zml.platform.rocm.computeCapability(platform) == .gfx942) .fly else .triton,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                    .nvfp4 => error.UnsupportedQuantization,
                };
            },
            .oneapi => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => .triton,
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => error.UnsupportedQuantization,
                };
            },
            .tpu => if (scheme == null) .mosaic_tpu else error.UnsupportedQuantization,
            .furiosa => if (scheme == null) .tcl else error.UnsupportedQuantization,
            .metal => b: {
                const s = scheme orelse break :b .metal;
                break :b switch (s) {
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .metal,
                    .mxfp4 => error.UnsupportedQuantization,
                };
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .stablehlo => true,
            .cute => cute.isAvailable(platform),
            .flashinfer_cutlass => flashinfer_cutlass.isAvailable(platform),
            .fly => switch (platform.target) {
                .rocm => zml.platform.rocm.computeCapability(platform) == .gfx942,
                else => false,
            },
            .triton => switch (platform.target) {
                .cuda, .rocm, .oneapi => true,
                else => false,
            },
            .mosaic_tpu => platform.target == .tpu,
            .metal => platform.target == .metal,
            .tcl => platform.target == .furiosa,
        };
    }

    /// The layout contract the expert weights must have for a given backend with scheme
    pub fn expertsLayout(backend: Backend, scheme: ?zml.Quantization.Scheme) !ExpertsLayout {
        return switch (backend) {
            .stablehlo => .{ .gate_up = .concatenated, .packing = .plain },
            .cute => if (scheme == .mxfp4 or scheme == .nvfp4) .{ .gate_up = .interleaved, .packing = .swizzled_scales } else error.UnsupportedQuantization,
            .flashinfer_cutlass => if (scheme) |s| switch (s) {
                .nvfp4 => .{ .gate_up = .concatenated, .packing = .flashinfer_nvfp4 },
                else => error.UnsupportedQuantization,
            } else error.UnsupportedQuantization,
            .mosaic_tpu, .metal, .tcl => .{ .gate_up = .concatenated, .packing = .plain },
            .triton, .fly => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else .{ .gate_up = .concatenated, .packing = .plain },
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .stablehlo => {},
            .cute => {},
            .flashinfer_cutlass => flashinfer_cutlass.register(platform),
            .triton, .fly => {},
            .mosaic_tpu => {},
            .metal => {},
            .tcl => {},
        };
    }
};

/// Check the shared token-wise contract for activations and routing tensors.
pub fn validateRoutingInputs(input: zml.Tensor, ids: zml.Tensor, weights: zml.Tensor) void {
    stdx.debug.assert(input.rank() == 2 and ids.rank() == 2 and weights.rank() == 2, "MoE expects input {{.t, .d}} and routing {{.t, .topk}}, got {f}, {f}, {f}", .{ input.shape(), ids.shape(), weights.shape() });
    stdx.debug.assert(input.dim(.t) == ids.dim(.t) and input.dim(.t) == weights.dim(.t), "MoE input and routing token counts must match", .{});
    stdx.debug.assert(ids.dim(.topk) == weights.dim(.topk), "MoE routing ids and weights must have matching top-k dimensions", .{});
    stdx.debug.assert(input.shape().hasTag(.t) == 0 and input.shape().hasTag(.d) == 1 and
        ids.shape().hasTag(.t) == 0 and ids.shape().hasTag(.topk) == 1 and
        weights.shape().hasTag(.t) == 0 and weights.shape().hasTag(.topk) == 1, "MoE expects ordered tags {{.t, .d}} for input and {{.t, .topk}} for routing", .{});
}

/// Input has shape {.t, .d}; routing IDs and weights have shape {.t, .topk}.
/// Returns {.t, .d}. Flatten batched input with merge(.{ .t = .{ .b, .s } }).
pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    backend: Backend,
    opts: Options,
) zml.Tensor {
    validateRoutingInputs(input, topk_ids, topk_weights);
    return switch (backend) {
        .stablehlo => stablehlo.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .cute => cute.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .flashinfer_cutlass => flashinfer_cutlass.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .triton => triton.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .fly => fly.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .mosaic_tpu => mosaic_tpu.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .metal => metal.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .tcl => tcl.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
    };
}
