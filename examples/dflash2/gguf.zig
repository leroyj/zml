//! Minimal GGUF v2/v3 reader for the Qwen3.8 GGUF checkpoints.
//!
//! The module parses the GGUF metadata and tensor table, maps the GGUF tensor
//! names to the Hugging Face names used by the `qwen3_5` model, and exposes a
//! `zml.safetensors.TensorRegistry` whose tensors are dequantized on the fly to
//! bfloat16 while they are streamed to the device.
//!
//! It also materializes a `config.json` and `tokenizer.json` derived from the
//! GGUF metadata into a small cache directory, so the existing model loading
//! path can be reused without creating a dequantized weight file.

const std = @import("std");

const zml = @import("zml");

const quants = @import("gguf/quants.zig");

const log = std.log.scoped(.gguf);

const GGUF_MAGIC = "GGUF";

pub const Value = union(enum) {
    u8: u8,
    i8: i8,
    u16: u16,
    i16: i16,
    u32: u32,
    i32: i32,
    f32: f32,
    bool: bool,
    string: []const u8,
    array: []const Value,
    u64: u64,
    i64: i64,
    f64: f64,
};

pub const TensorInfo = struct {
    name: []const u8,
    dims: []const u64,
    quant: quants.Quant,
    offset: u64,
    numel: u64,
};

pub const Context = struct {
    arena: std.heap.ArenaAllocator,
    mapped: []align(std.heap.page_size_min) const u8,
    file_uri: []const u8,
    file_size: u64,
    metadata: std.StringArrayHashMapUnmanaged(Value),
    tensors: []TensorInfo,
    data_offset: u64,
    num_layers: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    head_k_dim: usize,
    head_v_dim: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !Context {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const arena_allocator = arena.allocator();

        const real_path = try std.Io.Dir.cwd().realPathFileAlloc(io, path, allocator);
        defer allocator.free(real_path);
        const file_uri = try arena_allocator.dupe(u8, real_path);

        const file = try std.Io.Dir.openFileAbsolute(io, file_uri, .{ .mode = .read_only });
        defer file.close(io);

        const stat = try file.stat(io);
        if (stat.size < 4) return error.InvalidGgufMagic;

        const mapped = try std.posix.mmap(
            null,
            @intCast(stat.size),
            .{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mapped);

        var cursor: Cursor = .{ .data = mapped };

        if (!std.mem.eql(u8, try cursor.readBytes(4), GGUF_MAGIC)) return error.InvalidGgufMagic;

        const version = try cursor.readInt(u32);
        if (version < 2 or version > 3) return error.UnsupportedGgufVersion;

        const tensor_count = try cursor.readInt(u64);
        const kv_count = try cursor.readInt(u64);

        var metadata: std.StringArrayHashMapUnmanaged(Value) = .empty;
        for (0..@intCast(kv_count)) |_| {
            const key = try cursor.readString();
            const value = try readValue(arena_allocator, &cursor);
            try metadata.put(arena_allocator, key, value);
        }

        const tensors = try arena_allocator.alloc(TensorInfo, @intCast(tensor_count));
        for (tensors) |*tensor| {
            const name = try cursor.readString();
            const n_dims = try cursor.readInt(u32);
            if (n_dims > 4) return error.TooManyDimensions;

            const dims = try arena_allocator.alloc(u64, n_dims);
            for (dims) |*dim| dim.* = try cursor.readInt(u64);

            const quant_id = try cursor.readInt(u32);
            const quant = try quants.Quant.fromId(quant_id);
            const offset = try cursor.readInt(u64);

            var numel: u64 = 1;
            for (dims) |dim| numel *= dim;

            tensor.* = .{
                .name = name,
                .dims = dims,
                .quant = quant,
                .offset = offset,
                .numel = numel,
            };
        }

        const alignment = blk: {
            if (metadata.get("general.alignment")) |value| {
                break :blk @as(u64, @intCast(intValue(value) orelse 32));
            }
            break :blk 32;
        };
        if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;

        const data_offset = std.mem.alignForward(u64, cursor.pos, alignment);

        const block_count = metadataInt(metadata, "dflash.block_count") orelse 0;

        return .{
            .arena = arena,
            .mapped = mapped,
            .file_uri = file_uri,
            .file_size = stat.size,
            .metadata = metadata,
            .tensors = tensors,
            .data_offset = data_offset,
            .num_layers = @intCast(block_count),
            .num_k_heads = 0,
            .num_v_per_k = 1,
            .head_k_dim = 0,
            .head_v_dim = 0,
        };
    }

    pub fn deinit(self: *Context) void {
        std.posix.munmap(self.mapped);
        self.arena.deinit();
    }

    pub fn get(self: *const Context, key: []const u8) ?Value {
        return self.metadata.get(key);
    }

    pub fn getInt(self: *const Context, key: []const u8) !i64 {
        const value = self.metadata.get(key) orelse return error.MissingMetadata;
        return intValue(value) orelse error.InvalidMetadataType;
    }

    pub fn getFloat(self: *const Context, key: []const u8) !f64 {
        const value = self.metadata.get(key) orelse return error.MissingMetadata;
        return floatValue(value) orelse error.InvalidMetadataType;
    }

    pub fn getString(self: *const Context, key: []const u8) ![]const u8 {
        const value = self.metadata.get(key) orelse return error.MissingMetadata;
        return switch (value) {
            .string => |s| s,
            else => error.InvalidMetadataType,
        };
    }

    pub fn getArray(self: *const Context, key: []const u8) ![]const Value {
        const value = self.metadata.get(key) orelse return error.MissingMetadata;
        return switch (value) {
            .array => |a| a,
            else => error.InvalidMetadataType,
        };
    }

    pub fn getIntArray(self: *Context, key: []const u8) ![]const i64 {
        const values = try self.getArray(key);
        const result = try self.arena.allocator().alloc(i64, values.len);
        for (values, 0..) |value, i| {
            result[i] = intValue(value) orelse return error.InvalidMetadataType;
        }
        return result;
    }

    pub fn getStringArray(self: *Context, key: []const u8) ![]const []const u8 {
        const values = try self.getArray(key);
        const result = try self.arena.allocator().alloc([]const u8, values.len);
        for (values, 0..) |value, i| {
            result[i] = switch (value) {
                .string => |s| s,
                else => return error.InvalidMetadataType,
            };
        }
        return result;
    }

    /// Builds a registry whose tensors are dequantized to bf16 on the fly.
    pub fn registry(self: *Context, allocator: std.mem.Allocator) !zml.safetensors.TensorRegistry {
        var result: zml.safetensors.TensorRegistry = .init(allocator);
        errdefer result.deinit();

        const arena_allocator = self.arena.allocator();

        for (self.tensors) |tensor| {
            const hf_name = try mapDflashName(arena_allocator, tensor.name) orelse continue;

            var dims_buffer: [4]i64 = undefined;
            const dims = dflashDims(tensor, &dims_buffer) orelse continue;

            const rows: usize = if (tensor.dims.len == 1) @intCast(tensor.dims[0]) else @intCast(tensor.dims[1]);
            const row_len: usize = if (tensor.dims.len == 1) 1 else @intCast(tensor.dims[0]);

            const ctx = try arena_allocator.create(DequantCtx);
            ctx.* = .{
                .data = self.mapped,
                .quant = tensor.quant,
                .offset = @intCast(self.data_offset + tensor.offset),
                .numel = @intCast(tensor.numel),
                .rows = rows,
                .row_len = row_len,
                .permute = .none,
                .perm_offset = 0,
                .head_dim = 0,
                .num_k_heads = 0,
                .num_v_per_k = 1,
                .transform = .none,
                .output = .bf16,
            };

            try result.registerTensor(.{
                .file_uri = self.file_uri,
                .name = hf_name,
                .shape = .init(dims, .bf16),
                .offset = self.data_offset + tensor.offset,
                .custom = .{
                    .ctx = ctx,
                    .start = DequantCtx.start,
                    .fill = DequantCtx.fill,
                    .stop = DequantCtx.stop,
                },
            });
        }

        return result;
    }

    pub fn ensureCacheDir(
        self: *Context,
        allocator: std.mem.Allocator,
        io: std.Io,
        environ_map: *const std.process.Environ.Map,
    ) ![]u8 {
        const cache_root = blk: {
            if (environ_map.get("XDG_CACHE_HOME")) |root| break :blk try allocator.dupe(u8, root);
            if (environ_map.get("HOME")) |home| break :blk try std.Io.Dir.path.join(allocator, &.{ home, ".cache" });
            break :blk try allocator.dupe(u8, "/tmp");
        };
        defer allocator.free(cache_root);

        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.file_uri);
        hasher.update(std.mem.asBytes(&self.file_size));
        const digest = hasher.final();

        var hash_buffer: [16]u8 = undefined;
        const hash = try std.fmt.bufPrint(&hash_buffer, "{x:0>16}", .{digest});

        const dir_path = try std.Io.Dir.path.join(allocator, &.{ cache_root, "zml", "gguf", hash });
        errdefer allocator.free(dir_path);

        try std.Io.Dir.cwd().createDirPath(io, dir_path);

        var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
        defer dir.close(io);

        if (dir.openFile(io, "config.json", .{ .mode = .read_only })) |file| {
            file.close(io);
        } else |_| {
            try self.writeConfig(io, dir);
        }

        if (dir.openFile(io, "tokenizer.json", .{ .mode = .read_only })) |file| {
            file.close(io);
        } else |_| {
            try self.writeTokenizer(io, dir);
        }

        return dir_path;
    }

    fn writeConfig(self: *Context, io: std.Io, dir: std.Io.Dir) !void {
        const file = try dir.createFile(io, "config.json", .{ .truncate = true });
        defer file.close(io);

        var buffer: [64 * 1024]u8 = undefined;
        var file_writer = file.writer(io, &buffer);
        defer file_writer.interface.flush() catch {};

        var stringify: std.json.Stringify = .{
            .writer = &file_writer.interface,
            .options = .{ .whitespace = .indent_2 },
        };

        const block_count: i64 = @intCast(self.num_layers);
        const hidden_size = try self.getInt("qwen35.embedding_length");
        const num_heads = try self.getInt("qwen35.attention.head_count");
        const num_kv_heads = try self.getInt("qwen35.attention.head_count_kv");
        const head_dim = try self.getInt("qwen35.attention.key_length");
        const rms_eps = try self.getFloat("qwen35.attention.layer_norm_rms_epsilon");
        const context_length = try self.getInt("qwen35.context_length");
        const rope_theta = try self.getFloat("qwen35.rope.freq_base");
        const rope_dim = try self.getInt("qwen35.rope.dimension_count");
        const conv_kernel = try self.getInt("qwen35.ssm.conv_kernel");
        const state_size = try self.getInt("qwen35.ssm.state_size");
        const group_count = try self.getInt("qwen35.ssm.group_count");
        const time_step_rank = try self.getInt("qwen35.ssm.time_step_rank");
        const inner_size = try self.getInt("qwen35.ssm.inner_size");
        const full_attention_interval = try self.getInt("qwen35.full_attention_interval");

        const mrope_section = try self.getIntArray("qwen35.rope.dimension_sections");
        const partial_rotary_factor = @as(f64, @floatFromInt(rope_dim)) / @as(f64, @floatFromInt(head_dim));
        const linear_value_head_dim = @divExact(inner_size, time_step_rank);

        try stringify.beginObject();
        try stringify.objectField("model_type");
        try stringify.write("qwen3_5");
        try stringify.objectField("text_config");
        try stringify.beginObject();

        try stringify.objectField("num_hidden_layers");
        try stringify.write(block_count);

        try stringify.objectField("layer_types");
        try stringify.beginArray();
        for (0..@intCast(block_count)) |i| {
            const is_full = (i + 1) % @as(usize, @intCast(full_attention_interval)) == 0;
            try stringify.write(if (is_full) "full_attention" else "linear_attention");
        }
        try stringify.endArray();

        try stringify.objectField("hidden_size");
        try stringify.write(hidden_size);
        try stringify.objectField("max_position_embeddings");
        try stringify.write(context_length);
        try stringify.objectField("rms_norm_eps");
        try stringify.write(rms_eps);
        try stringify.objectField("head_dim");
        try stringify.write(head_dim);
        try stringify.objectField("num_attention_heads");
        try stringify.write(num_heads);
        try stringify.objectField("num_key_value_heads");
        try stringify.write(num_kv_heads);

        try stringify.objectField("rope_parameters");
        try stringify.beginObject();
        try stringify.objectField("mrope_section");
        try stringify.beginArray();
        for (mrope_section[0..@min(3, mrope_section.len)]) |section| try stringify.write(section);
        try stringify.endArray();
        try stringify.objectField("partial_rotary_factor");
        try stringify.write(partial_rotary_factor);
        try stringify.objectField("rope_theta");
        try stringify.write(rope_theta);
        try stringify.endObject();

        try stringify.objectField("linear_conv_kernel_dim");
        try stringify.write(conv_kernel);
        try stringify.objectField("linear_key_head_dim");
        try stringify.write(state_size);
        try stringify.objectField("linear_num_key_heads");
        try stringify.write(group_count);
        try stringify.objectField("linear_num_value_heads");
        try stringify.write(time_step_rank);
        try stringify.objectField("linear_value_head_dim");
        try stringify.write(linear_value_head_dim);

        try stringify.endObject();
        try stringify.endObject();

        try file_writer.interface.flush();
    }

    fn writeTokenizer(self: *Context, io: std.Io, dir: std.Io.Dir) !void {
        const tokens = try self.getStringArray("tokenizer.ggml.tokens");
        const token_types = try self.getIntArray("tokenizer.ggml.token_type");
        const merges = try self.getStringArray("tokenizer.ggml.merges");

        if (tokens.len != token_types.len) return error.InvalidTokenizerMetadata;

        const file = try dir.createFile(io, "tokenizer.json", .{ .truncate = true });
        defer file.close(io);

        var buffer: [256 * 1024]u8 = undefined;
        var file_writer = file.writer(io, &buffer);
        defer file_writer.interface.flush() catch {};

        var stringify: std.json.Stringify = .{
            .writer = &file_writer.interface,
            .options = .{},
        };

        try stringify.beginObject();

        try stringify.objectField("version");
        try stringify.write("1.0");
        try stringify.objectField("truncation");
        try stringify.write(null);
        try stringify.objectField("padding");
        try stringify.write(null);

        try stringify.objectField("added_tokens");
        try stringify.beginArray();
        for (tokens, token_types, 0..) |token, token_type, id| {
            // GGUF token types: 1 = normal, 3 = control, 4 = user defined,
            // 5 = unused (padding). Only control and user-defined tokens are
            // added tokens; unused entries are padding and are omitted.
            if (token_type != 3 and token_type != 4) continue;

            try stringify.beginObject();
            try stringify.objectField("id");
            try stringify.write(id);
            try stringify.objectField("content");
            try stringify.write(token);
            try stringify.objectField("single_word");
            try stringify.write(false);
            try stringify.objectField("lstrip");
            try stringify.write(false);
            try stringify.objectField("rstrip");
            try stringify.write(false);
            try stringify.objectField("normalized");
            try stringify.write(false);
            try stringify.objectField("special");
            try stringify.write(token_type == 3);
            try stringify.endObject();
        }
        try stringify.endArray();

        try stringify.objectField("normalizer");
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("NFC");
        try stringify.endObject();

        try stringify.objectField("pre_tokenizer");
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("Sequence");
        try stringify.objectField("pretokenizers");
        try stringify.beginArray();
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("Split");
        try stringify.objectField("pattern");
        try stringify.beginObject();
        try stringify.objectField("Regex");
        try stringify.write(QWEN_PRETOKENIZER_REGEX);
        try stringify.endObject();
        try stringify.objectField("behavior");
        try stringify.write("Isolated");
        try stringify.objectField("invert");
        try stringify.write(false);
        try stringify.endObject();
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("ByteLevel");
        try stringify.objectField("add_prefix_space");
        try stringify.write(false);
        try stringify.objectField("trim_offsets");
        try stringify.write(false);
        try stringify.objectField("use_regex");
        try stringify.write(false);
        try stringify.endObject();
        try stringify.endArray();
        try stringify.endObject();

        try stringify.objectField("post_processor");
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("ByteLevel");
        try stringify.objectField("add_prefix_space");
        try stringify.write(false);
        try stringify.objectField("trim_offsets");
        try stringify.write(false);
        try stringify.objectField("use_regex");
        try stringify.write(false);
        try stringify.endObject();

        try stringify.objectField("decoder");
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("ByteLevel");
        try stringify.objectField("add_prefix_space");
        try stringify.write(false);
        try stringify.objectField("trim_offsets");
        try stringify.write(false);
        try stringify.objectField("use_regex");
        try stringify.write(false);
        try stringify.endObject();

        try stringify.objectField("model");
        try stringify.beginObject();
        try stringify.objectField("type");
        try stringify.write("BPE");
        try stringify.objectField("dropout");
        try stringify.write(null);
        try stringify.objectField("unk_token");
        try stringify.write(null);
        try stringify.objectField("continuing_subword_prefix");
        try stringify.write("");
        try stringify.objectField("end_of_word_suffix");
        try stringify.write("");
        try stringify.objectField("fuse_unk");
        try stringify.write(false);
        try stringify.objectField("byte_fallback");
        try stringify.write(false);
        try stringify.objectField("ignore_merges");
        try stringify.write(false);

        try stringify.objectField("vocab");
        try stringify.beginObject();
        for (tokens, token_types, 0..) |token, token_type, id| {
            if (token_type != 1) continue;
            try stringify.objectField(token);
            try stringify.write(id);
        }
        try stringify.endObject();

        try stringify.objectField("merges");
        try stringify.beginArray();
        for (merges) |merge| try stringify.write(merge);
        try stringify.endArray();

        try stringify.endObject();
        try stringify.endObject();

        try file_writer.interface.flush();
    }
};

const QWEN_PRETOKENIZER_REGEX =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const Cursor = struct {
    data: []const u8,
    pos: usize = 0,

    fn readInt(self: *Cursor, comptime T: type) !T {
        const size = @sizeOf(T);
        if (self.pos + size > self.data.len) return error.UnexpectedEndOfFile;
        const value = std.mem.readInt(T, self.data[self.pos..][0..size], .little);
        self.pos += size;
        return value;
    }

    fn readBytes(self: *Cursor, len: usize) ![]const u8 {
        if (self.pos + len > self.data.len) return error.UnexpectedEndOfFile;
        const bytes = self.data[self.pos..][0..len];
        self.pos += len;
        return bytes;
    }

    fn readString(self: *Cursor) ![]const u8 {
        const len = try self.readInt(u64);
        if (len > 1 << 30) return error.InvalidGgufString;
        return self.readBytes(@intCast(len));
    }
};

fn readValue(allocator: std.mem.Allocator, cursor: *Cursor) !Value {
    const value_type = try cursor.readInt(u32);
    return readValueOfType(allocator, cursor, value_type);
}

fn readValueOfType(allocator: std.mem.Allocator, cursor: *Cursor, value_type: u32) !Value {
    return switch (value_type) {
        0 => .{ .u8 = try cursor.readInt(u8) },
        1 => .{ .i8 = try cursor.readInt(i8) },
        2 => .{ .u16 = try cursor.readInt(u16) },
        3 => .{ .i16 = try cursor.readInt(i16) },
        4 => .{ .u32 = try cursor.readInt(u32) },
        5 => .{ .i32 = try cursor.readInt(i32) },
        6 => .{ .f32 = @bitCast(try cursor.readInt(u32)) },
        7 => .{ .bool = (try cursor.readInt(u8)) != 0 },
        8 => .{ .string = try cursor.readString() },
        9 => blk: {
            const element_type = try cursor.readInt(u32);
            const len = try cursor.readInt(u64);
            if (len > 1 << 30) return error.InvalidGgufArray;
            const values = try allocator.alloc(Value, @intCast(len));
            for (values) |*value| {
                value.* = try readValueOfType(allocator, cursor, element_type);
            }
            break :blk .{ .array = values };
        },
        10 => .{ .u64 = try cursor.readInt(u64) },
        11 => .{ .i64 = try cursor.readInt(i64) },
        12 => .{ .f64 = @bitCast(try cursor.readInt(u64)) },
        else => error.InvalidGgufValueType,
    };
}

fn metadataInt(metadata: anytype, key: []const u8) ?i64 {
    const value = metadata.get(key) orelse return null;
    return intValue(value);
}

fn intValue(value: Value) ?i64 {
    return switch (value) {
        .u8 => |x| x,
        .i8 => |x| x,
        .u16 => |x| x,
        .i16 => |x| x,
        .u32 => |x| x,
        .i32 => |x| x,
        .u64 => |x| std.math.cast(i64, x),
        .i64 => |x| x,
        else => null,
    };
}

fn floatValue(value: Value) ?f64 {
    return switch (value) {
        .f32 => |x| x,
        .f64 => |x| x,
        .u8 => |x| @floatFromInt(x),
        .i8 => |x| @floatFromInt(x),
        .u16 => |x| @floatFromInt(x),
        .i16 => |x| @floatFromInt(x),
        .u32 => |x| @floatFromInt(x),
        .i32 => |x| @floatFromInt(x),
        .u64 => |x| @floatFromInt(x),
        .i64 => |x| @floatFromInt(x),
        else => null,
    };
}

const root_tensor_map = .{
    .{ "token_embd.weight", "model.language_model.embed_tokens.weight" },
    .{ "output_norm.weight", "model.language_model.norm.weight" },
    .{ "output.weight", "lm_head.weight" },
};

const layer_tensor_map = .{
    .{ "attn_norm.weight", "input_layernorm.weight" },
    .{ "post_attention_norm.weight", "post_attention_layernorm.weight" },
    .{ "attn_q.weight", "self_attn.q_proj.weight" },
    .{ "attn_k.weight", "self_attn.k_proj.weight" },
    .{ "attn_v.weight", "self_attn.v_proj.weight" },
    .{ "attn_output.weight", "self_attn.o_proj.weight" },
    .{ "attn_q_norm.weight", "self_attn.q_norm.weight" },
    .{ "attn_k_norm.weight", "self_attn.k_norm.weight" },
    .{ "attn_qkv.weight", "linear_attn.in_proj_qkv.weight" },
    .{ "attn_gate.weight", "linear_attn.in_proj_z.weight" },
    .{ "ssm_alpha.weight", "linear_attn.in_proj_a.weight" },
    .{ "ssm_beta.weight", "linear_attn.in_proj_b.weight" },
    .{ "ssm_out.weight", "linear_attn.out_proj.weight" },
    .{ "ssm_conv1d.weight", "linear_attn.conv1d.weight" },
    .{ "ssm_dt.bias", "linear_attn.dt_bias" },
    .{ "ssm_a", "linear_attn.A_log" },
    .{ "ssm_norm.weight", "linear_attn.norm.weight" },
    .{ "ffn_gate.weight", "mlp.gate_proj.weight" },
    .{ "ffn_up.weight", "mlp.up_proj.weight" },
    .{ "ffn_down.weight", "mlp.down_proj.weight" },
};

fn mapDflashName(allocator: std.mem.Allocator, name: []const u8) !?[]const u8 {
    if (std.mem.eql(u8, name, "fc.weight")) return "fc.weight";
    if (std.mem.eql(u8, name, "enc.output_norm.weight")) return "hidden_norm.weight";
    if (std.mem.eql(u8, name, "output_norm.weight")) return "norm.weight";
    if (std.mem.eql(u8, name, "selector_hidden.weight")) return "selector_hidden.weight";
    if (std.mem.eql(u8, name, "selector_predecessor.weight")) return "selector_predecessor.weight";
    if (std.mem.eql(u8, name, "selector_successor.weight")) return "selector_successor.weight";

    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const rest = name[4..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = rest[0..dot];
    const suffix = rest[dot + 1 ..];

    const mapped_suffix: []const u8 = if (std.mem.eql(u8, suffix, "attn_norm.weight"))
        "input_layernorm.weight"
    else if (std.mem.eql(u8, suffix, "attn_q.weight"))
        "self_attn.q_proj.weight"
    else if (std.mem.eql(u8, suffix, "attn_k.weight"))
        "self_attn.k_proj.weight"
    else if (std.mem.eql(u8, suffix, "attn_v.weight"))
        "self_attn.v_proj.weight"
    else if (std.mem.eql(u8, suffix, "attn_output.weight"))
        "self_attn.o_proj.weight"
    else if (std.mem.eql(u8, suffix, "attn_q_norm.weight"))
        "self_attn.q_norm.weight"
    else if (std.mem.eql(u8, suffix, "attn_k_norm.weight"))
        "self_attn.k_norm.weight"
    else if (std.mem.eql(u8, suffix, "attn_conv_proj.weight"))
        "self_attn.attn_conv_proj.weight"
    else if (std.mem.eql(u8, suffix, "attn_conv_base"))
        "self_attn.attn_conv_base"
    else if (std.mem.eql(u8, suffix, "ffn_norm.weight"))
        "post_attention_layernorm.weight"
    else if (std.mem.eql(u8, suffix, "ffn_gate.weight"))
        "mlp.gate_proj.weight"
    else if (std.mem.eql(u8, suffix, "ffn_up.weight"))
        "mlp.up_proj.weight"
    else if (std.mem.eql(u8, suffix, "ffn_down.weight"))
        "mlp.down_proj.weight"
    else if (std.mem.eql(u8, suffix, "ffn_conv_proj.weight"))
        "ffn_conv_proj.weight"
    else if (std.mem.eql(u8, suffix, "ffn_conv_base"))
        "ffn_conv_base"
    else
        return null;

    return try std.fmt.allocPrint(allocator, "layers.{s}.{s}", .{ layer, mapped_suffix });
}

fn dflashDims(tensor: TensorInfo, buffer: *[4]i64) ?[]const i64 {
    if (tensor.dims.len > buffer.len) return null;
    if (std.mem.endsWith(u8, tensor.name, "_conv_base")) {
        for (0..tensor.dims.len) |i| buffer[i] = @intCast(tensor.dims[i]);
        return buffer[0..tensor.dims.len];
    }
    for (0..tensor.dims.len) |i| {
        buffer[i] = @intCast(tensor.dims[tensor.dims.len - 1 - i]);
    }
    return buffer[0..tensor.dims.len];
}

fn mapName(allocator: std.mem.Allocator, name: []const u8, num_layers: usize) !?[]const u8 {
    if (std.mem.startsWith(u8, name, "blk.")) {
        const rest = name["blk.".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const layer_index = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
        if (layer_index >= num_layers) return null;

        const suffix = rest[dot + 1 ..];
        inline for (layer_tensor_map) |entry| {
            if (std.mem.eql(u8, suffix, entry[0])) {
                return try std.fmt.allocPrint(allocator, "model.language_model.layers.{d}.{s}", .{ layer_index, entry[1] });
            }
        }
        return null;
    }

    inline for (root_tensor_map) |entry| {
        if (std.mem.eql(u8, name, entry[0])) {
            return try allocator.dupe(u8, entry[1]);
        }
    }

    return null;
}

fn hfDims(tensor: TensorInfo, buffer: *[4]i64) ?[]const i64 {
    if (std.mem.endsWith(u8, tensor.name, ".ssm_conv1d.weight")) {
        if (tensor.dims.len != 2) return null;
        buffer[0] = @intCast(tensor.dims[1]);
        buffer[1] = 1;
        buffer[2] = @intCast(tensor.dims[0]);
        return buffer[0..3];
    }

    if (tensor.dims.len > buffer.len) return null;
    for (0..tensor.dims.len) |i| {
        buffer[i] = @intCast(tensor.dims[tensor.dims.len - 1 - i]);
    }
    return buffer[0..tensor.dims.len];
}

const chunk_elements: usize = 32 * 1024;

const Permute = enum {
    none,
    row,
    column,
};

const Transform = enum {
    none,
    minus_one,
    neg_log,
};

const Layout = struct {
    permute: Permute = .none,
    perm_offset: usize = 0,
    head_dim: usize = 1,
    transform: Transform = .none,
};

const Output = enum { bf16, q8_quants, q8_scales, q4_qs, q4_sc, q4_mn, q4_d, q4_dmin, q5_qs, q5_qh, q5_sc, q5_mn, q5_d, q5_dmin, q6_ql0, q6_ql1, q6_qh, q6_sc1, q6_sc2, q6_sc3, q6_sc4, q6_d, iq4_qs, iq4_scl, iq4_sch, iq4_d };

const DequantCtx = struct {
    data: []const u8,
    quant: quants.Quant,
    offset: usize,
    numel: usize,
    rows: usize,
    row_len: usize,
    permute: Permute,
    perm_offset: usize,
    head_dim: usize,
    num_k_heads: usize,
    num_v_per_k: usize,
    transform: Transform,
    output: Output = .bf16,

    const ReadState = struct {
        ctx: *const DequantCtx,
        blocks_done: usize,
        elems_done: usize,
        out_row: usize,
        out: []u8,
        out_len: usize,
        out_pos: usize,
        f32_row: []f32,
        f32_buffer: [256]f32,
    };

    fn start(ctx: *const anyopaque, io: std.Io, tensor: *const zml.safetensors.Tensor) anyerror!*anyopaque {
        _ = io;
        _ = tensor;
        const self: *const DequantCtx = @ptrCast(@alignCast(ctx));

        const allocator = std.heap.page_allocator;
        const state = try allocator.create(ReadState);
        errdefer allocator.destroy(state);

        const out_len: usize = switch (self.output) {
            .bf16 => if (self.permute == .none) chunk_elements * 2 else self.row_len * 2,
            .q8_quants => if (self.permute == .none) chunk_elements else self.row_len,
            .q8_scales => if (self.permute == .none) chunk_elements / 32 * 2 else self.row_len / 32 * 2,
            .q4_qs => if (self.permute == .none) chunk_elements / 2 else self.row_len / 2,
            .q4_sc, .q4_mn => if (self.permute == .none) chunk_elements / 256 * 8 else self.row_len / 256 * 8,
            .q4_d, .q4_dmin => if (self.permute == .none) chunk_elements / 256 * 2 else self.row_len / 256 * 2,
            .q5_qs => if (self.permute == .none) chunk_elements / 2 else self.row_len / 2,
            .q5_qh => if (self.permute == .none) chunk_elements / 8 else self.row_len / 8,
            .q5_sc, .q5_mn => if (self.permute == .none) chunk_elements / 256 * 8 else self.row_len / 256 * 8,
            .q5_d, .q5_dmin => if (self.permute == .none) chunk_elements / 256 * 2 else self.row_len / 256 * 2,
            .q6_ql0, .q6_ql1, .q6_qh => if (self.permute == .none) chunk_elements / 4 else self.row_len / 4,
            .q6_sc1, .q6_sc2, .q6_sc3, .q6_sc4 => if (self.permute == .none) chunk_elements / 256 * 4 else self.row_len / 256 * 4,
            .q6_d => if (self.permute == .none) chunk_elements / 256 * 2 else self.row_len / 256 * 2,
            .iq4_qs => if (self.permute == .none) chunk_elements / 2 else self.row_len / 2,
            .iq4_scl => if (self.permute == .none) chunk_elements / 64 else self.row_len / 64,
            .iq4_sch, .iq4_d => if (self.permute == .none) chunk_elements / 256 * 2 else self.row_len / 256 * 2,
        };

        const out = try allocator.alloc(u8, out_len);
        errdefer allocator.free(out);

        const needs_f32_row = switch (self.output) {
            .q8_quants, .q8_scales => self.permute != .none,
            else => false,
        };
        const f32_row = try allocator.alloc(f32, if (needs_f32_row) self.row_len else 0);
        errdefer allocator.free(f32_row);

        state.* = .{
            .ctx = self,
            .blocks_done = 0,
            .elems_done = 0,
            .out_row = 0,
            .out = out,
            .out_len = 0,
            .out_pos = 0,
            .f32_row = f32_row,
            .f32_buffer = undefined,
        };
        return state;
    }

    fn stop(state: *anyopaque, io: std.Io) void {
        _ = io;
        const self: *ReadState = @ptrCast(@alignCast(state));
        std.heap.page_allocator.free(self.f32_row);
        std.heap.page_allocator.free(self.out);
        std.heap.page_allocator.destroy(self);
    }

    fn fill(state: *anyopaque, io: std.Io, writer: *std.Io.Writer, max_bytes: usize) anyerror!usize {
        const self: *ReadState = @ptrCast(@alignCast(state));
        const ctx = self.ctx;

        var written: usize = 0;
        while (written < max_bytes) {
            if (self.out_pos == self.out_len) {
                switch (ctx.permute) {
                    .none => {
                        if (self.elems_done == ctx.numel) break;
                        try decodeChunk(self, io);
                    },
                    .row, .column => {
                        if (self.out_row == ctx.rows) break;
                        try decodeRow(self, io);
                    },
                }
            }

            const n = @min(self.out_len - self.out_pos, max_bytes - written);
            try writer.writeAll(self.out[self.out_pos .. self.out_pos + n]);
            self.out_pos += n;
            written += n;
        }

        return written;
    }

    fn decodeChunk(self: *ReadState, io: std.Io) !void {
        _ = io;
        const ctx = self.ctx;
        const elements_per_block = ctx.quant.elementsPerBlock();
        const block_size = ctx.quant.blockSize();

        const remaining = ctx.numel - self.elems_done;
        var n_elems: usize = @min(remaining, chunk_elements);
        n_elems -= n_elems % elements_per_block;
        if (n_elems == 0) return error.InvalidTensorSize;

        const n_blocks = n_elems / elements_per_block;
        const src_len = n_blocks * block_size;
        const src_offset = ctx.offset + self.blocks_done * block_size;
        if (src_offset + src_len > ctx.data.len) return error.UnexpectedEndOfFile;
        const src = ctx.data[src_offset..][0..src_len];

        switch (ctx.output) {
            .q4_qs, .q4_sc, .q4_mn, .q4_d, .q4_dmin, .q5_qs, .q5_qh, .q5_sc, .q5_mn, .q5_d, .q5_dmin, .q6_ql0, .q6_ql1, .q6_qh, .q6_sc1, .q6_sc2, .q6_sc3, .q6_sc4, .q6_d, .iq4_qs, .iq4_scl, .iq4_sch, .iq4_d => {
                var out_off: usize = 0;
                for (0..n_blocks) |block_index| {
                    const block = src[block_index * block_size ..][0..block_size];
                    out_off += writeKPart(ctx.output, block, self.out[out_off..]);
                }
                self.blocks_done += n_blocks;
                self.elems_done += n_elems;
                self.out_len = out_off;
                self.out_pos = 0;
                return;
            },
            else => {},
        }

        var out_off: usize = 0;
        switch (ctx.quant) {
            inline else => |q| {
                const comptime_block_size = comptime q.blockSize();
                const comptime_epb = comptime q.elementsPerBlock();
                for (0..n_blocks) |block_index| {
                    const block = src[block_index * comptime_block_size ..][0..comptime_block_size];
                    quants.dequantBlockKind(q, block, self.f32_buffer[0..comptime_epb]);
                    switch (ctx.output) {
                        .bf16 => {
                            for (self.f32_buffer[0..comptime_epb]) |value| {
                                const bits = f32ToBf16(applyTransform(ctx.transform, value));
                                self.out[out_off + 0] = @truncate(bits);
                                self.out[out_off + 1] = @truncate(bits >> 8);
                                out_off += 2;
                            }
                        },
                        .q8_quants, .q8_scales => {
                            var sub: usize = 0;
                            while (sub < comptime_epb) : (sub += 32) {
                                var transformed: [32]f32 = undefined;
                                for (self.f32_buffer[sub..][0..32], 0..) |value, j| {
                                    transformed[j] = applyTransform(ctx.transform, value);
                                }
                                const scale = q8Scale(&transformed);
                                if (ctx.output == .q8_quants) {
                                    for (transformed, 0..) |value, j| {
                                        self.out[out_off + j] = @bitCast(q8Quant(value, scale));
                                    }
                                    out_off += 32;
                                } else {
                                    const bits: u16 = @bitCast(@as(f16, @floatCast(scale)));
                                    self.out[out_off + 0] = @truncate(bits);
                                    self.out[out_off + 1] = @truncate(bits >> 8);
                                    out_off += 2;
                                }
                            }
                        },
                        else => unreachable,
                    }
                }
            },
        }

        self.blocks_done += n_blocks;
        self.elems_done += n_elems;
        self.out_len = out_off;
        self.out_pos = 0;
    }

    fn decodeRow(self: *ReadState, io: std.Io) !void {
        _ = io;
        const ctx = self.ctx;
        const elements_per_block = ctx.quant.elementsPerBlock();
        const block_size = ctx.quant.blockSize();
        const blocks_per_row = ctx.row_len / elements_per_block;
        const src_len = blocks_per_row * block_size;

        const source_row = sourceRow(ctx, self.out_row);
        const src_offset = ctx.offset + source_row * src_len;
        if (src_offset + src_len > ctx.data.len) return error.UnexpectedEndOfFile;
        const src = ctx.data[src_offset..][0..src_len];

        switch (ctx.output) {
            .q4_qs, .q4_sc, .q4_mn, .q4_d, .q4_dmin, .q5_qs, .q5_qh, .q5_sc, .q5_mn, .q5_d, .q5_dmin, .q6_ql0, .q6_ql1, .q6_qh, .q6_sc1, .q6_sc2, .q6_sc3, .q6_sc4, .q6_d, .iq4_qs, .iq4_scl, .iq4_sch, .iq4_d => {
                var out_off: usize = 0;
                for (0..blocks_per_row) |block_index| {
                    const block = src[block_index * block_size ..][0..block_size];
                    out_off += writeKPart(ctx.output, block, self.out[out_off..]);
                }
                self.out_row += 1;
                self.out_len = out_off;
                self.out_pos = 0;
                return;
            },
            else => {},
        }

        switch (ctx.output) {
            .bf16 => {
                var col: usize = 0;
                switch (ctx.quant) {
                    inline else => |q| {
                        const comptime_block_size = comptime q.blockSize();
                        const comptime_epb = comptime q.elementsPerBlock();
                        for (0..blocks_per_row) |block_index| {
                            const block = src[block_index * comptime_block_size ..][0..comptime_block_size];
                            quants.dequantBlockKind(q, block, self.f32_buffer[0..comptime_epb]);
                            for (self.f32_buffer[0..comptime_epb]) |value| {
                                const bits = f32ToBf16(applyTransform(ctx.transform, value));
                                const out_col = switch (ctx.permute) {
                                    .column => inverseFlat(ctx, col),
                                    else => col,
                                };
                                self.out[out_col * 2 + 0] = @truncate(bits);
                                self.out[out_col * 2 + 1] = @truncate(bits >> 8);
                                col += 1;
                            }
                        }
                    },
                }
                self.out_len = ctx.row_len * 2;
            },
            .q8_quants, .q8_scales => {
                var col: usize = 0;
                switch (ctx.quant) {
                    inline else => |q| {
                        const comptime_block_size = comptime q.blockSize();
                        const comptime_epb = comptime q.elementsPerBlock();
                        for (0..blocks_per_row) |block_index| {
                            const block = src[block_index * comptime_block_size ..][0..comptime_block_size];
                            quants.dequantBlockKind(q, block, self.f32_buffer[0..comptime_epb]);
                            for (self.f32_buffer[0..comptime_epb]) |value| {
                                const out_col = switch (ctx.permute) {
                                    .column => inverseFlat(ctx, col),
                                    else => col,
                                };
                                self.f32_row[out_col] = applyTransform(ctx.transform, value);
                                col += 1;
                            }
                        }
                    },
                }

                var out_off: usize = 0;
                var i: usize = 0;
                while (i < ctx.row_len) : (i += 32) {
                    const values = self.f32_row[i..][0..32];
                    const scale = q8Scale(values);
                    if (ctx.output == .q8_quants) {
                        for (values, 0..) |value, j| {
                            self.out[out_off + j] = @bitCast(q8Quant(value, scale));
                        }
                        out_off += 32;
                    } else {
                        const bits: u16 = @bitCast(@as(f16, @floatCast(scale)));
                        self.out[out_off + 0] = @truncate(bits);
                        self.out[out_off + 1] = @truncate(bits >> 8);
                        out_off += 2;
                    }
                }
                self.out_len = out_off;
            },
            else => unreachable,
        }

        self.out_row += 1;
        self.out_pos = 0;
    }
};

fn getScaleMinK4(j: usize, q: *const [12]u8) struct { u8, u8 } {
    if (j < 4) return .{ q[j] & 63, q[j + 4] & 63 };
    return .{
        (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4),
        (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
    };
}

fn writeKPart(output: Output, block: []const u8, out: []u8) usize {
    switch (output) {
        .q4_qs => {
            @memcpy(out[0..128], block[16..144]);
            return 128;
        },
        .q5_qs => {
            @memcpy(out[0..128], block[48..176]);
            return 128;
        },
        .q5_qh => {
            @memcpy(out[0..32], block[16..48]);
            return 32;
        },
        .q4_sc, .q4_mn, .q5_sc, .q5_mn => {
            var scales: [12]u8 = undefined;
            @memcpy(&scales, block[4..16]);
            for (0..8) |j| {
                const sc, const mn = getScaleMinK4(j, &scales);
                out[j] = if (output == .q4_sc or output == .q5_sc) sc else mn;
            }
            return 8;
        },
        .q4_d, .q5_d => {
            @memcpy(out[0..2], block[0..2]);
            return 2;
        },
        .q4_dmin, .q5_dmin => {
            @memcpy(out[0..2], block[2..4]);
            return 2;
        },
        .q6_ql0 => {
            @memcpy(out[0..32], block[0..32]);
            @memcpy(out[32..64], block[64..96]);
            return 64;
        },
        .q6_ql1 => {
            @memcpy(out[0..32], block[32..64]);
            @memcpy(out[32..64], block[96..128]);
            return 64;
        },
        .q6_qh => {
            @memcpy(out[0..64], block[128..192]);
            return 64;
        },
        .q6_sc1, .q6_sc2, .q6_sc3, .q6_sc4 => {
            const base: usize = switch (output) {
                .q6_sc1 => 0,
                .q6_sc2 => 2,
                .q6_sc3 => 4,
                .q6_sc4 => 6,
                else => unreachable,
            };
            out[0] = block[192 + base + 0];
            out[1] = block[192 + base + 1];
            out[2] = block[192 + base + 8];
            out[3] = block[192 + base + 9];
            return 4;
        },
        .q6_d => {
            @memcpy(out[0..2], block[208..210]);
            return 2;
        },
        .iq4_qs => {
            @memcpy(out[0..128], block[8..136]);
            return 128;
        },
        .iq4_scl => {
            @memcpy(out[0..4], block[4..8]);
            return 4;
        },
        .iq4_sch => {
            @memcpy(out[0..2], block[2..4]);
            return 2;
        },
        .iq4_d => {
            @memcpy(out[0..2], block[0..2]);
            return 2;
        },
        else => unreachable,
    }
}

fn q8Scale(values: []const f32) f32 {
    var max_abs: f32 = 0.0;
    for (values) |value| max_abs = @max(max_abs, @abs(value));
    return max_abs / 127.0;
}

fn q8Quant(value: f32, scale: f32) i8 {
    if (scale == 0.0) return 0;
    const q = @round(value / scale);
    return @intFromFloat(std.math.clamp(q, -127.0, 127.0));
}

fn applyTransform(transform: Transform, value: f32) f32 {
    return switch (transform) {
        .none => value,
        .minus_one => value - 1.0,
        .neg_log => if (value < 0.0) @log(-value) else 0.0,
    };
}

/// Maps a tiled V-head index (GGUF order) to the grouped V-head index (HF order).
fn inverseFlat(ctx: *const DequantCtx, index: usize) usize {
    const head = index / ctx.head_dim;
    const offset = index % ctx.head_dim;
    const k = head % ctx.num_k_heads;
    const v = head / ctx.num_k_heads;
    return (k * ctx.num_v_per_k + v) * ctx.head_dim + offset;
}

/// Maps a grouped V-head index (HF order) to the tiled V-head index (GGUF order).
fn forwardFlat(ctx: *const DequantCtx, index: usize) usize {
    const head = index / ctx.head_dim;
    const offset = index % ctx.head_dim;
    const k = head / ctx.num_v_per_k;
    const v = head % ctx.num_v_per_k;
    return (v * ctx.num_k_heads + k) * ctx.head_dim + offset;
}

fn sourceRow(ctx: *const DequantCtx, out_row: usize) usize {
    if (ctx.permute != .row or out_row < ctx.perm_offset) return out_row;
    return ctx.perm_offset + forwardFlat(ctx, out_row - ctx.perm_offset);
}

fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x8000) >> 16);
}

test "map qwen3.8 tensor names" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    try std.testing.expectEqualStrings(
        "model.language_model.embed_tokens.weight",
        (try mapName(allocator, "token_embd.weight", 64)).?,
    );
    try std.testing.expectEqualStrings(
        "lm_head.weight",
        (try mapName(allocator, "output.weight", 64)).?,
    );
    try std.testing.expectEqualStrings(
        "model.language_model.layers.3.self_attn.q_proj.weight",
        (try mapName(allocator, "blk.3.attn_q.weight", 64)).?,
    );
    try std.testing.expectEqualStrings(
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        (try mapName(allocator, "blk.0.ssm_conv1d.weight", 64)).?,
    );
    try std.testing.expect((try mapName(allocator, "blk.64.attn_q.weight", 64)) == null);
    try std.testing.expect((try mapName(allocator, "blk.0.nextn.eh_proj.weight", 64)) == null);
}

test "conv1d shape" {
    var buffer: [4]i64 = undefined;
    const tensor: TensorInfo = .{
        .name = "blk.0.ssm_conv1d.weight",
        .dims = &.{ 4, 10240 },
        .quant = .f32,
        .offset = 0,
        .numel = 40960,
    };
    try std.testing.expectEqualSlices(i64, &.{ 10240, 1, 4 }, hfDims(tensor, &buffer).?);
}

test "v-head reorder" {
    const ctx: DequantCtx = .{
        .file_uri = "",
        .quant = .f32,
        .offset = 0,
        .numel = 0,
        .rows = 0,
        .row_len = 0,
        .permute = .row,
        .perm_offset = 0,
        .head_dim = 1,
        .num_k_heads = 2,
        .num_v_per_k = 3,
        .transform = .none,
    };

    // Tiled order for K=2, r=3 is [G0v0, G1v0, G0v1, G1v1, G0v2, G1v2].
    const expected_tiled = [_]usize{ 0, 3, 1, 4, 2, 5 };
    for (expected_tiled, 0..) |tiled, grouped| {
        try std.testing.expectEqual(tiled, forwardFlat(&ctx, grouped));
        try std.testing.expectEqual(grouped, inverseFlat(&ctx, tiled));
    }
}

test "tensor transforms" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), applyTransform(.minus_one, 1.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), applyTransform(.neg_log, -1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), applyTransform(.none, 2.0), 1e-6);
}
