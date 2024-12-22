/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/forces.ts":
/*!***********************!*\
  !*** ./src/forces.ts ***!
  \***********************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.build = build;
function build(width, height, grid_x, grid_spacing, n_elems) {
    return `
@group(0) @binding(0)
var<storage, read_write > x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write > y_pos: array<f32>;

@group(0) @binding(2)
var<storage, read_write > intensity_map: array<u32>;

@group(0) @binding(3)
var<storage, read_write > x_pos_out: array<f32>;

@group(0) @binding(4)
var<storage, read_write > y_pos_out: array<f32>;

@group(0) @binding(5)
var<uniform>offset: u32;

@group(0) @binding(6)
var<uniform>stride: u32;



fn pixelToIntensity(_px: u32) -> f32 {
  var px = _px;
  let r = f32(px % 256);
  px /= u32(256);
  let g = f32(px % 256);
  px /= u32(256);
  let b = f32(px % 256);
  px /= u32(256);
  let a = f32(px % 256);
  let intensity: f32 = (a / 255) * (1 - (0.2126 * r + 0.7152 * g + 0.0722 * b));

  return intensity;
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id)
global_id : vec3u,

  @builtin(local_invocation_id)
local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let start = offset + (stride * global_id.x);
  for (var c = start; c < (start + stride); c+=u32(1)) {
    let i = c;
    let grid_x = i % ${grid_x};
    let grid_y = i / ${grid_x};
    let origin_x = grid_x * ${grid_spacing};
    let origin_y = grid_y * ${grid_spacing};

    if (i > ${n_elems}) {
      continue;
    }

    let x = x_pos[i];
    let y = y_pos[i];

    var dir_x: f32 = 0;
    var dir_y: f32 = 0;
    var coeff: f32 = 0;

    let region = 20;
    for (var s_y: i32 = 0; s_y <= region; s_y++) {
      for (var s_x: i32 = 0; s_x <= region; s_x++) {
        let ds_y = s_y - region / 2;
        let ds_x = s_x - region / 2;

        let base_y = i32(origin_y);
        let base_x = i32(origin_x);
        let f_y = base_y + ds_y;
        let f_x = base_x + ds_x;
        let d_x: f32 = f32(f_x) + 0.5 - x;
        let d_y: f32 = f32(f_y) + 0.5 - y;

        if (ds_y == 0 && ds_x == 0) {
          let local_coeff = f32(200);
          coeff += local_coeff;
          dir_x += local_coeff * f32(f_x);
          dir_y += local_coeff * f32(f_y);
          continue;
        }

        if (f_y >= 0 && f_y < ${height} && f_x >= 0 && f_x < ${width}) {
          let f_i = f_y * ${width} + f_x;
          let intensity = pixelToIntensity(intensity_map[f_i]);
          let local_coeff = f32(100) * intensity;
          coeff += local_coeff;
          dir_x += local_coeff * f32(f_x);
          dir_y += local_coeff * f32(f_y);
        }
      }
    }

    let total_coeff = coeff;
    if (total_coeff != 0) {
      var d_x = dir_x / total_coeff - x;
      var d_y = dir_y / total_coeff - y;

      let dist2 = d_x * d_x + d_y * d_y;
      let max_dist2 = f32(region * region);

      var speed = dist2 / max_dist2;
      if (dist2 < f32(5)) {
        speed = f32(1);
      } else {
        speed = f32(0.5);
      }

      d_x *= speed;
      d_y *= speed;

      x_pos_out[i] = x + d_x;
      y_pos_out[i] = y + d_y;
    } else {
      x_pos_out[i] = x;
      y_pos_out[i] = y;
    }
  }
}
`;
}


/***/ }),

/***/ "./src/init.ts":
/*!*********************!*\
  !*** ./src/init.ts ***!
  \*********************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.build = build;
function build(n_elems, grid_x, grid_spacing) {
    return `
@group(0) @binding(0)
var<storage, read_write> x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y_pos: array<f32>;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  if (global_id.x < ${n_elems}) {
      let y = global_id.x / ${grid_x};
      let x = global_id.x % ${grid_x};

      x_pos[global_id.x] = f32(x * ${grid_spacing});
      y_pos[global_id.x] = f32(y * ${grid_spacing});
  }
}
`;
}


/***/ }),

/***/ "./src/meshdeformation.ts":
/*!********************************!*\
  !*** ./src/meshdeformation.ts ***!
  \********************************/
/***/ ((__unused_webpack_module, exports, __webpack_require__) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.MeshDeformation = void 0;
const init_1 = __webpack_require__(/*! ./init */ "./src/init.ts");
const forces_1 = __webpack_require__(/*! ./forces */ "./src/forces.ts");
const render_1 = __webpack_require__(/*! ./render */ "./src/render.ts");
const edges = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [1, 1],
    [-1, -1],
];
class MeshDeformation {
    ctx;
    grid_x;
    grid_y;
    grid_spacing;
    min_dist;
    max_dist;
    radius;
    reinit;
    draw_edges;
    n_elems;
    x_pos;
    y_pos;
    init_module;
    initBindGroupLayout;
    init_computePipeline;
    initialization_done;
    device;
    force_module;
    active_buffer_id;
    x_pos_buffers;
    y_pos_buffers;
    // Buffers to read values back to the CPU for drawing
    staging_x_buf;
    staging_y_buf;
    // Buffer to write value to from the CPU for adjusting weights
    staging_intensity_buf;
    intensity_map_buf;
    offset_buf;
    stride_buf;
    image_buf;
    image_reset;
    force_bind_group_layout;
    render_to_buf_module;
    render_to_buf_bind_group_layout;
    render_to_buf_compute_pipeline;
    render_module;
    render_pipeline;
    render_pass_descriptor;
    vertex_buffer;
    render_bind_group_layout;
    constructor(ctx, grid_x, grid_y, grid_spacing, min_dist, max_dist, radius) {
        this.ctx = ctx;
        this.grid_x = grid_x;
        this.grid_y = grid_y;
        this.grid_spacing = grid_spacing;
        this.min_dist = min_dist;
        this.max_dist = max_dist;
        this.radius = radius;
        this.draw_edges = true;
        this.reinit = false;
        this.n_elems = this.grid_x * this.grid_y;
        this.initialization_done = this.async_init();
    }
    async async_init() {
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        console.log("Create compute shader");
        this.force_module = this.device.createShaderModule({
            code: (0, forces_1.build)(this.ctx.canvas.width, this.ctx.canvas.height, this.grid_x, this.grid_spacing, this.n_elems),
        });
        console.log("done Create compute shader");
        this.active_buffer_id = 0;
        this.x_pos_buffers = [
            this.device.createBuffer({
                label: "x_pos[0]",
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
            this.device.createBuffer({
                label: "x_pos[1]",
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
        ];
        this.y_pos_buffers = [
            this.device.createBuffer({
                label: "y_pos[0]",
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
            this.device.createBuffer({
                label: "y_pos[1]",
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
        ];
        this.staging_x_buf = this.device.createBuffer({
            label: "staging_x_buf",
            size: this.n_elems * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.staging_y_buf = this.device.createBuffer({
            label: "staging_y_buf",
            size: this.n_elems * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.staging_intensity_buf = this.device.createBuffer({
            label: "staging_intensity_buf",
            size: this.ctx.canvas.width * this.ctx.canvas.height * 4,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        });
        this.intensity_map_buf = this.device.createBuffer({
            label: "intensity_buf",
            size: this.ctx.canvas.width * this.ctx.canvas.height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.offset_buf = this.device.createBuffer({
            label: "offset_buf",
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.stride_buf = this.device.createBuffer({
            label: "stride_buf",
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.image_buf = this.device.createBuffer({
            label: "image_buf",
            size: this.ctx.canvas.width * this.ctx.canvas.height * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.image_reset = new Float32Array(this.ctx.canvas.width * this.ctx.canvas.height * 4);
        console.log("done allocate buffers");
        this.force_bind_group_layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
            ],
        });
        // intialize this.x_pos_buffers[this.active_buffer_id] and
        // this.y_pos_buffers[*] to be a grid
        const init_shader = (0, init_1.build)(this.n_elems, this.grid_x, this.grid_spacing);
        this.initBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
        console.log("Create init shader");
        this.init_module = this.device.createShaderModule({
            code: init_shader,
        });
        console.log("done Create init shader");
        this.init_computePipeline = this.device.createComputePipeline({
            label: "compute force",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.initBindGroupLayout],
            }),
            compute: {
                module: this.init_module,
                entryPoint: "main",
            },
        });
        const commandEncoder = this.device.createCommandEncoder();
        const bindGroup = this.device.createBindGroup({
            layout: this.initBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.x_pos_buffers[this.active_buffer_id],
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.y_pos_buffers[this.active_buffer_id],
                    },
                }
            ],
        });
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.init_computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(this.n_elems / 256));
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(this.x_pos_buffers[this.active_buffer_id], 0, this.staging_x_buf, 0, this.staging_x_buf.size);
        commandEncoder.copyBufferToBuffer(this.y_pos_buffers[this.active_buffer_id], 0, this.staging_y_buf, 0, this.staging_y_buf.size);
        this.device.queue.submit([commandEncoder.finish()]);
        // intialize this.x_pos_buffers[this.active_buffer_id] and
        // this.y_pos_buffers[*] to be a grid
        const render_to_buffer_shader = (0, render_1.build)(this.ctx.canvas.width, this.ctx.canvas.height, this.grid_x, this.grid_spacing, this.n_elems, this.radius);
        this.render_to_buf_bind_group_layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    },
                },
            ],
        });
        this.render_to_buf_module = this.device.createShaderModule({
            code: render_to_buffer_shader,
        });
        this.render_to_buf_compute_pipeline = this.device.createComputePipeline({
            label: "compute force",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.render_to_buf_bind_group_layout],
            }),
            compute: {
                module: this.render_to_buf_module,
                entryPoint: "main",
            },
        });
        this.ctx.configure({
            device: this.device,
            format: navigator.gpu.getPreferredCanvasFormat(),
        });
        const vertices = new Float32Array([
            -1, -1,
            1, -1,
            -1, 1,
            -1, 1,
            1, -1,
            1, 1,
        ]);
        this.vertex_buffer = this.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertex_buffer, 0, vertices, 0, vertices.length);
        const shaders = `
@group(0) @binding(0)
var<storage, read_write> image_map: array<f32>;

@vertex
fn vertex_main(@location(0) position: vec2f) -> @builtin(position) vec4f
{
  return vec4f(position.x, position.y, 0, 1);
}

@fragment
fn fragment_main(@builtin(position) position: vec4f) -> @location(0) vec4f
{
  let i = 4 * u32(position.x - 500 + position.y * ${this.ctx.canvas.width});
  return vec4f(
    image_map[i + 0],
    image_map[i + 1],
    image_map[i + 2],
    image_map[i + 3]);
}
`;
        const shaderModule = this.device.createShaderModule({
            code: shaders,
        });
        this.render_bind_group_layout = this.device.createBindGroupLayout({
            label: 'render_bind_group_layout',
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
        const vertexBuffers = [
            {
                attributes: [
                    {
                        shaderLocation: 0, // position
                        offset: 0,
                        format: "float32x2",
                    },
                ],
                arrayStride: 4 * 2,
                stepMode: "vertex",
            },
        ];
        const pipelineDescriptor = {
            vertex: {
                module: shaderModule,
                entryPoint: "vertex_main",
                buffers: vertexBuffers,
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fragment_main",
                targets: [
                    {
                        format: navigator.gpu.getPreferredCanvasFormat(),
                    },
                ],
            },
            primitive: {
                topology: "triangle-list",
            },
            layout: this.device.createPipelineLayout({
                label: 'render_bind_group_pipeline_layout',
                bindGroupLayouts: [this.render_bind_group_layout],
            })
        };
        this.render_pipeline = this.device.createRenderPipeline(pipelineDescriptor);
        const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };
        this.render_pass_descriptor = {
            colorAttachments: [
                {
                    clearValue: clearColor,
                    loadOp: "clear",
                    storeOp: "store",
                    view: this.ctx.getCurrentTexture().createView(),
                },
            ],
        };
        console.log("done async init");
    }
    async applyForce(ctx) {
        let idata = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
        // console.log(`b0 ${idata[0]}, ${idata[1]}, ${idata[2]}, ${idata[3]}`);
        // console.log(`Writing ${this.intensity_map_buf.size}/${idata.length} bytes for imap`);
        let start = performance.now();
        this.device.queue.writeBuffer(this.intensity_map_buf, 0, idata.buffer, 0, this.intensity_map_buf.size);
        window.write_time += performance.now() - start;
        let input_x = this.x_pos_buffers[this.active_buffer_id];
        let input_y = this.y_pos_buffers[this.active_buffer_id];
        let output_x = this.x_pos_buffers[1 - this.active_buffer_id];
        let output_y = this.y_pos_buffers[1 - this.active_buffer_id];
        if (this.reinit) {
            const bindGroup = this.device.createBindGroup({
                layout: this.initBindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.x_pos_buffers[this.active_buffer_id],
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.y_pos_buffers[this.active_buffer_id],
                        },
                    }
                ],
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.init_computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.n_elems / 256));
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
            this.reinit = false;
        }
        const wg_x = 64;
        const stride = 8;
        const dispatch_x = (256 / wg_x);
        let buffers = [input_x, input_y, this.intensity_map_buf, output_x, output_y, this.offset_buf, this.stride_buf];
        const bindGroup = this.device.createBindGroup({
            layout: this.force_bind_group_layout,
            entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
        });
        for (let offset = 0; offset < this.n_elems; offset += (stride * wg_x * dispatch_x)) {
            this.device.queue.writeBuffer(this.offset_buf, 0, new Uint32Array([offset]).buffer, 0, 4);
            this.device.queue.writeBuffer(this.stride_buf, 0, new Uint32Array([stride]).buffer, 0, 4);
            const computePipeline = this.device.createComputePipeline({
                label: "forcepipeline",
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.force_bind_group_layout],
                }),
                compute: {
                    module: this.force_module,
                    entryPoint: "main",
                },
            });
            // console.log("created pipeline");
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(dispatch_x, 1, 1);
            passEncoder.end();
            // console.log("encoded compute");
            this.device.queue.submit([commandEncoder.finish()]);
            // console.log("queue", offset);
        }
        this.device.queue.writeBuffer(this.image_buf, 0, this.image_reset);
        for (let offset = 0; offset < this.n_elems; offset += 256) {
            this.device.queue.writeBuffer(this.offset_buf, 0, new Uint32Array([offset]).buffer, 0, 4);
            const computePipeline = this.device.createComputePipeline({
                label: "render_to_buf_pipeline",
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.render_to_buf_bind_group_layout],
                }),
                compute: {
                    module: this.render_to_buf_module,
                    entryPoint: "main",
                },
            });
            let buffers = [output_x, output_y, this.intensity_map_buf, this.image_buf, this.offset_buf];
            const bindGroup = this.device.createBindGroup({
                layout: this.render_to_buf_bind_group_layout,
                entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(1, 1, 1);
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        const copy_output_commandEncoder = this.device.createCommandEncoder();
        // Copy output buffer to staging buffer
        copy_output_commandEncoder.copyBufferToBuffer(output_x, 0, this.staging_x_buf, 0, this.staging_x_buf.size);
        // console.log("x copying", this.staging_x_buf.size, "bytes");
        copy_output_commandEncoder.copyBufferToBuffer(output_y, 0, this.staging_y_buf, 0, this.staging_y_buf.size);
        // console.log("y copying", this.staging_x_buf.size, "bytes");
        // console.log("encoded copy to buffers", this.active_buffer_id);
        // End frame by passing array of command buffers to command queue for execution
        this.device.queue.submit([copy_output_commandEncoder.finish()]);
        // console.log("done submit to queue");
        {
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginRenderPass(this.render_pass_descriptor);
            passEncoder.setPipeline(this.render_pipeline);
            passEncoder.setVertexBuffer(0, this.vertex_buffer);
            const bindGroup = this.device.createBindGroup({
                label: 'render_bind_group',
                layout: this.render_bind_group_layout,
                entries: [{ binding: 0, resource: { buffer: this.image_buf } }],
            });
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.draw(6);
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        // Swap input and output:
        this.active_buffer_id = 1 - this.active_buffer_id;
    }
}
exports.MeshDeformation = MeshDeformation;


/***/ }),

/***/ "./src/render.ts":
/*!***********************!*\
  !*** ./src/render.ts ***!
  \***********************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.build = build;
function build(width, height, grid_x, grid_spacing, n_elems, radius) {
    return `
@group(0) @binding(0)
var<storage, read_write > x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write > y_pos: array<f32>;

@group(0) @binding(2)
var<storage, read_write> intensity_map: array<u32>;

@group(0) @binding(3)
var<storage, read_write > image_map: array<f32>;

@group(0) @binding(4)
var<uniform>offset: u32;


@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id)
global_id : vec3u,

  @builtin(local_invocation_id)
local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let i = offset + global_id.x;

  let grid_x = i % ${grid_x};
  let grid_y = i / ${grid_x};
  let origin_x = grid_x * ${grid_spacing};
  let origin_y = grid_y * ${grid_spacing};

  if (i > ${n_elems}) {
    return;
  }

  // var px = intensity_map[origin_x + origin_y * ${width}];
  // let r = f32(px % 256) / 256;
  // px /= u32(256);
  // let g = f32(px % 256) / 256;
  // px /= u32(256);
  // let b = f32(px % 256) / 256;
  let r = f32(1);
  let g = f32(0);
  let b = f32(0);

  let x = x_pos[i];
  let y = y_pos[i];

  for (var yi = -20; yi <= 20; yi++) {
    for (var xi = -20; xi <= 20; xi++) {
      let ox = i32(origin_x) + xi;
      let oy = i32(origin_y) + yi;
      if (ox < 0 || ox > ${width} || oy < 0 || oy > ${height}) {
        continue;
      }
      let dx = f32(ox) - x;
      let dy = f32(oy) - y;

      let d = sqrt(dx * dx + dy * dy);
      if (abs(d - ${radius}) < 1) {
        let out = ox + oy * ${width};
        image_map[4 * out + 0] = r;
        image_map[4 * out + 1] = g;
        image_map[4 * out + 2] = b;
        image_map[4 * out + 3] = f32(1);
      }
    }
  }
}
`;
}


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
// This entry needs to be wrapped in an IIFE because it needs to be isolated against other modules in the chunk.
(() => {
var exports = __webpack_exports__;
/*!*********************!*\
  !*** ./src/main.ts ***!
  \*********************/

Object.defineProperty(exports, "__esModule", ({ value: true }));
const meshdeformation_1 = __webpack_require__(/*! ./meshdeformation */ "./src/meshdeformation.ts");
let stop = false;
async function setupWebcam() {
    const video = document.createElement("video");
    const constraints = { video: true };
    try {
        if (video.srcObject) {
            const stream = video.srcObject;
            stream.getTracks().forEach(function (track) {
                track.stop();
            });
            video.srcObject = null;
        }
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.play();
    }
    catch (err) {
        alert("Error initializing webcam! " + err);
        console.log(err);
    }
    return video;
}
async function main(container) {
    let canvas = document.createElement("canvas");
    canvas.width = 1000;
    canvas.height = 1000;
    container.appendChild(canvas);
    // let ctx = canvas.getContext("2d");
    let ctx = canvas.getContext("webgpu");
    console.log("Created context for main canvas");
    let canvas2 = document.createElement("canvas");
    canvas2.width = 1000;
    canvas2.height = 1000;
    let ctx2 = canvas2.getContext("2d");
    // document.body.appendChild(canvas2);
    canvas.addEventListener("click", (e) => {
        let el = e.target;
        const rect = el.getBoundingClientRect();
        const x = el.width * (e.clientX - rect.left) / rect.width;
        const y = el.height * (e.clientY - rect.top) / rect.height;
        ctx2.beginPath();
        ctx2.fillStyle = "black";
        ctx2.arc(x, y, 100, 0, 2 * Math.PI);
        ctx2.fill();
    });
    container.appendChild(document.createElement("br"));
    const clear = document.createElement("button");
    clear.innerText = "clear";
    clear.addEventListener("click", () => {
        ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
    });
    container.appendChild(clear);
    const edges = document.createElement("button");
    edges.innerText = "edges";
    edges.addEventListener("click", () => {
        md.draw_edges = !md.draw_edges;
    });
    container.appendChild(edges);
    const video = await setupWebcam();
    console.log("Created context for interactive canvas");
    window.n_steps_per_frame = 1;
    let n_elems = 200;
    let spacing = ctx.canvas.width / n_elems;
    let md = new meshdeformation_1.MeshDeformation(ctx, n_elems, n_elems, spacing, spacing / 4, spacing * 4, 1);
    window.t_per_render = 0;
    window.n_renders = 0;
    window.write_time = 0;
    window.interval = 0;
    let theta = 0;
    let last_start = 0;
    md.initialization_done.then(() => {
        const f = async () => {
            let start = performance.now();
            for (let i = 0; i < window.n_steps_per_frame; i++) {
                await md.applyForce(ctx2);
            }
            let end = performance.now();
            window.t_per_render += end - start;
            window.n_renders += 1;
            if (video.readyState == 4) {
                ctx2.drawImage(video, 0, 0, ctx2.canvas.width, ctx2.canvas.height);
            }
            // ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
            let x = Math.sin(theta) * 450 + 500;
            let y = Math.cos(theta) * 450 + 500;
            ctx2.beginPath();
            ctx2.fillStyle = "black";
            ctx2.arc(x, y, 100, 0, 2 * Math.PI);
            ctx2.fill();
            theta += 0.1;
            window.interval += start - last_start;
            last_start = start;
            if (!stop) {
                requestAnimationFrame(f);
            }
        };
        requestAnimationFrame(f);
        window.t_per_draw = 0;
        window.t_per_read = 0;
        window.n_draws = 0;
    });
    window.stats = () => {
        let w = window;
        console.log("avg_interval", w.interval / w.n_renders);
        console.log("avg_t_per_render", w.t_per_render / w.n_renders);
        console.log("avg_t_per_write", w.write_time / w.n_renders);
        console.log("avg_t_per_draw", w.t_per_draw / w.n_draws);
        console.log("avg_t_per_read", w.t_per_read / w.n_draws);
    };
    function cancel() {
        stop = true;
    }
    window.md = md;
    window.ctx2 = ctx2;
    window.cancel = cancel;
}
document.addEventListener("DOMContentLoaded", () => {
    let container = document.getElementById("container");
    main(container);
});

})();

/******/ })()
;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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QyxDQUFDO0lBRUYsTUFBYyxDQUFDLEtBQUssR0FBRyxHQUFHLEVBQUU7UUFDM0IsSUFBSSxDQUFDLEdBQUcsTUFBYSxDQUFDO1FBQ3RCLE9BQU8sQ0FBQyxHQUFHLENBQUMsY0FBYyxFQUFFLENBQUMsQ0FBQyxRQUFRLEdBQUcsQ0FBQyxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQ3RELE9BQU8sQ0FBQyxHQUFHLENBQUMsa0JBQWtCLEVBQUUsQ0FBQyxDQUFDLFlBQVksR0FBRyxDQUFDLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDOUQsT0FBTyxDQUFDLEdBQUcsQ0FBQyxpQkFBaUIsRUFBRSxDQUFDLENBQUMsVUFBVSxHQUFHLENBQUMsQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUMzRCxPQUFPLENBQUMsR0FBRyxDQUFDLGdCQUFnQixFQUFFLENBQUMsQ0FBQyxVQUFVLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ3hELE9BQU8sQ0FBQyxHQUFHLENBQUMsZ0JBQWdCLEVBQUUsQ0FBQyxDQUFDLFVBQVUsR0FBRyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDMUQsQ0FBQztJQUdELFNBQVMsTUFBTTtRQUNiLElBQUksR0FBRyxJQUFJLENBQUM7SUFDZCxDQUFDO0lBQ0EsTUFBYyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7SUFDdkIsTUFBYyxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7SUFDM0IsTUFBYyxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7QUFDbEMsQ0FBQztBQUVELFFBQVEsQ0FBQyxnQkFBZ0IsQ0FBQyxrQkFBa0IsRUFBRSxHQUFHLEVBQUU7SUFDakQsSUFBSSxTQUFTLEdBQUcsUUFBUSxDQUFDLGNBQWMsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUNyRCxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7QUFDbEIsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9mb3JjZXMudHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvaW5pdC50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9tZXNoZGVmb3JtYXRpb24udHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvcmVuZGVyLnRzIiwid2VicGFjazovL3lvdXJQcm9qZWN0L3dlYnBhY2svYm9vdHN0cmFwIiwid2VicGFjazovL3lvdXJQcm9qZWN0Ly4vc3JjL21haW4udHMiXSwic291cmNlc0NvbnRlbnQiOlsiZXhwb3J0IGZ1bmN0aW9uIGJ1aWxkKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBncmlkX3g6IG51bWJlciwgZ3JpZF9zcGFjaW5nOiBudW1iZXIsIG5fZWxlbXM6IG51bWJlcikge1xuICByZXR1cm4gYFxuQGdyb3VwKDApIEBiaW5kaW5nKDApXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygyKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiBpbnRlbnNpdHlfbWFwOiBhcnJheTx1MzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMylcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNSlcbnZhcjx1bmlmb3JtPm9mZnNldDogdTMyO1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNilcbnZhcjx1bmlmb3JtPnN0cmlkZTogdTMyO1xuXG5cblxuZm4gcGl4ZWxUb0ludGVuc2l0eShfcHg6IHUzMikgLT4gZjMyIHtcbiAgdmFyIHB4ID0gX3B4O1xuICBsZXQgciA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgZyA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgYiA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgYSA9IGYzMihweCAlIDI1Nik7XG4gIGxldCBpbnRlbnNpdHk6IGYzMiA9IChhIC8gMjU1KSAqICgxIC0gKDAuMjEyNiAqIHIgKyAwLjcxNTIgKiBnICsgMC4wNzIyICogYikpO1xuXG4gIHJldHVybiBpbnRlbnNpdHk7XG59XG5cbkBjb21wdXRlIEB3b3JrZ3JvdXBfc2l6ZSg2NCwgMSwgMSlcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuZ2xvYmFsX2lkIDogdmVjM3UsXG5cbiAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZClcbmxvY2FsX2lkIDogdmVjM3UsXG4pIHtcbiAgLy8gQ29vcmRpbmF0ZXMgb2YgcGFydGljbGUgZm9yIHRoaXMgdGhyZWFkXG4gIGxldCBzdGFydCA9IG9mZnNldCArIChzdHJpZGUgKiBnbG9iYWxfaWQueCk7XG4gIGZvciAodmFyIGMgPSBzdGFydDsgYyA8IChzdGFydCArIHN0cmlkZSk7IGMrPXUzMigxKSkge1xuICAgIGxldCBpID0gYztcbiAgICBsZXQgZ3JpZF94ID0gaSAlICR7Z3JpZF94fTtcbiAgICBsZXQgZ3JpZF95ID0gaSAvICR7Z3JpZF94fTtcbiAgICBsZXQgb3JpZ2luX3ggPSBncmlkX3ggKiAke2dyaWRfc3BhY2luZ307XG4gICAgbGV0IG9yaWdpbl95ID0gZ3JpZF95ICogJHtncmlkX3NwYWNpbmd9O1xuXG4gICAgaWYgKGkgPiAke25fZWxlbXN9KSB7XG4gICAgICBjb250aW51ZTtcbiAgICB9XG5cbiAgICBsZXQgeCA9IHhfcG9zW2ldO1xuICAgIGxldCB5ID0geV9wb3NbaV07XG5cbiAgICB2YXIgZGlyX3g6IGYzMiA9IDA7XG4gICAgdmFyIGRpcl95OiBmMzIgPSAwO1xuICAgIHZhciBjb2VmZjogZjMyID0gMDtcblxuICAgIGxldCByZWdpb24gPSAyMDtcbiAgICBmb3IgKHZhciBzX3k6IGkzMiA9IDA7IHNfeSA8PSByZWdpb247IHNfeSsrKSB7XG4gICAgICBmb3IgKHZhciBzX3g6IGkzMiA9IDA7IHNfeCA8PSByZWdpb247IHNfeCsrKSB7XG4gICAgICAgIGxldCBkc195ID0gc195IC0gcmVnaW9uIC8gMjtcbiAgICAgICAgbGV0IGRzX3ggPSBzX3ggLSByZWdpb24gLyAyO1xuXG4gICAgICAgIGxldCBiYXNlX3kgPSBpMzIob3JpZ2luX3kpO1xuICAgICAgICBsZXQgYmFzZV94ID0gaTMyKG9yaWdpbl94KTtcbiAgICAgICAgbGV0IGZfeSA9IGJhc2VfeSArIGRzX3k7XG4gICAgICAgIGxldCBmX3ggPSBiYXNlX3ggKyBkc194O1xuICAgICAgICBsZXQgZF94OiBmMzIgPSBmMzIoZl94KSArIDAuNSAtIHg7XG4gICAgICAgIGxldCBkX3k6IGYzMiA9IGYzMihmX3kpICsgMC41IC0geTtcblxuICAgICAgICBpZiAoZHNfeSA9PSAwICYmIGRzX3ggPT0gMCkge1xuICAgICAgICAgIGxldCBsb2NhbF9jb2VmZiA9IGYzMigyMDApO1xuICAgICAgICAgIGNvZWZmICs9IGxvY2FsX2NvZWZmO1xuICAgICAgICAgIGRpcl94ICs9IGxvY2FsX2NvZWZmICogZjMyKGZfeCk7XG4gICAgICAgICAgZGlyX3kgKz0gbG9jYWxfY29lZmYgKiBmMzIoZl95KTtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGlmIChmX3kgPj0gMCAmJiBmX3kgPCAke2hlaWdodH0gJiYgZl94ID49IDAgJiYgZl94IDwgJHt3aWR0aH0pIHtcbiAgICAgICAgICBsZXQgZl9pID0gZl95ICogJHt3aWR0aH0gKyBmX3g7XG4gICAgICAgICAgbGV0IGludGVuc2l0eSA9IHBpeGVsVG9JbnRlbnNpdHkoaW50ZW5zaXR5X21hcFtmX2ldKTtcbiAgICAgICAgICBsZXQgbG9jYWxfY29lZmYgPSBmMzIoMTAwKSAqIGludGVuc2l0eTtcbiAgICAgICAgICBjb2VmZiArPSBsb2NhbF9jb2VmZjtcbiAgICAgICAgICBkaXJfeCArPSBsb2NhbF9jb2VmZiAqIGYzMihmX3gpO1xuICAgICAgICAgIGRpcl95ICs9IGxvY2FsX2NvZWZmICogZjMyKGZfeSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICBsZXQgdG90YWxfY29lZmYgPSBjb2VmZjtcbiAgICBpZiAodG90YWxfY29lZmYgIT0gMCkge1xuICAgICAgdmFyIGRfeCA9IGRpcl94IC8gdG90YWxfY29lZmYgLSB4O1xuICAgICAgdmFyIGRfeSA9IGRpcl95IC8gdG90YWxfY29lZmYgLSB5O1xuXG4gICAgICBsZXQgZGlzdDIgPSBkX3ggKiBkX3ggKyBkX3kgKiBkX3k7XG4gICAgICBsZXQgbWF4X2Rpc3QyID0gZjMyKHJlZ2lvbiAqIHJlZ2lvbik7XG5cbiAgICAgIHZhciBzcGVlZCA9IGRpc3QyIC8gbWF4X2Rpc3QyO1xuICAgICAgaWYgKGRpc3QyIDwgZjMyKDUpKSB7XG4gICAgICAgIHNwZWVkID0gZjMyKDEpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgc3BlZWQgPSBmMzIoMC41KTtcbiAgICAgIH1cblxuICAgICAgZF94ICo9IHNwZWVkO1xuICAgICAgZF95ICo9IHNwZWVkO1xuXG4gICAgICB4X3Bvc19vdXRbaV0gPSB4ICsgZF94O1xuICAgICAgeV9wb3Nfb3V0W2ldID0geSArIGRfeTtcbiAgICB9IGVsc2Uge1xuICAgICAgeF9wb3Nfb3V0W2ldID0geDtcbiAgICAgIHlfcG9zX291dFtpXSA9IHk7XG4gICAgfVxuICB9XG59XG5gXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gYnVpbGQobl9lbGVtczogbnVtYmVyLCBncmlkX3g6IG51bWJlciwgZ3JpZF9zcGFjaW5nOiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGU+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDI1NilcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuICBnbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxuICBsb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIGlmIChnbG9iYWxfaWQueCA8ICR7bl9lbGVtc30pIHtcbiAgICAgIGxldCB5ID0gZ2xvYmFsX2lkLnggLyAke2dyaWRfeH07XG4gICAgICBsZXQgeCA9IGdsb2JhbF9pZC54ICUgJHtncmlkX3h9O1xuXG4gICAgICB4X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeCAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gICAgICB5X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeSAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gIH1cbn1cbmBcbn1cbiIsImltcG9ydCB7IGJ1aWxkIGFzIGluaXRidWlsZCB9IGZyb20gJy4vaW5pdCc7XG5pbXBvcnQgeyBidWlsZCBhcyBmb3JjZXNidWlsZCB9IGZyb20gJy4vZm9yY2VzJztcbmltcG9ydCB7IGJ1aWxkIGFzIHJlbmRlcmJ1aWxkIH0gZnJvbSAnLi9yZW5kZXInO1xuXG5jb25zdCBlZGdlcyA9IFtcbiAgWy0xLCAwXSxcbiAgWzEsIDBdLFxuICBbMCwgLTFdLFxuICBbMCwgMV0sXG4gIFsxLCAxXSxcbiAgWy0xLCAtMV0sXG5dO1xuXG5leHBvcnQgY2xhc3MgTWVzaERlZm9ybWF0aW9uIHtcbiAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG4gIGdyaWRfeDogbnVtYmVyO1xuICBncmlkX3k6IG51bWJlcjtcbiAgZ3JpZF9zcGFjaW5nOiBudW1iZXI7XG4gIG1pbl9kaXN0OiBudW1iZXI7XG4gIG1heF9kaXN0OiBudW1iZXI7XG4gIHJhZGl1czogbnVtYmVyO1xuXG4gIHJlaW5pdDogYm9vbGVhbjtcbiAgZHJhd19lZGdlczogYm9vbGVhbjtcblxuICBuX2VsZW1zOiBudW1iZXI7XG4gIHhfcG9zOiBGbG9hdDMyQXJyYXk7XG4gIHlfcG9zOiBGbG9hdDMyQXJyYXk7XG5cbiAgaW5pdF9tb2R1bGU6IEdQVVNoYWRlck1vZHVsZTtcbiAgaW5pdEJpbmRHcm91cExheW91dDogR1BVQmluZEdyb3VwTGF5b3V0O1xuICBpbml0X2NvbXB1dGVQaXBlbGluZTogR1BVQ29tcHV0ZVBpcGVsaW5lO1xuXG4gIGluaXRpYWxpemF0aW9uX2RvbmU6IFByb21pc2U8dm9pZD47XG4gIGRldmljZTogR1BVRGV2aWNlO1xuICBmb3JjZV9tb2R1bGU6IEdQVVNoYWRlck1vZHVsZTtcbiAgYWN0aXZlX2J1ZmZlcl9pZDogbnVtYmVyO1xuICB4X3Bvc19idWZmZXJzOiBbR1BVQnVmZmVyLCBHUFVCdWZmZXJdO1xuICB5X3Bvc19idWZmZXJzOiBbR1BVQnVmZmVyLCBHUFVCdWZmZXJdO1xuICAvLyBCdWZmZXJzIHRvIHJlYWQgdmFsdWVzIGJhY2sgdG8gdGhlIENQVSBmb3IgZHJhd2luZ1xuICBzdGFnaW5nX3hfYnVmOiBHUFVCdWZmZXI7XG4gIHN0YWdpbmdfeV9idWY6IEdQVUJ1ZmZlcjtcbiAgLy8gQnVmZmVyIHRvIHdyaXRlIHZhbHVlIHRvIGZyb20gdGhlIENQVSBmb3IgYWRqdXN0aW5nIHdlaWdodHNcbiAgc3RhZ2luZ19pbnRlbnNpdHlfYnVmOiBHUFVCdWZmZXI7XG4gIGludGVuc2l0eV9tYXBfYnVmOiBHUFVCdWZmZXI7XG5cbiAgb2Zmc2V0X2J1ZjogR1BVQnVmZmVyO1xuICBzdHJpZGVfYnVmOiBHUFVCdWZmZXI7XG5cbiAgaW1hZ2VfYnVmOiBHUFVCdWZmZXI7XG4gIGltYWdlX3Jlc2V0OiBGbG9hdDMyQXJyYXk7XG5cbiAgZm9yY2VfYmluZF9ncm91cF9sYXlvdXQ6IEdQVUJpbmRHcm91cExheW91dDtcblxuXG4gIHJlbmRlcl90b19idWZfbW9kdWxlOiBHUFVTaGFkZXJNb2R1bGU7XG4gIHJlbmRlcl90b19idWZfYmluZF9ncm91cF9sYXlvdXQ6IEdQVUJpbmRHcm91cExheW91dDtcbiAgcmVuZGVyX3RvX2J1Zl9jb21wdXRlX3BpcGVsaW5lOiBHUFVDb21wdXRlUGlwZWxpbmU7XG5cbiAgcmVuZGVyX21vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICByZW5kZXJfcGlwZWxpbmU6IEdQVVJlbmRlclBpcGVsaW5lO1xuICByZW5kZXJfcGFzc19kZXNjcmlwdG9yOiBHUFVSZW5kZXJQYXNzRGVzY3JpcHRvcjtcbiAgdmVydGV4X2J1ZmZlcjogR1BVQnVmZmVyO1xuICByZW5kZXJfYmluZF9ncm91cF9sYXlvdXQ6IEdQVUJpbmRHcm91cExheW91dDtcblxuICBjb25zdHJ1Y3RvcihcbiAgICBjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCxcbiAgICBncmlkX3g6IG51bWJlcixcbiAgICBncmlkX3k6IG51bWJlcixcbiAgICBncmlkX3NwYWNpbmc6IG51bWJlcixcbiAgICBtaW5fZGlzdDogbnVtYmVyLFxuICAgIG1heF9kaXN0OiBudW1iZXIsXG4gICAgcmFkaXVzOiBudW1iZXJcbiAgKSB7XG4gICAgdGhpcy5jdHggPSBjdHg7XG4gICAgdGhpcy5ncmlkX3ggPSBncmlkX3g7XG4gICAgdGhpcy5ncmlkX3kgPSBncmlkX3k7XG4gICAgdGhpcy5ncmlkX3NwYWNpbmcgPSBncmlkX3NwYWNpbmc7XG4gICAgdGhpcy5taW5fZGlzdCA9IG1pbl9kaXN0O1xuICAgIHRoaXMubWF4X2Rpc3QgPSBtYXhfZGlzdDtcbiAgICB0aGlzLnJhZGl1cyA9IHJhZGl1cztcblxuICAgIHRoaXMuZHJhd19lZGdlcyA9IHRydWU7XG4gICAgdGhpcy5yZWluaXQgPSBmYWxzZTtcblxuICAgIHRoaXMubl9lbGVtcyA9IHRoaXMuZ3JpZF94ICogdGhpcy5ncmlkX3k7XG5cbiAgICB0aGlzLmluaXRpYWxpemF0aW9uX2RvbmUgPSB0aGlzLmFzeW5jX2luaXQoKTtcbiAgfVxuXG4gIGFzeW5jIGFzeW5jX2luaXQoKSB7XG4gICAgY29uc3QgYWRhcHRlciA9IGF3YWl0IG5hdmlnYXRvci5ncHUucmVxdWVzdEFkYXB0ZXIoKTtcbiAgICB0aGlzLmRldmljZSA9IGF3YWl0IGFkYXB0ZXIucmVxdWVzdERldmljZSgpO1xuICAgIGNvbnNvbGUubG9nKFwiQ3JlYXRlIGNvbXB1dGUgc2hhZGVyXCIpO1xuICAgIHRoaXMuZm9yY2VfbW9kdWxlID0gdGhpcy5kZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcbiAgICAgIGNvZGU6IGZvcmNlc2J1aWxkKHRoaXMuY3R4LmNhbnZhcy53aWR0aCwgdGhpcy5jdHguY2FudmFzLmhlaWdodCwgdGhpcy5ncmlkX3gsIHRoaXMuZ3JpZF9zcGFjaW5nLCB0aGlzLm5fZWxlbXMpLFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBDcmVhdGUgY29tcHV0ZSBzaGFkZXJcIik7XG5cbiAgICB0aGlzLmFjdGl2ZV9idWZmZXJfaWQgPSAwO1xuICAgIHRoaXMueF9wb3NfYnVmZmVycyA9IFtcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInhfcG9zWzBdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieF9wb3NbMV1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgXTtcbiAgICB0aGlzLnlfcG9zX2J1ZmZlcnMgPSBbXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ5X3Bvc1swXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInlfcG9zWzFdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgIF07XG5cbiAgICB0aGlzLnN0YWdpbmdfeF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwic3RhZ2luZ194X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5NQVBfUkVBRCB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIHRoaXMuc3RhZ2luZ195X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX3lfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG5cbiAgICB0aGlzLnN0YWdpbmdfaW50ZW5zaXR5X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX2ludGVuc2l0eV9idWZcIixcbiAgICAgIHNpemU6IHRoaXMuY3R4LmNhbnZhcy53aWR0aCAqIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9XUklURSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDLFxuICAgIH0pO1xuICAgIHRoaXMuaW50ZW5zaXR5X21hcF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwiaW50ZW5zaXR5X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuXG4gICAgdGhpcy5vZmZzZXRfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcIm9mZnNldF9idWZcIixcbiAgICAgIHNpemU6IDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuVU5JRk9STSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIHRoaXMuc3RyaWRlX2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdHJpZGVfYnVmXCIsXG4gICAgICBzaXplOiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlVOSUZPUk0gfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcblxuICAgIHRoaXMuaW1hZ2VfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcImltYWdlX2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcbiAgICB0aGlzLmltYWdlX3Jlc2V0ID0gbmV3IEZsb2F0MzJBcnJheSh0aGlzLmN0eC5jYW52YXMud2lkdGggKiB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0ICogNCk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFsbG9jYXRlIGJ1ZmZlcnNcIik7XG5cbiAgICB0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMixcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDMsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA0LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJ1bmlmb3JtXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDYsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwidW5pZm9ybVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgLy8gaW50aWFsaXplIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdIGFuZFxuICAgIC8vIHRoaXMueV9wb3NfYnVmZmVyc1sqXSB0byBiZSBhIGdyaWRcbiAgICBjb25zdCBpbml0X3NoYWRlciA9IGluaXRidWlsZCh0aGlzLm5fZWxlbXMsIHRoaXMuZ3JpZF94LCB0aGlzLmdyaWRfc3BhY2luZyk7XG4gICAgdGhpcy5pbml0QmluZEdyb3VwTGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIGNvbnNvbGUubG9nKFwiQ3JlYXRlIGluaXQgc2hhZGVyXCIpO1xuICAgIHRoaXMuaW5pdF9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogaW5pdF9zaGFkZXIsXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIENyZWF0ZSBpbml0IHNoYWRlclwiKTtcbiAgICB0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgIGxhYmVsOiBcImNvbXB1dGUgZm9yY2VcIixcbiAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbdGhpcy5pbml0QmluZEdyb3VwTGF5b3V0XSxcbiAgICAgIH0pLFxuICAgICAgY29tcHV0ZToge1xuICAgICAgICBtb2R1bGU6IHRoaXMuaW5pdF9tb2R1bGUsXG4gICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgfSxcbiAgICB9KTtcbiAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG5cbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiB0aGlzLmluaXRCaW5kR3JvdXBMYXlvdXQsXG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH1cbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luQ29tcHV0ZVBhc3MoKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZSh0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoTWF0aC5jZWlsKHRoaXMubl9lbGVtcyAvIDI1NikpO1xuICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemVcbiAgICApO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemVcbiAgICApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcblxuXG4gICAgLy8gaW50aWFsaXplIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdIGFuZFxuICAgIC8vIHRoaXMueV9wb3NfYnVmZmVyc1sqXSB0byBiZSBhIGdyaWRcbiAgICBjb25zdCByZW5kZXJfdG9fYnVmZmVyX3NoYWRlciA9IHJlbmRlcmJ1aWxkKFxuICAgICAgdGhpcy5jdHguY2FudmFzLndpZHRoLFxuICAgICAgdGhpcy5jdHguY2FudmFzLmhlaWdodCxcbiAgICAgIHRoaXMuZ3JpZF94LFxuICAgICAgdGhpcy5ncmlkX3NwYWNpbmcsXG4gICAgICB0aGlzLm5fZWxlbXMsXG4gICAgICB0aGlzLnJhZGl1c1xuICAgICk7XG4gICAgdGhpcy5yZW5kZXJfdG9fYnVmX2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMixcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDMsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA0LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInVuaWZvcm1cIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIHRoaXMucmVuZGVyX3RvX2J1Zl9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogcmVuZGVyX3RvX2J1ZmZlcl9zaGFkZXIsXG4gICAgfSk7XG4gICAgdGhpcy5yZW5kZXJfdG9fYnVmX2NvbXB1dGVfcGlwZWxpbmUgPSB0aGlzLmRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgbGFiZWw6IFwiY29tcHV0ZSBmb3JjZVwiLFxuICAgICAgbGF5b3V0OiB0aGlzLmRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XG4gICAgICAgIGJpbmRHcm91cExheW91dHM6IFt0aGlzLnJlbmRlcl90b19idWZfYmluZF9ncm91cF9sYXlvdXRdLFxuICAgICAgfSksXG4gICAgICBjb21wdXRlOiB7XG4gICAgICAgIG1vZHVsZTogdGhpcy5yZW5kZXJfdG9fYnVmX21vZHVsZSxcbiAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICB9LFxuICAgIH0pO1xuXG5cbiAgICAodGhpcy5jdHggYXMgYW55KS5jb25maWd1cmUoe1xuICAgICAgZGV2aWNlOiB0aGlzLmRldmljZSxcbiAgICAgIGZvcm1hdDogbmF2aWdhdG9yLmdwdS5nZXRQcmVmZXJyZWRDYW52YXNGb3JtYXQoKSxcbiAgICB9KTtcblxuICAgIGNvbnN0IHZlcnRpY2VzID0gbmV3IEZsb2F0MzJBcnJheShbXG4gICAgICAtMSwgLTEsXG4gICAgICAxLCAtMSxcbiAgICAgIC0xLCAxLFxuICAgICAgLTEsIDEsXG4gICAgICAxLCAtMSxcbiAgICAgIDEsIDEsXG4gICAgXSk7XG4gICAgdGhpcy52ZXJ0ZXhfYnVmZmVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIHNpemU6IHZlcnRpY2VzLmJ5dGVMZW5ndGgsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuVkVSVEVYIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgdGhpcy5kZXZpY2UucXVldWUud3JpdGVCdWZmZXIodGhpcy52ZXJ0ZXhfYnVmZmVyLCAwLCB2ZXJ0aWNlcywgMCwgdmVydGljZXMubGVuZ3RoKTtcblxuICAgIGNvbnN0IHNoYWRlcnMgPSBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiBpbWFnZV9tYXA6IGFycmF5PGYzMj47XG5cbkB2ZXJ0ZXhcbmZuIHZlcnRleF9tYWluKEBsb2NhdGlvbigwKSBwb3NpdGlvbjogdmVjMmYpIC0+IEBidWlsdGluKHBvc2l0aW9uKSB2ZWM0Zlxue1xuICByZXR1cm4gdmVjNGYocG9zaXRpb24ueCwgcG9zaXRpb24ueSwgMCwgMSk7XG59XG5cbkBmcmFnbWVudFxuZm4gZnJhZ21lbnRfbWFpbihAYnVpbHRpbihwb3NpdGlvbikgcG9zaXRpb246IHZlYzRmKSAtPiBAbG9jYXRpb24oMCkgdmVjNGZcbntcbiAgbGV0IGkgPSA0ICogdTMyKHBvc2l0aW9uLnggLSA1MDAgKyBwb3NpdGlvbi55ICogJHt0aGlzLmN0eC5jYW52YXMud2lkdGh9KTtcbiAgcmV0dXJuIHZlYzRmKFxuICAgIGltYWdlX21hcFtpICsgMF0sXG4gICAgaW1hZ2VfbWFwW2kgKyAxXSxcbiAgICBpbWFnZV9tYXBbaSArIDJdLFxuICAgIGltYWdlX21hcFtpICsgM10pO1xufVxuYDtcbiAgICBjb25zdCBzaGFkZXJNb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogc2hhZGVycyxcbiAgICB9KTtcblxuICAgIHRoaXMucmVuZGVyX2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGxhYmVsOiAncmVuZGVyX2JpbmRfZ3JvdXBfbGF5b3V0JyxcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuRlJBR01FTlQsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIGNvbnN0IHZlcnRleEJ1ZmZlcnMgPSBbXG4gICAgICB7XG4gICAgICAgIGF0dHJpYnV0ZXM6IFtcbiAgICAgICAgICB7XG4gICAgICAgICAgICBzaGFkZXJMb2NhdGlvbjogMCwgLy8gcG9zaXRpb25cbiAgICAgICAgICAgIG9mZnNldDogMCxcbiAgICAgICAgICAgIGZvcm1hdDogXCJmbG9hdDMyeDJcIixcbiAgICAgICAgICB9LFxuICAgICAgICBdLFxuICAgICAgICBhcnJheVN0cmlkZTogNCAqIDIsXG4gICAgICAgIHN0ZXBNb2RlOiBcInZlcnRleFwiLFxuICAgICAgfSxcbiAgICBdO1xuICAgIGNvbnN0IHBpcGVsaW5lRGVzY3JpcHRvciA9IHtcbiAgICAgIHZlcnRleDoge1xuICAgICAgICBtb2R1bGU6IHNoYWRlck1vZHVsZSxcbiAgICAgICAgZW50cnlQb2ludDogXCJ2ZXJ0ZXhfbWFpblwiLFxuICAgICAgICBidWZmZXJzOiB2ZXJ0ZXhCdWZmZXJzLFxuICAgICAgfSxcbiAgICAgIGZyYWdtZW50OiB7XG4gICAgICAgIG1vZHVsZTogc2hhZGVyTW9kdWxlLFxuICAgICAgICBlbnRyeVBvaW50OiBcImZyYWdtZW50X21haW5cIixcbiAgICAgICAgdGFyZ2V0czogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIGZvcm1hdDogbmF2aWdhdG9yLmdwdS5nZXRQcmVmZXJyZWRDYW52YXNGb3JtYXQoKSxcbiAgICAgICAgICB9LFxuICAgICAgICBdLFxuICAgICAgfSxcbiAgICAgIHByaW1pdGl2ZToge1xuICAgICAgICB0b3BvbG9neTogXCJ0cmlhbmdsZS1saXN0XCIsXG4gICAgICB9LFxuICAgICAgbGF5b3V0OiB0aGlzLmRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XG4gICAgICAgIGxhYmVsOiAncmVuZGVyX2JpbmRfZ3JvdXBfcGlwZWxpbmVfbGF5b3V0JyxcbiAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3RoaXMucmVuZGVyX2JpbmRfZ3JvdXBfbGF5b3V0XSxcbiAgICAgIH0pXG4gICAgfTtcblxuICAgIHRoaXMucmVuZGVyX3BpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlUmVuZGVyUGlwZWxpbmUoXG4gICAgICBwaXBlbGluZURlc2NyaXB0b3IgYXMgR1BVUmVuZGVyUGlwZWxpbmVEZXNjcmlwdG9yKTtcblxuICAgIGNvbnN0IGNsZWFyQ29sb3IgPSB7IHI6IDAuMCwgZzogMC41LCBiOiAxLjAsIGE6IDEuMCB9O1xuXG4gICAgdGhpcy5yZW5kZXJfcGFzc19kZXNjcmlwdG9yID0ge1xuICAgICAgY29sb3JBdHRhY2htZW50czogW1xuICAgICAgICB7XG4gICAgICAgICAgY2xlYXJWYWx1ZTogY2xlYXJDb2xvcixcbiAgICAgICAgICBsb2FkT3A6IFwiY2xlYXJcIixcbiAgICAgICAgICBzdG9yZU9wOiBcInN0b3JlXCIsXG4gICAgICAgICAgdmlldzogKHRoaXMuY3R4IGFzIGFueSkuZ2V0Q3VycmVudFRleHR1cmUoKS5jcmVhdGVWaWV3KCksXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0gYXMgR1BVUmVuZGVyUGFzc0Rlc2NyaXB0b3I7XG5cbiAgICBjb25zb2xlLmxvZyhcImRvbmUgYXN5bmMgaW5pdFwiKTtcbiAgfVxuXG5cbiAgYXN5bmMgYXBwbHlGb3JjZShjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCkge1xuICAgIGxldCBpZGF0YSA9IGN0eC5nZXRJbWFnZURhdGEoMCwgMCwgY3R4LmNhbnZhcy53aWR0aCwgY3R4LmNhbnZhcy5oZWlnaHQpLmRhdGE7XG4gICAgLy8gY29uc29sZS5sb2coYGIwICR7aWRhdGFbMF19LCAke2lkYXRhWzFdfSwgJHtpZGF0YVsyXX0sICR7aWRhdGFbM119YCk7XG4gICAgLy8gY29uc29sZS5sb2coYFdyaXRpbmcgJHt0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemV9LyR7aWRhdGEubGVuZ3RofSBieXRlcyBmb3IgaW1hcGApO1xuICAgIGxldCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKFxuICAgICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiwgMCwgaWRhdGEuYnVmZmVyLCAwLCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemUpO1xuICAgICh3aW5kb3cgYXMgYW55KS53cml0ZV90aW1lICs9IHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQ7XG5cbiAgICBsZXQgaW5wdXRfeCA9IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBpbnB1dF95ID0gdGhpcy55X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG4gICAgbGV0IG91dHB1dF94ID0gdGhpcy54X3Bvc19idWZmZXJzWzEgLSB0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBvdXRwdXRfeSA9IHRoaXMueV9wb3NfYnVmZmVyc1sxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcblxuXG4gICAgaWYgKHRoaXMucmVpbml0KSB7XG4gICAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgICBsYXlvdXQ6IHRoaXMuaW5pdEJpbmRHcm91cExheW91dCxcbiAgICAgICAgZW50cmllczogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgICBidWZmZXI6IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9LFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIGJpbmRpbmc6IDEsXG4gICAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgICBidWZmZXI6IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9XG4gICAgICAgIF0sXG4gICAgICB9KTtcblxuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZSh0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKE1hdGguY2VpbCh0aGlzLm5fZWxlbXMgLyAyNTYpKTtcbiAgICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgICAgdGhpcy5yZWluaXQgPSBmYWxzZTtcbiAgICB9XG5cblxuICAgIGNvbnN0IHdnX3ggPSA2NDtcbiAgICBjb25zdCBzdHJpZGUgPSA4O1xuICAgIGNvbnN0IGRpc3BhdGNoX3ggPSAoMjU2IC8gd2dfeCk7XG4gICAgbGV0IGJ1ZmZlcnMgPSBbaW5wdXRfeCwgaW5wdXRfeSwgdGhpcy5pbnRlbnNpdHlfbWFwX2J1Ziwgb3V0cHV0X3gsIG91dHB1dF95LCB0aGlzLm9mZnNldF9idWYsIHRoaXMuc3RyaWRlX2J1Zl07XG4gICAgY29uc3QgYmluZEdyb3VwID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcbiAgICAgIGxheW91dDogdGhpcy5mb3JjZV9iaW5kX2dyb3VwX2xheW91dCxcbiAgICAgIGVudHJpZXM6IGJ1ZmZlcnMubWFwKChiLCBpKSA9PiB7IHJldHVybiB7IGJpbmRpbmc6IGksIHJlc291cmNlOiB7IGJ1ZmZlcjogYiB9IH07IH0pXG4gICAgfSk7XG5cbiAgICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCB0aGlzLm5fZWxlbXM7IG9mZnNldCArPSAoc3RyaWRlICogd2dfeCAqIGRpc3BhdGNoX3gpKSB7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5vZmZzZXRfYnVmLCAwLCBuZXcgVWludDMyQXJyYXkoW29mZnNldF0pLmJ1ZmZlciwgMCwgNCk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5zdHJpZGVfYnVmLCAwLCBuZXcgVWludDMyQXJyYXkoW3N0cmlkZV0pLmJ1ZmZlciwgMCwgNCk7XG5cbiAgICAgIGNvbnN0IGNvbXB1dGVQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbXB1dGVQaXBlbGluZSh7XG4gICAgICAgIGxhYmVsOiBcImZvcmNlcGlwZWxpbmVcIixcbiAgICAgICAgbGF5b3V0OiB0aGlzLmRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XG4gICAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3RoaXMuZm9yY2VfYmluZF9ncm91cF9sYXlvdXRdLFxuICAgICAgICB9KSxcbiAgICAgICAgY29tcHV0ZToge1xuICAgICAgICAgIG1vZHVsZTogdGhpcy5mb3JjZV9tb2R1bGUsXG4gICAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICAgIH0sXG4gICAgICB9KTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiY3JlYXRlZCBwaXBlbGluZVwiKTtcblxuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZShjb21wdXRlUGlwZWxpbmUpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoZGlzcGF0Y2hfeCwgMSwgMSk7XG4gICAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb21wdXRlXCIpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgICAgLy8gY29uc29sZS5sb2coXCJxdWV1ZVwiLCBvZmZzZXQpO1xuICAgIH1cblxuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKHRoaXMuaW1hZ2VfYnVmLCAwLCB0aGlzLmltYWdlX3Jlc2V0KTtcbiAgICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCB0aGlzLm5fZWxlbXM7IG9mZnNldCArPSAyNTYpIHtcbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKFxuICAgICAgICB0aGlzLm9mZnNldF9idWYsIDAsIG5ldyBVaW50MzJBcnJheShbb2Zmc2V0XSkuYnVmZmVyLCAwLCA0KTtcblxuICAgICAgY29uc3QgY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgICAgbGFiZWw6IFwicmVuZGVyX3RvX2J1Zl9waXBlbGluZVwiLFxuICAgICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbdGhpcy5yZW5kZXJfdG9fYnVmX2JpbmRfZ3JvdXBfbGF5b3V0XSxcbiAgICAgICAgfSksXG4gICAgICAgIGNvbXB1dGU6IHtcbiAgICAgICAgICBtb2R1bGU6IHRoaXMucmVuZGVyX3RvX2J1Zl9tb2R1bGUsXG4gICAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICAgIH0sXG4gICAgICB9KTtcblxuICAgICAgbGV0IGJ1ZmZlcnMgPSBbb3V0cHV0X3gsIG91dHB1dF95LCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLCB0aGlzLmltYWdlX2J1ZiwgdGhpcy5vZmZzZXRfYnVmXTtcbiAgICAgIGNvbnN0IGJpbmRHcm91cCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cCh7XG4gICAgICAgIGxheW91dDogdGhpcy5yZW5kZXJfdG9fYnVmX2JpbmRfZ3JvdXBfbGF5b3V0LFxuICAgICAgICBlbnRyaWVzOiBidWZmZXJzLm1hcCgoYiwgaSkgPT4geyByZXR1cm4geyBiaW5kaW5nOiBpLCByZXNvdXJjZTogeyBidWZmZXI6IGIgfSB9OyB9KVxuICAgICAgfSk7XG5cbiAgICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAgIGNvbnN0IHBhc3NFbmNvZGVyID0gY29tbWFuZEVuY29kZXIuYmVnaW5Db21wdXRlUGFzcygpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUoY29tcHV0ZVBpcGVsaW5lKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKDEsIDEsIDEpO1xuICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG4gICAgfVxuXG4gICAgY29uc3QgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgIC8vIENvcHkgb3V0cHV0IGJ1ZmZlciB0byBzdGFnaW5nIGJ1ZmZlclxuICAgIGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIG91dHB1dF94LCAwLCB0aGlzLnN0YWdpbmdfeF9idWYsIDAsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcInggY29weWluZ1wiLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSwgXCJieXRlc1wiKTtcbiAgICBjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICBvdXRwdXRfeSwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLCAwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJ5IGNvcHlpbmdcIiwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUsIFwiYnl0ZXNcIik7XG4gICAgLy8gY29uc29sZS5sb2coXCJlbmNvZGVkIGNvcHkgdG8gYnVmZmVyc1wiLCB0aGlzLmFjdGl2ZV9idWZmZXJfaWQpO1xuXG4gICAgLy8gRW5kIGZyYW1lIGJ5IHBhc3NpbmcgYXJyYXkgb2YgY29tbWFuZCBidWZmZXJzIHRvIGNvbW1hbmQgcXVldWUgZm9yIGV4ZWN1dGlvblxuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImRvbmUgc3VibWl0IHRvIHF1ZXVlXCIpO1xuICAgIHtcbiAgICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAgIGNvbnN0IHBhc3NFbmNvZGVyID0gY29tbWFuZEVuY29kZXIuYmVnaW5SZW5kZXJQYXNzKHRoaXMucmVuZGVyX3Bhc3NfZGVzY3JpcHRvcik7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZSh0aGlzLnJlbmRlcl9waXBlbGluZSk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRWZXJ0ZXhCdWZmZXIoMCwgdGhpcy52ZXJ0ZXhfYnVmZmVyKTtcbiAgICAgIGNvbnN0IGJpbmRHcm91cCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cCh7XG4gICAgICAgIGxhYmVsOiAncmVuZGVyX2JpbmRfZ3JvdXAnLFxuICAgICAgICBsYXlvdXQ6IHRoaXMucmVuZGVyX2JpbmRfZ3JvdXBfbGF5b3V0LFxuICAgICAgICBlbnRyaWVzOiBbe2JpbmRpbmc6IDAsIHJlc291cmNlOiB7YnVmZmVyOiB0aGlzLmltYWdlX2J1Zn19XSxcbiAgICAgIH0pO1xuICAgICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgICBwYXNzRW5jb2Rlci5kcmF3KDYpO1xuICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XG5cbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICB9XG5cbiAgICAvLyBTd2FwIGlucHV0IGFuZCBvdXRwdXQ6XG4gICAgdGhpcy5hY3RpdmVfYnVmZmVyX2lkID0gMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZDtcbiAgfVxufVxuIiwiZXhwb3J0IGZ1bmN0aW9uIGJ1aWxkKFxuICB3aWR0aDogbnVtYmVyLFxuICBoZWlnaHQ6IG51bWJlcixcbiAgZ3JpZF94OiBudW1iZXIsXG4gIGdyaWRfc3BhY2luZzogbnVtYmVyLFxuICBuX2VsZW1zOiBudW1iZXIsXG4gIHJhZGl1czogbnVtYmVyLFxuKSB7XG4gIHJldHVybiBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygxKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDIpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZT4gaW50ZW5zaXR5X21hcDogYXJyYXk8dTMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDMpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IGltYWdlX21hcDogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDQpXG52YXI8dW5pZm9ybT5vZmZzZXQ6IHUzMjtcblxuXG5AY29tcHV0ZSBAd29ya2dyb3VwX3NpemUoMjU2LCAxLCAxKVxuZm4gbWFpbihcbiAgQGJ1aWx0aW4oZ2xvYmFsX2ludm9jYXRpb25faWQpXG5nbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxubG9jYWxfaWQgOiB2ZWMzdSxcbikge1xuICAvLyBDb29yZGluYXRlcyBvZiBwYXJ0aWNsZSBmb3IgdGhpcyB0aHJlYWRcbiAgbGV0IGkgPSBvZmZzZXQgKyBnbG9iYWxfaWQueDtcblxuICBsZXQgZ3JpZF94ID0gaSAlICR7Z3JpZF94fTtcbiAgbGV0IGdyaWRfeSA9IGkgLyAke2dyaWRfeH07XG4gIGxldCBvcmlnaW5feCA9IGdyaWRfeCAqICR7Z3JpZF9zcGFjaW5nfTtcbiAgbGV0IG9yaWdpbl95ID0gZ3JpZF95ICogJHtncmlkX3NwYWNpbmd9O1xuXG4gIGlmIChpID4gJHtuX2VsZW1zfSkge1xuICAgIHJldHVybjtcbiAgfVxuXG4gIC8vIHZhciBweCA9IGludGVuc2l0eV9tYXBbb3JpZ2luX3ggKyBvcmlnaW5feSAqICR7d2lkdGh9XTtcbiAgLy8gbGV0IHIgPSBmMzIocHggJSAyNTYpIC8gMjU2O1xuICAvLyBweCAvPSB1MzIoMjU2KTtcbiAgLy8gbGV0IGcgPSBmMzIocHggJSAyNTYpIC8gMjU2O1xuICAvLyBweCAvPSB1MzIoMjU2KTtcbiAgLy8gbGV0IGIgPSBmMzIocHggJSAyNTYpIC8gMjU2O1xuICBsZXQgciA9IGYzMigxKTtcbiAgbGV0IGcgPSBmMzIoMCk7XG4gIGxldCBiID0gZjMyKDApO1xuXG4gIGxldCB4ID0geF9wb3NbaV07XG4gIGxldCB5ID0geV9wb3NbaV07XG5cbiAgZm9yICh2YXIgeWkgPSAtMjA7IHlpIDw9IDIwOyB5aSsrKSB7XG4gICAgZm9yICh2YXIgeGkgPSAtMjA7IHhpIDw9IDIwOyB4aSsrKSB7XG4gICAgICBsZXQgb3ggPSBpMzIob3JpZ2luX3gpICsgeGk7XG4gICAgICBsZXQgb3kgPSBpMzIob3JpZ2luX3kpICsgeWk7XG4gICAgICBpZiAob3ggPCAwIHx8IG94ID4gJHt3aWR0aH0gfHwgb3kgPCAwIHx8IG95ID4gJHtoZWlnaHR9KSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbGV0IGR4ID0gZjMyKG94KSAtIHg7XG4gICAgICBsZXQgZHkgPSBmMzIob3kpIC0geTtcblxuICAgICAgbGV0IGQgPSBzcXJ0KGR4ICogZHggKyBkeSAqIGR5KTtcbiAgICAgIGlmIChhYnMoZCAtICR7cmFkaXVzfSkgPCAxKSB7XG4gICAgICAgIGxldCBvdXQgPSBveCArIG95ICogJHt3aWR0aH07XG4gICAgICAgIGltYWdlX21hcFs0ICogb3V0ICsgMF0gPSByO1xuICAgICAgICBpbWFnZV9tYXBbNCAqIG91dCArIDFdID0gZztcbiAgICAgICAgaW1hZ2VfbWFwWzQgKiBvdXQgKyAyXSA9IGI7XG4gICAgICAgIGltYWdlX21hcFs0ICogb3V0ICsgM10gPSBmMzIoMSk7XG4gICAgICB9XG4gICAgfVxuICB9XG59XG5gXG59XG4iLCIvLyBUaGUgbW9kdWxlIGNhY2hlXG52YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG5cbi8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG5mdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuXHR2YXIgY2FjaGVkTW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXTtcblx0aWYgKGNhY2hlZE1vZHVsZSAhPT0gdW5kZWZpbmVkKSB7XG5cdFx0cmV0dXJuIGNhY2hlZE1vZHVsZS5leHBvcnRzO1xuXHR9XG5cdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG5cdHZhciBtb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdID0ge1xuXHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcblx0XHQvLyBubyBtb2R1bGUubG9hZGVkIG5lZWRlZFxuXHRcdGV4cG9ydHM6IHt9XG5cdH07XG5cblx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG5cdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG5cdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG5cdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbn1cblxuIiwiaW1wb3J0IHsgTWVzaERlZm9ybWF0aW9uIH0gZnJvbSAnLi9tZXNoZGVmb3JtYXRpb24nO1xuXG5sZXQgc3RvcCA9IGZhbHNlO1xuXG5hc3luYyBmdW5jdGlvbiBzZXR1cFdlYmNhbSgpIHtcbiAgY29uc3QgdmlkZW8gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidmlkZW9cIik7XG4gIGNvbnN0IGNvbnN0cmFpbnRzID0geyB2aWRlbzogdHJ1ZSB9XG5cbiAgdHJ5IHtcbiAgICBpZiAodmlkZW8uc3JjT2JqZWN0KSB7XG4gICAgICBjb25zdCBzdHJlYW0gPSB2aWRlby5zcmNPYmplY3Q7XG4gICAgICBzdHJlYW0uZ2V0VHJhY2tzKCkuZm9yRWFjaChmdW5jdGlvbih0cmFjazogYW55KSB7XG4gICAgICAgIHRyYWNrLnN0b3AoKTtcbiAgICAgIH0pO1xuICAgICAgdmlkZW8uc3JjT2JqZWN0ID0gbnVsbDtcbiAgICB9XG5cbiAgICBjb25zdCBzdHJlYW0gPSBhd2FpdCBuYXZpZ2F0b3IubWVkaWFEZXZpY2VzLmdldFVzZXJNZWRpYShjb25zdHJhaW50cyk7XG4gICAgdmlkZW8uc3JjT2JqZWN0ID0gc3RyZWFtO1xuICAgIHZpZGVvLnBsYXkoKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgYWxlcnQoXCJFcnJvciBpbml0aWFsaXppbmcgd2ViY2FtISBcIiArIGVycik7XG4gICAgY29uc29sZS5sb2coZXJyKTtcbiAgfVxuICByZXR1cm4gdmlkZW87XG59XG5cbmFzeW5jIGZ1bmN0aW9uIG1haW4oY29udGFpbmVyOiBIVE1MRWxlbWVudCkge1xuICBsZXQgY2FudmFzID0gKGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJjYW52YXNcIikgYXMgSFRNTENhbnZhc0VsZW1lbnQpO1xuICBjYW52YXMud2lkdGggPSAxMDAwO1xuICBjYW52YXMuaGVpZ2h0ID0gMTAwMDtcbiAgY29udGFpbmVyLmFwcGVuZENoaWxkKGNhbnZhcyk7XG5cbiAgLy8gbGV0IGN0eCA9IGNhbnZhcy5nZXRDb250ZXh0KFwiMmRcIik7XG4gIGxldCBjdHggPSBjYW52YXMuZ2V0Q29udGV4dChcIndlYmdwdVwiKTtcbiAgY29uc29sZS5sb2coXCJDcmVhdGVkIGNvbnRleHQgZm9yIG1haW4gY2FudmFzXCIpO1xuXG4gIGxldCBjYW52YXMyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImNhbnZhc1wiKSBhcyBIVE1MQ2FudmFzRWxlbWVudDtcbiAgY2FudmFzMi53aWR0aCA9IDEwMDA7XG4gIGNhbnZhczIuaGVpZ2h0ID0gMTAwMDtcbiAgbGV0IGN0eDIgPSBjYW52YXMyLmdldENvbnRleHQoXCIyZFwiKTtcbiAgLy8gZG9jdW1lbnQuYm9keS5hcHBlbmRDaGlsZChjYW52YXMyKTtcblxuICBjYW52YXMuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsIChlKSA9PiB7XG4gICAgbGV0IGVsID0gZS50YXJnZXQgYXMgSFRNTENhbnZhc0VsZW1lbnQ7XG4gICAgY29uc3QgcmVjdCA9IGVsLmdldEJvdW5kaW5nQ2xpZW50UmVjdCgpO1xuICAgIGNvbnN0IHggPSBlbC53aWR0aCAqIChlLmNsaWVudFggLSByZWN0LmxlZnQpIC8gcmVjdC53aWR0aDtcbiAgICBjb25zdCB5ID0gZWwuaGVpZ2h0ICogKGUuY2xpZW50WSAtIHJlY3QudG9wKSAvIHJlY3QuaGVpZ2h0O1xuXG4gICAgY3R4Mi5iZWdpblBhdGgoKTtcbiAgICBjdHgyLmZpbGxTdHlsZSA9IFwiYmxhY2tcIjtcbiAgICBjdHgyLmFyYyh4LCB5LCAxMDAsIDAsIDIgKiBNYXRoLlBJKTtcbiAgICBjdHgyLmZpbGwoKTtcbiAgfSk7XG5cbiAgY29udGFpbmVyLmFwcGVuZENoaWxkKGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJiclwiKSk7XG5cbiAgY29uc3QgY2xlYXIgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiYnV0dG9uXCIpO1xuICBjbGVhci5pbm5lclRleHQgPSBcImNsZWFyXCI7XG4gIGNsZWFyLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgY3R4Mi5jbGVhclJlY3QoMCwgMCwgY3R4Mi5jYW52YXMud2lkdGgsIGN0eDIuY2FudmFzLmhlaWdodCk7XG4gIH0pO1xuICBjb250YWluZXIuYXBwZW5kQ2hpbGQoY2xlYXIpO1xuXG4gIGNvbnN0IGVkZ2VzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImJ1dHRvblwiKTtcbiAgZWRnZXMuaW5uZXJUZXh0ID0gXCJlZGdlc1wiO1xuICBlZGdlcy5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgIG1kLmRyYXdfZWRnZXMgPSAhbWQuZHJhd19lZGdlcztcbiAgfSk7XG4gIGNvbnRhaW5lci5hcHBlbmRDaGlsZChlZGdlcyk7XG5cbiAgY29uc3QgdmlkZW8gPSBhd2FpdCBzZXR1cFdlYmNhbSgpO1xuXG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBpbnRlcmFjdGl2ZSBjYW52YXNcIik7XG5cbiAgKHdpbmRvdyBhcyBhbnkpLm5fc3RlcHNfcGVyX2ZyYW1lID0gMTtcblxuICBsZXQgbl9lbGVtcyA9IDIwMDtcbiAgbGV0IHNwYWNpbmcgPSBjdHguY2FudmFzLndpZHRoIC8gbl9lbGVtcztcbiAgbGV0IG1kID0gbmV3IE1lc2hEZWZvcm1hdGlvbihjdHgsIG5fZWxlbXMsIG5fZWxlbXMsIHNwYWNpbmcsIHNwYWNpbmcgLyA0LCBzcGFjaW5nICogNCwgMSk7XG4gICh3aW5kb3cgYXMgYW55KS50X3Blcl9yZW5kZXIgPSAwO1xuICAod2luZG93IGFzIGFueSkubl9yZW5kZXJzID0gMDtcbiAgKHdpbmRvdyBhcyBhbnkpLndyaXRlX3RpbWUgPSAwO1xuICAod2luZG93IGFzIGFueSkuaW50ZXJ2YWwgPSAwO1xuICBsZXQgdGhldGEgPSAwO1xuICBsZXQgbGFzdF9zdGFydCA9IDA7XG4gIG1kLmluaXRpYWxpemF0aW9uX2RvbmUudGhlbigoKSA9PiB7XG4gICAgY29uc3QgZiA9IGFzeW5jICgpID0+IHtcbiAgICAgIGxldCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCAod2luZG93IGFzIGFueSkubl9zdGVwc19wZXJfZnJhbWU7IGkrKykge1xuICAgICAgICBhd2FpdCBtZC5hcHBseUZvcmNlKGN0eDIpO1xuICAgICAgfVxuICAgICAgbGV0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX3JlbmRlciArPSBlbmQgLSBzdGFydDtcbiAgICAgICh3aW5kb3cgYXMgYW55KS5uX3JlbmRlcnMgKz0gMTtcblxuXG4gICAgICBpZiAodmlkZW8ucmVhZHlTdGF0ZSA9PSA0KSB7XG4gICAgICAgIGN0eDIuZHJhd0ltYWdlKHZpZGVvLCAwLCAwLCBjdHgyLmNhbnZhcy53aWR0aCwgY3R4Mi5jYW52YXMuaGVpZ2h0KTtcbiAgICAgIH1cbiAgICAgIC8vIGN0eDIuY2xlYXJSZWN0KDAsIDAsIGN0eDIuY2FudmFzLndpZHRoLCBjdHgyLmNhbnZhcy5oZWlnaHQpO1xuICAgICAgbGV0IHggPSBNYXRoLnNpbih0aGV0YSkgKiA0NTAgKyA1MDA7XG4gICAgICBsZXQgeSA9IE1hdGguY29zKHRoZXRhKSAqIDQ1MCArIDUwMDtcbiAgICAgIGN0eDIuYmVnaW5QYXRoKCk7XG4gICAgICBjdHgyLmZpbGxTdHlsZSA9IFwiYmxhY2tcIjtcbiAgICAgIGN0eDIuYXJjKHgsIHksIDEwMCwgMCwgMiAqIE1hdGguUEkpO1xuICAgICAgY3R4Mi5maWxsKCk7XG5cbiAgICAgIHRoZXRhICs9IDAuMTtcblxuICAgICAgKHdpbmRvdyBhcyBhbnkpLmludGVydmFsICs9IHN0YXJ0IC0gbGFzdF9zdGFydDtcbiAgICAgIGxhc3Rfc3RhcnQgPSBzdGFydDtcbiAgICAgIGlmICghc3RvcCkge1xuICAgICAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZilcbiAgICAgIH1cbiAgICB9O1xuICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmKTtcblxuICAgICh3aW5kb3cgYXMgYW55KS50X3Blcl9kcmF3ID0gMDtcbiAgICAod2luZG93IGFzIGFueSkudF9wZXJfcmVhZCA9IDA7XG4gICAgKHdpbmRvdyBhcyBhbnkpLm5fZHJhd3MgPSAwO1xuICB9KTtcblxuICAod2luZG93IGFzIGFueSkuc3RhdHMgPSAoKSA9PiB7XG4gICAgbGV0IHcgPSB3aW5kb3cgYXMgYW55O1xuICAgIGNvbnNvbGUubG9nKFwiYXZnX2ludGVydmFsXCIsIHcuaW50ZXJ2YWwgLyB3Lm5fcmVuZGVycyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdfdF9wZXJfcmVuZGVyXCIsIHcudF9wZXJfcmVuZGVyIC8gdy5uX3JlbmRlcnMpO1xuICAgIGNvbnNvbGUubG9nKFwiYXZnX3RfcGVyX3dyaXRlXCIsIHcud3JpdGVfdGltZSAvIHcubl9yZW5kZXJzKTtcbiAgICBjb25zb2xlLmxvZyhcImF2Z190X3Blcl9kcmF3XCIsIHcudF9wZXJfZHJhdyAvIHcubl9kcmF3cyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdfdF9wZXJfcmVhZFwiLCB3LnRfcGVyX3JlYWQgLyB3Lm5fZHJhd3MpO1xuICB9XG5cblxuICBmdW5jdGlvbiBjYW5jZWwoKSB7XG4gICAgc3RvcCA9IHRydWU7XG4gIH1cbiAgKHdpbmRvdyBhcyBhbnkpLm1kID0gbWQ7XG4gICh3aW5kb3cgYXMgYW55KS5jdHgyID0gY3R4MjtcbiAgKHdpbmRvdyBhcyBhbnkpLmNhbmNlbCA9IGNhbmNlbDtcbn1cblxuZG9jdW1lbnQuYWRkRXZlbnRMaXN0ZW5lcihcIkRPTUNvbnRlbnRMb2FkZWRcIiwgKCkgPT4ge1xuICBsZXQgY29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJjb250YWluZXJcIik7XG4gIG1haW4oY29udGFpbmVyKTtcbn0pO1xuXG4iXSwibmFtZXMiOltdLCJzb3VyY2VSb290IjoiIn0=