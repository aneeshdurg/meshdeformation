/******/ (() => { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./src/constraint.ts":
/*!***************************!*\
  !*** ./src/constraint.ts ***!
  \***************************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.build = build;
const edges = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [1, 1],
    [-1, -1],
    [1, -1],
    [-1, 1],
];
function build(grid_x, grid_y, min_dist, max_dist) {
    let src = `
@group(0) @binding(0)
var<storage, read_write > x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write > y_pos: array<f32>;

@group(0) @binding(2)
var<storage, read_write > x_pos_out: array<f32>;

@group(0) @binding(3)
var<storage, read_write > y_pos_out: array<f32>;

@group(0) @binding(4)
var<uniform>offset: u32;

@group(0) @binding(5)
var<storage, read_write>debug: array<u32>;


@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id)
global_id : vec3u,

  @builtin(local_invocation_id)
local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let i = offset + global_id.x;
  let grid_y = i32(i / ${grid_x});
  let grid_x = i32(i % ${grid_x});

  let curr_x = x_pos_out[offset + global_id.x];
  let curr_y = y_pos_out[offset + global_id.x];
  let org_x = x_pos[offset + global_id.x];
  let org_y = y_pos[offset + global_id.x];

  var invalid = false;
`;
    for (let edge of edges) {
        src += `
  {
    let e_gx = grid_x + ${edge[0]};
    let e_gy = grid_y + ${edge[1]};
    if (e_gx > 0 && e_gx < ${grid_x} && e_gy > 0 && e_gy < ${grid_y}) {
      let e_i = e_gy * ${grid_x} + e_gx;
      let n_x = x_pos_out[e_i];
      let n_y = y_pos_out[e_i];
      let d_x = n_x - curr_x;
      let d_y = n_y - curr_y;

      let d_n2 = d_x * d_x + d_y * d_y;
      if (d_n2 < (${min_dist} * ${min_dist}) || d_n2 > (${max_dist} * ${max_dist})) {
        invalid = true;
      }
    }
  }
`;
    }
    src += `
  if (invalid) {
    x_pos_out[i] = x_pos[i];
    y_pos_out[i] = y_pos[i];
  }
`;
    src += "}\n";
    return src;
}


/***/ }),

/***/ "./src/forces.ts":
/*!***********************!*\
  !*** ./src/forces.ts ***!
  \***********************/
/***/ ((__unused_webpack_module, exports) => {


Object.defineProperty(exports, "__esModule", ({ value: true }));
exports.build = build;
function build(width, height, grid_x, grid_spacing) {
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
  let i = offset + global_id.x;

  let grid_x = i % ${grid_x};
  let grid_y = i / ${grid_x};

  let x = x_pos[i];
  let y = y_pos[i];

  var dir_x: f32 = 0;
  var dir_y: f32 = 0;
  var coeff: f32 = 0;

  // Coordinates to lookup in intensity_map
  for (var s_y: i32 = 0; s_y < 50; s_y++) {
    for (var s_z: i32 = 0; s_z < 50; s_z++) {
      let f_y = i32(floor(y)) + i32(50 * local_id.y) + s_y - 25;
      let f_x = i32(floor(x)) + i32(50 * local_id.z) + s_z - 25;
      let d_x: f32 = f32(f_x) - x;
      let d_y: f32 = f32(f_y) - y;
      let r2: f32 = d_x * d_x + d_y * d_y;
      let r: f32 = sqrt(r2);

      // Find the force exerted on the particle by contents of the intesity map.
      if (f_y >= 0 && f_y < ${height} && f_x >= 0 && f_x < ${width}) {
        let f_i = f_y * ${width} + f_x;
        let intensity = pixelToIntensity(intensity_map[f_i]);

        if (r != 0) {
          let local_coeff = 100 * intensity / r2;
          // atomicAdd(& coeff, u32(10000 * local_coeff));
          // atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
          // atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
          coeff += local_coeff;
          dir_x += local_coeff * d_x / r;
          dir_y += local_coeff * d_y / r;
        }
      }
    }
  }

  if (local_id.y == 0 && local_id.z == 0) {
    let origin_x = grid_x * ${grid_spacing};
    let origin_y = grid_y * ${grid_spacing};
    let d_x: f32 = f32(origin_x) - x;
    let d_y: f32 = f32(origin_y) - y;
    let r2: f32 = d_x * d_x + d_y * d_y;
    let r: f32 = sqrt(r2);

    let local_coeff = f32(5) / r2;
    // atomicAdd(& coeff, u32(10000 * local_coeff));
    // atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
    // atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
    coeff += local_coeff;
    dir_x += local_coeff * d_x / r;
    dir_y += local_coeff * d_y / r;
  }

  // Wait for all workgroup threads to finish simulating
  // workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    // let total_coeff = f32(atomicLoad(& coeff)) / 10000;
    let total_coeff = coeff;
    if (total_coeff != 0) {
      // var d_x = f32(atomicLoad(& dir_x)) / (1000 * total_coeff);
      // var d_y = f32(atomicLoad(& dir_y)) / (1000 * total_coeff);
      var d_x = dir_x / total_coeff;
      var d_y = dir_y / total_coeff;

      let s_dx = sign(d_x);
      let s_dy = sign(d_y);
      let a_dx = abs(d_x);
      let a_dy = abs(d_y);

      d_x = s_dx * min(a_dx, f32(0.5));
      d_y = s_dy * min(a_dy, f32(0.5));

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
const constraint_1 = __webpack_require__(/*! ./constraint */ "./src/constraint.ts");
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
    draw_edges;
    n_elems;
    x_pos;
    y_pos;
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
    force_bind_group_layout;
    constraint_module;
    constraint_bind_group_layout;
    debug_buf;
    constructor(ctx, grid_x, grid_y, grid_spacing, min_dist, max_dist, radius) {
        this.ctx = ctx;
        this.grid_x = grid_x;
        this.grid_y = grid_y;
        this.grid_spacing = grid_spacing;
        this.min_dist = min_dist;
        this.max_dist = max_dist;
        this.radius = radius;
        this.draw_edges = true;
        this.n_elems = this.grid_x * this.grid_y;
        this.initialization_done = this.async_init();
    }
    async async_init() {
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        console.log("Create compute shader");
        this.force_module = this.device.createShaderModule({
            code: (0, forces_1.build)(this.ctx.canvas.width, this.ctx.canvas.height, this.grid_x, this.grid_spacing),
        });
        console.log("done Create compute shader");
        this.debug_buf = this.device.createBuffer({
            label: "debug",
            size: this.n_elems * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
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
            ],
        });
        // intialize this.x_pos_buffers[this.active_buffer_id] and
        // this.y_pos_buffers[*] to be a grid
        const init_shader = (0, init_1.build)(this.n_elems, this.grid_x, this.grid_spacing);
        const initBindGroupLayout = this.device.createBindGroupLayout({
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
        const init_module = this.device.createShaderModule({
            code: init_shader,
        });
        console.log("done Create init shader");
        const computePipeline = this.device.createComputePipeline({
            label: "compute force",
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [initBindGroupLayout],
            }),
            compute: {
                module: init_module,
                entryPoint: "main",
            },
        });
        const commandEncoder = this.device.createCommandEncoder();
        const bindGroup = this.device.createBindGroup({
            layout: initBindGroupLayout,
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
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(this.n_elems / 256));
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(this.x_pos_buffers[this.active_buffer_id], 0, this.staging_x_buf, 0, this.staging_x_buf.size);
        commandEncoder.copyBufferToBuffer(this.y_pos_buffers[this.active_buffer_id], 0, this.staging_y_buf, 0, this.staging_y_buf.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.updateCPUpos();
        console.log("done async init");
        const constraint_src = (0, constraint_1.build)(this.grid_x, this.grid_y, this.min_dist, this.max_dist);
        // console.log(constraint_src);
        this.constraint_module = this.device.createShaderModule({
            code: constraint_src,
        });
        this.constraint_bind_group_layout = this.device.createBindGroupLayout({
            label: "constraint_bind_group_layout",
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
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
    }
    async updateCPUpos() {
        // console.log("Map buffers for reading");
        let m_x = this.staging_x_buf.mapAsync(GPUMapMode.READ, 0, this.staging_x_buf.size);
        let m_y = this.staging_y_buf.mapAsync(GPUMapMode.READ, 0, this.staging_y_buf.size);
        await m_x;
        await m_y;
        // console.log("copying x buffer to CPU");
        const copyArrayBufferX = this.staging_x_buf.getMappedRange(0, this.staging_x_buf.size);
        const dataX = copyArrayBufferX.slice();
        this.x_pos = new Float32Array(dataX);
        // console.log("copying y buffer to CPU");
        const copyArrayBufferY = this.staging_y_buf.getMappedRange(0, this.staging_y_buf.size);
        const dataY = copyArrayBufferY.slice();
        this.y_pos = new Float32Array(dataY);
        // console.log("unmap buffers");
        this.staging_x_buf.unmap();
        this.staging_y_buf.unmap();
        // console.log("Done updateCPUpos");
    }
    async applyForce(ctx) {
        let idata = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
        // console.log(`b0 ${idata[0]}, ${idata[1]}, ${idata[2]}, ${idata[3]}`);
        // console.log(`Writing ${this.intensity_map_buf.size}/${idata.length} bytes for imap`);
        this.device.queue.writeBuffer(this.intensity_map_buf, 0, idata.buffer, 0, this.intensity_map_buf.size);
        let input_x = this.x_pos_buffers[this.active_buffer_id];
        let input_y = this.y_pos_buffers[this.active_buffer_id];
        let output_x = this.x_pos_buffers[1 - this.active_buffer_id];
        let output_y = this.y_pos_buffers[1 - this.active_buffer_id];
        const wg_x = 64;
        const dispatch_x = 256 / wg_x;
        let buffers = [input_x, input_y, this.intensity_map_buf, output_x, output_y, this.offset_buf];
        const bindGroup = this.device.createBindGroup({
            layout: this.force_bind_group_layout,
            entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
        });
        for (let offset = 0; offset < this.n_elems; offset += (wg_x * dispatch_x)) {
            let input = new Uint32Array([offset]);
            this.device.queue.writeBuffer(this.offset_buf, 0, input.buffer, 0, 4);
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
        }
        let c_buffers = [input_x, input_y, output_x, output_y, this.offset_buf, this.debug_buf];
        const c_bindGroup = this.device.createBindGroup({
            label: "constraint_bind_group",
            layout: this.constraint_bind_group_layout,
            entries: c_buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
        });
        for (let offset = 0; offset < this.n_elems; offset += 256) {
            let input = new Uint32Array([offset]);
            this.device.queue.writeBuffer(this.offset_buf, 0, input.buffer, 0, 4);
            const computePipeline = this.device.createComputePipeline({
                label: "constraintpipeline",
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.constraint_bind_group_layout],
                }),
                compute: {
                    module: this.constraint_module,
                    entryPoint: "main",
                },
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, c_bindGroup);
            passEncoder.dispatchWorkgroups(1, 1, 1);
            passEncoder.end();
            // console.log("encoded compute");
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
        // Swap input and output:
        this.active_buffer_id = 1 - this.active_buffer_id;
        // await this.updateCPUpos();
        // console.log("done applyForce");
    }
    draw() {
        this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
        for (let yidx = 0; yidx < this.grid_y; yidx++) {
            for (let xidx = 0; xidx < this.grid_x; xidx++) {
                let i = yidx * this.grid_x + xidx;
                let x = this.x_pos[i];
                let y = this.y_pos[i];
                this.ctx.strokeStyle = "#ff00005f";
                this.ctx.beginPath();
                this.ctx.arc(x, y, this.radius, 0, 2 * Math.PI);
                this.ctx.stroke();
                if (this.draw_edges) {
                    for (let edge of edges) {
                        let j_xidx = xidx + edge[0];
                        let j_yidx = yidx + edge[1];
                        if (j_xidx < 0 || j_xidx >= this.grid_x || j_yidx < 0 || j_yidx >= this.grid_y) {
                            continue;
                        }
                        let j = j_yidx * this.grid_x + j_xidx;
                        let j_x = this.x_pos[j];
                        let j_y = this.y_pos[j];
                        this.ctx.beginPath();
                        this.ctx.moveTo(x, y);
                        this.ctx.lineTo(j_x, j_y);
                        this.ctx.stroke();
                    }
                }
            }
        }
    }
}
exports.MeshDeformation = MeshDeformation;


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
document.addEventListener("DOMContentLoaded", () => {
    let canvas = document.getElementById("mycanvas");
    canvas.width = 1000;
    canvas.height = 1000;
    let ctx = canvas.getContext("2d");
    console.log("Created context for main canvas");
    let canvas2 = document.getElementById("mycanvas2");
    canvas2.width = 1000;
    canvas2.height = 1000;
    let ctx2 = canvas2.getContext("2d");
    canvas2.addEventListener("click", (e) => {
        let el = e.target;
        const rect = el.getBoundingClientRect();
        const x = el.width * (e.clientX - rect.left) / rect.width;
        const y = el.height * (e.clientY - rect.top) / rect.height;
        ctx2.beginPath();
        ctx2.fillStyle = "black";
        ctx2.arc(x, y, 100, 0, 2 * Math.PI);
        ctx2.fill();
    });
    console.log("Created context for interactive canvas");
    window.n_steps_per_frame = 1;
    let md = new meshdeformation_1.MeshDeformation(ctx, 50, 50, ctx.canvas.width / 50, 10, 100, 1);
    window.t_per_render = 0;
    window.n_renders = 0;
    md.initialization_done.then(() => {
        const f = async () => {
            let start = performance.now();
            md.draw();
            for (let i = 0; i < window.n_steps_per_frame; i++) {
                await md.applyForce(ctx2);
                // await md.updateCPUpos();
            }
            await md.device.queue.onSubmittedWorkDone();
            let end = performance.now();
            window.t_per_render += end - start;
            window.n_renders += 1;
            if (!stop) {
                requestAnimationFrame(f);
                // setTimeout(() => {
                //   requestAnimationFrame(f)
                // }, 1);
            }
        };
        requestAnimationFrame(f);
        window.t_per_draw = 0;
        window.n_draws = 0;
        const g = async () => {
            let start = performance.now();
            await md.updateCPUpos();
            md.draw();
            let end = performance.now();
            window.t_per_draw += end - start;
            window.n_draws += 1;
            setTimeout(() => {
                requestAnimationFrame(g);
            }, 30);
        };
        requestAnimationFrame(g);
    });
    window.stats = () => {
        console.log("t_per_render", window.t_per_render);
        console.log("n_renders", window.n_renders);
        console.log("avg", window.t_per_render / window.n_renders);
        console.log("t_per_draw", window.t_per_draw);
        console.log("n_draws", window.n_draws);
        console.log("avg", window.t_per_draw / window.n_draws);
    };
    function cancel() {
        stop = true;
    }
    window.md = md;
    window.ctx2 = ctx2;
    window.cancel = cancel;
});

})();

/******/ })()
;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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Q0FBQyxTQUFTLENBQUMsQ0FBQztRQUMzRCxPQUFPLENBQUMsR0FBRyxDQUFDLFlBQVksRUFBRSxNQUFNLENBQUMsVUFBVSxDQUFDLENBQUM7UUFDN0MsT0FBTyxDQUFDLEdBQUcsQ0FBQyxTQUFTLEVBQUUsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQ3ZDLE9BQU8sQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLE1BQU0sQ0FBQyxVQUFVLEdBQUcsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQ3pELENBQUM7SUFHRCxTQUFTLE1BQU07UUFDYixJQUFJLEdBQUcsSUFBSSxDQUFDO0lBQ2QsQ0FBQztJQUNBLE1BQWMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDO0lBQ3ZCLE1BQWMsQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDO0lBQzNCLE1BQWMsQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDO0FBQ2xDLENBQUMsQ0FBQyxDQUFDIiwic291cmNlcyI6WyJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvY29uc3RyYWludC50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9mb3JjZXMudHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvaW5pdC50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9tZXNoZGVmb3JtYXRpb24udHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3Qvd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvbWFpbi50cyJdLCJzb3VyY2VzQ29udGVudCI6WyJjb25zdCBlZGdlcyA9IFtcbiAgWy0xLCAwXSxcbiAgWzEsIDBdLFxuICBbMCwgLTFdLFxuICBbMCwgMV0sXG4gIFsxLCAxXSxcbiAgWy0xLCAtMV0sXG4gIFsxLCAtMV0sXG4gIFstMSwgMV0sXG5dO1xuXG5cbmV4cG9ydCBmdW5jdGlvbiBidWlsZChncmlkX3g6IG51bWJlciwgZ3JpZF95OiBudW1iZXIsIG1pbl9kaXN0OiBudW1iZXIsIG1heF9kaXN0OiBudW1iZXIpIHtcbiAgbGV0IHNyYyA9IGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDEpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHlfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMilcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMylcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNClcbnZhcjx1bmlmb3JtPm9mZnNldDogdTMyO1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPmRlYnVnOiBhcnJheTx1MzI+O1xuXG5cbkBjb21wdXRlIEB3b3JrZ3JvdXBfc2l6ZSgyNTYpXG5mbiBtYWluKFxuICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZClcbmdsb2JhbF9pZCA6IHZlYzN1LFxuXG4gIEBidWlsdGluKGxvY2FsX2ludm9jYXRpb25faWQpXG5sb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIC8vIENvb3JkaW5hdGVzIG9mIHBhcnRpY2xlIGZvciB0aGlzIHRocmVhZFxuICBsZXQgaSA9IG9mZnNldCArIGdsb2JhbF9pZC54O1xuICBsZXQgZ3JpZF95ID0gaTMyKGkgLyAke2dyaWRfeH0pO1xuICBsZXQgZ3JpZF94ID0gaTMyKGkgJSAke2dyaWRfeH0pO1xuXG4gIGxldCBjdXJyX3ggPSB4X3Bvc19vdXRbb2Zmc2V0ICsgZ2xvYmFsX2lkLnhdO1xuICBsZXQgY3Vycl95ID0geV9wb3Nfb3V0W29mZnNldCArIGdsb2JhbF9pZC54XTtcbiAgbGV0IG9yZ194ID0geF9wb3Nbb2Zmc2V0ICsgZ2xvYmFsX2lkLnhdO1xuICBsZXQgb3JnX3kgPSB5X3Bvc1tvZmZzZXQgKyBnbG9iYWxfaWQueF07XG5cbiAgdmFyIGludmFsaWQgPSBmYWxzZTtcbmA7XG5cbiAgZm9yIChsZXQgZWRnZSBvZiBlZGdlcykge1xuICAgIHNyYyArPSBgXG4gIHtcbiAgICBsZXQgZV9neCA9IGdyaWRfeCArICR7ZWRnZVswXX07XG4gICAgbGV0IGVfZ3kgPSBncmlkX3kgKyAke2VkZ2VbMV19O1xuICAgIGlmIChlX2d4ID4gMCAmJiBlX2d4IDwgJHtncmlkX3h9ICYmIGVfZ3kgPiAwICYmIGVfZ3kgPCAke2dyaWRfeX0pIHtcbiAgICAgIGxldCBlX2kgPSBlX2d5ICogJHtncmlkX3h9ICsgZV9neDtcbiAgICAgIGxldCBuX3ggPSB4X3Bvc19vdXRbZV9pXTtcbiAgICAgIGxldCBuX3kgPSB5X3Bvc19vdXRbZV9pXTtcbiAgICAgIGxldCBkX3ggPSBuX3ggLSBjdXJyX3g7XG4gICAgICBsZXQgZF95ID0gbl95IC0gY3Vycl95O1xuXG4gICAgICBsZXQgZF9uMiA9IGRfeCAqIGRfeCArIGRfeSAqIGRfeTtcbiAgICAgIGlmIChkX24yIDwgKCR7bWluX2Rpc3R9ICogJHttaW5fZGlzdH0pIHx8IGRfbjIgPiAoJHttYXhfZGlzdH0gKiAke21heF9kaXN0fSkpIHtcbiAgICAgICAgaW52YWxpZCA9IHRydWU7XG4gICAgICB9XG4gICAgfVxuICB9XG5gXG4gIH1cblxuICBzcmMgKz0gYFxuICBpZiAoaW52YWxpZCkge1xuICAgIHhfcG9zX291dFtpXSA9IHhfcG9zW2ldO1xuICAgIHlfcG9zX291dFtpXSA9IHlfcG9zW2ldO1xuICB9XG5gXG5cbiAgc3JjICs9IFwifVxcblwiO1xuICByZXR1cm4gc3JjXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gYnVpbGQod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGdyaWRfeDogbnVtYmVyLCBncmlkX3NwYWNpbmc6IG51bWJlcikge1xuICByZXR1cm4gYFxuQGdyb3VwKDApIEBiaW5kaW5nKDApXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygyKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiBpbnRlbnNpdHlfbWFwOiBhcnJheTx1MzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMylcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3Nfb3V0OiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoNSlcbnZhcjx1bmlmb3JtPm9mZnNldDogdTMyO1xuXG5cblxuZm4gcGl4ZWxUb0ludGVuc2l0eShfcHg6IHUzMikgLT4gZjMyIHtcbiAgdmFyIHB4ID0gX3B4O1xuICBsZXQgciA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgZyA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgYiA9IGYzMihweCAlIDI1Nik7XG4gIHB4IC89IHUzMigyNTYpO1xuICBsZXQgYSA9IGYzMihweCAlIDI1Nik7XG4gIGxldCBpbnRlbnNpdHk6IGYzMiA9IChhIC8gMjU1KSAqICgxIC0gKDAuMjEyNiAqIHIgKyAwLjcxNTIgKiBnICsgMC4wNzIyICogYikpO1xuXG4gIHJldHVybiBpbnRlbnNpdHk7XG59XG5cbkBjb21wdXRlIEB3b3JrZ3JvdXBfc2l6ZSg2NCwgMSwgMSlcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuZ2xvYmFsX2lkIDogdmVjM3UsXG5cbiAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZClcbmxvY2FsX2lkIDogdmVjM3UsXG4pIHtcbiAgLy8gQ29vcmRpbmF0ZXMgb2YgcGFydGljbGUgZm9yIHRoaXMgdGhyZWFkXG4gIGxldCBpID0gb2Zmc2V0ICsgZ2xvYmFsX2lkLng7XG5cbiAgbGV0IGdyaWRfeCA9IGkgJSAke2dyaWRfeH07XG4gIGxldCBncmlkX3kgPSBpIC8gJHtncmlkX3h9O1xuXG4gIGxldCB4ID0geF9wb3NbaV07XG4gIGxldCB5ID0geV9wb3NbaV07XG5cbiAgdmFyIGRpcl94OiBmMzIgPSAwO1xuICB2YXIgZGlyX3k6IGYzMiA9IDA7XG4gIHZhciBjb2VmZjogZjMyID0gMDtcblxuICAvLyBDb29yZGluYXRlcyB0byBsb29rdXAgaW4gaW50ZW5zaXR5X21hcFxuICBmb3IgKHZhciBzX3k6IGkzMiA9IDA7IHNfeSA8IDUwOyBzX3krKykge1xuICAgIGZvciAodmFyIHNfejogaTMyID0gMDsgc196IDwgNTA7IHNfeisrKSB7XG4gICAgICBsZXQgZl95ID0gaTMyKGZsb29yKHkpKSArIGkzMig1MCAqIGxvY2FsX2lkLnkpICsgc195IC0gMjU7XG4gICAgICBsZXQgZl94ID0gaTMyKGZsb29yKHgpKSArIGkzMig1MCAqIGxvY2FsX2lkLnopICsgc196IC0gMjU7XG4gICAgICBsZXQgZF94OiBmMzIgPSBmMzIoZl94KSAtIHg7XG4gICAgICBsZXQgZF95OiBmMzIgPSBmMzIoZl95KSAtIHk7XG4gICAgICBsZXQgcjI6IGYzMiA9IGRfeCAqIGRfeCArIGRfeSAqIGRfeTtcbiAgICAgIGxldCByOiBmMzIgPSBzcXJ0KHIyKTtcblxuICAgICAgLy8gRmluZCB0aGUgZm9yY2UgZXhlcnRlZCBvbiB0aGUgcGFydGljbGUgYnkgY29udGVudHMgb2YgdGhlIGludGVzaXR5IG1hcC5cbiAgICAgIGlmIChmX3kgPj0gMCAmJiBmX3kgPCAke2hlaWdodH0gJiYgZl94ID49IDAgJiYgZl94IDwgJHt3aWR0aH0pIHtcbiAgICAgICAgbGV0IGZfaSA9IGZfeSAqICR7d2lkdGh9ICsgZl94O1xuICAgICAgICBsZXQgaW50ZW5zaXR5ID0gcGl4ZWxUb0ludGVuc2l0eShpbnRlbnNpdHlfbWFwW2ZfaV0pO1xuXG4gICAgICAgIGlmIChyICE9IDApIHtcbiAgICAgICAgICBsZXQgbG9jYWxfY29lZmYgPSAxMDAgKiBpbnRlbnNpdHkgLyByMjtcbiAgICAgICAgICAvLyBhdG9taWNBZGQoJiBjb2VmZiwgdTMyKDEwMDAwICogbG9jYWxfY29lZmYpKTtcbiAgICAgICAgICAvLyBhdG9taWNBZGQoJiBkaXJfeCwgaTMyKDEwMDAgKiBsb2NhbF9jb2VmZiAqIGRfeCAvIHIpKTtcbiAgICAgICAgICAvLyBhdG9taWNBZGQoJiBkaXJfeSwgaTMyKDEwMDAgKiBsb2NhbF9jb2VmZiAqIGRfeSAvIHIpKTtcbiAgICAgICAgICBjb2VmZiArPSBsb2NhbF9jb2VmZjtcbiAgICAgICAgICBkaXJfeCArPSBsb2NhbF9jb2VmZiAqIGRfeCAvIHI7XG4gICAgICAgICAgZGlyX3kgKz0gbG9jYWxfY29lZmYgKiBkX3kgLyByO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgaWYgKGxvY2FsX2lkLnkgPT0gMCAmJiBsb2NhbF9pZC56ID09IDApIHtcbiAgICBsZXQgb3JpZ2luX3ggPSBncmlkX3ggKiAke2dyaWRfc3BhY2luZ307XG4gICAgbGV0IG9yaWdpbl95ID0gZ3JpZF95ICogJHtncmlkX3NwYWNpbmd9O1xuICAgIGxldCBkX3g6IGYzMiA9IGYzMihvcmlnaW5feCkgLSB4O1xuICAgIGxldCBkX3k6IGYzMiA9IGYzMihvcmlnaW5feSkgLSB5O1xuICAgIGxldCByMjogZjMyID0gZF94ICogZF94ICsgZF95ICogZF95O1xuICAgIGxldCByOiBmMzIgPSBzcXJ0KHIyKTtcblxuICAgIGxldCBsb2NhbF9jb2VmZiA9IGYzMig1KSAvIHIyO1xuICAgIC8vIGF0b21pY0FkZCgmIGNvZWZmLCB1MzIoMTAwMDAgKiBsb2NhbF9jb2VmZikpO1xuICAgIC8vIGF0b21pY0FkZCgmIGRpcl94LCBpMzIoMTAwMCAqIGxvY2FsX2NvZWZmICogZF94IC8gcikpO1xuICAgIC8vIGF0b21pY0FkZCgmIGRpcl95LCBpMzIoMTAwMCAqIGxvY2FsX2NvZWZmICogZF95IC8gcikpO1xuICAgIGNvZWZmICs9IGxvY2FsX2NvZWZmO1xuICAgIGRpcl94ICs9IGxvY2FsX2NvZWZmICogZF94IC8gcjtcbiAgICBkaXJfeSArPSBsb2NhbF9jb2VmZiAqIGRfeSAvIHI7XG4gIH1cblxuICAvLyBXYWl0IGZvciBhbGwgd29ya2dyb3VwIHRocmVhZHMgdG8gZmluaXNoIHNpbXVsYXRpbmdcbiAgLy8gd29ya2dyb3VwQmFycmllcigpO1xuXG4gIC8vIE9uIGEgc2luZ2xlIHRocmVhZCwgdXBkYXRlIHRoZSBvdXRwdXQgcG9zaXRpb24gZm9yIHRoZSBjdXJyZW50IHBhcnRpY2xlXG4gIGlmIChsb2NhbF9pZC55ID09IDAgJiYgbG9jYWxfaWQueiA9PSAwKSB7XG4gICAgLy8gbGV0IHRvdGFsX2NvZWZmID0gZjMyKGF0b21pY0xvYWQoJiBjb2VmZikpIC8gMTAwMDA7XG4gICAgbGV0IHRvdGFsX2NvZWZmID0gY29lZmY7XG4gICAgaWYgKHRvdGFsX2NvZWZmICE9IDApIHtcbiAgICAgIC8vIHZhciBkX3ggPSBmMzIoYXRvbWljTG9hZCgmIGRpcl94KSkgLyAoMTAwMCAqIHRvdGFsX2NvZWZmKTtcbiAgICAgIC8vIHZhciBkX3kgPSBmMzIoYXRvbWljTG9hZCgmIGRpcl95KSkgLyAoMTAwMCAqIHRvdGFsX2NvZWZmKTtcbiAgICAgIHZhciBkX3ggPSBkaXJfeCAvIHRvdGFsX2NvZWZmO1xuICAgICAgdmFyIGRfeSA9IGRpcl95IC8gdG90YWxfY29lZmY7XG5cbiAgICAgIGxldCBzX2R4ID0gc2lnbihkX3gpO1xuICAgICAgbGV0IHNfZHkgPSBzaWduKGRfeSk7XG4gICAgICBsZXQgYV9keCA9IGFicyhkX3gpO1xuICAgICAgbGV0IGFfZHkgPSBhYnMoZF95KTtcblxuICAgICAgZF94ID0gc19keCAqIG1pbihhX2R4LCBmMzIoMC41KSk7XG4gICAgICBkX3kgPSBzX2R5ICogbWluKGFfZHksIGYzMigwLjUpKTtcblxuICAgICAgeF9wb3Nfb3V0W2ldID0geCArIGRfeDtcbiAgICAgIHlfcG9zX291dFtpXSA9IHkgKyBkX3k7XG4gICAgfSBlbHNlIHtcbiAgICAgIHhfcG9zX291dFtpXSA9IHg7XG4gICAgICB5X3Bvc19vdXRbaV0gPSB5O1xuICAgIH1cbiAgfVxufVxuYFxufVxuIiwiZXhwb3J0IGZ1bmN0aW9uIGJ1aWxkKG5fZWxlbXM6IG51bWJlciwgZ3JpZF94OiBudW1iZXIsIGdyaWRfc3BhY2luZzogbnVtYmVyKSB7XG4gIHJldHVybiBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB4X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDEpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZT4geV9wb3M6IGFycmF5PGYzMj47XG5cbkBjb21wdXRlIEB3b3JrZ3JvdXBfc2l6ZSgyNTYpXG5mbiBtYWluKFxuICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZClcbiAgZ2xvYmFsX2lkIDogdmVjM3UsXG5cbiAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZClcbiAgbG9jYWxfaWQgOiB2ZWMzdSxcbikge1xuICBpZiAoZ2xvYmFsX2lkLnggPCAke25fZWxlbXN9KSB7XG4gICAgICBsZXQgeSA9IGdsb2JhbF9pZC54IC8gJHtncmlkX3h9O1xuICAgICAgbGV0IHggPSBnbG9iYWxfaWQueCAlICR7Z3JpZF94fTtcblxuICAgICAgeF9wb3NbZ2xvYmFsX2lkLnhdID0gZjMyKHggKiAke2dyaWRfc3BhY2luZ30pO1xuICAgICAgeV9wb3NbZ2xvYmFsX2lkLnhdID0gZjMyKHkgKiAke2dyaWRfc3BhY2luZ30pO1xuICB9XG59XG5gXG59XG4iLCJpbXBvcnQgeyBidWlsZCBhcyBpbml0YnVpbGQgfSBmcm9tICcuL2luaXQnO1xuaW1wb3J0IHsgYnVpbGQgYXMgZm9yY2VzYnVpbGQgfSBmcm9tICcuL2ZvcmNlcyc7XG5pbXBvcnQgeyBidWlsZCBhcyBjb25zdHJhaW50YnVpbGQgfSBmcm9tICcuL2NvbnN0cmFpbnQnO1xuXG5jb25zdCBlZGdlcyA9IFtcbiAgWy0xLCAwXSxcbiAgWzEsIDBdLFxuICBbMCwgLTFdLFxuICBbMCwgMV0sXG4gIFsxLCAxXSxcbiAgWy0xLCAtMV0sXG5dO1xuXG5leHBvcnQgY2xhc3MgTWVzaERlZm9ybWF0aW9uIHtcbiAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG4gIGdyaWRfeDogbnVtYmVyO1xuICBncmlkX3k6IG51bWJlcjtcbiAgZ3JpZF9zcGFjaW5nOiBudW1iZXI7XG4gIG1pbl9kaXN0OiBudW1iZXI7XG4gIG1heF9kaXN0OiBudW1iZXI7XG4gIHJhZGl1czogbnVtYmVyO1xuXG4gIGRyYXdfZWRnZXM6IGJvb2w7XG5cbiAgbl9lbGVtczogbnVtYmVyO1xuICB4X3BvczogRmxvYXQzMkFycmF5O1xuICB5X3BvczogRmxvYXQzMkFycmF5O1xuXG4gIGluaXRpYWxpemF0aW9uX2RvbmU6IFByb21pc2U8dm9pZD47XG4gIGRldmljZTogR1BVRGV2aWNlO1xuICBmb3JjZV9tb2R1bGU6IEdQVVNoYWRlck1vZHVsZTtcbiAgYWN0aXZlX2J1ZmZlcl9pZDogbnVtYmVyO1xuICB4X3Bvc19idWZmZXJzOiBbR1BVQnVmZmVyLCBHUFVCdWZmZXJdO1xuICB5X3Bvc19idWZmZXJzOiBbR1BVQnVmZmVyLCBHUFVCdWZmZXJdO1xuICAvLyBCdWZmZXJzIHRvIHJlYWQgdmFsdWVzIGJhY2sgdG8gdGhlIENQVSBmb3IgZHJhd2luZ1xuICBzdGFnaW5nX3hfYnVmOiBHUFVCdWZmZXI7XG4gIHN0YWdpbmdfeV9idWY6IEdQVUJ1ZmZlcjtcbiAgLy8gQnVmZmVyIHRvIHdyaXRlIHZhbHVlIHRvIGZyb20gdGhlIENQVSBmb3IgYWRqdXN0aW5nIHdlaWdodHNcbiAgc3RhZ2luZ19pbnRlbnNpdHlfYnVmOiBHUFVCdWZmZXI7XG4gIGludGVuc2l0eV9tYXBfYnVmOiBHUFVCdWZmZXI7XG5cbiAgb2Zmc2V0X2J1ZjogR1BVQnVmZmVyO1xuXG4gIGZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0OiBHUFVCaW5kR3JvdXBMYXlvdXQ7XG5cbiAgY29uc3RyYWludF9tb2R1bGU6IEdQVVNoYWRlck1vZHVsZTtcbiAgY29uc3RyYWludF9iaW5kX2dyb3VwX2xheW91dDogR1BVQmluZEdyb3VwTGF5b3V0O1xuXG4gIGRlYnVnX2J1ZjogR1BVQnVmZmVyO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgIGN0eDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJELFxuICAgIGdyaWRfeDogbnVtYmVyLFxuICAgIGdyaWRfeTogbnVtYmVyLFxuICAgIGdyaWRfc3BhY2luZzogbnVtYmVyLFxuICAgIG1pbl9kaXN0OiBudW1iZXIsXG4gICAgbWF4X2Rpc3Q6IG51bWJlcixcbiAgICByYWRpdXM6IG51bWJlclxuICApIHtcbiAgICB0aGlzLmN0eCA9IGN0eDtcbiAgICB0aGlzLmdyaWRfeCA9IGdyaWRfeDtcbiAgICB0aGlzLmdyaWRfeSA9IGdyaWRfeTtcbiAgICB0aGlzLmdyaWRfc3BhY2luZyA9IGdyaWRfc3BhY2luZztcbiAgICB0aGlzLm1pbl9kaXN0ID0gbWluX2Rpc3Q7XG4gICAgdGhpcy5tYXhfZGlzdCA9IG1heF9kaXN0O1xuICAgIHRoaXMucmFkaXVzID0gcmFkaXVzO1xuXG4gICAgdGhpcy5kcmF3X2VkZ2VzID0gdHJ1ZTtcblxuICAgIHRoaXMubl9lbGVtcyA9IHRoaXMuZ3JpZF94ICogdGhpcy5ncmlkX3k7XG5cbiAgICB0aGlzLmluaXRpYWxpemF0aW9uX2RvbmUgPSB0aGlzLmFzeW5jX2luaXQoKTtcbiAgfVxuXG4gIGFzeW5jIGFzeW5jX2luaXQoKSB7XG4gICAgY29uc3QgYWRhcHRlciA9IGF3YWl0IG5hdmlnYXRvci5ncHUucmVxdWVzdEFkYXB0ZXIoKTtcbiAgICB0aGlzLmRldmljZSA9IGF3YWl0IGFkYXB0ZXIucmVxdWVzdERldmljZSgpO1xuICAgIGNvbnNvbGUubG9nKFwiQ3JlYXRlIGNvbXB1dGUgc2hhZGVyXCIpO1xuICAgIHRoaXMuZm9yY2VfbW9kdWxlID0gdGhpcy5kZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcbiAgICAgIGNvZGU6IGZvcmNlc2J1aWxkKHRoaXMuY3R4LmNhbnZhcy53aWR0aCwgdGhpcy5jdHguY2FudmFzLmhlaWdodCwgdGhpcy5ncmlkX3gsIHRoaXMuZ3JpZF9zcGFjaW5nKSxcbiAgICB9KTtcbiAgICBjb25zb2xlLmxvZyhcImRvbmUgQ3JlYXRlIGNvbXB1dGUgc2hhZGVyXCIpO1xuXG4gICAgdGhpcy5kZWJ1Z19idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwiZGVidWdcIixcbiAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgfSk7XG5cbiAgICB0aGlzLmFjdGl2ZV9idWZmZXJfaWQgPSAwO1xuICAgIHRoaXMueF9wb3NfYnVmZmVycyA9IFtcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInhfcG9zWzBdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieF9wb3NbMV1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgXTtcbiAgICB0aGlzLnlfcG9zX2J1ZmZlcnMgPSBbXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ5X3Bvc1swXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInlfcG9zWzFdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgIF07XG5cbiAgICB0aGlzLnN0YWdpbmdfeF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwic3RhZ2luZ194X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5NQVBfUkVBRCB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIHRoaXMuc3RhZ2luZ195X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX3lfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG5cbiAgICB0aGlzLnN0YWdpbmdfaW50ZW5zaXR5X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX2ludGVuc2l0eV9idWZcIixcbiAgICAgIHNpemU6IHRoaXMuY3R4LmNhbnZhcy53aWR0aCAqIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9XUklURSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDLFxuICAgIH0pO1xuICAgIHRoaXMuaW50ZW5zaXR5X21hcF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwiaW50ZW5zaXR5X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuXG4gICAgdGhpcy5vZmZzZXRfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcIm9mZnNldF9idWZcIixcbiAgICAgIHNpemU6IDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuVU5JRk9STSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBhbGxvY2F0ZSBidWZmZXJzXCIpO1xuXG4gICAgdGhpcy5mb3JjZV9iaW5kX2dyb3VwX2xheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDIsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAzLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNCxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDUsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwidW5pZm9ybVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgLy8gaW50aWFsaXplIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdIGFuZFxuICAgIC8vIHRoaXMueV9wb3NfYnVmZmVyc1sqXSB0byBiZSBhIGdyaWRcbiAgICBjb25zdCBpbml0X3NoYWRlciA9IGluaXRidWlsZCh0aGlzLm5fZWxlbXMsIHRoaXMuZ3JpZF94LCB0aGlzLmdyaWRfc3BhY2luZyk7XG4gICAgY29uc3QgaW5pdEJpbmRHcm91cExheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zb2xlLmxvZyhcIkNyZWF0ZSBpbml0IHNoYWRlclwiKTtcbiAgICBjb25zdCBpbml0X21vZHVsZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZVNoYWRlck1vZHVsZSh7XG4gICAgICBjb2RlOiBpbml0X3NoYWRlcixcbiAgICB9KTtcbiAgICBjb25zb2xlLmxvZyhcImRvbmUgQ3JlYXRlIGluaXQgc2hhZGVyXCIpO1xuICAgIGNvbnN0IGNvbXB1dGVQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbXB1dGVQaXBlbGluZSh7XG4gICAgICBsYWJlbDogXCJjb21wdXRlIGZvcmNlXCIsXG4gICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW2luaXRCaW5kR3JvdXBMYXlvdXRdLFxuICAgICAgfSksXG4gICAgICBjb21wdXRlOiB7XG4gICAgICAgIG1vZHVsZTogaW5pdF9tb2R1bGUsXG4gICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgfSxcbiAgICB9KTtcbiAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG5cbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiBpbml0QmluZEdyb3VwTGF5b3V0LFxuICAgICAgZW50cmllczogW1xuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMCxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9XG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUoY29tcHV0ZVBpcGVsaW5lKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoTWF0aC5jZWlsKHRoaXMubl9lbGVtcyAvIDI1NikpO1xuICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemVcbiAgICApO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemVcbiAgICApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcblxuICAgIGF3YWl0IHRoaXMudXBkYXRlQ1BVcG9zKCk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFzeW5jIGluaXRcIik7XG5cbiAgICBjb25zdCBjb25zdHJhaW50X3NyYyA9IGNvbnN0cmFpbnRidWlsZCh0aGlzLmdyaWRfeCwgdGhpcy5ncmlkX3ksIHRoaXMubWluX2Rpc3QsIHRoaXMubWF4X2Rpc3QpO1xuICAgIC8vIGNvbnNvbGUubG9nKGNvbnN0cmFpbnRfc3JjKTtcbiAgICB0aGlzLmNvbnN0cmFpbnRfbW9kdWxlID0gdGhpcy5kZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcbiAgICAgIGNvZGU6IGNvbnN0cmFpbnRfc3JjLFxuICAgIH0pO1xuICAgIHRoaXMuY29uc3RyYWludF9iaW5kX2dyb3VwX2xheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBsYWJlbDogXCJjb25zdHJhaW50X2JpbmRfZ3JvdXBfbGF5b3V0XCIsXG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDIsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAzLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNCxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJ1bmlmb3JtXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDUsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuICB9XG5cbiAgYXN5bmMgdXBkYXRlQ1BVcG9zKCkge1xuICAgIC8vIGNvbnNvbGUubG9nKFwiTWFwIGJ1ZmZlcnMgZm9yIHJlYWRpbmdcIik7XG4gICAgbGV0IG1feCA9IHRoaXMuc3RhZ2luZ194X2J1Zi5tYXBBc3luYyhHUFVNYXBNb2RlLlJFQUQsIDAsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplKTtcbiAgICBsZXQgbV95ID0gdGhpcy5zdGFnaW5nX3lfYnVmLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemUpO1xuICAgIGF3YWl0IG1feDtcbiAgICBhd2FpdCBtX3k7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcImNvcHlpbmcgeCBidWZmZXIgdG8gQ1BVXCIpO1xuICAgIGNvbnN0IGNvcHlBcnJheUJ1ZmZlclggPSB0aGlzLnN0YWdpbmdfeF9idWYuZ2V0TWFwcGVkUmFuZ2UoMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIGNvbnN0IGRhdGFYID0gY29weUFycmF5QnVmZmVyWC5zbGljZSgpO1xuICAgIHRoaXMueF9wb3MgPSBuZXcgRmxvYXQzMkFycmF5KGRhdGFYKTtcblxuICAgIC8vIGNvbnNvbGUubG9nKFwiY29weWluZyB5IGJ1ZmZlciB0byBDUFVcIik7XG4gICAgY29uc3QgY29weUFycmF5QnVmZmVyWSA9IHRoaXMuc3RhZ2luZ195X2J1Zi5nZXRNYXBwZWRSYW5nZSgwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgY29uc3QgZGF0YVkgPSBjb3B5QXJyYXlCdWZmZXJZLnNsaWNlKCk7XG4gICAgdGhpcy55X3BvcyA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YVkpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJ1bm1hcCBidWZmZXJzXCIpO1xuICAgIHRoaXMuc3RhZ2luZ194X2J1Zi51bm1hcCgpO1xuICAgIHRoaXMuc3RhZ2luZ195X2J1Zi51bm1hcCgpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJEb25lIHVwZGF0ZUNQVXBvc1wiKTtcbiAgfVxuXG4gIGFzeW5jIGFwcGx5Rm9yY2UoY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQpIHtcbiAgICBsZXQgaWRhdGEgPSBjdHguZ2V0SW1hZ2VEYXRhKDAsIDAsIGN0eC5jYW52YXMud2lkdGgsIGN0eC5jYW52YXMuaGVpZ2h0KS5kYXRhO1xuICAgIC8vIGNvbnNvbGUubG9nKGBiMCAke2lkYXRhWzBdfSwgJHtpZGF0YVsxXX0sICR7aWRhdGFbMl19LCAke2lkYXRhWzNdfWApO1xuICAgIC8vIGNvbnNvbGUubG9nKGBXcml0aW5nICR7dGhpcy5pbnRlbnNpdHlfbWFwX2J1Zi5zaXplfS8ke2lkYXRhLmxlbmd0aH0gYnl0ZXMgZm9yIGltYXBgKTtcbiAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgIHRoaXMuaW50ZW5zaXR5X21hcF9idWYsIDAsIGlkYXRhLmJ1ZmZlciwgMCwgdGhpcy5pbnRlbnNpdHlfbWFwX2J1Zi5zaXplKTtcblxuICAgIGxldCBpbnB1dF94ID0gdGhpcy54X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG4gICAgbGV0IGlucHV0X3kgPSB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgb3V0cHV0X3ggPSB0aGlzLnhfcG9zX2J1ZmZlcnNbMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG4gICAgbGV0IG91dHB1dF95ID0gdGhpcy55X3Bvc19idWZmZXJzWzEgLSB0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuXG4gICAgY29uc3Qgd2dfeCA9IDY0O1xuICAgIGNvbnN0IGRpc3BhdGNoX3ggPSAyNTYgLyB3Z194O1xuICAgIGxldCBidWZmZXJzID0gW2lucHV0X3gsIGlucHV0X3ksIHRoaXMuaW50ZW5zaXR5X21hcF9idWYsIG91dHB1dF94LCBvdXRwdXRfeSwgdGhpcy5vZmZzZXRfYnVmXTtcbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiB0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0LFxuICAgICAgZW50cmllczogYnVmZmVycy5tYXAoKGIsIGkpID0+IHsgcmV0dXJuIHsgYmluZGluZzogaSwgcmVzb3VyY2U6IHsgYnVmZmVyOiBiIH0gfTsgfSlcbiAgICB9KTtcblxuICAgIGZvciAobGV0IG9mZnNldCA9IDA7IG9mZnNldCA8IHRoaXMubl9lbGVtczsgb2Zmc2V0ICs9ICh3Z194ICogZGlzcGF0Y2hfeCkpIHtcbiAgICAgIGxldCBpbnB1dCA9IG5ldyBVaW50MzJBcnJheShbb2Zmc2V0XSk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5vZmZzZXRfYnVmLCAwLCBpbnB1dC5idWZmZXIsIDAsIDQpO1xuXG4gICAgICBjb25zdCBjb21wdXRlUGlwZWxpbmUgPSB0aGlzLmRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgICBsYWJlbDogXCJmb3JjZXBpcGVsaW5lXCIsXG4gICAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICAgIGJpbmRHcm91cExheW91dHM6IFt0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0XSxcbiAgICAgICAgfSksXG4gICAgICAgIGNvbXB1dGU6IHtcbiAgICAgICAgICBtb2R1bGU6IHRoaXMuZm9yY2VfbW9kdWxlLFxuICAgICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgICB9LFxuICAgICAgfSk7XG4gICAgICAvLyBjb25zb2xlLmxvZyhcImNyZWF0ZWQgcGlwZWxpbmVcIik7XG5cbiAgICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAgIGNvbnN0IHBhc3NFbmNvZGVyID0gY29tbWFuZEVuY29kZXIuYmVnaW5Db21wdXRlUGFzcygpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUoY29tcHV0ZVBpcGVsaW5lKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKGRpc3BhdGNoX3gsIDEsIDEpO1xuICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XG4gICAgICAvLyBjb25zb2xlLmxvZyhcImVuY29kZWQgY29tcHV0ZVwiKTtcbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICB9XG5cbiAgICBsZXQgY19idWZmZXJzID0gW2lucHV0X3gsIGlucHV0X3ksIG91dHB1dF94LCBvdXRwdXRfeSwgdGhpcy5vZmZzZXRfYnVmLCB0aGlzLmRlYnVnX2J1Zl07XG4gICAgY29uc3QgY19iaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGFiZWw6IFwiY29uc3RyYWludF9iaW5kX2dyb3VwXCIsXG4gICAgICBsYXlvdXQ6IHRoaXMuY29uc3RyYWludF9iaW5kX2dyb3VwX2xheW91dCxcbiAgICAgIGVudHJpZXM6IGNfYnVmZmVycy5tYXAoKGIsIGkpID0+IHsgcmV0dXJuIHsgYmluZGluZzogaSwgcmVzb3VyY2U6IHsgYnVmZmVyOiBiIH0gfTsgfSlcbiAgICB9KTtcbiAgICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCB0aGlzLm5fZWxlbXM7IG9mZnNldCArPSAyNTYpIHtcbiAgICAgIGxldCBpbnB1dCA9IG5ldyBVaW50MzJBcnJheShbb2Zmc2V0XSk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5vZmZzZXRfYnVmLCAwLCBpbnB1dC5idWZmZXIsIDAsIDQpO1xuXG4gICAgICBjb25zdCBjb21wdXRlUGlwZWxpbmUgPSB0aGlzLmRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgICBsYWJlbDogXCJjb25zdHJhaW50cGlwZWxpbmVcIixcbiAgICAgICAgbGF5b3V0OiB0aGlzLmRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XG4gICAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3RoaXMuY29uc3RyYWludF9iaW5kX2dyb3VwX2xheW91dF0sXG4gICAgICAgIH0pLFxuICAgICAgICBjb21wdXRlOiB7XG4gICAgICAgICAgbW9kdWxlOiB0aGlzLmNvbnN0cmFpbnRfbW9kdWxlLFxuICAgICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgICB9LFxuICAgICAgfSk7XG4gICAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG4gICAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luQ29tcHV0ZVBhc3MoKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldFBpcGVsaW5lKGNvbXB1dGVQaXBlbGluZSk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgY19iaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKDEsIDEsIDEpO1xuICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XG4gICAgICAvLyBjb25zb2xlLmxvZyhcImVuY29kZWQgY29tcHV0ZVwiKTtcbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICB9XG5cbiAgICBjb25zdCBjb3B5X291dHB1dF9jb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG4gICAgLy8gQ29weSBvdXRwdXQgYnVmZmVyIHRvIHN0YWdpbmcgYnVmZmVyXG4gICAgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgb3V0cHV0X3gsIDAsIHRoaXMuc3RhZ2luZ194X2J1ZiwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwieCBjb3B5aW5nXCIsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplLCBcImJ5dGVzXCIpO1xuICAgIGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIG91dHB1dF95LCAwLCB0aGlzLnN0YWdpbmdfeV9idWYsIDAsIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcInkgY29weWluZ1wiLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSwgXCJieXRlc1wiKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImVuY29kZWQgY29weSB0byBidWZmZXJzXCIsIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZCk7XG5cbiAgICAvLyBFbmQgZnJhbWUgYnkgcGFzc2luZyBhcnJheSBvZiBjb21tYW5kIGJ1ZmZlcnMgdG8gY29tbWFuZCBxdWV1ZSBmb3IgZXhlY3V0aW9uXG4gICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgIC8vIGNvbnNvbGUubG9nKFwiZG9uZSBzdWJtaXQgdG8gcXVldWVcIik7XG5cbiAgICAvLyBTd2FwIGlucHV0IGFuZCBvdXRwdXQ6XG4gICAgdGhpcy5hY3RpdmVfYnVmZmVyX2lkID0gMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZDtcblxuICAgIC8vIGF3YWl0IHRoaXMudXBkYXRlQ1BVcG9zKCk7XG4gICAgLy8gY29uc29sZS5sb2coXCJkb25lIGFwcGx5Rm9yY2VcIik7XG4gIH1cblxuICBkcmF3KCkge1xuICAgIHRoaXMuY3R4LmNsZWFyUmVjdCgwLCAwLCB0aGlzLmN0eC5jYW52YXMud2lkdGgsIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQpO1xuICAgIGZvciAobGV0IHlpZHggPSAwOyB5aWR4IDwgdGhpcy5ncmlkX3k7IHlpZHgrKykge1xuICAgICAgZm9yIChsZXQgeGlkeCA9IDA7IHhpZHggPCB0aGlzLmdyaWRfeDsgeGlkeCsrKSB7XG4gICAgICAgIGxldCBpID0geWlkeCAqIHRoaXMuZ3JpZF94ICsgeGlkeDtcblxuICAgICAgICBsZXQgeCA9IHRoaXMueF9wb3NbaV07XG4gICAgICAgIGxldCB5ID0gdGhpcy55X3Bvc1tpXTtcblxuICAgICAgICB0aGlzLmN0eC5zdHJva2VTdHlsZSA9IFwiI2ZmMDAwMDVmXCI7XG4gICAgICAgIHRoaXMuY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICB0aGlzLmN0eC5hcmMoeCwgeSwgdGhpcy5yYWRpdXMsIDAsIDIgKiBNYXRoLlBJKTtcbiAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG5cbiAgICAgICAgaWYgKHRoaXMuZHJhd19lZGdlcykge1xuICAgICAgICAgIGZvciAobGV0IGVkZ2Ugb2YgZWRnZXMpIHtcbiAgICAgICAgICAgIGxldCBqX3hpZHggPSB4aWR4ICsgZWRnZVswXTtcbiAgICAgICAgICAgIGxldCBqX3lpZHggPSB5aWR4ICsgZWRnZVsxXTtcbiAgICAgICAgICAgIGlmIChqX3hpZHggPCAwIHx8IGpfeGlkeCA+PSB0aGlzLmdyaWRfeCB8fCBqX3lpZHggPCAwIHx8IGpfeWlkeCA+PSB0aGlzLmdyaWRfeSkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgbGV0IGogPSBqX3lpZHggKiB0aGlzLmdyaWRfeCArIGpfeGlkeDtcblxuICAgICAgICAgICAgbGV0IGpfeCA9IHRoaXMueF9wb3Nbal07XG4gICAgICAgICAgICBsZXQgal95ID0gdGhpcy55X3Bvc1tqXTtcblxuICAgICAgICAgICAgdGhpcy5jdHguYmVnaW5QYXRoKCk7XG4gICAgICAgICAgICB0aGlzLmN0eC5tb3ZlVG8oeCwgeSk7XG4gICAgICAgICAgICB0aGlzLmN0eC5saW5lVG8oal94LCBqX3kpO1xuICAgICAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG4iLCIvLyBUaGUgbW9kdWxlIGNhY2hlXG52YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG5cbi8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG5mdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuXHR2YXIgY2FjaGVkTW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXTtcblx0aWYgKGNhY2hlZE1vZHVsZSAhPT0gdW5kZWZpbmVkKSB7XG5cdFx0cmV0dXJuIGNhY2hlZE1vZHVsZS5leHBvcnRzO1xuXHR9XG5cdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG5cdHZhciBtb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdID0ge1xuXHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcblx0XHQvLyBubyBtb2R1bGUubG9hZGVkIG5lZWRlZFxuXHRcdGV4cG9ydHM6IHt9XG5cdH07XG5cblx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG5cdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG5cdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG5cdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbn1cblxuIiwiaW1wb3J0IHsgTWVzaERlZm9ybWF0aW9uIH0gZnJvbSAnLi9tZXNoZGVmb3JtYXRpb24nO1xuXG5sZXQgc3RvcCA9IGZhbHNlO1xuXG5kb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKFwiRE9NQ29udGVudExvYWRlZFwiLCAoKSA9PiB7XG4gIGxldCBjYW52YXMgPSAoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJteWNhbnZhc1wiKSBhcyBIVE1MQ2FudmFzRWxlbWVudCk7XG4gIGNhbnZhcy53aWR0aCA9IDEwMDA7XG4gIGNhbnZhcy5oZWlnaHQgPSAxMDAwO1xuXG4gIGxldCBjdHggPSBjYW52YXMuZ2V0Q29udGV4dChcIjJkXCIpO1xuICBjb25zb2xlLmxvZyhcIkNyZWF0ZWQgY29udGV4dCBmb3IgbWFpbiBjYW52YXNcIik7XG5cbiAgbGV0IGNhbnZhczIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcIm15Y2FudmFzMlwiKSBhcyBIVE1MQ2FudmFzRWxlbWVudDtcbiAgY2FudmFzMi53aWR0aCA9IDEwMDA7XG4gIGNhbnZhczIuaGVpZ2h0ID0gMTAwMDtcbiAgbGV0IGN0eDIgPSBjYW52YXMyLmdldENvbnRleHQoXCIyZFwiKTtcbiAgY2FudmFzMi5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKGUpID0+IHtcbiAgICBsZXQgZWwgPSBlLnRhcmdldCBhcyBIVE1MQ2FudmFzRWxlbWVudDtcbiAgICBjb25zdCByZWN0ID0gZWwuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7XG4gICAgY29uc3QgeCA9IGVsLndpZHRoICogKGUuY2xpZW50WCAtIHJlY3QubGVmdCkgLyByZWN0LndpZHRoO1xuICAgIGNvbnN0IHkgPSBlbC5oZWlnaHQgKiAoZS5jbGllbnRZIC0gcmVjdC50b3ApIC8gcmVjdC5oZWlnaHQ7XG5cbiAgICBjdHgyLmJlZ2luUGF0aCgpO1xuICAgIGN0eDIuZmlsbFN0eWxlID0gXCJibGFja1wiO1xuICAgIGN0eDIuYXJjKHgsIHksIDEwMCwgMCwgMiAqIE1hdGguUEkpO1xuICAgIGN0eDIuZmlsbCgpO1xuICB9KTtcblxuICBjb25zb2xlLmxvZyhcIkNyZWF0ZWQgY29udGV4dCBmb3IgaW50ZXJhY3RpdmUgY2FudmFzXCIpO1xuXG4gICh3aW5kb3cgYXMgYW55KS5uX3N0ZXBzX3Blcl9mcmFtZSA9IDE7XG5cbiAgbGV0IG1kID0gbmV3IE1lc2hEZWZvcm1hdGlvbihjdHgsIDUwLCA1MCwgY3R4LmNhbnZhcy53aWR0aCAvIDUwLCAxMCwgMTAwLCAxKTtcbiAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX3JlbmRlciA9IDA7XG4gICh3aW5kb3cgYXMgYW55KS5uX3JlbmRlcnMgPSAwO1xuICBtZC5pbml0aWFsaXphdGlvbl9kb25lLnRoZW4oKCkgPT4ge1xuICAgIGNvbnN0IGYgPSBhc3luYyAoKSA9PiB7XG4gICAgICBsZXQgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgIG1kLmRyYXcoKTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgKHdpbmRvdyBhcyBhbnkpLm5fc3RlcHNfcGVyX2ZyYW1lOyBpKyspIHtcbiAgICAgICAgYXdhaXQgbWQuYXBwbHlGb3JjZShjdHgyKTtcbiAgICAgICAgLy8gYXdhaXQgbWQudXBkYXRlQ1BVcG9zKCk7XG4gICAgICB9XG4gICAgICBhd2FpdCBtZC5kZXZpY2UucXVldWUub25TdWJtaXR0ZWRXb3JrRG9uZSgpO1xuICAgICAgbGV0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX3JlbmRlciArPSBlbmQgLSBzdGFydDtcbiAgICAgICh3aW5kb3cgYXMgYW55KS5uX3JlbmRlcnMgKz0gMTtcbiAgICAgIGlmICghc3RvcCkge1xuICAgICAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZilcbiAgICAgICAgLy8gc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgIC8vICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpXG4gICAgICAgIC8vIH0sIDEpO1xuICAgICAgfVxuICAgIH07XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpO1xuXG4gICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX2RyYXcgPSAwO1xuICAgICh3aW5kb3cgYXMgYW55KS5uX2RyYXdzID0gMDtcbiAgICBjb25zdCBnID0gYXN5bmMgKCkgPT4ge1xuICAgICAgbGV0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgICBhd2FpdCBtZC51cGRhdGVDUFVwb3MoKTtcbiAgICAgIG1kLmRyYXcoKTtcbiAgICAgIGxldCBlbmQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgICh3aW5kb3cgYXMgYW55KS50X3Blcl9kcmF3ICs9IGVuZCAtIHN0YXJ0O1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLm5fZHJhd3MgKz0gMTtcbiAgICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZylcbiAgICAgIH0sIDMwKTtcbiAgICB9O1xuICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShnKTtcbiAgfSk7XG5cbiAgKHdpbmRvdyBhcyBhbnkpLnN0YXRzID0gKCkgPT4ge1xuICAgIGNvbnNvbGUubG9nKFwidF9wZXJfcmVuZGVyXCIsIHdpbmRvdy50X3Blcl9yZW5kZXIpO1xuICAgIGNvbnNvbGUubG9nKFwibl9yZW5kZXJzXCIsIHdpbmRvdy5uX3JlbmRlcnMpO1xuICAgIGNvbnNvbGUubG9nKFwiYXZnXCIsIHdpbmRvdy50X3Blcl9yZW5kZXIgLyB3aW5kb3cubl9yZW5kZXJzKTtcbiAgICBjb25zb2xlLmxvZyhcInRfcGVyX2RyYXdcIiwgd2luZG93LnRfcGVyX2RyYXcpO1xuICAgIGNvbnNvbGUubG9nKFwibl9kcmF3c1wiLCB3aW5kb3cubl9kcmF3cyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdcIiwgd2luZG93LnRfcGVyX2RyYXcgLyB3aW5kb3cubl9kcmF3cyk7XG4gIH1cblxuXG4gIGZ1bmN0aW9uIGNhbmNlbCgpIHtcbiAgICBzdG9wID0gdHJ1ZTtcbiAgfVxuICAod2luZG93IGFzIGFueSkubWQgPSBtZDtcbiAgKHdpbmRvdyBhcyBhbnkpLmN0eDIgPSBjdHgyO1xuICAod2luZG93IGFzIGFueSkuY2FuY2VsID0gY2FuY2VsO1xufSk7XG5cbiJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==