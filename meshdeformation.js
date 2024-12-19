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


var<workgroup>dir_x: atomic<i32>;
var<workgroup>dir_y: atomic<i32>;
var<workgroup>coeff: atomic<u32>;

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

@compute @workgroup_size(1, 1, 1)
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
          atomicAdd(& coeff, u32(10000 * local_coeff));
          atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
          atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
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

    let local_coeff = f32(10) / r2;
    atomicAdd(& coeff, u32(10000 * local_coeff));
    atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
    atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
  }

  // Wait for all workgroup threads to finish simulating
  // workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    let total_coeff = f32(atomicLoad(& coeff)) / 10000;
    if (total_coeff != 0) {
      var d_x = f32(atomicLoad(& dir_x)) / (1000 * total_coeff);
      var d_y = f32(atomicLoad(& dir_y)) / (1000 * total_coeff);

      let s_dx = sign(d_x);
      let s_dy = sign(d_y);
      let a_dx = abs(d_x);
      let a_dy = abs(d_y);

      d_x = s_dx * min(a_dx, f32(0.5));
      d_y = s_dy * min(a_dy, f32(0.5));

      x_pos_out[offset + global_id.x] = x + d_x;
      y_pos_out[offset + global_id.x] = y + d_y;
    } else {
      x_pos_out[offset + global_id.x] = x;
      y_pos_out[offset + global_id.x] = y;
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
        const dispatch_x = 256;
        for (let offset = 0; offset < this.n_elems; offset += dispatch_x) {
            let input = new Uint32Array([offset]);
            this.device.queue.writeBuffer(this.offset_buf, 0, input.buffer, 0, 4);
            let buffers = [input_x, input_y, this.intensity_map_buf, output_x, output_y, this.offset_buf];
            const bindGroup = this.device.createBindGroup({
                layout: this.force_bind_group_layout,
                entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
            });
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
        let buffers = [input_x, input_y, output_x, output_y, this.offset_buf, this.debug_buf];
        const bindGroup = this.device.createBindGroup({
            label: "constraint_bind_group",
            layout: this.constraint_bind_group_layout,
            entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
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
            passEncoder.setBindGroup(0, bindGroup);
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
    let md = new meshdeformation_1.MeshDeformation(ctx, 25, 25, ctx.canvas.width / 25, 10, 100, 5);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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LENBQUMsQ0FBQztRQUM3QyxPQUFPLENBQUMsR0FBRyxDQUFDLFNBQVMsRUFBRSxNQUFNLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDdkMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLFVBQVUsR0FBRyxNQUFNLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDekQsQ0FBQztJQUdELFNBQVMsTUFBTTtRQUNiLElBQUksR0FBRyxJQUFJLENBQUM7SUFDZCxDQUFDO0lBQ0EsTUFBYyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7SUFDdkIsTUFBYyxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7SUFDM0IsTUFBYyxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7QUFDbEMsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9jb25zdHJhaW50LnRzIiwid2VicGFjazovL3lvdXJQcm9qZWN0Ly4vc3JjL2ZvcmNlcy50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9pbml0LnRzIiwid2VicGFjazovL3lvdXJQcm9qZWN0Ly4vc3JjL21lc2hkZWZvcm1hdGlvbi50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC93ZWJwYWNrL2Jvb3RzdHJhcCIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbImNvbnN0IGVkZ2VzID0gW1xuICBbLTEsIDBdLFxuICBbMSwgMF0sXG4gIFswLCAtMV0sXG4gIFswLCAxXSxcbiAgWzEsIDFdLFxuICBbLTEsIC0xXSxcbiAgWzEsIC0xXSxcbiAgWy0xLCAxXSxcbl07XG5cblxuZXhwb3J0IGZ1bmN0aW9uIGJ1aWxkKGdyaWRfeDogbnVtYmVyLCBncmlkX3k6IG51bWJlciwgbWluX2Rpc3Q6IG51bWJlciwgbWF4X2Rpc3Q6IG51bWJlcikge1xuICBsZXQgc3JjID0gYFxuQGdyb3VwKDApIEBiaW5kaW5nKDApXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geV9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygyKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygzKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZyg0KVxudmFyPHVuaWZvcm0+b2Zmc2V0OiB1MzI7XG5cbkBncm91cCgwKSBAYmluZGluZyg1KVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGU+ZGVidWc6IGFycmF5PHUzMj47XG5cblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDI1NilcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuZ2xvYmFsX2lkIDogdmVjM3UsXG5cbiAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZClcbmxvY2FsX2lkIDogdmVjM3UsXG4pIHtcbiAgLy8gQ29vcmRpbmF0ZXMgb2YgcGFydGljbGUgZm9yIHRoaXMgdGhyZWFkXG4gIGxldCBpID0gb2Zmc2V0ICsgZ2xvYmFsX2lkLng7XG4gIGxldCBncmlkX3kgPSBpMzIoaSAvICR7Z3JpZF94fSk7XG4gIGxldCBncmlkX3ggPSBpMzIoaSAlICR7Z3JpZF94fSk7XG5cbiAgbGV0IGN1cnJfeCA9IHhfcG9zX291dFtvZmZzZXQgKyBnbG9iYWxfaWQueF07XG4gIGxldCBjdXJyX3kgPSB5X3Bvc19vdXRbb2Zmc2V0ICsgZ2xvYmFsX2lkLnhdO1xuICBsZXQgb3JnX3ggPSB4X3Bvc1tvZmZzZXQgKyBnbG9iYWxfaWQueF07XG4gIGxldCBvcmdfeSA9IHlfcG9zW29mZnNldCArIGdsb2JhbF9pZC54XTtcblxuICB2YXIgaW52YWxpZCA9IGZhbHNlO1xuYDtcblxuICBmb3IgKGxldCBlZGdlIG9mIGVkZ2VzKSB7XG4gICAgc3JjICs9IGBcbiAge1xuICAgIGxldCBlX2d4ID0gZ3JpZF94ICsgJHtlZGdlWzBdfTtcbiAgICBsZXQgZV9neSA9IGdyaWRfeSArICR7ZWRnZVsxXX07XG4gICAgaWYgKGVfZ3ggPiAwICYmIGVfZ3ggPCAke2dyaWRfeH0gJiYgZV9neSA+IDAgJiYgZV9neSA8ICR7Z3JpZF95fSkge1xuICAgICAgbGV0IGVfaSA9IGVfZ3kgKiAke2dyaWRfeH0gKyBlX2d4O1xuICAgICAgbGV0IG5feCA9IHhfcG9zX291dFtlX2ldO1xuICAgICAgbGV0IG5feSA9IHlfcG9zX291dFtlX2ldO1xuICAgICAgbGV0IGRfeCA9IG5feCAtIGN1cnJfeDtcbiAgICAgIGxldCBkX3kgPSBuX3kgLSBjdXJyX3k7XG5cbiAgICAgIGxldCBkX24yID0gZF94ICogZF94ICsgZF95ICogZF95O1xuICAgICAgaWYgKGRfbjIgPCAoJHttaW5fZGlzdH0gKiAke21pbl9kaXN0fSkgfHwgZF9uMiA+ICgke21heF9kaXN0fSAqICR7bWF4X2Rpc3R9KSkge1xuICAgICAgICBpbnZhbGlkID0gdHJ1ZTtcbiAgICAgIH1cbiAgICB9XG4gIH1cbmBcbiAgfVxuXG4gIHNyYyArPSBgXG4gIGlmIChpbnZhbGlkKSB7XG4gICAgeF9wb3Nfb3V0W2ldID0geF9wb3NbaV07XG4gICAgeV9wb3Nfb3V0W2ldID0geV9wb3NbaV07XG4gIH1cbmBcblxuICBzcmMgKz0gXCJ9XFxuXCI7XG4gIHJldHVybiBzcmNcbn1cbiIsImV4cG9ydCBmdW5jdGlvbiBidWlsZCh3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlciwgZ3JpZF94OiBudW1iZXIsIGdyaWRfc3BhY2luZzogbnVtYmVyKSB7XG4gIHJldHVybiBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygxKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDIpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IGludGVuc2l0eV9tYXA6IGFycmF5PHUzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygzKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZyg0KVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZyg1KVxudmFyPHVuaWZvcm0+b2Zmc2V0OiB1MzI7XG5cblxudmFyPHdvcmtncm91cD5kaXJfeDogYXRvbWljPGkzMj47XG52YXI8d29ya2dyb3VwPmRpcl95OiBhdG9taWM8aTMyPjtcbnZhcjx3b3JrZ3JvdXA+Y29lZmY6IGF0b21pYzx1MzI+O1xuXG5mbiBwaXhlbFRvSW50ZW5zaXR5KF9weDogdTMyKSAtPiBmMzIge1xuICB2YXIgcHggPSBfcHg7XG4gIGxldCByID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBnID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBiID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBhID0gZjMyKHB4ICUgMjU2KTtcbiAgbGV0IGludGVuc2l0eTogZjMyID0gKGEgLyAyNTUpICogKDEgLSAoMC4yMTI2ICogciArIDAuNzE1MiAqIGcgKyAwLjA3MjIgKiBiKSk7XG5cbiAgcmV0dXJuIGludGVuc2l0eTtcbn1cblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDEsIDEsIDEpXG5mbiBtYWluKFxuICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZClcbmdsb2JhbF9pZCA6IHZlYzN1LFxuXG4gIEBidWlsdGluKGxvY2FsX2ludm9jYXRpb25faWQpXG5sb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIC8vIENvb3JkaW5hdGVzIG9mIHBhcnRpY2xlIGZvciB0aGlzIHRocmVhZFxuICBsZXQgaSA9IG9mZnNldCArIGdsb2JhbF9pZC54O1xuXG4gIGxldCBncmlkX3ggPSBpICUgJHtncmlkX3h9O1xuICBsZXQgZ3JpZF95ID0gaSAvICR7Z3JpZF94fTtcblxuICBsZXQgeCA9IHhfcG9zW2ldO1xuICBsZXQgeSA9IHlfcG9zW2ldO1xuXG4gIC8vIENvb3JkaW5hdGVzIHRvIGxvb2t1cCBpbiBpbnRlbnNpdHlfbWFwXG4gIGZvciAodmFyIHNfeTogaTMyID0gMDsgc195IDwgNTA7IHNfeSsrKSB7XG4gICAgZm9yICh2YXIgc196OiBpMzIgPSAwOyBzX3ogPCA1MDsgc196KyspIHtcbiAgICAgIGxldCBmX3kgPSBpMzIoZmxvb3IoeSkpICsgaTMyKDUwICogbG9jYWxfaWQueSkgKyBzX3kgLSAyNTtcbiAgICAgIGxldCBmX3ggPSBpMzIoZmxvb3IoeCkpICsgaTMyKDUwICogbG9jYWxfaWQueikgKyBzX3ogLSAyNTtcbiAgICAgIGxldCBkX3g6IGYzMiA9IGYzMihmX3gpIC0geDtcbiAgICAgIGxldCBkX3k6IGYzMiA9IGYzMihmX3kpIC0geTtcbiAgICAgIGxldCByMjogZjMyID0gZF94ICogZF94ICsgZF95ICogZF95O1xuICAgICAgbGV0IHI6IGYzMiA9IHNxcnQocjIpO1xuXG4gICAgICAvLyBGaW5kIHRoZSBmb3JjZSBleGVydGVkIG9uIHRoZSBwYXJ0aWNsZSBieSBjb250ZW50cyBvZiB0aGUgaW50ZXNpdHkgbWFwLlxuICAgICAgaWYgKGZfeSA+PSAwICYmIGZfeSA8ICR7aGVpZ2h0fSAmJiBmX3ggPj0gMCAmJiBmX3ggPCAke3dpZHRofSkge1xuICAgICAgICBsZXQgZl9pID0gZl95ICogJHt3aWR0aH0gKyBmX3g7XG4gICAgICAgIGxldCBpbnRlbnNpdHkgPSBwaXhlbFRvSW50ZW5zaXR5KGludGVuc2l0eV9tYXBbZl9pXSk7XG5cbiAgICAgICAgaWYgKHIgIT0gMCkge1xuICAgICAgICAgIGxldCBsb2NhbF9jb2VmZiA9IDEwMCAqIGludGVuc2l0eSAvIHIyO1xuICAgICAgICAgIGF0b21pY0FkZCgmIGNvZWZmLCB1MzIoMTAwMDAgKiBsb2NhbF9jb2VmZikpO1xuICAgICAgICAgIGF0b21pY0FkZCgmIGRpcl94LCBpMzIoMTAwMCAqIGxvY2FsX2NvZWZmICogZF94IC8gcikpO1xuICAgICAgICAgIGF0b21pY0FkZCgmIGRpcl95LCBpMzIoMTAwMCAqIGxvY2FsX2NvZWZmICogZF95IC8gcikpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgaWYgKGxvY2FsX2lkLnkgPT0gMCAmJiBsb2NhbF9pZC56ID09IDApIHtcbiAgICBsZXQgb3JpZ2luX3ggPSBncmlkX3ggKiAke2dyaWRfc3BhY2luZ307XG4gICAgbGV0IG9yaWdpbl95ID0gZ3JpZF95ICogJHtncmlkX3NwYWNpbmd9O1xuICAgIGxldCBkX3g6IGYzMiA9IGYzMihvcmlnaW5feCkgLSB4O1xuICAgIGxldCBkX3k6IGYzMiA9IGYzMihvcmlnaW5feSkgLSB5O1xuICAgIGxldCByMjogZjMyID0gZF94ICogZF94ICsgZF95ICogZF95O1xuICAgIGxldCByOiBmMzIgPSBzcXJ0KHIyKTtcblxuICAgIGxldCBsb2NhbF9jb2VmZiA9IGYzMigxMCkgLyByMjtcbiAgICBhdG9taWNBZGQoJiBjb2VmZiwgdTMyKDEwMDAwICogbG9jYWxfY29lZmYpKTtcbiAgICBhdG9taWNBZGQoJiBkaXJfeCwgaTMyKDEwMDAgKiBsb2NhbF9jb2VmZiAqIGRfeCAvIHIpKTtcbiAgICBhdG9taWNBZGQoJiBkaXJfeSwgaTMyKDEwMDAgKiBsb2NhbF9jb2VmZiAqIGRfeSAvIHIpKTtcbiAgfVxuXG4gIC8vIFdhaXQgZm9yIGFsbCB3b3JrZ3JvdXAgdGhyZWFkcyB0byBmaW5pc2ggc2ltdWxhdGluZ1xuICAvLyB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgLy8gT24gYSBzaW5nbGUgdGhyZWFkLCB1cGRhdGUgdGhlIG91dHB1dCBwb3NpdGlvbiBmb3IgdGhlIGN1cnJlbnQgcGFydGljbGVcbiAgaWYgKGxvY2FsX2lkLnkgPT0gMCAmJiBsb2NhbF9pZC56ID09IDApIHtcbiAgICBsZXQgdG90YWxfY29lZmYgPSBmMzIoYXRvbWljTG9hZCgmIGNvZWZmKSkgLyAxMDAwMDtcbiAgICBpZiAodG90YWxfY29lZmYgIT0gMCkge1xuICAgICAgdmFyIGRfeCA9IGYzMihhdG9taWNMb2FkKCYgZGlyX3gpKSAvICgxMDAwICogdG90YWxfY29lZmYpO1xuICAgICAgdmFyIGRfeSA9IGYzMihhdG9taWNMb2FkKCYgZGlyX3kpKSAvICgxMDAwICogdG90YWxfY29lZmYpO1xuXG4gICAgICBsZXQgc19keCA9IHNpZ24oZF94KTtcbiAgICAgIGxldCBzX2R5ID0gc2lnbihkX3kpO1xuICAgICAgbGV0IGFfZHggPSBhYnMoZF94KTtcbiAgICAgIGxldCBhX2R5ID0gYWJzKGRfeSk7XG5cbiAgICAgIGRfeCA9IHNfZHggKiBtaW4oYV9keCwgZjMyKDAuNSkpO1xuICAgICAgZF95ID0gc19keSAqIG1pbihhX2R5LCBmMzIoMC41KSk7XG5cbiAgICAgIHhfcG9zX291dFtvZmZzZXQgKyBnbG9iYWxfaWQueF0gPSB4ICsgZF94O1xuICAgICAgeV9wb3Nfb3V0W29mZnNldCArIGdsb2JhbF9pZC54XSA9IHkgKyBkX3k7XG4gICAgfSBlbHNlIHtcbiAgICAgIHhfcG9zX291dFtvZmZzZXQgKyBnbG9iYWxfaWQueF0gPSB4O1xuICAgICAgeV9wb3Nfb3V0W29mZnNldCArIGdsb2JhbF9pZC54XSA9IHk7XG4gICAgfVxuICB9XG59XG5gXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gYnVpbGQobl9lbGVtczogbnVtYmVyLCBncmlkX3g6IG51bWJlciwgZ3JpZF9zcGFjaW5nOiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGU+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDI1NilcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuICBnbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxuICBsb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIGlmIChnbG9iYWxfaWQueCA8ICR7bl9lbGVtc30pIHtcbiAgICAgIGxldCB5ID0gZ2xvYmFsX2lkLnggLyAke2dyaWRfeH07XG4gICAgICBsZXQgeCA9IGdsb2JhbF9pZC54ICUgJHtncmlkX3h9O1xuXG4gICAgICB4X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeCAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gICAgICB5X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeSAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gIH1cbn1cbmBcbn1cbiIsImltcG9ydCB7IGJ1aWxkIGFzIGluaXRidWlsZCB9IGZyb20gJy4vaW5pdCc7XG5pbXBvcnQgeyBidWlsZCBhcyBmb3JjZXNidWlsZCB9IGZyb20gJy4vZm9yY2VzJztcbmltcG9ydCB7IGJ1aWxkIGFzIGNvbnN0cmFpbnRidWlsZCB9IGZyb20gJy4vY29uc3RyYWludCc7XG5cbmNvbnN0IGVkZ2VzID0gW1xuICBbLTEsIDBdLFxuICBbMSwgMF0sXG4gIFswLCAtMV0sXG4gIFswLCAxXSxcbiAgWzEsIDFdLFxuICBbLTEsIC0xXSxcbl07XG5cbmV4cG9ydCBjbGFzcyBNZXNoRGVmb3JtYXRpb24ge1xuICBjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRDtcbiAgZ3JpZF94OiBudW1iZXI7XG4gIGdyaWRfeTogbnVtYmVyO1xuICBncmlkX3NwYWNpbmc6IG51bWJlcjtcbiAgbWluX2Rpc3Q6IG51bWJlcjtcbiAgbWF4X2Rpc3Q6IG51bWJlcjtcbiAgcmFkaXVzOiBudW1iZXI7XG5cbiAgZHJhd19lZGdlczogYm9vbDtcblxuICBuX2VsZW1zOiBudW1iZXI7XG4gIHhfcG9zOiBGbG9hdDMyQXJyYXk7XG4gIHlfcG9zOiBGbG9hdDMyQXJyYXk7XG5cbiAgaW5pdGlhbGl6YXRpb25fZG9uZTogUHJvbWlzZTx2b2lkPjtcbiAgZGV2aWNlOiBHUFVEZXZpY2U7XG4gIGZvcmNlX21vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICBhY3RpdmVfYnVmZmVyX2lkOiBudW1iZXI7XG4gIHhfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIHlfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIC8vIEJ1ZmZlcnMgdG8gcmVhZCB2YWx1ZXMgYmFjayB0byB0aGUgQ1BVIGZvciBkcmF3aW5nXG4gIHN0YWdpbmdfeF9idWY6IEdQVUJ1ZmZlcjtcbiAgc3RhZ2luZ195X2J1ZjogR1BVQnVmZmVyO1xuICAvLyBCdWZmZXIgdG8gd3JpdGUgdmFsdWUgdG8gZnJvbSB0aGUgQ1BVIGZvciBhZGp1c3Rpbmcgd2VpZ2h0c1xuICBzdGFnaW5nX2ludGVuc2l0eV9idWY6IEdQVUJ1ZmZlcjtcbiAgaW50ZW5zaXR5X21hcF9idWY6IEdQVUJ1ZmZlcjtcblxuICBvZmZzZXRfYnVmOiBHUFVCdWZmZXI7XG5cbiAgZm9yY2VfYmluZF9ncm91cF9sYXlvdXQ6IEdQVUJpbmRHcm91cExheW91dDtcblxuICBjb25zdHJhaW50X21vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICBjb25zdHJhaW50X2JpbmRfZ3JvdXBfbGF5b3V0OiBHUFVCaW5kR3JvdXBMYXlvdXQ7XG5cbiAgZGVidWdfYnVmOiBHUFVCdWZmZXI7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQsXG4gICAgZ3JpZF94OiBudW1iZXIsXG4gICAgZ3JpZF95OiBudW1iZXIsXG4gICAgZ3JpZF9zcGFjaW5nOiBudW1iZXIsXG4gICAgbWluX2Rpc3Q6IG51bWJlcixcbiAgICBtYXhfZGlzdDogbnVtYmVyLFxuICAgIHJhZGl1czogbnVtYmVyXG4gICkge1xuICAgIHRoaXMuY3R4ID0gY3R4O1xuICAgIHRoaXMuZ3JpZF94ID0gZ3JpZF94O1xuICAgIHRoaXMuZ3JpZF95ID0gZ3JpZF95O1xuICAgIHRoaXMuZ3JpZF9zcGFjaW5nID0gZ3JpZF9zcGFjaW5nO1xuICAgIHRoaXMubWluX2Rpc3QgPSBtaW5fZGlzdDtcbiAgICB0aGlzLm1heF9kaXN0ID0gbWF4X2Rpc3Q7XG4gICAgdGhpcy5yYWRpdXMgPSByYWRpdXM7XG5cbiAgICB0aGlzLmRyYXdfZWRnZXMgPSB0cnVlO1xuXG4gICAgdGhpcy5uX2VsZW1zID0gdGhpcy5ncmlkX3ggKiB0aGlzLmdyaWRfeTtcblxuICAgIHRoaXMuaW5pdGlhbGl6YXRpb25fZG9uZSA9IHRoaXMuYXN5bmNfaW5pdCgpO1xuICB9XG5cbiAgYXN5bmMgYXN5bmNfaW5pdCgpIHtcbiAgICBjb25zdCBhZGFwdGVyID0gYXdhaXQgbmF2aWdhdG9yLmdwdS5yZXF1ZXN0QWRhcHRlcigpO1xuICAgIHRoaXMuZGV2aWNlID0gYXdhaXQgYWRhcHRlci5yZXF1ZXN0RGV2aWNlKCk7XG4gICAgY29uc29sZS5sb2coXCJDcmVhdGUgY29tcHV0ZSBzaGFkZXJcIik7XG4gICAgdGhpcy5mb3JjZV9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogZm9yY2VzYnVpbGQodGhpcy5jdHguY2FudmFzLndpZHRoLCB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0LCB0aGlzLmdyaWRfeCwgdGhpcy5ncmlkX3NwYWNpbmcpLFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBDcmVhdGUgY29tcHV0ZSBzaGFkZXJcIik7XG5cbiAgICB0aGlzLmRlYnVnX2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJkZWJ1Z1wiLFxuICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICB9KTtcblxuICAgIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZCA9IDA7XG4gICAgdGhpcy54X3Bvc19idWZmZXJzID0gW1xuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieF9wb3NbMF1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ4X3Bvc1sxXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICBdO1xuICAgIHRoaXMueV9wb3NfYnVmZmVycyA9IFtcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInlfcG9zWzBdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieV9wb3NbMV1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgXTtcblxuICAgIHRoaXMuc3RhZ2luZ194X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX3hfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0YWdpbmdfeV9idWZcIixcbiAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1JFQUQgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcblxuICAgIHRoaXMuc3RhZ2luZ19pbnRlbnNpdHlfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0YWdpbmdfaW50ZW5zaXR5X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1dSSVRFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkMsXG4gICAgfSk7XG4gICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJpbnRlbnNpdHlfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLmN0eC5jYW52YXMud2lkdGggKiB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0ICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG5cbiAgICB0aGlzLm9mZnNldF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwib2Zmc2V0X2J1ZlwiLFxuICAgICAgc2l6ZTogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5VTklGT1JNIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFsbG9jYXRlIGJ1ZmZlcnNcIik7XG5cbiAgICB0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMixcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDMsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA0LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJ1bmlmb3JtXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICAvLyBpbnRpYWxpemUgdGhpcy54X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF0gYW5kXG4gICAgLy8gdGhpcy55X3Bvc19idWZmZXJzWypdIHRvIGJlIGEgZ3JpZFxuICAgIGNvbnN0IGluaXRfc2hhZGVyID0gaW5pdGJ1aWxkKHRoaXMubl9lbGVtcywgdGhpcy5ncmlkX3gsIHRoaXMuZ3JpZF9zcGFjaW5nKTtcbiAgICBjb25zdCBpbml0QmluZEdyb3VwTGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIGNvbnNvbGUubG9nKFwiQ3JlYXRlIGluaXQgc2hhZGVyXCIpO1xuICAgIGNvbnN0IGluaXRfbW9kdWxlID0gdGhpcy5kZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcbiAgICAgIGNvZGU6IGluaXRfc2hhZGVyLFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBDcmVhdGUgaW5pdCBzaGFkZXJcIik7XG4gICAgY29uc3QgY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgIGxhYmVsOiBcImNvbXB1dGUgZm9yY2VcIixcbiAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbaW5pdEJpbmRHcm91cExheW91dF0sXG4gICAgICB9KSxcbiAgICAgIGNvbXB1dGU6IHtcbiAgICAgICAgbW9kdWxlOiBpbml0X21vZHVsZSxcbiAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICB9LFxuICAgIH0pO1xuICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcblxuICAgIGNvbnN0IGJpbmRHcm91cCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cCh7XG4gICAgICBsYXlvdXQ6IGluaXRCaW5kR3JvdXBMYXlvdXQsXG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH1cbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luQ29tcHV0ZVBhc3MoKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZShjb21wdXRlUGlwZWxpbmUpO1xuICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgIHBhc3NFbmNvZGVyLmRpc3BhdGNoV29ya2dyb3VwcyhNYXRoLmNlaWwodGhpcy5uX2VsZW1zIC8gMjU2KSk7XG4gICAgcGFzc0VuY29kZXIuZW5kKCk7XG4gICAgY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgdGhpcy54X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF0sIDAsXG4gICAgICB0aGlzLnN0YWdpbmdfeF9idWYsIDAsXG4gICAgICB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZVxuICAgICk7XG4gICAgY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgdGhpcy55X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF0sIDAsXG4gICAgICB0aGlzLnN0YWdpbmdfeV9idWYsIDAsXG4gICAgICB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZVxuICAgICk7XG4gICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuXG4gICAgYXdhaXQgdGhpcy51cGRhdGVDUFVwb3MoKTtcbiAgICBjb25zb2xlLmxvZyhcImRvbmUgYXN5bmMgaW5pdFwiKTtcblxuICAgIGNvbnN0IGNvbnN0cmFpbnRfc3JjID0gY29uc3RyYWludGJ1aWxkKHRoaXMuZ3JpZF94LCB0aGlzLmdyaWRfeSwgdGhpcy5taW5fZGlzdCwgdGhpcy5tYXhfZGlzdCk7XG4gICAgLy8gY29uc29sZS5sb2coY29uc3RyYWludF9zcmMpO1xuICAgIHRoaXMuY29uc3RyYWludF9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogY29uc3RyYWludF9zcmMsXG4gICAgfSk7XG4gICAgdGhpcy5jb25zdHJhaW50X2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGxhYmVsOiBcImNvbnN0cmFpbnRfYmluZF9ncm91cF9sYXlvdXRcIixcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMixcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDMsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA0LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInVuaWZvcm1cIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG4gIH1cblxuICBhc3luYyB1cGRhdGVDUFVwb3MoKSB7XG4gICAgLy8gY29uc29sZS5sb2coXCJNYXAgYnVmZmVycyBmb3IgcmVhZGluZ1wiKTtcbiAgICBsZXQgbV94ID0gdGhpcy5zdGFnaW5nX3hfYnVmLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIGxldCBtX3kgPSB0aGlzLnN0YWdpbmdfeV9idWYubWFwQXN5bmMoR1BVTWFwTW9kZS5SRUFELCAwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgYXdhaXQgbV94O1xuICAgIGF3YWl0IG1feTtcblxuICAgIC8vIGNvbnNvbGUubG9nKFwiY29weWluZyB4IGJ1ZmZlciB0byBDUFVcIik7XG4gICAgY29uc3QgY29weUFycmF5QnVmZmVyWCA9IHRoaXMuc3RhZ2luZ194X2J1Zi5nZXRNYXBwZWRSYW5nZSgwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgY29uc3QgZGF0YVggPSBjb3B5QXJyYXlCdWZmZXJYLnNsaWNlKCk7XG4gICAgdGhpcy54X3BvcyA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YVgpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJjb3B5aW5nIHkgYnVmZmVyIHRvIENQVVwiKTtcbiAgICBjb25zdCBjb3B5QXJyYXlCdWZmZXJZID0gdGhpcy5zdGFnaW5nX3lfYnVmLmdldE1hcHBlZFJhbmdlKDAsIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplKTtcbiAgICBjb25zdCBkYXRhWSA9IGNvcHlBcnJheUJ1ZmZlclkuc2xpY2UoKTtcbiAgICB0aGlzLnlfcG9zID0gbmV3IEZsb2F0MzJBcnJheShkYXRhWSk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcInVubWFwIGJ1ZmZlcnNcIik7XG4gICAgdGhpcy5zdGFnaW5nX3hfYnVmLnVubWFwKCk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmLnVubWFwKCk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcIkRvbmUgdXBkYXRlQ1BVcG9zXCIpO1xuICB9XG5cbiAgYXN5bmMgYXBwbHlGb3JjZShjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCkge1xuICAgIGxldCBpZGF0YSA9IGN0eC5nZXRJbWFnZURhdGEoMCwgMCwgY3R4LmNhbnZhcy53aWR0aCwgY3R4LmNhbnZhcy5oZWlnaHQpLmRhdGE7XG4gICAgLy8gY29uc29sZS5sb2coYGIwICR7aWRhdGFbMF19LCAke2lkYXRhWzFdfSwgJHtpZGF0YVsyXX0sICR7aWRhdGFbM119YCk7XG4gICAgLy8gY29uc29sZS5sb2coYFdyaXRpbmcgJHt0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemV9LyR7aWRhdGEubGVuZ3RofSBieXRlcyBmb3IgaW1hcGApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKFxuICAgICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiwgMCwgaWRhdGEuYnVmZmVyLCAwLCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemUpO1xuXG4gICAgbGV0IGlucHV0X3ggPSB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgaW5wdXRfeSA9IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBvdXRwdXRfeCA9IHRoaXMueF9wb3NfYnVmZmVyc1sxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgb3V0cHV0X3kgPSB0aGlzLnlfcG9zX2J1ZmZlcnNbMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG5cbiAgICBjb25zdCBkaXNwYXRjaF94ID0gMjU2O1xuICAgIGZvciAobGV0IG9mZnNldCA9IDA7IG9mZnNldCA8IHRoaXMubl9lbGVtczsgb2Zmc2V0ICs9IGRpc3BhdGNoX3gpIHtcbiAgICAgIGxldCBpbnB1dCA9IG5ldyBVaW50MzJBcnJheShbb2Zmc2V0XSk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5vZmZzZXRfYnVmLCAwLCBpbnB1dC5idWZmZXIsIDAsIDQpO1xuXG4gICAgICBsZXQgYnVmZmVycyA9IFtpbnB1dF94LCBpbnB1dF95LCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLCBvdXRwdXRfeCwgb3V0cHV0X3ksIHRoaXMub2Zmc2V0X2J1Zl07XG4gICAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgICBsYXlvdXQ6IHRoaXMuZm9yY2VfYmluZF9ncm91cF9sYXlvdXQsXG4gICAgICAgIGVudHJpZXM6IGJ1ZmZlcnMubWFwKChiLCBpKSA9PiB7IHJldHVybiB7IGJpbmRpbmc6IGksIHJlc291cmNlOiB7IGJ1ZmZlcjogYiB9IH07IH0pXG4gICAgICB9KTtcblxuICAgICAgY29uc3QgY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgICAgbGFiZWw6IFwiZm9yY2VwaXBlbGluZVwiLFxuICAgICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbdGhpcy5mb3JjZV9iaW5kX2dyb3VwX2xheW91dF0sXG4gICAgICAgIH0pLFxuICAgICAgICBjb21wdXRlOiB7XG4gICAgICAgICAgbW9kdWxlOiB0aGlzLmZvcmNlX21vZHVsZSxcbiAgICAgICAgICBlbnRyeVBvaW50OiBcIm1haW5cIixcbiAgICAgICAgfSxcbiAgICAgIH0pO1xuICAgICAgLy8gY29uc29sZS5sb2coXCJjcmVhdGVkIHBpcGVsaW5lXCIpO1xuXG4gICAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG4gICAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luQ29tcHV0ZVBhc3MoKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldFBpcGVsaW5lKGNvbXB1dGVQaXBlbGluZSk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICAgIHBhc3NFbmNvZGVyLmRpc3BhdGNoV29ya2dyb3VwcyhkaXNwYXRjaF94LCAxLCAxKTtcbiAgICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgICAgLy8gY29uc29sZS5sb2coXCJlbmNvZGVkIGNvbXB1dGVcIik7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG4gICAgfVxuXG4gICAgbGV0IGJ1ZmZlcnMgPSBbaW5wdXRfeCwgaW5wdXRfeSwgb3V0cHV0X3gsIG91dHB1dF95LCB0aGlzLm9mZnNldF9idWYsIHRoaXMuZGVidWdfYnVmXTtcbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGFiZWw6IFwiY29uc3RyYWludF9iaW5kX2dyb3VwXCIsXG4gICAgICBsYXlvdXQ6IHRoaXMuY29uc3RyYWludF9iaW5kX2dyb3VwX2xheW91dCxcbiAgICAgIGVudHJpZXM6IGJ1ZmZlcnMubWFwKChiLCBpKSA9PiB7IHJldHVybiB7IGJpbmRpbmc6IGksIHJlc291cmNlOiB7IGJ1ZmZlcjogYiB9IH07IH0pXG4gICAgfSk7XG4gICAgZm9yIChsZXQgb2Zmc2V0ID0gMDsgb2Zmc2V0IDwgdGhpcy5uX2VsZW1zOyBvZmZzZXQgKz0gMjU2KSB7XG4gICAgICBsZXQgaW5wdXQgPSBuZXcgVWludDMyQXJyYXkoW29mZnNldF0pO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUud3JpdGVCdWZmZXIoXG4gICAgICAgIHRoaXMub2Zmc2V0X2J1ZiwgMCwgaW5wdXQuYnVmZmVyLCAwLCA0KTtcblxuICAgICAgY29uc3QgY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgICAgbGFiZWw6IFwiY29uc3RyYWludHBpcGVsaW5lXCIsXG4gICAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICAgIGJpbmRHcm91cExheW91dHM6IFt0aGlzLmNvbnN0cmFpbnRfYmluZF9ncm91cF9sYXlvdXRdLFxuICAgICAgICB9KSxcbiAgICAgICAgY29tcHV0ZToge1xuICAgICAgICAgIG1vZHVsZTogdGhpcy5jb25zdHJhaW50X21vZHVsZSxcbiAgICAgICAgICBlbnRyeVBvaW50OiBcIm1haW5cIixcbiAgICAgICAgfSxcbiAgICAgIH0pO1xuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZShjb21wdXRlUGlwZWxpbmUpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoMSwgMSwgMSk7XG4gICAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb21wdXRlXCIpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgIH1cblxuICAgIGNvbnN0IGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAvLyBDb3B5IG91dHB1dCBidWZmZXIgdG8gc3RhZ2luZyBidWZmZXJcbiAgICBjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICBvdXRwdXRfeCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJ4IGNvcHlpbmdcIiwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUsIFwiYnl0ZXNcIik7XG4gICAgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgb3V0cHV0X3ksIDAsIHRoaXMuc3RhZ2luZ195X2J1ZiwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemUpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwieSBjb3B5aW5nXCIsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplLCBcImJ5dGVzXCIpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb3B5IHRvIGJ1ZmZlcnNcIiwgdGhpcy5hY3RpdmVfYnVmZmVyX2lkKTtcblxuICAgIC8vIEVuZCBmcmFtZSBieSBwYXNzaW5nIGFycmF5IG9mIGNvbW1hbmQgYnVmZmVycyB0byBjb21tYW5kIHF1ZXVlIGZvciBleGVjdXRpb25cbiAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJkb25lIHN1Ym1pdCB0byBxdWV1ZVwiKTtcblxuICAgIC8vIFN3YXAgaW5wdXQgYW5kIG91dHB1dDpcbiAgICB0aGlzLmFjdGl2ZV9idWZmZXJfaWQgPSAxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkO1xuXG4gICAgLy8gYXdhaXQgdGhpcy51cGRhdGVDUFVwb3MoKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImRvbmUgYXBwbHlGb3JjZVwiKTtcbiAgfVxuXG4gIGRyYXcoKSB7XG4gICAgdGhpcy5jdHguY2xlYXJSZWN0KDAsIDAsIHRoaXMuY3R4LmNhbnZhcy53aWR0aCwgdGhpcy5jdHguY2FudmFzLmhlaWdodCk7XG4gICAgZm9yIChsZXQgeWlkeCA9IDA7IHlpZHggPCB0aGlzLmdyaWRfeTsgeWlkeCsrKSB7XG4gICAgICBmb3IgKGxldCB4aWR4ID0gMDsgeGlkeCA8IHRoaXMuZ3JpZF94OyB4aWR4KyspIHtcbiAgICAgICAgbGV0IGkgPSB5aWR4ICogdGhpcy5ncmlkX3ggKyB4aWR4O1xuXG4gICAgICAgIGxldCB4ID0gdGhpcy54X3Bvc1tpXTtcbiAgICAgICAgbGV0IHkgPSB0aGlzLnlfcG9zW2ldO1xuXG4gICAgICAgIHRoaXMuY3R4LnN0cm9rZVN0eWxlID0gXCIjZmYwMDAwNWZcIjtcbiAgICAgICAgdGhpcy5jdHguYmVnaW5QYXRoKCk7XG4gICAgICAgIHRoaXMuY3R4LmFyYyh4LCB5LCB0aGlzLnJhZGl1cywgMCwgMiAqIE1hdGguUEkpO1xuICAgICAgICB0aGlzLmN0eC5zdHJva2UoKTtcblxuICAgICAgICBpZiAodGhpcy5kcmF3X2VkZ2VzKSB7XG4gICAgICAgICAgZm9yIChsZXQgZWRnZSBvZiBlZGdlcykge1xuICAgICAgICAgICAgbGV0IGpfeGlkeCA9IHhpZHggKyBlZGdlWzBdO1xuICAgICAgICAgICAgbGV0IGpfeWlkeCA9IHlpZHggKyBlZGdlWzFdO1xuICAgICAgICAgICAgaWYgKGpfeGlkeCA8IDAgfHwgal94aWR4ID49IHRoaXMuZ3JpZF94IHx8IGpfeWlkeCA8IDAgfHwgal95aWR4ID49IHRoaXMuZ3JpZF95KSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgaiA9IGpfeWlkeCAqIHRoaXMuZ3JpZF94ICsgal94aWR4O1xuXG4gICAgICAgICAgICBsZXQgal94ID0gdGhpcy54X3Bvc1tqXTtcbiAgICAgICAgICAgIGxldCBqX3kgPSB0aGlzLnlfcG9zW2pdO1xuXG4gICAgICAgICAgICB0aGlzLmN0eC5iZWdpblBhdGgoKTtcbiAgICAgICAgICAgIHRoaXMuY3R4Lm1vdmVUbyh4LCB5KTtcbiAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVUbyhqX3gsIGpfeSk7XG4gICAgICAgICAgICB0aGlzLmN0eC5zdHJva2UoKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cbiIsIi8vIFRoZSBtb2R1bGUgY2FjaGVcbnZhciBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX18gPSB7fTtcblxuLy8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbmZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG5cdHZhciBjYWNoZWRNb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdO1xuXHRpZiAoY2FjaGVkTW9kdWxlICE9PSB1bmRlZmluZWQpIHtcblx0XHRyZXR1cm4gY2FjaGVkTW9kdWxlLmV4cG9ydHM7XG5cdH1cblx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcblx0dmFyIG1vZHVsZSA9IF9fd2VicGFja19tb2R1bGVfY2FjaGVfX1ttb2R1bGVJZF0gPSB7XG5cdFx0Ly8gbm8gbW9kdWxlLmlkIG5lZWRlZFxuXHRcdC8vIG5vIG1vZHVsZS5sb2FkZWQgbmVlZGVkXG5cdFx0ZXhwb3J0czoge31cblx0fTtcblxuXHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cblx0X193ZWJwYWNrX21vZHVsZXNfX1ttb2R1bGVJZF0obW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cblx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcblx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xufVxuXG4iLCJpbXBvcnQgeyBNZXNoRGVmb3JtYXRpb24gfSBmcm9tICcuL21lc2hkZWZvcm1hdGlvbic7XG5cbmxldCBzdG9wID0gZmFsc2U7XG5cbmRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoXCJET01Db250ZW50TG9hZGVkXCIsICgpID0+IHtcbiAgbGV0IGNhbnZhcyA9IChkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcIm15Y2FudmFzXCIpIGFzIEhUTUxDYW52YXNFbGVtZW50KTtcbiAgY2FudmFzLndpZHRoID0gMTAwMDtcbiAgY2FudmFzLmhlaWdodCA9IDEwMDA7XG5cbiAgbGV0IGN0eCA9IGNhbnZhcy5nZXRDb250ZXh0KFwiMmRcIik7XG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBtYWluIGNhbnZhc1wiKTtcblxuICBsZXQgY2FudmFzMiA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwibXljYW52YXMyXCIpIGFzIEhUTUxDYW52YXNFbGVtZW50O1xuICBjYW52YXMyLndpZHRoID0gMTAwMDtcbiAgY2FudmFzMi5oZWlnaHQgPSAxMDAwO1xuICBsZXQgY3R4MiA9IGNhbnZhczIuZ2V0Q29udGV4dChcIjJkXCIpO1xuICBjYW52YXMyLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoZSkgPT4ge1xuICAgIGxldCBlbCA9IGUudGFyZ2V0IGFzIEhUTUxDYW52YXNFbGVtZW50O1xuICAgIGNvbnN0IHJlY3QgPSBlbC5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcbiAgICBjb25zdCB4ID0gZWwud2lkdGggKiAoZS5jbGllbnRYIC0gcmVjdC5sZWZ0KSAvIHJlY3Qud2lkdGg7XG4gICAgY29uc3QgeSA9IGVsLmhlaWdodCAqIChlLmNsaWVudFkgLSByZWN0LnRvcCkgLyByZWN0LmhlaWdodDtcblxuICAgIGN0eDIuYmVnaW5QYXRoKCk7XG4gICAgY3R4Mi5maWxsU3R5bGUgPSBcImJsYWNrXCI7XG4gICAgY3R4Mi5hcmMoeCwgeSwgMTAwLCAwLCAyICogTWF0aC5QSSk7XG4gICAgY3R4Mi5maWxsKCk7XG4gIH0pO1xuXG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBpbnRlcmFjdGl2ZSBjYW52YXNcIik7XG5cbiAgKHdpbmRvdyBhcyBhbnkpLm5fc3RlcHNfcGVyX2ZyYW1lID0gMTtcblxuICBsZXQgbWQgPSBuZXcgTWVzaERlZm9ybWF0aW9uKGN0eCwgMjUsIDI1LCBjdHguY2FudmFzLndpZHRoIC8gMjUsIDEwLCAxMDAsIDUpO1xuICAod2luZG93IGFzIGFueSkudF9wZXJfcmVuZGVyID0gMDtcbiAgKHdpbmRvdyBhcyBhbnkpLm5fcmVuZGVycyA9IDA7XG4gIG1kLmluaXRpYWxpemF0aW9uX2RvbmUudGhlbigoKSA9PiB7XG4gICAgY29uc3QgZiA9IGFzeW5jICgpID0+IHtcbiAgICAgIGxldCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgbWQuZHJhdygpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCAod2luZG93IGFzIGFueSkubl9zdGVwc19wZXJfZnJhbWU7IGkrKykge1xuICAgICAgICBhd2FpdCBtZC5hcHBseUZvcmNlKGN0eDIpO1xuICAgICAgICAvLyBhd2FpdCBtZC51cGRhdGVDUFVwb3MoKTtcbiAgICAgIH1cbiAgICAgIGF3YWl0IG1kLmRldmljZS5xdWV1ZS5vblN1Ym1pdHRlZFdvcmtEb25lKCk7XG4gICAgICBsZXQgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgICAod2luZG93IGFzIGFueSkudF9wZXJfcmVuZGVyICs9IGVuZCAtIHN0YXJ0O1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLm5fcmVuZGVycyArPSAxO1xuICAgICAgaWYgKCFzdG9wKSB7XG4gICAgICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmKVxuICAgICAgICAvLyBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgLy8gICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZilcbiAgICAgICAgLy8gfSwgMSk7XG4gICAgICB9XG4gICAgfTtcbiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZik7XG5cbiAgICAod2luZG93IGFzIGFueSkudF9wZXJfZHJhdyA9IDA7XG4gICAgKHdpbmRvdyBhcyBhbnkpLm5fZHJhd3MgPSAwO1xuICAgIGNvbnN0IGcgPSBhc3luYyAoKSA9PiB7XG4gICAgICBsZXQgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgIGF3YWl0IG1kLnVwZGF0ZUNQVXBvcygpO1xuICAgICAgbWQuZHJhdygpO1xuICAgICAgbGV0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX2RyYXcgKz0gZW5kIC0gc3RhcnQ7XG4gICAgICAod2luZG93IGFzIGFueSkubl9kcmF3cyArPSAxO1xuICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShnKVxuICAgICAgfSwgMzApO1xuICAgIH07XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGcpO1xuICB9KTtcblxuICAod2luZG93IGFzIGFueSkuc3RhdHMgPSAoKSA9PiB7XG4gICAgY29uc29sZS5sb2coXCJ0X3Blcl9yZW5kZXJcIiwgd2luZG93LnRfcGVyX3JlbmRlcik7XG4gICAgY29uc29sZS5sb2coXCJuX3JlbmRlcnNcIiwgd2luZG93Lm5fcmVuZGVycyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdcIiwgd2luZG93LnRfcGVyX3JlbmRlciAvIHdpbmRvdy5uX3JlbmRlcnMpO1xuICAgIGNvbnNvbGUubG9nKFwidF9wZXJfZHJhd1wiLCB3aW5kb3cudF9wZXJfZHJhdyk7XG4gICAgY29uc29sZS5sb2coXCJuX2RyYXdzXCIsIHdpbmRvdy5uX2RyYXdzKTtcbiAgICBjb25zb2xlLmxvZyhcImF2Z1wiLCB3aW5kb3cudF9wZXJfZHJhdyAvIHdpbmRvdy5uX2RyYXdzKTtcbiAgfVxuXG5cbiAgZnVuY3Rpb24gY2FuY2VsKCkge1xuICAgIHN0b3AgPSB0cnVlO1xuICB9XG4gICh3aW5kb3cgYXMgYW55KS5tZCA9IG1kO1xuICAod2luZG93IGFzIGFueSkuY3R4MiA9IGN0eDI7XG4gICh3aW5kb3cgYXMgYW55KS5jYW5jZWwgPSBjYW5jZWw7XG59KTtcblxuIl0sIm5hbWVzIjpbXSwic291cmNlUm9vdCI6IiJ9