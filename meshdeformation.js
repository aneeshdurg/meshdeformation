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

      let speed = dist2 / max_dist2;

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
    force_bind_group_layout;
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
        await this.updateCPUpos();
        console.log("done async init");
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
    let canvas2 = document.createElement("canvas");
    canvas2.width = 1000;
    canvas2.height = 1000;
    let ctx2 = canvas2.getContext("2d");
    // let idata: Uint8ClampedArray<ArrayBufferLike> = new Uint8ClampedArray(ctx2.canvas.width * ctx.canvas.height * 4);
    canvas.addEventListener("click", (e) => {
        let el = e.target;
        const rect = el.getBoundingClientRect();
        const x = el.width * (e.clientX - rect.left) / rect.width;
        const y = el.height * (e.clientY - rect.top) / rect.height;
        ctx2.beginPath();
        ctx2.fillStyle = "black";
        ctx2.arc(x, y, 100, 0, 2 * Math.PI);
        ctx2.fill();
        // idata = ctx2.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
    });
    document.getElementById("clear").addEventListener("click", () => {
        ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
    });
    document.getElementById("edges").addEventListener("click", () => {
        md.draw_edges = !md.draw_edges;
    });
    console.log("Created context for interactive canvas");
    window.n_steps_per_frame = 1;
    let n_elems = 100;
    let spacing = ctx.canvas.width / n_elems;
    let md = new meshdeformation_1.MeshDeformation(ctx, n_elems, n_elems, spacing, spacing / 4, spacing * 4, 1);
    window.t_per_render = 0;
    window.n_renders = 0;
    window.write_time = 0;
    md.initialization_done.then(() => {
        const f = async () => {
            let start = performance.now();
            for (let i = 0; i < window.n_steps_per_frame; i++) {
                await md.applyForce(ctx2);
            }
            let end = performance.now();
            window.t_per_render += end - start;
            window.n_renders += 1;
            if (!stop) {
                requestAnimationFrame(f);
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
        let w = window;
        console.log("t_per_render", w.t_per_render);
        console.log("n_renders", w.n_renders);
        console.log("avg", w.t_per_render / w.n_renders);
        console.log("t_per_write", w.write_time);
        console.log("  avg", w.write_time / w.n_renders);
        console.log("t_per_draw", w.t_per_draw);
        console.log("n_draws", w.n_draws);
        console.log("avg", w.t_per_draw / w.n_draws);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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QUFjLENBQUMsRUFBRSxHQUFHLEVBQUUsQ0FBQztJQUN2QixNQUFjLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztJQUMzQixNQUFjLENBQUMsTUFBTSxHQUFHLE1BQU0sQ0FBQztBQUNsQyxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXMiOlsid2VicGFjazovL3lvdXJQcm9qZWN0Ly4vc3JjL2ZvcmNlcy50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9pbml0LnRzIiwid2VicGFjazovL3lvdXJQcm9qZWN0Ly4vc3JjL21lc2hkZWZvcm1hdGlvbi50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC93ZWJwYWNrL2Jvb3RzdHJhcCIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9tYWluLnRzIl0sInNvdXJjZXNDb250ZW50IjpbImV4cG9ydCBmdW5jdGlvbiBidWlsZCh3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlciwgZ3JpZF94OiBudW1iZXIsIGdyaWRfc3BhY2luZzogbnVtYmVyLCBuX2VsZW1zOiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDEpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHlfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMilcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4gaW50ZW5zaXR5X21hcDogYXJyYXk8dTMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDMpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHhfcG9zX291dDogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDQpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHlfcG9zX291dDogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDUpXG52YXI8dW5pZm9ybT5vZmZzZXQ6IHUzMjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDYpXG52YXI8dW5pZm9ybT5zdHJpZGU6IHUzMjtcblxuXG5cbmZuIHBpeGVsVG9JbnRlbnNpdHkoX3B4OiB1MzIpIC0+IGYzMiB7XG4gIHZhciBweCA9IF9weDtcbiAgbGV0IHIgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGcgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGIgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGEgPSBmMzIocHggJSAyNTYpO1xuICBsZXQgaW50ZW5zaXR5OiBmMzIgPSAoYSAvIDI1NSkgKiAoMSAtICgwLjIxMjYgKiByICsgMC43MTUyICogZyArIDAuMDcyMiAqIGIpKTtcblxuICByZXR1cm4gaW50ZW5zaXR5O1xufVxuXG5AY29tcHV0ZSBAd29ya2dyb3VwX3NpemUoNjQsIDEsIDEpXG5mbiBtYWluKFxuICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZClcbmdsb2JhbF9pZCA6IHZlYzN1LFxuXG4gIEBidWlsdGluKGxvY2FsX2ludm9jYXRpb25faWQpXG5sb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIC8vIENvb3JkaW5hdGVzIG9mIHBhcnRpY2xlIGZvciB0aGlzIHRocmVhZFxuICBsZXQgc3RhcnQgPSBvZmZzZXQgKyAoc3RyaWRlICogZ2xvYmFsX2lkLngpO1xuICBmb3IgKHZhciBjID0gc3RhcnQ7IGMgPCAoc3RhcnQgKyBzdHJpZGUpOyBjKz11MzIoMSkpIHtcbiAgICBsZXQgaSA9IGM7XG4gICAgbGV0IGdyaWRfeCA9IGkgJSAke2dyaWRfeH07XG4gICAgbGV0IGdyaWRfeSA9IGkgLyAke2dyaWRfeH07XG4gICAgbGV0IG9yaWdpbl94ID0gZ3JpZF94ICogJHtncmlkX3NwYWNpbmd9O1xuICAgIGxldCBvcmlnaW5feSA9IGdyaWRfeSAqICR7Z3JpZF9zcGFjaW5nfTtcblxuICAgIGlmIChpID4gJHtuX2VsZW1zfSkge1xuICAgICAgY29udGludWU7XG4gICAgfVxuXG4gICAgbGV0IHggPSB4X3Bvc1tpXTtcbiAgICBsZXQgeSA9IHlfcG9zW2ldO1xuXG4gICAgdmFyIGRpcl94OiBmMzIgPSAwO1xuICAgIHZhciBkaXJfeTogZjMyID0gMDtcbiAgICB2YXIgY29lZmY6IGYzMiA9IDA7XG5cbiAgICBsZXQgcmVnaW9uID0gMjA7XG4gICAgZm9yICh2YXIgc195OiBpMzIgPSAwOyBzX3kgPD0gcmVnaW9uOyBzX3krKykge1xuICAgICAgZm9yICh2YXIgc194OiBpMzIgPSAwOyBzX3ggPD0gcmVnaW9uOyBzX3grKykge1xuICAgICAgICBsZXQgZHNfeSA9IHNfeSAtIHJlZ2lvbiAvIDI7XG4gICAgICAgIGxldCBkc194ID0gc194IC0gcmVnaW9uIC8gMjtcblxuICAgICAgICBsZXQgYmFzZV95ID0gaTMyKG9yaWdpbl95KTtcbiAgICAgICAgbGV0IGJhc2VfeCA9IGkzMihvcmlnaW5feCk7XG4gICAgICAgIGxldCBmX3kgPSBiYXNlX3kgKyBkc195O1xuICAgICAgICBsZXQgZl94ID0gYmFzZV94ICsgZHNfeDtcbiAgICAgICAgbGV0IGRfeDogZjMyID0gZjMyKGZfeCkgKyAwLjUgLSB4O1xuICAgICAgICBsZXQgZF95OiBmMzIgPSBmMzIoZl95KSArIDAuNSAtIHk7XG5cbiAgICAgICAgaWYgKGRzX3kgPT0gMCAmJiBkc194ID09IDApIHtcbiAgICAgICAgICBsZXQgbG9jYWxfY29lZmYgPSBmMzIoMjAwKTtcbiAgICAgICAgICBjb2VmZiArPSBsb2NhbF9jb2VmZjtcbiAgICAgICAgICBkaXJfeCArPSBsb2NhbF9jb2VmZiAqIGYzMihmX3gpO1xuICAgICAgICAgIGRpcl95ICs9IGxvY2FsX2NvZWZmICogZjMyKGZfeSk7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cblxuICAgICAgICBpZiAoZl95ID49IDAgJiYgZl95IDwgJHtoZWlnaHR9ICYmIGZfeCA+PSAwICYmIGZfeCA8ICR7d2lkdGh9KSB7XG4gICAgICAgICAgbGV0IGZfaSA9IGZfeSAqICR7d2lkdGh9ICsgZl94O1xuICAgICAgICAgIGxldCBpbnRlbnNpdHkgPSBwaXhlbFRvSW50ZW5zaXR5KGludGVuc2l0eV9tYXBbZl9pXSk7XG4gICAgICAgICAgbGV0IGxvY2FsX2NvZWZmID0gZjMyKDEwMCkgKiBpbnRlbnNpdHk7XG4gICAgICAgICAgY29lZmYgKz0gbG9jYWxfY29lZmY7XG4gICAgICAgICAgZGlyX3ggKz0gbG9jYWxfY29lZmYgKiBmMzIoZl94KTtcbiAgICAgICAgICBkaXJfeSArPSBsb2NhbF9jb2VmZiAqIGYzMihmX3kpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgbGV0IHRvdGFsX2NvZWZmID0gY29lZmY7XG4gICAgaWYgKHRvdGFsX2NvZWZmICE9IDApIHtcbiAgICAgIHZhciBkX3ggPSBkaXJfeCAvIHRvdGFsX2NvZWZmIC0geDtcbiAgICAgIHZhciBkX3kgPSBkaXJfeSAvIHRvdGFsX2NvZWZmIC0geTtcblxuICAgICAgbGV0IGRpc3QyID0gZF94ICogZF94ICsgZF95ICogZF95O1xuICAgICAgbGV0IG1heF9kaXN0MiA9IGYzMihyZWdpb24gKiByZWdpb24pO1xuXG4gICAgICBsZXQgc3BlZWQgPSBkaXN0MiAvIG1heF9kaXN0MjtcblxuICAgICAgZF94ICo9IHNwZWVkO1xuICAgICAgZF95ICo9IHNwZWVkO1xuXG4gICAgICB4X3Bvc19vdXRbaV0gPSB4ICsgZF94O1xuICAgICAgeV9wb3Nfb3V0W2ldID0geSArIGRfeTtcbiAgICB9IGVsc2Uge1xuICAgICAgeF9wb3Nfb3V0W2ldID0geDtcbiAgICAgIHlfcG9zX291dFtpXSA9IHk7XG4gICAgfVxuICB9XG59XG5gXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gYnVpbGQobl9lbGVtczogbnVtYmVyLCBncmlkX3g6IG51bWJlciwgZ3JpZF9zcGFjaW5nOiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGU+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDI1NilcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuICBnbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxuICBsb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIGlmIChnbG9iYWxfaWQueCA8ICR7bl9lbGVtc30pIHtcbiAgICAgIGxldCB5ID0gZ2xvYmFsX2lkLnggLyAke2dyaWRfeH07XG4gICAgICBsZXQgeCA9IGdsb2JhbF9pZC54ICUgJHtncmlkX3h9O1xuXG4gICAgICB4X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeCAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gICAgICB5X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeSAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gIH1cbn1cbmBcbn1cbiIsImltcG9ydCB7IGJ1aWxkIGFzIGluaXRidWlsZCB9IGZyb20gJy4vaW5pdCc7XG5pbXBvcnQgeyBidWlsZCBhcyBmb3JjZXNidWlsZCB9IGZyb20gJy4vZm9yY2VzJztcblxuY29uc3QgZWRnZXMgPSBbXG4gIFstMSwgMF0sXG4gIFsxLCAwXSxcbiAgWzAsIC0xXSxcbiAgWzAsIDFdLFxuICBbMSwgMV0sXG4gIFstMSwgLTFdLFxuXTtcblxuZXhwb3J0IGNsYXNzIE1lc2hEZWZvcm1hdGlvbiB7XG4gIGN0eDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuICBncmlkX3g6IG51bWJlcjtcbiAgZ3JpZF95OiBudW1iZXI7XG4gIGdyaWRfc3BhY2luZzogbnVtYmVyO1xuICBtaW5fZGlzdDogbnVtYmVyO1xuICBtYXhfZGlzdDogbnVtYmVyO1xuICByYWRpdXM6IG51bWJlcjtcblxuICByZWluaXQ6IGJvb2xlYW47XG4gIGRyYXdfZWRnZXM6IGJvb2xlYW47XG5cbiAgbl9lbGVtczogbnVtYmVyO1xuICB4X3BvczogRmxvYXQzMkFycmF5O1xuICB5X3BvczogRmxvYXQzMkFycmF5O1xuXG4gIGluaXRfbW9kdWxlOiBHUFVTaGFkZXJNb2R1bGU7XG4gIGluaXRCaW5kR3JvdXBMYXlvdXQ6IEdQVUJpbmRHcm91cExheW91dDtcbiAgaW5pdF9jb21wdXRlUGlwZWxpbmU6IEdQVUNvbXB1dGVQaXBlbGluZTtcblxuICBpbml0aWFsaXphdGlvbl9kb25lOiBQcm9taXNlPHZvaWQ+O1xuICBkZXZpY2U6IEdQVURldmljZTtcbiAgZm9yY2VfbW9kdWxlOiBHUFVTaGFkZXJNb2R1bGU7XG4gIGFjdGl2ZV9idWZmZXJfaWQ6IG51bWJlcjtcbiAgeF9wb3NfYnVmZmVyczogW0dQVUJ1ZmZlciwgR1BVQnVmZmVyXTtcbiAgeV9wb3NfYnVmZmVyczogW0dQVUJ1ZmZlciwgR1BVQnVmZmVyXTtcbiAgLy8gQnVmZmVycyB0byByZWFkIHZhbHVlcyBiYWNrIHRvIHRoZSBDUFUgZm9yIGRyYXdpbmdcbiAgc3RhZ2luZ194X2J1ZjogR1BVQnVmZmVyO1xuICBzdGFnaW5nX3lfYnVmOiBHUFVCdWZmZXI7XG4gIC8vIEJ1ZmZlciB0byB3cml0ZSB2YWx1ZSB0byBmcm9tIHRoZSBDUFUgZm9yIGFkanVzdGluZyB3ZWlnaHRzXG4gIHN0YWdpbmdfaW50ZW5zaXR5X2J1ZjogR1BVQnVmZmVyO1xuICBpbnRlbnNpdHlfbWFwX2J1ZjogR1BVQnVmZmVyO1xuXG4gIG9mZnNldF9idWY6IEdQVUJ1ZmZlcjtcbiAgc3RyaWRlX2J1ZjogR1BVQnVmZmVyO1xuXG4gIGZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0OiBHUFVCaW5kR3JvdXBMYXlvdXQ7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQsXG4gICAgZ3JpZF94OiBudW1iZXIsXG4gICAgZ3JpZF95OiBudW1iZXIsXG4gICAgZ3JpZF9zcGFjaW5nOiBudW1iZXIsXG4gICAgbWluX2Rpc3Q6IG51bWJlcixcbiAgICBtYXhfZGlzdDogbnVtYmVyLFxuICAgIHJhZGl1czogbnVtYmVyXG4gICkge1xuICAgIHRoaXMuY3R4ID0gY3R4O1xuICAgIHRoaXMuZ3JpZF94ID0gZ3JpZF94O1xuICAgIHRoaXMuZ3JpZF95ID0gZ3JpZF95O1xuICAgIHRoaXMuZ3JpZF9zcGFjaW5nID0gZ3JpZF9zcGFjaW5nO1xuICAgIHRoaXMubWluX2Rpc3QgPSBtaW5fZGlzdDtcbiAgICB0aGlzLm1heF9kaXN0ID0gbWF4X2Rpc3Q7XG4gICAgdGhpcy5yYWRpdXMgPSByYWRpdXM7XG5cbiAgICB0aGlzLmRyYXdfZWRnZXMgPSB0cnVlO1xuICAgIHRoaXMucmVpbml0ID0gZmFsc2U7XG5cbiAgICB0aGlzLm5fZWxlbXMgPSB0aGlzLmdyaWRfeCAqIHRoaXMuZ3JpZF95O1xuXG4gICAgdGhpcy5pbml0aWFsaXphdGlvbl9kb25lID0gdGhpcy5hc3luY19pbml0KCk7XG4gIH1cblxuICBhc3luYyBhc3luY19pbml0KCkge1xuICAgIGNvbnN0IGFkYXB0ZXIgPSBhd2FpdCBuYXZpZ2F0b3IuZ3B1LnJlcXVlc3RBZGFwdGVyKCk7XG4gICAgdGhpcy5kZXZpY2UgPSBhd2FpdCBhZGFwdGVyLnJlcXVlc3REZXZpY2UoKTtcbiAgICBjb25zb2xlLmxvZyhcIkNyZWF0ZSBjb21wdXRlIHNoYWRlclwiKTtcbiAgICB0aGlzLmZvcmNlX21vZHVsZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZVNoYWRlck1vZHVsZSh7XG4gICAgICBjb2RlOiBmb3JjZXNidWlsZCh0aGlzLmN0eC5jYW52YXMud2lkdGgsIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQsIHRoaXMuZ3JpZF94LCB0aGlzLmdyaWRfc3BhY2luZywgdGhpcy5uX2VsZW1zKSxcbiAgICB9KTtcbiAgICBjb25zb2xlLmxvZyhcImRvbmUgQ3JlYXRlIGNvbXB1dGUgc2hhZGVyXCIpO1xuXG4gICAgdGhpcy5hY3RpdmVfYnVmZmVyX2lkID0gMDtcbiAgICB0aGlzLnhfcG9zX2J1ZmZlcnMgPSBbXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ4X3Bvc1swXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInhfcG9zWzFdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgIF07XG4gICAgdGhpcy55X3Bvc19idWZmZXJzID0gW1xuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieV9wb3NbMF1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ5X3Bvc1sxXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICBdO1xuXG4gICAgdGhpcy5zdGFnaW5nX3hfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0YWdpbmdfeF9idWZcIixcbiAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1JFQUQgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcbiAgICB0aGlzLnN0YWdpbmdfeV9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwic3RhZ2luZ195X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5NQVBfUkVBRCB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuXG4gICAgdGhpcy5zdGFnaW5nX2ludGVuc2l0eV9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwic3RhZ2luZ19pbnRlbnNpdHlfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLmN0eC5jYW52YXMud2lkdGggKiB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0ICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5NQVBfV1JJVEUgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQyxcbiAgICB9KTtcbiAgICB0aGlzLmludGVuc2l0eV9tYXBfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcImludGVuc2l0eV9idWZcIixcbiAgICAgIHNpemU6IHRoaXMuY3R4LmNhbnZhcy53aWR0aCAqIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcblxuICAgIHRoaXMub2Zmc2V0X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJvZmZzZXRfYnVmXCIsXG4gICAgICBzaXplOiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlVOSUZPUk0gfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcbiAgICB0aGlzLnN0cmlkZV9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwic3RyaWRlX2J1ZlwiLFxuICAgICAgc2l6ZTogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5VTklGT1JNIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFsbG9jYXRlIGJ1ZmZlcnNcIik7XG5cbiAgICB0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMixcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDMsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA0LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJ1bmlmb3JtXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDYsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwidW5pZm9ybVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgLy8gaW50aWFsaXplIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdIGFuZFxuICAgIC8vIHRoaXMueV9wb3NfYnVmZmVyc1sqXSB0byBiZSBhIGdyaWRcbiAgICBjb25zdCBpbml0X3NoYWRlciA9IGluaXRidWlsZCh0aGlzLm5fZWxlbXMsIHRoaXMuZ3JpZF94LCB0aGlzLmdyaWRfc3BhY2luZyk7XG4gICAgdGhpcy5pbml0QmluZEdyb3VwTGF5b3V0ID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwTGF5b3V0KHtcbiAgICAgIGVudHJpZXM6IFtcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIGNvbnNvbGUubG9nKFwiQ3JlYXRlIGluaXQgc2hhZGVyXCIpO1xuICAgIHRoaXMuaW5pdF9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogaW5pdF9zaGFkZXIsXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIENyZWF0ZSBpbml0IHNoYWRlclwiKTtcbiAgICB0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tcHV0ZVBpcGVsaW5lKHtcbiAgICAgIGxhYmVsOiBcImNvbXB1dGUgZm9yY2VcIixcbiAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbdGhpcy5pbml0QmluZEdyb3VwTGF5b3V0XSxcbiAgICAgIH0pLFxuICAgICAgY29tcHV0ZToge1xuICAgICAgICBtb2R1bGU6IHRoaXMuaW5pdF9tb2R1bGUsXG4gICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgfSxcbiAgICB9KTtcbiAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG5cbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiB0aGlzLmluaXRCaW5kR3JvdXBMYXlvdXQsXG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgIHJlc291cmNlOiB7XG4gICAgICAgICAgICBidWZmZXI6IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgIH0sXG4gICAgICAgIH1cbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zdCBwYXNzRW5jb2RlciA9IGNvbW1hbmRFbmNvZGVyLmJlZ2luQ29tcHV0ZVBhc3MoKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZSh0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoTWF0aC5jZWlsKHRoaXMubl9lbGVtcyAvIDI1NikpO1xuICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemVcbiAgICApO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemVcbiAgICApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcblxuICAgIGF3YWl0IHRoaXMudXBkYXRlQ1BVcG9zKCk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFzeW5jIGluaXRcIik7XG4gIH1cblxuICBhc3luYyB1cGRhdGVDUFVwb3MoKSB7XG4gICAgLy8gY29uc29sZS5sb2coXCJNYXAgYnVmZmVycyBmb3IgcmVhZGluZ1wiKTtcbiAgICBsZXQgbV94ID0gdGhpcy5zdGFnaW5nX3hfYnVmLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIGxldCBtX3kgPSB0aGlzLnN0YWdpbmdfeV9idWYubWFwQXN5bmMoR1BVTWFwTW9kZS5SRUFELCAwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgYXdhaXQgbV94O1xuICAgIGF3YWl0IG1feTtcblxuICAgIC8vIGNvbnNvbGUubG9nKFwiY29weWluZyB4IGJ1ZmZlciB0byBDUFVcIik7XG4gICAgY29uc3QgY29weUFycmF5QnVmZmVyWCA9IHRoaXMuc3RhZ2luZ194X2J1Zi5nZXRNYXBwZWRSYW5nZSgwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgY29uc3QgZGF0YVggPSBjb3B5QXJyYXlCdWZmZXJYLnNsaWNlKCk7XG4gICAgdGhpcy54X3BvcyA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YVgpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJjb3B5aW5nIHkgYnVmZmVyIHRvIENQVVwiKTtcbiAgICBjb25zdCBjb3B5QXJyYXlCdWZmZXJZID0gdGhpcy5zdGFnaW5nX3lfYnVmLmdldE1hcHBlZFJhbmdlKDAsIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplKTtcbiAgICBjb25zdCBkYXRhWSA9IGNvcHlBcnJheUJ1ZmZlclkuc2xpY2UoKTtcbiAgICB0aGlzLnlfcG9zID0gbmV3IEZsb2F0MzJBcnJheShkYXRhWSk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcInVubWFwIGJ1ZmZlcnNcIik7XG4gICAgdGhpcy5zdGFnaW5nX3hfYnVmLnVubWFwKCk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmLnVubWFwKCk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcIkRvbmUgdXBkYXRlQ1BVcG9zXCIpO1xuICB9XG5cbiAgYXN5bmMgYXBwbHlGb3JjZShjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCkge1xuICAgIGxldCBpZGF0YSA9IGN0eC5nZXRJbWFnZURhdGEoMCwgMCwgY3R4LmNhbnZhcy53aWR0aCwgY3R4LmNhbnZhcy5oZWlnaHQpLmRhdGE7XG4gICAgLy8gY29uc29sZS5sb2coYGIwICR7aWRhdGFbMF19LCAke2lkYXRhWzFdfSwgJHtpZGF0YVsyXX0sICR7aWRhdGFbM119YCk7XG4gICAgLy8gY29uc29sZS5sb2coYFdyaXRpbmcgJHt0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemV9LyR7aWRhdGEubGVuZ3RofSBieXRlcyBmb3IgaW1hcGApO1xuICAgIGxldCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKFxuICAgICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiwgMCwgaWRhdGEuYnVmZmVyLCAwLCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemUpO1xuICAgICh3aW5kb3cgYXMgYW55KS53cml0ZV90aW1lICs9IHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQ7XG5cbiAgICBsZXQgaW5wdXRfeCA9IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBpbnB1dF95ID0gdGhpcy55X3Bvc19idWZmZXJzW3RoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG4gICAgbGV0IG91dHB1dF94ID0gdGhpcy54X3Bvc19idWZmZXJzWzEgLSB0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBvdXRwdXRfeSA9IHRoaXMueV9wb3NfYnVmZmVyc1sxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcblxuXG4gICAgaWYgKHRoaXMucmVpbml0KSB7XG4gICAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgICBsYXlvdXQ6IHRoaXMuaW5pdEJpbmRHcm91cExheW91dCxcbiAgICAgICAgZW50cmllczogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIGJpbmRpbmc6IDAsXG4gICAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgICBidWZmZXI6IHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9LFxuICAgICAgICAgIHtcbiAgICAgICAgICAgIGJpbmRpbmc6IDEsXG4gICAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgICBidWZmZXI6IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLFxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9XG4gICAgICAgIF0sXG4gICAgICB9KTtcblxuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZSh0aGlzLmluaXRfY29tcHV0ZVBpcGVsaW5lKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKE1hdGguY2VpbCh0aGlzLm5fZWxlbXMgLyAyNTYpKTtcbiAgICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgICAgdGhpcy5yZWluaXQgPSBmYWxzZTtcbiAgICB9XG5cblxuICAgIGNvbnN0IHdnX3ggPSA2NDtcbiAgICBjb25zdCBzdHJpZGUgPSA4O1xuICAgIGNvbnN0IGRpc3BhdGNoX3ggPSAoMjU2IC8gd2dfeCk7XG4gICAgbGV0IGJ1ZmZlcnMgPSBbaW5wdXRfeCwgaW5wdXRfeSwgdGhpcy5pbnRlbnNpdHlfbWFwX2J1Ziwgb3V0cHV0X3gsIG91dHB1dF95LCB0aGlzLm9mZnNldF9idWYsIHRoaXMuc3RyaWRlX2J1Zl07XG4gICAgY29uc3QgYmluZEdyb3VwID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcbiAgICAgIGxheW91dDogdGhpcy5mb3JjZV9iaW5kX2dyb3VwX2xheW91dCxcbiAgICAgIGVudHJpZXM6IGJ1ZmZlcnMubWFwKChiLCBpKSA9PiB7IHJldHVybiB7IGJpbmRpbmc6IGksIHJlc291cmNlOiB7IGJ1ZmZlcjogYiB9IH07IH0pXG4gICAgfSk7XG5cbiAgICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCB0aGlzLm5fZWxlbXM7IG9mZnNldCArPSAoc3RyaWRlICogd2dfeCAqIGRpc3BhdGNoX3gpKSB7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5vZmZzZXRfYnVmLCAwLCBuZXcgVWludDMyQXJyYXkoW29mZnNldF0pLmJ1ZmZlciwgMCwgNCk7XG4gICAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgICAgdGhpcy5zdHJpZGVfYnVmLCAwLCBuZXcgVWludDMyQXJyYXkoW3N0cmlkZV0pLmJ1ZmZlciwgMCwgNCk7XG5cbiAgICAgIGNvbnN0IGNvbXB1dGVQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbXB1dGVQaXBlbGluZSh7XG4gICAgICAgIGxhYmVsOiBcImZvcmNlcGlwZWxpbmVcIixcbiAgICAgICAgbGF5b3V0OiB0aGlzLmRldmljZS5jcmVhdGVQaXBlbGluZUxheW91dCh7XG4gICAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3RoaXMuZm9yY2VfYmluZF9ncm91cF9sYXlvdXRdLFxuICAgICAgICB9KSxcbiAgICAgICAgY29tcHV0ZToge1xuICAgICAgICAgIG1vZHVsZTogdGhpcy5mb3JjZV9tb2R1bGUsXG4gICAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICAgIH0sXG4gICAgICB9KTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiY3JlYXRlZCBwaXBlbGluZVwiKTtcblxuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZShjb21wdXRlUGlwZWxpbmUpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoZGlzcGF0Y2hfeCwgMSwgMSk7XG4gICAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb21wdXRlXCIpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgICAgLy8gY29uc29sZS5sb2coXCJxdWV1ZVwiLCBvZmZzZXQpO1xuICAgIH1cblxuICAgIGNvbnN0IGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAvLyBDb3B5IG91dHB1dCBidWZmZXIgdG8gc3RhZ2luZyBidWZmZXJcbiAgICBjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICBvdXRwdXRfeCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJ4IGNvcHlpbmdcIiwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUsIFwiYnl0ZXNcIik7XG4gICAgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgb3V0cHV0X3ksIDAsIHRoaXMuc3RhZ2luZ195X2J1ZiwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemUpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwieSBjb3B5aW5nXCIsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplLCBcImJ5dGVzXCIpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb3B5IHRvIGJ1ZmZlcnNcIiwgdGhpcy5hY3RpdmVfYnVmZmVyX2lkKTtcblxuICAgIC8vIEVuZCBmcmFtZSBieSBwYXNzaW5nIGFycmF5IG9mIGNvbW1hbmQgYnVmZmVycyB0byBjb21tYW5kIHF1ZXVlIGZvciBleGVjdXRpb25cbiAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJkb25lIHN1Ym1pdCB0byBxdWV1ZVwiKTtcblxuICAgIC8vIFN3YXAgaW5wdXQgYW5kIG91dHB1dDpcbiAgICB0aGlzLmFjdGl2ZV9idWZmZXJfaWQgPSAxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkO1xuXG4gICAgLy8gYXdhaXQgdGhpcy51cGRhdGVDUFVwb3MoKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImRvbmUgYXBwbHlGb3JjZVwiKTtcbiAgfVxuXG4gIGRyYXcoKSB7XG4gICAgdGhpcy5jdHguY2xlYXJSZWN0KDAsIDAsIHRoaXMuY3R4LmNhbnZhcy53aWR0aCwgdGhpcy5jdHguY2FudmFzLmhlaWdodCk7XG4gICAgZm9yIChsZXQgeWlkeCA9IDA7IHlpZHggPCB0aGlzLmdyaWRfeTsgeWlkeCsrKSB7XG4gICAgICBmb3IgKGxldCB4aWR4ID0gMDsgeGlkeCA8IHRoaXMuZ3JpZF94OyB4aWR4KyspIHtcbiAgICAgICAgbGV0IGkgPSB5aWR4ICogdGhpcy5ncmlkX3ggKyB4aWR4O1xuXG4gICAgICAgIGxldCB4ID0gdGhpcy54X3Bvc1tpXTtcbiAgICAgICAgbGV0IHkgPSB0aGlzLnlfcG9zW2ldO1xuXG4gICAgICAgIHRoaXMuY3R4LnN0cm9rZVN0eWxlID0gXCIjZmYwMDAwNWZcIjtcbiAgICAgICAgdGhpcy5jdHguYmVnaW5QYXRoKCk7XG4gICAgICAgIHRoaXMuY3R4LmFyYyh4LCB5LCB0aGlzLnJhZGl1cywgMCwgMiAqIE1hdGguUEkpO1xuICAgICAgICB0aGlzLmN0eC5zdHJva2UoKTtcblxuICAgICAgICBpZiAodGhpcy5kcmF3X2VkZ2VzKSB7XG4gICAgICAgICAgZm9yIChsZXQgZWRnZSBvZiBlZGdlcykge1xuICAgICAgICAgICAgbGV0IGpfeGlkeCA9IHhpZHggKyBlZGdlWzBdO1xuICAgICAgICAgICAgbGV0IGpfeWlkeCA9IHlpZHggKyBlZGdlWzFdO1xuICAgICAgICAgICAgaWYgKGpfeGlkeCA8IDAgfHwgal94aWR4ID49IHRoaXMuZ3JpZF94IHx8IGpfeWlkeCA8IDAgfHwgal95aWR4ID49IHRoaXMuZ3JpZF95KSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBsZXQgaiA9IGpfeWlkeCAqIHRoaXMuZ3JpZF94ICsgal94aWR4O1xuXG4gICAgICAgICAgICBsZXQgal94ID0gdGhpcy54X3Bvc1tqXTtcbiAgICAgICAgICAgIGxldCBqX3kgPSB0aGlzLnlfcG9zW2pdO1xuXG4gICAgICAgICAgICB0aGlzLmN0eC5iZWdpblBhdGgoKTtcbiAgICAgICAgICAgIHRoaXMuY3R4Lm1vdmVUbyh4LCB5KTtcbiAgICAgICAgICAgIHRoaXMuY3R4LmxpbmVUbyhqX3gsIGpfeSk7XG4gICAgICAgICAgICB0aGlzLmN0eC5zdHJva2UoKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cbiIsIi8vIFRoZSBtb2R1bGUgY2FjaGVcbnZhciBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX18gPSB7fTtcblxuLy8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbmZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblx0Ly8gQ2hlY2sgaWYgbW9kdWxlIGlzIGluIGNhY2hlXG5cdHZhciBjYWNoZWRNb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdO1xuXHRpZiAoY2FjaGVkTW9kdWxlICE9PSB1bmRlZmluZWQpIHtcblx0XHRyZXR1cm4gY2FjaGVkTW9kdWxlLmV4cG9ydHM7XG5cdH1cblx0Ly8gQ3JlYXRlIGEgbmV3IG1vZHVsZSAoYW5kIHB1dCBpdCBpbnRvIHRoZSBjYWNoZSlcblx0dmFyIG1vZHVsZSA9IF9fd2VicGFja19tb2R1bGVfY2FjaGVfX1ttb2R1bGVJZF0gPSB7XG5cdFx0Ly8gbm8gbW9kdWxlLmlkIG5lZWRlZFxuXHRcdC8vIG5vIG1vZHVsZS5sb2FkZWQgbmVlZGVkXG5cdFx0ZXhwb3J0czoge31cblx0fTtcblxuXHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cblx0X193ZWJwYWNrX21vZHVsZXNfX1ttb2R1bGVJZF0obW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cblx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcblx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xufVxuXG4iLCJpbXBvcnQgeyBNZXNoRGVmb3JtYXRpb24gfSBmcm9tICcuL21lc2hkZWZvcm1hdGlvbic7XG5cbmxldCBzdG9wID0gZmFsc2U7XG5cbmRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoXCJET01Db250ZW50TG9hZGVkXCIsICgpID0+IHtcbiAgbGV0IGNhbnZhcyA9IChkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcIm15Y2FudmFzXCIpIGFzIEhUTUxDYW52YXNFbGVtZW50KTtcbiAgY2FudmFzLndpZHRoID0gMTAwMDtcbiAgY2FudmFzLmhlaWdodCA9IDEwMDA7XG5cbiAgbGV0IGN0eCA9IGNhbnZhcy5nZXRDb250ZXh0KFwiMmRcIik7XG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBtYWluIGNhbnZhc1wiKTtcblxuICBsZXQgY2FudmFzMiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJjYW52YXNcIikgYXMgSFRNTENhbnZhc0VsZW1lbnQ7XG4gIGNhbnZhczIud2lkdGggPSAxMDAwO1xuICBjYW52YXMyLmhlaWdodCA9IDEwMDA7XG4gIGxldCBjdHgyID0gY2FudmFzMi5nZXRDb250ZXh0KFwiMmRcIik7XG5cbiAgLy8gbGV0IGlkYXRhOiBVaW50OENsYW1wZWRBcnJheTxBcnJheUJ1ZmZlckxpa2U+ID0gbmV3IFVpbnQ4Q2xhbXBlZEFycmF5KGN0eDIuY2FudmFzLndpZHRoICogY3R4LmNhbnZhcy5oZWlnaHQgKiA0KTtcbiAgY2FudmFzLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoZSkgPT4ge1xuICAgIGxldCBlbCA9IGUudGFyZ2V0IGFzIEhUTUxDYW52YXNFbGVtZW50O1xuICAgIGNvbnN0IHJlY3QgPSBlbC5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcbiAgICBjb25zdCB4ID0gZWwud2lkdGggKiAoZS5jbGllbnRYIC0gcmVjdC5sZWZ0KSAvIHJlY3Qud2lkdGg7XG4gICAgY29uc3QgeSA9IGVsLmhlaWdodCAqIChlLmNsaWVudFkgLSByZWN0LnRvcCkgLyByZWN0LmhlaWdodDtcblxuICAgIGN0eDIuYmVnaW5QYXRoKCk7XG4gICAgY3R4Mi5maWxsU3R5bGUgPSBcImJsYWNrXCI7XG4gICAgY3R4Mi5hcmMoeCwgeSwgMTAwLCAwLCAyICogTWF0aC5QSSk7XG4gICAgY3R4Mi5maWxsKCk7XG5cbiAgICAvLyBpZGF0YSA9IGN0eDIuZ2V0SW1hZ2VEYXRhKDAsIDAsIGN0eC5jYW52YXMud2lkdGgsIGN0eC5jYW52YXMuaGVpZ2h0KS5kYXRhO1xuICB9KTtcblxuICBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcImNsZWFyXCIpLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoKSA9PiB7XG4gICAgY3R4Mi5jbGVhclJlY3QoMCwgMCwgY3R4Mi5jYW52YXMud2lkdGgsIGN0eDIuY2FudmFzLmhlaWdodCk7XG4gIH0pO1xuXG4gIGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiZWRnZXNcIikuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICBtZC5kcmF3X2VkZ2VzID0gIW1kLmRyYXdfZWRnZXM7XG4gIH0pO1xuXG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBpbnRlcmFjdGl2ZSBjYW52YXNcIik7XG5cbiAgKHdpbmRvdyBhcyBhbnkpLm5fc3RlcHNfcGVyX2ZyYW1lID0gMTtcblxuICBsZXQgbl9lbGVtcyA9IDEwMDtcbiAgbGV0IHNwYWNpbmcgPSBjdHguY2FudmFzLndpZHRoIC8gbl9lbGVtcztcbiAgbGV0IG1kID0gbmV3IE1lc2hEZWZvcm1hdGlvbihjdHgsIG5fZWxlbXMsIG5fZWxlbXMsIHNwYWNpbmcsIHNwYWNpbmcgLyA0LCBzcGFjaW5nICogNCwgMSk7XG4gICh3aW5kb3cgYXMgYW55KS50X3Blcl9yZW5kZXIgPSAwO1xuICAod2luZG93IGFzIGFueSkubl9yZW5kZXJzID0gMDtcbiAgKHdpbmRvdyBhcyBhbnkpLndyaXRlX3RpbWUgPSAwO1xuICBtZC5pbml0aWFsaXphdGlvbl9kb25lLnRoZW4oKCkgPT4ge1xuICAgIGNvbnN0IGYgPSBhc3luYyAoKSA9PiB7XG4gICAgICBsZXQgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgKHdpbmRvdyBhcyBhbnkpLm5fc3RlcHNfcGVyX2ZyYW1lOyBpKyspIHtcbiAgICAgICAgYXdhaXQgbWQuYXBwbHlGb3JjZShjdHgyKTtcbiAgICAgIH1cbiAgICAgIGxldCBlbmQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgICh3aW5kb3cgYXMgYW55KS50X3Blcl9yZW5kZXIgKz0gZW5kIC0gc3RhcnQ7XG4gICAgICAod2luZG93IGFzIGFueSkubl9yZW5kZXJzICs9IDE7XG4gICAgICBpZiAoIXN0b3ApIHtcbiAgICAgICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpXG4gICAgICB9XG4gICAgfTtcbiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZik7XG5cbiAgICAod2luZG93IGFzIGFueSkudF9wZXJfZHJhdyA9IDA7XG4gICAgKHdpbmRvdyBhcyBhbnkpLm5fZHJhd3MgPSAwO1xuICAgIGNvbnN0IGcgPSBhc3luYyAoKSA9PiB7XG4gICAgICBsZXQgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICAgIGF3YWl0IG1kLnVwZGF0ZUNQVXBvcygpO1xuICAgICAgbWQuZHJhdygpO1xuICAgICAgbGV0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX2RyYXcgKz0gZW5kIC0gc3RhcnQ7XG4gICAgICAod2luZG93IGFzIGFueSkubl9kcmF3cyArPSAxO1xuICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShnKVxuICAgICAgfSwgMzApO1xuICAgIH07XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGcpO1xuICB9KTtcblxuICAod2luZG93IGFzIGFueSkuc3RhdHMgPSAoKSA9PiB7XG4gICAgbGV0IHcgPSB3aW5kb3cgYXMgYW55O1xuICAgIGNvbnNvbGUubG9nKFwidF9wZXJfcmVuZGVyXCIsIHcudF9wZXJfcmVuZGVyKTtcbiAgICBjb25zb2xlLmxvZyhcIm5fcmVuZGVyc1wiLCB3Lm5fcmVuZGVycyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdcIiwgdy50X3Blcl9yZW5kZXIgLyB3Lm5fcmVuZGVycyk7XG4gICAgY29uc29sZS5sb2coXCJ0X3Blcl93cml0ZVwiLCB3LndyaXRlX3RpbWUpO1xuICAgIGNvbnNvbGUubG9nKFwiICBhdmdcIiwgdy53cml0ZV90aW1lIC8gdy5uX3JlbmRlcnMpO1xuICAgIGNvbnNvbGUubG9nKFwidF9wZXJfZHJhd1wiLCB3LnRfcGVyX2RyYXcpO1xuICAgIGNvbnNvbGUubG9nKFwibl9kcmF3c1wiLCB3Lm5fZHJhd3MpO1xuICAgIGNvbnNvbGUubG9nKFwiYXZnXCIsIHcudF9wZXJfZHJhdyAvIHcubl9kcmF3cyk7XG4gIH1cblxuXG4gIGZ1bmN0aW9uIGNhbmNlbCgpIHtcbiAgICBzdG9wID0gdHJ1ZTtcbiAgfVxuICAod2luZG93IGFzIGFueSkubWQgPSBtZDtcbiAgKHdpbmRvdyBhcyBhbnkpLmN0eDIgPSBjdHgyO1xuICAod2luZG93IGFzIGFueSkuY2FuY2VsID0gY2FuY2VsO1xufSk7XG5cbiJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==