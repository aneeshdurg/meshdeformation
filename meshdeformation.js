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
            if (video.readyState != 4) {
                return;
            }
            ctx2.drawImage(video, 0, 0, ctx2.canvas.width, ctx2.canvas.height);
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
}
document.addEventListener("DOMContentLoaded", () => {
    let container = document.getElementById("container");
    main(container);
});

})();

/******/ })()
;
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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Q0FBQyxDQUFDO1FBQ2pELE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxFQUFFLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUN4QyxPQUFPLENBQUMsR0FBRyxDQUFDLFNBQVMsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDbEMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLFVBQVUsR0FBRyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQUdELFNBQVMsTUFBTTtRQUNiLElBQUksR0FBRyxJQUFJLENBQUM7SUFDZCxDQUFDO0lBQ0EsTUFBYyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7SUFDdkIsTUFBYyxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7SUFDM0IsTUFBYyxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7QUFDbEMsQ0FBQztBQUVELFFBQVEsQ0FBQyxnQkFBZ0IsQ0FBQyxrQkFBa0IsRUFBRSxHQUFHLEVBQUU7SUFDakQsSUFBSSxTQUFTLEdBQUcsUUFBUSxDQUFDLGNBQWMsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUNyRCxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7QUFDbEIsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9mb3JjZXMudHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvaW5pdC50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9tZXNoZGVmb3JtYXRpb24udHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3Qvd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvbWFpbi50cyJdLCJzb3VyY2VzQ29udGVudCI6WyJleHBvcnQgZnVuY3Rpb24gYnVpbGQod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGdyaWRfeDogbnVtYmVyLCBncmlkX3NwYWNpbmc6IG51bWJlciwgbl9lbGVtczogbnVtYmVyKSB7XG4gIHJldHVybiBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4geF9wb3M6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygxKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDIpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IGludGVuc2l0eV9tYXA6IGFycmF5PHUzMj47XG5cbkBncm91cCgwKSBAYmluZGluZygzKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZyg0KVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB5X3Bvc19vdXQ6IGFycmF5PGYzMj47XG5cbkBncm91cCgwKSBAYmluZGluZyg1KVxudmFyPHVuaWZvcm0+b2Zmc2V0OiB1MzI7XG5cbkBncm91cCgwKSBAYmluZGluZyg2KVxudmFyPHVuaWZvcm0+c3RyaWRlOiB1MzI7XG5cblxuXG5mbiBwaXhlbFRvSW50ZW5zaXR5KF9weDogdTMyKSAtPiBmMzIge1xuICB2YXIgcHggPSBfcHg7XG4gIGxldCByID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBnID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBiID0gZjMyKHB4ICUgMjU2KTtcbiAgcHggLz0gdTMyKDI1Nik7XG4gIGxldCBhID0gZjMyKHB4ICUgMjU2KTtcbiAgbGV0IGludGVuc2l0eTogZjMyID0gKGEgLyAyNTUpICogKDEgLSAoMC4yMTI2ICogciArIDAuNzE1MiAqIGcgKyAwLjA3MjIgKiBiKSk7XG5cbiAgcmV0dXJuIGludGVuc2l0eTtcbn1cblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDY0LCAxLCAxKVxuZm4gbWFpbihcbiAgQGJ1aWx0aW4oZ2xvYmFsX2ludm9jYXRpb25faWQpXG5nbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxubG9jYWxfaWQgOiB2ZWMzdSxcbikge1xuICAvLyBDb29yZGluYXRlcyBvZiBwYXJ0aWNsZSBmb3IgdGhpcyB0aHJlYWRcbiAgbGV0IHN0YXJ0ID0gb2Zmc2V0ICsgKHN0cmlkZSAqIGdsb2JhbF9pZC54KTtcbiAgZm9yICh2YXIgYyA9IHN0YXJ0OyBjIDwgKHN0YXJ0ICsgc3RyaWRlKTsgYys9dTMyKDEpKSB7XG4gICAgbGV0IGkgPSBjO1xuICAgIGxldCBncmlkX3ggPSBpICUgJHtncmlkX3h9O1xuICAgIGxldCBncmlkX3kgPSBpIC8gJHtncmlkX3h9O1xuICAgIGxldCBvcmlnaW5feCA9IGdyaWRfeCAqICR7Z3JpZF9zcGFjaW5nfTtcbiAgICBsZXQgb3JpZ2luX3kgPSBncmlkX3kgKiAke2dyaWRfc3BhY2luZ307XG5cbiAgICBpZiAoaSA+ICR7bl9lbGVtc30pIHtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cblxuICAgIGxldCB4ID0geF9wb3NbaV07XG4gICAgbGV0IHkgPSB5X3Bvc1tpXTtcblxuICAgIHZhciBkaXJfeDogZjMyID0gMDtcbiAgICB2YXIgZGlyX3k6IGYzMiA9IDA7XG4gICAgdmFyIGNvZWZmOiBmMzIgPSAwO1xuXG4gICAgbGV0IHJlZ2lvbiA9IDIwO1xuICAgIGZvciAodmFyIHNfeTogaTMyID0gMDsgc195IDw9IHJlZ2lvbjsgc195KyspIHtcbiAgICAgIGZvciAodmFyIHNfeDogaTMyID0gMDsgc194IDw9IHJlZ2lvbjsgc194KyspIHtcbiAgICAgICAgbGV0IGRzX3kgPSBzX3kgLSByZWdpb24gLyAyO1xuICAgICAgICBsZXQgZHNfeCA9IHNfeCAtIHJlZ2lvbiAvIDI7XG5cbiAgICAgICAgbGV0IGJhc2VfeSA9IGkzMihvcmlnaW5feSk7XG4gICAgICAgIGxldCBiYXNlX3ggPSBpMzIob3JpZ2luX3gpO1xuICAgICAgICBsZXQgZl95ID0gYmFzZV95ICsgZHNfeTtcbiAgICAgICAgbGV0IGZfeCA9IGJhc2VfeCArIGRzX3g7XG4gICAgICAgIGxldCBkX3g6IGYzMiA9IGYzMihmX3gpICsgMC41IC0geDtcbiAgICAgICAgbGV0IGRfeTogZjMyID0gZjMyKGZfeSkgKyAwLjUgLSB5O1xuXG4gICAgICAgIGlmIChkc195ID09IDAgJiYgZHNfeCA9PSAwKSB7XG4gICAgICAgICAgbGV0IGxvY2FsX2NvZWZmID0gZjMyKDIwMCk7XG4gICAgICAgICAgY29lZmYgKz0gbG9jYWxfY29lZmY7XG4gICAgICAgICAgZGlyX3ggKz0gbG9jYWxfY29lZmYgKiBmMzIoZl94KTtcbiAgICAgICAgICBkaXJfeSArPSBsb2NhbF9jb2VmZiAqIGYzMihmX3kpO1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG5cbiAgICAgICAgaWYgKGZfeSA+PSAwICYmIGZfeSA8ICR7aGVpZ2h0fSAmJiBmX3ggPj0gMCAmJiBmX3ggPCAke3dpZHRofSkge1xuICAgICAgICAgIGxldCBmX2kgPSBmX3kgKiAke3dpZHRofSArIGZfeDtcbiAgICAgICAgICBsZXQgaW50ZW5zaXR5ID0gcGl4ZWxUb0ludGVuc2l0eShpbnRlbnNpdHlfbWFwW2ZfaV0pO1xuICAgICAgICAgIGxldCBsb2NhbF9jb2VmZiA9IGYzMigxMDApICogaW50ZW5zaXR5O1xuICAgICAgICAgIGNvZWZmICs9IGxvY2FsX2NvZWZmO1xuICAgICAgICAgIGRpcl94ICs9IGxvY2FsX2NvZWZmICogZjMyKGZfeCk7XG4gICAgICAgICAgZGlyX3kgKz0gbG9jYWxfY29lZmYgKiBmMzIoZl95KTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIGxldCB0b3RhbF9jb2VmZiA9IGNvZWZmO1xuICAgIGlmICh0b3RhbF9jb2VmZiAhPSAwKSB7XG4gICAgICB2YXIgZF94ID0gZGlyX3ggLyB0b3RhbF9jb2VmZiAtIHg7XG4gICAgICB2YXIgZF95ID0gZGlyX3kgLyB0b3RhbF9jb2VmZiAtIHk7XG5cbiAgICAgIGxldCBkaXN0MiA9IGRfeCAqIGRfeCArIGRfeSAqIGRfeTtcbiAgICAgIGxldCBtYXhfZGlzdDIgPSBmMzIocmVnaW9uICogcmVnaW9uKTtcblxuICAgICAgbGV0IHNwZWVkID0gZGlzdDIgLyBtYXhfZGlzdDI7XG5cbiAgICAgIGRfeCAqPSBzcGVlZDtcbiAgICAgIGRfeSAqPSBzcGVlZDtcblxuICAgICAgeF9wb3Nfb3V0W2ldID0geCArIGRfeDtcbiAgICAgIHlfcG9zX291dFtpXSA9IHkgKyBkX3k7XG4gICAgfSBlbHNlIHtcbiAgICAgIHhfcG9zX291dFtpXSA9IHg7XG4gICAgICB5X3Bvc19vdXRbaV0gPSB5O1xuICAgIH1cbiAgfVxufVxuYFxufVxuIiwiZXhwb3J0IGZ1bmN0aW9uIGJ1aWxkKG5fZWxlbXM6IG51bWJlciwgZ3JpZF94OiBudW1iZXIsIGdyaWRfc3BhY2luZzogbnVtYmVyKSB7XG4gIHJldHVybiBgXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMClcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB4X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDEpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZT4geV9wb3M6IGFycmF5PGYzMj47XG5cbkBjb21wdXRlIEB3b3JrZ3JvdXBfc2l6ZSgyNTYpXG5mbiBtYWluKFxuICBAYnVpbHRpbihnbG9iYWxfaW52b2NhdGlvbl9pZClcbiAgZ2xvYmFsX2lkIDogdmVjM3UsXG5cbiAgQGJ1aWx0aW4obG9jYWxfaW52b2NhdGlvbl9pZClcbiAgbG9jYWxfaWQgOiB2ZWMzdSxcbikge1xuICBpZiAoZ2xvYmFsX2lkLnggPCAke25fZWxlbXN9KSB7XG4gICAgICBsZXQgeSA9IGdsb2JhbF9pZC54IC8gJHtncmlkX3h9O1xuICAgICAgbGV0IHggPSBnbG9iYWxfaWQueCAlICR7Z3JpZF94fTtcblxuICAgICAgeF9wb3NbZ2xvYmFsX2lkLnhdID0gZjMyKHggKiAke2dyaWRfc3BhY2luZ30pO1xuICAgICAgeV9wb3NbZ2xvYmFsX2lkLnhdID0gZjMyKHkgKiAke2dyaWRfc3BhY2luZ30pO1xuICB9XG59XG5gXG59XG4iLCJpbXBvcnQgeyBidWlsZCBhcyBpbml0YnVpbGQgfSBmcm9tICcuL2luaXQnO1xuaW1wb3J0IHsgYnVpbGQgYXMgZm9yY2VzYnVpbGQgfSBmcm9tICcuL2ZvcmNlcyc7XG5cbmNvbnN0IGVkZ2VzID0gW1xuICBbLTEsIDBdLFxuICBbMSwgMF0sXG4gIFswLCAtMV0sXG4gIFswLCAxXSxcbiAgWzEsIDFdLFxuICBbLTEsIC0xXSxcbl07XG5cbmV4cG9ydCBjbGFzcyBNZXNoRGVmb3JtYXRpb24ge1xuICBjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRDtcbiAgZ3JpZF94OiBudW1iZXI7XG4gIGdyaWRfeTogbnVtYmVyO1xuICBncmlkX3NwYWNpbmc6IG51bWJlcjtcbiAgbWluX2Rpc3Q6IG51bWJlcjtcbiAgbWF4X2Rpc3Q6IG51bWJlcjtcbiAgcmFkaXVzOiBudW1iZXI7XG5cbiAgcmVpbml0OiBib29sZWFuO1xuICBkcmF3X2VkZ2VzOiBib29sZWFuO1xuXG4gIG5fZWxlbXM6IG51bWJlcjtcbiAgeF9wb3M6IEZsb2F0MzJBcnJheTtcbiAgeV9wb3M6IEZsb2F0MzJBcnJheTtcblxuICBpbml0X21vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICBpbml0QmluZEdyb3VwTGF5b3V0OiBHUFVCaW5kR3JvdXBMYXlvdXQ7XG4gIGluaXRfY29tcHV0ZVBpcGVsaW5lOiBHUFVDb21wdXRlUGlwZWxpbmU7XG5cbiAgaW5pdGlhbGl6YXRpb25fZG9uZTogUHJvbWlzZTx2b2lkPjtcbiAgZGV2aWNlOiBHUFVEZXZpY2U7XG4gIGZvcmNlX21vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICBhY3RpdmVfYnVmZmVyX2lkOiBudW1iZXI7XG4gIHhfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIHlfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIC8vIEJ1ZmZlcnMgdG8gcmVhZCB2YWx1ZXMgYmFjayB0byB0aGUgQ1BVIGZvciBkcmF3aW5nXG4gIHN0YWdpbmdfeF9idWY6IEdQVUJ1ZmZlcjtcbiAgc3RhZ2luZ195X2J1ZjogR1BVQnVmZmVyO1xuICAvLyBCdWZmZXIgdG8gd3JpdGUgdmFsdWUgdG8gZnJvbSB0aGUgQ1BVIGZvciBhZGp1c3Rpbmcgd2VpZ2h0c1xuICBzdGFnaW5nX2ludGVuc2l0eV9idWY6IEdQVUJ1ZmZlcjtcbiAgaW50ZW5zaXR5X21hcF9idWY6IEdQVUJ1ZmZlcjtcblxuICBvZmZzZXRfYnVmOiBHUFVCdWZmZXI7XG4gIHN0cmlkZV9idWY6IEdQVUJ1ZmZlcjtcblxuICBmb3JjZV9iaW5kX2dyb3VwX2xheW91dDogR1BVQmluZEdyb3VwTGF5b3V0O1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgIGN0eDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJELFxuICAgIGdyaWRfeDogbnVtYmVyLFxuICAgIGdyaWRfeTogbnVtYmVyLFxuICAgIGdyaWRfc3BhY2luZzogbnVtYmVyLFxuICAgIG1pbl9kaXN0OiBudW1iZXIsXG4gICAgbWF4X2Rpc3Q6IG51bWJlcixcbiAgICByYWRpdXM6IG51bWJlclxuICApIHtcbiAgICB0aGlzLmN0eCA9IGN0eDtcbiAgICB0aGlzLmdyaWRfeCA9IGdyaWRfeDtcbiAgICB0aGlzLmdyaWRfeSA9IGdyaWRfeTtcbiAgICB0aGlzLmdyaWRfc3BhY2luZyA9IGdyaWRfc3BhY2luZztcbiAgICB0aGlzLm1pbl9kaXN0ID0gbWluX2Rpc3Q7XG4gICAgdGhpcy5tYXhfZGlzdCA9IG1heF9kaXN0O1xuICAgIHRoaXMucmFkaXVzID0gcmFkaXVzO1xuXG4gICAgdGhpcy5kcmF3X2VkZ2VzID0gdHJ1ZTtcbiAgICB0aGlzLnJlaW5pdCA9IGZhbHNlO1xuXG4gICAgdGhpcy5uX2VsZW1zID0gdGhpcy5ncmlkX3ggKiB0aGlzLmdyaWRfeTtcblxuICAgIHRoaXMuaW5pdGlhbGl6YXRpb25fZG9uZSA9IHRoaXMuYXN5bmNfaW5pdCgpO1xuICB9XG5cbiAgYXN5bmMgYXN5bmNfaW5pdCgpIHtcbiAgICBjb25zdCBhZGFwdGVyID0gYXdhaXQgbmF2aWdhdG9yLmdwdS5yZXF1ZXN0QWRhcHRlcigpO1xuICAgIHRoaXMuZGV2aWNlID0gYXdhaXQgYWRhcHRlci5yZXF1ZXN0RGV2aWNlKCk7XG4gICAgY29uc29sZS5sb2coXCJDcmVhdGUgY29tcHV0ZSBzaGFkZXJcIik7XG4gICAgdGhpcy5mb3JjZV9tb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogZm9yY2VzYnVpbGQodGhpcy5jdHguY2FudmFzLndpZHRoLCB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0LCB0aGlzLmdyaWRfeCwgdGhpcy5ncmlkX3NwYWNpbmcsIHRoaXMubl9lbGVtcyksXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIENyZWF0ZSBjb21wdXRlIHNoYWRlclwiKTtcblxuICAgIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZCA9IDA7XG4gICAgdGhpcy54X3Bvc19idWZmZXJzID0gW1xuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieF9wb3NbMF1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBsYWJlbDogXCJ4X3Bvc1sxXVwiLFxuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICBdO1xuICAgIHRoaXMueV9wb3NfYnVmZmVycyA9IFtcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIGxhYmVsOiBcInlfcG9zWzBdXCIsXG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgbGFiZWw6IFwieV9wb3NbMV1cIixcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgXTtcblxuICAgIHRoaXMuc3RhZ2luZ194X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJzdGFnaW5nX3hfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0YWdpbmdfeV9idWZcIixcbiAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1JFQUQgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcblxuICAgIHRoaXMuc3RhZ2luZ19pbnRlbnNpdHlfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0YWdpbmdfaW50ZW5zaXR5X2J1ZlwiLFxuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1dSSVRFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkMsXG4gICAgfSk7XG4gICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBsYWJlbDogXCJpbnRlbnNpdHlfYnVmXCIsXG4gICAgICBzaXplOiB0aGlzLmN0eC5jYW52YXMud2lkdGggKiB0aGlzLmN0eC5jYW52YXMuaGVpZ2h0ICogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG5cbiAgICB0aGlzLm9mZnNldF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgbGFiZWw6IFwib2Zmc2V0X2J1ZlwiLFxuICAgICAgc2l6ZTogNCxcbiAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5VTklGT1JNIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgdGhpcy5zdHJpZGVfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIGxhYmVsOiBcInN0cmlkZV9idWZcIixcbiAgICAgIHNpemU6IDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuVU5JRk9STSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBhbGxvY2F0ZSBidWZmZXJzXCIpO1xuXG4gICAgdGhpcy5mb3JjZV9iaW5kX2dyb3VwX2xheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDIsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAzLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogNCxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDUsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwidW5pZm9ybVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA2LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInVuaWZvcm1cIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIC8vIGludGlhbGl6ZSB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSBhbmRcbiAgICAvLyB0aGlzLnlfcG9zX2J1ZmZlcnNbKl0gdG8gYmUgYSBncmlkXG4gICAgY29uc3QgaW5pdF9zaGFkZXIgPSBpbml0YnVpbGQodGhpcy5uX2VsZW1zLCB0aGlzLmdyaWRfeCwgdGhpcy5ncmlkX3NwYWNpbmcpO1xuICAgIHRoaXMuaW5pdEJpbmRHcm91cExheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zb2xlLmxvZyhcIkNyZWF0ZSBpbml0IHNoYWRlclwiKTtcbiAgICB0aGlzLmluaXRfbW9kdWxlID0gdGhpcy5kZXZpY2UuY3JlYXRlU2hhZGVyTW9kdWxlKHtcbiAgICAgIGNvZGU6IGluaXRfc2hhZGVyLFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBDcmVhdGUgaW5pdCBzaGFkZXJcIik7XG4gICAgdGhpcy5pbml0X2NvbXB1dGVQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbXB1dGVQaXBlbGluZSh7XG4gICAgICBsYWJlbDogXCJjb21wdXRlIGZvcmNlXCIsXG4gICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW3RoaXMuaW5pdEJpbmRHcm91cExheW91dF0sXG4gICAgICB9KSxcbiAgICAgIGNvbXB1dGU6IHtcbiAgICAgICAgbW9kdWxlOiB0aGlzLmluaXRfbW9kdWxlLFxuICAgICAgICBlbnRyeVBvaW50OiBcIm1haW5cIixcbiAgICAgIH0sXG4gICAgfSk7XG4gICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuXG4gICAgY29uc3QgYmluZEdyb3VwID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcbiAgICAgIGxheW91dDogdGhpcy5pbml0QmluZEdyb3VwTGF5b3V0LFxuICAgICAgZW50cmllczogW1xuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMCxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9XG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUodGhpcy5pbml0X2NvbXB1dGVQaXBlbGluZSk7XG4gICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKE1hdGguY2VpbCh0aGlzLm5fZWxlbXMgLyAyNTYpKTtcbiAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICBjb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSwgMCxcbiAgICAgIHRoaXMuc3RhZ2luZ194X2J1ZiwgMCxcbiAgICAgIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplXG4gICAgKTtcbiAgICBjb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSwgMCxcbiAgICAgIHRoaXMuc3RhZ2luZ195X2J1ZiwgMCxcbiAgICAgIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplXG4gICAgKTtcbiAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG5cbiAgICBhd2FpdCB0aGlzLnVwZGF0ZUNQVXBvcygpO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBhc3luYyBpbml0XCIpO1xuICB9XG5cbiAgYXN5bmMgdXBkYXRlQ1BVcG9zKCkge1xuICAgIC8vIGNvbnNvbGUubG9nKFwiTWFwIGJ1ZmZlcnMgZm9yIHJlYWRpbmdcIik7XG4gICAgbGV0IG1feCA9IHRoaXMuc3RhZ2luZ194X2J1Zi5tYXBBc3luYyhHUFVNYXBNb2RlLlJFQUQsIDAsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplKTtcbiAgICBsZXQgbV95ID0gdGhpcy5zdGFnaW5nX3lfYnVmLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemUpO1xuICAgIGF3YWl0IG1feDtcbiAgICBhd2FpdCBtX3k7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcImNvcHlpbmcgeCBidWZmZXIgdG8gQ1BVXCIpO1xuICAgIGNvbnN0IGNvcHlBcnJheUJ1ZmZlclggPSB0aGlzLnN0YWdpbmdfeF9idWYuZ2V0TWFwcGVkUmFuZ2UoMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIGNvbnN0IGRhdGFYID0gY29weUFycmF5QnVmZmVyWC5zbGljZSgpO1xuICAgIHRoaXMueF9wb3MgPSBuZXcgRmxvYXQzMkFycmF5KGRhdGFYKTtcblxuICAgIC8vIGNvbnNvbGUubG9nKFwiY29weWluZyB5IGJ1ZmZlciB0byBDUFVcIik7XG4gICAgY29uc3QgY29weUFycmF5QnVmZmVyWSA9IHRoaXMuc3RhZ2luZ195X2J1Zi5nZXRNYXBwZWRSYW5nZSgwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgY29uc3QgZGF0YVkgPSBjb3B5QXJyYXlCdWZmZXJZLnNsaWNlKCk7XG4gICAgdGhpcy55X3BvcyA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YVkpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJ1bm1hcCBidWZmZXJzXCIpO1xuICAgIHRoaXMuc3RhZ2luZ194X2J1Zi51bm1hcCgpO1xuICAgIHRoaXMuc3RhZ2luZ195X2J1Zi51bm1hcCgpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJEb25lIHVwZGF0ZUNQVXBvc1wiKTtcbiAgfVxuXG4gIGFzeW5jIGFwcGx5Rm9yY2UoY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQpIHtcbiAgICBsZXQgaWRhdGEgPSBjdHguZ2V0SW1hZ2VEYXRhKDAsIDAsIGN0eC5jYW52YXMud2lkdGgsIGN0eC5jYW52YXMuaGVpZ2h0KS5kYXRhO1xuICAgIC8vIGNvbnNvbGUubG9nKGBiMCAke2lkYXRhWzBdfSwgJHtpZGF0YVsxXX0sICR7aWRhdGFbMl19LCAke2lkYXRhWzNdfWApO1xuICAgIC8vIGNvbnNvbGUubG9nKGBXcml0aW5nICR7dGhpcy5pbnRlbnNpdHlfbWFwX2J1Zi5zaXplfS8ke2lkYXRhLmxlbmd0aH0gYnl0ZXMgZm9yIGltYXBgKTtcbiAgICBsZXQgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgICB0aGlzLmRldmljZS5xdWV1ZS53cml0ZUJ1ZmZlcihcbiAgICAgIHRoaXMuaW50ZW5zaXR5X21hcF9idWYsIDAsIGlkYXRhLmJ1ZmZlciwgMCwgdGhpcy5pbnRlbnNpdHlfbWFwX2J1Zi5zaXplKTtcbiAgICAod2luZG93IGFzIGFueSkud3JpdGVfdGltZSArPSBwZXJmb3JtYW5jZS5ub3coKSAtIHN0YXJ0O1xuXG4gICAgbGV0IGlucHV0X3ggPSB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgaW5wdXRfeSA9IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBvdXRwdXRfeCA9IHRoaXMueF9wb3NfYnVmZmVyc1sxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgb3V0cHV0X3kgPSB0aGlzLnlfcG9zX2J1ZmZlcnNbMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG5cblxuICAgIGlmICh0aGlzLnJlaW5pdCkge1xuICAgICAgY29uc3QgYmluZEdyb3VwID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcbiAgICAgICAgbGF5b3V0OiB0aGlzLmluaXRCaW5kR3JvdXBMYXlvdXQsXG4gICAgICAgIGVudHJpZXM6IFtcbiAgICAgICAgICB7XG4gICAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgICAgcmVzb3VyY2U6IHtcbiAgICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICBiaW5kaW5nOiAxLFxuICAgICAgICAgICAgcmVzb3VyY2U6IHtcbiAgICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICAgIH0sXG4gICAgICAgICAgfVxuICAgICAgICBdLFxuICAgICAgfSk7XG5cbiAgICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAgIGNvbnN0IHBhc3NFbmNvZGVyID0gY29tbWFuZEVuY29kZXIuYmVnaW5Db21wdXRlUGFzcygpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUodGhpcy5pbml0X2NvbXB1dGVQaXBlbGluZSk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICAgIHBhc3NFbmNvZGVyLmRpc3BhdGNoV29ya2dyb3VwcyhNYXRoLmNlaWwodGhpcy5uX2VsZW1zIC8gMjU2KSk7XG4gICAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICAgIHRoaXMucmVpbml0ID0gZmFsc2U7XG4gICAgfVxuXG5cbiAgICBjb25zdCB3Z194ID0gNjQ7XG4gICAgY29uc3Qgc3RyaWRlID0gODtcbiAgICBjb25zdCBkaXNwYXRjaF94ID0gKDI1NiAvIHdnX3gpO1xuICAgIGxldCBidWZmZXJzID0gW2lucHV0X3gsIGlucHV0X3ksIHRoaXMuaW50ZW5zaXR5X21hcF9idWYsIG91dHB1dF94LCBvdXRwdXRfeSwgdGhpcy5vZmZzZXRfYnVmLCB0aGlzLnN0cmlkZV9idWZdO1xuICAgIGNvbnN0IGJpbmRHcm91cCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cCh7XG4gICAgICBsYXlvdXQ6IHRoaXMuZm9yY2VfYmluZF9ncm91cF9sYXlvdXQsXG4gICAgICBlbnRyaWVzOiBidWZmZXJzLm1hcCgoYiwgaSkgPT4geyByZXR1cm4geyBiaW5kaW5nOiBpLCByZXNvdXJjZTogeyBidWZmZXI6IGIgfSB9OyB9KVxuICAgIH0pO1xuXG4gICAgZm9yIChsZXQgb2Zmc2V0ID0gMDsgb2Zmc2V0IDwgdGhpcy5uX2VsZW1zOyBvZmZzZXQgKz0gKHN0cmlkZSAqIHdnX3ggKiBkaXNwYXRjaF94KSkge1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUud3JpdGVCdWZmZXIoXG4gICAgICAgIHRoaXMub2Zmc2V0X2J1ZiwgMCwgbmV3IFVpbnQzMkFycmF5KFtvZmZzZXRdKS5idWZmZXIsIDAsIDQpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUud3JpdGVCdWZmZXIoXG4gICAgICAgIHRoaXMuc3RyaWRlX2J1ZiwgMCwgbmV3IFVpbnQzMkFycmF5KFtzdHJpZGVdKS5idWZmZXIsIDAsIDQpO1xuXG4gICAgICBjb25zdCBjb21wdXRlUGlwZWxpbmUgPSB0aGlzLmRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgICBsYWJlbDogXCJmb3JjZXBpcGVsaW5lXCIsXG4gICAgICAgIGxheW91dDogdGhpcy5kZXZpY2UuY3JlYXRlUGlwZWxpbmVMYXlvdXQoe1xuICAgICAgICAgIGJpbmRHcm91cExheW91dHM6IFt0aGlzLmZvcmNlX2JpbmRfZ3JvdXBfbGF5b3V0XSxcbiAgICAgICAgfSksXG4gICAgICAgIGNvbXB1dGU6IHtcbiAgICAgICAgICBtb2R1bGU6IHRoaXMuZm9yY2VfbW9kdWxlLFxuICAgICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgICB9LFxuICAgICAgfSk7XG4gICAgICAvLyBjb25zb2xlLmxvZyhcImNyZWF0ZWQgcGlwZWxpbmVcIik7XG5cbiAgICAgIGNvbnN0IGNvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAgIGNvbnN0IHBhc3NFbmNvZGVyID0gY29tbWFuZEVuY29kZXIuYmVnaW5Db21wdXRlUGFzcygpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUoY29tcHV0ZVBpcGVsaW5lKTtcbiAgICAgIHBhc3NFbmNvZGVyLnNldEJpbmRHcm91cCgwLCBiaW5kR3JvdXApO1xuICAgICAgcGFzc0VuY29kZXIuZGlzcGF0Y2hXb3JrZ3JvdXBzKGRpc3BhdGNoX3gsIDEsIDEpO1xuICAgICAgcGFzc0VuY29kZXIuZW5kKCk7XG4gICAgICAvLyBjb25zb2xlLmxvZyhcImVuY29kZWQgY29tcHV0ZVwiKTtcbiAgICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwicXVldWVcIiwgb2Zmc2V0KTtcbiAgICB9XG5cbiAgICBjb25zdCBjb3B5X291dHB1dF9jb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG4gICAgLy8gQ29weSBvdXRwdXQgYnVmZmVyIHRvIHN0YWdpbmcgYnVmZmVyXG4gICAgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgb3V0cHV0X3gsIDAsIHRoaXMuc3RhZ2luZ194X2J1ZiwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwieCBjb3B5aW5nXCIsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplLCBcImJ5dGVzXCIpO1xuICAgIGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIG91dHB1dF95LCAwLCB0aGlzLnN0YWdpbmdfeV9idWYsIDAsIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcInkgY29weWluZ1wiLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSwgXCJieXRlc1wiKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImVuY29kZWQgY29weSB0byBidWZmZXJzXCIsIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZCk7XG5cbiAgICAvLyBFbmQgZnJhbWUgYnkgcGFzc2luZyBhcnJheSBvZiBjb21tYW5kIGJ1ZmZlcnMgdG8gY29tbWFuZCBxdWV1ZSBmb3IgZXhlY3V0aW9uXG4gICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgIC8vIGNvbnNvbGUubG9nKFwiZG9uZSBzdWJtaXQgdG8gcXVldWVcIik7XG5cbiAgICAvLyBTd2FwIGlucHV0IGFuZCBvdXRwdXQ6XG4gICAgdGhpcy5hY3RpdmVfYnVmZmVyX2lkID0gMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZDtcblxuICAgIC8vIGF3YWl0IHRoaXMudXBkYXRlQ1BVcG9zKCk7XG4gICAgLy8gY29uc29sZS5sb2coXCJkb25lIGFwcGx5Rm9yY2VcIik7XG4gIH1cblxuICBkcmF3KCkge1xuICAgIHRoaXMuY3R4LmNsZWFyUmVjdCgwLCAwLCB0aGlzLmN0eC5jYW52YXMud2lkdGgsIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQpO1xuICAgIGZvciAobGV0IHlpZHggPSAwOyB5aWR4IDwgdGhpcy5ncmlkX3k7IHlpZHgrKykge1xuICAgICAgZm9yIChsZXQgeGlkeCA9IDA7IHhpZHggPCB0aGlzLmdyaWRfeDsgeGlkeCsrKSB7XG4gICAgICAgIGxldCBpID0geWlkeCAqIHRoaXMuZ3JpZF94ICsgeGlkeDtcblxuICAgICAgICBsZXQgeCA9IHRoaXMueF9wb3NbaV07XG4gICAgICAgIGxldCB5ID0gdGhpcy55X3Bvc1tpXTtcblxuICAgICAgICB0aGlzLmN0eC5zdHJva2VTdHlsZSA9IFwiI2ZmMDAwMDVmXCI7XG4gICAgICAgIHRoaXMuY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICB0aGlzLmN0eC5hcmMoeCwgeSwgdGhpcy5yYWRpdXMsIDAsIDIgKiBNYXRoLlBJKTtcbiAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG5cbiAgICAgICAgaWYgKHRoaXMuZHJhd19lZGdlcykge1xuICAgICAgICAgIGZvciAobGV0IGVkZ2Ugb2YgZWRnZXMpIHtcbiAgICAgICAgICAgIGxldCBqX3hpZHggPSB4aWR4ICsgZWRnZVswXTtcbiAgICAgICAgICAgIGxldCBqX3lpZHggPSB5aWR4ICsgZWRnZVsxXTtcbiAgICAgICAgICAgIGlmIChqX3hpZHggPCAwIHx8IGpfeGlkeCA+PSB0aGlzLmdyaWRfeCB8fCBqX3lpZHggPCAwIHx8IGpfeWlkeCA+PSB0aGlzLmdyaWRfeSkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgbGV0IGogPSBqX3lpZHggKiB0aGlzLmdyaWRfeCArIGpfeGlkeDtcblxuICAgICAgICAgICAgbGV0IGpfeCA9IHRoaXMueF9wb3Nbal07XG4gICAgICAgICAgICBsZXQgal95ID0gdGhpcy55X3Bvc1tqXTtcblxuICAgICAgICAgICAgdGhpcy5jdHguYmVnaW5QYXRoKCk7XG4gICAgICAgICAgICB0aGlzLmN0eC5tb3ZlVG8oeCwgeSk7XG4gICAgICAgICAgICB0aGlzLmN0eC5saW5lVG8oal94LCBqX3kpO1xuICAgICAgICAgICAgdGhpcy5jdHguc3Ryb2tlKCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG4iLCIvLyBUaGUgbW9kdWxlIGNhY2hlXG52YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG5cbi8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG5mdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuXHR2YXIgY2FjaGVkTW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXTtcblx0aWYgKGNhY2hlZE1vZHVsZSAhPT0gdW5kZWZpbmVkKSB7XG5cdFx0cmV0dXJuIGNhY2hlZE1vZHVsZS5leHBvcnRzO1xuXHR9XG5cdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG5cdHZhciBtb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdID0ge1xuXHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcblx0XHQvLyBubyBtb2R1bGUubG9hZGVkIG5lZWRlZFxuXHRcdGV4cG9ydHM6IHt9XG5cdH07XG5cblx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG5cdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG5cdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG5cdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbn1cblxuIiwiaW1wb3J0IHsgTWVzaERlZm9ybWF0aW9uIH0gZnJvbSAnLi9tZXNoZGVmb3JtYXRpb24nO1xuXG5sZXQgc3RvcCA9IGZhbHNlO1xuXG5hc3luYyBmdW5jdGlvbiBzZXR1cFdlYmNhbSgpIHtcbiAgY29uc3QgdmlkZW8gPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwidmlkZW9cIik7XG4gIGNvbnN0IGNvbnN0cmFpbnRzID0geyB2aWRlbzogdHJ1ZSB9XG5cbiAgdHJ5IHtcbiAgICBpZiAodmlkZW8uc3JjT2JqZWN0KSB7XG4gICAgICBjb25zdCBzdHJlYW0gPSB2aWRlby5zcmNPYmplY3Q7XG4gICAgICBzdHJlYW0uZ2V0VHJhY2tzKCkuZm9yRWFjaChmdW5jdGlvbih0cmFjazogYW55KSB7XG4gICAgICAgIHRyYWNrLnN0b3AoKTtcbiAgICAgIH0pO1xuICAgICAgdmlkZW8uc3JjT2JqZWN0ID0gbnVsbDtcbiAgICB9XG5cbiAgICBjb25zdCBzdHJlYW0gPSBhd2FpdCBuYXZpZ2F0b3IubWVkaWFEZXZpY2VzLmdldFVzZXJNZWRpYShjb25zdHJhaW50cyk7XG4gICAgdmlkZW8uc3JjT2JqZWN0ID0gc3RyZWFtO1xuICAgIHZpZGVvLnBsYXkoKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gICAgYWxlcnQoXCJFcnJvciBpbml0aWFsaXppbmcgd2ViY2FtISBcIiArIGVycik7XG4gICAgY29uc29sZS5sb2coZXJyKTtcbiAgfVxuICByZXR1cm4gdmlkZW87XG59XG5cbmFzeW5jIGZ1bmN0aW9uIG1haW4oY29udGFpbmVyOiBIVE1MRWxlbWVudCkge1xuICBsZXQgY2FudmFzID0gKGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJjYW52YXNcIikgYXMgSFRNTENhbnZhc0VsZW1lbnQpO1xuICBjYW52YXMud2lkdGggPSAxMDAwO1xuICBjYW52YXMuaGVpZ2h0ID0gMTAwMDtcbiAgY29udGFpbmVyLmFwcGVuZENoaWxkKGNhbnZhcyk7XG5cbiAgbGV0IGN0eCA9IGNhbnZhcy5nZXRDb250ZXh0KFwiMmRcIik7XG4gIGNvbnNvbGUubG9nKFwiQ3JlYXRlZCBjb250ZXh0IGZvciBtYWluIGNhbnZhc1wiKTtcblxuICBsZXQgY2FudmFzMiA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJjYW52YXNcIikgYXMgSFRNTENhbnZhc0VsZW1lbnQ7XG4gIGNhbnZhczIud2lkdGggPSAxMDAwO1xuICBjYW52YXMyLmhlaWdodCA9IDEwMDA7XG4gIGxldCBjdHgyID0gY2FudmFzMi5nZXRDb250ZXh0KFwiMmRcIik7XG5cbiAgLy8gbGV0IGlkYXRhOiBVaW50OENsYW1wZWRBcnJheTxBcnJheUJ1ZmZlckxpa2U+ID0gbmV3IFVpbnQ4Q2xhbXBlZEFycmF5KGN0eDIuY2FudmFzLndpZHRoICogY3R4LmNhbnZhcy5oZWlnaHQgKiA0KTtcbiAgY2FudmFzLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCAoZSkgPT4ge1xuICAgIGxldCBlbCA9IGUudGFyZ2V0IGFzIEhUTUxDYW52YXNFbGVtZW50O1xuICAgIGNvbnN0IHJlY3QgPSBlbC5nZXRCb3VuZGluZ0NsaWVudFJlY3QoKTtcbiAgICBjb25zdCB4ID0gZWwud2lkdGggKiAoZS5jbGllbnRYIC0gcmVjdC5sZWZ0KSAvIHJlY3Qud2lkdGg7XG4gICAgY29uc3QgeSA9IGVsLmhlaWdodCAqIChlLmNsaWVudFkgLSByZWN0LnRvcCkgLyByZWN0LmhlaWdodDtcblxuICAgIGN0eDIuYmVnaW5QYXRoKCk7XG4gICAgY3R4Mi5maWxsU3R5bGUgPSBcImJsYWNrXCI7XG4gICAgY3R4Mi5hcmMoeCwgeSwgMTAwLCAwLCAyICogTWF0aC5QSSk7XG4gICAgY3R4Mi5maWxsKCk7XG4gIH0pO1xuXG4gIGNvbnRhaW5lci5hcHBlbmRDaGlsZChkb2N1bWVudC5jcmVhdGVFbGVtZW50KFwiYnJcIikpO1xuXG4gIGNvbnN0IGNsZWFyID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudChcImJ1dHRvblwiKTtcbiAgY2xlYXIuaW5uZXJUZXh0ID0gXCJjbGVhclwiO1xuICBjbGVhci5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKCkgPT4ge1xuICAgIGN0eDIuY2xlYXJSZWN0KDAsIDAsIGN0eDIuY2FudmFzLndpZHRoLCBjdHgyLmNhbnZhcy5oZWlnaHQpO1xuICB9KTtcbiAgY29udGFpbmVyLmFwcGVuZENoaWxkKGNsZWFyKTtcblxuICBjb25zdCBlZGdlcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoXCJidXR0b25cIik7XG4gIGVkZ2VzLmlubmVyVGV4dCA9IFwiZWRnZXNcIjtcbiAgZWRnZXMuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsICgpID0+IHtcbiAgICBtZC5kcmF3X2VkZ2VzID0gIW1kLmRyYXdfZWRnZXM7XG4gIH0pO1xuICBjb250YWluZXIuYXBwZW5kQ2hpbGQoZWRnZXMpO1xuXG4gIGNvbnN0IHZpZGVvID0gYXdhaXQgc2V0dXBXZWJjYW0oKTtcblxuICBjb25zb2xlLmxvZyhcIkNyZWF0ZWQgY29udGV4dCBmb3IgaW50ZXJhY3RpdmUgY2FudmFzXCIpO1xuXG4gICh3aW5kb3cgYXMgYW55KS5uX3N0ZXBzX3Blcl9mcmFtZSA9IDE7XG5cbiAgbGV0IG5fZWxlbXMgPSAxMDA7XG4gIGxldCBzcGFjaW5nID0gY3R4LmNhbnZhcy53aWR0aCAvIG5fZWxlbXM7XG4gIGxldCBtZCA9IG5ldyBNZXNoRGVmb3JtYXRpb24oY3R4LCBuX2VsZW1zLCBuX2VsZW1zLCBzcGFjaW5nLCBzcGFjaW5nIC8gNCwgc3BhY2luZyAqIDQsIDEpO1xuICAod2luZG93IGFzIGFueSkudF9wZXJfcmVuZGVyID0gMDtcbiAgKHdpbmRvdyBhcyBhbnkpLm5fcmVuZGVycyA9IDA7XG4gICh3aW5kb3cgYXMgYW55KS53cml0ZV90aW1lID0gMDtcbiAgbWQuaW5pdGlhbGl6YXRpb25fZG9uZS50aGVuKCgpID0+IHtcbiAgICBjb25zdCBmID0gYXN5bmMgKCkgPT4ge1xuICAgICAgbGV0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8ICh3aW5kb3cgYXMgYW55KS5uX3N0ZXBzX3Blcl9mcmFtZTsgaSsrKSB7XG4gICAgICAgIGF3YWl0IG1kLmFwcGx5Rm9yY2UoY3R4Mik7XG4gICAgICB9XG4gICAgICBsZXQgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgICAod2luZG93IGFzIGFueSkudF9wZXJfcmVuZGVyICs9IGVuZCAtIHN0YXJ0O1xuICAgICAgKHdpbmRvdyBhcyBhbnkpLm5fcmVuZGVycyArPSAxO1xuICAgICAgaWYgKCFzdG9wKSB7XG4gICAgICAgIHJlcXVlc3RBbmltYXRpb25GcmFtZShmKVxuICAgICAgfVxuICAgIH07XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpO1xuXG4gICAgKHdpbmRvdyBhcyBhbnkpLnRfcGVyX2RyYXcgPSAwO1xuICAgICh3aW5kb3cgYXMgYW55KS5uX2RyYXdzID0gMDtcbiAgICBjb25zdCBnID0gYXN5bmMgKCkgPT4ge1xuICAgICAgaWYgKHZpZGVvLnJlYWR5U3RhdGUgIT0gNCkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG4gICAgICBjdHgyLmRyYXdJbWFnZSh2aWRlbywgMCwgMCwgY3R4Mi5jYW52YXMud2lkdGgsIGN0eDIuY2FudmFzLmhlaWdodCk7XG5cbiAgICAgIGxldCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgICAgYXdhaXQgbWQudXBkYXRlQ1BVcG9zKCk7XG4gICAgICBtZC5kcmF3KCk7XG4gICAgICBsZXQgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgICAod2luZG93IGFzIGFueSkudF9wZXJfZHJhdyArPSBlbmQgLSBzdGFydDtcbiAgICAgICh3aW5kb3cgYXMgYW55KS5uX2RyYXdzICs9IDE7XG4gICAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGcpXG4gICAgICB9LCAzMCk7XG4gICAgfTtcbiAgICByZXF1ZXN0QW5pbWF0aW9uRnJhbWUoZyk7XG4gIH0pO1xuXG4gICh3aW5kb3cgYXMgYW55KS5zdGF0cyA9ICgpID0+IHtcbiAgICBsZXQgdyA9IHdpbmRvdyBhcyBhbnk7XG4gICAgY29uc29sZS5sb2coXCJ0X3Blcl9yZW5kZXJcIiwgdy50X3Blcl9yZW5kZXIpO1xuICAgIGNvbnNvbGUubG9nKFwibl9yZW5kZXJzXCIsIHcubl9yZW5kZXJzKTtcbiAgICBjb25zb2xlLmxvZyhcImF2Z1wiLCB3LnRfcGVyX3JlbmRlciAvIHcubl9yZW5kZXJzKTtcbiAgICBjb25zb2xlLmxvZyhcInRfcGVyX3dyaXRlXCIsIHcud3JpdGVfdGltZSk7XG4gICAgY29uc29sZS5sb2coXCIgIGF2Z1wiLCB3LndyaXRlX3RpbWUgLyB3Lm5fcmVuZGVycyk7XG4gICAgY29uc29sZS5sb2coXCJ0X3Blcl9kcmF3XCIsIHcudF9wZXJfZHJhdyk7XG4gICAgY29uc29sZS5sb2coXCJuX2RyYXdzXCIsIHcubl9kcmF3cyk7XG4gICAgY29uc29sZS5sb2coXCJhdmdcIiwgdy50X3Blcl9kcmF3IC8gdy5uX2RyYXdzKTtcbiAgfVxuXG5cbiAgZnVuY3Rpb24gY2FuY2VsKCkge1xuICAgIHN0b3AgPSB0cnVlO1xuICB9XG4gICh3aW5kb3cgYXMgYW55KS5tZCA9IG1kO1xuICAod2luZG93IGFzIGFueSkuY3R4MiA9IGN0eDI7XG4gICh3aW5kb3cgYXMgYW55KS5jYW5jZWwgPSBjYW5jZWw7XG59XG5cbmRvY3VtZW50LmFkZEV2ZW50TGlzdGVuZXIoXCJET01Db250ZW50TG9hZGVkXCIsICgpID0+IHtcbiAgbGV0IGNvbnRhaW5lciA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiY29udGFpbmVyXCIpO1xuICBtYWluKGNvbnRhaW5lcik7XG59KTtcblxuIl0sIm5hbWVzIjpbXSwic291cmNlUm9vdCI6IiJ9