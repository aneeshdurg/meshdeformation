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
function build(width, height) {
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

@compute @workgroup_size(1, 10, 10)
fn main(
  @builtin(global_invocation_id)
global_id : vec3u,

  @builtin(local_invocation_id)
local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let x = x_pos[offset + global_id.x];
  let y = y_pos[offset + global_id.x];

  // Coordinates to lookup in intensity_map
  for (var s_y: i32 = 0; s_y < 5; s_y++) {
    for (var s_z: i32 = 0; s_z < 5; s_z++) {
      let f_y = i32(floor(y)) + i32(5 * local_id.y) + s_y - 25;
      let f_x = i32(floor(x)) + i32(5 * local_id.z) + s_z - 25;
      let d_x: f32 = f32(f_x) - x;
      let d_y: f32 = f32(f_y) - y;
      let r2: f32 = d_x * d_x + d_y * d_y;
      let r: f32 = sqrt(r2);

      // Find the force exerted on the particle by contents of the intesity map.
      if (f_y >= 0 && f_y < ${height} && f_x >= 0 && f_x < ${width}) {
        let f_i = f_y * ${width} + f_x;
        let intensity = pixelToIntensity(intensity_map[f_i]);

        if (r != 0) {
          let local_coeff = u32(10000 * 100 * intensity / r2);
          atomicAdd(& coeff, local_coeff);
          atomicAdd(& dir_x, i32(1000 * d_x / r));
          atomicAdd(& dir_y, i32(1000 * d_y / r));
        }
      }
    }
  }

  // Wait for all workgroup threads to finish simulating
  workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    let total_coeff = f32(atomicLoad(& coeff)) / 10000;
    if (total_coeff != 0) {
      x_pos_out[offset + global_id.x] = x + f32(atomicLoad(& dir_x)) / (1000 * total_coeff);
      y_pos_out[offset + global_id.x] = y + f32(atomicLoad(& dir_y)) / (1000 * total_coeff);
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

/***/ "./src/script.ts":
/*!***********************!*\
  !*** ./src/script.ts ***!
  \***********************/
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
    radius;
    n_elems;
    x_pos;
    y_pos;
    shader;
    initialization_done;
    device;
    shaderModule;
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
    bindGroupLayout;
    constructor(ctx, grid_x, grid_y, grid_spacing, min_dist, radius) {
        this.ctx = ctx;
        this.grid_x = grid_x;
        this.grid_y = grid_y;
        this.grid_spacing = grid_spacing;
        this.min_dist = min_dist;
        this.radius = radius;
        this.n_elems = this.grid_x * this.grid_y;
        this.shader = (0, forces_1.build)(this.ctx.canvas.width, this.ctx.canvas.height);
        this.initialization_done = this.async_init();
    }
    async async_init() {
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();
        console.log("Create compute shader");
        this.shaderModule = this.device.createShaderModule({
            code: this.shader,
        });
        console.log("done Create compute shader");
        this.active_buffer_id = 0;
        this.x_pos_buffers = [
            this.device.createBuffer({
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
            this.device.createBuffer({
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
        ];
        this.y_pos_buffers = [
            this.device.createBuffer({
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
            this.device.createBuffer({
                size: this.n_elems * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }),
        ];
        this.staging_x_buf = this.device.createBuffer({
            size: this.n_elems * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.staging_y_buf = this.device.createBuffer({
            size: this.n_elems * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.staging_intensity_buf = this.device.createBuffer({
            size: this.ctx.canvas.width * this.ctx.canvas.height * 4,
            usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        });
        this.intensity_map_buf = this.device.createBuffer({
            size: this.ctx.canvas.width * this.ctx.canvas.height * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.offset_buf = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        console.log("done allocate buffers");
        this.bindGroupLayout = this.device.createBindGroupLayout({
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
        const dispatch_x = 2;
        for (let offset = 0; offset < this.n_elems; offset += dispatch_x) {
            let input = new Uint32Array([offset]);
            this.device.queue.writeBuffer(this.offset_buf, 0, input.buffer, 0, 4);
            let buffers = [input_x, input_y, this.intensity_map_buf, output_x, output_y, this.offset_buf];
            const bindGroup = this.device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: buffers.map((b, i) => { return { binding: i, resource: { buffer: b } }; })
            });
            const computePipeline = this.device.createComputePipeline({
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.bindGroupLayout],
                }),
                compute: {
                    module: this.shaderModule,
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
const script_1 = __webpack_require__(/*! ./script */ "./src/script.ts");
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
    let md = new script_1.MeshDeformation(ctx, 20, 20, ctx.canvas.width / 20, 10, 5);
    md.initialization_done.then(() => {
        const f = async () => {
            md.draw();
            await md.applyForce(ctx2);
            await md.updateCPUpos();
            if (!stop) {
                setTimeout(() => {
                    requestAnimationFrame(f);
                }, 1);
            }
        };
        requestAnimationFrame(f);
    });
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,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LENBQUMsQ0FBQyxDQUFDO0lBQ3hFLEVBQUUsQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1FBQy9CLE1BQU0sQ0FBQyxHQUFHLEtBQUssSUFBSSxFQUFFO1lBQ25CLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQztZQUNWLE1BQU0sRUFBRSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQztZQUMxQixNQUFNLEVBQUUsQ0FBQyxZQUFZLEVBQUUsQ0FBQztZQUN4QixJQUFJLENBQUMsSUFBSSxFQUFFLENBQUM7Z0JBQ1YsVUFBVSxDQUFDLEdBQUcsRUFBRTtvQkFDZCxxQkFBcUIsQ0FBQyxDQUFDLENBQUM7Z0JBQzFCLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUNSLENBQUM7UUFDSCxDQUFDLENBQUM7UUFDRixxQkFBcUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMzQixDQUFDLENBQUMsQ0FBQztJQUdILFNBQVMsTUFBTTtRQUNiLElBQUksR0FBRyxJQUFJLENBQUM7SUFDZCxDQUFDO0lBQ0EsTUFBYyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7SUFDdkIsTUFBYyxDQUFDLElBQUksR0FBRyxJQUFJLENBQUM7SUFDM0IsTUFBYyxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7QUFDbEMsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9mb3JjZXMudHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvaW5pdC50cyIsIndlYnBhY2s6Ly95b3VyUHJvamVjdC8uL3NyYy9zY3JpcHQudHMiLCJ3ZWJwYWNrOi8veW91clByb2plY3Qvd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8veW91clByb2plY3QvLi9zcmMvbWFpbi50cyJdLCJzb3VyY2VzQ29udGVudCI6WyJleHBvcnQgZnVuY3Rpb24gYnVpbGQod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGUgPiB4X3BvczogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDEpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHlfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMilcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlID4gaW50ZW5zaXR5X21hcDogYXJyYXk8dTMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDMpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHhfcG9zX291dDogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDQpXG52YXI8c3RvcmFnZSwgcmVhZF93cml0ZSA+IHlfcG9zX291dDogYXJyYXk8ZjMyPjtcblxuQGdyb3VwKDApIEBiaW5kaW5nKDUpXG52YXI8dW5pZm9ybT5vZmZzZXQ6IHUzMjtcblxuXG52YXI8d29ya2dyb3VwPmRpcl94OiBhdG9taWM8aTMyPjtcbnZhcjx3b3JrZ3JvdXA+ZGlyX3k6IGF0b21pYzxpMzI+O1xudmFyPHdvcmtncm91cD5jb2VmZjogYXRvbWljPHUzMj47XG5cbmZuIHBpeGVsVG9JbnRlbnNpdHkoX3B4OiB1MzIpIC0+IGYzMiB7XG4gIHZhciBweCA9IF9weDtcbiAgbGV0IHIgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGcgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGIgPSBmMzIocHggJSAyNTYpO1xuICBweCAvPSB1MzIoMjU2KTtcbiAgbGV0IGEgPSBmMzIocHggJSAyNTYpO1xuICBsZXQgaW50ZW5zaXR5OiBmMzIgPSAoYSAvIDI1NSkgKiAoMSAtICgwLjIxMjYgKiByICsgMC43MTUyICogZyArIDAuMDcyMiAqIGIpKTtcblxuICByZXR1cm4gaW50ZW5zaXR5O1xufVxuXG5AY29tcHV0ZSBAd29ya2dyb3VwX3NpemUoMSwgMTAsIDEwKVxuZm4gbWFpbihcbiAgQGJ1aWx0aW4oZ2xvYmFsX2ludm9jYXRpb25faWQpXG5nbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxubG9jYWxfaWQgOiB2ZWMzdSxcbikge1xuICAvLyBDb29yZGluYXRlcyBvZiBwYXJ0aWNsZSBmb3IgdGhpcyB0aHJlYWRcbiAgbGV0IHggPSB4X3Bvc1tvZmZzZXQgKyBnbG9iYWxfaWQueF07XG4gIGxldCB5ID0geV9wb3Nbb2Zmc2V0ICsgZ2xvYmFsX2lkLnhdO1xuXG4gIC8vIENvb3JkaW5hdGVzIHRvIGxvb2t1cCBpbiBpbnRlbnNpdHlfbWFwXG4gIGZvciAodmFyIHNfeTogaTMyID0gMDsgc195IDwgNTsgc195KyspIHtcbiAgICBmb3IgKHZhciBzX3o6IGkzMiA9IDA7IHNfeiA8IDU7IHNfeisrKSB7XG4gICAgICBsZXQgZl95ID0gaTMyKGZsb29yKHkpKSArIGkzMig1ICogbG9jYWxfaWQueSkgKyBzX3kgLSAyNTtcbiAgICAgIGxldCBmX3ggPSBpMzIoZmxvb3IoeCkpICsgaTMyKDUgKiBsb2NhbF9pZC56KSArIHNfeiAtIDI1O1xuICAgICAgbGV0IGRfeDogZjMyID0gZjMyKGZfeCkgLSB4O1xuICAgICAgbGV0IGRfeTogZjMyID0gZjMyKGZfeSkgLSB5O1xuICAgICAgbGV0IHIyOiBmMzIgPSBkX3ggKiBkX3ggKyBkX3kgKiBkX3k7XG4gICAgICBsZXQgcjogZjMyID0gc3FydChyMik7XG5cbiAgICAgIC8vIEZpbmQgdGhlIGZvcmNlIGV4ZXJ0ZWQgb24gdGhlIHBhcnRpY2xlIGJ5IGNvbnRlbnRzIG9mIHRoZSBpbnRlc2l0eSBtYXAuXG4gICAgICBpZiAoZl95ID49IDAgJiYgZl95IDwgJHsgaGVpZ2h0IH0gJiYgZl94ID49IDAgJiYgZl94IDwgJHsgd2lkdGggfSkge1xuICAgICAgICBsZXQgZl9pID0gZl95ICogJHsgd2lkdGggfSArIGZfeDtcbiAgICAgICAgbGV0IGludGVuc2l0eSA9IHBpeGVsVG9JbnRlbnNpdHkoaW50ZW5zaXR5X21hcFtmX2ldKTtcblxuICAgICAgICBpZiAociAhPSAwKSB7XG4gICAgICAgICAgbGV0IGxvY2FsX2NvZWZmID0gdTMyKDEwMDAwICogMTAwICogaW50ZW5zaXR5IC8gcjIpO1xuICAgICAgICAgIGF0b21pY0FkZCgmIGNvZWZmLCBsb2NhbF9jb2VmZik7XG4gICAgICAgICAgYXRvbWljQWRkKCYgZGlyX3gsIGkzMigxMDAwICogZF94IC8gcikpO1xuICAgICAgICAgIGF0b21pY0FkZCgmIGRpcl95LCBpMzIoMTAwMCAqIGRfeSAvIHIpKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIC8vIFdhaXQgZm9yIGFsbCB3b3JrZ3JvdXAgdGhyZWFkcyB0byBmaW5pc2ggc2ltdWxhdGluZ1xuICB3b3JrZ3JvdXBCYXJyaWVyKCk7XG5cbiAgLy8gT24gYSBzaW5nbGUgdGhyZWFkLCB1cGRhdGUgdGhlIG91dHB1dCBwb3NpdGlvbiBmb3IgdGhlIGN1cnJlbnQgcGFydGljbGVcbiAgaWYgKGxvY2FsX2lkLnkgPT0gMCAmJiBsb2NhbF9pZC56ID09IDApIHtcbiAgICBsZXQgdG90YWxfY29lZmYgPSBmMzIoYXRvbWljTG9hZCgmIGNvZWZmKSkgLyAxMDAwMDtcbiAgICBpZiAodG90YWxfY29lZmYgIT0gMCkge1xuICAgICAgeF9wb3Nfb3V0W29mZnNldCArIGdsb2JhbF9pZC54XSA9IHggKyBmMzIoYXRvbWljTG9hZCgmIGRpcl94KSkgLyAoMTAwMCAqIHRvdGFsX2NvZWZmKTtcbiAgICAgIHlfcG9zX291dFtvZmZzZXQgKyBnbG9iYWxfaWQueF0gPSB5ICsgZjMyKGF0b21pY0xvYWQoJiBkaXJfeSkpIC8gKDEwMDAgKiB0b3RhbF9jb2VmZik7XG4gICAgfSBlbHNlIHtcbiAgICAgIHhfcG9zX291dFtvZmZzZXQgKyBnbG9iYWxfaWQueF0gPSB4O1xuICAgICAgeV9wb3Nfb3V0W29mZnNldCArIGdsb2JhbF9pZC54XSA9IHk7XG4gICAgfVxuICB9XG59XG5gXG59XG4iLCJleHBvcnQgZnVuY3Rpb24gYnVpbGQobl9lbGVtczogbnVtYmVyLCBncmlkX3g6IG51bWJlciwgZ3JpZF9zcGFjaW5nOiBudW1iZXIpIHtcbiAgcmV0dXJuIGBcbkBncm91cCgwKSBAYmluZGluZygwKVxudmFyPHN0b3JhZ2UsIHJlYWRfd3JpdGU+IHhfcG9zOiBhcnJheTxmMzI+O1xuXG5AZ3JvdXAoMCkgQGJpbmRpbmcoMSlcbnZhcjxzdG9yYWdlLCByZWFkX3dyaXRlPiB5X3BvczogYXJyYXk8ZjMyPjtcblxuQGNvbXB1dGUgQHdvcmtncm91cF9zaXplKDI1NilcbmZuIG1haW4oXG4gIEBidWlsdGluKGdsb2JhbF9pbnZvY2F0aW9uX2lkKVxuICBnbG9iYWxfaWQgOiB2ZWMzdSxcblxuICBAYnVpbHRpbihsb2NhbF9pbnZvY2F0aW9uX2lkKVxuICBsb2NhbF9pZCA6IHZlYzN1LFxuKSB7XG4gIGlmIChnbG9iYWxfaWQueCA8ICR7bl9lbGVtc30pIHtcbiAgICAgIGxldCB5ID0gZ2xvYmFsX2lkLnggLyAke2dyaWRfeH07XG4gICAgICBsZXQgeCA9IGdsb2JhbF9pZC54ICUgJHtncmlkX3h9O1xuXG4gICAgICB4X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeCAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gICAgICB5X3Bvc1tnbG9iYWxfaWQueF0gPSBmMzIoeSAqICR7Z3JpZF9zcGFjaW5nfSk7XG4gIH1cbn1cbmBcbn1cbiIsImltcG9ydCB7IGJ1aWxkIGFzIGluaXRidWlsZCB9IGZyb20gJy4vaW5pdCc7XG5pbXBvcnQgeyBidWlsZCBhcyBmb3JjZXNidWlsZCB9IGZyb20gJy4vZm9yY2VzJztcblxuY29uc3QgZWRnZXMgPSBbXG4gIFstMSwgMF0sXG4gIFsxLCAwXSxcbiAgWzAsIC0xXSxcbiAgWzAsIDFdLFxuICBbMSwgMV0sXG4gIFstMSwgLTFdLFxuXTtcblxuZXhwb3J0IGNsYXNzIE1lc2hEZWZvcm1hdGlvbiB7XG4gIGN0eDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuICBncmlkX3g6IG51bWJlcjtcbiAgZ3JpZF95OiBudW1iZXI7XG4gIGdyaWRfc3BhY2luZzogbnVtYmVyO1xuICBtaW5fZGlzdDogbnVtYmVyO1xuICByYWRpdXM6IG51bWJlcjtcblxuICBuX2VsZW1zOiBudW1iZXI7XG4gIHhfcG9zOiBGbG9hdDMyQXJyYXk7XG4gIHlfcG9zOiBGbG9hdDMyQXJyYXk7XG5cbiAgc2hhZGVyOiBzdHJpbmc7XG5cbiAgaW5pdGlhbGl6YXRpb25fZG9uZTogUHJvbWlzZTx2b2lkPjtcbiAgZGV2aWNlOiBHUFVEZXZpY2U7XG4gIHNoYWRlck1vZHVsZTogR1BVU2hhZGVyTW9kdWxlO1xuICBhY3RpdmVfYnVmZmVyX2lkOiBudW1iZXI7XG4gIHhfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIHlfcG9zX2J1ZmZlcnM6IFtHUFVCdWZmZXIsIEdQVUJ1ZmZlcl07XG4gIC8vIEJ1ZmZlcnMgdG8gcmVhZCB2YWx1ZXMgYmFjayB0byB0aGUgQ1BVIGZvciBkcmF3aW5nXG4gIHN0YWdpbmdfeF9idWY6IEdQVUJ1ZmZlcjtcbiAgc3RhZ2luZ195X2J1ZjogR1BVQnVmZmVyO1xuICAvLyBCdWZmZXIgdG8gd3JpdGUgdmFsdWUgdG8gZnJvbSB0aGUgQ1BVIGZvciBhZGp1c3Rpbmcgd2VpZ2h0c1xuICBzdGFnaW5nX2ludGVuc2l0eV9idWY6IEdQVUJ1ZmZlcjtcbiAgaW50ZW5zaXR5X21hcF9idWY6IEdQVUJ1ZmZlcjtcblxuICBvZmZzZXRfYnVmOiBHUFVCdWZmZXI7XG5cbiAgYmluZEdyb3VwTGF5b3V0OiBHUFVCaW5kR3JvdXBMYXlvdXQ7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQsXG4gICAgZ3JpZF94OiBudW1iZXIsXG4gICAgZ3JpZF95OiBudW1iZXIsXG4gICAgZ3JpZF9zcGFjaW5nOiBudW1iZXIsXG4gICAgbWluX2Rpc3Q6IG51bWJlcixcbiAgICByYWRpdXM6IG51bWJlclxuICApIHtcbiAgICB0aGlzLmN0eCA9IGN0eDtcbiAgICB0aGlzLmdyaWRfeCA9IGdyaWRfeDtcbiAgICB0aGlzLmdyaWRfeSA9IGdyaWRfeTtcbiAgICB0aGlzLmdyaWRfc3BhY2luZyA9IGdyaWRfc3BhY2luZztcbiAgICB0aGlzLm1pbl9kaXN0ID0gbWluX2Rpc3Q7XG4gICAgdGhpcy5yYWRpdXMgPSByYWRpdXM7XG5cbiAgICB0aGlzLm5fZWxlbXMgPSB0aGlzLmdyaWRfeCAqIHRoaXMuZ3JpZF95O1xuXG4gICAgdGhpcy5zaGFkZXIgPSBmb3JjZXNidWlsZCh0aGlzLmN0eC5jYW52YXMud2lkdGgsIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQpO1xuICAgIHRoaXMuaW5pdGlhbGl6YXRpb25fZG9uZSA9IHRoaXMuYXN5bmNfaW5pdCgpO1xuICB9XG5cbiAgYXN5bmMgYXN5bmNfaW5pdCgpIHtcbiAgICBjb25zdCBhZGFwdGVyID0gYXdhaXQgbmF2aWdhdG9yLmdwdS5yZXF1ZXN0QWRhcHRlcigpO1xuICAgIHRoaXMuZGV2aWNlID0gYXdhaXQgYWRhcHRlci5yZXF1ZXN0RGV2aWNlKCk7XG4gICAgY29uc29sZS5sb2coXCJDcmVhdGUgY29tcHV0ZSBzaGFkZXJcIik7XG4gICAgdGhpcy5zaGFkZXJNb2R1bGUgPSB0aGlzLmRldmljZS5jcmVhdGVTaGFkZXJNb2R1bGUoe1xuICAgICAgY29kZTogdGhpcy5zaGFkZXIsXG4gICAgfSk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIENyZWF0ZSBjb21wdXRlIHNoYWRlclwiKTtcblxuICAgIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZCA9IDA7XG4gICAgdGhpcy54X3Bvc19idWZmZXJzID0gW1xuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgICB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDXG4gICAgICB9KSxcbiAgICBdO1xuICAgIHRoaXMueV9wb3NfYnVmZmVycyA9IFtcbiAgICAgIHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICAgIHVzYWdlOiBHUFVCdWZmZXJVc2FnZS5TVE9SQUdFIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9TUkNcbiAgICAgIH0pLFxuICAgICAgdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgICAgc2l6ZTogdGhpcy5uX2VsZW1zICogNCxcbiAgICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLlNUT1JBR0UgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX1NSQ1xuICAgICAgfSksXG4gICAgXTtcblxuICAgIHRoaXMuc3RhZ2luZ194X2J1ZiA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJ1ZmZlcih7XG4gICAgICBzaXplOiB0aGlzLm5fZWxlbXMgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9SRUFEIHwgR1BVQnVmZmVyVXNhZ2UuQ09QWV9EU1QsXG4gICAgfSk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIHNpemU6IHRoaXMubl9lbGVtcyAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuTUFQX1JFQUQgfCBHUFVCdWZmZXJVc2FnZS5DT1BZX0RTVCxcbiAgICB9KTtcblxuICAgIHRoaXMuc3RhZ2luZ19pbnRlbnNpdHlfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIHNpemU6IHRoaXMuY3R4LmNhbnZhcy53aWR0aCAqIHRoaXMuY3R4LmNhbnZhcy5oZWlnaHQgKiA0LFxuICAgICAgdXNhZ2U6IEdQVUJ1ZmZlclVzYWdlLk1BUF9XUklURSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfU1JDLFxuICAgIH0pO1xuICAgIHRoaXMuaW50ZW5zaXR5X21hcF9idWYgPSB0aGlzLmRldmljZS5jcmVhdGVCdWZmZXIoe1xuICAgICAgc2l6ZTogdGhpcy5jdHguY2FudmFzLndpZHRoICogdGhpcy5jdHguY2FudmFzLmhlaWdodCAqIDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuU1RPUkFHRSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuXG4gICAgdGhpcy5vZmZzZXRfYnVmID0gdGhpcy5kZXZpY2UuY3JlYXRlQnVmZmVyKHtcbiAgICAgIHNpemU6IDQsXG4gICAgICB1c2FnZTogR1BVQnVmZmVyVXNhZ2UuVU5JRk9STSB8IEdQVUJ1ZmZlclVzYWdlLkNPUFlfRFNULFxuICAgIH0pO1xuICAgIGNvbnNvbGUubG9nKFwiZG9uZSBhbGxvY2F0ZSBidWZmZXJzXCIpO1xuXG4gICAgdGhpcy5iaW5kR3JvdXBMYXlvdXQgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXBMYXlvdXQoe1xuICAgICAgZW50cmllczogW1xuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMCxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDEsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAyLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMyxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIGJpbmRpbmc6IDQsXG4gICAgICAgICAgdmlzaWJpbGl0eTogR1BVU2hhZGVyU3RhZ2UuQ09NUFVURSxcbiAgICAgICAgICBidWZmZXI6IHtcbiAgICAgICAgICAgIHR5cGU6IFwic3RvcmFnZVwiLFxuICAgICAgICAgIH0sXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiA1LFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInVuaWZvcm1cIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgXSxcbiAgICB9KTtcblxuICAgIC8vIGludGlhbGl6ZSB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSBhbmRcbiAgICAvLyB0aGlzLnlfcG9zX2J1ZmZlcnNbKl0gdG8gYmUgYSBncmlkXG4gICAgY29uc3QgaW5pdF9zaGFkZXIgPSBpbml0YnVpbGQodGhpcy5uX2VsZW1zLCB0aGlzLmdyaWRfeCwgdGhpcy5ncmlkX3NwYWNpbmcpO1xuXG4gICAgY29uc3QgaW5pdEJpbmRHcm91cExheW91dCA9IHRoaXMuZGV2aWNlLmNyZWF0ZUJpbmRHcm91cExheW91dCh7XG4gICAgICBlbnRyaWVzOiBbXG4gICAgICAgIHtcbiAgICAgICAgICBiaW5kaW5nOiAwLFxuICAgICAgICAgIHZpc2liaWxpdHk6IEdQVVNoYWRlclN0YWdlLkNPTVBVVEUsXG4gICAgICAgICAgYnVmZmVyOiB7XG4gICAgICAgICAgICB0eXBlOiBcInN0b3JhZ2VcIixcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICB2aXNpYmlsaXR5OiBHUFVTaGFkZXJTdGFnZS5DT01QVVRFLFxuICAgICAgICAgIGJ1ZmZlcjoge1xuICAgICAgICAgICAgdHlwZTogXCJzdG9yYWdlXCIsXG4gICAgICAgICAgfSxcbiAgICAgICAgfSxcbiAgICAgIF0sXG4gICAgfSk7XG5cbiAgICBjb25zb2xlLmxvZyhcIkNyZWF0ZSBpbml0IHNoYWRlclwiKTtcbiAgICBjb25zdCBpbml0X21vZHVsZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZVNoYWRlck1vZHVsZSh7XG4gICAgICBjb2RlOiBpbml0X3NoYWRlcixcbiAgICB9KTtcbiAgICBjb25zb2xlLmxvZyhcImRvbmUgQ3JlYXRlIGluaXQgc2hhZGVyXCIpO1xuICAgIGNvbnN0IGNvbXB1dGVQaXBlbGluZSA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbXB1dGVQaXBlbGluZSh7XG4gICAgICBsYWJlbDogXCJjb21wdXRlIGZvcmNlXCIsXG4gICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgYmluZEdyb3VwTGF5b3V0czogW2luaXRCaW5kR3JvdXBMYXlvdXRdLFxuICAgICAgfSksXG4gICAgICBjb21wdXRlOiB7XG4gICAgICAgIG1vZHVsZTogaW5pdF9tb2R1bGUsXG4gICAgICAgIGVudHJ5UG9pbnQ6IFwibWFpblwiLFxuICAgICAgfSxcbiAgICB9KTtcbiAgICBjb25zdCBjb21tYW5kRW5jb2RlciA9IHRoaXMuZGV2aWNlLmNyZWF0ZUNvbW1hbmRFbmNvZGVyKCk7XG5cbiAgICBjb25zdCBiaW5kR3JvdXAgPSB0aGlzLmRldmljZS5jcmVhdGVCaW5kR3JvdXAoe1xuICAgICAgbGF5b3V0OiBpbml0QmluZEdyb3VwTGF5b3V0LFxuICAgICAgZW50cmllczogW1xuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMCxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgYmluZGluZzogMSxcbiAgICAgICAgICByZXNvdXJjZToge1xuICAgICAgICAgICAgYnVmZmVyOiB0aGlzLnlfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXSxcbiAgICAgICAgICB9LFxuICAgICAgICB9XG4gICAgICBdLFxuICAgIH0pO1xuXG4gICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgcGFzc0VuY29kZXIuc2V0UGlwZWxpbmUoY29tcHV0ZVBpcGVsaW5lKTtcbiAgICBwYXNzRW5jb2Rlci5zZXRCaW5kR3JvdXAoMCwgYmluZEdyb3VwKTtcbiAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoTWF0aC5jZWlsKHRoaXMubl9lbGVtcyAvIDI1NikpO1xuICAgIHBhc3NFbmNvZGVyLmVuZCgpO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueF9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemVcbiAgICApO1xuICAgIGNvbW1hbmRFbmNvZGVyLmNvcHlCdWZmZXJUb0J1ZmZlcihcbiAgICAgIHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLCAwLFxuICAgICAgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemVcbiAgICApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLnN1Ym1pdChbY29tbWFuZEVuY29kZXIuZmluaXNoKCldKTtcblxuICAgIGF3YWl0IHRoaXMudXBkYXRlQ1BVcG9zKCk7XG4gICAgY29uc29sZS5sb2coXCJkb25lIGFzeW5jIGluaXRcIik7XG4gIH1cblxuICBhc3luYyB1cGRhdGVDUFVwb3MoKSB7XG4gICAgLy8gY29uc29sZS5sb2coXCJNYXAgYnVmZmVycyBmb3IgcmVhZGluZ1wiKTtcbiAgICBsZXQgbV94ID0gdGhpcy5zdGFnaW5nX3hfYnVmLm1hcEFzeW5jKEdQVU1hcE1vZGUuUkVBRCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUpO1xuICAgIGxldCBtX3kgPSB0aGlzLnN0YWdpbmdfeV9idWYubWFwQXN5bmMoR1BVTWFwTW9kZS5SRUFELCAwLCB0aGlzLnN0YWdpbmdfeV9idWYuc2l6ZSk7XG4gICAgYXdhaXQgbV94O1xuICAgIGF3YWl0IG1feTtcblxuICAgIC8vIGNvbnNvbGUubG9nKFwiY29weWluZyB4IGJ1ZmZlciB0byBDUFVcIik7XG4gICAgY29uc3QgY29weUFycmF5QnVmZmVyWCA9IHRoaXMuc3RhZ2luZ194X2J1Zi5nZXRNYXBwZWRSYW5nZSgwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgY29uc3QgZGF0YVggPSBjb3B5QXJyYXlCdWZmZXJYLnNsaWNlKCk7XG4gICAgdGhpcy54X3BvcyA9IG5ldyBGbG9hdDMyQXJyYXkoZGF0YVgpO1xuXG4gICAgLy8gY29uc29sZS5sb2coXCJjb3B5aW5nIHkgYnVmZmVyIHRvIENQVVwiKTtcbiAgICBjb25zdCBjb3B5QXJyYXlCdWZmZXJZID0gdGhpcy5zdGFnaW5nX3lfYnVmLmdldE1hcHBlZFJhbmdlKDAsIHRoaXMuc3RhZ2luZ195X2J1Zi5zaXplKTtcbiAgICBjb25zdCBkYXRhWSA9IGNvcHlBcnJheUJ1ZmZlclkuc2xpY2UoKTtcbiAgICB0aGlzLnlfcG9zID0gbmV3IEZsb2F0MzJBcnJheShkYXRhWSk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcInVubWFwIGJ1ZmZlcnNcIik7XG4gICAgdGhpcy5zdGFnaW5nX3hfYnVmLnVubWFwKCk7XG4gICAgdGhpcy5zdGFnaW5nX3lfYnVmLnVubWFwKCk7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhcIkRvbmUgdXBkYXRlQ1BVcG9zXCIpO1xuICB9XG5cbiAgYXN5bmMgYXBwbHlGb3JjZShjdHg6IENhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCkge1xuICAgIGxldCBpZGF0YSA9IGN0eC5nZXRJbWFnZURhdGEoMCwgMCwgY3R4LmNhbnZhcy53aWR0aCwgY3R4LmNhbnZhcy5oZWlnaHQpLmRhdGE7XG4gICAgLy8gY29uc29sZS5sb2coYGIwICR7aWRhdGFbMF19LCAke2lkYXRhWzFdfSwgJHtpZGF0YVsyXX0sICR7aWRhdGFbM119YCk7XG4gICAgLy8gY29uc29sZS5sb2coYFdyaXRpbmcgJHt0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemV9LyR7aWRhdGEubGVuZ3RofSBieXRlcyBmb3IgaW1hcGApO1xuICAgIHRoaXMuZGV2aWNlLnF1ZXVlLndyaXRlQnVmZmVyKFxuICAgICAgdGhpcy5pbnRlbnNpdHlfbWFwX2J1ZiwgMCwgaWRhdGEuYnVmZmVyLCAwLCB0aGlzLmludGVuc2l0eV9tYXBfYnVmLnNpemUpO1xuXG4gICAgbGV0IGlucHV0X3ggPSB0aGlzLnhfcG9zX2J1ZmZlcnNbdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgaW5wdXRfeSA9IHRoaXMueV9wb3NfYnVmZmVyc1t0aGlzLmFjdGl2ZV9idWZmZXJfaWRdO1xuICAgIGxldCBvdXRwdXRfeCA9IHRoaXMueF9wb3NfYnVmZmVyc1sxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkXTtcbiAgICBsZXQgb3V0cHV0X3kgPSB0aGlzLnlfcG9zX2J1ZmZlcnNbMSAtIHRoaXMuYWN0aXZlX2J1ZmZlcl9pZF07XG5cbiAgICBjb25zdCBkaXNwYXRjaF94ID0gMjtcbiAgICBmb3IgKGxldCBvZmZzZXQgPSAwOyBvZmZzZXQgPCB0aGlzLm5fZWxlbXM7IG9mZnNldCArPSBkaXNwYXRjaF94KSB7XG4gICAgICBsZXQgaW5wdXQgPSBuZXcgVWludDMyQXJyYXkoW29mZnNldF0pO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUud3JpdGVCdWZmZXIoXG4gICAgICAgIHRoaXMub2Zmc2V0X2J1ZiwgMCwgaW5wdXQuYnVmZmVyLCAwLCA0KTtcblxuICAgICAgbGV0IGJ1ZmZlcnMgPSBbaW5wdXRfeCwgaW5wdXRfeSwgdGhpcy5pbnRlbnNpdHlfbWFwX2J1Ziwgb3V0cHV0X3gsIG91dHB1dF95LCB0aGlzLm9mZnNldF9idWZdO1xuICAgICAgY29uc3QgYmluZEdyb3VwID0gdGhpcy5kZXZpY2UuY3JlYXRlQmluZEdyb3VwKHtcbiAgICAgICAgbGF5b3V0OiB0aGlzLmJpbmRHcm91cExheW91dCxcbiAgICAgICAgZW50cmllczogYnVmZmVycy5tYXAoKGIsIGkpID0+IHsgcmV0dXJuIHsgYmluZGluZzogaSwgcmVzb3VyY2U6IHsgYnVmZmVyOiBiIH0gfTsgfSlcbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCBjb21wdXRlUGlwZWxpbmUgPSB0aGlzLmRldmljZS5jcmVhdGVDb21wdXRlUGlwZWxpbmUoe1xuICAgICAgICBsYXlvdXQ6IHRoaXMuZGV2aWNlLmNyZWF0ZVBpcGVsaW5lTGF5b3V0KHtcbiAgICAgICAgICBiaW5kR3JvdXBMYXlvdXRzOiBbdGhpcy5iaW5kR3JvdXBMYXlvdXRdLFxuICAgICAgICB9KSxcbiAgICAgICAgY29tcHV0ZToge1xuICAgICAgICAgIG1vZHVsZTogdGhpcy5zaGFkZXJNb2R1bGUsXG4gICAgICAgICAgZW50cnlQb2ludDogXCJtYWluXCIsXG4gICAgICAgIH0sXG4gICAgICB9KTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiY3JlYXRlZCBwaXBlbGluZVwiKTtcblxuICAgICAgY29uc3QgY29tbWFuZEVuY29kZXIgPSB0aGlzLmRldmljZS5jcmVhdGVDb21tYW5kRW5jb2RlcigpO1xuICAgICAgY29uc3QgcGFzc0VuY29kZXIgPSBjb21tYW5kRW5jb2Rlci5iZWdpbkNvbXB1dGVQYXNzKCk7XG4gICAgICBwYXNzRW5jb2Rlci5zZXRQaXBlbGluZShjb21wdXRlUGlwZWxpbmUpO1xuICAgICAgcGFzc0VuY29kZXIuc2V0QmluZEdyb3VwKDAsIGJpbmRHcm91cCk7XG4gICAgICBwYXNzRW5jb2Rlci5kaXNwYXRjaFdvcmtncm91cHMoZGlzcGF0Y2hfeCwgMSwgMSk7XG4gICAgICBwYXNzRW5jb2Rlci5lbmQoKTtcbiAgICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb21wdXRlXCIpO1xuICAgICAgdGhpcy5kZXZpY2UucXVldWUuc3VibWl0KFtjb21tYW5kRW5jb2Rlci5maW5pc2goKV0pO1xuICAgIH1cblxuICAgIGNvbnN0IGNvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyID0gdGhpcy5kZXZpY2UuY3JlYXRlQ29tbWFuZEVuY29kZXIoKTtcbiAgICAvLyBDb3B5IG91dHB1dCBidWZmZXIgdG8gc3RhZ2luZyBidWZmZXJcbiAgICBjb3B5X291dHB1dF9jb21tYW5kRW5jb2Rlci5jb3B5QnVmZmVyVG9CdWZmZXIoXG4gICAgICBvdXRwdXRfeCwgMCwgdGhpcy5zdGFnaW5nX3hfYnVmLCAwLCB0aGlzLnN0YWdpbmdfeF9idWYuc2l6ZSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJ4IGNvcHlpbmdcIiwgdGhpcy5zdGFnaW5nX3hfYnVmLnNpemUsIFwiYnl0ZXNcIik7XG4gICAgY29weV9vdXRwdXRfY29tbWFuZEVuY29kZXIuY29weUJ1ZmZlclRvQnVmZmVyKFxuICAgICAgb3V0cHV0X3ksIDAsIHRoaXMuc3RhZ2luZ195X2J1ZiwgMCwgdGhpcy5zdGFnaW5nX3lfYnVmLnNpemUpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwieSBjb3B5aW5nXCIsIHRoaXMuc3RhZ2luZ194X2J1Zi5zaXplLCBcImJ5dGVzXCIpO1xuICAgIC8vIGNvbnNvbGUubG9nKFwiZW5jb2RlZCBjb3B5IHRvIGJ1ZmZlcnNcIiwgdGhpcy5hY3RpdmVfYnVmZmVyX2lkKTtcblxuICAgIC8vIEVuZCBmcmFtZSBieSBwYXNzaW5nIGFycmF5IG9mIGNvbW1hbmQgYnVmZmVycyB0byBjb21tYW5kIHF1ZXVlIGZvciBleGVjdXRpb25cbiAgICB0aGlzLmRldmljZS5xdWV1ZS5zdWJtaXQoW2NvcHlfb3V0cHV0X2NvbW1hbmRFbmNvZGVyLmZpbmlzaCgpXSk7XG4gICAgLy8gY29uc29sZS5sb2coXCJkb25lIHN1Ym1pdCB0byBxdWV1ZVwiKTtcblxuICAgIC8vIFN3YXAgaW5wdXQgYW5kIG91dHB1dDpcbiAgICB0aGlzLmFjdGl2ZV9idWZmZXJfaWQgPSAxIC0gdGhpcy5hY3RpdmVfYnVmZmVyX2lkO1xuXG4gICAgLy8gYXdhaXQgdGhpcy51cGRhdGVDUFVwb3MoKTtcbiAgICAvLyBjb25zb2xlLmxvZyhcImRvbmUgYXBwbHlGb3JjZVwiKTtcbiAgfVxuXG4gIGRyYXcoKSB7XG4gICAgdGhpcy5jdHguY2xlYXJSZWN0KDAsIDAsIHRoaXMuY3R4LmNhbnZhcy53aWR0aCwgdGhpcy5jdHguY2FudmFzLmhlaWdodCk7XG4gICAgZm9yIChsZXQgeWlkeCA9IDA7IHlpZHggPCB0aGlzLmdyaWRfeTsgeWlkeCsrKSB7XG4gICAgICBmb3IgKGxldCB4aWR4ID0gMDsgeGlkeCA8IHRoaXMuZ3JpZF94OyB4aWR4KyspIHtcbiAgICAgICAgbGV0IGkgPSB5aWR4ICogdGhpcy5ncmlkX3ggKyB4aWR4O1xuXG4gICAgICAgIGxldCB4ID0gdGhpcy54X3Bvc1tpXTtcbiAgICAgICAgbGV0IHkgPSB0aGlzLnlfcG9zW2ldO1xuXG4gICAgICAgIHRoaXMuY3R4LnN0cm9rZVN0eWxlID0gXCIjZmYwMDAwNWZcIjtcbiAgICAgICAgdGhpcy5jdHguYmVnaW5QYXRoKCk7XG4gICAgICAgIHRoaXMuY3R4LmFyYyh4LCB5LCB0aGlzLnJhZGl1cywgMCwgMiAqIE1hdGguUEkpO1xuICAgICAgICB0aGlzLmN0eC5zdHJva2UoKTtcblxuICAgICAgICBmb3IgKGxldCBlZGdlIG9mIGVkZ2VzKSB7XG4gICAgICAgICAgbGV0IGpfeGlkeCA9IHhpZHggKyBlZGdlWzBdO1xuICAgICAgICAgIGxldCBqX3lpZHggPSB5aWR4ICsgZWRnZVsxXTtcbiAgICAgICAgICBpZiAoal94aWR4IDwgMCB8fCBqX3hpZHggPj0gdGhpcy5ncmlkX3ggfHwgal95aWR4IDwgMCB8fCBqX3lpZHggPj0gdGhpcy5ncmlkX3kpIHtcbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGxldCBqID0gal95aWR4ICogdGhpcy5ncmlkX3ggKyBqX3hpZHg7XG5cbiAgICAgICAgICBsZXQgal94ID0gdGhpcy54X3Bvc1tqXTtcbiAgICAgICAgICBsZXQgal95ID0gdGhpcy55X3Bvc1tqXTtcblxuICAgICAgICAgIHRoaXMuY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICAgIHRoaXMuY3R4Lm1vdmVUbyh4LCB5KTtcbiAgICAgICAgICB0aGlzLmN0eC5saW5lVG8oal94LCBqX3kpO1xuICAgICAgICAgIHRoaXMuY3R4LnN0cm9rZSgpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG4iLCIvLyBUaGUgbW9kdWxlIGNhY2hlXG52YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG5cbi8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG5mdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuXHR2YXIgY2FjaGVkTW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXTtcblx0aWYgKGNhY2hlZE1vZHVsZSAhPT0gdW5kZWZpbmVkKSB7XG5cdFx0cmV0dXJuIGNhY2hlZE1vZHVsZS5leHBvcnRzO1xuXHR9XG5cdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG5cdHZhciBtb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdID0ge1xuXHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcblx0XHQvLyBubyBtb2R1bGUubG9hZGVkIG5lZWRlZFxuXHRcdGV4cG9ydHM6IHt9XG5cdH07XG5cblx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG5cdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG5cdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG5cdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbn1cblxuIiwiaW1wb3J0IHsgTWVzaERlZm9ybWF0aW9uIH0gZnJvbSAnLi9zY3JpcHQnO1xuXG5sZXQgc3RvcCA9IGZhbHNlO1xuXG5kb2N1bWVudC5hZGRFdmVudExpc3RlbmVyKFwiRE9NQ29udGVudExvYWRlZFwiLCAoKSA9PiB7XG4gIGxldCBjYW52YXMgPSAoZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJteWNhbnZhc1wiKSBhcyBIVE1MQ2FudmFzRWxlbWVudCk7XG4gIGNhbnZhcy53aWR0aCA9IDEwMDA7XG4gIGNhbnZhcy5oZWlnaHQgPSAxMDAwO1xuXG4gIGxldCBjdHggPSBjYW52YXMuZ2V0Q29udGV4dChcIjJkXCIpO1xuICBjb25zb2xlLmxvZyhcIkNyZWF0ZWQgY29udGV4dCBmb3IgbWFpbiBjYW52YXNcIik7XG5cbiAgbGV0IGNhbnZhczIgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcIm15Y2FudmFzMlwiKSBhcyBIVE1MQ2FudmFzRWxlbWVudDtcbiAgY2FudmFzMi53aWR0aCA9IDEwMDA7XG4gIGNhbnZhczIuaGVpZ2h0ID0gMTAwMDtcbiAgbGV0IGN0eDIgPSBjYW52YXMyLmdldENvbnRleHQoXCIyZFwiKTtcbiAgY2FudmFzMi5hZGRFdmVudExpc3RlbmVyKFwiY2xpY2tcIiwgKGUpID0+IHtcbiAgICBsZXQgZWwgPSBlLnRhcmdldCBhcyBIVE1MQ2FudmFzRWxlbWVudDtcbiAgICBjb25zdCByZWN0ID0gZWwuZ2V0Qm91bmRpbmdDbGllbnRSZWN0KCk7XG4gICAgY29uc3QgeCA9IGVsLndpZHRoICogKGUuY2xpZW50WCAtIHJlY3QubGVmdCkgLyByZWN0LndpZHRoO1xuICAgIGNvbnN0IHkgPSBlbC5oZWlnaHQgKiAoZS5jbGllbnRZIC0gcmVjdC50b3ApIC8gcmVjdC5oZWlnaHQ7XG5cbiAgICBjdHgyLmJlZ2luUGF0aCgpO1xuICAgIGN0eDIuZmlsbFN0eWxlID0gXCJibGFja1wiO1xuICAgIGN0eDIuYXJjKHgsIHksIDEwMCwgMCwgMiAqIE1hdGguUEkpO1xuICAgIGN0eDIuZmlsbCgpO1xuICB9KTtcblxuICBjb25zb2xlLmxvZyhcIkNyZWF0ZWQgY29udGV4dCBmb3IgaW50ZXJhY3RpdmUgY2FudmFzXCIpO1xuXG4gIGxldCBtZCA9IG5ldyBNZXNoRGVmb3JtYXRpb24oY3R4LCAyMCwgMjAsIGN0eC5jYW52YXMud2lkdGggLyAyMCwgMTAsIDUpO1xuICBtZC5pbml0aWFsaXphdGlvbl9kb25lLnRoZW4oKCkgPT4ge1xuICAgIGNvbnN0IGYgPSBhc3luYyAoKSA9PiB7XG4gICAgICBtZC5kcmF3KCk7XG4gICAgICBhd2FpdCBtZC5hcHBseUZvcmNlKGN0eDIpO1xuICAgICAgYXdhaXQgbWQudXBkYXRlQ1BVcG9zKCk7XG4gICAgICBpZiAoIXN0b3ApIHtcbiAgICAgICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICAgICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpXG4gICAgICAgIH0sIDEpO1xuICAgICAgfVxuICAgIH07XG4gICAgcmVxdWVzdEFuaW1hdGlvbkZyYW1lKGYpO1xuICB9KTtcblxuXG4gIGZ1bmN0aW9uIGNhbmNlbCgpIHtcbiAgICBzdG9wID0gdHJ1ZTtcbiAgfVxuICAod2luZG93IGFzIGFueSkubWQgPSBtZDtcbiAgKHdpbmRvdyBhcyBhbnkpLmN0eDIgPSBjdHgyO1xuICAod2luZG93IGFzIGFueSkuY2FuY2VsID0gY2FuY2VsO1xufSk7XG5cbiJdLCJuYW1lcyI6W10sInNvdXJjZVJvb3QiOiIifQ==