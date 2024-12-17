/// <reference types="./types" />

const max_vel = [1, 1];
const edges = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
  [1, 1],
  [-1, -1],
];

class MeshDeformation {
  ctx: CanvasRenderingContext2D;
  grid_x: number;
  grid_y: number;
  grid_spacing: number;
  min_dist: number;
  radius: number;

  n_elems: number;
  velocity_x: Float32Array;
  velocity_y: Float32Array;
  x_pos: Float32Array;
  y_pos: Float32Array;

  shader: string;

  initialization_done: Promise<void>;
  device: GPUDevice;
  shaderModule: GPUShaderModule;
  active_buffer_id: number;
  x_pos_buffers: [GPUBuffer, GPUBuffer];
  y_pos_buffers: [GPUBuffer, GPUBuffer];
  // Buffers to read values back to the CPU for drawing
  staging_x_buf: GPUBuffer;
  staging_y_buf: GPUBuffer;
  // Buffer to write value to from the CPU for adjusting weights
  staging_intensity_buf: GPUBuffer;
  intensity_map_buf: GPUBuffer;

  bindGroupLayout: GPUBindGroupLayout;

  constructor(
    ctx: CanvasRenderingContext2D,
    grid_x: number,
    grid_y: number,
    grid_spacing: number,
    min_dist: number,
    radius: number
  ) {
    this.ctx = ctx;
    this.grid_x = grid_x;
    this.grid_y = grid_y;
    this.grid_spacing = grid_spacing;
    this.min_dist = min_dist;
    this.radius = radius;

    this.n_elems = this.grid_x * this.grid_y;

    // Velocity of each grid element (update from previous iter)
    this.velocity_x = new Float32Array(this.n_elems);
    this.velocity_y = new Float32Array(this.n_elems);
    // Position of each grid element
    this.x_pos = new Float32Array(this.n_elems);
    this.y_pos = new Float32Array(this.n_elems);
    for (let i = 0; i < this.n_elems; i++) {
      let y = Math.floor(i / this.grid_x);
      let x = i % this.grid_x;
      this.x_pos[i] = x * this.grid_spacing;
      this.y_pos[i] = y * this.grid_spacing;
    }
    this.shader = `
@group(0) @binding(0)
var<storage, read> x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read> y_pos: array<f32>;

@group(0) @binding(2)
var<storage, read> intensity_map: array<u32>;

@group(0) @binding(3)
var<storage, read_write> x_pos_out: array<f32>;

@group(0) @binding(4)
var<storage, read_write> y_pos_out: array<f32>;


var<workgroup> dir_x: atomic<f32>;
var<workgroup> dir_y: atomic<f32>;
var<workgroup> coeff: atomic<f32>;

fn pixelToIntensity(px: u32) -> f32 {
  let px = intensity_map[f_i];
  let r = f32(px % 256);
  px /= 256;
  let g = f32(px % 256);
  px /= 256;
  let b = f32(px % 256);
  px /= 256;
  let a = f32(px % 256);
  let intensity: f32 = (a / 255) * (1 - (0.2126 * r + 0.7152 * g + 0.0722 * b));

  return intensity;
}

@compute @workgroup_size(1, 50, 50)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let x = x_pos[global_id.x];
  let y = y_pos[global_id.x];

  // Coordinates to lookup in intensity_map
  let f_y = i32(floor(y)) + local_id.y - 25;
  let f_x = i32(floor(x)) + local_id.z - 25;
  let d_x = f32(x_f) - x;
  let d_y = f32(y_f) - y;
  let r2 = d_x * d_x + d_y * d_y;
  let r = sqrt(r2);

  // Find the force exerted on the particle by contents of the intesity map.
  if (f_y >= 0 && f_y < ${this.grid_y} && f_x >= 0 && f_x < ${this.grid_x}) {
    let f_i = f_y * ${this.grid_x} + f_x;
    let intensity = pixelToIntensity(intensity_map[f_i]);

    if (r != 0) {
      let local_coeff = 100 * intensity / r2;
      atomicAdd(&coeff, local_coeff);
      atomicAdd(&dir_x, d_x / r);
      atomicAdd(&dir_y, d_y / r);
    }
  }

  // Wait for all workgroup threads to finish simulating
  workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    let total_coeff = atomicLoad(&coeff);
    if (total_coeff != 0) {
      x_pos_out[global_id.x] = x + atomicLoad(&dir_x) / total_coeff;
      y_pos_out[global_id.y] = y + atomicLoad(&dir_y) / total_coeff;
    }
  }
}
`;
    this.initialization_done = this.async_init();
  }

  async async_init() {
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();
    this.shaderModule = this.device.createShaderModule({
      code: this.shader,
    });

    this.active_buffer_id = 0;
    this.x_pos_buffers = [
      this.device.createBuffer({
        size: this.n_elems * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      this.device.createBuffer({
        size: this.n_elems * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
    ];
    this.y_pos_buffers = [
      this.device.createBuffer({
        size: this.n_elems * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      }),
      this.device.createBuffer({
        size: this.n_elems * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
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
      size: this.n_elems * 4,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    });
    this.intensity_map_buf = this.device.createBuffer({
      size: this.ctx.canvas.width * this.ctx.canvas.height * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

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
      ],
    });

    // intialize this.x_pos_buffers[this.active_buffer_id] and
    // this.y_pos_buffers[*] to be a grid
    const init_shader = `
@group(0) @binding(0)
var<storage, read_write> x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y_pos: array<f32>;

@compute @workgroup_size(1024)
fn main(
  @builtin(global_invocation_id)
  global_id : vec3u,

  @builtin(local_invocation_id)
  local_id : vec3u,
) {
  if (global_invocation_id.x < ${this.n_elems}) {
      let y = global_invocation_id.x / ${this.grid_x};
      let x = global_invocation_id.x % ${this.grid_x};

      x_pos[global_invocation_id.x] = x * ${this.grid_spacing};
      y_pos[global_invocation_id.x] = y * ${this.grid_spacing};
  }
}
`;

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

    const init_module = this.device.createShaderModule({
      code: init_shader,
    })
    const computePipeline = this.device.createComputePipeline({
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
    passEncoder.dispatchWorkgroups(Math.ceil(this.n_elems / 1024));
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
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

  async applyForce(ctx: CanvasRenderingContext2D) {
    let idata = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
    // TODO write idata into this.intensity_map_buf
    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
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
        },
        {
          binding: 2,
          resource: {
            buffer: this.intensity_map_buf,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.x_pos_buffers[1 - this.active_buffer_id],
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.y_pos_buffers[1 - this.active_buffer_id],
          },
        }
      ],
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
    this.active_buffer_id = 1 - this.active_buffer_id;

    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.staging_intensity_buf,
      0, // Source offset
      this.intensity_map_buf,
      0, // Destination offset
      this.intensity_map_buf.size, // Length, in bytes
    );

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(NUM_ELEMENTS / 64));
    passEncoder.end();

    // Copy output buffer to staging buffer
    commandEncoder.copyBufferToBuffer(
      this.x_pos_buffers[this.active_buffer_id],
      0, // Source offset
      this.staging_x_buf,
      0, // Destination offset
      this.staging_x_buf.size, // Length, in bytes
    );
    commandEncoder.copyBufferToBuffer(
      this.y_pos_buffers[this.active_buffer_id],
      0, // Source offset
      this.staging_y_buf,
      0, // Destination offset
      this.staging_y_buf.size, // Length, in bytes
    );

    // End frame by passing array of command buffers to command queue for execution
    this.device.queue.submit([commandEncoder.finish()]);

    let m_x = this.staging_x_buf.mapAsync(GPUMapMode.READ, 0, this.staging_x_buf.size);
    let m_y = this.staging_y_buf.mapAsync(GPUMapMode.READ, 0, this.staging_y_buf.size);
    await m_x;
    await m_y;

    const copyArrayBufferX = this.staging_x_buf.getMappedRange(0, this.staging_x_buf.size);
    const dataX = copyArrayBufferX.slice();
    this.staging_x_buf.unmap();
    this.x_pos = new Float32Array(dataX);

    const copyArrayBufferY = this.staging_y_buf.getMappedRange(0, this.staging_y_buf.size);
    const dataY = copyArrayBufferY.slice();
    this.y_pos = new Float32Array(dataY);
  }
}
