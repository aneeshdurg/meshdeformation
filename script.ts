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

  offset_buf: GPUBuffer;

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

@group(0) @binding(5)
var<uniform> offset: u32;


var<workgroup> dir_x: atomic<i32>;
var<workgroup> dir_y: atomic<i32>;
var<workgroup> coeff: atomic<u32>;

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
      let d_x = f32(f_x) - x;
      let d_y = f32(f_y) - y;
      let r2 = d_x * d_x + d_y * d_y;
      let r = sqrt(r2);

      // Find the force exerted on the particle by contents of the intesity map.
      if (f_y >= 0 && f_y < ${this.grid_y} && f_x >= 0 && f_x < ${this.grid_x}) {
        let f_i = f_y * ${this.grid_x} + f_x;
        let intensity = pixelToIntensity(intensity_map[f_i]);

        if (r != 0) {
          let local_coeff = u32(10000 * 100 * intensity / r2);
          atomicAdd(&coeff, local_coeff);
          atomicAdd(&dir_x, i32(1000 * d_x / r));
          atomicAdd(&dir_y, i32(1000 * d_y / r));
        }
      }
    }
  }

  // Wait for all workgroup threads to finish simulating
  workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    let total_coeff = f32(atomicLoad(&coeff)) / 10000;
    if (total_coeff != 0) {
      x_pos_out[offset + global_id.x] = x + f32(atomicLoad(&dir_x)) / (1000 * total_coeff);
      y_pos_out[offset + global_id.x] = y + f32(atomicLoad(&dir_y)) / (1000 * total_coeff);
    } else {
      x_pos_out[offset + global_id.x] = x;
      y_pos_out[offset + global_id.x] = y;
    }
    // x_pos_out[offset + global_id.x] = x + 5;
    // y_pos_out[offset + global_id.x] = y + 5;
  }
}
`;
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
    const init_shader = `
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
  if (global_id.x < ${this.n_elems}) {
      let y = global_id.x / ${this.grid_x};
      let x = global_id.x % ${this.grid_x};

      x_pos[global_id.x] = f32(x * ${this.grid_spacing});
      y_pos[global_id.x] = f32(y * ${this.grid_spacing});
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
    commandEncoder.copyBufferToBuffer(
      this.x_pos_buffers[this.active_buffer_id], 0,
      this.staging_x_buf, 0,
      this.staging_x_buf.size
    );
    commandEncoder.copyBufferToBuffer(
      this.y_pos_buffers[this.active_buffer_id], 0,
      this.staging_y_buf, 0,
      this.staging_y_buf.size
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await this.updateCPUpos();
    console.log("done async init");
  }

  async updateCPUpos() {
    console.log("Map buffers for reading");
    let m_x = this.staging_x_buf.mapAsync(GPUMapMode.READ, 0, this.staging_x_buf.size);
    let m_y = this.staging_y_buf.mapAsync(GPUMapMode.READ, 0, this.staging_y_buf.size);
    await m_x;
    await m_y;

    console.log("copying x buffer to CPU");
    const copyArrayBufferX = this.staging_x_buf.getMappedRange(0, this.staging_x_buf.size);
    const dataX = copyArrayBufferX.slice();
    this.x_pos = new Float32Array(dataX);

    console.log("copying y buffer to CPU");
    const copyArrayBufferY = this.staging_y_buf.getMappedRange(0, this.staging_y_buf.size);
    const dataY = copyArrayBufferY.slice();
    this.y_pos = new Float32Array(dataY);

    console.log("unmap buffers");
    this.staging_x_buf.unmap();
    this.staging_y_buf.unmap();

    console.log("Done updateCPUpos");
  }

  async applyForce(ctx: CanvasRenderingContext2D) {
    let idata = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
    this.device.queue.writeBuffer(
      this.intensity_map_buf, 0, idata.buffer, this.intensity_map_buf.size);

    let input_x = this.x_pos_buffers[this.active_buffer_id];
    let input_y = this.y_pos_buffers[this.active_buffer_id];
    let output_x = this.x_pos_buffers[1 - this.active_buffer_id];
    let output_y = this.y_pos_buffers[1 - this.active_buffer_id];

    const dispatch_x = 2;
    for (let offset = 0; offset < this.n_elems; offset += dispatch_x) {
      let input = new Uint32Array([offset]);
      this.device.queue.writeBuffer(
        this.offset_buf, 0, input.buffer, 0, 4);

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
      // Set active to output
      console.log("created pipeline");

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
    copy_output_commandEncoder.copyBufferToBuffer(
      output_x, 0, this.staging_x_buf, 0, this.staging_x_buf.size);
    console.log("x copying", this.staging_x_buf.size, "bytes");
    copy_output_commandEncoder.copyBufferToBuffer(
      output_y, 0, this.staging_y_buf, 0, this.staging_y_buf.size);
    console.log("y copying", this.staging_x_buf.size, "bytes");
    console.log("encoded copy to buffers", this.active_buffer_id);

    // End frame by passing array of command buffers to command queue for execution
    this.device.queue.submit([copy_output_commandEncoder.finish()]);
    console.log("done submit to queue");

    // Swap input and output:
    this.active_buffer_id = 1 - this.active_buffer_id;

    await this.updateCPUpos();
    console.log("done applyForce");
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
