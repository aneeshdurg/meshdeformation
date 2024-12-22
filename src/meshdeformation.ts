import { build as initbuild } from './init';
import { build as forcesbuild } from './forces';
import { build as renderbuild } from './render';

const edges = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
  [1, 1],
  [-1, -1],
];

export class MeshDeformation {
  ctx: CanvasRenderingContext2D;
  grid_x: number;
  grid_y: number;
  grid_spacing: number;
  min_dist: number;
  max_dist: number;
  radius: number;

  reinit: boolean;
  draw_edges: boolean;

  n_elems: number;
  x_pos: Float32Array;
  y_pos: Float32Array;

  init_module: GPUShaderModule;
  initBindGroupLayout: GPUBindGroupLayout;
  init_computePipeline: GPUComputePipeline;

  initialization_done: Promise<void>;
  device: GPUDevice;
  force_module: GPUShaderModule;
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
  stride_buf: GPUBuffer;

  image_buf: GPUBuffer;
  image_reset: Float32Array;

  force_bind_group_layout: GPUBindGroupLayout;


  render_to_buf_module: GPUShaderModule;
  render_to_buf_bind_group_layout: GPUBindGroupLayout;
  render_to_buf_compute_pipeline: GPUComputePipeline;

  render_module: GPUShaderModule;
  render_pipeline: GPURenderPipeline;
  render_pass_descriptor: GPURenderPassDescriptor;
  vertex_buffer: GPUBuffer;
  render_bind_group_layout: GPUBindGroupLayout;

  constructor(
    ctx: CanvasRenderingContext2D,
    grid_x: number,
    grid_y: number,
    grid_spacing: number,
    min_dist: number,
    max_dist: number,
    radius: number
  ) {
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
      code: forcesbuild(this.ctx.canvas.width, this.ctx.canvas.height, this.grid_x, this.grid_spacing, this.n_elems),
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
    const init_shader = initbuild(this.n_elems, this.grid_x, this.grid_spacing);
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


    // intialize this.x_pos_buffers[this.active_buffer_id] and
    // this.y_pos_buffers[*] to be a grid
    const render_to_buffer_shader = renderbuild(
      this.ctx.canvas.width,
      this.ctx.canvas.height,
      this.grid_x,
      this.grid_spacing,
      this.n_elems,
      this.radius
    );
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


    (this.ctx as any).configure({
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

    this.render_pipeline = this.device.createRenderPipeline(
      pipelineDescriptor as GPURenderPipelineDescriptor);

    const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 };

    this.render_pass_descriptor = {
      colorAttachments: [
        {
          clearValue: clearColor,
          loadOp: "clear",
          storeOp: "store",
          view: (this.ctx as any).getCurrentTexture().createView(),
        },
      ],
    } as GPURenderPassDescriptor;

    console.log("done async init");
  }


  async applyForce(ctx: CanvasRenderingContext2D) {
    let idata = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
    // console.log(`b0 ${idata[0]}, ${idata[1]}, ${idata[2]}, ${idata[3]}`);
    // console.log(`Writing ${this.intensity_map_buf.size}/${idata.length} bytes for imap`);
    let start = performance.now();
    this.device.queue.writeBuffer(
      this.intensity_map_buf, 0, idata.buffer, 0, this.intensity_map_buf.size);
    (window as any).write_time += performance.now() - start;

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
      this.device.queue.writeBuffer(
        this.offset_buf, 0, new Uint32Array([offset]).buffer, 0, 4);
      this.device.queue.writeBuffer(
        this.stride_buf, 0, new Uint32Array([stride]).buffer, 0, 4);

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
      this.device.queue.writeBuffer(
        this.offset_buf, 0, new Uint32Array([offset]).buffer, 0, 4);

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
    copy_output_commandEncoder.copyBufferToBuffer(
      output_x, 0, this.staging_x_buf, 0, this.staging_x_buf.size);
    // console.log("x copying", this.staging_x_buf.size, "bytes");
    copy_output_commandEncoder.copyBufferToBuffer(
      output_y, 0, this.staging_y_buf, 0, this.staging_y_buf.size);
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
        entries: [{binding: 0, resource: {buffer: this.image_buf}}],
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
