export function build(n_elems: number, grid_x: number, grid_spacing: number) {
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
`
}
