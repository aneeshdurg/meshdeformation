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


export function build(grid_x: number, grid_y: number, min_dist: number, max_dist: number) {
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
`
  }

  src += `
  if (invalid) {
    x_pos_out[i] = x_pos[i];
    y_pos_out[i] = y_pos[i];
  }
`

  src += "}\n";
  return src
}
