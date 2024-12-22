export function build(
  width: number,
  height: number,
  grid_x: number,
  grid_spacing: number,
  n_elems: number,
  radius: number,
) {
  return `
@group(0) @binding(0)
var<storage, read_write > x_pos: array<f32>;

@group(0) @binding(1)
var<storage, read_write > y_pos: array<f32>;

@group(0) @binding(2)
var<storage, read_write> intensity_map: array<u32>;

@group(0) @binding(3)
var<storage, read_write > image_map: array<f32>;

@group(0) @binding(4)
var<uniform>offset: u32;


@compute @workgroup_size(256, 1, 1)
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
  let origin_x = grid_x * ${grid_spacing};
  let origin_y = grid_y * ${grid_spacing};

  if (i > ${n_elems}) {
    return;
  }

  // var px = intensity_map[origin_x + origin_y * ${width}];
  // let r = f32(px % 256) / 256;
  // px /= u32(256);
  // let g = f32(px % 256) / 256;
  // px /= u32(256);
  // let b = f32(px % 256) / 256;
  let r = f32(1);
  let g = f32(0);
  let b = f32(0);

  let x = x_pos[i];
  let y = y_pos[i];

  for (var yi = -20; yi <= 20; yi++) {
    for (var xi = -20; xi <= 20; xi++) {
      let ox = i32(origin_x) + xi;
      let oy = i32(origin_y) + yi;
      if (ox < 0 || ox > ${width} || oy < 0 || oy > ${height}) {
        continue;
      }
      let dx = f32(ox) - x;
      let dy = f32(oy) - y;

      let d = sqrt(dx * dx + dy * dy);
      if (abs(d - ${radius}) < 1) {
        let out = ox + oy * ${width};
        image_map[4 * out + 0] = r;
        image_map[4 * out + 1] = g;
        image_map[4 * out + 2] = b;
        image_map[4 * out + 3] = f32(1);
      }
    }
  }
}
`
}
