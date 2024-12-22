export function build(width: number, height: number, grid_x: number, grid_spacing: number, n_elems: number) {
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

      var speed = dist2 / max_dist2;
      if (dist2 < f32(5)) {
        speed = f32(1);
      } else {
        speed = f32(0.5);
      }

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
`
}
