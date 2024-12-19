export function build(width: number, height: number, grid_x: number, grid_spacing: number) {
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
  let i = offset + global_id.x;

  let grid_x = i % ${grid_x};
  let grid_y = i / ${grid_x};

  let x = x_pos[i];
  let y = y_pos[i];

  var dir_x: f32 = 0;
  var dir_y: f32 = 0;
  var coeff: f32 = 0;

  // Coordinates to lookup in intensity_map
  for (var s_y: i32 = 0; s_y < 50; s_y++) {
    for (var s_z: i32 = 0; s_z < 50; s_z++) {
      let f_y = i32(floor(y)) + i32(50 * local_id.y) + s_y - 25;
      let f_x = i32(floor(x)) + i32(50 * local_id.z) + s_z - 25;
      let d_x: f32 = f32(f_x) - x;
      let d_y: f32 = f32(f_y) - y;
      let r2: f32 = d_x * d_x + d_y * d_y;
      let r: f32 = sqrt(r2);

      // Find the force exerted on the particle by contents of the intesity map.
      if (f_y >= 0 && f_y < ${height} && f_x >= 0 && f_x < ${width}) {
        let f_i = f_y * ${width} + f_x;
        let intensity = pixelToIntensity(intensity_map[f_i]);

        if (r != 0) {
          let local_coeff = 100 * intensity / r2;
          // atomicAdd(& coeff, u32(10000 * local_coeff));
          // atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
          // atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
          coeff += local_coeff;
          dir_x += local_coeff * d_x / r;
          dir_y += local_coeff * d_y / r;
        }
      }
    }
  }

  if (local_id.y == 0 && local_id.z == 0) {
    let origin_x = grid_x * ${grid_spacing};
    let origin_y = grid_y * ${grid_spacing};
    let d_x: f32 = f32(origin_x) - x;
    let d_y: f32 = f32(origin_y) - y;
    let r2: f32 = d_x * d_x + d_y * d_y;
    let r: f32 = sqrt(r2);

    let local_coeff = f32(5) / r2;
    // atomicAdd(& coeff, u32(10000 * local_coeff));
    // atomicAdd(& dir_x, i32(1000 * local_coeff * d_x / r));
    // atomicAdd(& dir_y, i32(1000 * local_coeff * d_y / r));
    coeff += local_coeff;
    dir_x += local_coeff * d_x / r;
    dir_y += local_coeff * d_y / r;
  }

  // Wait for all workgroup threads to finish simulating
  // workgroupBarrier();

  // On a single thread, update the output position for the current particle
  if (local_id.y == 0 && local_id.z == 0) {
    // let total_coeff = f32(atomicLoad(& coeff)) / 10000;
    let total_coeff = coeff;
    if (total_coeff != 0) {
      // var d_x = f32(atomicLoad(& dir_x)) / (1000 * total_coeff);
      // var d_y = f32(atomicLoad(& dir_y)) / (1000 * total_coeff);
      var d_x = dir_x / total_coeff;
      var d_y = dir_y / total_coeff;

      let s_dx = sign(d_x);
      let s_dy = sign(d_y);
      let a_dx = abs(d_x);
      let a_dy = abs(d_y);

      d_x = s_dx * min(a_dx, f32(0.5));
      d_y = s_dy * min(a_dy, f32(0.5));

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
