export function build(width: number, height: number) {
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
      if (f_y >= 0 && f_y < ${ height } && f_x >= 0 && f_x < ${ width }) {
        let f_i = f_y * ${ width } + f_x;
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
`
}
