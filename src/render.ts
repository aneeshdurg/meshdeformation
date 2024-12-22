export function build(
  width: number,
  height: number,
  grid_x: number,
  grid_y: number,
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

@group(0) @binding(5)
var<uniform>mode: u32;


@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(global_invocation_id)
global_id : vec3u,

  @builtin(local_invocation_id)
local_id : vec3u,
) {
  // Coordinates of particle for this thread
  let i = offset + global_id.x;
  if (i > ${n_elems}) {
    return;
  }

  let grid_x = i % ${grid_x};
  let grid_y = i / ${grid_x};
  let origin_x = grid_x * ${grid_spacing};
  let origin_y = grid_y * ${grid_spacing};

  let right_xi = grid_x + 1;
  let right_yi = grid_y;
  let right_i = right_xi + right_yi * ${grid_x};
  let right_is_valid = (right_xi < ${grid_x}) && (right_yi < ${grid_y});
  var right_x:f32 = 0;
  var right_y:f32 = 0;
  if (right_is_valid) {
    right_x = x_pos[right_i];
    right_y = y_pos[right_i];
  }

  let down_xi = grid_x;
  let down_yi = grid_y + 1;
  let down_i = down_xi + down_yi * ${grid_x};
  let down_is_valid = (down_xi < ${grid_x}) && (down_yi < ${grid_y});
  var down_x:f32 = 0;
  var down_y:f32 = 0;
  if (down_is_valid) {
    down_x = x_pos[down_i];
    down_y = y_pos[down_i];
  }

  let diag_xi = grid_x + 1;
  let diag_yi = grid_y + 1;
  let diag_i = diag_xi + diag_yi * ${grid_x};
  let diag_is_valid = (diag_xi < ${grid_x}) && (diag_yi < ${grid_y});
  var diag_x:f32 = 0;
  var diag_y:f32 = 0;
  if (diag_is_valid) {
    diag_x = x_pos[diag_i];
    diag_y = y_pos[diag_i];
  }

  var px = intensity_map[origin_x + origin_y * ${width}];
  let i_r = f32(px % 256) / 256;
  px /= u32(256);
  let i_g = f32(px % 256) / 256;
  px /= u32(256);
  let i_b = f32(px % 256) / 256;

  let x = x_pos[i];
  let y = y_pos[i];

  let dox = f32(origin_x) - x;
  let doy = f32(origin_y) - y;
  let dist_to_origin = sqrt(dox * dox + doy * doy);

  let r = (f32(1) - dist_to_origin) * f32(1) + dist_to_origin * i_r;
  let g = (f32(1) - dist_to_origin) * f32(0) + dist_to_origin * i_g;
  let b = (f32(1) - dist_to_origin) * f32(0) + dist_to_origin * i_b;
  // let r = f32(1);
  // let g = f32(0);
  // let b = f32(0);

  for (var yi = -20; yi <= 20; yi++) {
    for (var xi = -20; xi <= 20; xi++) {
      let ox = i32(origin_x) + xi;
      let oy = i32(origin_y) + yi;
      if (ox < 0 || ox > ${width} || oy < 0 || oy > ${height}) {
        continue;
      }
      let out = ox + oy * ${width};
      let dx = f32(ox) - x;
      let dy = f32(oy) - y;

      if ((mode % 2) == 1) {
        let d = sqrt(dx * dx + dy * dy);
        if (abs(d - ${radius}) < 1) {
          image_map[4 * out + 0] = r;
          image_map[4 * out + 1] = g;
          image_map[4 * out + 2] = b;
          image_map[4 * out + 3] = f32(1);
          continue;
        }
      }
      if ((mode / 2) %2 == 1) {
        let line_thickness = f32(0.5);

        if (right_is_valid && f32(ox) > x) {
          // check if (ox, oy) is close to the line between current and right
          var a: f32 = 0;
          var b: f32 = 0;
          var c: f32 = 0;

          if (right_x != x) {
            a = -1.0;
            b = (y - right_y) / (x - right_x);
            c = y - b * x;
          } else if (right_y != y) {
            a = (x - right_x) / (y - right_y);
            b = -1.0;
            c = x - a * y;
          }

          let d_l = abs(b * f32(ox) + a * f32(oy) + c)/sqrt(a * a + b * b);
          if (d_l < line_thickness) {
            image_map[4 * out + 0] = r;
            image_map[4 * out + 1] = g;
            image_map[4 * out + 2] = b;
            image_map[4 * out + 3] = f32(1);
            continue;
          }
        }
        if (down_is_valid && f32(oy) > y) {
          // check if (ox, oy) is close to the line between current and down
          var a: f32 = 0;
          var b: f32 = 0;
          var c: f32 = 0;

          if (down_x != x) {
            a = -1.0;
            b = (y - down_y) / (x - down_x);
            c = y - b * x;
          } else if (down_y != y) {
            a = (x - down_x) / (y - down_y);
            b = -1.0;
            c = x - a * y;
          }

          let d_l = abs(b * f32(ox) + a * f32(oy) + c)/sqrt(a * a + b * b);
          if (d_l < line_thickness) {
            image_map[4 * out + 0] = r;
            image_map[4 * out + 1] = g;
            image_map[4 * out + 2] = b;
            image_map[4 * out + 3] = f32(1);
            continue;
          }
        }
        if (diag_is_valid && f32(ox) > x && f32(oy) > y) {
          // check if (ox, oy) is close to the line between current and diag
          var a: f32 = 0;
          var b: f32 = 0;
          var c: f32 = 0;

          if (diag_x != x) {
            a = -1.0;
            b = (y - diag_y) / (x - diag_x);
            c = y - b * x;
          } else if (diag_y != y) {
            a = (x - diag_x) / (y - diag_y);
            b = -1.0;
            c = x - a * y;
          }

          let d_l = abs(b * f32(ox) + a * f32(oy) + c)/sqrt(a * a + b * b);
          if (d_l < line_thickness) {
            image_map[4 * out + 0] = r;
            image_map[4 * out + 1] = g;
            image_map[4 * out + 2] = b;
            image_map[4 * out + 3] = f32(1);
            continue;
          }
        }
      }
    }
  }
}
`
}
