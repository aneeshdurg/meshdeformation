import { MeshDeformation } from './meshdeformation';

let stop = false;

async function setupWebcam() {
  const video = document.createElement("video");
  const constraints = { video: true }

  try {
    if (video.srcObject) {
      const stream = video.srcObject;
      stream.getTracks().forEach(function(track: any) {
        track.stop();
      });
      video.srcObject = null;
    }

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    video.play();
  } catch (err) {
    alert("Error initializing webcam! " + err);
    console.log(err);
  }
  return video;
}

async function main(container: HTMLElement) {
  let canvas = (document.createElement("canvas") as HTMLCanvasElement);
  canvas.width = 1000;
  canvas.height = 1000;
  container.appendChild(canvas);

  // let ctx = canvas.getContext("2d");
  let ctx = canvas.getContext("webgpu");
  console.log("Created context for main canvas");

  let canvas2 = document.createElement("canvas") as HTMLCanvasElement;
  canvas2.width = 1000;
  canvas2.height = 1000;
  let ctx2 = canvas2.getContext("2d");
  // document.body.appendChild(canvas2);

  canvas.addEventListener("click", (e) => {
    let el = e.target as HTMLCanvasElement;
    const rect = el.getBoundingClientRect();
    const x = el.width * (e.clientX - rect.left) / rect.width;
    const y = el.height * (e.clientY - rect.top) / rect.height;

    ctx2.beginPath();
    ctx2.fillStyle = "black";
    ctx2.arc(x, y, 100, 0, 2 * Math.PI);
    ctx2.fill();
  });

  container.appendChild(document.createElement("br"));

  const clear = document.createElement("button");
  clear.innerText = "clear";
  clear.addEventListener("click", () => {
    ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
  });
  container.appendChild(clear);

  const edges = document.createElement("button");
  edges.innerText = "edges";
  edges.addEventListener("click", () => {
    md.draw_edges = !md.draw_edges;
  });
  container.appendChild(edges);

  const video = await setupWebcam();

  console.log("Created context for interactive canvas");

  (window as any).n_steps_per_frame = 1;

  let n_elems = 200;
  let spacing = ctx.canvas.width / n_elems;
  let md = new MeshDeformation(ctx, n_elems, n_elems, spacing, spacing / 4, spacing * 4, 1);
  (window as any).t_per_render = 0;
  (window as any).n_renders = 0;
  (window as any).write_time = 0;
  (window as any).interval = 0;
  let theta = 0;
  let last_start = 0;
  md.initialization_done.then(() => {
    const f = async () => {
      let start = performance.now();
      for (let i = 0; i < (window as any).n_steps_per_frame; i++) {
        await md.applyForce(ctx2);
      }
      let end = performance.now();
      (window as any).t_per_render += end - start;
      (window as any).n_renders += 1;


      if (video.readyState == 4) {
        ctx2.drawImage(video, 0, 0, ctx2.canvas.width, ctx2.canvas.height);
      }
      // ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
      let x = Math.sin(theta) * 450 + 500;
      let y = Math.cos(theta) * 450 + 500;
      ctx2.beginPath();
      ctx2.fillStyle = "black";
      ctx2.arc(x, y, 100, 0, 2 * Math.PI);
      ctx2.fill();

      theta += 0.1;

      (window as any).interval += start - last_start;
      last_start = start;
      if (!stop) {
        requestAnimationFrame(f)
      }
    };
    requestAnimationFrame(f);

    (window as any).t_per_draw = 0;
    (window as any).t_per_read = 0;
    (window as any).n_draws = 0;
  });

  (window as any).stats = () => {
    let w = window as any;
    console.log("avg_interval", w.interval / w.n_renders);
    console.log("avg_t_per_render", w.t_per_render / w.n_renders);
    console.log("avg_t_per_write", w.write_time / w.n_renders);
    console.log("avg_t_per_draw", w.t_per_draw / w.n_draws);
    console.log("avg_t_per_read", w.t_per_read / w.n_draws);
  }


  function cancel() {
    stop = true;
  }
  (window as any).md = md;
  (window as any).ctx2 = ctx2;
  (window as any).cancel = cancel;
}

document.addEventListener("DOMContentLoaded", () => {
  let container = document.getElementById("container");
  main(container);
});

