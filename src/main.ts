import { MeshDeformation } from './meshdeformation';
import * as HME from "h264-mp4-encoder";

let stop = false;

function _download(data_blob, filename) {
  const downloader = document.createElement('a');
  downloader.setAttribute('href', URL.createObjectURL(data_blob));
  downloader.setAttribute('download', filename);
  downloader.style.display = "none";
  document.body.appendChild(downloader);

  downloader.click();

  document.body.removeChild(downloader);
}

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

  const circles = [];

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

  const encoder = await HME.createH264MP4Encoder()
  encoder.width = 1000;
  encoder.height = 1000;
  encoder.initialize();

  let frame_cnt = 0;
  let n_frames = 300;

  let record_canvas = document.createElement("canvas");
  record_canvas.width = 1000;
  record_canvas.height = 1000;
  let record_ctx = record_canvas.getContext("2d");

  let record_frame = () => {
    if (frame_cnt < n_frames) {
      record_ctx.clearRect(0, 0, record_ctx.canvas.width, record_ctx.canvas.height);
      record_ctx.fillStyle = "white";
      record_ctx.rect(0, 0, record_ctx.canvas.width, record_ctx.canvas.height);
      record_ctx.fill();
      record_ctx.drawImage(canvas, 0, 0, record_canvas.width, record_canvas.height);
      encoder.addFrameRgba(
        record_ctx.getImageData(0, 0, record_canvas.width, record_canvas.height).data
      );
      frame_cnt += 1;
      console.log(frame_cnt);
      if (frame_cnt == n_frames) {
        encoder.finalize();
        const data = encoder.FS.readFile(encoder.outputFilename);
        const blob = new Blob([new Uint8Array(data)], { type: 'octet/stream' });
        _download(blob, "raindrop_large.mp4");
      }
    }
  };

  let n_elems = 200;
  let spacing = ctx.canvas.width / n_elems;
  let md = new MeshDeformation(ctx, n_elems, n_elems, spacing, spacing / 4, spacing * 4, 1);
  (window as any).t_per_render = 0;
  (window as any).n_renders = 0;
  (window as any).write_time = 0;
  (window as any).interval = 0;
  let theta = 0;
  let last_start = 0;
  // md.draw_edges = false;
  md.initialization_done.then(() => {
    const f = async () => {
      let start = performance.now();
      for (let i = 0; i < (window as any).n_steps_per_frame; i++) {
        await md.applyForce(ctx2);
      }
      let end = performance.now();
      (window as any).t_per_render += end - start;
      (window as any).n_renders += 1;

      if (Math.random() < 0.5) {
        let x = Math.random() * ctx2.canvas.width;
        let y = Math.random() * ctx2.canvas.width;
        let r = Math.random() * 50 + 50;
        if (circles.length > 10 && Math.random() < 0.75) {
          circles.shift();
        }
        circles.push([x, y, r]);
      }
      ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
      for (let circle of circles) {
        let x = circle[0];
        let y = circle[1];
        let r = circle[2];
        ctx2.beginPath();
        ctx2.fillStyle = "black";
        ctx2.arc(x, y, r, 0, 2 * Math.PI);
        ctx2.fill();
      }

      // if (video.readyState == 4) {
      //   ctx2.drawImage(video, 0, 0, ctx2.canvas.width, ctx2.canvas.height);
      // }
      // ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
      // let x = Math.sin(theta) * 450 + 500;
      // let y = Math.cos(theta) * 450 + 500;
      // ctx2.beginPath();
      // ctx2.fillStyle = "black";
      // ctx2.arc(x, y, 100, 0, 2 * Math.PI);
      // ctx2.fill();

      theta += 0.1;

      record_frame();

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

