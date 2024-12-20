import { MeshDeformation } from './meshdeformation';

let stop = false;

document.addEventListener("DOMContentLoaded", () => {
  let canvas = (document.getElementById("mycanvas") as HTMLCanvasElement);
  canvas.width = 1000;
  canvas.height = 1000;

  let ctx = canvas.getContext("2d");
  console.log("Created context for main canvas");

  let canvas2 = document.createElement("canvas") as HTMLCanvasElement;
  canvas2.width = 1000;
  canvas2.height = 1000;
  let ctx2 = canvas2.getContext("2d");

  // let idata: Uint8ClampedArray<ArrayBufferLike> = new Uint8ClampedArray(ctx2.canvas.width * ctx.canvas.height * 4);
  canvas.addEventListener("click", (e) => {
    let el = e.target as HTMLCanvasElement;
    const rect = el.getBoundingClientRect();
    const x = el.width * (e.clientX - rect.left) / rect.width;
    const y = el.height * (e.clientY - rect.top) / rect.height;

    ctx2.beginPath();
    ctx2.fillStyle = "black";
    ctx2.arc(x, y, 100, 0, 2 * Math.PI);
    ctx2.fill();

    // idata = ctx2.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height).data;
  });

  document.getElementById("clear").addEventListener("click", () => {
    ctx2.clearRect(0, 0, ctx2.canvas.width, ctx2.canvas.height);
  });

  document.getElementById("edges").addEventListener("click", () => {
    md.draw_edges = !md.draw_edges;
  });

  console.log("Created context for interactive canvas");

  (window as any).n_steps_per_frame = 1;

  let n_elems = 100;
  let spacing = ctx.canvas.width / n_elems;
  let md = new MeshDeformation(ctx, n_elems, n_elems, spacing, spacing / 4, spacing * 4, 1);
  (window as any).t_per_render = 0;
  (window as any).n_renders = 0;
  (window as any).write_time = 0;
  md.initialization_done.then(() => {
    const f = async () => {
      let start = performance.now();
      for (let i = 0; i < (window as any).n_steps_per_frame; i++) {
        await md.applyForce(ctx2);
      }
      let end = performance.now();
      (window as any).t_per_render += end - start;
      (window as any).n_renders += 1;
      if (!stop) {
        requestAnimationFrame(f)
      }
    };
    requestAnimationFrame(f);

    (window as any).t_per_draw = 0;
    (window as any).n_draws = 0;
    const g = async () => {
      let start = performance.now();
      await md.updateCPUpos();
      md.draw();
      let end = performance.now();
      (window as any).t_per_draw += end - start;
      (window as any).n_draws += 1;
      setTimeout(() => {
        requestAnimationFrame(g)
      }, 30);
    };
    requestAnimationFrame(g);
  });

  (window as any).stats = () => {
    let w = window as any;
    console.log("t_per_render", w.t_per_render);
    console.log("n_renders", w.n_renders);
    console.log("avg", w.t_per_render / w.n_renders);
    console.log("t_per_write", w.write_time);
    console.log("  avg", w.write_time / w.n_renders);
    console.log("t_per_draw", w.t_per_draw);
    console.log("n_draws", w.n_draws);
    console.log("avg", w.t_per_draw / w.n_draws);
  }


  function cancel() {
    stop = true;
  }
  (window as any).md = md;
  (window as any).ctx2 = ctx2;
  (window as any).cancel = cancel;
});

