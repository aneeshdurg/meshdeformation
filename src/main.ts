import { MeshDeformation } from './meshdeformation';

let stop = false;

document.addEventListener("DOMContentLoaded", () => {
  let canvas = (document.getElementById("mycanvas") as HTMLCanvasElement);
  canvas.width = 1000;
  canvas.height = 1000;

  let ctx = canvas.getContext("2d");
  console.log("Created context for main canvas");

  let canvas2 = document.getElementById("mycanvas2") as HTMLCanvasElement;
  canvas2.width = 1000;
  canvas2.height = 1000;
  let ctx2 = canvas2.getContext("2d");
  canvas2.addEventListener("click", (e) => {
    let el = e.target as HTMLCanvasElement;
    const rect = el.getBoundingClientRect();
    const x = el.width * (e.clientX - rect.left) / rect.width;
    const y = el.height * (e.clientY - rect.top) / rect.height;

    ctx2.beginPath();
    ctx2.fillStyle = "black";
    ctx2.arc(x, y, 100, 0, 2 * Math.PI);
    ctx2.fill();
  });

  console.log("Created context for interactive canvas");

  (window as any).n_steps_per_frame = 1;

  let md = new MeshDeformation(ctx, 25, 25, ctx.canvas.width / 25, 10, 100, 5);
  (window as any).t_per_render = 0;
  (window as any).n_renders = 0;
  md.initialization_done.then(() => {
    const f = async () => {
      let start = performance.now();
      md.draw();
      for (let i = 0; i < (window as any).n_steps_per_frame; i++) {
        await md.applyForce(ctx2);
        // await md.updateCPUpos();
      }
      await md.device.queue.onSubmittedWorkDone();
      let end = performance.now();
      (window as any).t_per_render += end - start;
      (window as any).n_renders += 1;
      if (!stop) {
        requestAnimationFrame(f)
        // setTimeout(() => {
        //   requestAnimationFrame(f)
        // }, 1);
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
    console.log("t_per_render", window.t_per_render);
    console.log("n_renders", window.n_renders);
    console.log("avg", window.t_per_render / window.n_renders);
    console.log("t_per_draw", window.t_per_draw);
    console.log("n_draws", window.n_draws);
    console.log("avg", window.t_per_draw / window.n_draws);
  }


  function cancel() {
    stop = true;
  }
  (window as any).md = md;
  (window as any).ctx2 = ctx2;
  (window as any).cancel = cancel;
});

