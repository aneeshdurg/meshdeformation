import { MeshDeformation } from './script';

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

  let md = new MeshDeformation(ctx, 20, 20, ctx.canvas.width / 20, 10, 5);
  md.initialization_done.then(() => {
    const f = async () => {
      md.draw();
      await md.applyForce(ctx2);
      await md.updateCPUpos();
      if (!stop) {
        setTimeout(() => {
          requestAnimationFrame(f)
        }, 1);
      }
    };
    requestAnimationFrame(f);
  });


  function cancel() {
    stop = true;
  }
  (window as any).md = md;
  (window as any).ctx2 = ctx2;
  (window as any).cancel = cancel;
});

