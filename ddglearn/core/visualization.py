"""Mesh visualization utilities — save animations as standalone HTML.

Uses meshplot for 3D rendering. Generates self-contained HTML files
with a slider that works in any browser (no Jupyter required).
"""
import numpy as np
from typing import Optional
import sys


def _ensure_display():
    """Ensure `display` builtin exists (meshplot needs it even outside Jupyter)."""
    if "display" not in dir(__builtins__) if isinstance(__builtins__, dict) else not hasattr(__builtins__, "display"):
        import builtins
        builtins.display = lambda *a, **kw: None


def save_mesh_animation(
    vertices: np.ndarray,
    faces: np.ndarray,
    frames: np.ndarray,
    filename: str = "animation.html",
    title: str = "Simulation",
    colormap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fps: int = 10,
):
    """Save a vertex-color animation as a standalone HTML file.

    Args:
        vertices: (n_verts, 3) vertex positions.
        faces: (n_faces, 3) face indices.
        frames: (n_frames, n_verts) per-frame vertex colors.
        filename: output HTML path.
        title: title displayed above the viewer.
        colormap: meshplot/Plotly colormap name.
        vmin / vmax: color range (auto-detected if None).
        fps: playback frames per second for auto-play.
    """
    _ensure_display()
    import meshplot as mp

    if vmin is None:
        vmin = float(frames.min())
    if vmax is None:
        vmax = float(frames.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    n_frames = len(frames)
    n_verts = len(vertices)
    n_faces = len(faces)

    v_list = vertices.tolist()
    f_list = faces.tolist()
    c_list = [((frames[i] - vmin) / (vmax - vmin)).tolist() for i in range(n_frames)]

    # Build a standalone HTML file with three.js via meshplot
    # We embed one meshplot viewer and swap vertex colors via JS
    p = mp.plot(vertices, faces, c=frames[0],
                shading={"colormap": colormap, "xmin": vmin, "xmax": vmax},
                return_plot=True)

    base_html = p.to_html()

    # Inject slider + JS color-swap logic
    frames_json = str(c_list).replace("Infinity", "1e308")
    slider_html = f"""
<div id="sim-controls" style="margin:12px 0; font-family:monospace;">
  <b>{title}</b><br>
  <input type="range" id="frame-slider" min="0" max="{n_frames - 1}" value="0"
         style="width:80%;">
  <span id="frame-label">frame 0 / {n_frames - 1}</span>
  <br>
  <button id="play-btn" style="margin-top:4px;">&#9654; Play</button>
</div>
<script>
(function() {{
  var frames = {frames_json};
  var nFrames = {n_frames};
  var fps = {fps};
  var playing = false;
  var timer = null;
  var slider = document.getElementById("frame-slider");
  var label  = document.getElementById("frame-label");
  var btn    = document.getElementById("play-btn");

  function setFrame(idx) {{
    slider.value = idx;
    label.textContent = "frame " + idx + " / " + (nFrames - 1);
    // meshplot stores the mesh in a global scope; find the geometry
    var geoms = [];
    // Walk the scene to find mesh objects
    if (typeof m !== "undefined" && m.scene) {{
      m.scene.traverse(function(obj) {{
        if (obj.geometry && obj.geometry.attributes && obj.geometry.attributes.color) {{
          geoms.push(obj);
        }}
      }});
    }}
    geoms.forEach(function(meshObj) {{
      var colors = frames[idx];
      var buf = meshObj.geometry.attributes.color;
      for (var i = 0; i < colors.length; i++) {{
        buf.array[i] = colors[i];
      }}
      buf.needsUpdate = true;
    }});
  }}

  slider.oninput = function() {{ setFrame(parseInt(this.value)); }};

  btn.onclick = function() {{
    if (playing) {{
      clearInterval(timer);
      btn.innerHTML = "&#9654; Play";
      playing = false;
    }} else {{
      playing = true;
      btn.innerHTML = "&#9646;&#9646; Pause";
      timer = setInterval(function() {{
        var cur = parseInt(slider.value);
        var next = (cur + 1) % nFrames;
        setFrame(next);
      }}, 1000 / fps);
    }}
  }};
}})();
</script>
"""
    # Inject slider before closing </body>
    if "</body>" in base_html:
        full_html = base_html.replace("</body>", slider_html + "</body>")
    else:
        full_html = base_html + slider_html

    with open(filename, "w") as fh:
        fh.write(full_html)

    print(f"  Saved animation: {filename} ({n_frames} frames)")


def save_mesh_snapshot(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    filename: str = "snapshot.html",
    title: str = "",
    colormap: str = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Save a single colored mesh as a standalone HTML file."""
    _ensure_display()
    import meshplot as mp

    if vmin is None:
        vmin = float(colors.min())
    if vmax is None:
        vmax = float(colors.max())

    p = mp.plot(vertices, faces, c=colors,
                shading={"colormap": colormap, "xmin": vmin, "xmax": vmax},
                return_plot=True)
    html = p.to_html()

    if title:
        title_html = f"<h3 style='font-family:monospace;margin:8px 0;'>{title}</h3>"
        if "<body>" in html:
            html = html.replace("<body>", f"<body>{title_html}")
        else:
            html = title_html + html

    with open(filename, "w") as fh:
        fh.write(html)

    print(f"  Saved snapshot: {filename}")


__all__ = ["save_mesh_animation", "save_mesh_snapshot"]
