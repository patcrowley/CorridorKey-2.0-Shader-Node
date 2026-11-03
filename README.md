# CorridorKey 2.0 for Blender

A Blender add-on that makes Corridor Digital's CorridorKey AI keyer work as a **shader node**. A footage plane shows its key in the 3D viewport and in renders. You get live previews, a cache on the timeline, F12 keying, SAM 3 alpha hints, a green/blue screen key and render-farm locking.

> Not made by or affiliated with Corridor Digital. CorridorKey is theirs; this add-on drives either their paid Blender add-on or the free, open-source CorridorKey 2.0.

![A green-screen plate keyed live on a plane in Blender's viewport, with the CorridorKey 2.0 sidebar, the rendered key on the timeline and the node in the Shader Editor](docs/images/overview.png)

## Features

- **Shader node:** add it from **Add > Texture > CorridorKey 2.0**. It has Image and Alpha Hint inputs, and Color and Alpha outputs.

  ![The CorridorKey 2.0 node between the footage and matte textures and the material's shaders](docs/images/nodes.png)
- **One-click Set Up:** right-click a footage plane and choose **Set Up CorridorKey 2.0**. It wires the node between the footage and the shader, and uses the plane's matte as the Alpha Hint. It also turns off a compositor key (Jetset/Autoshot) that would cover CorridorKey's.
- **Controls in the 3D viewport sidebar:** press N and open the **CorridorKey 2.0** tab. A Getting Started checklist covers the engine, footage, saved file and Render Folder, and warnings catch common problems: full paths, missing files, compositing.
- **Previews:**
  - Preview Frame, plus **Live**, which keys each frame you stop on.
  - **Quick Preview** for speed.
  - **Cache Ahead**, which keys frames outward from the playhead while Live is on, like After Effects.
  - The model stays loaded between previews, so later frames take about a second.
- **Timeline bars:** blue for the rendered key (it grows frame by frame while a render runs), green for previews, orange for frames being keyed.

  ![The blue rendered-key bar across the whole shot on Blender's timeline](docs/images/timeline.png)
- **Render box:**
  - **Whole Shot**, or **Some Frames** to re-render or test a stretch, into the current key or a new test folder.
  - F12 keys the current frame first.
  - Delete Render.

  <img src="docs/images/render_box.png" alt="The Render box: Whole Shot or Some Frames, Render Shot, the F12 option, the key's folder, Open Folder, Delete Render and Lock" width="260">
- **Lock for Render Farm:** freezes the finished key into plain Blender nodes, so the material renders the same on machines without the add-on (SheepIt, farms, other computers). Unlock hands it back.
- **SAM 3 alpha hints:** type what to keep (for example `person`) and make a matte for one frame or the whole shot. SAM 3 runs only when needed: it gets the GPU while it runs and frees it when it's done.
- **Screen Key:** a simple keyer for screen that CorridorKey leaves behind. Pick the screen's colour with the eyedropper, or use the Green or Blue preset. **Clean the Key** multiplies the key's alpha by it live, inside the node. **Bake into Alpha Hint** multiplies it into your hint for every frame, so CorridorKey itself treats that screen as background. Tune it with **Show Matte**.
- **Two engines:**
  - **Corridor Add-on (paid):** Corridor Digital's CorridorKey add-on for Blender.
  - **Free (GitHub):** [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0), with a one-click installer (Windows, NVIDIA GPU).

## Requirements

- Blender 5.0 or later
- An NVIDIA GPU for the free engine and SAM 3 (Windows, or Linux with a manual setup)
- One of the two engines above

## Install

1. Download the add-on zip from [`dist/`](dist), for example `CorridorKey_2.0-v2.12.0.zip`. Use that file rather than GitHub's **Code > Download ZIP**, which also packs the README and license. Or zip the add-on files yourself: `__init__.py`, `free_server.py`, `sam3_worker.py`, `screen_key_worker.py`, `blender_manifest.toml`, `QUICKSTART.html`.
2. In Blender, go to **Edit > Preferences > Get Extensions**, open the menu at the top right, and choose **Install from Disk...** with that zip.
3. Open the 3D viewport sidebar (N), then the **CorridorKey 2.0** tab, and click **?** for the quick-start guide.

<img src="docs/images/guide.png" alt="The first page of the quick-start guide that the ? button opens" width="520">

## Files

| File | What it is |
| --- | --- |
| `__init__.py` | The add-on: the node, panels, previews, renders, locking, SAM 3 hints |
| `free_server.py` | Worker that runs the free CorridorKey 2.0 engine with its own Python |
| `sam3_worker.py` | Worker that runs SAM 3 for alpha hints with its own Python |
| `screen_key_worker.py` | Worker that bakes the screen key into the Alpha Hint, in a background Blender |
| `QUICKSTART.html` | The quick-start guide the **?** button opens |
| `blender_manifest.toml` | Blender extension manifest |
| `dist/` | Ready-to-install add-on zips, one per version |
| `docs/images/` | The pictures in this README |

## Third-party software

- **CorridorKey** is by Corridor Digital. The free engine is [nikopueringer/CorridorKey2.0](https://github.com/nikopueringer/CorridorKey2.0), installed separately into its own folder.
- **SAM 3** is by Meta ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)), under the SAM License. The installer downloads it, with its weights, from an ungated mirror ([cubert-gmbh/sam3](https://huggingface.co/cubert-gmbh/sam3)). Neither is included in this repository.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
