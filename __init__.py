"""CorridorKey 2.0: CorridorKey in the Shader Editor and the 3D viewport (Add > Texture > CorridorKey 2.0).

Shader nodes can't run CorridorKey's model any more than compositor nodes can, so this node works the
way CorridorKey's own compositor node does: the key is rendered by CorridorKey's runtime, outside
Blender, and the node reads the finished frames through an Image Texture inside its group.

Each shader node drives a CorridorKey compositor node kept in a hidden compositor group
(".CorridorKey Shader Sources"). That is what lets every CorridorKey control, and its Render button,
appear in the Shader Editor's sidebar unchanged. Until a render lands, the node shows the footage.

Like Corridor's node, the shader node is a plain Blender group: a .blend that uses it opens and
renders its last key on any Blender, a render farm's included, without this add-on.

Two engines, chosen in the add-on's preferences (Automatic picks the paid one when it's enabled):
- Corridor Digital's paid CorridorKey add-on, whose own controls and Render are used as-is;
- the free CorridorKey 2.0 from GitHub (github.com/nikopueringer/CorridorKey2.0), run from its own
  folder and Python through free_server.py, with this add-on's own controls and Render.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import textwrap
import sys
import tempfile
import threading
import time
import types
import uuid

import bpy
from bpy.app.handlers import persistent

NAME = "CorridorKey 2.0"
MARK = "corridorkey_shader"  # on the shader group's inner tree
SOURCE = "corridorkey_source"  # on the inner tree: the uid of the compositor node it shows
SOURCES = ".CorridorKey Shader Sources"  # the hidden compositor group holding those nodes
TEXTURE = "CorridorKey Frames"  # the Image Texture inside the shader group
RESULT = "CorridorKey Result"  # Corridor's name for the Image node holding a finished render
SEEN = "corridorkey_shader_seen"  # on a footage node: the footage it was last placed for
SOURCE_FREE = "corridorkey_source_free"  # on the inner tree: the uid of its free-engine source
FREE_MARK = "corridorkey2_free"  # on a free source's inner tree
FREE_REPO = "https://github.com/nikopueringer/CorridorKey2.0"
ADDON = __package__
_NUMBER = re.compile(r"(\d+)(?=\.[^.]+$)")
_TICK_S = 0.25
_PLATES = (".exr", ".png", ".jpg", ".jpeg")
_MOVIES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mxf", ".m4v")
_SETTLE_S = 0.4  # Live: how long the playhead rests on a frame before it is keyed

# Previews are kept on disk (next to the .blend, by settings) so they outlive the session; what is
# being keyed right now and the panel's line live only in this session.
PREVIEW_JOBS: dict[str, dict] = {}  # source uid -> the preview being keyed
PREVIEW_LINE: dict[str, str] = {}  # source uid -> what the preview row says
RANGE_LINES: set[str] = set()  # source uids whose PREVIEW_LINE is a range render's (the Render box's)
_SETTLING: dict[str, tuple] = {}  # source uid -> (frame, since) for Live
MARKS: dict = {"key": [], "preview": [], "busy": []}  # the timeline's spans, in scene frames
_RENDERING_WITH_KEY = [False]  # an F12 render that keyed its frame first: keep that key in


# --- CorridorKey's own add-on --------------------------------------------------------------------


def ck(sub: str = ""):
    """CorridorKey's add-on module (or one of its modules), or None while it isn't enabled."""
    for name in bpy.context.preferences.addons.keys():
        if name == "corridorkey" or name.endswith(".corridorkey"):
            try:
                return importlib.import_module(f"{name}.{sub}" if sub else name)
            except ImportError:
                return None
    return None


# --- engines: Corridor's paid add-on, or the free CorridorKey 2.0 from GitHub ----------------------


def prefs():
    addon = bpy.context.preferences.addons.get(ADDON)
    return addon.preferences if addon is not None else None


def engine() -> str:
    """'paid' or 'free'."""
    chosen = getattr(prefs(), "engine", "AUTO")
    if chosen == "PAID":
        return "paid"
    if chosen == "FREE":
        return "free"
    return "paid" if ck() is not None else "free"


def engine_label() -> str:
    return "Corridor's CorridorKey add-on" if engine() == "paid" else "CorridorKey 2.0 (free, GitHub)"


_FREE_WEIGHTS = (
    ("weights/corridorkey_v2.safetensors", "weights/corridorkey_v2.pt"),
    ("weights/foundation_stats.safetensors", "weights/foundation_stats.pt"),
    ("vfms/rvm/pretrain_rvm_small16_256_204031069.npz",),
    ("vfms/moge/model.pt",),
    ("vfms/cradio/C-RADIOv4-SO400M",),
)
_FREE_GUESSES = ("CorridorKey2.0", "CorridorKey2.0-master", "CorridorKey 2.0", "CorridorKey2")


def _is_free_root(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "infer.py")) and os.path.isdir(
        os.path.join(path, "corridorkey")
    )


def free_root() -> str:
    """The free CorridorKey 2.0 folder: the one in the preferences, else a usual place."""
    chosen = getattr(prefs(), "free_root", "")
    if chosen:
        return os.path.normpath(bpy.path.abspath(chosen))
    found, when = _GUESSED
    if time.monotonic() - when < 5.0:
        return found
    _GUESSED[:] = [_guess_free_root(), time.monotonic()]
    return _GUESSED[0]


_GUESSED: list = ["", -100.0]  # the last search for an install, and when


def _guess_free_root() -> str:
    home = os.path.expanduser("~")
    for base in (home, os.path.join(home, "Downloads"), os.path.join(home, "Documents"),
                 os.path.join(home, "Desktop"), "C:\\" if sys.platform == "win32" else "/opt"):
        for name in _FREE_GUESSES:
            candidate = os.path.join(base, name)
            if _is_free_root(candidate):
                return candidate
    return ""


def free_python(root: str) -> str:
    chosen = getattr(prefs(), "free_python", "")
    if chosen:
        return os.path.normpath(bpy.path.abspath(chosen))
    for parts in (("venv", "Scripts", "python.exe"), (".venv", "Scripts", "python.exe"),
                  ("venv", "bin", "python"), (".venv", "bin", "python")):
        candidate = os.path.join(root, *parts)
        if os.path.isfile(candidate):
            return candidate
    return ""


def free_setup_problem() -> str | None:
    """Why the free engine can't run yet, in a line, or None."""
    if sys.platform == "darwin":
        return "The free CorridorKey 2.0 needs an NVIDIA GPU (Windows or Linux)."
    root = free_root()
    if not root:
        return "Install the free CorridorKey 2.0, or set its folder in the preferences."
    if not _is_free_root(root):
        return "The folder in the preferences isn't CorridorKey 2.0."
    if not os.path.isfile(free_python(root)):
        return "CorridorKey 2.0 has no venv yet: run Install in the preferences."
    for options in _FREE_WEIGHTS:
        if not any(os.path.exists(os.path.join(root, o)) for o in options):
            return "CorridorKey 2.0's weights are missing: run Install in the preferences."
    return None


def engine_available() -> bool:
    return ck() is not None if engine() == "paid" else free_setup_problem() is None


def _engine_changed(_self, _context) -> None:
    close_warm()


# --- the hidden compositor sources -----------------------------------------------------------------


def is_free_source(node) -> bool:
    return (
        node is not None
        and node.bl_idname == "CompositorNodeGroup"
        and node.node_tree is not None
        and bool(node.node_tree.get(FREE_MARK))
    )


def is_source(node) -> bool:
    if is_free_source(node):
        return bool(node.corridorkey2.uid)
    ck_node = ck("node")
    return ck_node is not None and ck_node.is_corridorkey(node)


def st(src):
    """A source's settings: the free engine's, or Corridor's own."""
    return src.corridorkey2 if is_free_source(src) else src.corridorkey


def iter_sources():
    for tree in bpy.data.node_groups:
        if tree.bl_idname == "CompositorNodeTree":
            for node in tree.nodes:
                if is_source(node):
                    yield node


def node_by_uid(uid: str):
    return next((n for n in iter_sources() if uid and st(n).uid == uid), None) if uid else None


def sources_tree() -> bpy.types.NodeTree:
    tree = bpy.data.node_groups.get(SOURCES)
    if tree is None or tree.bl_idname != "CompositorNodeTree":
        tree = bpy.data.node_groups.new(SOURCES, "CompositorNodeTree")
    tree.use_fake_user = True  # kept in the .blend though nothing uses it as a group
    return tree


def wired_image_node(src: bpy.types.Node, index: int) -> bpy.types.Node | None:
    links = src.inputs[index].links if len(src.inputs) > index else ()
    upstream = links[0].from_node if links else None
    if upstream is not None and upstream.bl_idname == "CompositorNodeImage":
        return upstream
    return None


def ensure_inputs(src: bpy.types.Node) -> None:
    """The Footage and Alpha Hint Image nodes wired into a source; put back if missing."""
    tree = src.id_data
    for index, label in ((0, "Footage"), (1, "Alpha Hint")):
        if wired_image_node(src, index) is None:
            image_node = tree.nodes.new("CompositorNodeImage")
            image_node.label = label
            image_node.location = (src.location.x - 300, src.location.y - 220 * index)
            tree.links.new(image_node.outputs["Image"], src.inputs[index])


def new_free_source() -> bpy.types.Node:
    """A free-engine source: a plain compositor group of this add-on's own, laid out like Corridor's
    (Image and Alpha Hint in, the footage passed through until a render lands), so it renders on
    any Blender."""
    tree = sources_tree()
    row = sum(1 for n in tree.nodes if is_source(n))
    inner = bpy.data.node_groups.new(".CorridorKey 2.0 Free", "CompositorNodeTree")
    inner[FREE_MARK] = 1
    inner.interface.new_socket("Image", in_out="INPUT", socket_type="NodeSocketColor")
    inner.interface.new_socket("Alpha Hint", in_out="INPUT", socket_type="NodeSocketColor")
    inner.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    group_in = inner.nodes.new("NodeGroupInput")
    group_out = inner.nodes.new("NodeGroupOutput")
    group_out.location = (400, 0)
    inner.links.new(group_in.outputs[0], group_out.inputs[0])
    src = tree.nodes.new("CompositorNodeGroup")
    src.node_tree = inner
    src.label = f"{NAME} (free)"
    src.width = 200
    src.location = (300, -450 * row)
    src.corridorkey2.uid = uuid.uuid4().hex
    ensure_inputs(src)
    if bpy.data.filepath:
        src.corridorkey2.render_folder = "//CorridorKey/"
    return src


DESPILL = 0.7  # this add-on's default despill, for either engine


def despill_default(src) -> None:
    """Corridor's node keeps its own despill default: set ours on it (on its scale, 0-1 or a
    percentage). The free engine's setting defaults to DESPILL already."""
    if is_free_source(src):
        return
    state = st(src)
    prop = state.bl_rna.properties.get("despill_amount")
    if prop is None or prop.type not in {"FLOAT", "INT"}:
        return
    value = DESPILL * 100 if prop.subtype == "PERCENTAGE" or prop.hard_max > 1.5 else DESPILL
    value = min(max(value, prop.hard_min), prop.hard_max)
    try:
        setattr(state, "despill_amount", round(value) if prop.type == "INT" else value)
    except (TypeError, AttributeError):
        pass


def new_source(kind: str | None = None) -> bpy.types.Node:
    """A source for `kind` ('paid' or 'free'; the current engine when None)."""
    if (kind or engine()) == "free":
        return new_free_source()
    ck_node, ck_ops = ck("node"), ck("operators")
    tree = sources_tree()
    row = sum(1 for n in tree.nodes if ck_node.is_corridorkey(n))
    src = ck_node.add_corridorkey_node(tree)
    src.location = (300, -450 * row)
    ensure_inputs(src)
    if bpy.data.filepath:  # a saved file gets a Render Folder beside it; change it in the sidebar
        src.corridorkey.render_folder = "//CorridorKey/"
    despill_default(src)
    if hasattr(ck_ops, "start_check"):
        ck_ops.start_check()
    return src


def source_key(kind: str | None = None) -> str:
    return SOURCE_FREE if (kind or engine()) == "free" else SOURCE


def source_of(node: bpy.types.Node) -> bpy.types.Node | None:
    """The node's source for the current engine."""
    tree = node.node_tree
    if tree is None:
        return None
    if engine() == "paid" and ck() is None:
        return None
    uid = tree.get(source_key())
    return node_by_uid(uid) if uid else None


def adopt_source(node: bpy.types.Node) -> bpy.types.Node | None:
    """Give a node a source for the current engine (a file made with the other engine, or a switch
    of engine): its footage and hint come from what's plugged in, as always."""
    if engine() == "paid" and ck() is None:
        return None
    src = new_source()
    node.node_tree[source_key()] = st(src).uid
    return src


def apply_display(src) -> None:
    if not is_free_source(src):
        ck("node").apply_display(src)
        return
    tree = src.node_tree
    group_in = next((n for n in tree.nodes if n.bl_idname == "NodeGroupInput"), None) or tree.nodes.new("NodeGroupInput")
    group_out = next((n for n in tree.nodes if n.bl_idname == "NodeGroupOutput"), None) or tree.nodes.new("NodeGroupOutput")
    result = tree.nodes.get(RESULT)
    shown = src.corridorkey2.display_rendered_frames and result is not None
    tree.links.new(result.outputs["Image"] if shown else group_in.outputs[0], group_out.inputs[0])


def footage_length(image: bpy.types.Image) -> tuple[int, int]:
    """(frame count, first frame number) of a movie, an image sequence or one image."""
    if image.source == "MOVIE":
        return max(image.frame_duration, 1), 1
    if image.source != "SEQUENCE":
        return 1, 1
    path = bpy.path.abspath(image.filepath, library=image.library)
    folder, name = os.path.split(path)
    found = _NUMBER.search(name)
    if not found or not os.path.isdir(folder):
        return max(image.frame_duration, 1), 1
    head, tail = name[: found.start()], name[found.end() :]
    numbers = sorted(
        int(f[len(head) : len(f) - len(tail)])
        for f in os.listdir(folder)
        if f.startswith(head)
        and f.endswith(tail)
        and f[len(head) : len(f) - len(tail)].isdigit()
    )
    return (len(numbers), numbers[0]) if numbers else (1, 1)


def place_footage(image_node: bpy.types.Node, scene: bpy.types.Scene) -> None:
    """Footage picked in the sidebar plays from the scene's first frame, its first plate there."""
    image = image_node.image
    key = f"{image.name}|{image.filepath}|{image.source}" if image is not None else ""
    if image_node.get(SEEN) == key:
        return
    image_node[SEEN] = key
    if image is None:
        return
    duration, first = footage_length(image)
    image_node.frame_start = scene.frame_start
    image_node.frame_offset = first - 1
    image_node.frame_duration = duration
    image_node.use_cyclic = False
    image_node.use_auto_refresh = True


# --- the shader node -------------------------------------------------------------------------------


def is_ck_shader(node: bpy.types.Node | None) -> bool:
    return (
        node is not None
        and node.bl_idname == "ShaderNodeGroup"
        and node.node_tree is not None
        and bool(node.node_tree.get(MARK))
    )


IMAGE_IN, HINT_IN = "Image", "Alpha Hint"  # the node's inputs, the compositor node's words


def build_shader_tree(tree: bpy.types.NodeTree) -> None:
    """The group's sockets and inside; also brings a node made by an earlier version up to date."""
    have = {(i.in_out, i.name) for i in tree.interface.items_tree if i.item_type == "SOCKET"}
    for name, in_out, kind in (
        ("Color", "OUTPUT", "NodeSocketColor"),
        ("Alpha", "OUTPUT", "NodeSocketFloat"),
        (IMAGE_IN, "INPUT", "NodeSocketColor"),
        (HINT_IN, "INPUT", "NodeSocketColor"),
    ):
        if (in_out, name) not in have:
            socket = tree.interface.new_socket(name, in_out=in_out, socket_type=kind)
            if name == "Alpha":
                socket.default_value = 1.0
            elif in_out == "INPUT":
                socket.default_value = (0.0, 0.0, 0.0, 1.0)
    nodes = tree.nodes
    if not any(n.bl_idname == "NodeGroupInput" for n in nodes):
        nodes.new("NodeGroupInput").location = (-450, 200)
    if not any(n.bl_idname == "NodeGroupOutput" for n in nodes):
        nodes.new("NodeGroupOutput").location = (150, 0)
    texture = nodes.get(TEXTURE)
    if texture is None:
        coords = nodes.new("ShaderNodeTexCoord")
        coords.location = (-450, -100)
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = texture.label = TEXTURE
        texture.extension = "CLIP"  # nothing outside the frame
        texture.location = (-200, -100)
        tree.links.new(coords.outputs["UV"], texture.inputs["Vector"])


def new_shader_tree(uid: str, key: str = SOURCE) -> bpy.types.NodeTree:
    tree = bpy.data.node_groups.new(".CorridorKey Shader", "ShaderNodeTree")
    tree[MARK] = 1
    tree[key] = uid
    build_shader_tree(tree)
    wire(tree, "plate")
    return tree


def _link(tree: bpy.types.NodeTree, from_socket, to_socket) -> bool:
    """Wire `to_socket` from `from_socket` (None: unwire it), only when it isn't already."""
    links = to_socket.links
    if from_socket is None:
        for link in list(links):
            tree.links.remove(link)
        return bool(links)
    if links and links[0].from_socket == from_socket:
        return False
    tree.links.new(from_socket, to_socket)
    return True


def wire(tree: bpy.types.NodeTree, mode: str, screen=None) -> bool:
    """What the node puts out. "key": the rendered key. "input": what's plugged in, passed
    through (the Alpha Hint, when plugged in, as its Alpha) so the shot looks as it did until a key
    lands. "plate": the footage picked in the sidebar. `screen` (the node's settings): with Clean
    the Key, the Alpha goes through the screen key; with Show Matte, the screen key is shown."""
    group_in = next(n for n in tree.nodes if n.bl_idname == "NodeGroupInput")
    group_out = next(n for n in tree.nodes if n.bl_idname == "NodeGroupOutput")
    texture = tree.nodes[TEXTURE]
    if mode == "input":
        color, alpha = group_in.outputs[IMAGE_IN], group_in.outputs[HINT_IN]
    else:
        color, alpha = texture.outputs["Color"], texture.outputs["Alpha"]
    changed = False
    keyer = tree.nodes.get(SK_MAP)
    if screen is not None and keyer is not None and (screen.screen_clean or screen.screen_show):
        matte = keyer.outputs[0]
        if screen.screen_show:
            color, alpha = matte, None  # white kept, black taken out; opaque
        else:
            multiply = tree.nodes[SK_MULT]
            changed |= _link(tree, alpha, multiply.inputs[0])
            changed |= _link(tree, matte, multiply.inputs[1])
            alpha = multiply.outputs[0]
    changed |= _link(tree, color, group_out.inputs["Color"])
    changed |= _link(tree, alpha, group_out.inputs["Alpha"])
    return changed


# --- the screen key: a simple green/blue keyer, live in the node, or baked into the Alpha Hint ------

SK_PLATE, SK_GAMMA, SK_SCORE = "CK Screen Footage", "CK Screen Gamma", "CK Screen Score"
SK_MAP, SK_MULT = "CK Screen Matte", "CK Screen Clean"
SK_GAMMA_VALUE = 1 / 2.2  # display-like values, as screen_key_worker.py judges them
SCREEN_GREEN, SCREEN_BLUE = (0.20, 0.62, 0.25), (0.15, 0.25, 0.65)  # typical screens, display-like


def _display(color) -> tuple[float, float, float]:
    return tuple(max(float(c), 0.0) ** SK_GAMMA_VALUE for c in color[:3])


def _linear(color) -> tuple[float, float, float]:
    return tuple(float(c) ** (1 / SK_GAMMA_VALUE) for c in color[:3])


def screen_direction(settings) -> tuple[float, float, float]:
    """The picked screen colour's hue as a unit vector (its display-like colour minus its grey
    level): a pixel's lean towards it is its dot product with this. Grey picks key green."""
    colour = _display(settings.screen_pick)
    mean = sum(colour) / 3
    chroma = [c - mean for c in colour]
    length = sum(c * c for c in chroma) ** 0.5
    if length < 0.02:
        colour = SCREEN_GREEN
        mean = sum(colour) / 3
        chroma = [c - mean for c in colour]
        length = sum(c * c for c in chroma) ** 0.5
    return tuple(c / length for c in chroma)


def _value_input(node, name: str):
    return next(s for s in node.inputs if s.name == name and s.type == "VALUE")


def ensure_screen_nodes(tree) -> bool:
    """The keyer inside the shader group (made once): the footage in display-like values, its
    dot product with the screen colour's direction, a smoothstep from kept to taken out."""
    if tree.nodes.get(SK_MULT) is not None and tree.nodes.get(SK_SCORE) is not None:
        return False
    for old in [n for n in tree.nodes if n.name.startswith("CK Screen")]:  # v2.10's green/blue graph
        tree.nodes.remove(old)
    nodes, links = tree.nodes, tree.links
    coords = next((n for n in nodes if n.bl_idname == "ShaderNodeTexCoord"), None) or nodes.new("ShaderNodeTexCoord")
    plate = nodes.new("ShaderNodeTexImage")
    plate.name = plate.label = SK_PLATE
    plate.extension = "CLIP"
    plate.location = (-200, -420)
    links.new(coords.outputs["UV"], plate.inputs["Vector"])
    gamma = nodes.new("ShaderNodeGamma")
    gamma.name = SK_GAMMA
    gamma.inputs["Gamma"].default_value = SK_GAMMA_VALUE
    gamma.location = (60, -420)
    links.new(plate.outputs["Color"], gamma.inputs["Color"])
    score = nodes.new("ShaderNodeVectorMath")
    score.name, score.operation = SK_SCORE, "DOT_PRODUCT"
    score.location = (260, -420)
    links.new(gamma.outputs["Color"], score.inputs[0])
    matte = nodes.new("ShaderNodeMapRange")
    matte.name, matte.data_type, matte.interpolation_type = SK_MAP, "FLOAT", "SMOOTHSTEP"
    matte.clamp = True
    matte.location = (460, -420)
    links.new(score.outputs["Value"], _value_input(matte, "Value"))
    _value_input(matte, "To Min").default_value = 1.0  # not screen: kept
    _value_input(matte, "To Max").default_value = 0.0  # screen: taken out
    multiply = nodes.new("ShaderNodeMath")
    multiply.name, multiply.operation = SK_MULT, "MULTIPLY"
    multiply.location = (660, -300)
    return True


def screen_update(tree, settings, plate) -> bool:
    """The keyer's footage (the plate, timed as it plays), screen colour and tuning, from the
    node's settings."""
    changed = ensure_screen_nodes(tree)
    nodes = tree.nodes
    footage = nodes[SK_PLATE]
    image = plate.image if plate is not None else None
    changed |= _set(footage, "image", image)
    if image is not None:
        user = footage.image_user
        for attr in ("frame_start", "frame_offset", "frame_duration"):
            changed |= _set(user, attr, getattr(plate, attr))
        changed |= _set(user, "use_auto_refresh", True)
        changed |= _set(user, "use_cyclic", False)
    direction = nodes[SK_SCORE].inputs[1]
    wanted = screen_direction(settings)
    if any(abs(a - b) > 1e-5 for a, b in zip(direction.default_value, wanted)):
        direction.default_value = wanted
        changed = True
    matte = nodes[SK_MAP]
    changed |= _set(_value_input(matte, "From Min"), "default_value", settings.screen_threshold)
    changed |= _set(_value_input(matte, "From Max"), "default_value",
                    settings.screen_threshold + max(settings.screen_softness, 0.001))
    return changed


def upstream_texture(node: bpy.types.Node, name: str) -> tuple[bpy.types.Node | None, bool]:
    """The Image Texture node plugged into input `name` (through reroutes), and whether anything
    is plugged in at all."""
    socket = node.inputs.get(name)
    if socket is None or not socket.is_linked:
        return None, False
    upstream = socket.links[0].from_node
    while upstream is not None and upstream.bl_idname == "NodeReroute":
        links = upstream.inputs[0].links
        upstream = links[0].from_node if links else None
    if upstream is not None and upstream.bl_idname == "ShaderNodeTexImage":
        return upstream, True
    return None, True


def take_wired(texture: bpy.types.Node, image_node: bpy.types.Node) -> None:
    """Hand CorridorKey the plugged-in Image Texture's image, timed as that texture plays it."""
    image, user = texture.image, texture.image_user
    _set(image_node, "image", image)
    if image is None:
        return
    for attr in ("frame_start", "frame_offset", "frame_duration"):
        _set(image_node, attr, getattr(user, attr))
    _set(image_node, "use_cyclic", False)
    _set(image_node, "use_auto_refresh", True)
    key = f"{image.name}|{image.filepath}|{image.source}"
    if image_node.get(SEEN) != key:
        image_node[SEEN] = key  # placed already: the sidebar's placing leaves it alone


def _shader_trees():
    for owners in (bpy.data.materials, bpy.data.worlds, bpy.data.lights, bpy.data.linestyles):
        for owner in owners:
            tree = getattr(owner, "node_tree", None)
            if tree is not None:
                yield tree
    for tree in bpy.data.node_groups:
        if tree.bl_idname == "ShaderNodeTree":
            yield tree


def shader_nodes():
    for tree in _shader_trees():
        for node in tree.nodes:
            if is_ck_shader(node):
                yield node


_LISTED: dict[str, tuple] = {}  # folder -> (its mtime, {frame number: file name}, the numbers)


def _listed(folder: str) -> tuple:
    try:
        stamp = os.stat(folder).st_mtime
    except OSError:
        return (None, {}, set())
    cached = _LISTED.get(folder)
    if cached is None or cached[0] != stamp:
        names = numbered(folder)
        cached = (stamp, names, set(names))
        _LISTED[folder] = cached
    return cached


def _frames_in(folder: str) -> set[int]:
    return _listed(folder)[2]


def _frame_file(folder: str, number: int) -> str | None:
    name = _listed(folder)[1].get(number)
    return os.path.join(folder, name) if name else None


def _has_frame(image_node: bpy.types.Node, frame: int | None) -> bool:
    """Whether a sequence holds the frame the scene is on (a partial render holds only some)."""
    image = image_node.image
    if frame is None or image is None or image.source != "SEQUENCE":
        return True
    folder = os.path.dirname(bpy.path.abspath(image.filepath, library=image.library))
    number = frame - image_node.frame_start + 1 + image_node.frame_offset
    return number in _frames_in(folder)


def shown_frames(src: bpy.types.Node, frame: int | None = None) -> bpy.types.Node | None:
    """What the node shows, as the compositor node does: the key once a render has landed and
    "Display rendered frames?" is ticked, the footage otherwise. On a frame a partial render
    didn't reach, the footage, not Blender's missing-image pink."""
    result = src.node_tree.nodes.get(RESULT) if src.node_tree is not None else None
    if (
        result is not None
        and result.image is not None
        and st(src).display_rendered_frames
        and _has_frame(result, frame)
    ):
        return result
    return wired_image_node(src, 0)


def _set(owner, attr: str, value) -> bool:
    if getattr(owner, attr) != value:
        setattr(owner, attr, value)
        return True
    return False


# --- the warm session: CorridorKey stays loaded between previews -----------------------------------
#
# A preview through Corridor's render job starts the runtime and loads the model every time (most of
# its ~10 s). The runtime's session protocol (the one Corridor's GPU Memory estimate keeps open)
# takes many requests on one session: load once (PrepareModel), then key on demand. Quiet for
# _PARK_S, the session parks (the weights leave the card for system memory, the card goes back to
# Blender); quiet for _CLOSE_S it closes. The worker thread never touches bpy.

_PARK_S = 20.0
_CLOSE_S = 600.0
_LEAVE_S = 10.0
_READY_S = 120.0
_LICENCE_RECHECK_S = 600.0


def _parsed(raw: str):
    try:
        line = json.loads(raw)
    except ValueError:
        return None
    return line if isinstance(line, dict) else None


class Ticket:
    """One preview's request to the warm session; read from the main thread like a render job."""

    def __init__(self, source, render_folder: str, controls, plan, leaving=()) -> None:
        self.source, self.render_folder, self.controls, self.plan = source, render_folder, controls, plan
        self.leaving = list(leaving)
        self.job_id = uuid.uuid4().hex
        self.state, self.line, self.message, self.output = "running", "waiting for CorridorKey", "", None
        self.sent = False
        self.cancel_asked = False
        self.session = None

    def snapshot(self):
        return types.SimpleNamespace(
            phase=types.SimpleNamespace(value=self.state),
            line=self.line, message=self.message, output=self.output,
        )

    def cancel(self) -> None:
        self.cancel_asked = True
        if self.session is not None:
            self.session.cancel(self)


class WarmSession:
    def __init__(self, modules: dict) -> None:
        self.m = modules  # Corridor's runtime, request, licence, operators and words modules
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._write_lock = threading.Lock()
        self._queue: list[Ticket] = []
        self._closing = False
        self._park_now = False
        self._process = None
        self._lines = None
        self._next_id = 1
        self._licensed_at = 0.0
        self.status = "starting"  # starting, loading, ready, parked, closed
        self._worker = threading.Thread(target=self._run, name="corridorkey2-warm", daemon=True)
        self._worker.start()

    # main thread
    def alive(self) -> bool:
        return self._worker.is_alive() and not self._closing

    def submit(self, ticket: Ticket) -> bool:
        with self._wake:
            if self._closing:
                return False
            ticket.session = self
            self._queue.append(ticket)
            self._wake.notify()
        return True

    def park(self) -> None:
        with self._wake:
            self._park_now = True
            self._wake.notify()

    def cancel(self, ticket: Ticket) -> None:
        with self._wake:
            if ticket in self._queue:
                self._queue.remove(ticket)
                ticket.state, ticket.line = "cancelled", "cancelled"
                return
            process = self._process
        if ticket.sent and process is not None:
            threading.Thread(
                target=self._write,
                args=(process, {"v": 1, "id": self._take_id(), "command": "CancelTemporalMedia",
                                "payload": {"job_id": ticket.job_id}}),
                daemon=True,
            ).start()

    def close(self) -> None:
        with self._wake:
            if self._closing:
                return
            self._closing = True
            for ticket in self._queue:
                ticket.state, ticket.line = "cancelled", "cancelled"
            self._queue.clear()
            process = self._process
            self._wake.notify()
        if process is not None:
            threading.Thread(target=self._leave, args=(process,), daemon=True).start()

    # worker
    def _take_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        return request_id

    def _write(self, process, line: dict) -> None:
        stdin = process.stdin
        if stdin is None:
            return
        try:
            with self._write_lock:
                stdin.write(json.dumps(line) + "\n")
                stdin.flush()
        except (OSError, ValueError):
            pass

    def _leave(self, process) -> None:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=_LEAVE_S)
        except subprocess.TimeoutExpired:
            process.kill()
        self.status = "closed"

    def _ask(self, process, command: str, payload: dict, timeout: float):
        request_id = self._take_id()
        self._write(process, {"v": 1, "id": request_id, "command": command, "payload": payload})
        answered = threading.Event()
        watchdog = threading.Timer(timeout, lambda: None if answered.is_set() else process.kill())
        watchdog.daemon = True
        watchdog.start()
        try:
            for raw in self._lines:
                line = _parsed(raw)
                if line is not None and line.get("id") == request_id and "ok" in line:
                    answered.set()
                    return line
        finally:
            watchdog.cancel()
        return None

    def _licensed(self, paths, ticket: Ticket) -> bool:
        if time.monotonic() - self._licensed_at < _LICENCE_RECHECK_S:
            return True
        licence = self.m["runtime"].read_licence(paths)
        if not self.m["licence"].renders(licence, int(time.time())):
            ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
            ticket.message = self.m["licence"].panel_text(licence)
            return False
        self._licensed_at = time.monotonic()
        return True

    def _open(self, ticket: Ticket):
        paths = self.m["runtime"].locate()
        if paths is None:
            ticket.state, ticket.line = "failed", self.m["words"].RUNTIME_MISSING
            return None
        if not self._licensed(paths, ticket):
            return None
        wait = getattr(self.m["operators"], "_wait_left", None)
        if wait is not None and ticket.leaving:
            wait(ticket.leaving)  # the GPU Memory estimate's model leaves the card first
        self.status, ticket.line = "starting", "starting CorridorKey"
        try:
            process = self.m["runtime"].spawn_session(paths)
        except OSError:
            ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
            return None
        with self._lock:
            self._process = process
        self._lines = iter(process.stdout or ())
        ready = threading.Event()
        watchdog = threading.Timer(_READY_S, lambda: None if ready.is_set() else process.kill())
        watchdog.daemon = True
        watchdog.start()
        try:
            for raw in self._lines:
                if (_parsed(raw) or {}).get("event") == "ready":
                    ready.set()
                    break
        finally:
            watchdog.cancel()
        self.status, ticket.line = "loading", "loading the model (first preview only)"
        prepared = self._ask(process, "PrepareModel", {}, 600.0) if ready.is_set() else None
        if prepared is None or prepared.get("ok") is not True:
            self._drop(process)
            ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
            return None
        self.status = "ready"
        return process

    def _drop(self, process) -> None:
        with self._lock:
            self._process = None
        self._leave(process)

    def _key(self, process, ticket: Ticket) -> bool:
        """Send one preview and read until its answer. False when the child has gone."""
        request_id = self._take_id()
        line = self.m["request"].process_shot_folder(
            request_id=request_id, job_id=ticket.job_id, source=ticket.source,
            render_folder=ticket.render_folder, controls=ticket.controls, plan=ticket.plan,
        )
        ticket.sent = True
        ticket.line = "keying"
        self._write(process, line)
        if ticket.cancel_asked:
            self._write(process, {"v": 1, "id": self._take_id(), "command": "CancelTemporalMedia",
                                  "payload": {"job_id": ticket.job_id}})
        for raw in self._lines:
            msg = _parsed(raw)
            if msg is None or msg.get("id") != request_id:
                continue
            if "ok" in msg:
                body = msg.get("body") if isinstance(msg.get("body"), dict) else {}
                if msg.get("ok") is not True:
                    message = body.get("message")
                    ticket.message = message if isinstance(message, str) else ""
                    ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
                elif body.get("status") == "cancelled":
                    kept = body.get("kept_directory")
                    ticket.output = kept if isinstance(kept, str) else None
                    ticket.state, ticket.line = "cancelled", "cancelled"
                else:
                    ticket.state, ticket.line = "done", "done"
                return True
            shot = msg.get("shot")
            if isinstance(shot, dict) and isinstance(shot.get("output"), str):
                ticket.output = shot["output"]
            progress = msg.get("progress")
            if isinstance(progress, dict):
                fraction = progress.get("fraction")
                detail = progress.get("detail") if isinstance(progress.get("detail"), str) else ""
                percent = f"{int(float(fraction) * 100)}% " if isinstance(fraction, (int, float)) else ""
                ticket.line = f"{percent}{detail}".strip() or "keying"
        ticket.state, ticket.line = "failed", self.m["words"].INTERRUPTED
        return False

    def _run(self) -> None:
        process, parked = None, False
        try:
            while True:
                with self._wake:
                    quiet_since = time.monotonic()
                    while not self._queue and not self._closing and not self._park_now:
                        idle = time.monotonic() - quiet_since
                        if process is None:
                            limit = 1.0  # nothing open and nothing asked: this session is done
                        else:
                            limit = _CLOSE_S if parked else _PARK_S
                        if idle >= limit:
                            break
                        self._wake.wait(timeout=limit - idle)
                    if self._closing:
                        return
                    park_now, self._park_now = self._park_now, False
                    ticket = self._queue.pop(0) if self._queue else None
                if ticket is None:
                    if process is None:
                        return
                    if parked and not park_now:
                        return  # quiet for the close spell
                    if not parked:
                        answer = self._ask(process, "ParkSession",
                                           {"model_id": self.m["request"].MODEL_ID,
                                            "device": self.m["request"].DEVICE}, 60.0)
                        if answer is None:
                            self._drop(process)
                            process = None
                        parked = True
                        self.status = "parked"
                    continue
                if ticket.cancel_asked:
                    ticket.state, ticket.line = "cancelled", "cancelled"
                    continue
                if process is None:
                    process = self._open(ticket)
                    if process is None:
                        if ticket.state == "running":
                            ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
                        continue
                elif not self._licensed(self.m["runtime"].locate(), ticket):
                    continue
                elif parked:  # the weights come back to the card from system memory
                    self.status, ticket.line = "loading", "waking CorridorKey"
                    woke = self._ask(process, "PrepareModel", {}, 600.0)
                    if woke is None or woke.get("ok") is not True:
                        self._drop(process)
                        process = self._open(ticket)
                        if process is None:
                            if ticket.state == "running":
                                ticket.state, ticket.line = "failed", self.m["words"].SHOT_NOT_RUN
                            parked = False
                            continue
                parked = False
                self.status = "keying"
                if not self._key(process, ticket):
                    self._drop(process)
                    process = None
                else:
                    self.status = "ready"
        except Exception:
            pass
        finally:
            with self._lock:
                self._closing = True
                remaining, self._queue = self._queue, []
            for ticket in remaining:
                ticket.state, ticket.line = "failed", "CorridorKey stopped"
            if process is not None:
                self._leave(process)
            self.status = "closed"


_WARM: list = [None]


def warm_session():
    session = _WARM[0]
    if session is None or not session.alive():
        modules = {name: ck(name) for name in ("runtime", "request", "licence", "operators", "words")}
        session = WarmSession(modules)
        _WARM[0] = session
    return session


def close_paid_warm() -> None:
    session = _WARM[0]
    if session is not None:
        session.close()
    _WARM[0] = None


def close_warm() -> None:
    close_paid_warm()
    close_free()


class FreeTicket:
    """One request to the free engine's worker; read from the main thread like a render job.
    `numbers`: the plate number of each input frame, in order (None: 1, 2, 3...)."""

    def __init__(self, request: dict, numbers, prefix: str = "frame_") -> None:
        self.request, self.numbers, self.prefix = request, numbers, prefix
        self.state, self.line, self.message, self.output = "running", "waiting for CorridorKey", "", None
        self.cancel_asked = False
        self.session = None

    def snapshot(self):
        return types.SimpleNamespace(
            phase=types.SimpleNamespace(value=self.state),
            line=self.line, message=self.message, output=self.output,
        )

    def cancel(self) -> None:
        self.cancel_asked = True
        if self.session is not None:
            self.session.cancel(self)


def exr_dir_of(request: dict) -> str:
    """Where the free runtime writes a request's EXRs (its raw_clip_name rule)."""
    path = os.path.normpath(request["input"])
    name = os.path.splitext(os.path.basename(path))[0]
    if name.lower() in {"input", "frames", "rgb"}:
        name = os.path.basename(os.path.dirname(path)) or name
    suffix = "cutout" if "cutout" in request["cfg"]["outputs"] else "alpha"
    return os.path.join(request["output_dir"], f"{name}_{suffix}_exr")


def renumber(exr_dir: str, numbers, prefix: str) -> int:
    """The free runtime numbers its frames 0, 1, 2...: name each after its plate."""
    if not os.path.isdir(exr_dir):
        return 0
    count = 0
    for name in sorted(os.listdir(exr_dir)):
        match = re.match(r"^(alpha|cutout|fg)_(\d+)\.exr$", name)
        if not match:
            continue
        index = int(match.group(2))
        number = numbers[index] if numbers is not None and index < len(numbers) else index + 1
        os.replace(os.path.join(exr_dir, name), os.path.join(exr_dir, f"{prefix}{number:06d}.exr"))
        count += 1
    return count


def _log_tail(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as log:
            lines = [l.strip() for l in log.read().splitlines() if l.strip()]
    except OSError:
        return ""
    for line in reversed(lines):
        if not line.startswith(("File ", "Traceback", "^", "~")):
            return line[-160:]
    return ""


class FreeSession:
    """The free CorridorKey 2.0 kept loaded: free_server.py run with its own Python, one request at
    a time. Quiet for _PARK_S it parks (the model to system memory); quiet for _CLOSE_S it closes.
    Cancelling a running request stops the worker (the free runtime can't be interrupted)."""

    def __init__(self, python: str, root: str) -> None:
        self.python, self.root = python, root
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: list[FreeTicket] = []
        self._current: FreeTicket | None = None
        self._closing = False
        self._park_now = False
        self._process = None
        self._next_id = 1
        self.status = "starting"
        self.gpu = ""
        folder = os.path.join(tempfile.gettempdir(), "CorridorKey Preview")
        os.makedirs(folder, exist_ok=True)
        self.log_path = os.path.join(folder, "free_engine.log")
        self._worker = threading.Thread(target=self._run, name="corridorkey2-free", daemon=True)
        self._worker.start()

    # main thread
    def alive(self) -> bool:
        return self._worker.is_alive() and not self._closing

    def submit(self, ticket: FreeTicket) -> bool:
        with self._wake:
            if self._closing:
                return False
            ticket.session = self
            self._queue.append(ticket)
            self._wake.notify()
        return True

    def park(self) -> None:
        with self._wake:
            self._park_now = True
            self._wake.notify()

    def cancel(self, ticket: FreeTicket) -> None:
        with self._wake:
            if ticket in self._queue:
                self._queue.remove(ticket)
                ticket.state, ticket.line = "cancelled", "cancelled"
                return
            process = self._process if self._current is ticket else None
        if process is not None:
            try:
                process.kill()
            except OSError:
                pass

    def close(self) -> None:
        with self._wake:
            if self._closing:
                return
            self._closing = True
            for ticket in self._queue:
                ticket.state, ticket.line = "cancelled", "cancelled"
            self._queue.clear()
            process = self._process
            self._wake.notify()
        if process is not None:
            threading.Thread(target=self._leave, args=(process,), daemon=True).start()

    # worker
    def _leave(self, process) -> None:
        try:
            if process.stdin is not None:
                process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            process.wait(timeout=_LEAVE_S)
        except subprocess.TimeoutExpired:
            process.kill()

    def _send(self, process, message: dict) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        try:
            process.stdin.write(json.dumps({"id": request_id, **message}) + "\n")
            process.stdin.flush()
        except (OSError, ValueError):
            pass
        return request_id

    def _answer(self, request_id: int):
        for raw in self._lines:
            line = _parsed(raw)
            if line is not None and line.get("id") == request_id and "ok" in line:
                return line
        return None

    def _open(self, ticket: FreeTicket):
        self.status, ticket.line = "starting", "starting CorridorKey 2.0"
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
        env["PYTHONUNBUFFERED"] = "1"
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "free_server.py")
        flags = 0x08000000 if sys.platform == "win32" else 0  # no console window
        try:
            log = open(self.log_path, "w", encoding="utf-8", errors="replace")
            process = subprocess.Popen(
                [self.python, "-u", server, self.root], cwd=self.root, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
            )
        except OSError as exc:
            ticket.state, ticket.line, ticket.message = "failed", "CorridorKey 2.0 didn't start", str(exc)
            return None
        with self._lock:
            self._process = process
        self._lines = iter(process.stdout)
        for raw in self._lines:
            event = (_parsed(raw) or {}).get("event")
            if event == "loading":
                self.status, ticket.line = "loading", "loading the model (first preview only)"
            elif event == "ready":
                self.gpu = (_parsed(raw) or {}).get("gpu", "")
                self.status = "ready"
                return process
            elif event == "error":
                ticket.state, ticket.line = "failed", "CorridorKey 2.0 didn't start"
                ticket.message = (_parsed(raw) or {}).get("message", "")
                self._drop(process)
                return None
        ticket.state, ticket.line = "failed", "CorridorKey 2.0 stopped while loading"
        ticket.message = _log_tail(self.log_path)
        self._drop(process)
        return None

    def _drop(self, process) -> None:
        with self._lock:
            self._process = None
        self._leave(process)

    def _key(self, process, ticket: FreeTicket) -> bool:
        with self._lock:
            self._current = ticket
        ticket.line = "keying"
        answer = self._answer(self._send(process, {"cmd": "key", **ticket.request}))
        with self._lock:
            self._current = None
        exr_dir = exr_dir_of(ticket.request)
        if answer is None:  # the worker stopped: cancelled (killed) or crashed
            ticket.output = exr_dir if renumber(exr_dir, ticket.numbers, ticket.prefix) else None
            if ticket.cancel_asked:
                ticket.state, ticket.line = "cancelled", "cancelled"
            else:
                ticket.state, ticket.line = "failed", "CorridorKey 2.0 stopped"
                ticket.message = _log_tail(self.log_path)
            return False
        if answer.get("ok") is not True:
            ticket.state, ticket.line = "failed", "Shot not run"
            ticket.message = answer.get("message", "")
            return True
        renumber(answer.get("exr_dir") or exr_dir, ticket.numbers, ticket.prefix)
        ticket.output = answer.get("exr_dir") or exr_dir
        ticket.state, ticket.line = "done", "done"
        return True

    def _run(self) -> None:
        process, parked = None, False
        try:
            while True:
                with self._wake:
                    quiet_since = time.monotonic()
                    while not self._queue and not self._closing and not self._park_now:
                        idle = time.monotonic() - quiet_since
                        limit = 1.0 if process is None else (_CLOSE_S if parked else _PARK_S)
                        if idle >= limit:
                            break
                        self._wake.wait(timeout=limit - idle)
                    if self._closing:
                        return
                    park_now, self._park_now = self._park_now, False
                    ticket = self._queue.pop(0) if self._queue else None
                if ticket is None:
                    if process is None or (parked and not park_now):
                        return
                    if not parked:
                        if self._answer(self._send(process, {"cmd": "park"})) is None:
                            self._drop(process)
                            process = None
                        parked, self.status = True, "parked"
                    continue
                if ticket.cancel_asked:
                    ticket.state, ticket.line = "cancelled", "cancelled"
                    continue
                if process is None:
                    process = self._open(ticket)
                    if process is None:
                        continue
                parked = False
                self.status = "keying"
                if not self._key(process, ticket):
                    self._drop(process)
                    process = None
                self.status = "ready" if process is not None else "closed"
        except Exception:
            pass
        finally:
            with self._lock:
                self._closing = True
                remaining, self._queue = self._queue, []
            for ticket in remaining:
                ticket.state, ticket.line = "failed", "CorridorKey 2.0 stopped"
            if process is not None:
                self._leave(process)
            self.status = "closed"


_FREE: list = [None]


def free_session() -> FreeSession:
    session = _FREE[0]
    root = free_root()
    python = free_python(root)
    if session is None or not session.alive() or session.root != root or session.python != python:
        if session is not None:
            session.close()
        session = FreeSession(python, root)
        _FREE[0] = session
    return session


def close_free() -> None:
    session = _FREE[0]
    if session is not None:
        session.close()
    _FREE[0] = None


def park_all() -> None:
    for session in (_WARM[0], _FREE[0]):
        if session is not None and session.alive():
            session.park()


def warm_alive() -> bool:
    session = _FREE[0] if engine() == "free" else _WARM[0]
    return session is not None and session.alive()


def warm_status() -> str:
    if engine() == "free":
        problem = free_setup_problem()
        if problem:
            return problem
        session = _FREE[0]
        if session is None or not session.alive():
            return "Not loaded: next preview takes ~25 s"
        return {
            "starting": "Starting CorridorKey 2.0...",
            "loading": "Loading the model...",
            "ready": "Loaded: previews are quick",
            "keying": "Loaded and keying",
            "parked": "Parked (GPU freed): still quick",
        }.get(session.status, "CorridorKey 2.0 loaded")
    session = _WARM[0]
    if session is None or not session.alive():
        return "Not loaded: next preview takes ~25 s"
    return {
        "starting": "Starting CorridorKey...",
        "loading": "Loading the model...",
        "ready": "Loaded: previews are quick",
        "keying": "Loaded and keying",
        "parked": "Parked (GPU freed): still quick",
    }.get(session.status, "CorridorKey loaded")


# --- previews: frames keyed on demand, cached on disk ---------------------------------------------


def numbered(folder: str) -> dict[int, str]:
    """The plates in a folder by frame number."""
    found: dict[int, str] = {}
    try:
        names = os.listdir(folder)
    except OSError:
        return found
    for name in names:
        match = _NUMBER.search(name)
        if match and name.lower().endswith(_PLATES):
            found.setdefault(int(match.group(1)), name)
    return found


def _place(path: str, folder: str, name: str | None = None) -> None:
    """A plate into a preview's folder: a hard link when the disk allows, a copy otherwise."""
    target = os.path.join(folder, name or os.path.basename(path))
    try:
        os.link(path, target)
    except OSError:
        shutil.copy2(path, target)


@dataclasses.dataclass(frozen=True)
class FreeControls:
    """The free engine's controls (what a preview's cache is filed by)."""

    input_linear: bool = False
    model_scale: float = 1.0
    auto_window: bool = True
    window: int = 80
    low_vram: bool = False
    use_alpha_hint: bool = True
    hint_frames: str = "ALL"
    hint_quality: float = 0.95
    carry_hint: bool = True
    carry_hint_quality: float = 0.95
    despill: float = 0.7
    output: str = "CUTOUT"
    preview_context: int = 4


def free_controls(state) -> FreeControls:
    r = lambda v: round(float(v), 6)  # noqa: E731  (float32 props read back with noise)
    return FreeControls(
        input_linear=state.input_linear,
        model_scale=float(state.internal_resolution),
        auto_window=state.time_awareness == "AUTO",
        window=state.window,
        low_vram=state.low_vram,
        use_alpha_hint=state.use_alpha_hint,
        hint_frames=state.hint_frames,
        hint_quality=r(state.hint_quality),
        carry_hint=state.carry_hint,
        carry_hint_quality=r(state.carry_hint_quality),
        despill=r(state.despill_strength),
        output=state.output,
        preview_context=state.preview_context,
    )


def controls_of(src):
    if is_free_source(src):
        return free_controls(src.corridorkey2)
    return ck("node").controls_of(src.corridorkey)


def preview_span(controls) -> int:
    if isinstance(controls, FreeControls):
        return controls.preview_context
    return controls.window if controls.time_awareness else 0


def hint_of(src, scene):
    """What's in the source's Alpha Hint: a sequence's folder, one image, or a video; or None."""
    if not is_free_source(src):
        return ck("node").wired_hint(src, scene)
    hint = wired_image_node(src, 1)
    image = hint.image if hint is not None else None
    if image is None or not image.filepath:
        return None
    path = bpy.path.abspath(image.filepath, library=image.library)
    return os.path.dirname(path) if image.source == "SEQUENCE" else path


def result_read(src) -> tuple[str, str]:
    """How a key's frames are read: (Blender colour space, alpha mode)."""
    if is_free_source(src):
        state = src.corridorkey2
        return ("Non-Color", "NONE") if state.output == "MATTE" else ("Linear Rec.709", "STRAIGHT")
    plate = wired_image_node(src, 0)
    controls = controls_of(src)
    return ck("source").result_read(
        plate.image.colorspace_settings.name, controls.input_linear, controls.output, controls.linearize
    )


def free_request(controls: FreeControls, input_path: str, output_dir: str, hint_first, hint_video) -> dict:
    cfg = {
        "outputs": ["alpha"] if controls.output == "MATTE" else ["cutout"],
        "auto_vram": controls.auto_window and not controls.low_vram,
        "hann_chunk": controls.window,
        "hann_stride": max(1, controls.window // 2),
        "low_vram": controls.low_vram,
        "carry_hint": controls.carry_hint,
        "hint_quality": controls.hint_quality,
        "carry_hint_quality": controls.carry_hint_quality,
        "model_scale": controls.model_scale,
        "frame_dir_linear": controls.input_linear,
        "despill_strength": controls.despill,
        "temp_dir": os.path.join(tempfile.gettempdir(), "corridorkey_v2"),
    }
    return {"input": input_path, "output_dir": output_dir, "hint_first": hint_first,
            "hint_video": hint_video, "cfg": cfg}


def stage_free_hint(hint, controls: FreeControls, numbers, root: str):
    """(first-frame hint, full hint) for the free runtime, lined up with `numbers` (the plate
    numbers being keyed, in order; None for a movie). A full hint is staged frame for frame; when
    some frames have none, the first frame's hint is used alone."""
    if not controls.use_alpha_hint or not hint:
        return None, None
    if os.path.isdir(hint):
        hints = numbered(hint)
        if not hints:
            return None, None
        if numbers is None:
            first = os.path.join(hint, hints[min(hints)])
            return (None, hint) if controls.hint_frames == "ALL" else (first, None)
        if controls.hint_frames == "ALL" and all(n in hints for n in numbers):
            folder = os.path.join(root, "hint")
            os.makedirs(folder, exist_ok=True)
            for n in numbers:
                name = hints[n]
                _place(os.path.join(hint, name), folder, f"{n:08d}{os.path.splitext(name)[1]}")
            return None, folder
        if numbers[0] in hints:
            return os.path.join(hint, hints[numbers[0]]), None
        return None, None
    if hint.lower().endswith(_MOVIES):
        return (None, hint) if numbers is None and controls.hint_frames == "ALL" else (None, None)
    return (hint, None) if os.path.isfile(hint) else (None, None)


def quick_controls(controls):
    """Quick Preview: the frame alone, without Time Awareness's neighbours. A Single Frame hint
    needs Time Awareness, so it keeps it."""
    if isinstance(controls, FreeControls):
        return dataclasses.replace(controls, preview_context=0) if controls.preview_context else controls
    if controls.time_awareness and not (controls.use_alpha_hint and controls.hint_frames == "single_frame"):
        return dataclasses.replace(controls, time_awareness=False)
    return controls


def quick_of(src: bpy.types.Node) -> bool:
    node = node_by_source(st(src).uid)
    return bool(node is not None and node.corridorkey_shader.quick_preview)


def preview_key(src: bpy.types.Node, scene: bpy.types.Scene, frame: int, quick: bool = False):
    """((footage number, controls, hint, footage folder), None) for `frame`, or (None, why it can't
    be previewed). `quick`: the controls of a Quick Preview."""
    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None:
        return None, "Plug footage into the node first."
    if image.source != "SEQUENCE":
        return None, "Previews need an image sequence, not a movie."
    folder = os.path.dirname(bpy.path.abspath(image.filepath, library=image.library))
    number = frame - plate.frame_start + 1 + plate.frame_offset
    hint = hint_of(src, scene)
    controls = controls_of(src)
    if quick:
        controls = quick_controls(controls)
    return (number, controls, hint, folder), None


def cache_root() -> str:
    if bpy.data.filepath:
        return bpy.path.abspath("//CorridorKey Previews")
    return os.path.join(tempfile.gettempdir(), "CorridorKey Preview", "cache")


def cache_dir(uid: str, controls, hint, folder) -> str:
    """Where previews made with exactly these settings, this hint and this footage are kept: a
    change of any of them is a different folder, so an old preview never passes for a new one."""
    blob = json.dumps([dataclasses.asdict(controls), hint, folder], sort_keys=True, default=str)
    return os.path.join(cache_root(), uid[:8], hashlib.sha1(blob.encode()).hexdigest()[:12])


def cache_folders(src: bpy.types.Node, scene: bpy.types.Scene, full_only: bool = False):
    """([cache folders for the node's current settings, full quality first, then Quick Preview's
    when it is on], the frame number the scene is on)."""
    key, _why = preview_key(src, scene, scene.frame_current)
    if key is None:
        return [], None
    number, controls, hint, folder = key
    uid = st(src).uid
    folders = [cache_dir(uid, controls, hint, folder)]
    if not full_only and quick_of(src):
        quick = quick_controls(controls)
        if quick != controls:
            folders.append(cache_dir(uid, quick, hint, folder))
    return folders, number


def cached_frames(src: bpy.types.Node, scene: bpy.types.Scene) -> set[int]:
    folders, _number = cache_folders(src, scene)
    found: set[int] = set()
    for folder in folders:
        found |= _frames_in(folder)
    return found


def current_preview(src: bpy.types.Node, scene: bpy.types.Scene, full_only: bool = False):
    """(cache folder, frame number) when the frame the scene is on has a preview (full quality
    first), else None."""
    folders, number = cache_folders(src, scene, full_only)
    for folder in folders:
        if number in _frames_in(folder):
            return folder, number
    return None


def shown_preview(src: bpy.types.Node, scene: bpy.types.Scene, full_only: bool = False):
    """The preview to show on this frame: one exists, and the rendered key (if it has this frame)
    is older than it. The newest of the two wins, so a render isn't hidden by the previews made
    before it, and a preview of changed settings isn't hidden by an older render."""
    preview = current_preview(src, scene, full_only)
    if preview is None:
        return None
    shown = shown_frames(src, scene.frame_current)
    if shown is None or shown.name != RESULT:
        return preview
    folder = os.path.dirname(bpy.path.abspath(shown.image.filepath, library=shown.image.library))
    number = scene.frame_current - shown.frame_start + 1 + shown.frame_offset
    rendered = _frame_file(folder, number)
    try:
        if rendered and os.path.getmtime(rendered) >= os.path.getmtime(
            os.path.join(preview[0], f"{preview[1]:06d}.exr")
        ):
            return None
    except OSError:
        pass
    return preview


def range_span(controls) -> int:
    """Frames either side a range render looks at (more than a preview: it's for keeps)."""
    if isinstance(controls, FreeControls):
        return max(controls.preview_context, 8)
    return controls.window if controls.time_awareness else 0


def start_preview(
    src: bpy.types.Node, scene: bpy.types.Scene, frames: list[int], full: bool = False,
    dest: dict | None = None,
) -> str | None:
    """Key `frames` (and, with Time Awareness, their neighbours) in one CorridorKey job, skipping
    any already cached. A Quick Preview (the node's toggle) keys each frame alone; `full` always
    uses the full settings. Returns why it can't, or None (started, all cached, or keying)."""
    free = is_free_source(src)
    if free:
        problem = free_setup_problem()
        if problem:
            return problem
    else:
        ck_node, ck_ops, ck_plan, ck_request = ck("node"), ck("operators"), ck("plan"), ck("request")
    uid = st(src).uid
    if uid in PREVIEW_JOBS:
        return None
    if full_render_running():
        return "Waiting: a CorridorKey render is running."
    if HINT_JOBS:
        return "Waiting: SAM 3 has the GPU while it makes the alpha hint."
    quick = not full and dest is None and quick_of(src)
    key, why = preview_key(src, scene, frames[0], quick=quick)
    if key is None:
        return why
    _number, controls, hint, folder = key
    plate = wired_image_node(src, 0)
    to_number = lambda f: f - plate.frame_start + 1 + plate.frame_offset  # noqa: E731
    target_dir = dest["folder"] if dest else cache_dir(uid, controls, hint, folder)
    have = set() if dest else _frames_in(target_dir)  # a range render keys every frame again
    plates = numbered(folder)
    numbers = [to_number(f) for f in frames]
    targets = sorted(n for n in numbers if n in plates and n not in have)
    if not targets:
        if len(frames) == 1 and numbers[0] not in plates:
            return f"No plate for frame {frames[0]}."
        return None
    span = range_span(controls) if dest else preview_span(controls)
    wanted = [n for n in range(targets[0] - span, targets[-1] + span + 1) if n in plates]

    root = os.path.join(tempfile.gettempdir(), "CorridorKey Preview", uid, uuid.uuid4().hex[:8])
    plate_dir, out_dir = os.path.join(root, "plates"), os.path.join(root, "out")
    os.makedirs(plate_dir)
    os.makedirs(out_dir)
    to_frame = lambda n: n - plate.frame_offset - 1 + plate.frame_start  # noqa: E731
    shown = (to_frame(targets[0]), to_frame(targets[-1]))
    label = f"Frame {shown[0]}" if len(targets) == 1 else f"Frames {shown[0]}-{shown[1]}"
    if quick and controls != controls_of(src):
        label += " (quick)"
    if dest:
        label = ("Test render " if dest["new"] else "Re-render ") + label[0].lower() + label[1:]
        RANGE_LINES.add(uid)
    else:
        RANGE_LINES.discard(uid)
    if free:  # names the free runtime sorts in frame order; it numbers its output in that order
        for n in wanted:
            name = plates[n]
            _place(os.path.join(folder, name), plate_dir, f"{n:08d}{os.path.splitext(name)[1]}")
        hint_first, hint_video = stage_free_hint(hint, controls, wanted, root)
        job = FreeTicket(free_request(controls, plate_dir, out_dir, hint_first, hint_video), wanted)
        if not free_session().submit(job):
            close_free()
            free_session().submit(job)
        PREVIEW_JOBS[uid] = {
            "job": job, "targets": set(targets), "dir": target_dir, "root": root,
            "label": label, "frames": [to_frame(n) for n in targets], "dest": dest,
        }
        PREVIEW_LINE[uid] = f"{label}: starting CorridorKey 2.0"
        return None
    for n in wanted:
        _place(os.path.join(folder, plates[n]), plate_dir)
    hint_path = None
    if hint and os.path.isdir(hint):  # a sequence: the same frames of it
        hints = numbered(hint)
        matched = [n for n in wanted if n in hints]
        if matched:
            hint_path = os.path.join(root, "hint")
            os.makedirs(hint_path)
            for n in matched:
                _place(os.path.join(hint, hints[n]), hint_path)
    elif hint and os.path.isfile(hint) and not hint.lower().endswith(_MOVIES):
        hint_path = hint  # one picked image

    footage = ck_node.wired_footage(src, scene)
    if footage is None:
        shutil.rmtree(root, ignore_errors=True)
        return "Plug footage into the node first."
    shot = ck_plan.ShotRange(first=wanted[0], count=len(wanted))
    plan = ck_plan.plan_pass(controls, hint_path, shot, footage.shot_frame)
    if isinstance(plan, str):  # Corridor's own words for why not
        shutil.rmtree(root, ignore_errors=True)
        return plan
    source = ck_request.Source(
        frames=plate_dir, name=f"{footage.source.name} preview", rate=footage.source.rate
    )
    leaving = ck_ops.close_estimates() if hasattr(ck_ops, "close_estimates") else []
    job = Ticket(source, out_dir, controls, plan, leaving)
    if not warm_session().submit(job):
        close_paid_warm()
        warm_session().submit(job)
    PREVIEW_JOBS[uid] = {
        "job": job, "targets": set(targets), "dir": target_dir, "root": root,
        "label": label, "frames": [to_frame(n) for n in targets], "dest": dest,
    }
    PREVIEW_LINE[uid] = f"{label}: starting CorridorKey"
    return None


def cancel_preview(uid: str) -> None:
    entry = PREVIEW_JOBS.get(uid)
    if entry is not None:
        entry["job"].cancel()


def clear_previews(uid: str) -> None:
    """Every preview this node has, on disk too."""
    cancel_preview(uid)
    PREVIEW_LINE.pop(uid, None)
    _SETTLING.pop(uid, None)
    shutil.rmtree(os.path.join(cache_root(), uid[:8]), ignore_errors=True)
    _LISTED.clear()


def land_previews() -> bool:
    """Previews that finished: their frames move into the cache."""
    changed = False
    for uid, entry in list(PREVIEW_JOBS.items()):
        view = entry["job"].snapshot()
        if view.phase.value == "running":
            line = f"{entry['label']}: {view.line}"
            if PREVIEW_LINE.get(uid) != line:
                PREVIEW_LINE[uid] = line
                changed = True
            continue
        del PREVIEW_JOBS[uid]
        kept = 0
        dest = entry.get("dest")
        if view.output and os.path.isdir(view.output):  # a cancelled job's kept frames count too
            os.makedirs(entry["dir"], exist_ok=True)
            for name in os.listdir(view.output):
                match = _NUMBER.search(name)
                if not (name.lower().endswith(".exr") and match):
                    continue
                number = int(match.group(1))
                if number in entry["targets"]:
                    final = (f"{dest['prefix']}{str(number).zfill(dest['width'])}.exr" if dest
                             else f"{number:06d}.exr")
                    target = os.path.join(entry["dir"], final)
                    if os.path.exists(target):
                        os.remove(target)
                    shutil.move(os.path.join(view.output, name), target)
                    kept += 1
        shutil.rmtree(entry["root"], ignore_errors=True)
        if dest:
            src = node_by_uid(uid)
            if src is not None and kept:
                range_landed(src, entry)
            PREVIEW_LINE[uid] = (f"{entry['label']}: {kept} frame{'s' if kept != 1 else ''} keyed."
                                 if kept else view.message or view.line or "The render didn't finish.")
            if src is not None:  # the Render box says how it went; the Preview row is free again
                st(src).render_line = PREVIEW_LINE.pop(uid)
                RANGE_LINES.discard(uid)
        elif kept:
            PREVIEW_LINE[uid] = f"{entry['label']}: previewed {kept} frame{'s' if kept != 1 else ''}."
        else:
            PREVIEW_LINE[uid] = view.message or view.line or "The preview didn't finish."
            CACHE_STALL[uid] = bpy.context.scene.frame_current  # Cache Ahead waits for the playhead to move
        changed = True
    return changed


def range_destination(src, into: str):
    """Where a range render's frames go: into the current key (named as its frames are), or a new
    Render N-range folder in the Render Folder. (dict, None) or (None, why not)."""
    state = st(src)
    result = src.node_tree.nodes.get(RESULT) if src.node_tree is not None else None
    if into == "CURRENT" and result is not None and result.image is not None:
        folder = os.path.dirname(bpy.path.abspath(result.image.filepath))
        names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".exr")) if os.path.isdir(folder) else []
        match = re.match(r"^(.*?)(\d+)\.exr$", names[0], re.IGNORECASE) if names else None
        if match:
            return {"folder": folder, "prefix": match.group(1), "width": len(match.group(2)), "new": False}, None
    if not state.render_folder:
        return None, "Choose a Render Folder first."
    root = bpy.path.abspath(state.render_folder)
    os.makedirs(root, exist_ok=True)
    plate = wired_image_node(src, 0)
    stem = plate.image.name.rsplit(".", 1)[0] if plate is not None and plate.image is not None else "key"
    prefix = re.sub(r"[_\-. ]*\d+$", "", stem) + "_"
    return {"folder": _next_render_folder(root) + "-range", "prefix": prefix, "width": 6, "new": True}, None


def range_landed(src, entry) -> None:
    """Show a range render's frames: the current key, updated; or the new test folder."""
    dest, state = entry["dest"], st(src)
    if dest["new"]:
        plate = wired_image_node(src, 0)
        state.result_frame_start, state.result_frame_offset = plate.frame_start, plate.frame_offset
        state.result_frame_duration = plate.frame_duration
        state.result_colorspace, state.result_alpha_mode = result_read(src)
    if is_free_source(src):
        free_show_result(src, dest["folder"])
    else:
        ck("node").show_result(src, dest["folder"])
    state.result_folder = dest["folder"]
    state.render_line = f"{entry['label']}: done"
    _LISTED.clear()


def preview_image(src: bpy.types.Node, folder: str, number: int) -> bpy.types.Image:
    """The cache folder as one image sequence, read as CorridorKey wrote it."""
    name = f".CorridorKey Preview {st(src).uid[:8]}"
    path = os.path.join(folder, f"{number:06d}.exr")
    image = bpy.data.images.get(name)
    if image is None:
        image = bpy.data.images.load(path)
        image.name = name
    elif os.path.normcase(os.path.dirname(bpy.path.abspath(image.filepath))) != os.path.normcase(folder):
        image.filepath = path
    _set(image, "source", "SEQUENCE")
    colorspace, alpha = result_read(src)
    try:
        _set(image.colorspace_settings, "name", colorspace)
    except TypeError:
        pass
    _set(image, "alpha_mode", alpha)
    return image


def _spans(frames) -> list[list[int]]:
    spans: list[list[int]] = []
    for frame in sorted(set(frames)):
        if spans and frame == spans[-1][1] + 1:
            spans[-1][1] = frame
        else:
            spans.append([frame, frame])
    return spans


def focus_node(context) -> bpy.types.Node | None:
    """The node the timeline shows: the active object's, else the file's first."""
    nodes = object_nodes(getattr(context, "active_object", None))
    return nodes[0] if nodes else next(shader_nodes(), None)


def timeline_marks(scene: bpy.types.Scene) -> dict:
    """Which scene frames have a full key (blue), a preview for the current settings (green), or
    are being keyed now (orange)."""
    marks = {"key": [], "preview": [], "busy": []}
    node = focus_node(bpy.context)
    src = source_of(node) if node is not None else None
    if src is None:
        return marks
    uid = st(src).uid
    keyed: set[int] = set()
    result = src.node_tree.nodes.get(RESULT) if src.node_tree is not None else None
    if result is not None and result.image is not None and result.image.source == "SEQUENCE":
        folder = os.path.dirname(bpy.path.abspath(result.image.filepath))
        keyed = {n - result.frame_offset - 1 + result.frame_start for n in _frames_in(folder)}
    plate = wired_image_node(src, 0)
    if plate is None:
        marks["key"] = _spans(keyed)
        return marks
    to_frame = lambda n: n - plate.frame_offset - 1 + plate.frame_start  # noqa: E731
    previewed = {to_frame(n) for n in cached_frames(src, scene)}
    done, busy, rendering = render_progress(uid)  # a render's bar grows frame by frame
    done_frames = {to_frame(n) for n in done}
    if rendering:
        keyed |= done_frames
    else:
        previewed |= done_frames
    marks["key"], marks["preview"] = _spans(keyed), _spans(previewed)
    marks["busy"] = _spans(to_frame(n) for n in busy - done)
    return marks


_INDEXED = re.compile(r"^(alpha|cutout|fg)_(\d+)\.exr$", re.IGNORECASE)
_PROGRESS: dict[str, tuple] = {}  # uid -> (when it was counted, the plate numbers keyed so far)
_PROGRESS_S = 1.0
_PAID_RUNS: dict[str, dict] = {}  # uid -> a paid full render seen running: its folder, its start


def _keyed_in(folders, numbers=None) -> set[int]:
    """The plate numbers of the EXRs written so far in `folders` (and two levels in). The free
    runtime names its frames 0, 1, 2...: `numbers` says which plate each one is."""
    found: set[int] = set()
    for top in folders:
        if not top or not os.path.isdir(top):
            continue
        depth = top.rstrip("\\/").count(os.sep)
        for here, dirs, files in os.walk(top):
            if here.rstrip("\\/").count(os.sep) - depth >= 2:
                dirs[:] = []
            for name in files:
                if not name.lower().endswith(".exr"):
                    continue
                match = _INDEXED.match(name)
                if match:
                    index = int(match.group(2))
                    if numbers is None:
                        found.add(index + 1)
                    elif index < len(numbers):
                        found.add(numbers[index])
                    continue
                match = _NUMBER.search(name)
                if match:
                    found.add(int(match.group(1)))
    return found


def _paid_render_folders(uid: str, src) -> list[str]:
    """Where a paid full render is writing: the Render Folder's folders made or written to since
    it started (Corridor names them; this only watches)."""
    root = bpy.path.abspath(st(src).render_folder) if st(src).render_folder else ""
    if not root or not os.path.isdir(root):
        return []
    run = _PAID_RUNS.get(uid)
    if run is None or run["root"] != root:
        run = _PAID_RUNS[uid] = {"root": root, "before": set(os.listdir(root)), "started": time.time()}
    folders = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        try:
            fresh = name not in run["before"] or os.path.getmtime(path) >= run["started"] - 2
        except OSError:
            continue
        if fresh and os.path.isdir(path):
            folders.append(path)
    return folders


def render_progress(uid: str) -> tuple[set[int], set[int], bool]:
    """(plate numbers keyed so far, plate numbers asked for, whether it's a render rather than a
    preview) for whatever this node is keying now. Counted once a second."""
    entry, free_render = PREVIEW_JOBS.get(uid), FREE_RENDERS.get(uid)
    paid = uid in set(getattr(ck("operators"), "JOBS", None) or ())
    if entry is None and free_render is None and not paid:
        _PROGRESS.pop(uid, None)
        return set(), set(), False
    src = node_by_uid(uid)
    if entry is not None:
        busy, rendering = set(entry["targets"]), bool(entry.get("dest"))
    elif free_render is not None:
        busy, rendering = set(free_render.get("numbers") or ()), True
    else:
        busy, rendering = set(), True
    now = time.monotonic()
    counted = _PROGRESS.get(uid)
    if counted is not None and now - counted[0] < _PROGRESS_S:
        return counted[1], busy, rendering
    done: set[int] = set()
    try:
        if entry is not None:
            job = entry["job"]
            if isinstance(job, FreeTicket):
                done = _keyed_in([exr_dir_of(job.request)], job.numbers)
            else:
                done = _keyed_in([os.path.join(entry["root"], "out")])
            done &= busy
        elif free_render is not None:
            done = _keyed_in([free_render["exr_dir"]], free_render["numbers"])
        elif src is not None:
            done = _keyed_in(_paid_render_folders(uid, src))
    except OSError:
        pass
    _PROGRESS[uid] = (now, done)
    return done, busy, rendering


def previews_off(uid: str) -> bool:
    """A full render landed: Live and Cache Ahead turn off on the nodes showing it."""
    changed = False
    for node in shader_nodes():
        src = source_of(node)
        if src is not None and st(src).uid == uid:
            settings = node.corridorkey_shader
            if settings.live_preview or settings.cache_ahead:
                settings.live_preview = settings.cache_ahead = False
                changed = True
    return changed


def land_paid_renders() -> bool:
    """A paid full render that has finished: once Corridor has shown the whole shot, previews
    stop (it is all keyed now)."""
    jobs = set(getattr(ck("operators"), "JOBS", None) or ())
    changed = False
    now = time.time()
    for uid in [u for u in _PAID_RUNS if u not in jobs]:
        run = _PAID_RUNS[uid]
        if "ended" not in run:
            run["ended"] = now
            _PROGRESS.pop(uid, None)
            _LISTED.clear()
            changed = True
        src = node_by_uid(uid)
        result = src.node_tree.nodes.get(RESULT) if src is not None and src.node_tree is not None else None
        landed = False
        if result is not None and result.image is not None:
            folder = os.path.dirname(bpy.path.abspath(result.image.filepath, library=result.image.library))
            try:  # the key now shown is this render's, and the whole of it
                landed = os.path.getmtime(folder) >= run["started"] - 2
            except OSError:
                landed = False
            if landed and "partial" not in os.path.basename(folder).lower():
                changed |= previews_off(uid)
        if landed or now - run["ended"] > 10:  # Corridor shows its result a moment after
            _PAID_RUNS.pop(uid)
    for uid in jobs:  # watch from the start, so its folder is told from the old ones
        src = node_by_uid(uid)
        if "ended" in _PAID_RUNS.get(uid, {}):
            del _PAID_RUNS[uid]
        if uid not in _PAID_RUNS and src is not None and not is_free_source(src):
            _paid_render_folders(uid, src)
    return changed


def draw_timeline() -> None:
    """Thin bars along the bottom of the timeline, like Blender's own cache lines."""
    if not (MARKS["key"] or MARKS["preview"] or MARKS["busy"]):
        return
    try:
        import gpu
        from gpu_extras.batch import batch_for_shader

        region = bpy.context.region
        view = region.view2d
        scale = bpy.context.preferences.system.ui_scale
        height, gap, base = 3 * scale, 1 * scale, 16 * scale
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        for kind, row, color in (
            ("key", 1, (0.33, 0.58, 1.0, 0.95)),
            ("preview", 0, (0.36, 0.86, 0.47, 0.95)),
            ("busy", 0, (1.0, 0.6, 0.15, 0.95)),
        ):
            verts, faces = [], []
            for first, last in MARKS[kind]:
                x0 = view.view_to_region(first - 0.5, 0, clip=False)[0]
                x1 = view.view_to_region(last + 0.5, 0, clip=False)[0]
                if x1 < 0 or x0 > region.width:
                    continue
                if x1 - x0 < 3 * scale:  # one frame stays visible when zoomed out
                    middle = (x0 + x1) / 2
                    x0, x1 = middle - 1.5 * scale, middle + 1.5 * scale
                x0, x1 = max(x0, 0), min(x1, region.width)
                y0 = base + row * (height + gap)
                n = len(verts)
                verts += [(x0, y0), (x1, y0), (x1, y0 + height), (x0, y0 + height)]
                faces += [(n, n + 1, n + 2), (n, n + 2, n + 3)]
            if verts:
                batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=faces)
                shader.uniform_float("color", color)
                batch.draw(shader)
        gpu.state.blend_set("NONE")
    except Exception:  # a draw callback must never take the timeline down
        pass


def sync(
    node: bpy.types.Node,
    src: bpy.types.Node,
    scene: bpy.types.Scene | None = None,
    allow_preview: bool = True,
) -> bool:
    texture = node.node_tree.nodes.get(TEXTURE)
    if texture is None:
        return False
    preview = (
        shown_preview(src, scene, full_only=_RENDERING_WITH_KEY[0])
        if allow_preview and scene is not None else None
    )
    settings = node.corridorkey_shader
    screen_changed = False
    if settings.screen_clean or settings.screen_show:
        screen_changed = screen_update(node.node_tree, settings, wired_image_node(src, 0))
    if settings.floor_name:
        floor = bpy.data.objects.get(settings.floor_name)
        if floor is not None and floor.get(FLOOR_TAG) == st(src).uid:
            floor_sync(floor, wired_image_node(src, 0))
    if preview is not None:  # this frame has been previewed: show the cache, played as a sequence
        image = preview_image(src, *preview)
        plate = wired_image_node(src, 0)
        changed = screen_changed | _set(texture, "image", image)
        user = texture.image_user
        for attr in ("frame_start", "frame_offset", "frame_duration"):
            changed |= _set(user, attr, getattr(plate, attr))
        changed |= _set(user, "use_auto_refresh", True)
        changed |= _set(user, "use_cyclic", False)
        return changed | wire(node.node_tree, "key", settings)
    shown = shown_frames(src, scene.frame_current if scene is not None else None)
    image = shown.image if shown is not None else None
    changed = screen_changed | _set(texture, "image", image)
    if image is not None:
        user = texture.image_user
        for attr in ("frame_start", "frame_offset", "frame_duration"):
            changed |= _set(user, attr, getattr(shown, attr))
        changed |= _set(user, "use_auto_refresh", True)
        changed |= _set(user, "use_cyclic", False)
    keyed = shown is not None and shown.name == RESULT
    plugged = node.inputs.get(IMAGE_IN) is not None and node.inputs[IMAGE_IN].is_linked
    changed |= wire(node.node_tree, "key" if keyed else ("input" if plugged else "plate"), settings)
    return changed


def tick() -> float:
    try:
        scene = bpy.context.scene
        rendering = any(bpy.app.is_job_running(j) for j in ("RENDER", "OBJECT_BAKE"))
        ck_ops = ck("operators")
        if scene is not None and not rendering and (ck_ops is not None or engine() == "free"):
            placed, changed = set(), land_previews()
            changed |= land_free_renders()
            changed |= land_paid_renders()
            changed |= land_hints()
            changed |= land_screen_hints()
            changed |= land_floor_bakes()
            if getattr(ck_ops, "JOBS", None):  # a paid render takes the card: previews wait
                for uid in list(PREVIEW_JOBS):
                    cancel_preview(uid)
                close_paid_warm()
            playing = any(
                w.screen is not None and w.screen.is_animation_playing
                for w in bpy.context.window_manager.windows
            )
            for node in list(shader_nodes()):
                if node.inputs.get(HINT_IN) is None:  # made before the node had inputs
                    build_shader_tree(node.node_tree)
                    changed = True
                if node.label in ("", "CorridorKey"):  # named before CorridorKey 2.0
                    node.label = NAME
                src = source_of(node)
                if src is None:
                    src = adopt_source(node)
                if src is None:
                    continue
                if st(src).uid not in placed:
                    placed.add(st(src).uid)
                    for index, name in ((0, IMAGE_IN), (1, HINT_IN)):
                        image_node = wired_image_node(src, index)
                        if image_node is None:
                            continue
                        texture, plugged = upstream_texture(node, name)
                        if texture is not None:
                            take_wired(texture, image_node)
                        elif not plugged:
                            place_footage(image_node, scene)
                    if node.corridorkey_shader.live_preview and not playing and not HINT_JOBS:
                        changed |= live_step(src, scene)
                    if (node.corridorkey_shader.cache_ahead and node.corridorkey_shader.live_preview
                            and not playing and not HINT_JOBS and engine_available()):
                        changed |= cache_step(node, src, scene)
                changed |= sync(node, src, scene)
            marks = timeline_marks(scene)
            if marks != MARKS:
                MARKS.update(marks)
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == "DOPESHEET_EDITOR":
                            area.tag_redraw()
            if changed:
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type in {"VIEW_3D", "NODE_EDITOR"}:
                            area.tag_redraw()
    except (ReferenceError, RuntimeError, AttributeError, OSError):
        pass  # never let one bad node stop the timer
    return _TICK_S


CACHE_STALL: dict[str, int] = {}  # source uid -> the frame where a Cache Ahead job failed


def cache_step(node, src, scene) -> bool:
    """Cache Ahead: with nothing else keying, key the nearest uncached frames around the resting
    playhead, a few at a time, working outwards until the radius is full."""
    uid = st(src).uid
    if uid in PREVIEW_JOBS or uid in HINT_JOBS or full_render_running():
        return False
    settled = _SETTLING.get(uid)
    frame, now = scene.frame_current, time.monotonic()
    if CACHE_STALL.get(uid) == frame:
        return False
    CACHE_STALL.pop(uid, None)
    if settled is None or settled[0] != frame:
        _SETTLING[uid] = (frame, now)
        return False
    if now - settled[1] < _SETTLE_S:
        return False
    key, _why = preview_key(src, scene, frame, quick=quick_of(src))
    plate = wired_image_node(src, 0)
    if key is None or plate is None:
        return False
    number, controls, hint, folder = key
    have = _frames_in(cache_dir(uid, controls, hint, folder))
    plates = numbered(folder)
    settings = node.corridorkey_shader
    to_frame = lambda n: n - plate.frame_offset - 1 + plate.frame_start  # noqa: E731
    for distance in range(0, settings.cache_radius + 1):
        for step in ((0,) if distance == 0 else (distance, -distance)):
            n = number + step
            if n in plates and n not in have:
                direction = 1 if step >= 0 else -1
                block = [n + direction * i for i in range(settings.cache_batch)]
                block = [b for b in block if b in plates and b not in have and abs(b - number) <= settings.cache_radius]
                frames = sorted(to_frame(b) for b in block)
                why = start_preview(src, scene, frames)
                if why and PREVIEW_LINE.get(uid) != why:
                    PREVIEW_LINE[uid] = why
                return uid in PREVIEW_JOBS
    return False


def live_step(src: bpy.types.Node, scene: bpy.types.Scene) -> bool:
    """Live: once the playhead has rested on a frame, key it (if it isn't keyed for these
    settings already)."""
    uid, frame, now = st(src).uid, scene.frame_current, time.monotonic()
    settled = _SETTLING.get(uid)
    if settled is None or settled[0] != frame:
        _SETTLING[uid] = (frame, now)
        return False
    if now - settled[1] < _SETTLE_S or uid in PREVIEW_JOBS:
        return False
    why = start_preview(src, scene, [frame])
    if why and PREVIEW_LINE.get(uid) != why:
        PREVIEW_LINE[uid] = why
        return True
    return uid in PREVIEW_JOBS


@persistent
def _show_no_previews(*_args) -> None:
    """Before a save or a render, every node shows its footage or its key, never a preview: a
    preview is one frame from a temp folder, and a render farm has neither. The one exception is
    an F12 render that keyed its own frame first."""
    park_all()  # Blender's render gets the card
    if _RENDERING_WITH_KEY[0]:
        return
    scene = bpy.context.scene
    for node in list(shader_nodes()):
        src = source_of(node)
        if src is not None:
            sync(node, src, scene, allow_preview=False)


@persistent
def _on_frame(scene, _depsgraph=None) -> None:
    """Every frame change, playback included: the previewed frames play from the cache, and a
    partial key falls back to the footage where it ends. While Blender renders, no previews."""
    try:
        rendering = bpy.app.is_job_running("RENDER")
        for node in list(shader_nodes()):
            src = source_of(node)
            if src is not None:
                sync(node, src, scene, allow_preview=_RENDERING_WITH_KEY[0] or not rendering)
    except (ReferenceError, RuntimeError, AttributeError, OSError):
        pass


@persistent
def _render_ended(*_args) -> None:
    _RENDERING_WITH_KEY[0] = False


@persistent
def _after_load(*_args) -> None:
    close_warm()
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=1.0, persistent=True)


@persistent
def _tidy_before_save(*_args) -> None:
    """Sources whose shader node was deleted go before a save (their renders stay on disk)."""
    tree = bpy.data.node_groups.get(SOURCES)
    if tree is None:
        return
    used = set()
    for n in shader_nodes():
        used |= {n.node_tree.get(SOURCE), n.node_tree.get(SOURCE_FREE)}
    for group in bpy.data.node_groups:  # locked nodes' originals, kept for Unlock
        if group.get(MARK) and group.use_fake_user:
            used |= {group.get(SOURCE), group.get(SOURCE_FREE)}
    for src in [n for n in tree.nodes if is_source(n)]:
        uid = st(src).uid
        if uid in used or full_render_running(uid):
            continue
        for index in (0, 1):
            image_node = wired_image_node(src, index)
            if image_node is not None:
                tree.nodes.remove(image_node)
        inner = src.node_tree
        tree.nodes.remove(src)
        if inner is not None and inner.users == 0:
            bpy.data.node_groups.remove(inner)


# --- operators -------------------------------------------------------------------------------------


def _in_shader_editor(context) -> bool:
    space = context.space_data
    return getattr(space, "tree_type", "") == "ShaderNodeTree" and space.edit_tree is not None


class CORRIDORKEY_SHADER_OT_add_node(bpy.types.Operator):
    bl_idname = "corridorkey_shader.add_node"
    bl_label = NAME
    bl_description = "Add a CorridorKey node: AI-keyed footage as a texture, with Color and Alpha"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context) -> bool:
        if not _in_shader_editor(context):
            return False
        if engine() == "paid" and ck() is None:
            cls.poll_message_set("Enable Corridor's CorridorKey add-on, or pick the free engine")
            return False
        return True

    def execute(self, context):
        tree = context.space_data.edit_tree
        src = new_source()
        for other in tree.nodes:
            other.select = False
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = new_shader_tree(st(src).uid, source_key())
        node.label = NAME
        node.width = 200
        node.location = context.space_data.cursor_location
        node.select = True
        tree.nodes.active = node
        return {"FINISHED"}

    def invoke(self, context, event):
        region = context.region
        if region is not None and region.type == "WINDOW":
            context.space_data.cursor_location_from_region(event.mouse_region_x, event.mouse_region_y)
        result = self.execute(context)
        if "FINISHED" in result:
            bpy.ops.node.translate_attach_remove_on_cancel("INVOKE_DEFAULT")
        return result


class CORRIDORKEY_SHADER_OT_repair(bpy.types.Operator):
    bl_idname = "corridorkey_shader.repair"
    bl_label = "Repair CorridorKey Node"
    bl_description = "Give this node a fresh CorridorKey source (its footage and settings start over)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context) -> bool:
        return (engine() == "free" or ck() is not None) and is_ck_shader(
            getattr(context, "active_node", None)
        )

    def execute(self, context):
        node = context.active_node
        src = source_of(node) or adopt_source(node)
        if src is None:
            return {"CANCELLED"}
        ensure_inputs(src)
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_connect(bpy.types.Operator):
    bl_idname = "corridorkey_shader.connect"
    bl_label = "Connect to Principled BSDF"
    bl_description = "Wire Color into Base Color and Alpha into Alpha on this material's Principled BSDF"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context) -> bool:
        node = getattr(context, "active_node", None)
        return is_ck_shader(node) and any(
            n.bl_idname == "ShaderNodeBsdfPrincipled" for n in node.id_data.nodes
        )

    def execute(self, context):
        node = context.active_node
        tree = node.id_data
        candidates = [n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"]
        wired = [n for n in candidates if n.outputs[0].is_linked]
        bsdf = (wired or candidates)[0]
        tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        tree.links.new(node.outputs["Alpha"], bsdf.inputs["Alpha"])
        return {"FINISHED"}


# --- preview and reset ----------------------------------------------------------------------------


class CorridorKeyShaderSettings(bpy.types.PropertyGroup):
    live_preview: bpy.props.BoolProperty(
        name="Live Preview",
        description=(
            "Key the frame you're on whenever you stop on a new frame or change a setting. "
            "Each preview is keyed by CorridorKey and takes a few seconds"
        ),
        default=False,
    )
    quick_preview: bpy.props.BoolProperty(
        name="Quick Preview",
        description=(
            "Previews key the frame alone, without Time Awareness's neighbouring frames: much "
            "faster while you dial in settings, but it can differ slightly from the final key. "
            "F12 and the full Render always use your full settings"
        ),
        default=True,
    )
    cache_ahead: bpy.props.BoolProperty(
        name="Cache Ahead", default=False,
        description=(
            "With Live on: while the playhead rests, keep previewing the frames around it, "
            "nearest first and outwards, like After Effects' background caching (green on the timeline)"
        ),
    )
    cache_radius: bpy.props.IntProperty(
        name="Frames Either Side", default=48, min=1, max=2000,
        description="How far from the playhead Cache Ahead goes",
    )
    cache_batch: bpy.props.IntProperty(
        name="Frames per Job", default=8, min=1, max=64,
        description="Frames keyed together: more is faster overall, fewer shows the nearest ones sooner",
    )
    render_scope: bpy.props.EnumProperty(
        name="Render",
        items=[("SHOT", "Whole Shot", "Key every frame of the shot into a new Render folder"),
               ("RANGE", "Some Frames", "Key or preview just the frames From-To: to fix or test a stretch")],
        default="SHOT",
    )
    range_start: bpy.props.IntProperty(name="From", default=1, description="First frame to render")
    range_end: bpy.props.IntProperty(name="To", default=10, description="Last frame to render")
    range_into: bpy.props.EnumProperty(
        name="Into",
        items=[("CURRENT", "Current Key", "Replace these frames in the key the node shows"),
               ("NEW", "New Test Folder", "Write them to a new Render N-range folder, leaving the key alone")],
        default="CURRENT",
    )
    hint_prompt: bpy.props.StringProperty(
        name="Find", default="person",
        description="What SAM 3 cuts out, in words: person, man in a suit, dog, red car...",
    )
    hint_mode: bpy.props.EnumProperty(
        name="Hint",
        items=[("SINGLE", "Single Frame", "A matte for one frame: CorridorKey carries it through the shot"),
               ("ALL", "All Frames", "SAM 3 tracks the subject through the shot: a matte for every frame")],
        default="ALL",
    )
    hint_range: bpy.props.EnumProperty(
        name="Frames",
        items=[("SHOT", "Whole Shot", "Every frame of the footage"),
               ("PREVIEW", "Preview Range", "The timeline's Preview Range")],
        default="SHOT",
    )
    hint_threshold: bpy.props.FloatProperty(
        name="Confidence", default=0.5, min=0.05, max=0.95,
        description="How sure SAM 3 must be: lower finds more, higher finds less",
    )
    screen_pick: bpy.props.FloatVectorProperty(
        name="Screen Colour", subtype="COLOR", size=3, min=0.0, soft_max=1.0,
        default=(0.033, 0.343, 0.051),  # a typical green screen (Green fills it in again)
        description="The screen's colour: click it and use the eyedropper on the screen in your "
                    "footage, or pick Green or Blue",
    )
    screen_threshold: bpy.props.FloatProperty(
        name="Threshold", default=0.04, min=0.0, max=0.5, step=0.5, precision=3,
        description="How green (or blue) a pixel must be before the screen key starts taking it "
                    "out: lower takes out more, including shadowed screen",
    )
    screen_softness: bpy.props.FloatProperty(
        name="Softness", default=0.08, min=0.001, max=0.5, step=0.5, precision=3,
        description="How gradually the screen key fades from kept to taken out: higher gives softer edges",
    )
    screen_clean: bpy.props.BoolProperty(
        name="Clean the Key",
        description="Take the screen colour out of the key live, inside this node: the key's alpha "
                    "times the screen key. Renders, and locks for a render farm, the same way",
        default=False, update=lambda self, _c: setattr(self, "ui_screen", True) if self.screen_clean else None,
    )
    ui_inputs: bpy.props.BoolProperty(name="Footage", default=True)
    ui_hint: bpy.props.BoolProperty(name="Alpha Hint", default=True)
    ui_sam3: bpy.props.BoolProperty(name="Alpha Hint Maker (SAM 3)", default=False)
    ui_screen: bpy.props.BoolProperty(name="Screen Key", default=False)
    ui_preview: bpy.props.BoolProperty(name="Preview", default=True)
    ui_render: bpy.props.BoolProperty(name="Render", default=True)
    ui_tips: bpy.props.BoolProperty(name="Tips", default=True)
    ui_floor: bpy.props.BoolProperty(name="Floor Shadows", default=True)
    floor_name: bpy.props.StringProperty(options={"HIDDEN"})
    screen_show: bpy.props.BoolProperty(
        name="Show Matte",
        description="Show the screen key itself (white kept, black taken out) while you tune it",
        default=False,
    )


def node_by_source(uid: str) -> bpy.types.Node | None:
    return next(
        (n for n in shader_nodes() if uid and uid in (n.node_tree.get(SOURCE), n.node_tree.get(SOURCE_FREE))),
        None,
    )


class _OnSource:
    """An operator that acts on the CorridorKey node whose source id it carries."""

    uid: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    def source(self):
        return node_by_uid(self.uid)


class CORRIDORKEY_SHADER_OT_preview(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.preview"
    bl_label = "Preview Frame"
    bl_description = "Key just the frame you're on, so you can judge the settings before a full render"

    def execute(self, context):
        src = self.source()
        if src is None:
            return {"CANCELLED"}
        why = start_preview(src, context.scene, [context.scene.frame_current])
        if why:
            PREVIEW_LINE[self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_preview_range(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.preview_range"
    bl_label = "Preview Frames"
    bl_description = (
        "Preview the frames From-To in one go (CorridorKey loads once), so you can play them "
        "back. Frames already previewed are skipped; nothing is written to your key"
    )

    def execute(self, context):
        src = self.source()
        node = node_by_source(self.uid)
        if src is None or node is None:
            return {"CANCELLED"}
        scene = context.scene
        first, last = sorted((node.corridorkey_shader.range_start, node.corridorkey_shader.range_end))
        frames = list(range(first, last + 1))
        why = start_preview(src, scene, frames)
        if why:
            PREVIEW_LINE[self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        if self.uid not in PREVIEW_JOBS:
            PREVIEW_LINE[self.uid] = "Every frame in the range is already previewed."
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_clear_previews(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.clear_previews"
    bl_label = "Clear Previews"
    bl_description = "Delete this node's cached previews (a full render's key is kept)"
    bl_options = {"REGISTER"}

    def execute(self, context):
        clear_previews(self.uid)
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_cancel_preview(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.cancel_preview"
    bl_label = "Stop Preview"
    bl_description = "Stop keying the preview frame"

    def execute(self, context):
        cancel_preview(self.uid)
        return {"FINISHED"}


def scene_nodes(scene: bpy.types.Scene) -> list[bpy.types.Node]:
    """The CorridorKey 2.0 nodes this scene renders: on its renderable objects and its world."""
    trees, found = [], []
    for obj in scene.objects:
        if obj.hide_render:
            continue
        for slot in getattr(obj, "material_slots", ()):
            if slot.material is not None and slot.material.node_tree is not None:
                trees.append(slot.material.node_tree)
    if scene.world is not None and scene.world.node_tree is not None:
        trees.append(scene.world.node_tree)
    for tree in trees:
        for node in tree.nodes:
            if is_ck_shader(node) and node not in found:
                found.append(node)
    return found


def cached_at(src: bpy.types.Node, scene: bpy.types.Scene, frame: int) -> bool:
    key, _why = preview_key(src, scene, frame)
    if key is None:
        return False
    number, controls, hint, folder = key
    return number in _frames_in(cache_dir(st(src).uid, controls, hint, folder))


def to_trash(path: str) -> bool:
    """Move a folder to the Recycle Bin (Windows) or the Trash (macOS); nothing else deletes."""
    path = os.path.abspath(path)
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16),
                ("fAnyOperationsAborted", wintypes.BOOL), ("hNameMappings", ctypes.c_void_p),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        flags = 0x0040 | 0x0010 | 0x0004 | 0x0400  # undo (Recycle Bin), no confirm, silent, no error UI
        source = ctypes.create_unicode_buffer(path + "\0")  # the list ends with a second null
        op = SHFILEOPSTRUCTW(None, 3, ctypes.cast(source, wintypes.LPCWSTR), None, flags, False, None, None)
        code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        return code == 0 and not os.path.exists(path)
    if sys.platform == "darwin":
        trash = os.path.expanduser("~/.Trash")
        target = os.path.join(trash, os.path.basename(path))
        n = 1
        while os.path.exists(target):
            target = os.path.join(trash, f"{os.path.basename(path)} {n}")
            n += 1
        shutil.move(path, target)
        return True
    return False


def render_folders(src: bpy.types.Node) -> list[str]:
    """The finished render(s) this node shows or keeps, only inside its own Render Folder."""
    state = st(src)
    root = bpy.path.abspath(state.render_folder) if state.render_folder else ""
    found = []
    for folder in (state.result_folder, state.kept_render):
        if not folder or not os.path.isdir(folder):
            continue
        folder = os.path.abspath(folder)
        inside = root and os.path.commonpath([folder, os.path.abspath(root)]) == os.path.abspath(root)
        if inside and folder != os.path.abspath(root) and folder not in found:
            found.append(folder)
    return found


class CORRIDORKEY_SHADER_OT_delete_render(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.delete_render"
    bl_label = "Delete Render"
    bl_description = (
        "Move this node's full CorridorKey render to the Recycle Bin. The node shows your footage "
        "again until you render. Previews are kept"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def invoke(self, context, event):
        src = self.source()
        folders = render_folders(src) if src is not None else []
        if not folders:
            self.report({"INFO"}, "This node has no finished render to delete.")
            return {"CANCELLED"}
        names = ", ".join(os.path.basename(f) for f in folders)
        return context.window_manager.invoke_confirm(
            self, event, title="Delete CorridorKey render?",
            message=f"Moves {names} to the Recycle Bin.", confirm_text="Delete", icon="WARNING",
        )

    def execute(self, context):
        src = self.source()
        if src is None:
            return {"CANCELLED"}
        ck_ops, ck_words = ck("operators"), ck("words")
        if full_render_running(st(src).uid):
            self.report({"WARNING"}, "Cancel the running render first.")
            return {"CANCELLED"}
        # let go of the frames first, so nothing shows a missing image
        tree = src.node_tree
        images = [n.image for n in tree.nodes if n.name == RESULT and n.image is not None]
        for n in [n for n in tree.nodes if n.name == RESULT]:
            tree.nodes.remove(n)
        apply_display(src)
        for node in list(shader_nodes()):
            if source_of(node) == src:
                texture = node.node_tree.nodes.get(TEXTURE)
                if texture is not None and texture.image in images:
                    texture.image = None
        for image in images:
            if image.users == 0:
                bpy.data.images.remove(image)
        failed = [f for f in render_folders(src) if not to_trash(f)]
        state = st(src)
        state.result_folder = ""
        state.kept_render = state.kept_render_owner = state.kept_render_stamp = ""
        if is_free_source(src):
            state.render_line = "Not rendered"
        else:
            state.render_line = getattr(ck_words, "NOT_RENDERED", "")
            if hasattr(ck_ops, "queue_folder_size"):
                ck_ops.queue_folder_size(state.uid)
        _LISTED.clear()
        for node in list(shader_nodes()):
            other = source_of(node)
            if other is not None:
                sync(node, other, context.scene)
        if failed:
            self.report({"WARNING"}, "Couldn't move to the Recycle Bin: " + ", ".join(failed))
            return {"CANCELLED"}
        self.report({"INFO"}, "Render moved to the Recycle Bin.")
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_render_image(bpy.types.Operator):
    bl_idname = "corridorkey_shader.render_image"
    bl_label = f"Render Image ({NAME})"
    bl_description = (
        "Key the frame you're on with CorridorKey first (or reuse its cached preview), "
        "then render it. Esc while it keys cancels"
    )

    def _plain_render(self):
        bpy.ops.render.render("INVOKE_DEFAULT")
        return {"FINISHED"}

    def invoke(self, context, event):
        scene = context.scene
        if (
            not scene.corridorkey_key_before_render
            or not engine_available()
            or bpy.app.is_job_running("RENDER")
        ):
            return self._plain_render()
        self.frame = scene.frame_current
        self.queue, self.started, self.notes = [], set(), []
        for node in scene_nodes(scene):
            src = source_of(node)
            if src is None:
                continue
            key, _why = preview_key(src, scene, self.frame)
            if key is not None and st(src).uid not in self.queue:
                self.queue.append(st(src).uid)
        if not self.queue:
            return self._plain_render()
        self._timer = context.window_manager.event_timer_add(0.2, window=context.window)
        context.window_manager.modal_handler_add(self)
        self._status(context)
        return {"RUNNING_MODAL"}

    def _status(self, context, text: str | None = None) -> None:
        if text is None:
            text = f"{NAME}: keying frame {self.frame} before rendering (Esc to cancel)"
        context.workspace.status_text_set(text)

    def _finish(self, context) -> None:
        context.window_manager.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type == "ESC":
            for uid in self.started:
                cancel_preview(uid)
            self._finish(context)
            self.report({"INFO"}, f"{NAME}: render cancelled")
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        land_previews()
        scene = context.scene
        while self.queue:
            uid = self.queue[0]
            src = node_by_uid(uid)
            if src is None or cached_at(src, scene, self.frame):
                self.queue.pop(0)
                continue
            if PREVIEW_JOBS:  # this node or another is keying (one model on the card at a time)
                break
            if uid in self.started:  # it ended without a frame
                self.notes.append(PREVIEW_LINE.get(uid, "a node couldn't be keyed"))
                self.queue.pop(0)
                continue
            why = start_preview(src, scene, [self.frame], full=True)
            self.started.add(uid)
            if why:
                self.notes.append(why)
                self.queue.pop(0)
                continue
            break
        if self.queue:
            uid = self.queue[0]
            line = PREVIEW_LINE.get(uid, "")
            self._status(context, f"{NAME}: keying frame {self.frame} before rendering, {line} (Esc to cancel)")
            return {"RUNNING_MODAL"}

        self._finish(context)
        if scene.frame_current != self.frame:
            scene.frame_set(self.frame)
        _RENDERING_WITH_KEY[0] = True
        for node in list(shader_nodes()):
            src = source_of(node)
            if src is not None:
                sync(node, src, scene)
        if self.notes:
            self.report({"WARNING"}, f"{NAME}: " + "; ".join(n for n in self.notes if n))
        bpy.ops.render.render("INVOKE_DEFAULT")
        if not bpy.app.is_job_running("RENDER"):  # the render didn't start
            _RENDERING_WITH_KEY[0] = False
        return {"FINISHED"}


# Corridor's settings a Reset leaves alone: where renders go, and which strip is a hint.
_KEEP_ON_RESET = {"render_folder", "hint_strip"}


class CORRIDORKEY_SHADER_OT_reset(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.reset"
    bl_label = "Reset"
    bl_description = (
        "Put every CorridorKey setting back to its default and clear the previews. "
        "Your footage, Render Folder and any finished key are kept"
    )
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, title="Reset CorridorKey?", message="Settings go back to defaults.", confirm_text="Reset"
        )

    def execute(self, context):
        src = self.source()
        if src is None:
            return {"CANCELLED"}
        state = st(src)
        for prop in state.bl_rna.properties:
            name = prop.identifier
            if name == "rna_type" or name in _KEEP_ON_RESET or prop.is_hidden or prop.is_readonly:
                continue
            state.property_unset(name)
        despill_default(src)
        apply_display(src)  # "Display rendered frames?" is ticked again
        ck_ops = ck("operators")
        if not is_free_source(src) and hasattr(ck_ops, "queue_estimate"):
            ck_ops.queue_estimate(state.uid)
        clear_previews(state.uid)
        node = node_by_source(state.uid)
        if node is not None:
            node.corridorkey_shader.live_preview = False
        return {"FINISHED"}


# --- sidebars (Shader Editor and 3D viewport) --------------------------------------------------------


# --- first-time help: the Getting Started checklist and the warnings ---------------------------------

GUIDE = "QUICKSTART.html"


def _lines(layout, text: str, icon: str = "NONE", width: int = 42) -> None:
    """A long message as several labels, the icon on the first only."""
    for i, part in enumerate(textwrap.wrap(text, width) or [""]):
        layout.label(text=part, icon=icon if i == 0 else ("BLANK1" if icon != "NONE" else "NONE"))


_FRACTION = re.compile(r"(\d+)\s*(?:of|/)\s*(\d+)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def progress_of(line: str) -> float | None:
    """How far along a status line says a job is ("12 of 240", "40%"), 0-1, or None."""
    found = _FRACTION.search(line or "")
    if found and int(found.group(2)) > 0:
        return min(1.0, int(found.group(1)) / int(found.group(2)))
    found = _PERCENT.search(line or "")
    return min(1.0, float(found.group(1)) / 100.0) if found else None


def draw_progress(layout, line: str, factor: float | None = None, icon: str = "TIME") -> None:
    """A running job: a filling bar when its progress is known, its line of text otherwise."""
    factor = progress_of(line) if factor is None else factor
    if factor is None or not hasattr(layout, "progress"):
        _lines(layout, line, icon, 34)
        return
    if len(line) <= 38:
        layout.progress(factor=factor, type="BAR", text=line)
        return
    found = _FRACTION.search(line)
    _lines(layout, line, icon, 34)
    layout.progress(factor=factor, type="BAR",
                    text=f"{found.group(1)} of {found.group(2)}" if found else f"{round(factor * 100)}%")


class CORRIDORKEY_SHADER_OT_pick_render_folder(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.pick_render_folder"
    bl_label = "Choose Render Folder"
    bl_description = "Choose the folder the rendered key is saved in"

    directory: bpy.props.StringProperty(subtype="DIR_PATH", options={"SKIP_SAVE"})

    def invoke(self, context, event):
        src = self.source()
        if src is not None and st(src).render_folder:
            self.directory = bpy.path.abspath(st(src).render_folder)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        src = self.source()
        if src is None or not self.directory:
            return {"CANCELLED"}
        folder = self.directory
        if bpy.data.filepath:
            try:  # keep it relative to the .blend, so the project can move
                folder = bpy.path.relpath(folder)
            except ValueError:  # another drive
                pass
        st(src).render_folder = folder
        return {"FINISHED"}


def blend_problem() -> str | None:
    """Why the .blend isn't a safe home for a key yet, or None."""
    path = bpy.data.filepath
    if not path:
        return "unsaved"
    folder = os.path.normcase(os.path.abspath(os.path.dirname(path)))
    temp = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
    name = os.path.basename(path).lower()
    if folder == temp or folder.startswith(temp + os.sep) or "autosave" in name or name == "quit.blend":
        return "temp"
    return None


def _image_path(image) -> str:
    return bpy.path.abspath(image.filepath, library=image.library) if image is not None else ""


def image_missing(image) -> bool:
    """An image that's neither packed nor on disk (a sequence: no frames in its folder)."""
    if image is None or image.packed_file is not None or image.source in {"GENERATED", "VIEWER"}:
        return False
    path = _image_path(image)
    if not image.filepath:
        return True
    if image.source == "SEQUENCE":
        return not _frames_in(os.path.dirname(path))
    return not os.path.isfile(path)


def image_absolute(image) -> bool:
    """A file path that breaks when the project folder moves (and could be relative)."""
    if image is None or image.packed_file is not None or not bpy.data.is_saved or not image.filepath:
        return False
    if image.filepath.startswith("//") or image.library is not None:
        return False
    here = os.path.splitdrive(os.path.abspath(bpy.data.filepath))[0].lower()
    there = os.path.splitdrive(_image_path(image))[0].lower()
    return here == there


def render_folder_of(src) -> str:
    folder = st(src).render_folder
    return os.path.normpath(bpy.path.abspath(folder)) if folder else ""


def checklist(src) -> list[tuple[str, bool, str]]:
    """(item, done, how it's done or what to do) for the four things a first key needs."""
    items = []
    if engine() == "free":
        problem = free_setup_problem()
        items.append(("Engine: free CorridorKey 2.0", problem is None, problem or "installed"))
    else:
        items.append(("Engine: Corridor's add-on", ck() is not None, "enabled" if ck() else "not enabled"))
    footage = wired_image_node(src, 0)
    image = footage.image if footage is not None else None
    if image is None:
        items.append(("Footage", False, "plug an image sequence or a movie into Image"))
    elif image_missing(image):
        items.append(("Footage", False, f"{image.name}: its files aren't where the .blend says"))
    else:
        items.append(("Footage", True, image.name))
    saved = blend_problem()
    items.append((
        "File saved", saved is None,
        {"unsaved": "save the .blend next to your footage",
         "temp": "this is an autosave in Temp: save it with your project"}.get(saved, os.path.basename(bpy.data.filepath)),
    ))
    folder = st(src).render_folder
    if not folder and saved == "unsaved":
        items.append(("Render Folder", False, "made next to the .blend when you save it, or choose one"))
    elif not folder:
        items.append(("Render Folder", False, "choose where keys are saved"))
    elif folder.startswith("//") and saved is not None:
        items.append(("Render Folder", False, "it's next to the .blend: save the file first"))
    else:
        items.append(("Render Folder", True, render_folder_of(src)))
    return items


def draw_getting_started(layout, node, src, context, in_shader_editor: bool) -> None:
    items = checklist(src)
    if all(done for _item, done, _note in items):
        return
    box = layout.box()
    row = box.row()
    row.label(text="Getting Started", icon="INFO")
    row.operator(CORRIDORKEY_SHADER_OT_open_guide.bl_idname, text="", icon="QUESTION")
    for item, done, note in items:
        col = box.column(align=True)
        col.label(text=item, icon="CHECKBOX_HLT" if done else "CHECKBOX_DEHLT")
        if done:
            continue
        _lines(col, note, "BLANK1", 34)
        if item.startswith("Engine"):
            if engine() == "free":
                col.operator(CORRIDORKEY_SHADER_OT_open_prefs.bl_idname, text="Install CorridorKey 2.0 (free)",
                             icon="IMPORT")
            else:
                col.operator(CORRIDORKEY_SHADER_OT_open_prefs.bl_idname, text="Pick an Engine", icon="PREFERENCES")
        elif item == "Footage":
            footage = wired_image_node(src, 0)
            if footage is not None and footage.image is not None:
                col.operator("file.find_missing_files", text="Find Missing Files", icon="VIEWZOOM")
            elif in_shader_editor:
                col.operator(CORRIDORKEY_SHADER_OT_connect.bl_idname, icon="LINKED")
        elif item == "File saved":
            col.operator("wm.save_as_mainfile", text="Save As...", icon="FILE_TICK")
        elif item == "Render Folder" and not st(src).render_folder:
            col.prop(st(src), "render_folder", text="")


def warnings(src, scene) -> list[tuple[str, str, str | None, str]]:
    """(icon, message, operator to fix it, its button text): what would spoil a key or a render."""
    found = []
    hint = wired_image_node(src, 1)
    footage = wired_image_node(src, 0)
    for label, image_node in (("Footage", footage), ("Alpha Hint", hint)):
        image = image_node.image if image_node is not None else None
        if image is None:
            continue
        if label == "Alpha Hint" and image_missing(image):
            found.append(("ERROR", f"Alpha Hint '{image.name}' isn't on disk: the key would ignore it.",
                          "file.find_missing_files", "Find Missing Files"))
        elif image_absolute(image) and not image_missing(image):
            found.append(("INFO", f"{label} uses a full path: moving the project folder will lose it "
                          "(and show pink frames).", "file.make_paths_relative", "Make Paths Relative"))
    result = src.node_tree.nodes.get(RESULT) if src.node_tree is not None else None
    if result is not None and result.image is not None and image_missing(result.image):
        found.append(("ERROR", "The rendered key's folder is gone (moved or deleted): the node shows "
                      "the footage until you render again.", None, ""))
    if scene.render.use_compositing and _autoshot_compositor(scene):
        found.append(("ERROR", "The compositor keys the raw plate again and will cover CorridorKey's "
                      "key in your renders.", CORRIDORKEY_SHADER_OT_compositing_off.bl_idname,
                      "Turn Off Compositing"))
    return found


def draw_warnings(layout, src, scene) -> None:
    for icon, message, operator, text in warnings(src, scene):
        box = layout.box()
        _lines(box, message, icon, 34)
        if operator:
            box.operator(operator, text=text)


class CORRIDORKEY_SHADER_OT_open_guide(bpy.types.Operator):
    bl_idname = "corridorkey_shader.open_guide"
    bl_label = "Quick Start Guide"
    bl_description = "Open the CorridorKey 2.0 quick-start guide in your browser"

    def execute(self, context):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), GUIDE)
        if not os.path.isfile(path):
            self.report({"WARNING"}, "The guide isn't installed with this copy of the add-on.")
            return {"CANCELLED"}
        bpy.ops.wm.url_open(url="file:///" + path.replace("\\", "/").lstrip("/"))
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_compositing_off(bpy.types.Operator):
    bl_idname = "corridorkey_shader.compositing_off"
    bl_label = "Turn Off Compositing"
    bl_description = (
        "Render without the compositor (Output Properties > Post Processing > Compositing), so its "
        "key of the raw plate doesn't cover CorridorKey's"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.render.use_compositing = False
        return {"FINISHED"}


def draw_node(layout: bpy.types.UILayout, node: bpy.types.Node, context, in_shader_editor: bool) -> None:
    preferences = prefs()
    if preferences is not None:
        row = layout.row(align=True)
        row.prop(preferences, "engine", text="Engine")
        row.operator(CORRIDORKEY_SHADER_OT_open_prefs.bl_idname, text="", icon="PREFERENCES")
        row.operator(CORRIDORKEY_SHADER_OT_open_guide.bl_idname, text="", icon="QUESTION")
    ck_panel = ck("panel")
    if engine() == "paid" and ck_panel is None:
        layout.label(text="Corridor's CorridorKey add-on isn't enabled.", icon="ERROR")
        layout.label(text="Enable it, or set Engine to Free (GitHub).")
        return
    src = source_of(node)
    footage = wired_image_node(src, 0) if src is not None else None
    hint = wired_image_node(src, 1) if src is not None else None
    if src is None or footage is None or hint is None:
        layout.label(text="This node's CorridorKey source is missing.", icon="ERROR")
        if in_shader_editor:
            layout.operator(CORRIDORKEY_SHADER_OT_repair.bl_idname)
        else:
            layout.label(text="Select the node in the Shader Editor to repair it.")
        return
    uid = st(src).uid
    draw_getting_started(layout, node, src, context, in_shader_editor)
    draw_warnings(layout, src, context.scene)

    scene = context.scene
    previewing = uid in PREVIEW_JOBS and not PREVIEW_JOBS[uid].get("dest")
    settings = node.corridorkey_shader
    note = "keying" if previewing else (
        ("live + cache ahead" if settings.cache_ahead else "live") if settings.live_preview else "")
    def reset_button(header):
        header.operator(CORRIDORKEY_SHADER_OT_reset.bl_idname, text="", icon="LOOP_BACK", emboss=False).uid = uid

    box = _fold(layout, settings, "ui_preview", "Preview", "HIDE_OFF", note, reset_button)
    row = box.row(align=True)
    row.scale_y = 1.5
    if previewing:
        row.operator(CORRIDORKEY_SHADER_OT_cancel_preview.bl_idname, icon="CANCEL").uid = uid
    else:
        row.operator(CORRIDORKEY_SHADER_OT_preview.bl_idname, icon="RENDER_STILL").uid = uid
    row.prop(node.corridorkey_shader, "live_preview", text="Live", toggle=True, icon="REC")
    if PREVIEW_LINE.get(uid) and uid not in RANGE_LINES:
        if previewing:
            draw_progress(box, PREVIEW_LINE[uid])
        else:
            _lines(box, PREVIEW_LINE[uid], "INFO", 34)
    box.prop(node.corridorkey_shader, "quick_preview")
    row = box.row(align=True)
    row.enabled = settings.live_preview  # caching works with Live
    row.prop(node.corridorkey_shader, "cache_ahead", toggle=True, icon="FF")
    sub = row.row(align=True)
    sub.enabled = settings.live_preview and settings.cache_ahead
    sub.prop(node.corridorkey_shader, "cache_radius", text="Radius")
    sub.prop(node.corridorkey_shader, "cache_batch", text="Batch")
    cached = len(cached_frames(src, scene))
    if cached:
        row = box.row()
        row.label(text=f"{cached} frame{'s' if cached != 1 else ''} cached for these settings")
        row.operator(CORRIDORKEY_SHADER_OT_clear_previews.bl_idname, text="", icon="TRASH").uid = uid

    draw_render(layout, node, src, scene)

    inputs = _fold(layout, node.corridorkey_shader, "ui_inputs", "Footage", "IMAGE_DATA")
    _draw_input(inputs, node, IMAGE_IN, footage)
    if in_shader_editor:
        inputs.operator(CORRIDORKEY_SHADER_OT_connect.bl_idname, icon="LINKED")
    draw_floor_shadows(layout, node, src, scene)
    layout.separator()

    # The engine's own settings sections, drawn for the hidden compositor node (their Render
    # section is in the Render box above). Ours slot in between theirs: the Alpha Hint above
    # Despill, the Tips above Help.
    hooks = {"before_despill": lambda where: draw_alpha_hint(where, node, src, scene, hint),
             "before_help": lambda where: draw_tips(where, node, src, scene)}
    if is_free_source(src):
        draw_free_controls(layout, src, **hooks)
    else:
        draw_paid_controls(layout, src, scene, **hooks)


def engine_column(layout, src):
    """A column for the engine's own sections. Their buttons act on the editor's active node, so
    the hidden source is handed to them as that node."""
    column = layout.column()
    column.context_pointer_set("active_node", src)
    column.use_property_split = True
    column.use_property_decorate = False
    return column


def draw_tips(layout, node, src, scene) -> None:
    """What the panel's colours and waits mean, in a section of their own."""
    uid = st(src).uid
    body = _fold(layout, node.corridorkey_shader, "ui_tips", "Tips", "INFO")
    if not cached_frames(src, scene) and uid not in PREVIEW_JOBS and not warm_alive():
        note = body.column(align=True)
        note.label(text="First preview: about 25 s", icon="TIME")
        note.label(text="(the model loads). Then ~1 s a frame.", icon="BLANK1")
    if shown_preview(src, scene) is not None:
        body.label(text="Showing the preview for this frame", icon="HIDE_OFF")
    body.label(text="Timeline: blue = key, green = preview", icon="TIME")
    body.label(text=warm_status(), icon="CHECKMARK" if warm_alive() else "SORTTIME")


def _draw_input(layout, node, name: str, image_node) -> None:
    """One of the node's inputs: the Image Texture plugged in, or the image picked here."""
    texture, plugged = upstream_texture(node, name)
    if texture is not None:
        shown = texture.image.name if texture.image is not None else "no image"
        layout.label(text=f"{shown} (plugged in)", icon="LINKED")
    elif plugged:
        layout.label(text=f"Plug an Image Texture node into {name}", icon="ERROR")
    else:
        layout.template_ID(image_node, "image", open="image.open")


def draw_alpha_hint(layout, node, src, scene, hint) -> None:
    """Everything about the Alpha Hint, at the bottom: the hint itself, the engine's hint
    settings, and the two ways to make or improve one (SAM 3, the screen key)."""
    settings = node.corridorkey_shader
    texture, plugged = upstream_texture(node, HINT_IN)
    image = texture.image if texture is not None else (hint.image if hint is not None and not plugged else None)
    busy = ("SAM 3 running" if st(src).uid in HINT_JOBS else
            "baking the screen key" if st(src).uid in SCREEN_JOBS else
            "screen key on" if settings.screen_clean else "")
    body = _fold(layout, settings, "ui_hint", "Alpha Hint", "MOD_MASK", busy or ("none" if image is None else ""))
    _draw_input(body, node, HINT_IN, hint)
    engine_settings = body.column()
    engine_settings.use_property_split = True
    engine_settings.use_property_decorate = False
    engine_settings.context_pointer_set("active_node", src)
    if is_free_source(src):
        draw_free_hint_settings(engine_settings, src)
    else:
        draw_paid_hint_settings(engine_settings, src)
    draw_hint_maker(body, node, src)
    draw_screen_key(body, node, src)


class CORRIDORKEY_SHADER_PT_node(bpy.types.Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Node"
    bl_label = NAME

    @classmethod
    def poll(cls, context) -> bool:
        node = getattr(context, "active_node", None)
        return _in_shader_editor(context) and (is_ck_shader(node) or is_locked(node))

    def draw(self, context):
        if is_locked(context.active_node):
            draw_locked(self.layout, context.active_node)
            return
        draw_node(self.layout, context.active_node, context, in_shader_editor=True)


def object_nodes(obj: bpy.types.Object | None) -> list[bpy.types.Node]:
    """The CorridorKey nodes on an object's materials, the active material's first."""
    if obj is None:
        return []
    slots = list(getattr(obj, "material_slots", ()))
    active = getattr(obj, "active_material_index", 0)
    ordered = slots[active : active + 1] + slots[:active] + slots[active + 1 :]
    found = []
    for slot in ordered:
        material = slot.material
        if material is not None and material.node_tree is not None:
            found += [n for n in material.node_tree.nodes if is_ck_shader(n)]
    return found


class CORRIDORKEY_SHADER_PT_viewport(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = NAME
    bl_label = NAME

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        nodes = object_nodes(obj)
        locked = locked_nodes(obj) if obj is not None else []
        for node in locked:
            draw_locked(layout, node)
        if len(locked_nodes()) > 1:
            layout.operator(CORRIDORKEY_SHADER_OT_unlock.bl_idname, text="Unlock All", icon="UNLOCKED").every = True
        if locked and not nodes:
            return
        if not nodes:
            if obj is not None and _material_of(obj) is not None:
                layout.label(text=f"{obj.name} shows footage.", icon="IMAGE_DATA")
                column = layout.column()
                column.scale_y = 1.6
                column.operator(CORRIDORKEY_SHADER_OT_setup_plane.bl_idname, icon="NODE_MATERIAL")
                layout.label(text="Puts CorridorKey between its footage and shader.")
            else:
                layout.label(text="Select a footage plane (a material")
                layout.label(text="with an Image Texture) to set it up.")
            preferences = prefs()
            if preferences is not None:
                layout.prop(preferences, "engine", text="Engine")
            layout.operator(CORRIDORKEY_SHADER_OT_open_guide.bl_idname, icon="QUESTION")
            return
        node = nodes[0]
        layout.label(text=_owner_name(node), icon="MATERIAL")
        if len(nodes) > 1:
            layout.label(text=f"{len(nodes) - 1} more CorridorKey node(s) on this object", icon="INFO")
        draw_node(layout, node, context, in_shader_editor=False)


def _owner_name(node: bpy.types.Node) -> str:
    tree = node.id_data
    for material in bpy.data.materials:
        if material.node_tree == tree:
            return material.name
    return tree.name


# --- render a range of frames ------------------------------------------------------------------------


class CORRIDORKEY_SHADER_OT_render_range(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.render_range"
    bl_label = "Render Range"
    bl_description = (
        "Key just these frames with your full settings: to fix frames in the current key, or to "
        "test settings on a stretch of the shot"
    )

    def execute(self, context):
        src = self.source()
        node = node_by_source(self.uid)
        if src is None or node is None:
            return {"CANCELLED"}
        settings = node.corridorkey_shader
        first, last = sorted((settings.range_start, settings.range_end))
        if full_render_running(self.uid):
            self.report({"WARNING"}, "A full render is running.")
            return {"CANCELLED"}
        if self.uid in PREVIEW_JOBS:
            PREVIEW_JOBS[self.uid]["job"].cancel()
            land_previews()
        dest, why = range_destination(src, settings.range_into)
        if dest is None:
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        why = start_preview(src, context.scene, list(range(first, last + 1)), full=True, dest=dest)
        if why:
            PREVIEW_LINE[self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_range_from_timeline(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.range_from_timeline"
    bl_label = "From Timeline"
    bl_description = "Use the timeline's Preview Range (or the frame you're on)"

    def execute(self, context):
        node = node_by_source(self.uid)
        if node is None:
            return {"CANCELLED"}
        scene, settings = context.scene, node.corridorkey_shader
        settings.render_scope = "RANGE"
        if scene.use_preview_range:
            settings.range_start, settings.range_end = scene.frame_preview_start, scene.frame_preview_end
        else:
            settings.range_start = settings.range_end = scene.frame_current
        return {"FINISHED"}


# --- lock for a render farm: the key as plain Blender nodes ------------------------------------------

LOCKED = "corridorkey_locked"  # on a locked copy of the inner tree: the name of the tree it froze


def is_locked(node: bpy.types.Node | None) -> bool:
    return (
        node is not None
        and node.bl_idname == "ShaderNodeGroup"
        and node.node_tree is not None
        and bool(node.node_tree.get(LOCKED))
    )


def locked_nodes(obj=None):
    """Locked CorridorKey nodes: on an object's materials, or (None) in the whole file."""
    if obj is not None:
        trees = [s.material.node_tree for s in getattr(obj, "material_slots", ())
                 if s.material is not None and s.material.node_tree is not None]
    else:
        trees = list(_shader_trees())
    return [n for tree in trees for n in tree.nodes if is_locked(n)]


def _numbers_in_scene(image_node, scene) -> set[int]:
    """The frame numbers an image node plays over the scene's frame range."""
    numbers = set()
    for frame in range(scene.frame_start, scene.frame_end + 1):
        local = frame - image_node.frame_start
        if 0 <= local < max(image_node.frame_duration, 1):
            numbers.add(local + 1 + image_node.frame_offset)
    return numbers


def lock_check(src, scene):
    """(the key's Image node, None) when the key is whole over the scene's frame range, else
    (None, why not)."""
    result = src.node_tree.nodes.get(RESULT) if src.node_tree is not None else None
    if result is None or result.image is None:
        return None, "There's no rendered key yet: render the shot first."
    image = result.image
    if image_missing(image):
        return None, "The rendered key's folder is gone: render the shot again."
    if image.source != "SEQUENCE":
        return result, None
    keyed = _frames_in(os.path.dirname(_image_path(image)))
    plate = wired_image_node(src, 0)
    wanted = _numbers_in_scene(result, scene)
    if plate is not None and plate.image is not None and plate.image.source == "SEQUENCE":
        wanted &= set(_frames_in(os.path.dirname(_image_path(plate.image))))  # frames the footage has
    missing = sorted(wanted - keyed)
    if missing:
        spans = _spans(n - result.frame_offset - 1 + result.frame_start for n in missing)
        shown = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in spans[:3]) + ("..." if len(spans) > 3 else "")
        return None, (f"{len(missing)} frame{'s' if len(missing) != 1 else ''} of the scene aren't keyed "
                      f"({shown}): key them with Render Range, or narrow the scene's frame range.")
    return result, None


def lock_node(node, scene) -> tuple[bool, str]:
    """Freeze a node: its group becomes a plain copy that plays the finished key, which renders
    the same anywhere, with or without this add-on. The original waits in the file for Unlock."""
    src = source_of(node)
    if src is None:
        return False, "Its CorridorKey source is missing."
    uid = st(src).uid
    if uid in PREVIEW_JOBS or full_render_running(uid):
        return False, "It's still keying: wait for it to finish."
    result, why = lock_check(src, scene)
    if result is None:
        return False, why
    image = result.image
    notes = []
    path = _image_path(image)
    if bpy.data.is_saved and not image.filepath.startswith("//"):
        try:
            image.filepath = bpy.path.relpath(path)
        except ValueError:  # another drive: it stays a full path
            pass
    if image.filepath.startswith("//.."):
        notes.append("its key is outside the project folder: send that folder with the .blend")
    elif not image.filepath.startswith("//"):
        notes.append("its key has a full path: the farm needs the same path, or save the .blend near the key")

    original = node.node_tree
    locked = original.copy()
    for key in (MARK, SOURCE, SOURCE_FREE):
        if key in locked:
            del locked[key]
    locked[LOCKED] = original.name
    locked.name = f"CorridorKey Locked ({_owner_name(node)})"
    texture = locked.nodes[TEXTURE]
    texture.image = image
    user = texture.image_user
    user.frame_start, user.frame_offset = result.frame_start, result.frame_offset
    user.frame_duration = max(result.frame_duration, 1)
    user.use_auto_refresh, user.use_cyclic = True, False
    settings = node.corridorkey_shader
    if settings.screen_clean:  # the screen key goes with it: the farm renders the same clean key
        plate = wired_image_node(src, 0)
        screen_update(locked, settings, plate)
        if plate is not None and plate.image is not None and not plate.image.filepath.startswith("//"):
            notes.append("Clean the Key reads the footage too: the farm needs it at the same path")
    wire(locked, "key", types.SimpleNamespace(screen_clean=settings.screen_clean, screen_show=False))
    original.use_fake_user = True  # kept, unused, for Unlock (and so its source isn't tidied away)
    node.node_tree = locked
    node.label = f"{NAME} (locked)"
    settings.live_preview = settings.cache_ahead = False
    settings.screen_show = False
    return True, "; ".join(notes)


def unlock_node(node) -> bool:
    locked = node.node_tree
    original = bpy.data.node_groups.get(locked.get(LOCKED, "")) if locked is not None else None
    if original is None or not original.get(MARK):
        return False
    node.node_tree = original
    original.use_fake_user = False
    node.label = NAME
    if locked.users == 0:
        bpy.data.node_groups.remove(locked)
    return True


class CORRIDORKEY_SHADER_OT_lock(bpy.types.Operator):
    bl_idname = "corridorkey_shader.lock"
    bl_label = "Lock for Render Farm"
    bl_description = (
        "Freeze the finished key into plain Blender nodes, so this material renders the same on "
        "machines without the add-on (render farms). The add-on stops changing it until you unlock"
    )
    bl_options = {"REGISTER", "UNDO"}

    every: bpy.props.BoolProperty(name="Every CorridorKey node in the file", default=False)

    @classmethod
    def description(cls, context, properties):
        """Why the button is greyed, when it is: the node's lock check, in its own words."""
        src = getattr(context, "active_node", None)
        if not properties.every and src is not None and getattr(src, "node_tree", None) is not None:
            try:
                result, why = lock_check(src, context.scene)
            except Exception:  # noqa: BLE001 - a tooltip must never fail
                result, why = True, ""
            if result is None and why:
                return f"{why}\n\n{cls.bl_description}"
        return cls.bl_description

    def execute(self, context):
        if self.every:
            nodes = list(shader_nodes())
        else:
            nodes = object_nodes(context.active_object)[:1]
            active = getattr(context, "active_node", None)
            if is_ck_shader(active) and _in_shader_editor(context):
                nodes = [active]
        if not nodes:
            self.report({"WARNING"}, "No unlocked CorridorKey nodes to lock.")
            return {"CANCELLED"}
        done, failed, notes = 0, [], []
        for node in nodes:
            owner = _owner_name(node)
            ok, message = lock_node(node, context.scene)
            if ok:
                done += 1
                if message:
                    notes.append(f"{owner}: {message}")
            else:
                failed.append(f"{owner}: {message}")
        if context.scene.render.use_compositing and _autoshot_compositor(context.scene):
            notes.append("compositing is on and keys the raw plate: turn it off before sending")
        for line in failed + notes:
            self.report({"WARNING"}, line)
        if not done:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Locked {done} node{'s' if done != 1 else ''} for rendering anywhere.")
        _LISTED.clear()
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_unlock(bpy.types.Operator):
    bl_idname = "corridorkey_shader.unlock"
    bl_label = "Unlock"
    bl_description = "Hand the node back to CorridorKey 2.0: previews, renders and settings work again"
    bl_options = {"REGISTER", "UNDO"}

    every: bpy.props.BoolProperty(name="Every locked node in the file", default=False)

    def execute(self, context):
        if self.every:
            nodes = locked_nodes()
        else:
            nodes = locked_nodes(context.active_object)[:1]
            active = getattr(context, "active_node", None)
            if is_locked(active) and _in_shader_editor(context):
                nodes = [active]
        undone = sum(unlock_node(n) for n in nodes)
        if not undone:
            self.report({"WARNING"}, "Nothing to unlock (the original node is gone).")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Unlocked {undone} node{'s' if undone != 1 else ''}.")
        return {"FINISHED"}


def draw_locked(layout, node) -> None:
    box = layout.box()
    box.label(text=f"{_owner_name(node)}: locked", icon="LOCKED")
    tree = node.node_tree
    texture = tree.nodes.get(TEXTURE) if tree is not None else None
    image = texture.image if texture is not None else None
    if image is not None:
        box.label(text=os.path.basename(os.path.dirname(_image_path(image))) or image.name, icon="IMAGE_DATA")
        if image_missing(image):
            box.label(text="Its key's frames aren't on disk.", icon="ERROR")
    _lines(box, "Plain Blender nodes play the finished key: it renders the same on a render farm "
                "without this add-on.", "INFO", 34)
    box.operator(CORRIDORKEY_SHADER_OT_unlock.bl_idname, icon="UNLOCKED")


# --- the alpha-hint maker: SAM 3 ---------------------------------------------------------------------

SAM3_MIRROR = "cubert-gmbh/sam3"  # an ungated, byte-identical mirror of Meta's facebook/sam3
SAM3_REPO = "https://github.com/facebookresearch/sam3"
_SAM3_FRAMES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")


def sam3_root() -> str:
    chosen = getattr(prefs(), "sam3_root", "")
    if chosen:
        return os.path.normpath(bpy.path.abspath(chosen))
    return os.path.join(os.path.expanduser("~"), "CorridorKey2.0-SAM3")


def sam3_python(root: str) -> str:
    for parts in (("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
        candidate = os.path.join(root, *parts)
        if os.path.isfile(candidate):
            return candidate
    return ""


def sam3_problem() -> str | None:
    if sys.platform == "darwin":
        return "SAM 3 needs an NVIDIA GPU (Windows or Linux)."
    root = sam3_root()
    if not sam3_python(root):
        return "SAM 3 isn't installed yet: Install it in the preferences."
    if not os.path.isfile(os.path.join(root, "weights", "sam3.pt")):
        return "SAM 3's weights are missing: run Install in the preferences."
    return None


HINT_JOBS: dict[str, dict] = {}  # source uid -> the hint being made


def _hint_reader(job: dict) -> None:
    """The worker's progress, read off Blender's thread (no bpy here)."""
    process = job["process"]
    for raw in process.stdout:
        message = _parsed(raw) or {}
        event = message.get("event")
        if event == "loading":
            job["line"] = "loading SAM 3" + (f" (slow: {job['busy']})" if job.get("busy") else "")
        elif event == "stage":
            job["timing"][message.get("stage", "")] = float(message.get("seconds", 0))
            if message.get("stage") == "load":
                job["line"] = "reading the frames" if job["mode"] != "SINGLE" else "finding it"
        elif event == "memory":
            job["memory"] = message.get("frames", "")
            job["free_gb"] = message.get("free_gb")
        elif event == "progress":
            job["done"], job["total"] = message.get("done", 0), message.get("total", 0)
            where = f", frames on the {job['memory']}" if job.get("memory") == "GPU" else (
                ", frames in system RAM" if job.get("memory") else "")
            slow = f" (slow: {job['busy']})" if job.get("busy") else ""
            job["line"] = f"{job['done']} of {job['total']} frames{where}{slow}"
        elif event == "error":
            job["message"] = message.get("message", "")
        elif event == "done":
            job["empty"] = message.get("empty", 0)
            job["finished"] = True
    process.wait()
    if job.get("cancelled"):
        job["state"] = "cancelled"
    elif job.get("finished"):
        job["state"] = "done"
    else:
        job["state"] = "failed"
        job["message"] = job.get("message") or _log_tail(job["log"]) or "SAM 3 stopped."


def _hint_folder(prompt: str) -> str:
    root = bpy.path.abspath("//CorridorKey/Alpha Hints") if bpy.data.filepath else os.path.join(
        tempfile.gettempdir(), "CorridorKey Alpha Hints")
    os.makedirs(root, exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", prompt).strip() or "hint"
    taken = [int(m.group(1)) for f in os.listdir(root) if (m := re.match(rf"^{re.escape(name)} (\d+)$", f))]
    return os.path.join(root, f"{name} {max(taken, default=0) + 1}")


def start_hint(node, scene) -> str | None:
    """Make an alpha hint with SAM 3 for the node's footage, in the background."""
    problem = sam3_problem()
    if problem:
        return problem
    src = source_of(node)
    if src is None:
        return "This node has no CorridorKey source yet."
    uid = st(src).uid
    if uid in HINT_JOBS:
        return None
    if full_render_running() or any(entry.get("dest") for entry in PREVIEW_JOBS.values()):
        return "A CorridorKey render is running: make the hint when it's done."
    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None or image.source != "SEQUENCE":
        return "SAM 3 needs the footage as an image sequence."
    folder = os.path.dirname(bpy.path.abspath(image.filepath, library=image.library))
    plates = {n: f for n, f in numbered(folder).items() if f.lower().endswith(_SAM3_FRAMES)}
    if not plates:
        return "SAM 3 reads PNG, JPEG, TIFF or WebP frames (not EXR)."
    settings = node.corridorkey_shader
    to_number = lambda f: f - plate.frame_start + 1 + plate.frame_offset  # noqa: E731
    current = to_number(scene.frame_current)
    if settings.hint_mode == "SINGLE":
        if is_free_source(src):  # the free engine's single-frame hint is the shot's first frame
            current = min(plates)
        if current not in plates:
            return f"No footage frame at frame {scene.frame_current}."
        numbers = [current]
    elif settings.hint_range == "PREVIEW" and scene.use_preview_range:
        wanted = range(to_number(scene.frame_preview_start), to_number(scene.frame_preview_end) + 1)
        numbers = [n for n in wanted if n in plates]
    else:
        numbers = sorted(plates)
    if not numbers:
        return "No footage frames in that range."
    root = sam3_root()
    work = os.path.join(tempfile.gettempdir(), "CorridorKey Alpha Hints", uuid.uuid4().hex[:8])
    frames_dir = os.path.join(work, "frames")
    os.makedirs(frames_dir)
    for i, n in enumerate(numbers):
        _place(os.path.join(folder, plates[n]), frames_dir, f"{i:06d}{os.path.splitext(plates[n])[1].lower()}")
    out_dir = _hint_folder(settings.hint_prompt)
    request = {
        "mode": "frame" if settings.hint_mode == "SINGLE" else "video",
        "frames_dir": frames_dir,
        "frame_paths": [os.path.join(folder, plates[n]) for n in numbers],
        "numbers": numbers,
        "prompt": settings.hint_prompt,
        "prompt_index": numbers.index(current) if current in numbers else 0,
        "threshold": round(settings.hint_threshold, 3),
        "checkpoint": os.path.join(root, "weights", "sam3.pt"),
        "out_dir": out_dir,
        "prefix": "hint_",
    }
    request_path = os.path.join(work, "request.json")
    with open(request_path, "w", encoding="utf-8") as f:
        json.dump(request, f)
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3_worker.py")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env["PYTHONUNBUFFERED"] = "1"
    log_path = os.path.join(work, "sam3.log")
    reload, paused = gpu_for_sam3()
    free_gb, others = gpu_others()
    busy = ""
    if free_gb is not None and free_gb < 6 and others:
        busy = f"only {free_gb:.1f} GB of GPU memory free: {', '.join(others[:3])} using it"
    try:
        process = subprocess.Popen(
            [sam3_python(root), "-u", worker, request_path], cwd=root, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=open(log_path, "w", encoding="utf-8", errors="replace"),
            text=True, encoding="utf-8", errors="replace",
            # no console window; ahead of other programs for the CPU while it runs
            creationflags=(0x08000000 | 0x00008000) if sys.platform == "win32" else 0,
        )
    except OSError as exc:
        shutil.rmtree(work, ignore_errors=True)
        gpu_back(paused)
        return f"SAM 3 didn't start: {exc}"
    job = {
        "reload": reload, "paused": paused, "timing": {}, "started": time.monotonic(), "busy": busy,
        "process": process, "state": "running", "line": "starting SAM 3", "done": 0,
        "total": len(numbers), "work": work, "out_dir": out_dir, "log": log_path,
        "node": node.name, "material": _owner_name(node), "mode": settings.hint_mode,
        "frame": scene.frame_current if settings.hint_mode == "SINGLE" and not is_free_source(src) else None,
        "prompt": settings.hint_prompt,
    }
    job["thread"] = threading.Thread(target=_hint_reader, args=(job,), daemon=True)
    job["thread"].start()
    HINT_JOBS[uid] = job
    return None


def gpu_others() -> tuple[float | None, list[str]]:
    """(GB of GPU memory free, the other programs on the GPU) from nvidia-smi, or (None, [])."""
    try:
        flags = 0x08000000 if sys.platform == "win32" else 0
        memory = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, creationflags=flags).stdout
        used, total = (float(x) for x in memory.strip().splitlines()[0].split(","))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5, creationflags=flags).stdout
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, []
    names = []
    for line in apps.strip().splitlines():
        pid, _sep, path = line.partition(",")
        if pid.strip() == str(os.getpid()):
            continue
        path = path.strip()
        lowered = path.lower()
        name = ("ComfyUI" if "comfy" in lowered else
                "another Blender" if lowered.endswith("blender.exe") else
                os.path.splitext(os.path.basename(path))[0] or path)
        if name not in names:
            names.append(name)
    return (total - used) / 1024, names


def gpu_for_sam3():
    """Everything SAM 3 can use while it runs: CorridorKey closes (not just parks) so its GPU
    memory is free, and Rendered viewports pause so Blender isn't drawing on the card too.
    (whether CorridorKey was loaded, the viewports paused)."""
    reload = warm_alive()
    for uid in list(PREVIEW_JOBS):
        cancel_preview(uid)
    land_previews()
    close_warm()
    paused = []
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D" and space.shading.type == "RENDERED":
                    space.shading.type = "SOLID"
                    paused.append(space)
    return reload, paused


def gpu_back(paused) -> None:
    """The viewports SAM 3 paused, rendered again (unless they were changed meanwhile)."""
    for space in paused:
        try:
            if space.shading.type == "SOLID":
                space.shading.type = "RENDERED"
        except (ReferenceError, AttributeError):
            pass


def _took(job) -> str:
    def seconds(value: float) -> str:
        value = int(round(value))
        return f"{value // 60}m {value % 60:02d}s" if value >= 60 else f"{value} s"

    parts = [f"{name} {seconds(job['timing'][key])}" for key, name in
             (("load", "start"), ("frames", "frames"), ("work", "finding" if job["mode"] == "SINGLE" else "tracking"))
             if key in job["timing"]]
    if job.get("memory"):
        parts.append(f"frames on the {job['memory']}" if job["memory"] == "GPU" else "frames in system RAM")
    if job.get("busy"):
        parts.append(job["busy"])
    total = seconds(time.monotonic() - job["started"])
    return f"{total} ({', '.join(parts)})" if parts else total


def cancel_hint(uid: str) -> None:
    job = HINT_JOBS.get(uid)
    if job is not None and job["state"] == "running":
        job["cancelled"] = True
        try:
            job["process"].kill()
        except OSError:
            pass


def _texture_of(node, socket_name: str):
    texture, _plugged = upstream_texture(node, socket_name)
    return texture


def plug_hint(node, job: dict) -> int:
    """Plug the mattes SAM 3 made into the node's Alpha Hint, timed as the footage plays, and set
    the hint up for one frame or all frames. Returns how many frames it has."""
    folder = job["out_dir"]
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png")) if os.path.isdir(folder) else []
    if not files:
        return 0
    tree = node.id_data
    footage = _texture_of(node, IMAGE_IN)
    image = bpy.data.images.load(os.path.join(folder, files[0]), check_existing=False)
    image.name = job.get("image_name") or f"SAM 3 hint: {job['prompt']}"
    image.source = "SEQUENCE" if len(files) > 1 else "FILE"
    try:
        image.colorspace_settings.name = "Non-Color"
    except TypeError:
        pass
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.label = job.get("label") or f"Alpha Hint (SAM 3: {job['prompt']})"
    if job.get("base"):  # a screen-key hint remembers the hint it was made from, for the next bake
        texture[SCREEN_BASE] = json.dumps(job["base"])
    texture.image = image
    texture.interpolation = "Linear"
    if footage is not None:
        texture.location = (footage.location.x, footage.location.y - 320)
        texture.extension = footage.extension
        if image.source == "SEQUENCE":
            for attr in ("frame_start", "frame_offset", "frame_duration"):
                setattr(texture.image_user, attr, getattr(footage.image_user, attr))
            texture.image_user.use_auto_refresh = True
            texture.image_user.use_cyclic = False
        if footage.inputs["Vector"].is_linked:
            tree.links.new(footage.inputs["Vector"].links[0].from_socket, texture.inputs["Vector"])
    else:
        texture.location = (node.location.x - 320, node.location.y - 200)
    tree.links.new(texture.outputs["Color"], node.inputs[HINT_IN])
    src = source_of(node)
    if src is not None:
        state = st(src)
        state.use_alpha_hint = True
        if is_free_source(src):
            state.hint_frames = "FIRST" if job["mode"] == "SINGLE" else "ALL"
            state.hint_quality = 0.75  # the free README's value for automated hints
        else:
            state.hint_frames = "single_frame" if job["mode"] == "SINGLE" else "all_frames"
            if job["mode"] == "SINGLE" and job["frame"] is not None:
                state.use_time_awareness = True  # Corridor's single-frame hint needs it
                state.reference_frame = str(job["frame"])
    return len(files)


def land_hints() -> bool:
    changed = False
    for uid, job in list(HINT_JOBS.items()):
        if job["state"] == "running":
            line = f"SAM 3: {job['line']}"
            if PREVIEW_LINE.get("hint:" + uid) != line:
                PREVIEW_LINE["hint:" + uid] = line
                changed = True
            continue
        del HINT_JOBS[uid]
        changed = True
        shutil.rmtree(job["work"], ignore_errors=True)
        gpu_back(job.get("paused") or [])
        material = bpy.data.materials.get(job["material"])
        tree = material.node_tree if material is not None else bpy.data.node_groups.get(job["material"])
        node = tree.nodes.get(job["node"]) if tree is not None else None
        if job["state"] == "done" and node is not None:
            count = plug_hint(node, job)
            note = f" ({job['empty']} with no {job['prompt']} found)" if job.get("empty") else ""
            PREVIEW_LINE["hint:" + uid] = (f"Alpha hint made: {count} frame{'s' if count != 1 else ''}"
                                           f" in {_took(job)}{note}.")
            src = source_of(node)
            if job.get("reload") and src is not None and not HINT_JOBS and engine_available():
                start_preview(src, bpy.context.scene, [bpy.context.scene.frame_current])  # CorridorKey back, with the new hint
        elif job["state"] == "cancelled":
            PREVIEW_LINE["hint:" + uid] = "Alpha hint cancelled."
        else:
            PREVIEW_LINE["hint:" + uid] = f"SAM 3: {job.get('message') or 'it stopped'}"
    return changed


SCREEN_BASE = "corridorkey_screen_base"  # on a screen-key hint's texture: the hint it was made from
SCREEN_JOBS: dict[str, dict] = {}  # source uid -> the screen key being baked into the hint


def _hint_base(node, src) -> dict | None:
    """The hint a screen-key bake multiplies into: the one plugged in now, or, when that is an
    earlier screen-key bake, the hint that bake was made from (so tuning and baking again never
    compounds). {"path", "kind": "SEQUENCE" | "FILE" | "MOVIE", "name"} or None."""
    texture = _texture_of(node, HINT_IN)
    if texture is not None and texture.get(SCREEN_BASE):
        try:
            return json.loads(texture[SCREEN_BASE])
        except (TypeError, ValueError):
            pass
    image = texture.image if texture is not None else None
    if image is None and not (node.inputs.get(HINT_IN) and node.inputs[HINT_IN].is_linked):
        hint = wired_image_node(src, 1)
        image = hint.image if hint is not None else None
    if image is None or not image.filepath:
        return None
    return {"path": bpy.path.abspath(image.filepath, library=image.library), "kind": image.source, "name": image.name}


def start_screen_hint(node, scene) -> str | None:
    """Bake the screen key into the Alpha Hint for the whole shot, in background Blenders: each
    frame's screen key times the current hint, plugged in when it's done."""
    src = source_of(node)
    if src is None:
        return "This node has no CorridorKey source yet."
    uid = st(src).uid
    if uid in SCREEN_JOBS:
        return None
    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None or image.source != "SEQUENCE":
        return "The screen key bakes from an image sequence (not a movie)."
    folder = os.path.dirname(bpy.path.abspath(image.filepath, library=image.library))
    plates = numbered(folder)
    if not plates:
        return "No footage frames found."
    numbers = sorted(plates)
    base = _hint_base(node, src)
    note = ""
    hints: dict[int, str | None] = {n: None for n in numbers}
    if base is not None and base["kind"] == "SEQUENCE":
        sequence = numbered(os.path.dirname(base["path"]))
        if len(set(sequence) & set(numbers)) >= len(numbers) / 2:  # numbered like the footage
            hints = {n: os.path.join(os.path.dirname(base["path"]), sequence[n]) if n in sequence else None
                     for n in numbers}
        else:  # numbered otherwise: frame for frame, in order
            ordered = [sequence[k] for k in sorted(sequence)]
            hints = {n: os.path.join(os.path.dirname(base["path"]), ordered[i]) if i < len(ordered) else None
                     for i, n in enumerate(numbers)}
    elif base is not None and base["kind"] in {"FILE", "TILED"} and os.path.isfile(base["path"]):
        hints = {n: base["path"] for n in numbers}
    elif base is not None:
        note = " (the hint is a movie: the screen key alone)"
        base = None
    settings = node.corridorkey_shader
    work = os.path.join(tempfile.gettempdir(), "CorridorKey Screen Key", uuid.uuid4().hex[:8])
    os.makedirs(work)
    out_dir = _hint_folder("screen key")
    workers = max(1, min(4, (os.cpu_count() or 4) // 4))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_key_worker.py")
    flags = 0x08000000 if sys.platform == "win32" else 0
    processes = []
    for index in range(workers):
        request = {
            "frames": [[n, os.path.join(folder, plates[n]), hints[n]] for n in numbers],
            "direction": list(screen_direction(settings)), "threshold": settings.screen_threshold,
            "softness": settings.screen_softness, "out_dir": out_dir, "prefix": "hint_",
            "slice": [index, workers],
        }
        path = os.path.join(work, f"request_{index}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(request, f)
        try:
            processes.append(subprocess.Popen(
                [bpy.app.binary_path, "-b", "--factory-startup", "--python", script, "--", path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            ))
        except OSError as exc:
            for process in processes:
                process.kill()
            shutil.rmtree(work, ignore_errors=True)
            return f"The screen key didn't start: {exc}"
    with_hint = f"times '{base['name']}'" if base else "on its own"
    SCREEN_JOBS[uid] = {
        "processes": processes, "work": work, "out_dir": out_dir, "total": len(numbers),
        "node": node.name, "material": _owner_name(node), "base": base, "started": time.monotonic(),
        "what": f"{screen_name(settings)} screen key {with_hint}{note}",
    }
    PREVIEW_LINE["screen:" + uid] = f"Baking the {SCREEN_JOBS[uid]['what']}..."
    return None


def cancel_screen_hint(uid: str) -> None:
    job = SCREEN_JOBS.get(uid)
    if job is not None:
        job["cancelled"] = True
        for process in job["processes"]:
            try:
                process.kill()
            except OSError:
                pass


def land_screen_hints() -> bool:
    """Screen-key bakes: how far along; the finished ones plugged into the Alpha Hint."""
    changed = False
    for uid, job in list(SCREEN_JOBS.items()):
        try:
            done = len([f for f in os.listdir(job["out_dir"]) if f.endswith(".png")])
        except OSError:
            done = 0
        if any(process.poll() is None for process in job["processes"]):
            line = f"Baking the {job['what']}: {done} of {job['total']} frames"
            if PREVIEW_LINE.get("screen:" + uid) != line:
                PREVIEW_LINE["screen:" + uid] = line
                changed = True
            continue
        del SCREEN_JOBS[uid]
        changed = True
        errors = sorted(f for f in os.listdir(job["out_dir"]) if f.startswith("error_")) if os.path.isdir(job["out_dir"]) else []
        message = ""
        for name in errors:
            with open(os.path.join(job["out_dir"], name), encoding="utf-8", errors="replace") as f:
                message = message or f.read().strip()[:160]
            os.remove(os.path.join(job["out_dir"], name))
        shutil.rmtree(job["work"], ignore_errors=True)
        material = bpy.data.materials.get(job["material"])
        tree = material.node_tree if material is not None else bpy.data.node_groups.get(job["material"])
        node = tree.nodes.get(job["node"]) if tree is not None else None
        if job.get("cancelled"):
            PREVIEW_LINE["screen:" + uid] = "Screen key bake stopped."
        elif done < job["total"] or node is None:
            PREVIEW_LINE["screen:" + uid] = f"Screen key bake stopped at {done} of {job['total']} frames" + (
                f": {message}" if message else ".")
        else:
            count = plug_hint(node, {
                "out_dir": job["out_dir"], "mode": "ALL", "frame": None, "prompt": "screen key",
                "image_name": "Screen key hint", "label": f"Alpha Hint (screen key: {job['what']})",
                "base": job["base"],
            })
            seconds = int(time.monotonic() - job["started"])
            PREVIEW_LINE["screen:" + uid] = (f"Alpha Hint is now the {job['what']}: {count} frames "
                                             f"in {seconds // 60}m {seconds % 60:02d}s.")
    return changed


class _Folded:
    """What a closed section draws into: everything, quietly, nowhere."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: self

    def __setattr__(self, _name, _value):
        pass


def _fold(layout, settings, prop: str, title: str, icon: str, note: str = "", header_extra=None):
    """A section that folds away (its open state saved on the node): the layout to draw into, or
    a stand-in that draws nothing while it's closed. `note` shows in the header, open or closed;
    `header_extra(header)` adds small buttons at the header's right."""
    if isinstance(layout, _Folded):  # inside a folded section: nothing to draw
        return layout
    header, body = layout.panel_prop(settings, prop)
    header.label(text=f"{title}: {note}" if note else title, icon=icon)
    if header_extra is not None:
        header_extra(header)
    return body if body is not None else _Folded()


def screen_name(settings) -> str:
    """The picked screen colour in words: green, blue, or its hex value."""
    colour = _display(settings.screen_pick)
    for name, preset in (("green", SCREEN_GREEN), ("blue", SCREEN_BLUE)):
        if all(abs(a - b) < 0.02 for a, b in zip(colour, preset)):
            return name
    return "#" + "".join(f"{round(min(c, 1.0) * 255):02x}" for c in colour)


class CORRIDORKEY_SHADER_OT_screen_preset(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.screen_preset"
    bl_label = "Screen Preset"
    bl_description = "Set the screen colour to a typical green or blue screen"
    bl_options = {"REGISTER", "UNDO"}

    colour: bpy.props.EnumProperty(items=[("GREEN", "Green", ""), ("BLUE", "Blue", "")], options={"SKIP_SAVE"})

    def execute(self, context):
        node = node_by_source(self.uid)
        if node is None:
            return {"CANCELLED"}
        node.corridorkey_shader.screen_pick = _linear(SCREEN_GREEN if self.colour == "GREEN" else SCREEN_BLUE)
        return {"FINISHED"}


def draw_screen_key(layout, node, src) -> None:
    uid, settings = st(src).uid, node.corridorkey_shader
    note = ("baking" if uid in SCREEN_JOBS else
            f"cleaning ({screen_name(settings)})" if settings.screen_clean else
            "showing the matte" if settings.screen_show else "")
    box = _fold(layout, settings, "ui_screen", "Screen Key", "IMAGE_ALPHA", note)
    row = box.row(align=True)
    row.prop(settings, "screen_pick", text="")
    for colour, text in (("GREEN", "Green"), ("BLUE", "Blue")):
        op = row.operator(CORRIDORKEY_SHADER_OT_screen_preset.bl_idname, text=text)
        op.uid, op.colour = uid, colour
    row = box.row(align=True)
    row.prop(settings, "screen_threshold")
    row.prop(settings, "screen_softness")
    row = box.row(align=True)
    row.prop(settings, "screen_clean", toggle=True, icon="BRUSH_DATA")
    row.prop(settings, "screen_show", toggle=True, icon="HIDE_OFF")
    if uid in SCREEN_JOBS:
        box.operator(CORRIDORKEY_SHADER_OT_cancel_screen_hint.bl_idname, icon="CANCEL").uid = uid
    else:
        box.operator(CORRIDORKEY_SHADER_OT_screen_hint.bl_idname, icon="IMPORT").uid = uid
    line = PREVIEW_LINE.get("screen:" + uid)
    if uid in SCREEN_JOBS and line:
        draw_progress(box, line)
    elif line:
        _lines(box, line, "BLANK1", 34)


class CORRIDORKEY_SHADER_OT_screen_hint(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.screen_hint"
    bl_label = "Bake into Alpha Hint"
    bl_description = (
        "Key the screen colour out of every frame and multiply it into the Alpha Hint (or use it "
        "as the hint when there is none), so CorridorKey itself treats the leftover screen as "
        "background. Runs in the background; the new hint plugs in when it's done"
    )

    def execute(self, context):
        node = node_by_source(self.uid)
        if node is None:
            return {"CANCELLED"}
        why = start_screen_hint(node, context.scene)
        if why:
            PREVIEW_LINE["screen:" + self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_cancel_screen_hint(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.cancel_screen_hint"
    bl_label = "Stop Baking"
    bl_description = "Stop baking the screen key into the Alpha Hint"

    def execute(self, context):
        cancel_screen_hint(self.uid)
        return {"FINISHED"}


def _node_of_uid(uid: str):
    return node_by_source(uid)


class CORRIDORKEY_SHADER_OT_make_hint(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.make_hint"
    bl_label = "Make Alpha Hint"
    bl_description = (
        "Cut out what you typed with SAM 3 and plug it into the Alpha Hint: for one frame, or "
        "tracked through the shot"
    )

    def execute(self, context):
        node = _node_of_uid(self.uid)
        if node is None:
            return {"CANCELLED"}
        why = start_hint(node, context.scene)
        if why:
            PREVIEW_LINE["hint:" + self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_cancel_hint(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.cancel_hint"
    bl_label = "Stop"
    bl_description = "Stop making the alpha hint"

    def execute(self, context):
        cancel_hint(self.uid)
        return {"FINISHED"}


SAM3_CHECK: dict = {"line": ""}


def _check_sam3(python: str, root: str) -> None:
    code = (
        "import torch, sam3;"
        "print('SAM 3 ready | torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|',"
        " torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no NVIDIA GPU found')"
    )
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    try:
        done = subprocess.run([python, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                              timeout=300, creationflags=0x08000000 if sys.platform == "win32" else 0)
        lines = (done.stdout or done.stderr or "").strip().splitlines()
        SAM3_CHECK["line"] = lines[-1][-120:] if lines else "No answer from SAM 3's Python."
    except (OSError, subprocess.TimeoutExpired) as exc:
        SAM3_CHECK["line"] = f"Couldn't run SAM 3's Python: {exc}"


class CORRIDORKEY_SHADER_OT_sam3_check(bpy.types.Operator):
    bl_idname = "corridorkey_shader.sam3_check"
    bl_label = "Check SAM 3"
    bl_description = "Check that SAM 3 is installed, has its weights and sees your GPU"

    def execute(self, context):
        problem = sam3_problem()
        if problem:
            SAM3_CHECK["line"] = problem
            return {"FINISHED"}
        root = sam3_root()
        SAM3_CHECK["line"] = "Checking (importing PyTorch and SAM 3)..."
        threading.Thread(target=_check_sam3, args=(sam3_python(root), root), daemon=True).start()
        return {"FINISHED"}


_SAM3_BAT = r"""@echo off
title SAM 3 for CorridorKey 2.0 - installing
echo Installing SAM 3 (Meta, SAM License) into "{root}"
echo This downloads PyTorch and SAM 3's weights (about 7 GB). Leave this window open.
echo.
where git >nul 2>nul || (echo Git isn't installed. Get it from https://git-scm.com/download/win, then run Install again. & goto fail)
if not exist "{root}" mkdir "{root}"
cd /d "{root}"
set "PY="
where py >nul 2>nul && (py -3.12 -c "" >nul 2>nul && set "PY=py -3.12")
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (echo Python 3.12 isn't installed. Get it from https://www.python.org/downloads/ ^(tick "Add to PATH"^), then run Install again. & goto fail)
if not exist "venv\Scripts\python.exe" (%PY% -m venv venv || goto fail)
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128 || goto fail
venv\Scripts\python.exe -m pip install "git+{repo}.git" einops psutil "setuptools<81" pycocotools scipy opencv-python-headless pillow huggingface_hub || goto fail
venv\Scripts\python.exe -m pip install "triton-windows<3.7"
echo.
echo Downloading SAM 3's weights from the {mirror} mirror (Meta's SAM License; a copy is saved beside them)...
venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; [d('{mirror}', f, local_dir='weights') for f in ('LICENSE', 'config.json', 'sam3.pt')]" || goto fail
echo.
echo Done. In Blender, press Check SAM 3 in the CorridorKey 2.0 preferences.
pause
exit /b 0
:fail
echo.
echo The install stopped (see above). Fix that and press Install again; finished steps are skipped.
pause
exit /b 1
"""


class CORRIDORKEY_SHADER_OT_sam3_install(bpy.types.Operator):
    bl_idname = "corridorkey_shader.sam3_install"
    bl_label = "Install SAM 3"
    bl_description = (
        "Download SAM 3 (Meta's Segment Anything 3, freely distributed under the SAM License) with "
        "its PyTorch, from GitHub and an ungated Hugging Face mirror (about 7 GB), in a window you "
        "can watch. Needs Git and Python 3.12"
    )

    @classmethod
    def poll(cls, context) -> bool:
        if sys.platform != "win32":
            cls.poll_message_set("The installer is for Windows: follow the SAM 3 README on Linux")
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, title="Install SAM 3?",
            message="About 7 GB. SAM 3 is Meta's, under the SAM License.", confirm_text="Install",
        )

    def execute(self, context):
        preferences = prefs()
        root = sam3_root()
        if not preferences.sam3_root:
            preferences.sam3_root = root
        script = os.path.join(tempfile.gettempdir(), "install_sam3_for_corridorkey.bat")
        with open(script, "w", encoding="utf-8") as bat:
            bat.write(_SAM3_BAT.format(root=root, repo=SAM3_REPO, mirror=SAM3_MIRROR))
        subprocess.Popen(["cmd", "/c", script], creationflags=subprocess.CREATE_NEW_CONSOLE)
        SAM3_CHECK["line"] = "Installing in the window that opened. Press Check SAM 3 when it's done."
        return {"FINISHED"}


def draw_hint_maker(layout, node, src) -> None:
    uid = st(src).uid
    settings = node.corridorkey_shader
    job = HINT_JOBS.get(uid)
    box = _fold(layout, settings, "ui_sam3", "Alpha Hint Maker (SAM 3)", "MOD_MASK",
                f"{job['done']} of {job['total']}" if job and job.get("total") and job.get("done") else
                ("running" if job else ""))
    problem = sam3_problem()
    if problem:
        box.label(text=problem, icon="ERROR")
        box.operator(CORRIDORKEY_SHADER_OT_open_prefs.bl_idname, text="Install SAM 3", icon="IMPORT")
        return
    box.prop(settings, "hint_prompt")
    box.row().prop(settings, "hint_mode", expand=True)
    if settings.hint_mode == "ALL":
        box.prop(settings, "hint_range")
    elif is_free_source(src):
        box.label(text="Free engine: the hint is made for the shot's first frame.", icon="INFO")
    box.prop(settings, "hint_threshold")
    if uid in HINT_JOBS:
        box.operator(CORRIDORKEY_SHADER_OT_cancel_hint.bl_idname, icon="CANCEL").uid = uid
    else:
        box.operator(CORRIDORKEY_SHADER_OT_make_hint.bl_idname, icon="MOD_MASK").uid = uid
    line = PREVIEW_LINE.get("hint:" + uid)
    if job and job.get("total"):
        draw_progress(box, line or "Making the Alpha Hint", factor=(job.get("done") or 0) / job["total"])
    elif line:
        _lines(box, line, "NONE", 40)


# --- floor shadows: the real shadows on the floor, kept, on a plane at the floor ----------------
#
# A see-through plane lies on the floor (Z = 0, where Jetset puts it). Its material looks at the
# footage through the camera (Window coordinates), so each point of the plane sees the plate pixel
# in front of it, and it follows any camera move by itself: the plane is the floor mask. Where the
# plate is darker than the clean floor colour, the plane darkens what's behind it by that much
# (a Transparent BSDF tinted by the ratio: a true multiply), only where the plate is still the
# screen's hue (so the actor, standing in front of it, is never darkened).

FLOOR_GROUP, FLOOR_TEX, FLOOR_NODE = "CK Floor Shadows", "CK Floor Footage", "CK Floor Shadows"
FLOOR_CLEAN = "CK Floor Clean"  # the baked clean-floor brightness, per frame
FLOOR_TAG = "corridorkey_floor"  # on the plane: the uid of the CorridorKey node it serves
FLOOR_VERSION = 2
FLOOR_JOBS: dict[str, dict] = {}  # source uid -> its clean-floor bake
FLOOR_RANGE = 15.0  # metres: how far the plane reaches where the view runs to the horizon
FLOOR_LIFT = 0.002  # metres above the floor, clear of a CG floor at Z = 0


def _feed(links, socket, value) -> None:
    if isinstance(value, bpy.types.NodeSocket):
        links.new(value, socket)
    else:
        socket.default_value = value


def ensure_floor_group():
    """The shadow maths as one node group (made once, remade when this add-on changes it)."""
    tree = bpy.data.node_groups.get(FLOOR_GROUP)
    if tree is not None and tree.get("ck_version") == FLOOR_VERSION:
        return tree
    kept = []  # the settings on the planes using an older version, put back after
    if tree is None:
        tree = bpy.data.node_groups.new(FLOOR_GROUP, "ShaderNodeTree")
    else:
        for material in bpy.data.materials:
            for n in (material.node_tree.nodes if material.node_tree else ()):
                if n.bl_idname == "ShaderNodeGroup" and n.node_tree == tree:
                    kept.append((n, {i.name: tuple(i.default_value) if hasattr(i.default_value, "__len__")
                                     else i.default_value for i in n.inputs if hasattr(i, "default_value")}))
    tree.nodes.clear()
    tree.interface.clear()
    tree["ck_version"] = FLOOR_VERSION
    face = tree.interface

    def add(name, kind, default, low=None, high=None, about=""):
        socket = face.new_socket(name, in_out="INPUT", socket_type=kind)
        socket.default_value = default
        if low is not None:
            socket.min_value, socket.max_value = low, high
        socket.description = about

    add("Footage", "NodeSocketColor", (0.5, 0.5, 0.5, 1.0), about="The plate, seen through the camera")
    add("Floor Colour", "NodeSocketColor", (0.05, 0.35, 0.06, 1.0),
        about="The floor where nothing shades it: anything darker reads as shadow")
    add("Density", "NodeSocketFloat", 1.0, 0.0, 2.0, "How dark the kept shadows are (1 = as shot)")
    add("Lift", "NodeSocketFloat", 0.1, 0.0, 0.9,
        "Ignore darkening weaker than this: uneven lighting and noise, not shadow")
    add("Keep Colour", "NodeSocketFloat", 0.0, 0.0, 1.0,
        "0: neutral shadows. 1: the shadows' own colour (can carry screen spill)")
    add("Screen Gate", "NodeSocketFloat", 0.03, 0.0, 0.5,
        "How much of the screen's hue a pixel needs to count as floor (keeps the actor clear)")
    add("Edge Feather", "NodeSocketFloat", 0.05, 0.0, 0.5, "Fade at the plane's edges (share of its size)")
    add("Show Matte", "NodeSocketFloat", 0.0, 0.0, 1.0, "Show the shadow matte itself, white on black")
    add("Clean Floor", "NodeSocketFloat", 0.5, 0.0, 1.0,
        "The unshadowed floor's brightness here, half scale (0.5 = the Floor Colour): the baked clean floor")
    face.new_socket("Shader", in_out="OUTPUT", socket_type="NodeSocketShader")

    nodes, links = tree.nodes, tree.links
    made = []

    def node(kind, **attrs):
        n = nodes.new(kind)
        for key, value in attrs.items():
            setattr(n, key, value)
        made.append(n)
        n.location = (200 * (len(made) % 12) - 1000, -180 * (len(made) // 12))
        return n

    def math(op, a, b=None, clamp=False):
        n = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        _feed(links, n.inputs[0], a)
        if b is not None:
            _feed(links, n.inputs[1], b)
        return n.outputs[0]

    def vmath(op, a, b=None, scale=None):
        n = node("ShaderNodeVectorMath", operation=op)
        _feed(links, n.inputs[0], a)
        if b is not None:
            _feed(links, n.inputs[1], b)
        if scale is not None:
            _feed(links, n.inputs["Scale"], scale)
        return n.outputs["Value" if op == "DOT_PRODUCT" else "Vector"]

    def smooth(value, low, high):
        n = node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP", clamp=True)
        _feed(links, _value_input(n, "Value"), value)
        _feed(links, _value_input(n, "From Min"), low)
        _feed(links, _value_input(n, "From Max"), high)
        return n.outputs["Result"]

    def mix(factor, a, b, clamp=False):
        n = node("ShaderNodeMix", data_type="RGBA", blend_type="MIX", clamp_result=clamp)
        _feed(links, n.inputs["Factor"], factor)
        colours = [s for s in n.inputs if s.type == "RGBA"]
        _feed(links, colours[0], a)
        _feed(links, colours[1], b)
        return next(s for s in n.outputs if s.type == "RGBA")

    def gamma(colour):
        n = node("ShaderNodeGamma")
        _feed(links, n.inputs["Color"], colour)
        n.inputs["Gamma"].default_value = SK_GAMMA_VALUE
        return n.outputs["Color"]

    group_in = node("NodeGroupInput")
    group_out = node("NodeGroupOutput")
    plate, clean = group_in.outputs["Footage"], group_in.outputs["Floor Colour"]

    # how much of the clean floor is there: 1 unshaded, 0 black; shadow is what's missing
    level = math("MULTIPLY", group_in.outputs["Clean Floor"], 2.0)  # stored at half scale
    there = math("DIVIDE", vmath("DOT_PRODUCT", plate, clean),
                 math("MAXIMUM", math("MULTIPLY", vmath("DOT_PRODUCT", clean, clean), level), 1e-6), clamp=True)
    shade = math("SUBTRACT", 1.0, there)
    lift = group_in.outputs["Lift"]
    shade = math("DIVIDE", math("SUBTRACT", shade, lift), math("MAXIMUM", math("SUBTRACT", 1.0, lift), 1e-3), clamp=True)
    shade = math("MULTIPLY", shade, group_in.outputs["Density"], clamp=True)

    # only where the plate still leans to the screen's hue (display-like values, as the screen key)
    clean_d = gamma(clean)
    mean = vmath("DOT_PRODUCT", clean_d, (1 / 3, 1 / 3, 1 / 3))
    hue = vmath("NORMALIZE", vmath("SUBTRACT", clean_d, vmath("SCALE", (1.0, 1.0, 1.0), scale=mean)))
    lean = vmath("DOT_PRODUCT", gamma(plate), hue)
    gate_low = group_in.outputs["Screen Gate"]
    floor = smooth(lean, gate_low, math("ADD", gate_low, 0.05))

    # faded towards the plane's edges
    coords = node("ShaderNodeTexCoord")
    split = node("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["UV"], split.inputs[0])
    u, v = split.outputs["X"], split.outputs["Y"]
    inner = math("MINIMUM", math("MINIMUM", u, math("SUBTRACT", 1.0, u)),
                 math("MINIMUM", v, math("SUBTRACT", 1.0, v)))
    edge = smooth(inner, 0.0, math("MAXIMUM", group_in.outputs["Edge Feather"], 1e-4))

    amount = math("MULTIPLY", math("MULTIPLY", shade, floor), edge, clamp=True)
    ratio = mix(1.0, (0, 0, 0, 1), vmath("DIVIDE", plate, vmath("MAXIMUM", clean, (1e-4, 1e-4, 1e-4))), clamp=True)
    tint = mix(group_in.outputs["Keep Colour"], (0, 0, 0, 1), ratio)
    multiply = mix(amount, (1, 1, 1, 1), tint, clamp=True)

    see_through = node("ShaderNodeBsdfTransparent")
    links.new(multiply, see_through.inputs["Color"])
    matte = node("ShaderNodeEmission")
    links.new(amount, matte.inputs["Color"])
    shader = node("ShaderNodeMixShader")
    links.new(group_in.outputs["Show Matte"], shader.inputs[0])
    links.new(see_through.outputs[0], shader.inputs[1])
    links.new(matte.outputs[0], shader.inputs[2])
    links.new(shader.outputs[0], group_out.inputs["Shader"])
    for n, values in kept:
        for name, value in values.items():
            socket = n.inputs.get(name)
            if socket is not None and hasattr(socket, "default_value"):
                try:
                    socket.default_value = value
                except (TypeError, ValueError):
                    pass
    return tree


def floor_object(node, src):
    """This node's floor-shadow plane, or None."""
    obj = bpy.data.objects.get(node.corridorkey_shader.floor_name) if node.corridorkey_shader.floor_name else None
    return obj if obj is not None and obj.get(FLOOR_TAG) == st(src).uid else None


def floor_group_node(obj):
    material = obj.active_material if obj is not None else None
    tree = material.node_tree if material is not None else None
    return tree.nodes.get(FLOOR_NODE) if tree is not None else None


def floor_sync(obj, plate) -> bool:
    """The plane's footage: the node's plate, timed as it plays (and its clean floor with it)."""
    material = obj.active_material
    nodes = material.node_tree.nodes if material is not None and material.node_tree else None
    texture = nodes.get(FLOOR_TEX) if nodes is not None else None
    if texture is None or plate is None:
        return False
    changed = _set(texture, "image", plate.image)
    clean = nodes.get(FLOOR_CLEAN)
    for node in (texture, clean):
        if node is not None and node.image is not None and plate.image is not None:
            user = node.image_user
            for attr in ("frame_start", "frame_offset", "frame_duration"):
                changed |= _set(user, attr, getattr(plate, attr))
            changed |= _set(user, "use_auto_refresh", True)
            changed |= _set(user, "use_cyclic", False)
    return changed


def floor_footprint(scene):
    """The four corners of the camera's view on the floor (Z = 0), the far ones reined in to
    FLOOR_RANGE where the view runs to the horizon. None without a camera above the floor."""
    camera = scene.camera
    if camera is None or camera.type != "CAMERA":
        return None
    from mathutils import Vector

    matrix = camera.matrix_world
    origin = matrix.translation
    if origin.z <= 0.01:
        return None
    corners = []
    for local in camera.data.view_frame(scene=scene):
        ray = (matrix @ local) - origin
        flat = Vector((ray.x, ray.y, 0.0))
        if ray.z < -1e-6:
            point = origin + ray * (-origin.z / ray.z)
            reach = (Vector((point.x, point.y, 0.0)) - Vector((origin.x, origin.y, 0.0))).length
            if reach > FLOOR_RANGE and flat.length > 1e-6:
                point = Vector((origin.x, origin.y, 0.0)) + flat.normalized() * FLOOR_RANGE
        elif flat.length > 1e-6:
            point = Vector((origin.x, origin.y, 0.0)) + flat.normalized() * FLOOR_RANGE
        else:
            return None
        corners.append(Vector((point.x, point.y, FLOOR_LIFT)))
    return corners


def floor_mesh(mesh, corners) -> None:
    """The plane's mesh: one quad through the corners, facing up, UVs 0-1 (for the edge fade)."""
    import bmesh

    a, b, c = corners[0], corners[1], corners[2]
    if (b - a).cross(c - a).z < 0:  # face up
        corners = list(reversed(corners))
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in corners]
    face = bm.faces.new(verts)
    uv = bm.loops.layers.uv.new("UVMap")
    for loop, coord in zip(face.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop[uv].uv = coord
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def _srgb_to_linear(values):
    import numpy as np

    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def sample_floor_colour(node, src, scene, obj):
    """The clean floor's colour on the frame you're on: the brighter part (75th-95th percentile)
    of the screen-coloured pixels the plane covers. (colour, None) or (None, why not)."""
    import numpy as np
    from bpy_extras.object_utils import world_to_camera_view

    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None or image.source not in {"SEQUENCE", "FILE"}:
        return None, "Sampling needs footage as images (not a movie): pick the floor colour instead."
    path = bpy.path.abspath(image.filepath, library=image.library)
    if image.source == "SEQUENCE":
        plates = numbered(os.path.dirname(path))
        if not plates:
            return None, "No footage frames found."
        number = scene.frame_current - plate.frame_start + 1 + plate.frame_offset
        nearest = min(plates, key=lambda n: abs(n - number))
        path = os.path.join(os.path.dirname(path), plates[nearest])
    loaded = bpy.data.images.load(path, check_existing=False)
    try:
        width, height = loaded.size
        flat = np.empty(width * height * 4, dtype=np.float32)
        loaded.pixels.foreach_get(flat)
        display = not loaded.is_float
    finally:
        bpy.data.images.remove(loaded)
    pixels = flat.reshape(height, width, 4)[..., :3]
    step = max(1, width // 480)
    pixels = pixels[::step, ::step]
    rows, cols = pixels.shape[:2]
    linear = _srgb_to_linear(pixels) if display else np.clip(pixels, 0.0, None)
    shown = pixels if display else np.clip(pixels, 0.0, None) ** SK_GAMMA_VALUE

    keep = np.ones((rows, cols), dtype=bool)
    camera = scene.camera
    if obj is not None and camera is not None:  # only what the plane covers
        corners = [world_to_camera_view(scene, camera, obj.matrix_world @ v.co) for v in obj.data.vertices]
        if len(corners) >= 3 and all(c.z > 0 for c in corners):
            ys, xs = np.mgrid[0:rows, 0:cols]
            px, py = (xs + 0.5) / cols, (ys + 0.5) / rows  # bottom-up, as Blender's rows run
            sides = []
            for i, c in enumerate(corners):
                d = corners[(i + 1) % len(corners)]
                sides.append((d.x - c.x) * (py - c.y) - (d.y - c.y) * (px - c.x))
            keep = np.all(np.stack(sides) >= 0, axis=0) | np.all(np.stack(sides) <= 0, axis=0)
    direction = np.asarray(screen_direction(node.corridorkey_shader), dtype=np.float32)
    keep &= (shown @ direction) > 0.08  # the screen's hue: floor, not actor or set
    found = linear[keep]
    if len(found) < 200:
        return None, "Too little screen floor in view here: try another frame, or pick the colour."
    bright = found @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    low, high = np.percentile(bright, [75, 95])
    chosen = found[(bright >= low) & (bright <= high)]
    colour = chosen.mean(axis=0) if len(chosen) else found.mean(axis=0)
    return tuple(float(c) for c in colour), None


def make_floor_shadows(node, src, scene):
    """The plane, its material and its floor colour. (object, note) or (None, why not)."""
    corners = floor_footprint(scene)
    if corners is None:
        return None, "Needs the scene camera above the floor (Z = 0)."
    plate = wired_image_node(src, 0)
    if plate is None or plate.image is None:
        return None, "Connect the footage first."
    owner = node.id_data
    name = f"{FLOOR_GROUP} ({owner.name})" if owner is not None else FLOOR_GROUP
    mesh = bpy.data.meshes.new(name)
    floor_mesh(mesh, corners)
    obj = bpy.data.objects.new(name, mesh)
    obj[FLOOR_TAG] = st(src).uid
    collection = scene.collection
    for view_layer_obj in (bpy.context.active_object,):
        if view_layer_obj is not None and view_layer_obj.users_collection:
            collection = view_layer_obj.users_collection[0]
    collection.objects.link(obj)
    for attr in ("visible_shadow", "visible_diffuse", "visible_glossy", "visible_transmission",
                 "visible_volume_scatter"):
        if hasattr(obj, attr):
            setattr(obj, attr, False)  # seen by the camera only: it lights and shades nothing

    material = bpy.data.materials.new(name)
    if hasattr(material, "use_nodes") and not material.use_nodes:
        try:
            material.use_nodes = True
        except (AttributeError, TypeError):
            pass
    tree = material.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    coords = tree.nodes.new("ShaderNodeTexCoord")
    coords.location = (-500, 0)
    texture = tree.nodes.new("ShaderNodeTexImage")
    texture.name = texture.label = FLOOR_TEX
    texture.extension = "CLIP"
    texture.location = (-300, 0)
    tree.links.new(coords.outputs["Window"], texture.inputs["Vector"])
    group = tree.nodes.new("ShaderNodeGroup")
    group.node_tree = ensure_floor_group()
    group.name = group.label = FLOOR_NODE
    group.location = (0, 0)
    tree.links.new(texture.outputs["Color"], group.inputs["Footage"])
    tree.links.new(group.outputs["Shader"], out.inputs["Surface"])
    clean = tree.nodes.new("ShaderNodeTexImage")
    clean.name = clean.label = FLOOR_CLEAN
    clean.extension, clean.interpolation = "EXTEND", "Linear"
    clean.location = (-300, -300)
    tree.links.new(coords.outputs["Window"], clean.inputs["Vector"])
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "BLENDED"  # EEVEE: a coloured see-through needs blending
    elif hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    material.use_backface_culling = False
    obj.data.materials.append(material)
    node.corridorkey_shader.floor_name = obj.name
    floor_sync(obj, plate)
    colour, why = sample_floor_colour(node, src, scene, obj)
    if colour is None:
        group.inputs["Floor Colour"].default_value = (*_linear(_display(node.corridorkey_shader.screen_pick)), 1.0)
        return obj, why + " Using the Screen Key colour for now."
    group.inputs["Floor Colour"].default_value = (*colour, 1.0)
    return obj, ""


def _hue_of(colour_linear) -> list[float]:
    """A colour's hue as a unit vector in display-like values (as screen_direction)."""
    shown = _display(colour_linear)
    mean = sum(shown) / 3
    chroma = [c - mean for c in shown]
    length = sum(c * c for c in chroma) ** 0.5
    if length < 0.02:
        return list(screen_direction(types.SimpleNamespace(screen_pick=_linear(SCREEN_GREEN))))
    return [c / length for c in chroma]


def start_floor_bake(node, src, scene) -> str | None:
    """Bake the clean floor for every frame, in background Blenders: how bright the unshadowed
    floor is across the frame, so the lighting's falloff isn't taken for shadow."""
    uid = st(src).uid
    if uid in FLOOR_JOBS:
        return None
    obj = floor_object(node, src)
    group = floor_group_node(obj)
    if group is None:
        return "Add Floor Shadows first."
    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None or image.source != "SEQUENCE":
        return "The clean floor bakes from an image sequence (not a movie)."
    folder = os.path.dirname(bpy.path.abspath(image.filepath, library=image.library))
    plates = numbered(folder)
    if not plates:
        return "No footage frames found."
    numbers = sorted(plates)
    root = bpy.path.abspath("//CorridorKey/Floor") if bpy.data.filepath else os.path.join(
        tempfile.gettempdir(), "CorridorKey Floor")
    os.makedirs(root, exist_ok=True)
    taken = [int(m.group(1)) for f in os.listdir(root) if (m := re.match(r"^clean floor (\d+)$", f))]
    out_dir = os.path.join(root, f"clean floor {max(taken, default=0) + 1}")
    os.makedirs(out_dir)
    work = os.path.join(tempfile.gettempdir(), "CorridorKey Floor", uuid.uuid4().hex[:8])
    os.makedirs(work)
    colour = list(group.inputs["Floor Colour"].default_value)[:3]
    workers = max(1, min(4, (os.cpu_count() or 4) // 4))
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screen_key_worker.py")
    flags = 0x08000000 if sys.platform == "win32" else 0
    processes = []
    for index in range(workers):
        request = {
            "mode": "clean_floor", "frames": [[n, os.path.join(folder, plates[n]), None] for n in numbers],
            "colour": colour, "direction": _hue_of(colour), "out_dir": out_dir, "prefix": "clean_",
            "slice": [index, workers],
        }
        path = os.path.join(work, f"request_{index}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(request, f)
        try:
            processes.append(subprocess.Popen(
                [bpy.app.binary_path, "-b", "--factory-startup", "--python", script, "--", path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            ))
        except OSError as exc:
            for process in processes:
                process.kill()
            shutil.rmtree(work, ignore_errors=True)
            return f"The clean-floor bake didn't start: {exc}"
    FLOOR_JOBS[uid] = {"processes": processes, "work": work, "out_dir": out_dir, "total": len(numbers),
                       "object": obj.name, "started": time.monotonic()}
    PREVIEW_LINE["floor:" + uid] = "Baking the clean floor..."
    return None


def cancel_floor_bake(uid: str) -> None:
    job = FLOOR_JOBS.get(uid)
    if job is not None:
        job["cancelled"] = True
        for process in job["processes"]:
            try:
                process.kill()
            except OSError:
                pass


def plug_clean_floor(obj, out_dir: str, plate) -> bool:
    """A finished clean-floor bake into the plane's material (or out of it, with no folder)."""
    material = obj.active_material if obj is not None else None
    tree = material.node_tree if material is not None else None
    group = tree.nodes.get(FLOOR_NODE) if tree is not None else None
    clean = tree.nodes.get(FLOOR_CLEAN) if tree is not None else None
    if group is None or clean is None:
        return False
    for link in list(group.inputs["Clean Floor"].links):
        tree.links.remove(link)
    if not out_dir:
        clean.image = None
        return True
    names = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    if not names:
        return False
    path = os.path.join(out_dir, names[0])
    if bpy.data.filepath:
        try:
            path = bpy.path.relpath(path)
        except ValueError:
            pass
    image = bpy.data.images.load(path, check_existing=False)
    image.source = "SEQUENCE"
    image.colorspace_settings.name = "Non-Color"
    image.name = f"Clean floor ({obj.name})"
    clean.image = image
    tree.links.new(clean.outputs["Color"], group.inputs["Clean Floor"])
    if plate is not None:
        floor_sync(obj, plate)
    return True


def land_floor_bakes() -> bool:
    """Clean-floor bakes: how far along; the finished ones plugged into their plane."""
    changed = False
    for uid, job in list(FLOOR_JOBS.items()):
        try:
            done = len([f for f in os.listdir(job["out_dir"]) if f.endswith(".png")])
        except OSError:
            done = 0
        if any(process.poll() is None for process in job["processes"]):
            line = f"Baking the clean floor: {done} of {job['total']} frames"
            if PREVIEW_LINE.get("floor:" + uid) != line:
                PREVIEW_LINE["floor:" + uid] = line
                changed = True
            continue
        del FLOOR_JOBS[uid]
        changed = True
        shutil.rmtree(job["work"], ignore_errors=True)
        errors = sorted(f for f in os.listdir(job["out_dir"]) if f.startswith("error_")) if os.path.isdir(job["out_dir"]) else []
        message = ""
        for name in errors:
            with open(os.path.join(job["out_dir"], name), encoding="utf-8", errors="replace") as f:
                message = message or f.read().strip()[:160]
            os.remove(os.path.join(job["out_dir"], name))
        obj = bpy.data.objects.get(job["object"])
        src = node_by_uid(uid)
        if job.get("cancelled"):
            PREVIEW_LINE["floor:" + uid] = "Clean-floor bake stopped."
        elif done < job["total"] or obj is None:
            PREVIEW_LINE["floor:" + uid] = f"Clean-floor bake stopped at {done} of {job['total']} frames" + (
                f": {message}" if message else ".")
        else:
            plug_clean_floor(obj, job["out_dir"], wired_image_node(src, 0) if src is not None else None)
            seconds = int(time.monotonic() - job["started"])
            PREVIEW_LINE["floor:" + uid] = (f"Clean floor baked: {done} frames in {seconds // 60}m "
                                            f"{seconds % 60:02d}s. Falloff no longer reads as shadow.")
    return changed


class CORRIDORKEY_SHADER_OT_floor_bake(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_bake"
    bl_label = "Bake Clean Floor"
    bl_description = (
        "Measure, frame by frame, how bright the floor is where nothing shades it (a smooth fit "
        "that ignores shadows), so uneven lighting isn't taken for shadow. Runs in the background"
    )

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        if node is None or src is None:
            return {"CANCELLED"}
        why = start_floor_bake(node, src, context.scene)
        if why:
            PREVIEW_LINE["floor:" + self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_bake_cancel(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_bake_cancel"
    bl_label = "Stop Baking"
    bl_description = "Stop baking the clean floor"

    def execute(self, context):
        cancel_floor_bake(self.uid)
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_unbake(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_unbake"
    bl_label = "Use the Floor Colour Only"
    bl_description = "Unplug the baked clean floor: the whole floor is compared with the Floor Colour"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        obj = floor_object(node, src) if node is not None and src is not None else None
        if obj is None:
            return {"CANCELLED"}
        plug_clean_floor(obj, "", None)
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_add(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_add"
    bl_label = "Add Floor Shadows"
    bl_description = (
        "Keep the real shadows on the floor: a see-through plane on the floor (Z = 0) that darkens "
        "what's behind it wherever the footage's floor is shadowed. It sees the footage through "
        "the camera, so it follows the camera by itself"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        if node is None or src is None:
            return {"CANCELLED"}
        obj, note = make_floor_shadows(node, src, context.scene)
        if obj is None:
            self.report({"WARNING"}, note)
            return {"CANCELLED"}
        PREVIEW_LINE["floor:" + self.uid] = note or "Floor colour sampled from this frame."
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_fit(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_fit"
    bl_label = "Fit to View"
    bl_description = "Reshape the floor plane to what the camera sees of the floor on this frame"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        obj = floor_object(node, src) if node is not None and src is not None else None
        corners = floor_footprint(context.scene)
        if obj is None or corners is None:
            return {"CANCELLED"}
        inverse = obj.matrix_world.inverted()
        floor_mesh(obj.data, [inverse @ p for p in corners])
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_sample(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_sample"
    bl_label = "Sample Floor Colour"
    bl_description = "Measure the clean floor's colour on the frame you're on (pick a frame with little shadow)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        obj = floor_object(node, src) if node is not None and src is not None else None
        group = floor_group_node(obj)
        if group is None:
            return {"CANCELLED"}
        colour, why = sample_floor_colour(node, src, context.scene, obj)
        if colour is None:
            PREVIEW_LINE["floor:" + self.uid] = why
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        group.inputs["Floor Colour"].default_value = (*colour, 1.0)
        PREVIEW_LINE["floor:" + self.uid] = f"Floor colour sampled on frame {context.scene.frame_current}."
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_matte(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_matte"
    bl_label = "Show Matte"
    bl_description = "Show the shadow matte itself (white = darkened), to tune Lift and Screen Gate"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        group = floor_group_node(floor_object(node, src)) if node is not None and src is not None else None
        if group is None:
            return {"CANCELLED"}
        socket = group.inputs["Show Matte"]
        socket.default_value = 0.0 if socket.default_value > 0.5 else 1.0
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_floor_remove(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.floor_remove"
    bl_label = "Remove Floor Shadows"
    bl_description = "Delete the floor-shadow plane and its material"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        node, src = node_by_source(self.uid), self.source()
        obj = floor_object(node, src) if node is not None and src is not None else None
        if obj is None:
            return {"CANCELLED"}
        cancel_floor_bake(self.uid)
        mesh, materials = obj.data, [m for m in obj.data.materials if m is not None]
        bpy.data.objects.remove(obj)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        for material in materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        node.corridorkey_shader.floor_name = ""
        PREVIEW_LINE.pop("floor:" + self.uid, None)
        return {"FINISHED"}


def draw_floor_shadows(layout, node, src, scene) -> None:
    uid, settings = st(src).uid, node.corridorkey_shader
    obj = floor_object(node, src)
    group = floor_group_node(obj)
    showing = group is not None and group.inputs["Show Matte"].default_value > 0.5
    note = ("baking" if uid in FLOOR_JOBS else "showing the matte" if showing else "on") if group is not None else ""
    box = _fold(layout, settings, "ui_floor", "Floor Shadows", "MESH_PLANE", note)
    if group is None:
        _lines(box, "Keep the real shadows on the floor: a see-through plane on the floor that "
                    "darkens what's behind it where the footage is shadowed. It follows the camera "
                    "by itself.", "INFO", 40)
        row = box.row()
        row.scale_y = 1.3
        row.operator(CORRIDORKEY_SHADER_OT_floor_add.bl_idname, icon="ADD").uid = uid
        return
    row = box.row(align=True)
    row.prop(group.inputs["Floor Colour"], "default_value", text="Floor")
    row.operator(CORRIDORKEY_SHADER_OT_floor_sample.bl_idname, text="", icon="EYEDROPPER").uid = uid
    column = box.column(align=True)
    for name in ("Density", "Lift", "Screen Gate", "Keep Colour", "Edge Feather"):
        column.prop(group.inputs[name], "default_value", text=name, slider=True)
    baked = bool(group.inputs["Clean Floor"].links)
    row = box.row(align=True)
    if uid in FLOOR_JOBS:
        row.operator(CORRIDORKEY_SHADER_OT_floor_bake_cancel.bl_idname, icon="CANCEL").uid = uid
    else:
        row.operator(CORRIDORKEY_SHADER_OT_floor_bake.bl_idname,
                     text="Re-bake Clean Floor" if baked else "Bake Clean Floor", icon="LIGHT_AREA").uid = uid
        if baked:
            row.operator(CORRIDORKEY_SHADER_OT_floor_unbake.bl_idname, text="", icon="X").uid = uid
    if not baked and uid not in FLOOR_JOBS:
        box.label(text="Uneven lighting reads as shadow until you bake.", icon="INFO")
    row = box.row(align=True)
    row.operator(CORRIDORKEY_SHADER_OT_floor_matte.bl_idname, icon="HIDE_OFF", depress=showing).uid = uid
    row.operator(CORRIDORKEY_SHADER_OT_floor_fit.bl_idname, icon="FULLSCREEN_ENTER").uid = uid
    row.operator(CORRIDORKEY_SHADER_OT_floor_remove.bl_idname, text="", icon="TRASH").uid = uid
    line = PREVIEW_LINE.get("floor:" + uid)
    if uid in FLOOR_JOBS and line:
        draw_progress(box, line)
    elif line:
        _lines(box, line, "BLANK1", 40)
    _lines(box, f"Plane: {obj.name}. Lines up through the camera; shrink it in Edit Mode to the "
                "green floor if walls get in.", "BLANK1", 40)


def _paid_op(name: str) -> str | None:
    """Corridor's operator `corridorkey.<name>`, when its add-on has it."""
    return f"corridorkey.{name}" if hasattr(bpy.ops.corridorkey, name) else None


def draw_render(layout, node, src, scene) -> None:
    """Everything that renders, in one box: the shot or some of its frames, F12, where the key
    goes, and locking it for a render farm."""
    uid, state, settings = st(src).uid, st(src), node.corridorkey_shader
    free = is_free_source(src)
    range_job = PREVIEW_JOBS.get(uid) if (PREVIEW_JOBS.get(uid) or {}).get("dest") else None
    shot_job = uid in FREE_RENDERS if free else uid in set(getattr(ck("operators"), "JOBS", None) or ())
    box = _fold(layout, settings, "ui_render", "Render", "RENDER_ANIMATION",
                "rendering" if (range_job or shot_job) else "")
    box.context_pointer_set("active_node", src)  # Corridor's buttons act on the active node
    busy = bool(range_job or shot_job)
    line = PREVIEW_LINE.get(uid) if uid in RANGE_LINES else state.render_line
    if busy and line:
        draw_progress(box, line)
    else:
        # one status line: how the key stands, and the GPU it needs against what there is
        row = box.row()
        status = row.row()
        status.label(text=(line or "not rendered").rstrip(".").capitalize(),
                     icon="CHECKMARK" if result_ready(src) else "RENDER_RESULT")
        gpu = "" if free else gpu_line(state)
        if gpu:
            side = row.row()
            side.alignment = "RIGHT"
            side.label(text=gpu, icon="MEMORY")
    if state.runtime_message:
        _lines(box, state.runtime_message, "ERROR", 34)

    box.row().prop(settings, "render_scope", expand=True)
    if settings.render_scope == "RANGE":
        row = box.row(align=True)
        row.prop(settings, "range_start")
        row.prop(settings, "range_end")
        row.operator(CORRIDORKEY_SHADER_OT_range_from_timeline.bl_idname, text="", icon="TIME").uid = uid
        box.row().prop(settings, "range_into", expand=True)

    row = box.row(align=True)
    row.scale_y = 1.5
    if range_job:
        row.operator(CORRIDORKEY_SHADER_OT_cancel_preview.bl_idname, text="Cancel Render", icon="CANCEL").uid = uid
    elif shot_job:
        if free:
            row.operator(CORRIDORKEY_SHADER_OT_free_cancel.bl_idname, text="Cancel Render", icon="CANCEL").uid = uid
        elif _paid_op("cancel"):
            row.operator(_paid_op("cancel"), text="Cancel Render", icon="CANCEL").node_uid = uid
    elif settings.render_scope == "RANGE":
        first, last = sorted((settings.range_start, settings.range_end))
        what = f"{first}-{last}" if last > first else f"{first}"
        row.operator(CORRIDORKEY_SHADER_OT_preview_range.bl_idname, text="Preview", icon="HIDE_OFF").uid = uid
        row.operator(CORRIDORKEY_SHADER_OT_render_range.bl_idname, text=f"Render {what}",
                     icon="RENDER_ANIMATION").uid = uid
    elif free:
        row.enabled = free_setup_problem() is None
        row.operator(CORRIDORKEY_SHADER_OT_free_render.bl_idname, text="Render Shot", icon="RENDER_ANIMATION").uid = uid
    elif _paid_op("render"):
        row.operator(_paid_op("render"), text="Render Shot", icon="RENDER_ANIMATION").node_uid = uid

    box.prop(scene, "corridorkey_key_before_render", text="Key the Frame Before F12")
    row = box.row()
    row.enabled = not shot_job
    row.prop(state, "display_rendered_frames", text="Show the Rendered Key")

    # the Render Folder as a line, not a path field (Blender paints those red at times when
    # nothing is wrong), with its size, and buttons to choose, open and empty it
    folder = state.render_folder or "not set"
    size = "" if free else getattr(state, "render_folder_size", "")
    row = box.row(align=True)
    row.label(text=f"{folder}  ({size})" if size else folder, icon="FILE_FOLDER")
    row.operator(CORRIDORKEY_SHADER_OT_pick_render_folder.bl_idname, text="", icon="FILEBROWSER").uid = uid
    row = box.row(align=True)
    if free:
        row.operator(CORRIDORKEY_SHADER_OT_free_open_folder.bl_idname, text="Open Folder", icon="FOLDER_REDIRECT").uid = uid
    elif _paid_op("open_render_folder"):
        row.operator(_paid_op("open_render_folder"), text="Open Folder", icon="FOLDER_REDIRECT").node_uid = uid
    row.operator(CORRIDORKEY_SHADER_OT_delete_render.bl_idname, text="Delete Render", icon="TRASH").uid = uid

    # Render Farm: one row; while there's nothing to lock the button is greyed and its tooltip
    # says why
    box.separator()
    result, _why = lock_check(src, scene)
    row = box.row(align=True)
    row.label(text="Render Farm", icon="NETWORK_DRIVE")
    buttons = row.row(align=True)
    buttons.enabled = result is not None
    buttons.operator(CORRIDORKEY_SHADER_OT_lock.bl_idname, text="Lock", icon="LOCKED")
    others = sum(1 for _n in shader_nodes())
    if others > 1:
        row.operator(CORRIDORKEY_SHADER_OT_lock.bl_idname, text=f"All {others}", icon="LOCKED").every = True


def result_ready(src) -> bool:
    """Whether a rendered key is plugged in to show."""
    result = src.node_tree.nodes.get(RESULT) if getattr(src, "node_tree", None) is not None else None
    return result is not None and result.image is not None


def gpu_line(state) -> str:
    """The GPU a render needs against what the card has, short: "GPU 5.1 / 9.9 GB"."""
    needs = getattr(state, "gpu_memory_required", "") or ""
    has = getattr(state, "gpu_memory_available", "") or ""
    if not re.search(r"\d", needs) or not re.search(r"\d", has):  # "Shown after the first render"
        return ""
    if needs.endswith(" GB") and has.endswith(" GB"):
        return f"{needs[:-3]} / {has}"
    return f"{needs} / {has}"


def draw_paid_hint_settings(layout, src) -> None:
    """Corridor's Alpha Hint settings, as its own panel draws them."""
    panel = ck("panel")
    if hasattr(panel, "_draw_alpha_hint"):
        state = panel.settings(src) if hasattr(panel, "settings") else st(src)
        panel._draw_alpha_hint(layout, src, state)


def draw_paid_controls(outer, src, scene, before_despill=None, before_help=None) -> None:
    """Corridor's sections for the node, as its own panel draws them, less its Render section
    (which is in the Render box), with ours drawn in between (`before_despill`, `before_help`
    draw into `outer`)."""
    panel, words, operators = ck("panel"), ck("words"), ck("operators")
    layout = engine_column(outer, src)
    needed = ("_line", "_section", "_draw_controls", "_draw_time_awareness", "_draw_alpha_hint",
              "_draw_output", "_draw_despill", "_draw_help")
    if not all(hasattr(panel, n) for n in needed):  # a newer Corridor add-on: its own panel, whole
        panel.draw_controls(layout, src)
        for hook in (before_despill, before_help):
            if hook:
                hook(outer)
        return
    state = panel.settings(src) if hasattr(panel, "settings") else st(src)
    status = operators.status_line(src, scene) if hasattr(operators, "status_line") else ""
    if status and status.strip("- "):  # a job's status (the render box has the rest)
        panel._line(layout, words.STATUS, status)
    if state.licence_line:
        panel._line(layout, words.LICENSE, state.licence_line)
        layout.operator("corridorkey.manage_license", text=words.MANAGE_LICENSE)
        layout.separator()
    hooks = {"despill": before_despill, "help": before_help}
    for key, title, draw, closed in (
        ("controls", words.SECTION_CONTROLS, panel._draw_controls, False),
        ("time_awareness", words.SECTION_TIME_AWARENESS, panel._draw_time_awareness, False),
        ("output", words.SECTION_OUTPUT, panel._draw_output, False),
        ("despill", words.SECTION_DESPILL, panel._draw_despill, False),
        ("help", words.SECTION_HELP, panel._draw_help, True),
    ):
        if hooks.get(key):
            hooks[key](outer)
            layout = engine_column(outer, src)  # Corridor's next sections go below ours
        # Corridor's sections under its own names (their open state shared with its panel),
        # with icons like ours
        header, body = layout.panel(f"corridorkey_{key}", default_closed=closed)
        header.label(text=title, icon=SECTION_ICONS.get(key, "NONE"))
        if body:
            draw(body, src, state)


# --- one-click set-up: turn a footage plane into a CorridorKey plane ----------------------------

_MATTE_WORDS = ("matte", "mask", "alpha", "_mat", "hint", "aimatte")


def _is_matte(texture) -> bool:
    text = " ".join(filter(None, (texture.name, texture.label, texture.image.name if texture.image else "",
                                  texture.image.filepath if texture.image else ""))).lower()
    if any(word in text for word in _MATTE_WORDS):
        return True
    targets = [l.to_socket for l in texture.outputs["Color"].links]
    return bool(targets) and all(t.type == "VALUE" or t.name in {"Fac", "Factor", "Alpha"} for t in targets)


def find_plane_textures(tree):
    """(footage texture, matte texture or None) on a footage plane's material."""
    textures = [n for n in tree.nodes if n.bl_idname == "ShaderNodeTexImage" and n.image is not None]
    moving = [n for n in textures if n.image.source in {"SEQUENCE", "MOVIE"}] or textures
    mattes = [n for n in moving if _is_matte(n)]
    plates = [n for n in moving if n not in mattes]
    footage = next((n for n in plates if n.outputs["Color"].is_linked), plates[0] if plates else None)
    matte = next((n for n in mattes if n.outputs["Color"].is_linked), mattes[0] if mattes else None)
    return footage, matte


def _material_of(obj):
    slots = [s for s in getattr(obj, "material_slots", ()) if s.material is not None and s.material.node_tree]
    if not slots:
        return None
    active = obj.active_material
    for material in ([active] if active else []) + [s.material for s in slots]:
        if material is not None and material.node_tree is not None and find_plane_textures(material.node_tree)[0]:
            return material
    return None


def _autoshot_compositor(scene) -> bool:
    tree = getattr(scene, "compositing_node_group", None) or getattr(scene, "node_tree", None)
    if tree is None:
        return False
    for n in tree.nodes:
        group = getattr(n, "node_tree", None)
        if n.bl_idname == "CompositorNodeGroup" and group is not None and (
                group.name.lower().startswith("autoshot") or "key" in n.name.lower()):
            return True
        if n.bl_idname == "CompositorNodeKeying":
            return True
    return False


class CORRIDORKEY_SHADER_OT_setup_plane(bpy.types.Operator):
    bl_idname = "corridorkey_shader.setup_plane"
    bl_label = "Set Up CorridorKey 2.0"
    bl_description = (
        "One click: put a CorridorKey 2.0 node between this plane's footage and its shader, use its "
        "matte (if it has one) as the Alpha Hint, and turn off a compositor key that would cover it"
    )
    bl_options = {"REGISTER", "UNDO"}

    use_matte: bpy.props.BoolProperty(name="Use the plane's matte as the Alpha Hint", default=True)
    disable_compositor: bpy.props.BoolProperty(
        name="Turn off the compositor's key", default=True,
        description="Jetset/Autoshot scenes key the raw plate again in the compositor, over CorridorKey's key",
    )
    make_hint: bpy.props.BoolProperty(
        name="Make an Alpha Hint with SAM 3 when there's no matte", default=True,
    )
    warm_up: bpy.props.BoolProperty(
        name="Preview this frame now", default=True,
        description="Key the frame you're on straight away: it loads the model in the background, so "
                    "later previews take about a second",
    )

    @classmethod
    def poll(cls, context) -> bool:
        obj = context.active_object
        if obj is None or _material_of(obj) is None:
            cls.poll_message_set("Select a plane whose material shows footage (an Image Texture)")
            return False
        if engine() == "paid" and ck() is None:
            cls.poll_message_set("Enable Corridor's CorridorKey add-on, or pick the free engine")
            return False
        return True

    def execute(self, context):
        obj = context.active_object
        material = _material_of(obj)
        tree = material.node_tree
        if any(is_locked(n) for n in tree.nodes):
            self.report({"INFO"}, f"{material.name} is locked for rendering: unlock it to change it.")
            return {"CANCELLED"}
        existing = [n for n in tree.nodes if is_ck_shader(n)]
        if existing:
            for n in tree.nodes:
                n.select = False
            existing[0].select = True
            tree.nodes.active = existing[0]
            self.report({"INFO"}, f"{material.name} already has a CorridorKey 2.0 node.")
            return {"CANCELLED"}
        footage, matte = find_plane_textures(tree)
        color_targets = [l.to_socket for l in footage.outputs["Color"].links]
        alpha_targets = [l.to_socket for l in footage.outputs["Alpha"].links]
        if matte is not None:
            alpha_targets += [l.to_socket for l in matte.outputs["Color"].links]
            alpha_targets += [l.to_socket for l in matte.outputs["Alpha"].links]

        src = new_source()
        node = tree.nodes.new("ShaderNodeGroup")
        node.node_tree = new_shader_tree(st(src).uid, source_key())
        node.label, node.width = NAME, 200
        node.location = (footage.location.x + 340, footage.location.y)
        tree.links.new(footage.outputs["Color"], node.inputs[IMAGE_IN])
        if matte is not None and self.use_matte:
            tree.links.new(matte.outputs["Color"], node.inputs[HINT_IN])
        for socket in color_targets:
            tree.links.new(node.outputs["Color"], socket)
        for socket in alpha_targets:
            tree.links.new(node.outputs["Alpha"], socket)

        made = []
        if not color_targets or not alpha_targets:
            made = self._shader_for(tree, node, color_targets, alpha_targets)
        if hasattr(material, "surface_render_method") and not alpha_targets:
            material.surface_render_method = "DITHERED"
        for n in tree.nodes:
            n.select = False
        node.select = True
        tree.nodes.active = node

        notes = [f"{material.name}: CorridorKey 2.0 set up"]
        if matte is not None and self.use_matte:
            notes.append(f"its matte '{matte.label or matte.image.name}' is the Alpha Hint")
        if made:
            notes.append("added " + ", ".join(made))
        scene = context.scene
        if self.disable_compositor and scene.render.use_compositing and _autoshot_compositor(scene):
            scene.render.use_compositing = False
            notes.append("compositing turned off (it keyed the raw plate over CorridorKey)")
        tick()  # the node takes its footage now
        hinting = False
        if matte is None and self.make_hint and sam3_problem() is None:
            why = start_hint(node, scene)
            hinting = not why
            notes.append("SAM 3 is making an Alpha Hint" if not why else f"no Alpha Hint: {why}")
        if self.warm_up and not hinting and engine_available():
            why = start_preview(src, scene, [scene.frame_current])
            if not why:  # the model loads now, so the first preview the user asks for is quick
                notes.append(f"previewing frame {scene.frame_current} (the model loads: about 25 s)")
        self.report({"INFO"}, "; ".join(notes) + ".")
        return {"FINISHED"}

    @staticmethod
    def _shader_for(tree, node, color_targets, alpha_targets) -> list[str]:
        """Give the key somewhere to go when the footage fed nothing (or nothing used its alpha)."""
        made = []
        output = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial" and n.is_active_output),
                      None) or next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
        if output is None:
            output = tree.nodes.new("ShaderNodeOutputMaterial")
            output.location = (node.location.x + 700, node.location.y)
            made.append("a Material Output")
        principled = next((s.node for s in color_targets if s.node.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if principled is not None and not alpha_targets:
            tree.links.new(node.outputs["Alpha"], principled.inputs["Alpha"])
            return made
        if color_targets and not alpha_targets:  # an Emission (or such): mix it with transparency
            shader = color_targets[0].node
            out_socket = next((o for o in shader.outputs if o.is_linked), shader.outputs[0])
            links = list(out_socket.links)
            destination = links[0].to_socket if links else output.inputs["Surface"]
        else:  # the footage fed nothing: an emissive, keyed surface
            emission = tree.nodes.new("ShaderNodeEmission")
            emission.location = (node.location.x + 260, node.location.y + 60)
            tree.links.new(node.outputs["Color"], emission.inputs["Color"])
            out_socket, destination = emission.outputs[0], output.inputs["Surface"]
            made.append("an Emission shader")
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        transparent.location = (node.location.x + 260, node.location.y - 140)
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.location = (node.location.x + 480, node.location.y)
        tree.links.new(node.outputs["Alpha"], mix.inputs[0])
        tree.links.new(transparent.outputs[0], mix.inputs[1])
        tree.links.new(out_socket, mix.inputs[2])
        tree.links.new(mix.outputs[0], destination)
        made.append("a transparency mix")
        return made


def _context_menu(self, context) -> None:
    obj = context.active_object
    if obj is not None and not object_nodes(obj) and not locked_nodes(obj) and _material_of(obj) is not None:
        self.layout.separator()
        self.layout.operator(CORRIDORKEY_SHADER_OT_setup_plane.bl_idname, icon="NODE_MATERIAL")


# --- the free engine: settings, full render, controls, setup --------------------------------------


def _free_display_changed(state, _context) -> None:
    for src in iter_sources():
        if is_free_source(src) and src.corridorkey2.uid == state.uid:
            apply_display(src)


class CorridorKeyFreeSettings(bpy.types.PropertyGroup):
    uid: bpy.props.StringProperty(options={"HIDDEN"})
    render_folder: bpy.props.StringProperty(
        name="Render Folder", subtype="DIR_PATH",
        description="Where full renders are written, each in a Render folder of its own",
    )
    input_linear: bpy.props.BoolProperty(
        name="Input is Linear?", default=False,
        description="Tick for scene-linear EXR plates (otherwise footage is read as sRGB)",
    )
    internal_resolution: bpy.props.EnumProperty(
        name="Internal Resolution",
        description="The resolution CorridorKey works at; the key is always written at full size",
        items=[("1.0", "Full", ""), ("0.75", "Three Quarters", ""), ("0.5", "Half", ""), ("0.25", "Quarter", "")],
        default="1.0",
    )
    time_awareness: bpy.props.EnumProperty(
        name="Time Awareness",
        description="How many frames CorridorKey looks at together in a full render",
        items=[("AUTO", "Automatic", "The largest window your GPU fits"), ("CUSTOM", "Custom", "A window you set")],
        default="AUTO",
    )
    window: bpy.props.IntProperty(
        name="Window", default=80, min=1, max=160,
        description="Frames looked at together in a full render (overlapping by half)",
    )
    preview_context: bpy.props.IntProperty(
        name="Preview Context", default=4, min=0, max=24,
        description="Frames either side a full-quality preview looks at (Quick Preview uses none)",
    )
    low_vram: bpy.props.BoolProperty(
        name="Low GPU Memory", default=False,
        description="Slower, staged keying for GPUs with little memory",
    )
    use_alpha_hint: bpy.props.BoolProperty(name="Use Alpha Hint?", default=True)
    hint_frames: bpy.props.EnumProperty(
        name="Hint Frames",
        items=[("ALL", "All Frames", "The hint guides every frame"),
               ("FIRST", "First Frame", "The hint guides the first frame; CorridorKey carries it on")],
        default="ALL",
    )
    hint_quality: bpy.props.FloatProperty(
        name="Hint Strength", default=0.95, min=0.0, max=1.0,
        description="How much CorridorKey trusts the hint (about 0.95 for good mattes, 0.75 for rough ones)",
    )
    carry_hint: bpy.props.BoolProperty(
        name="Carry Hint", default=True,
        description="Carry CorridorKey's own matte between time windows",
    )
    carry_hint_quality: bpy.props.FloatProperty(
        name="Carried Hint Strength", default=0.95, min=0.0, max=1.0,
    )
    despill_strength: bpy.props.FloatProperty(
        name="Despill", default=DESPILL, min=0.0, max=1.0,
        description="How much green or blue spill is taken out of the cut-out (0 none, 1 full)",
    )
    output: bpy.props.EnumProperty(
        name="Output",
        items=[("CUTOUT", "Cut-Out", "The despilled foreground with its alpha"), ("MATTE", "Matte", "The alpha alone")],
        default="CUTOUT",
    )
    display_rendered_frames: bpy.props.BoolProperty(
        name="Display rendered frames?", default=True, update=_free_display_changed,
    )
    render_line: bpy.props.StringProperty(default="Not rendered", options={"HIDDEN"})
    runtime_message: bpy.props.StringProperty(options={"HIDDEN"})
    result_folder: bpy.props.StringProperty(options={"HIDDEN"})
    kept_render: bpy.props.StringProperty(options={"HIDDEN"})
    kept_render_owner: bpy.props.StringProperty(options={"HIDDEN"})
    kept_render_stamp: bpy.props.StringProperty(options={"HIDDEN"})
    result_frame_start: bpy.props.IntProperty(default=1, options={"HIDDEN"})
    result_frame_offset: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    result_frame_duration: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    result_colorspace: bpy.props.StringProperty(options={"HIDDEN"})
    result_alpha_mode: bpy.props.StringProperty(default="STRAIGHT", options={"HIDDEN"})


FREE_RENDERS: dict[str, dict] = {}  # source uid -> the free engine's full render in flight


def full_render_running(uid: str | None = None) -> bool:
    jobs = set(getattr(ck("operators"), "JOBS", None) or ()) | set(FREE_RENDERS)
    return bool(jobs) if uid is None else uid in jobs


def _next_render_folder(root: str) -> str:
    taken = [int(m.group(1)) for name in os.listdir(root) if (m := re.match(r"^Render (\d+)", name))]
    return os.path.join(root, f"Render {max(taken, default=0) + 1}")


def start_free_render(src, scene) -> str | None:
    """The free engine's Render: the whole shot, keyed into a new Render folder."""
    state = src.corridorkey2
    uid = state.uid
    if uid in FREE_RENDERS:
        return None
    problem = free_setup_problem()
    if problem:
        return problem
    plate = wired_image_node(src, 0)
    image = plate.image if plate is not None else None
    if image is None or image.source not in {"SEQUENCE", "MOVIE"}:
        return "Plug in footage: an image sequence or a movie."
    if not state.render_folder:
        state.render_line = "Choose a Render Folder"
        return state.render_line
    root = bpy.path.abspath(state.render_folder)
    os.makedirs(root, exist_ok=True)
    controls = free_controls(state)  # a preview keying now finishes first: one model, one queue
    working = os.path.join(root, f".corridorkey_working_{uuid.uuid4().hex[:8]}")
    os.makedirs(working)
    footage = bpy.path.abspath(image.filepath, library=image.library)
    if image.source == "SEQUENCE":
        folder = os.path.dirname(footage)
        plates = numbered(folder)
        if not plates:
            shutil.rmtree(working, ignore_errors=True)
            return "No frames found in the footage's folder."
        numbers = sorted(plates)
        input_path = os.path.join(working, "plates")
        os.makedirs(input_path)
        for n in numbers:
            _place(os.path.join(folder, plates[n]), input_path, f"{n:08d}{os.path.splitext(plates[n])[1]}")
    else:
        numbers, input_path = None, footage
    hint_first, hint_video = stage_free_hint(hint_of(src, scene), controls, numbers, working)
    out_dir = os.path.join(working, "out")
    request = free_request(controls, input_path, out_dir, hint_first, hint_video)
    clip = os.path.splitext(os.path.basename(footage))[0] if numbers is None else image.name.rsplit(".", 1)[0]
    prefix = re.sub(r"[_\-. ]*\d+$", "", clip) + "_"
    ticket = FreeTicket(request, numbers, prefix=prefix)
    if not free_session().submit(ticket):
        close_free()
        free_session().submit(ticket)
    total = len(numbers) if numbers is not None else max(plate.frame_duration, 1)
    FREE_RENDERS[uid] = {
        "ticket": ticket, "working": working, "root": root, "numbers": numbers, "total": total,
        "exr_dir": exr_dir_of(request), "started": time.monotonic(),
    }
    state.result_frame_start, state.result_frame_offset = plate.frame_start, plate.frame_offset
    state.result_frame_duration = plate.frame_duration
    state.result_colorspace, state.result_alpha_mode = result_read(src)
    state.render_line, state.runtime_message = "Rendering: starting CorridorKey 2.0", ""
    return None


def free_show_result(src, folder: str) -> bool:
    """Point the source's output at the frames in `folder` (as Corridor's node does)."""
    state, tree = src.corridorkey2, src.node_tree
    previous = [n.image for n in tree.nodes if n.name == RESULT and n.image is not None]
    for old in [n for n in tree.nodes if n.name == RESULT]:
        tree.nodes.remove(old)
    frames = sorted(f for f in os.listdir(folder) if f.lower().endswith(".exr")) if os.path.isdir(folder) else []
    if not frames:
        apply_display(src)
        return False
    image = bpy.data.images.load(os.path.join(folder, frames[0]), check_existing=True)
    image.source = "SEQUENCE" if len(frames) > 1 else "FILE"
    image.reload()
    try:
        image.colorspace_settings.name = state.result_colorspace or "Linear Rec.709"
    except TypeError:
        pass
    image.alpha_mode = state.result_alpha_mode or "STRAIGHT"
    result = tree.nodes.new("CompositorNodeImage")
    result.name = RESULT
    result.image = image
    result.location = (200, -200)
    result.frame_start = state.result_frame_start
    result.frame_offset = state.result_frame_offset
    result.frame_duration = state.result_frame_duration or len(frames)
    result.use_cyclic = False
    apply_display(src)
    for old_image in previous:
        if old_image != image and old_image.users == 0:
            bpy.data.images.remove(old_image)
    return True


def land_free_renders() -> bool:
    """Narrate the free engine's full renders; land the ones that ended."""
    changed = False
    for uid, entry in list(FREE_RENDERS.items()):
        src = node_by_uid(uid)
        ticket = entry["ticket"]
        if src is None:  # the node went: let its render go
            ticket.cancel()
        if ticket.state == "running":
            if src is not None:
                try:
                    done = len([f for f in os.listdir(entry["exr_dir"]) if f.endswith(".exr")])
                except OSError:
                    done = 0
                if done:
                    line = f"Rendering {done} of {entry['total']}"
                    elapsed = time.monotonic() - entry["started"]
                    if done > 2:
                        left = int(elapsed / done * (entry["total"] - done))
                        line += f" ({left // 60}m {left % 60:02d}s left)" if left >= 60 else f" ({left}s left)"
                else:
                    line = f"Rendering: {ticket.line}"
                if src.corridorkey2.render_line != line:
                    src.corridorkey2.render_line = line
                    changed = True
            continue
        del FREE_RENDERS[uid]
        changed = True
        if src is None:
            shutil.rmtree(entry["working"], ignore_errors=True)
            continue
        state = src.corridorkey2
        frames = len([f for f in os.listdir(ticket.output) if f.endswith(".exr")]) if ticket.output and os.path.isdir(ticket.output) else 0
        complete = ticket.state == "done" and frames >= entry["total"]
        if frames:
            final = _next_render_folder(entry["root"])
            if not complete:
                final += "-partial"
            os.replace(ticket.output, final)
            state.result_folder = final
            free_show_result(src, final)
            if complete:
                state.kept_render, state.kept_render_owner = final, uid
                previews_off(uid)
            state.render_line = "Shot rendered" if complete else f"Saved {frames} of {entry['total']} frames"
        else:
            state.render_line = "Cancelled" if ticket.state == "cancelled" else "Shot not run"
        state.runtime_message = ticket.message if ticket.state == "failed" else ""
        shutil.rmtree(entry["working"], ignore_errors=True)
        _LISTED.clear()
    return changed


class CORRIDORKEY_SHADER_OT_free_render(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.free_render"
    bl_label = "Render"
    bl_description = "Key the whole shot with CorridorKey 2.0 (free) into a new folder in the Render Folder"

    def execute(self, context):
        src = self.source()
        if src is None:
            return {"CANCELLED"}
        why = start_free_render(src, context.scene)
        if why:
            self.report({"WARNING"}, why)
            return {"CANCELLED"}
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_free_cancel(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.free_cancel"
    bl_label = "Cancel"
    bl_description = "Stop the render; the frames already keyed are kept"

    def execute(self, context):
        entry = FREE_RENDERS.get(self.uid)
        if entry is None:
            return {"CANCELLED"}
        entry["ticket"].cancel()
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_free_open_folder(_OnSource, bpy.types.Operator):
    bl_idname = "corridorkey_shader.free_open_folder"
    bl_label = "Open Render Folder"

    def execute(self, context):
        src = self.source()
        folder = bpy.path.abspath(src.corridorkey2.render_folder) if src is not None else ""
        if not folder or not os.path.isdir(folder):
            self.report({"WARNING"}, "The Render Folder doesn't exist yet.")
            return {"CANCELLED"}
        bpy.ops.wm.path_open(filepath=folder)
        return {"FINISHED"}


SECTION_ICONS = {"controls": "OPTIONS", "time_awareness": "TIME", "time": "TIME", "output": "OUTPUT",
                 "despill": "MATFLUID", "help": "HELP"}


def _section(layout, key: str, title: str, closed: bool = False):
    header, body = layout.panel(f"corridorkey2_free_{key}", default_closed=closed)
    header.label(text=title, icon=SECTION_ICONS.get(key, "NONE"))
    return body


def draw_free_hint_settings(layout, src) -> None:
    """The free engine's Alpha Hint settings."""
    state = src.corridorkey2
    layout.prop(state, "use_alpha_hint")
    column = layout.column()
    column.enabled = state.use_alpha_hint
    column.prop(state, "hint_frames")
    column.prop(state, "hint_quality")
    column.prop(state, "carry_hint")
    row = column.row()
    row.enabled = state.carry_hint
    row.prop(state, "carry_hint_quality")


def draw_free_controls(outer, src, before_despill=None, before_help=None) -> None:
    """The free engine's sections, with ours in between: the Alpha Hint above Output and Despill,
    the Tips last (the free engine has no Help section)."""
    state = src.corridorkey2
    layout = engine_column(outer, src)
    session = _FREE[0]
    if session is not None and session.alive() and session.gpu:
        layout.label(text=f"GPU: {session.gpu}", icon="MEMORY")

    body = _section(layout, "controls", "Controls")
    if body:
        body.prop(state, "input_linear")
        body.prop(state, "internal_resolution")

    body = _section(layout, "time", "Time Awareness")
    if body:
        body.prop(state, "time_awareness")
        row = body.row()
        row.enabled = state.time_awareness == "CUSTOM"
        row.prop(state, "window")
        body.prop(state, "preview_context")
        body.prop(state, "low_vram")

    if before_despill:
        before_despill(outer)
        layout = engine_column(outer, src)
    body = _section(layout, "output", "Output and Despill")
    if body:
        body.prop(state, "output")
        row = body.row()
        row.enabled = state.output == "CUTOUT"
        row.prop(state, "despill_strength")
    if before_help:
        before_help(outer)


FREE_CHECK: dict = {"line": "", "running": False}


def _check_free(python: str, root: str) -> None:
    code = (
        "import torch;"
        "print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(),"
        " '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no NVIDIA GPU found')"
    )
    flags = 0x08000000 if sys.platform == "win32" else 0
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    try:
        done = subprocess.run([python, "-c", code], cwd=root, env=env, capture_output=True, text=True,
                              timeout=300, creationflags=flags)
        lines = (done.stdout or done.stderr or "").strip().splitlines()
        FREE_CHECK["line"] = lines[-1][-120:] if lines else "No answer from CorridorKey 2.0's Python."
    except (OSError, subprocess.TimeoutExpired) as exc:
        FREE_CHECK["line"] = f"Couldn't run CorridorKey 2.0's Python: {exc}"
    FREE_CHECK["running"] = False


class CORRIDORKEY_SHADER_OT_free_check(bpy.types.Operator):
    bl_idname = "corridorkey_shader.free_check"
    bl_label = "Check Setup"
    bl_description = "Check that CorridorKey 2.0 (free) is installed, has its weights and sees your GPU"

    def execute(self, context):
        problem = free_setup_problem()
        if problem:
            FREE_CHECK["line"] = problem
            return {"FINISHED"}
        root = free_root()
        FREE_CHECK.update(line="Checking (importing PyTorch, a few seconds)...", running=True)
        threading.Thread(target=_check_free, args=(free_python(root), root), daemon=True).start()
        return {"FINISHED"}


_INSTALL_BAT = r"""@echo off
title CorridorKey 2.0 (free) - installing
echo Installing CorridorKey 2.0 (free) into "{root}"
echo This downloads several GB (PyTorch and CorridorKey's models). Leave this window open.
echo.
where git >nul 2>nul || (echo Git isn't installed. Get it from https://git-scm.com/download/win, then run Install again. & goto fail)
if not exist "{root}\infer.py" (git clone {repo}.git "{root}" || goto fail)
cd /d "{root}"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (where python >nul 2>nul && set "PY=python")
if not defined PY (echo Python 3 isn't installed. Get it from https://www.python.org/downloads/ ^(tick "Add to PATH"^), then run Install again. & goto fail)
if not exist "venv\Scripts\python.exe" (%PY% -m venv venv || goto fail)
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128 || goto fail
venv\Scripts\python.exe -m pip install -r requirements.txt || goto fail
venv\Scripts\python.exe scripts\download_weights.py || goto fail
echo.
echo Done. In Blender, press Check Setup in the CorridorKey 2.0 preferences.
pause
exit /b 0
:fail
echo.
echo The install stopped (see above). Fix that and press Install again; finished steps are skipped.
pause
exit /b 1
"""


class CORRIDORKEY_SHADER_OT_free_install(bpy.types.Operator):
    bl_idname = "corridorkey_shader.free_install"
    bl_label = "Install CorridorKey 2.0 (free)"
    bl_description = (
        "Download the free CorridorKey 2.0 from GitHub with its PyTorch and models (several GB) "
        "into the folder below, in a window you can watch. Needs Git and Python 3"
    )

    @classmethod
    def poll(cls, context) -> bool:
        if sys.platform != "win32":
            cls.poll_message_set("The installer is for Windows: follow the GitHub README on Linux")
            return False
        return True

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(
            self, event, title="Install CorridorKey 2.0 (free)?",
            message="Downloads several GB from GitHub, PyTorch and Hugging Face.", confirm_text="Install",
        )

    def execute(self, context):
        preferences = prefs()
        root = free_root() or os.path.join(os.path.expanduser("~"), "CorridorKey2.0")
        if not preferences.free_root:
            preferences.free_root = root
        os.makedirs(os.path.dirname(root), exist_ok=True)
        script = os.path.join(tempfile.gettempdir(), "install_corridorkey2_free.bat")
        with open(script, "w", encoding="utf-8") as bat:
            bat.write(_INSTALL_BAT.format(root=root, repo=FREE_REPO))
        subprocess.Popen(["cmd", "/c", script], creationflags=subprocess.CREATE_NEW_CONSOLE)
        FREE_CHECK["line"] = "Installing in the window that opened. Press Check Setup when it's done."
        return {"FINISHED"}


class CORRIDORKEY_SHADER_OT_open_prefs(bpy.types.Operator):
    bl_idname = "corridorkey_shader.open_prefs"
    bl_label = "CorridorKey 2.0 Preferences"
    bl_description = "Choose the engine and set up the free CorridorKey 2.0"

    def execute(self, context):
        import addon_utils

        try:
            addon_utils.module_bl_info(sys.modules[ADDON])["show_expanded"] = True
        except (KeyError, AttributeError, TypeError):
            pass
        context.preferences.active_section = "ADDONS"
        context.window_manager.addon_search = "CorridorKey 2.0"
        bpy.ops.screen.userpref_show(section="ADDONS")
        return {"FINISHED"}


class CorridorKey2Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON

    engine: bpy.props.EnumProperty(
        name="Engine",
        description="What keys the footage",
        items=[
            ("AUTO", "Automatic", "Corridor's paid add-on when it's enabled, otherwise the free GitHub version"),
            ("PAID", "Corridor Add-on (paid)", "Corridor Digital's CorridorKey add-on and runtime"),
            ("FREE", "Free (GitHub)", "The free CorridorKey 2.0 from GitHub, run with its own Python"),
        ],
        default="AUTO",
        update=_engine_changed,
    )
    free_root: bpy.props.StringProperty(
        name="CorridorKey 2.0 Folder", subtype="DIR_PATH",
        description="The folder of the free CorridorKey 2.0 (the one with infer.py). Empty: found in your usual folders",
        update=_engine_changed,
    )
    free_python: bpy.props.StringProperty(
        name="Python", subtype="FILE_PATH",
        description="The Python CorridorKey 2.0 runs with. Empty: the folder's venv",
        update=_engine_changed,
    )
    sam3_root: bpy.props.StringProperty(
        name="SAM 3 Folder", subtype="DIR_PATH",
        description="Where SAM 3 is installed (its venv and weights). Empty: CorridorKey2.0-SAM3 in your home folder",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "engine")
        layout.label(text=f"In use: {engine_label()}", icon="CHECKMARK" if engine_available() else "ERROR")
        box = layout.box()
        box.label(text="Free CorridorKey 2.0 (GitHub)", icon="URL")
        box.prop(self, "free_root")
        found = free_root()
        if not self.free_root and found:
            box.label(text=f"Found: {found}", icon="FILE_FOLDER")
        box.prop(self, "free_python")
        problem = free_setup_problem()
        box.label(text=problem or "Ready: installed, with its weights.", icon="ERROR" if problem else "CHECKMARK")
        row = box.row(align=True)
        row.operator(CORRIDORKEY_SHADER_OT_free_install.bl_idname, icon="IMPORT")
        row.operator(CORRIDORKEY_SHADER_OT_free_check.bl_idname, icon="VIEWZOOM")
        if FREE_CHECK["line"]:
            for part in textwrap.wrap(FREE_CHECK["line"], 70):
                box.label(text=part)
        box.operator("wm.url_open", text="CorridorKey 2.0 on GitHub", icon="URL").url = FREE_REPO
        box.label(text="Needs an NVIDIA RTX 30-series or newer, Git and Python 3 (for Install).", icon="INFO")

        box = layout.box()
        box.label(text="Alpha Hint Maker: SAM 3 (Meta, SAM License)", icon="MOD_MASK")
        box.prop(self, "sam3_root")
        problem = sam3_problem()
        box.label(text=problem or "Ready: SAM 3 and its weights are installed.", icon="ERROR" if problem else "CHECKMARK")
        row = box.row(align=True)
        row.operator(CORRIDORKEY_SHADER_OT_sam3_install.bl_idname, icon="IMPORT")
        row.operator(CORRIDORKEY_SHADER_OT_sam3_check.bl_idname, icon="VIEWZOOM")
        if SAM3_CHECK["line"]:
            for part in textwrap.wrap(SAM3_CHECK["line"], 70):
                box.label(text=part)
        row = box.row(align=True)
        row.operator("wm.url_open", text="SAM 3 on GitHub", icon="URL").url = SAM3_REPO
        row.operator("wm.url_open", text="Weights mirror", icon="URL").url = f"https://huggingface.co/{SAM3_MIRROR}"
        box.label(text="Weights: the ungated mirror of facebook/sam3, with Meta's licence beside them.", icon="INFO")


# --- registration ----------------------------------------------------------------------------------

_CLASSES = (
    CorridorKeyShaderSettings,
    CorridorKeyFreeSettings,
    CorridorKey2Preferences,
    CORRIDORKEY_SHADER_OT_free_render,
    CORRIDORKEY_SHADER_OT_free_cancel,
    CORRIDORKEY_SHADER_OT_free_open_folder,
    CORRIDORKEY_SHADER_OT_free_check,
    CORRIDORKEY_SHADER_OT_free_install,
    CORRIDORKEY_SHADER_OT_open_prefs,
    CORRIDORKEY_SHADER_OT_render_range,
    CORRIDORKEY_SHADER_OT_range_from_timeline,
    CORRIDORKEY_SHADER_OT_make_hint,
    CORRIDORKEY_SHADER_OT_cancel_hint,
    CORRIDORKEY_SHADER_OT_screen_hint,
    CORRIDORKEY_SHADER_OT_screen_preset,
    CORRIDORKEY_SHADER_OT_pick_render_folder,
    CORRIDORKEY_SHADER_OT_floor_add,
    CORRIDORKEY_SHADER_OT_floor_fit,
    CORRIDORKEY_SHADER_OT_floor_sample,
    CORRIDORKEY_SHADER_OT_floor_matte,
    CORRIDORKEY_SHADER_OT_floor_remove,
    CORRIDORKEY_SHADER_OT_floor_bake,
    CORRIDORKEY_SHADER_OT_floor_bake_cancel,
    CORRIDORKEY_SHADER_OT_floor_unbake,
    CORRIDORKEY_SHADER_OT_cancel_screen_hint,
    CORRIDORKEY_SHADER_OT_sam3_check,
    CORRIDORKEY_SHADER_OT_sam3_install,
    CORRIDORKEY_SHADER_OT_setup_plane,
    CORRIDORKEY_SHADER_OT_open_guide,
    CORRIDORKEY_SHADER_OT_lock,
    CORRIDORKEY_SHADER_OT_unlock,
    CORRIDORKEY_SHADER_OT_compositing_off,
    CORRIDORKEY_SHADER_OT_add_node,
    CORRIDORKEY_SHADER_OT_repair,
    CORRIDORKEY_SHADER_OT_connect,
    CORRIDORKEY_SHADER_OT_preview,
    CORRIDORKEY_SHADER_OT_preview_range,
    CORRIDORKEY_SHADER_OT_clear_previews,
    CORRIDORKEY_SHADER_OT_cancel_preview,
    CORRIDORKEY_SHADER_OT_render_image,
    CORRIDORKEY_SHADER_OT_delete_render,
    CORRIDORKEY_SHADER_OT_reset,
    CORRIDORKEY_SHADER_PT_node,
    CORRIDORKEY_SHADER_PT_viewport,
)
_MENU = "NODE_MT_category_shader_texture"  # Add > Texture, beside Image Texture
@persistent
def _render_folder_on_save(*args) -> None:
    """A node made before the file was first saved gets its Render Folder beside the .blend
    (set as the save starts, so it's saved with the file)."""
    tree = bpy.data.node_groups.get(SOURCES)
    target = args[0] if args and isinstance(args[0], str) else bpy.data.filepath
    if tree is None or not target:
        return
    for src in [n for n in tree.nodes if is_source(n)]:
        if not st(src).render_folder:
            st(src).render_folder = "//CorridorKey/"


_HANDLERS = (
    ("load_post", _after_load),
    ("save_pre", _render_folder_on_save),
    ("save_pre", _show_no_previews),
    ("save_pre", _tidy_before_save),
    ("render_init", _show_no_previews),
    ("frame_change_post", _on_frame),
    ("render_complete", _render_ended),
    ("render_cancel", _render_ended),
)
_KEYMAPS: list = []  # F12, taken over while the add-on is on
_DRAW: list = []  # the timeline's draw handler


def _add_to_menu(self, context) -> None:
    self.layout.operator(CORRIDORKEY_SHADER_OT_add_node.bl_idname, text=NAME)


def _add_to_render_menu(self, context) -> None:
    self.layout.operator(CORRIDORKEY_SHADER_OT_render_image.bl_idname, icon="RENDER_STILL")
    if any(True for _n in shader_nodes()):
        self.layout.operator(CORRIDORKEY_SHADER_OT_lock.bl_idname, text="Lock CorridorKey for Render Farm",
                             icon="LOCKED").every = True
    if locked_nodes():
        self.layout.operator(CORRIDORKEY_SHADER_OT_unlock.bl_idname, text="Unlock CorridorKey",
                             icon="UNLOCKED").every = True


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.ShaderNodeGroup.corridorkey_shader = bpy.props.PointerProperty(
        type=CorridorKeyShaderSettings
    )
    bpy.types.CompositorNodeGroup.corridorkey2 = bpy.props.PointerProperty(type=CorridorKeyFreeSettings)
    bpy.types.Scene.corridorkey_key_before_render = bpy.props.BoolProperty(
        name="Key Frame Before F12 Render",
        description=(
            "When you render an image (F12), key the current frame with CorridorKey first, "
            "or use its cached preview, so the render has a fresh key"
        ),
        default=True,
    )
    menu = getattr(bpy.types, _MENU, None)
    if menu is not None:
        menu.append(_add_to_menu)
    bpy.types.TOPBAR_MT_render.prepend(_add_to_render_menu)
    bpy.types.VIEW3D_MT_object_context_menu.append(_context_menu)
    keyconfig = bpy.context.window_manager.keyconfigs.addon
    if keyconfig is not None:
        keymap = keyconfig.keymaps.new(name="Screen", space_type="EMPTY")
        item = keymap.keymap_items.new(CORRIDORKEY_SHADER_OT_render_image.bl_idname, "F12", "PRESS")
        _KEYMAPS.append((keymap, item))
    for name, fn in _HANDLERS:
        getattr(bpy.app.handlers, name).append(fn)
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=1.0, persistent=True)
    _DRAW.append(
        bpy.types.SpaceDopeSheetEditor.draw_handler_add(draw_timeline, (), "WINDOW", "POST_PIXEL")
    )


def unregister() -> None:
    for uid in list(HINT_JOBS):
        cancel_hint(uid)
    for entry in FREE_RENDERS.values():
        entry["ticket"].cancel()
    FREE_RENDERS.clear()
    close_warm()
    for handle in _DRAW:
        bpy.types.SpaceDopeSheetEditor.draw_handler_remove(handle, "WINDOW")
    _DRAW.clear()
    if bpy.app.timers.is_registered(tick):
        bpy.app.timers.unregister(tick)
    for uid in list(PREVIEW_JOBS):
        cancel_preview(uid)
    for name, fn in _HANDLERS:
        handlers = getattr(bpy.app.handlers, name)
        if fn in handlers:
            handlers.remove(fn)
    for keymap, item in _KEYMAPS:
        keymap.keymap_items.remove(item)
    _KEYMAPS.clear()
    bpy.types.TOPBAR_MT_render.remove(_add_to_render_menu)
    bpy.types.VIEW3D_MT_object_context_menu.remove(_context_menu)
    menu = getattr(bpy.types, _MENU, None)
    if menu is not None:
        menu.remove(_add_to_menu)
    del bpy.types.Scene.corridorkey_key_before_render
    del bpy.types.CompositorNodeGroup.corridorkey2
    del bpy.types.ShaderNodeGroup.corridorkey_shader
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
