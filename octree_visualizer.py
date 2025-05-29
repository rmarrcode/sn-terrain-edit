import argparse
import struct
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

##########################################
#  DATA CLASSES (ported from Unity C#)   #
##########################################

@dataclass
class VoxelData:
    blocktype: int
    signed_distance: float

    @property
    def solid(self) -> bool:
        """A voxel is considered solid when the signed distance is positive (same heuristic Unity uses)."""
        return self.signed_distance > 0


@dataclass
class OctNodeData:
    """Exact binary layout of one entry in an optoctree file."""
    type: int
    density: int  # stored as signed byte in file (-128 … 127)
    child_position: int  # 16-bit unsigned offset to first child

    @classmethod
    def from_bytes(cls, data: bytes, offset: int) -> Tuple["OctNodeData", int]:
        # layout: [type:1][density:1][childLow:1][childHigh:1]
        type_val = data[offset]
        density_val = struct.unpack("b", data[offset + 1 : offset + 2])[0]
        child_pos = struct.unpack("<H", data[offset + 2 : offset + 4])[0]
        return cls(type_val, density_val, child_pos), offset + 4


################################
#  OCTREE REPRESENTATION       #
################################

CORNER_OFFSETS = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
]

class OctNode:
    """Port of ReefEditor.VoxelEditing.OctNode (subset that is relevant for mesh generation)."""

    __slots__ = (
        "position",
        "size",
        "voxel_data",
        "vertex_index",
        "corners_solid_info",
        "has_children",
        "children",
    )

    def __init__(self, position: Tuple[int, int, int], size: int):
        self.position: Tuple[int, int, int] = position
        self.size: int = size
        self.voxel_data = VoxelData(0, -2)  # default empty voxel
        self.vertex_index: int = -1
        self.corners_solid_info: List[bool] = [False] * 8
        self.has_children: bool = False
        self.children: List[Optional["OctNode"]] = [None] * 8

    # ---------- reading ----------
    def _decode_density(self, voxel_type: int, density_byte: int) -> float:
        if density_byte == 0:
            return 2.0 if voxel_type > 0 else -2.0
        return (density_byte - 126) / 126.0

    def read_array(self, data_array: List[OctNodeData], my_pos: int):
        node_data = data_array[my_pos]
        self.voxel_data = VoxelData(node_data.type, self._decode_density(node_data.type, node_data.density))

        if node_data.child_position > 0:
            self.has_children = True
            for i in range(8):
                child_position = tuple(
                    self.position[j] + CORNER_OFFSETS[i][j] * self.size // 2 for j in range(3)
                )
                child = OctNode(child_position, self.size // 2)
                child.read_array(data_array, node_data.child_position + i)
                self.children[i] = child

    # ---------- mesh helpers ----------
    def fill_vertex_list(self, verts: List[Tuple[float, float, float]]):
        if self.has_children:
            for c in self.children:
                if c is not None:
                    c.fill_vertex_list(verts)
        else:
            self.vertex_index = len(verts)
            half = 0.5 * self.size
            verts.append((self.position[0] + half, self.position[1] + half, self.position[2] + half))

    def update_edges_solidity(self, metaspace: "VoxelMetaspace"):
        for i in range(8):
            # convert local offset to world voxel position
            vx = self.position[0] + CORNER_OFFSETS[i][0] * self.size
            vy = self.position[1] + CORNER_OFFSETS[i][1] * self.size
            vz = self.position[2] + CORNER_OFFSETS[i][2] * self.size
            voxel = metaspace.get_voxel((vx, vy, vz))
            self.corners_solid_info[i] = voxel.solid if voxel else False

        if self.has_children:
            for c in self.children:
                if c is not None:
                    c.update_edges_solidity(metaspace)

######################################
#   METASPACE – container of octrees #
######################################

class VoxelMetaspace:
    """A minimal counterpart to Unity's VoxelMetaspace used only for queries during mesh building."""

    def __init__(self, root_nodes: List[OctNode]):
        self.root_nodes = root_nodes
        # index root nodes by their global tree coordinate (xTree, yTree, zTree)
        self._tree_size = 32  # each root represents a 32³ region
        self._lookup: Dict[Tuple[int, int, int], OctNode] = {}
        for node in root_nodes:
            key = (node.position[0] // self._tree_size,
                   node.position[1] // self._tree_size,
                   node.position[2] // self._tree_size)
            self._lookup[key] = node

    def _find_root(self, voxel: Tuple[int, int, int]) -> Optional[OctNode]:
        key = (
            voxel[0] // self._tree_size,
            voxel[1] // self._tree_size,
            voxel[2] // self._tree_size,
        )
        return self._lookup.get(key)

    def get_voxel(self, voxel: Tuple[int, int, int]) -> Optional[VoxelData]:
        root = self._find_root(voxel)
        if not root:
            return None
        # Recursively descend – max depth 5 as in Unity
        return self._get_voxel_rec(root, voxel, height=5)

    def _get_voxel_rec(self, node: OctNode, voxel: Tuple[int, int, int], height: int) -> Optional[VoxelData]:
        if node.has_children and height > 0:
            for child in node.children:
                if child and _voxel_in_node(voxel, child):
                    return self._get_voxel_rec(child, voxel, height - 1)
        return node.voxel_data


def _voxel_in_node(voxel: Tuple[int, int, int], node: OctNode) -> bool:
    x, y, z = voxel
    px, py, pz = node.position
    s = node.size
    return px <= x < px + s and py <= y < py + s and pz <= z < pz + s

########################################
#  MESH-BUILDER (Dual Contouring port) #
########################################

# --- Constant masks copied verbatim from MeshBuilderV2.cs ---
PROCESS_EDGE_MASK = [
    [3, 2, 1, 0],  # right
    [7, 5, 6, 4],  # up
    [11, 10, 9, 8],  # forward
]

EDGE_VMAP = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],  # x-axis
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],  # y-axis
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],  # z-axis
]

EDGE_PROC_EDGE_MASK = [
    [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
    [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
    [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]],
]

FACE_PROC_FACE_MASK = [
    [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
    [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
    [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]],
]

FACE_PROC_EDGE_MASK = [
    [[1, 4, 0, 5, 1, 1], [1, 6, 2, 7, 3, 1], [0, 4, 6, 0, 2, 2], [0, 5, 7, 1, 3, 2]],
    [[0, 2, 3, 0, 1, 0], [0, 6, 7, 4, 5, 0], [1, 2, 0, 6, 4, 2], [1, 3, 1, 7, 5, 2]],
    [[1, 1, 0, 3, 2, 0], [1, 5, 4, 7, 6, 0], [0, 1, 5, 0, 4, 1], [0, 3, 7, 2, 6, 1]],
]

CELL_PROC_FACE_MASK = [
    [0, 4, 0],
    [1, 5, 0],
    [2, 6, 0],
    [3, 7, 0],
    [0, 2, 1],
    [4, 6, 1],
    [1, 3, 1],
    [5, 7, 1],
    [0, 1, 2],
    [2, 3, 2],
    [4, 5, 2],
    [6, 7, 2],
]

CELL_PROC_EDGE_MASK = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2],
]

# --- MeshBuilder functions ---

def generate_mesh(root: OctNode, metaspace: VoxelMetaspace) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
    verts: List[Tuple[float, float, float]] = []
    root.fill_vertex_list(verts)
    root.update_edges_solidity(metaspace)

    indices: List[int] = []
    _cell_proc(indices, root)

    # convert pairwise indices to triangles
    tris = [(indices[i], indices[i + 1], indices[i + 2]) for i in range(0, len(indices), 3)]
    return verts, tris


def _cell_proc(index_buffer: List[int], node: OctNode):
    if node.has_children:
        for child in node.children:
            if child is not None:
                _cell_proc(index_buffer, child)

        # faces
        for mask in CELL_PROC_FACE_MASK:
            a, b, dir_ = mask
            face_nodes = [node.children[a], node.children[b]]
            _face_proc(index_buffer, face_nodes, dir_)

        # edges
        for mask in CELL_PROC_EDGE_MASK:
            n0, n1, n2, n3, dir_ = mask
            edge_nodes = [node.children[n0], node.children[n1], node.children[n2], node.children[n3]]
            _edge_proc(index_buffer, edge_nodes, dir_)


def _face_proc(index_buffer: List[int], nodes: List[OctNode], dir_: int):
    if nodes[0].has_children or nodes[1].has_children:
        # recurse
        for i in range(4):
            c0, c1 = FACE_PROC_FACE_MASK[dir_][i][0:2]
            child_nodes = [
                nodes[0].children[c0] if nodes[0].has_children else nodes[0],
                nodes[1].children[c1] if nodes[1].has_children else nodes[1],
            ]
            _face_proc(index_buffer, child_nodes, FACE_PROC_FACE_MASK[dir_][i][2])

        # edges inside face
        for i in range(4):
            row = FACE_PROC_EDGE_MASK[dir_][i]
            orders = [[0, 0, 1, 1], [0, 1, 0, 1]]
            order = orders[row[0]]
            edge_nodes = []
            for j in range(4):
                src = order[j]
                child_idx = row[j + 1]
                edge_nodes.append(
                    nodes[src].children[child_idx] if nodes[src].has_children else nodes[src]
                )
            _edge_proc(index_buffer, edge_nodes, row[5])
    # else nothing – face between two leaf nodes handled by edge proc


def _edge_proc(index_buffer: List[int], nodes: List[OctNode], dir_: int):
    if all(not n.has_children for n in nodes):
        _make_edge(index_buffer, nodes, dir_)
    else:
        for i in range(2):
            row = EDGE_PROC_EDGE_MASK[dir_][i]
            edge_nodes = []
            for j in range(4):
                src_node = nodes[j]
                child_idx = row[j]
                edge_nodes.append(src_node.children[child_idx] if src_node.has_children else src_node)
            _edge_proc(index_buffer, edge_nodes, row[4])


def _make_edge(index_buffer: List[int], nodes: List[OctNode], dir_: int):
    min_size = 1e9
    min_index = 0
    flip = False
    indices = [-1, -1, -1, -1]
    sign_change = [False] * 4

    for i in range(4):
        edge = PROCESS_EDGE_MASK[dir_][i]
        c1, c2 = EDGE_VMAP[edge]
        solid1 = nodes[i].corners_solid_info[c1]
        solid2 = nodes[i].corners_solid_info[c2]

        if nodes[i].size < min_size:
            min_size = nodes[i].size
            min_index = i
            flip = solid1

        indices[i] = nodes[i].vertex_index
        sign_change[i] = (solid1 and not solid2) or (not solid1 and solid2)

    if sign_change[min_index]:
        if not flip:
            index_buffer.extend([indices[0], indices[1], indices[3], indices[0], indices[3], indices[2]])
        else:
            index_buffer.extend([indices[0], indices[3], indices[1], indices[0], indices[2], indices[3]])

########################################
#  BATCH FILE READING                  #
########################################

def load_batch(file_path: str, batch_index: Tuple[int, int, int]) -> List[OctNode]:
    """Faithful port of BatchReadWriter.ReadBatch"""
    octrees: List[OctNode] = []
    with open(file_path, "rb") as f:
        _ = f.read(4)  # skip int header
        data = f.read()

    # determine expected counts
    dims = [5, 5, 5]
    if batch_index[0] == 25:
        dims[0] = 3
    if batch_index[2] == 25:
        dims[2] = 3
    expected_octrees = dims[0] * dims[1] * dims[2]

    curr = 0
    count = 0
    while curr < len(data) and count < expected_octrees:
        node_count = data[curr] + 256 * data[curr + 1]  # ushort little-endian
        curr += 2
        nodes_array: List[OctNodeData] = []
        for i in range(node_count):
            od, _ = OctNodeData.from_bytes(data, curr + i * 4)
            nodes_array.append(od)
        curr += node_count * 4

        x = count // (dims[2] * dims[1])
        y = (count % (dims[2] * dims[1])) // dims[2]
        z = count % dims[2]
        pos = (
            batch_index[0] * 160 + x * 32,
            batch_index[1] * 160 + y * 32,
            batch_index[2] * 160 + z * 32,
        )
        root = OctNode(pos, 32)
        root.read_array(nodes_array, 0)
        octrees.append(root)
        count += 1

    return octrees

########################################
#  VISUALIZATION                       #
########################################

def visualize(octrees: List[OctNode], elev: float = 30, azim: float = 45):
    metaspace = VoxelMetaspace(octrees)
    all_verts: List[Tuple[float, float, float]] = []
    all_tris: List[Tuple[int, int, int]] = []
    vert_offset = 0

    for tree in octrees:
        verts, tris = generate_mesh(tree, metaspace)
        all_verts.extend(verts)
        all_tris.extend([(a + vert_offset, b + vert_offset, c + vert_offset) for a, b, c in tris])
        vert_offset += len(verts)

    if not all_tris:
        print("No triangles generated – maybe the batch is empty or all voxels are uniform.")
        return

    # Swap Y and Z so that the Z axis points up (Unity style Z-up display)
    verts_np = np.array(all_verts)
    verts_plot = verts_np[:, [0, 2, 1]]  # (x,z,y)
    tris_np = np.array(all_tris)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts_plot[tris_np], alpha=0.6)
    mesh.set_facecolor("#6495ED")
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)

    # scale axes
    xyz_min = verts_plot.min(axis=0)
    xyz_max = verts_plot.max(axis=0)
    ax.set_xlim(xyz_min[0], xyz_max[0])
    ax.set_ylim(xyz_min[1], xyz_max[1])  # depth axis (former Y)
    ax.set_zlim(xyz_min[2], xyz_max[2])  # up axis (former Y -> now Z)
    ax.set_box_aspect((xyz_max - xyz_min))
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")
    plt.title(f"Octree batch mesh (dual contouring) - elev={elev}°, azim={azim}°")
    plt.show()

########################################
#  ENTRY POINT                         #
########################################

# Hard-coded path to your Subnautica CompiledOctreesCache directory
DEFAULT_CACHE_DIR = (
    "/Users/ryanmarr/Library/Application Support/Steam/steamapps/common/"
    "Subnautica/Subnautica.app/Contents/Resources/Data/StreamingAssets/"
    "SNUnmanagedData/Build18/CompiledOctreesCache"
)

def main():
    p = argparse.ArgumentParser(
        description="Visualize Subnautica .optoctrees batches like the Unity editor (path is hard-coded)."
    )
    p.add_argument("bx", type=int, help="Batch X index")
    p.add_argument("by", type=int, help="Batch Y index")
    p.add_argument("bz", type=int, help="Batch Z index")
    p.add_argument("--elev", type=float, default=30, help="Elevation angle (default: 30)")
    p.add_argument("--azim", type=float, default=45, help="Azimuth angle (default: 45)")
    args = p.parse_args()

    batch_idx = (args.bx, args.by, args.bz)
    batch_file = f"{DEFAULT_CACHE_DIR}/compiled-batch-{batch_idx[0]}-{batch_idx[1]}-{batch_idx[2]}.optoctrees"

    print(f"Loading batch file: {batch_file}")
    trees = load_batch(batch_file, batch_idx)
    print(f"Loaded {len(trees)} octrees from batch {batch_idx}.")
    visualize(trees, elev=args.elev, azim=args.azim)

if __name__ == "__main__":
    main() 