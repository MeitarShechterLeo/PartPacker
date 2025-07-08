"""
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
"""

import sys

sys.path.append(".")

import argparse
import glob
import os
import pickle
import subprocess
import tempfile
import json

import numpy as np
import tqdm
import trimesh
from meshiki import Mesh


def safe_collision_detection_subprocess(manager_data, return_names=True, return_data=False, timeout_seconds=23):
    """
    Safely perform collision detection using subprocess for maximum isolation.
    
    Args:
        manager_data: dict of mesh objects
        return_names: whether to return collision pair names
        return_data: whether to return collision data
        timeout_seconds: maximum time to wait for collision detection
    
    Returns:
        tuple: (is_collide, collide_pairs, collide_data) or (is_collide, collide_pairs) depending on return_data
    """
    # Create temporary files for data exchange
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        
        input_path = input_file.name
        output_path = output_file.name
    
    try:
        # Prepare data for subprocess
        # Convert mesh objects to serializable format (vertices and faces)
        serializable_data = {}
        for name, mesh in manager_data.items():
            serializable_data[name] = {
                'vertices': mesh.vertices.tolist(),
                'faces': mesh.faces.tolist()
            }
        
        # Write input data
        with open(input_path, 'w') as f:
            json.dump({
                'manager_data': serializable_data,
                'return_names': return_names,
                'return_data': return_data
            }, f)
        
        # Create the subprocess script
        subprocess_script = f'''
import sys
import json
import numpy as np
import trimesh

def collision_worker_subprocess(manager_data, return_names=True, return_data=False, timeout_seconds=20):
    try:
        # Reconstruct the collision manager
        manager = trimesh.collision.CollisionManager()
        
        # Reconstruct mesh objects from serialized data
        for name, mesh_data in manager_data.items():
            vertices = np.array(mesh_data['vertices'])
            faces = np.array(mesh_data['faces'])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            manager.add_object(name, mesh)
        
        # Perform collision detection
        result = manager.in_collision_internal(return_names=return_names, return_data=return_data)
        
        # Convert result to serializable format
        if isinstance(result, tuple):
            if len(result) == 3:
                is_collide, collide_pairs, collide_data = result
                collide_pairs = list(collide_pairs) if collide_pairs else []
                collide_data_serializable = []
                for data in collide_data:
                    collide_data_serializable.append({{
                        'names': list(data.names),
                        'depth': float(data.depth) if hasattr(data, 'depth') else 0.0
                    }})
                return {{'success': True, 'result': [is_collide, collide_pairs, collide_data_serializable]}}
            elif len(result) == 2:
                is_collide, collide_pairs = result
                collide_pairs = list(collide_pairs) if collide_pairs else []
                return {{'success': True, 'result': [is_collide, collide_pairs, []]}}
            else:
                return {{'success': True, 'result': [False, [], []]}}
        else:
            return {{'success': True, 'result': [False, [], []]}}
            
    except Exception as e:
        return {{'success': False, 'error': str(e)}}

if __name__ == "__main__":
    # Read input data
    with open("{input_path}", 'r') as f:
        data = json.load(f)
    
    # Run collision detection
    result = collision_worker_subprocess(
        data['manager_data'], 
        data['return_names'], 
        data['return_data']
    )
    
    # Write output data
    with open("{output_path}", 'w') as f:
        json.dump(result, f)
'''
        
        # Write subprocess script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_path = script_file.name
            script_file.write(subprocess_script)
        
        try:
            # Run subprocess
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise Exception("Subprocess timed out")
            
            # Check if subprocess failed
            if process.returncode != 0:
                raise Exception(f"Subprocess failed with return code {process.returncode}: {stderr.decode()}")
            
            # Read result
            with open(output_path, 'r') as f:
                result_data = json.load(f)
            
            if not result_data.get('success', False):
                raise Exception(f"Collision detection failed: {result_data.get('error', 'Unknown error')}")
            
            # Convert result back to expected format
            result = result_data['result']
            if len(result) == 3:
                is_collide, collide_pairs, collide_data_serializable = result
                # Convert lists back to sets
                collide_pairs = set(collide_pairs) if collide_pairs else set()
                # Reconstruct collision data objects if needed
                collide_data = []
                if return_data and collide_data_serializable:
                    # Create simple collision data objects
                    for data in collide_data_serializable:
                        class SimpleCollisionData:
                            def __init__(self, names, depth):
                                self.names = set(names)
                                self.depth = depth
                        collide_data.append(SimpleCollisionData(data['names'], data['depth']))
                
                return is_collide, collide_pairs, collide_data
            else:
                return False, set(), []
                
        finally:
            # Clean up script file
            try:
                os.unlink(script_path)
            except:
                pass
                
    finally:
        # Clean up temporary files
        try:
            os.unlink(input_path)
            os.unlink(output_path)
        except:
            pass


def safe_collision_detection(manager_data, return_names=True, return_data=False, timeout_seconds=23):
    """
    Safely perform collision detection using subprocess for maximum isolation.
    This works in all environments including daemon processes.
    
    Args:
        manager_data: dict of mesh objects
        return_names: whether to return collision pair names
        return_data: whether to return collision data
        timeout_seconds: maximum time to wait for collision detection
    
    Returns:
        tuple: (is_collide, collide_pairs, collide_data) or (is_collide, collide_pairs) depending on return_data
    """
    return safe_collision_detection_subprocess(manager_data, return_names, return_data, timeout_seconds)


class NamedDisjointSet:
    def __init__(self, names):
        # names: list of str
        self.parent = {name: name for name in names}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x, y):
        self.parent[self.find(x)] = self.find(y)

    def get_groups(self):
        groups = {}
        for name in self.parent:
            root = self.find(name)
            if root not in groups:
                groups[root] = []
            groups[root].append(name)
        return groups


def is_single_layer_plane(mesh: trimesh.Trimesh, coplane_thresh: float = 1):
    # check if a mesh is just a single-layer plane
    if mesh.is_watertight:
        return False
    face_normals = mesh.face_normals
    diff = np.linalg.norm(np.abs(face_normals) - np.abs(face_normals[0]))
    return diff.max() < coplane_thresh


def calc_intersection_union(bounds1, bounds2):
    # bounds: [2, 3]
    bmin1, bmax1 = bounds1[0], bounds1[1]
    bmin2, bmax2 = bounds2[0], bounds2[1]
    # intersection
    bmin = np.maximum(bmin1, bmin2)
    bmax = np.minimum(bmax1, bmax2)
    if np.any(bmin >= bmax):
        intersection = 0
    else:
        intersection = np.prod(bmax - bmin)
    # union
    vol1 = np.prod(bmax1 - bmin1)
    vol2 = np.prod(bmax2 - bmin2)
    union = vol1 + vol2 - intersection
    return intersection, union


def is_coplanar_and_convex(vertices, coplanar_thresh: float = 1):
    # vertices: [N, 3], assume ordered to form a coplanar polygon
    # note the last vertex is the same as the first vertex, i.e. ABCA
    # we are actually not requiring perfect coplanarity, the thresh of 1 means 60 degree tolerance...

    # Need at least 3 vertices to form a polygon
    if len(vertices) < 3:
        return False

    # Convert to numpy array if not already
    points = np.array(vertices)
    n_points = len(points)

    # Normal of the first triangle
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Test if coplanar using the normal of fan-cut triangles
    for i in range(2, n_points - 2):
        v1 = points[i] - points[0]
        v2 = points[i + 1] - points[0]
        normal_cur = np.cross(v1, v2)
        normal_cur = normal_cur / (np.linalg.norm(normal_cur) + 1e-12)
        diff = np.linalg.norm(np.abs(normal_cur) - np.abs(normal))
        if diff > coplanar_thresh:
            # print(f'not coplanar: {normal_cur} != {normal} (diff = {diff:.4f}) at {i}-{i+1}')
            return False  # not coplanar

    # Find basis vectors for the 2D plane
    # First basis vector can be the normalized vector from points[0] to points[1]
    basis1 = v1 / (np.linalg.norm(v1) + 1e-12)
    # Second basis vector is perpendicular to both normal and basis1
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / (np.linalg.norm(basis2) + 1e-12)

    # Project all points onto the 2D plane
    points_2d = np.zeros((n_points, 2))
    for i in range(n_points):
        v = points[i] - points[0]
        points_2d[i, 0] = np.dot(v, basis1)
        points_2d[i, 1] = np.dot(v, basis2)

    # Check if polygon is convex by using the cross product
    # For a convex polygon, all cross products should have the same sign
    sign = 0
    for i in range(n_points - 1):
        j = (i + 1) % n_points
        k = (i + 2) % n_points

        # Vectors from point i to j and j to k
        v1 = points_2d[j] - points_2d[i]
        v2 = points_2d[k] - points_2d[j]

        # 2D cross product
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

        # Check for consistent sign of cross product
        if abs(cross_product) > 1e-2:  # Skip collinear points
            current_sign = np.sign(cross_product)
            if sign == 0:
                sign = current_sign
            elif sign != current_sign:
                # print(f'not convex: {i}-{j} and {j}-{k}, cross_product = {cross_product:.4f}')
                return False  # not convex

    return True


def stitch_nonwatertight_mesh(mesh: trimesh.Trimesh, eps: float = 1e-2):
    # mesh will be inplace modified
    # return a flag denoting if there are still open boundaries unfixed

    # manager = trimesh.collision.CollisionManager()
    # manager.add_object('main', mesh)

    # watertight mesh doesn't need to be stitched
    if mesh.is_watertight:
        return False

    # planar mesh cannot be stitched
    if is_single_layer_plane(mesh, eps):
        return True

    # the following is modified from trimesh.repair.stitch
    # fan_faces = trimesh.repair.stitch(mesh)

    nonwatertight = False

    from trimesh.path.exchange.misc import faces_to_path

    faces = np.arange(len(mesh.faces))

    # get a sequence of vertex indices representing the
    # boundary of the specified faces
    # will be referencing the same indexes of `mesh.vertices`
    boundaries = [
        e.points for e in faces_to_path(mesh, faces)["entities"] if len(e.points) > 3 and e.points[0] == e.points[-1]
    ]

    # get properties to avoid querying in loop
    vertices = mesh.vertices
    normals = mesh.face_normals

    # find which faces are associated with an edge
    edges_face = mesh.edges_face
    tree_edge = mesh.edges_sorted_tree

    # MODIFIED: if any two boundary edges share close vertices, we discard both since they may connect
    mask = np.ones(len(boundaries), dtype=bool)
    for i in range(len(boundaries)):
        for j in range(i + 1, len(boundaries)):
            verts_i = vertices[boundaries[i]]  # [N, 3]
            verts_j = vertices[boundaries[j]]  # [M, 3]
            # check pair-wise distance
            dists = np.linalg.norm(verts_i[:, None, :] - verts_j[None, :, :], axis=-1)  # [N, M]
            num_close = np.sum(dists < 1e-6)
            if num_close >= 4 or (num_close / verts_i.shape[0] >= 0.5) or (num_close / verts_j.shape[0] >= 0.5):
                mask[i] = False
                mask[j] = False
                # print(f'discarding boundary {i} and {j} because of close vertices')
    boundaries = [boundaries[i] for i in range(len(boundaries)) if mask[i]]

    # MODIFIED: we only keep coplanar & convex fans
    fans = []
    for vert_indices in boundaries:

        # the fan should be coplanar and convex
        verts = vertices[vert_indices]
        if not is_coplanar_and_convex(verts):
            nonwatertight = True
            continue

        fan = np.column_stack(
            (np.ones(len(vert_indices) - 3, dtype=int) * vert_indices[0], vert_indices[1:-2], vert_indices[2:-1])
        )  # [N, 3]

        fans.append(fan)

    # now we do a normal check against an adjacent face
    # to see if each region needs to be flipped
    for i, t in zip(range(len(fans)), fans):
        # get the edges from the original mesh
        # for the first `n` new triangles
        e = t[:10, 1:].copy()
        e.sort(axis=1)

        # find which indexes of `mesh.edges` these
        # new edges correspond with by finding edges
        # that exactly correspond with the tree
        query = tree_edge.query_ball_point(e, r=1e-10)
        if len(query) == 0:
            continue
        # stack all the indices that exist
        edge_index = np.concatenate(query)

        # get the normals from the original mesh
        original = normals[edges_face[edge_index]]

        # calculate the normals for a few new faces
        check, valid = trimesh.triangles.normals(vertices[t[:3]])
        if not valid.any():
            continue
        # take the first valid normal from our new faces
        check = check[0]

        # if our new faces are reversed from the original
        # Adjacent face flip them along their axis
        sign = np.dot(original, check)
        if sign.mean() < 0:
            fans[i] = np.fliplr(t)

    if len(fans) > 0:
        fans = np.vstack(fans)
        mesh.faces = np.concatenate([mesh.faces, fans])

    return nonwatertight


def smart_grouping(meshes: dict):
    # meshes: {name: trimesh.Trimesh, ...}

    # use collision manager to find all colliding pairs
    # manager = trimesh.collision.CollisionManager()
    # for name, mesh in meshes.items():
    #     manager.add_object(name, mesh)

    # is_collide, collide_pairs = manager.in_collision_internal(return_names=True)

    is_collide, collide_pairs, _ = safe_collision_detection(meshes, return_names=True, return_data=False)
    # print(f'[INFO] num_collide = {len(collide_pairs)}, {collide_pairs}')

    if not is_collide:
        return meshes

    # pre-calculate some stat for each mesh
    name_to_stat = {}
    total_volume = 0
    max_extent = 0
    num_submeshes = 0
    num_meshes = len(meshes)
    for name, mesh in meshes.items():
        name_to_stat[name] = {}
        submeshes = mesh.split()
        name_to_stat[name]["volume"] = []
        name_to_stat[name]["extent"] = []
        num_submeshes += len(submeshes)
        for submesh in submeshes:
            if submesh.is_watertight:
                name_to_stat[name]["volume"].append(submesh.volume)
                total_volume += name_to_stat[name]["volume"][-1]
            name_to_stat[name]["extent"].append(np.max(submesh.extents))
            max_extent = max(max_extent, name_to_stat[name]["extent"][-1])
        name_to_stat[name]["volume"] = (
            np.mean(name_to_stat[name]["volume"]) if len(name_to_stat[name]["volume"]) > 0 else np.inf
        )
        name_to_stat[name]["extent"] = (
            np.max(name_to_stat[name]["extent"]) if len(name_to_stat[name]["extent"]) > 0 else np.inf
        )

    # use a disjoint set to record grouping
    ds = NamedDisjointSet(list(meshes.keys()))

    # decide the merging thresh adaptively (based on the number of meshes, average volume and extent)
    # very empirical...
    if num_meshes <= 16:
        tol_volume = 0.05 * total_volume
        tol_extent = 0.05 * max_extent
    else:
        tol_volume = 0.1 * total_volume
        tol_extent = 0.1 * max_extent

    # loop each pair, determine if they should be grouped together
    for name1, name2 in collide_pairs:

        if ds.find(name1) == ds.find(name2):
            continue

        # single-layer plane should be merged
        if is_single_layer_plane(meshes[name1]) or is_single_layer_plane(meshes[name2]):
            # print(f'[INFO] merge {name1} and {name2} because of single-layer plane')
            ds.merge(name1, name2)
            continue

        # too small component
        if (name_to_stat[name1]["volume"] < tol_volume and name_to_stat[name1]["extent"] < tol_extent) or (
            name_to_stat[name2]["volume"] < tol_volume and name_to_stat[name2]["extent"] < tol_extent
        ):
            # print(f'[INFO] merge {name1} and {name2} because of small volume')
            ds.merge(name1, name2)
            continue

        # overlaps a lot should be merged (just use bounding box IoU)
        bounds1 = meshes[name1].bounds  # [2, 3]
        bounds2 = meshes[name2].bounds  # [2, 3]
        vol_intersect, vol_union = calc_intersection_union(bounds1, bounds2)
        vol_iou = vol_intersect / vol_union
        if vol_iou > 0.5:
            # print(f'[INFO] merge {name1} and {name2} because of large IoU')
            ds.merge(name1, name2)
            continue

    # merge groups
    groups = ds.get_groups()
    for group in groups.values():
        if len(group) <= 1:
            continue
        # print(f'[INFO] merge group: {group}')
        new_name = "_".join(group)
        new_mesh = trimesh.util.concatenate(list(meshes[name] for name in group))

        # merge close vertices, and clean up
        new_mesh.merge_vertices(merge_tex=True, merge_norm=True)
        new_mesh.update_faces(new_mesh.unique_faces() & new_mesh.nondegenerate_faces())
        new_mesh.fix_normals()

        meshes[new_name] = new_mesh
        for name in group:
            del meshes[name]

    # print(f"[INFO] after grouping, num_meshes = {len(meshes)}")
    return meshes


def merge_odd_loops(meshes: dict, graph: dict, penetration_depths: dict):
    # meshes: {name: trimesh.Trimesh, ...}
    # graph: {name: [neighbor, ...], ...}
    # penetration_depths: {(name1, name2): depth, ...}

    # find out odd loops in the graph, and try to merge vertex pair of largest penetration depth to make them even loops
    all_loops = []
    visited = set()

    def dfs(node, parent, start_node, path):
        visited.add(node)
        path.append(node)
        for neighbor in graph[node]:
            if neighbor == parent:
                continue
            if neighbor == start_node and len(path) >= 3:
                all_loops.append(path.copy())
            elif neighbor not in visited:
                dfs(neighbor, node, start_node, path)
        path.pop()
        visited.remove(node)

    for node in graph:
        dfs(node, None, node, [])
        visited.add(node)

    unique_loops = []
    loop_keys = set()
    for loop in all_loops:
        loop_key = frozenset(loop)
        if len(loop) > 2 and loop_key not in loop_keys:
            loop_keys.add(loop_key)
            unique_loops.append(loop)
            # print(f'[INFO] find loop: {loop}')

    # convert loops to list of edges
    loop_edges = []
    for loop in unique_loops:
        edges = set()
        for i in range(len(loop)):
            j = (i + 1) % len(loop)
            edges.add(tuple(sorted([loop[i], loop[j]])))
        loop_edges.append(edges)

    # merge vertex pair of largest penetration depth to make them even loops
    ds = NamedDisjointSet(list(meshes.keys()))

    while True:  # have to loop multiple times to make sure there is no odd loop...

        for i in range(len(loop_edges)):
            edges = loop_edges[i]
            if len(edges) % 2 == 0:  # even
                continue

            # find the largest penetration depth
            max_penetration_depth = 0
            max_edge = None
            for edge in edges:
                if penetration_depths[edge] > max_penetration_depth:
                    max_penetration_depth = penetration_depths[edge]
                    max_edge = edge

            # remove this edge from other loops if it exists (this will affect other loops, even make already even loops become odd)
            for j in range(len(loop_edges)):
                if max_edge in loop_edges[j]:
                    loop_edges[j].remove(max_edge)

            # merge the vertex pair of largest penetration depth
            # print(f'[INFO] merge {max_edge[0]} and {max_edge[1]}')
            ds.merge(max_edge[0], max_edge[1])

        has_odd_loop = False
        for i in range(len(loop_edges)):
            edges = loop_edges[i]
            if len(edges) % 2 == 1:
                has_odd_loop = True
                break

        if not has_odd_loop:
            break

    # merge groups
    graph_new = graph.copy()
    groups = ds.get_groups()
    for group in groups.values():
        if len(group) <= 1:
            continue
        # print(f'[INFO] merge group: {group}')
        new_name = "_".join(group)
        new_mesh = trimesh.util.concatenate(list(meshes[name] for name in group))
        meshes[new_name] = new_mesh
        graph_new[new_name] = set()
        for name in group:
            del meshes[name]
            for neighbor in graph[name]:
                if neighbor not in group:
                    graph_new[new_name].add(neighbor)
                    graph_new[neighbor].remove(name)
                    graph_new[neighbor].add(new_name)
            del graph_new[name]

    # print(f"[INFO] after merging odd loops, num_meshes = {len(meshes)}")
    return meshes, graph_new


def normalize_scene(scene):
    ### box normalize into [-1, 1]
    bounds = scene.bounds  # [2, 3]
    center = scene.centroid  # [3]
    # print(f'[INFO] center = {center}, bounds = {bounds}')
    scale = 0.95 * 1 / np.max(bounds[1] - bounds[0])
    transform_normalize = np.eye(4)
    transform_normalize[:3, 3] = -center
    transform_normalize[:3, :3] = np.diag(np.array([scale, scale, scale]))
    # print(transform_normalize)
    scene.apply_transform(transform_normalize)

    ### apply transform to vertices
    meshes = {}
    scene_graph = scene.graph.to_flattened()
    for k, v in scene_graph.items():
        name = v["geometry"]
        if name in scene.geometry and isinstance(scene.geometry[name], trimesh.Trimesh):
            transform = v["transform"]
            mesh: trimesh.Trimesh = scene.geometry[name].apply_transform(transform)
            # drop all textures since we only need geom for parted data
            mesh.visual = trimesh.visual.ColorVisuals()
            # clean up
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
            mesh.fix_normals()
            meshes[name] = mesh

    return meshes

def color_meshes(meshes, model_name, no_dilate=False, no_merge_odd_loops=False, verbose=False, dilate_size=2/512):
    # build an undirected collision graph
    # manager = trimesh.collision.CollisionManager()
    new_meshes = {}
    for name, mesh in meshes.items():
        # scale up the mesh a little bit to take count of the collision margin
        mesh_dilated = mesh.copy()
        if not no_dilate:
            center = mesh_dilated.centroid
            vertices = mesh_dilated.vertices - center
            max_radius = np.max(np.linalg.norm(vertices, axis=-1))
            scale = (max_radius + dilate_size) / max_radius
            # print(f'[INFO] dilate {name} by {scale}')
            mesh_dilated.vertices = vertices * scale + center
        # manager.add_object(name, mesh_dilated)
        new_meshes[name] = mesh_dilated

    # is_collide, collide_pairs, collide_data = manager.in_collision_internal(return_names=True, return_data=True)
    is_collide, collide_pairs, collide_data = safe_collision_detection(new_meshes, return_names=True, return_data=True)

    graph = {name: set() for name in meshes.keys()}
    penetration_depths = {}

    for data in collide_data:
        name1, name2 = list(data.names)
        graph[name1].add(name2)
        graph[name2].add(name1)
        name_key = tuple(sorted([name1, name2]))
        penetration_depth = data.depth
        penetration_depths[name_key] = penetration_depth

    # merge odd loops
    if not no_merge_odd_loops:
        # if the graph is too complex, we will skip since it takes forever
        num_edges = sum(len(edges) for edges in graph.values())
        if num_edges > 100:
            # print(f"[WARN] skip {model_name} because of too many edges: {num_edges}")
            pass
        else:
            meshes, graph = merge_odd_loops(meshes, graph, penetration_depths)

    if verbose:
        print(graph)

    # sort objects by distance to center
    name_to_centers = {}
    for name, mesh in meshes.items():
        vmin = np.min(mesh.vertices, axis=0)
        vmax = np.max(mesh.vertices, axis=0)
        name_to_centers[name] = (vmin + vmax) / 2

    name_with_dist = []  # [(name, dist), ...]
    for name, mesh_center in name_to_centers.items():
        dist = np.linalg.norm(mesh_center)
        name_with_dist.append((name, dist))
    name_with_dist.sort(key=lambda x: x[1])

    # we will start graph coloring from center to border
    name_to_color = {}
    queue = []
    initial_color = 0
    print_warn = False
    for name, dist in name_with_dist:
        if name not in name_to_color:
            name_to_color[name] = initial_color
            initial_color = 1 - initial_color
            queue.append(name)
            while len(queue) > 0:
                name = queue.pop(0)
                for neighbor in graph[name]:
                    if neighbor not in name_to_color:
                        name_to_color[neighbor] = 1 - name_to_color[name]
                        if verbose:
                            print(f"[INFO] color {neighbor} with {name_to_color[neighbor]}")
                        queue.append(neighbor)
                    else:
                        if name_to_color[neighbor] == name_to_color[name]:
                            if verbose: 
                                print(f"[WARN] {name} and {neighbor} have the same color!")
                            else:
                                print_warn = True

    # if not verbose and print_warn:
    #     print(f"[WARN] {model_name} has at least one pair of meshes with the same color!")

    # get the two parts
    mesh_color0 = []
    mesh_color1 = []
    for name, color in name_to_color.items():
        if color == 0:
            mesh_color0.append(meshes[name])
        else:
            mesh_color1.append(meshes[name])

    return mesh_color0, mesh_color1

def run(path):
    print(f"[INFO] processing {path}")

    mesh = trimesh.load(path)

    if not opt.force_cc and isinstance(mesh, trimesh.Scene) and len(mesh.geometry) > 1:
        print(f"[INFO] scene: {len(mesh.geometry)} meshes")
        scene = mesh

    else:
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        # use meshiki backend
        mesh = Mesh(mesh.vertices, mesh.faces, verbose=opt.verbose, clean=False)
        mesh.smart_group_components()
        scene = mesh.export_components_as_trimesh_scene()

        # use trimesh
        # meshes = mesh.split(only_watertight=False)
        # # print(meshes)
        # scene = trimesh.Scene()
        # for mesh in meshes:
        #     scene.add_geometry(mesh)
        print(f"[INFO] mesh: {len(scene.geometry)} components")

    meshes = normalize_scene(scene)

    ### smart grouping to avoid too many single-layer surface or too small objects
    if not opt.no_smart_group:
        meshes = smart_grouping(meshes)

    ### stitch open boundaries to make each mesh watertight
    if not opt.no_stitch:
        for name, mesh in meshes.items():
            stitch_nonwatertight_mesh(mesh)

    ### coloring
    mesh_color0, mesh_color1 = color_meshes(meshes, path, opt.no_dilate, opt.no_merge_odd_loops, opt.verbose, opt.dilate_size)

    ### convert to a single mesh and export as glb
    mesh_color0 = trimesh.util.concatenate(mesh_color0)
    mesh_color1 = trimesh.util.concatenate(mesh_color1)
    name = os.path.splitext(os.path.basename(path))[0]

    # export separately
    mesh_color0.export(f"{opt.workspace}/{name}_color0.obj")
    mesh_color1.export(f"{opt.workspace}/{name}_color1.obj")

    # export together (offsetted)
    mesh_color1.vertices += [0, 0, 1]
    mesh_all = trimesh.util.concatenate([mesh_color0, mesh_color1])
    mesh_all.export(f"{opt.workspace}/{name}.obj")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path", type=str, help="path to the mesh file or folder")
    parser.add_argument("--verbose", action="store_true", help="print verbose output")
    parser.add_argument("--force_cc", action="store_true", help="force to use connected components and ignore glb groups")
    parser.add_argument("--no_smart_group", action="store_true", help="do not perform smart grouping")
    parser.add_argument("--no_stitch", action="store_true", help="do not stitch open boundaries")
    parser.add_argument("--no_merge_odd_loops", action="store_true", help="do not merge odd loops")
    parser.add_argument("--no_dilate", action="store_true", help="do not dilate the mesh")
    parser.add_argument("--dilate_size", type=float, default=2 / 512, help="dilate size")
    parser.add_argument("--workspace", type=str, default="output", help="path to the output folder")
    opt = parser.parse_args()

    os.makedirs(opt.workspace, exist_ok=True)

    if os.path.isdir(opt.test_path):
        file_paths = glob.glob(os.path.join(opt.test_path, "*"))
        for path in tqdm.tqdm(file_paths):
            run(path)
    else:
        run(opt.test_path)
