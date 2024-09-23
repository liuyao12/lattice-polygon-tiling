import numpy as np
import matplotlib.pyplot as plt


named_tiles = {
    'turtle': {
        'boundary': [
            [3, 0], # [3, -1]
            [2, 1],
            [1, 2],
            [-1, 1],
            [-1, 2],
            [-2, 2],
            [-3, 0], # [-3, 1]
            [-2, -1],
            [-3, -2],
            [-3, -3],
            [-1, -2],
            [0, -3],
            [1, -2],
            [2, -2]
        ],
        'interior': [
            [2, 0],
            [2, -1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, 1],
            [0, 0],
            [0, -1],
            [0, -2],
            [-1, 0],
            [-1, -1],
            [-2, 1],
            [-2, 0],
            [-2, -2]
        ]
    },
    'hat': {
        'boundary': [
            [1, 0],
            [2, 0],
            [3, 1],
            [2, 2],
            [0, 1],
            [0, 2],
            [-1, 2],
            [-2, 2],
            [-3, 1],
            [-2, 0],
            [-3, -2],
            [-2, -2],
            [-2, -3],
            [0, -2]
        ],
        'interior': [
            [2, 1],
            [1, 1],
            [0, 0],
            [0, -1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [-1, -2],
            [-2, 1],
            [-2, -1]
        ]
    },
    'triskelion': {
        'boundary': [
            [1, 0],
            [2, 2],
            [1, 3],
            [0, 2],
            [0, 1],
            [-2, 0],
            [-3, -2],
            [-2, -2],
            [-1, -1],
            [0, -2],
            [2, -1],
            [2, 0]
        ],
        'interior': [
            [0, 0],
            [1, 1],
            [1, 2],
            [-1, 0],
            [-2, -1],
            [0, -1],
            [1, -1]
        ]
    },
    'pent': {
        'boundary': [
            [1, 0],
            [0, 0],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1]
        ],
        'interior': [
            [-0.5, -0.5]
        ]
    },
    'hex2': {
        'boundary': [
            [2, 0], 
            [2, 1], 
            [2, 2], 
            [1, 2], 
            [0, 2], 
            [-1, 1], 
            [-2, 0], 
            [-2, -1], 
            [-2, -2], 
            [-1, -2], 
            [0, -2], 
            [1, -1]
        ],
        'interior': [
            [1, 1], 
            [1, 0], 
            [0, -1], 
            [0, 1], 
            [0, 0], 
            [0, -1], 
            [-1, 0], 
            [-1, -1]
        ]
    },
    'star12': {
        'boundary': [
            [4, 0],
            [5, 1],
            [4, 2],
            [5, 4],
            [4, 4],
            [4, 5],
            [2, 4],
            [1, 5],
            [0, 4],
            [-1, 4],
            [-2, 2],
            [-4, 1],
            [-4, 0],
            [-5, -1],
            [-4, -2],
            [-5, -4],
            [-4, -4],
            [-4, -5],
            [-2, -4],
            [-1, -5],
            [0, -4],
            [1, -4],
            [2, -2],
            [4, -1]
        ],
        'interior': [
            [4, 1],
            [4, 3],
            [3, -1],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [2, -1],
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [1, -3],
            [1, -2],
            [1, -1],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4],
            [0, -3],
            [0, -2],
            [0, -1],
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [-1, 3],
            [-1, 2],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [-1, -2],
            [-1, -3],
            [-1, -4],
            [-2, 1],
            [-2, 0],
            [-2, -1],
            [-2, -2],
            [-2, -3],
            [-3, 1],
            [-3, 0],
            [-3, -1],
            [-3, -2],
            [-3, -3],
            [-3, -4],
            [-4, -1],
            [-4, -3]
        ]
    },
    'simplex': {
        'boundary': [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 2],
            [0, 1],
            [-1, 0]
        ],
        'interior': [
        ]
    },
    'simplex2': {
        'boundary': [
            [0, 0],
            [1, 0],
            [2, 0],
            [1, 1],
            [0, 2],
            [0, 1]
        ],
        'interior': []
    },
    'wedge': {
        'boundary': [
            [1, 2],
            [0, 1],
            [-1, 0],
            [-1, -1],
            [0, -1]
        ],
        'interior': [
            [0, 0]
        ]
    },
    'vase': {
        'boundary': [
            [1, 0],
            [2, 2],
            [1, 2],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-2, -1],
            [-1, -1],
            [0, 0],
            [0, -1],
            [1, 0]
        ],
        'interior': [
            [1, 1]
        ]
    },
    'heesch': {
        'boundary': [
            [1, 0],
            [2, 0],
            [1, 1],
            [0, 1],
            [1, 2],
            [0, 2],
            [-1, 2],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [-1, -2],
            [0, -2],
            [1, -2],
            [2, -2],
            [1, -1],
            [0, -1]
        ],
        'interior': [
            [0, 0]
        ]
    }
}

reflections = {
    "A2": {
        "s1": [[-1, 0], [1, 1]],
        "s2": [[1, 1], [0, -1]]
    },
    "B2": {
        "s1": [[-1, 0], [0, 1]],
        "s2": [[1, 0], [0, -1]]
    },
}

rotations = {
    "A2": {
        "r1": [[1, 1], [-1, 0]]
    },
    "B2": {
        "r1": [[0, 1], [-1, 0]]
    }
}


for lattice_type, order in [['A2', 6], ['B2', 4]]:
    R = np.array(rotations[lattice_type]['r1'])
    R_new = R
    for i in range(2, order + 1):
        R_new = R_new @ R
        rotations[lattice_type]['r'+str(i % order)] = R_new 

print(rotations)

def display_tiles(tiles, lattice_type='A2', ax=None, erase=True):
    if erase:
        ax.clear() # redraw everything
        ax.set_title(f"{len(tiles)} tiles (max {max_on_display})")
    else:
        ax.set_title(f"{len(tiles)} tiles (max {max_on_display})")
        tiles = tiles[tiles_on_display:] # only draw additional tiles
    
    transformation_matrix = np.array([[1, 0], 
                                      [-1/2, np.sqrt(3)/2]]) if lattice_type == 'A2' else np.eye(2)
    
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'red']

    for i, tile in enumerate(tiles):
        color = colors[int(tile.move[0][-1]) % 6]

        points = tile.boundary @ transformation_matrix
        points = np.vstack([points, points[0]])
        x, y = points[:, 0], points[:, 1]
        ax.plot(x, y, color=color, linewidth=1, label=f'Polygon {i+1}')
        
        # if tile.move[0][1] != '0':
        #     mask = tile.mask
        #     interior_points = np.argwhere(mask.mask == 2) + np.array([mask.x1, mask.y1], dtype=int)
        #     interior_points = interior_points @ transformation_matrix
        #     x, y = interior_points[:, 0], interior_points[:, 1]
        #     ax.scatter(x, y, color=color, s=1, label=f'Polygon {i+1}')

        for marking in tile.markings:
            points = marking @ transformation_matrix
            x, y = points[:, 0], points[:, 1]
            ax.plot(x, y, color=color, linewidth=1, label=f'Polygon {i+1}')
    
    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.01)

    # print("Press Enter to continue...")
    # input()  # Wait for the user to press Enter


def get_tile(name):
    return [np.array(named_tiles.get(name).get(key)) for key in ['boundary', 'interior']]


def tileMask(tile):
    min_x, min_y = np.min(tile[0], axis=0)
    max_x, max_y = np.max(tile[0], axis=0)

    mask = np.zeros((max_x - min_x + 1, max_y - min_y + 1), dtype=int)
    for point in tile[0]:
        x, y = point
        mask[x - min_x, y - min_y] = 1
    for point in tile[1]:
        x, y = point
        mask[x - min_x, y - min_y] = 2
    
    return Mask(mask, min_x, min_y)

class Tile:
    def __init__(self, boundary, mask, markings=[], move=['r0', [0, 0]]):
        self.boundary = boundary
        self.mask = mask
        self.markings = markings
        self.move = move  # [str, np.array([x, y])]

    def translate(self, vector=None):
        if vector is None:
            vector = np.array([0, 0], dtype=int)
        new_vector = [self.move[1][i] + vector[i] for i in range(2)]
        return Tile(boundary=self.boundary + vector,
                    mask=self.mask.translate(vector),
                    markings=[_ + vector for _ in self.markings], 
                    move=[self.move[0], new_vector])
    
    def sameAs(self, other):
        return self.move[0] == other.move[0] and np.all(self.move[1] == other.move[1])

    def shift_boundary(self, shift):
        self.boundary = np.concatenate((self.boundary[shift:], self.boundary[:shift+1]), axis=0)

class TileNode:
    def __init__(self, tiles=[], boundary=None, union_mask=None, deadends=[]):
        self.boundary = boundary
        self.union_mask = union_mask
        self.children = []
        self.tiles = tiles.copy()
        self.deadends = deadends

class Mask:
    def __init__(self, mask, x1, y1):
        self.mask = mask.copy()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + mask.shape[0]
        self.y2 = y1 + mask.shape[1]

    def translate(self, vector):
        return Mask(self.mask, self.x1 + vector[0], self.y1 + vector[1])

    def is_contained_in(self, other):
        return other.x1 <= self.x1 and other.x2 >= self.x2 and other.y1 <= self.y1 and other.y2 >= self.y2

    def get_intersection_box(self, other):
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if x1 >= x2 or y1 >= y2:
            return None
        return x1, y1, x2, y2

    def get_union_box(self, other):
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        x2, y2 = max(self.x2, other.x2), max(self.y2, other.y2)
        return x1, y1, x2, y2
    
    def copy(self):
        return Mask(self.mask, self.x1, self.y1)

def overlap(mask1, mask2, containment):
    if containment:
        x1, x2 = mask1.x1 - mask2.x1, mask1.x2 - mask2.x1
        y1, y2 = mask1.y1 - mask2.y1, mask1.y2 - mask2.y1
        return np.any(mask1.mask + mask2.mask[x1:x2, y1:y2] > 2)
    else:
        inter_box = mask1.get_intersection_box(mask2)
        if inter_box is None:
            return 0
        x1, y1, x2, y2 = inter_box
        mask1_overlap = mask1.mask[x1-mask1.x1:x2-mask1.x1, y1-mask1.y1:y2-mask1.y1]
        mask2_overlap = mask2.mask[x1-mask2.x1:x2-mask2.x1, y1-mask2.y1:y2-mask2.y1]
        return np.any(mask1_overlap + mask2_overlap > 2)

def merge(tile, boundary, union_mask):
    mask = tile.mask
    containment = mask.is_contained_in(union_mask)
    if overlap(mask, union_mask, containment):
        return None, "overlap"
    
    u, d = 1, 0 # for upstream and downstream along the boundary
    while np.all(boundary[u+1] == tile.boundary[-2-u]):
        u += 1
    while np.all(boundary[-2-d] == tile.boundary[1+d]):
        d += 1
    while np.all(boundary[u+1] == boundary[-2-d]): # close up any bubble
        u += 1
        d += 1
        print('enclave closing...', [u, d])
    if np.all(2 * boundary[-2-d] - 2 * tile.boundary[1+d] == 3 * boundary[-3-d] - 3 * tile.boundary[2+d]):
        # print('backclaw skipped')
        return None, 'backclaw skipped'
    common_points = [p1 for p1 in boundary[-7-d:-2-d] for p2 in tile.boundary[1+d:-1-u] if np.array_equal(p1, p2)]
    if len(common_points) > 0:
        # print('enclave skipped', common_points)
        return None, 'enclave skipped'

    new_boundary = np.concatenate((boundary[u:-1-d], tile.boundary[d:-u]), axis=0)

    if containment:
        x1, y1 = union_mask.x1, union_mask.y1
        new_mask = Mask(union_mask.mask.copy(), x1, y1)
    else:
        x1, y1, x2, y2 = mask.get_union_box(union_mask)
        new_mask = Mask(np.zeros((x2-x1, y2-y1), dtype=int), x1, y1)
        new_mask.mask[union_mask.x1-x1:union_mask.x2-x1, union_mask.y1-y1:union_mask.y2-y1] = union_mask.mask
    
    new_mask.mask[mask.x1-x1:mask.x2-x1, mask.y1-y1:mask.y2-y1] += mask.mask
    for pt in [tile.boundary[d], tile.boundary[-1-u]]:
        new_mask.mask[pt[0]-x1, pt[1]-y1] = 1

    return new_boundary, new_mask, boundary[:u] 

def border_length(path):
    return sum(x['border'].shape[0] for x in path)

def addChildren(node, lattice_type, counts, ax):
    global tiles_on_display, max_on_display
    k = 1 # tiling with a k-fold rotational symmetry; k=1 means no symmetry
    perimeter = (node.boundary.shape[0] - 1) // k

    # node.deadends = [path for path in node.deadends if border_length(path) < perimeter]
    matches = [path for path in node.deadends if np.all(path[0]['border'][0] == node.boundary[0])]
    new_deadends = []
    first_edge = node.boundary[1] - node.boundary[0]

    keys = rototiles.keys()
    # if counts[0] < 6: 
    #     keys = [key for key in keys if key[:2] == 't0']
    # else:
    #     keys = [key for key in keys if key[:2] != 't2']
    for tile_key in keys:
        tile_idx = int(tile_key.split('_')[0][-1]) # 0, 1, ...
        base_tile = rototiles[tile_key]
        tile_boundary = base_tile.boundary
        for j in range(len(tile_boundary)):
            # find the edge of tile that matches with the first edge to be tiled against
            if not np.all(tile_boundary[j-1] - tile_boundary[j] == first_edge):
                continue # no match, skip

            vector = node.boundary[0] - tile_boundary[j]
            new_tile = base_tile.translate(vector)
            new_tile.shift_boundary(j) # shift the matching edge to be at index 0
            
            results = merge(new_tile, node.boundary, node.union_mask)
            if results[0] is None:
                continue # new_tile doesn't fit

            matches2 = [path for path in matches if len(path) > 1 and path[0]['tile'].move == new_tile.move]
            if len(matches2) > 0: # and all(border_length(path) <= perimeter for path in matches2):
                matches3 = []
                for path in matches2:
                    concat_border = np.concatenate([link['border'] for link in path], axis=0)
                    length = concat_border.shape[0]
                    if length <= perimeter and np.all(concat_border == node.boundary[:length]):
                        pass
                    else:
                        matches3.append(path) # boundary ending has changed, may be able to tile
                matches4 = []
                for path in matches3:
                    if len(path) == 0:
                        continue
                    new_boundary = node.boundary
                    union_mask = node.union_mask
                    new_tiles = []
                    i = 0
                    while i < len(path) - 2:
                        res = merge(path[i]['tile'], new_boundary, union_mask)
                        if res[0] is None:
                            print('breaking out @', i, len(path), res[1])
                            print(path[i]['border'], new_boundary[:5])
                            if i > 0 and any(path[:i] == x for x in [y for y in matches4 if len(y) == i]):
                                print("repeated, not added")
                                new_tiles = []
                            else:
                                print("adding to matches4")
                                matches4.append(path[:i])
                            break
                        else:
                            new_boundary, union_mask, border = res
                            new_tiles.append(path[i]['tile'])
                            i += 1
                    if len(new_tiles) == 0:
                        path.clear()
                        continue
                    all_tiles = node.tiles + new_tiles
                    display_tiles(all_tiles, lattice_type, ax, True)
                    tiles_on_display = len(all_tiles)
                    if tiles_on_display > max_on_display:
                        max_on_display = tiles_on_display
                        
                    pass_on = []
                    for p in new_deadends:
                        n = 0
                        while n < len(p) and not np.any(np.all(new_boundary[0] == p[n]['border'], axis=1)):
                            n += 1
                        if n < len(p):
                            pass_on.append(p[n:])
                    if len(pass_on) > 0:
                        print('passing on on', len(pass_on), 'of', len(node.deadends), '+', len(new_deadends))
                    child_node = TileNode(all_tiles, new_boundary, union_mask, pass_on)
                    return_paths = addChildren(child_node, lattice_type, counts, ax)
                    # print([1 for p in return_paths if len(path[:i] + p) == 0])
                    new_deadends += [path[:i] + p for p in return_paths]
                    path.clear()
                # if len(matches3) == 0:
                continue

            new_boundary, union_mask, border = results
            all_tiles = node.tiles + [new_tile]

            new_counts = [_ for _ in counts]
            new_counts[tile_idx] += 1

            # for i in range(1, k):  # making k-fold symmetric tiling
            #     R = rotations.get(lattice_type).get('r' + str(i * 6 // k))
            #     key = tile_key[:-1] + str((int(tile_key[-1]) + i * 6 // k) % 6)
            #     new_tile = rototiles[key].translate(vector @ R)
            #     new_tile.shift_boundary(j)
            #     shift = perimeter - len(border)
            #     new_boundary = np.concatenate((new_boundary[shift:-1], new_boundary[:shift+1]), axis=0)
            #     results = merge(new_tile, new_boundary, union_mask)
            #     if results[0] is None:
            #         continue 
            #     new_boundary, union_mask, border = results
            #     all_tiles.append(new_tile)
            #     new_counts[tile_idx] += 1
            
            display_tiles(all_tiles, lattice_type, ax, len(all_tiles) <= tiles_on_display)
            tiles_on_display = len(all_tiles)
            if len(all_tiles) > max_on_display:
                max_on_display = len(all_tiles)
            # input()

            pass_on = []
            # for path in node.deadends + new_deadends:
            #     n = 0
            #     while n < len(path) and not np.any(np.all(new_boundary[0] == path[n]['border'], axis=1)):
            #         n += 1
            #     if n < len(path):
            #         pass_on.append(path[n:])
            if len(pass_on) > 0:
                print('passing on', len(pass_on), 'of', len(node.deadends), '+', len(new_deadends))

            child_node = TileNode(all_tiles, new_boundary, union_mask, pass_on)
            return_paths = addChildren(child_node, lattice_type, new_counts, ax)
            for return_path in return_paths:
                concat_path = [{'border': border, 'tile': new_tile}] + return_path
                # length = sum(link['border'].shape[0] for link in concat_path)
                # path_border = np.concatenate([link['border'] for link in concat_path], axis=0)
                # if np.all(path_border == node.boundary[:length]):
                new_deadends.append(concat_path)
                
    if len(new_deadends) == 0:
        return [[{'border': node.boundary[:14], 'tile': None}]]
    
    return new_deadends


lattice_type = 'A2'
S = np.array(reflections.get(lattice_type).get('s1'))

turtle = get_tile('turtle')
turtle.append(np.array([[2, 1], [0, -3]]))

turtle_s = [(turtle[i] @ S) for i in range(2)]
turtle_s[0] = turtle_s[0][::-1]
turtle_s.append(np.array([[2, 1], [0, -3]]))
turtle_s.append(np.array([[4, 2], [-4, -2]]))
turtle_s.append(np.array([[-1, 1], [1, -1]]))

hat = get_tile('hat')
hat.append(np.array([[-1, 2], [-2, 0]]))

hat_s = [(hat[i] @ S)[::-1] for i in range(2)]
hat_s.append(np.array([[1, 1], [-1, -3]]))
hat_s.append(np.array([[4, 1], [3, 2], [-1, 0], [-2, 1]]))

hex2 = get_tile('hex2')
triskelion = get_tile('triskelion')
star12 = get_tile('star12')

pent = get_tile('pent')
wedge = get_tile('wedge')
wedge_s = [(_ @ S)[::-1] for _ in wedge]
vase = get_tile('vase')

simplex = get_tile('simplex')
simplex2 = get_tile('simplex2')


# lattice_type = 'B2'
# S = np.array(reflections.get(lattice_type).get('s1'))
# heesch = get_tile('heesch')
# heesch_s = [(heesch[i][::-1] @ S) for i in range(2)]



prototiles = [turtle, turtle_s] # the set of shapes to be used in tiling
# prototiles = [simplex, simplex2]
# prototiles = [vase]
# prototiles.append(triskelion)
# prototiles = [hat, hat_s]

# prototiles = [heesch, heesch_s]

rototiles = dict() # rotated prototiles, or rototiles for short
for tile_idx, tile in enumerate(prototiles):
    order = 6 if lattice_type == 'A2' else 4
    for i in range(order):
        R = rotations.get(lattice_type).get('r'+str(i))
        tile_r = [points @ R if points.shape[-1] == 2 else points for points in tile]
        key = 't' + str(tile_idx) + '_r' + str(i)
        rototiles[key] = Tile(boundary=tile_r[0],
                                mask=tileMask(tile_r),
                                markings=[tile_r[i] for i in range(2, len(tile_r))],
                                move=[key, [0, 0]]
                                )
print(rototiles.keys())

# R = rotations.get(lattice_type).get('r5')
root_tile = turtle_s # [_ @ R for _ in turtle_s]
# root_tile = heesch
# root_tile = vase
root_boundary = root_tile[0]
root_boundary = np.append(root_boundary, root_boundary[0:1], axis=0)
root_mask = tileMask(root_tile)
tiles = [Tile(boundary=root_boundary,
            mask=root_mask,
            markings=[root_tile[i] for i in range(2, len(root_tile))],
            move=['root_r0', [0, 0]]
)]

root_node = TileNode(tiles, root_boundary, root_mask)

print("Starting at ", root_boundary[0])

plt.ion()
fig, ax = plt.subplots()

tiles_on_display, max_on_display = 0, 0
display_tiles(tiles, lattice_type, ax)

addChildren(root_node, lattice_type, [0 for _ in prototiles], ax)

# display_tiles([{'polygon': turtle}], 'A2', ax)

plt.ioff()
plt.show()

print("Done")
