def get_neighbour_vertex(edges, index):
    vi = []
    for edge in edges:
        if edge[0] == index:
            vi.append(edge[1])
        elif edge[1] == index:
            vi.append(edge[0])
    return vi


def get_all_neighbours(verts, edges):
    neighbour_dict = {}
    i = 0
    while i < len(verts):
        vi = []
        for edge in edges:
            if edge[0] == i:
                vi.append(edge[1].item())
            elif edge[1] == i:
                vi.append(edge[0].item())
        neighbour_dict[i] = vi
        i += 1
    return neighbour_dict
