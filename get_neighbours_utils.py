def get_all_neighbours(verts, L):
    neighbour_dict = {}
    i = 0
    while i < len(verts):
        vi = []
        j = 0
        while j < len(verts):
            if (L[i][j] != 0) and (L[i][j] != -1):
                vi.append(j)
            j += 1
        neighbour_dict[i] = vi
        i += 1
    return neighbour_dict
