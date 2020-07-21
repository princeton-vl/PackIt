import numpy as np

from packing.packing_heuristic import get_sha_dims


def indexed_convolution(voxel, _filter, x_indices, y_indices, z_indices,
                        filter_values_same=False):
    """ Performs convolution only at specific locations

        Args:
            voxel (np.array[,,]):
            _filter (np.array[,,]):
            x_indices (np.array[]):
            y_indices (np.array[]):
            z_indices (np.array[]):
            filter_values_same (bool): whether all the values in the filter are same
                for example this should be true for avg_filter
                if same, we can speed up the code
    """

    features = np.zeros((len(x_indices), len(y_indices), len(z_indices)))
    x_filter, y_filter, z_filter = np.shape(_filter)
    for i in range(len(x_indices)):
        for j in range(len(y_indices)):
            for k in range(len(z_indices)):
                x_index = x_indices[i]
                y_index = y_indices[j]
                z_index = z_indices[k]
                cur_voxel = voxel[x_index:x_index + x_filter,
                                  y_index:y_index + y_filter,
                                  z_index:z_index + z_filter]
                if filter_values_same:
                    features[i, j, k] = _filter[0, 0, 0] * np.sum(cur_voxel)
                else:
                    features[i, j, k] = np.sum(cur_voxel * _filter)

    return features


def extract_coarse_fea(voxel, fea_per_dim, norm_fea=True):
    """ To extract coarse features from a voxel

        Args:
            voxel (np.array[,,]):
            fea_per_dim (list[3]):

        Returns:
            features (np.array[fea_per_dim[0]*fea_per_dim[1]*fea_per_dim[2]]):
    """
    # finding all the filter size
    x, y, z = np.shape(voxel)
    x_filter = np.ceil(x / fea_per_dim[0]).astype('int')
    y_filter = np.ceil(y / fea_per_dim[1]).astype('int')
    z_filter = np.ceil(z / fea_per_dim[2]).astype('int')
    avg_filter = (np.ones((x_filter, y_filter, z_filter))
                  / (x_filter * y_filter * z_filter))

    # finding the location for getting the features
    x_indices = np.ceil(
        np.linspace(
            start=0, stop=(x - x_filter), num=fea_per_dim[0],
            endpoint=True)).astype('int')
    y_indices = np.ceil(
        np.linspace(
            start=0, stop=(y - y_filter), num=fea_per_dim[1],
            endpoint=True)).astype('int')
    z_indices = np.ceil(
        np.linspace(
            start=0, stop=(z - z_filter), num=fea_per_dim[2],
            endpoint=True)).astype('int')

    features = indexed_convolution(voxel, avg_filter, x_indices, y_indices,
                                   z_indices, filter_values_same=True)
    features = np.reshape(features,
                          [fea_per_dim[0] * fea_per_dim[1] * fea_per_dim[2]])
    if norm_fea:
        features = (2 * features) - 1

    return features


def extract_fine_fea(voxel, fea_per_dim, norm_fea=True):
    """ It first finds the smallest cube that encompassed the voxel and then
    extracts features from it
    """

    if np.sum(voxel == 0):
        return np.zeros((fea_per_dim[0] * fea_per_dim[1] * fea_per_dim[2]))
    else:
        x_start, x_end, y_start, y_end, z_start, z_end = get_sha_dims(
            voxel, return_start_end=True)
        voxel_fine = voxel[x_start:x_end + 1,
                           y_start:y_end + 1,
                           z_start:z_end + 1]

        return extract_coarse_fea(voxel_fine, fea_per_dim, norm_fea)
