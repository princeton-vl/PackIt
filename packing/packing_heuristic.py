import numpy as np
import scipy as sci
from packing.packing_env import PackingEnv

# maps the dimension to the rotation action
# dim is a tuple with the dimesions arranged in assending order
# rot is a tuple od best possible rotation actions
DIM_ROT = {
    ('x', 'y', 'z'): (0, 2, 8, 10),
    ('x', 'z', 'y'): (4, 6, 12, 14),
    ('y', 'x', 'z'): (16, 18, 20, 22),
    ('y', 'z', 'x'): (5, 7, 13, 15),
    ('z', 'x', 'y'): (17, 19, 21, 23),
    ('z', 'y', 'x'): (1, 3, 9, 11)
}


# source: https://docs.python.org/3/howto/sorting.html
def cmp_to_key(mycmp):
    """ Convert a cmp= function into a key= function.

        Eg:
        >>> sorted([5, 2, 4, 1, 3], key=cmp_to_key(reverse_numeric))
        [5, 4, 3, 2, 1]
    """
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


def sha_lar(obs):
    """
        Choses the largest available shape
        Args:
            obs (dict): decoded dictionary from packing env
    """

    sha_rep = obs["sha_rep"]
    sha_rep = np.reshape(sha_rep, (PackingEnv.MAX_NUM_SHA, -1))
    sha_rep = (sha_rep * np.expand_dims(obs["sha_mask"], axis=1))
    sha_vol = np.sum(sha_rep, axis=1)
    return np.argmax(sha_vol)


def sha_lar_n(obs, n):
    """
        Choses the largest n shapes out of the available ones
        If less that n shapes available, we repeat -1 for the
        remaining shapes and 0 score
        If 0 shapes available, an all zero vector returned
        Args:
            obs (dict): decoded dictionary from packing env
        Return:
            shape, score
    """

    sha_rep = obs["sha_rep"]
    sha_rep = np.reshape(sha_rep, (PackingEnv.MAX_NUM_SHA, -1))
    sha_rep = (sha_rep * np.expand_dims(obs["sha_mask"], axis=1))
    sha_vol = np.sum(sha_rep, axis=1)

    num_sha_nz = np.count_nonzero(sha_vol)
    sha_vol_arg_des = np.argsort(sha_vol)[::-1]
    shape = -1 * np.ones((n), dtype=np.int32)
    score = np.ones((n))

    num_sha = min(num_sha_nz, n)
    for i in range(num_sha):
        shape[i] = sha_vol_arg_des[i]
        score[i] = sha_vol[sha_vol_arg_des[i]]

    return shape, score


def sha_all_sorted(obs):
    """
        All the possible shapes that one can chose in a sorted
        order
        Args:
            obs (dict): decoded dictionary from packing env
        Return:
            sha_pos nparrent(num_pos_sha):
    """

    sha_rep = obs["sha_rep"]
    sha_rep = np.reshape(sha_rep, (PackingEnv.MAX_NUM_SHA, -1))
    sha_rep = (sha_rep * np.expand_dims(obs["sha_mask"], axis=1))
    sha_vol = np.sum(sha_rep, axis=1)

    num_sha_nz = np.count_nonzero(sha_vol)
    sha_pos = np.argsort(sha_vol)[::-1][0: num_sha_nz]

    return sha_pos


def sha_rand(obs):
    """
        Choses the largest available shape
        Args:
            obs (dict): decoded dictionary from packing env
    """

    sha_mask = obs["sha_mask"]
    sha_mask = np.where(sha_mask == 1)[0]

    return np.random.choice(sha_mask, 1)


def mov_best(obs):
    """
        Args:
            obs (dict): decoded dictionary from packing env
    """
    pos_mov = obs["pos_mov"]
    index = np.where(pos_mov == 1)

    if index[0].size == 0:
        return 0
    else:
        """
        same format to PackingEnv.get_get_action
            return (int)(unity_gt_action[2] \
                 + (unity_gt_action[1] * self.MOV_RES ) \
                 + (unity_gt_action[0] * (self.MOV_RES ** 2)))
        """
        return int(index[2][0]
                   + index[1][0] * PackingEnv.MOV_RES
                   + index[0][0] * (PackingEnv.MOV_RES ** 2))


def mov_rand(obs):
    """
        Args:
            obs (dict): decoded dictionary from packing env
    """
    pos_mov = obs["pos_mov"]
    index = np.where(pos_mov == 1)

    if index[0].size == 0:
        return 0
    else:
        # Return random integers from the “discrete uniform” distribution
        # of the specified dtype in the “half-open” interval [low, high).
        rand_loc = np.random.randint(low=0, high=len(index[0]))
        """
        same format to PackingEnv.get_get_action
            return (int)(unity_gt_action[2] \
                 + (unity_gt_action[1] * self.MOV_RES ) \
                 + (unity_gt_action[0] * (self.MOV_RES ** 2)))
        """
        return int(index[2][rand_loc]
                   + index[1][rand_loc] * PackingEnv.MOV_RES
                   + index[0][rand_loc] * (PackingEnv.MOV_RES ** 2))


def rot_pos(obs):
    """
        Returns the first possible rotation
        Args:
            obs (dict): decoded dictionary from packing env
    """
    pos_rot = obs["pos_rot"]
    index = np.where(pos_rot == 1)

    if index[0].size == 0:
        return 0
    else:
        return index[0][0]


def rot_rand(obs):
    """
        Returns a random rotation out of the possible ones
    """

    pos_rot = obs["pos_rot"]
    cho_sha_rep = obs["cho_sha_rep"]

    # if a shape does not fit, these values can be 0
    # however, to sync the evs we still execute actions
    if np.sum(pos_rot) == 0 or np.sum(cho_sha_rep) == 0:
        return 0
    else:
        pos_rot = np.where(pos_rot == 1)[0]
        return np.random.choice(pos_rot, 1)


def get_sha_dims(sha_rep, return_start_end=False):
    """
        Returns the x, y and z dimensions of the cube that tightly encloses
        the shape

    Args:
        shape_rep(numpy.array(,,)):
        return_start_end(bool): whether to return start end indices or
            dimensions
    """

    sha_rep_x = np.sum(sha_rep, axis=(1, 2), keepdims=False)
    sha_rep_y = np.sum(sha_rep, axis=(0, 2), keepdims=False)
    sha_rep_z = np.sum(sha_rep, axis=(0, 1), keepdims=False)

    x_start = np.where(sha_rep_x > 0)[0][0]
    x_end = np.where(sha_rep_x > 0)[0][-1]
    y_start = np.where(sha_rep_y > 0)[0][0]
    y_end = np.where(sha_rep_y > 0)[0][-1]
    z_start = np.where(sha_rep_z > 0)[0][0]
    z_end = np.where(sha_rep_z > 0)[0][-1]

    if return_start_end:
        return x_start, x_end, y_start, y_end, z_start, z_end

    else:
        x_dim = x_end - x_start
        y_dim = y_end - y_start
        z_dim = z_end - z_start
        return x_dim, y_dim, z_dim


def get_dim_order(x, y, z):
    """
        Returns a tuple with the order of dimensions. Tuple can be used with
        DIM_ROT.
    """

    if x <= y and x <= z:
        if y <= z:
            return ('x', 'y', 'z')
        else:
            return ('x', 'z', 'y')

    elif y <= x and y <= z:
        if x <= z:
            return ('y', 'x', 'z')
        else:
            return ('y', 'z', 'x')
    else:
        if x <= y:
            return ('z', 'x', 'y')
        else:
            return ('z', 'y', 'x')


def rot_best(obs):
    """
        Returns a rotation based on heuristic of minimizing height
        Only compatible with mov_best
        Should only be used with a rot_before_mov environment
    """

    cho_sha_rep = obs["cho_sha_rep"]
    if np.sum(cho_sha_rep) == 0:
        return 0

    x_dim, y_dim, z_dim = get_sha_dims(cho_sha_rep)
    dim_order = get_dim_order(x_dim, y_dim, z_dim)
    best_rots = DIM_ROT[dim_order]

    return best_rots[0]


def cmp_func(ref, a1, a2):
    """Comparison function for tuples a1 and a2 w.r.t the ref tuple

        if a1 better than a2 return -1
        if a2 better than a1 return 1
        else return 0
        this assignment done as soted return ascending order
    """

    for i in range(len(ref)):
        if a1.index(ref[i]) < a2.index(ref[i]):
            return -1
        elif a1.index(ref[i]) > a2.index(ref[i]):
            return 1
        else:
            continue

    return 0


def rot_best_pos(obs):
    """
        Returns the best possible rotation action out of the possible ones.
        Should be used after a move selection function.
    """

    pos_rot = obs["pos_rot"]
    cho_sha_rep = obs["cho_sha_rep"]

    # if a shape does not fit, these values can be 0
    # however, to sync the evs we still execute actions
    if np.sum(pos_rot) == 0 or np.sum(cho_sha_rep) == 0:
        return 0

    x_dim, y_dim, z_dim = get_sha_dims(cho_sha_rep)
    dim_order = get_dim_order(x_dim, y_dim, z_dim)

    def cmp_func_temp(x, y):
        return cmp_func(dim_order, x, y)

    all_dim_orders = [key for key, value in DIM_ROT.items()]
    sorted_dim_orders = sorted(all_dim_orders,
                               key=cmp_to_key(cmp_func_temp))

    for dim_order in sorted_dim_orders:
        rots = DIM_ROT[dim_order]
        for rot in rots:
            if pos_rot[rot] == 1:
                return rot

    return -1


class HeuristicModel(object):

    def __init__(self, pol_1, pol_2, pol_3):
        """
            Args:
                pol_1: func to be called for step_typ == 1
                pol_2: func to be called for step_typ == 2
                pol_3: func to be called for step_typ == 3
        """

        self.pol_1 = pol_1
        self.pol_2 = pol_2
        self.pol_3 = pol_3

    def action_best(self, obs):
        """
            Args:
                obs(nparray(n_envs, obs_size) or sci.sparse)

            Return:
                action(nparray(n_envs))
        """
        assert len(obs.shape) == 2
        action = np.zeros(np.shape(obs)[0], dtype=np.int32)

        if sci.sparse.issparse(obs):
            obs = obs.toarray()

        for i in range(np.shape(obs)[0]):
            _obs = obs[i]
            _obs = PackingEnv._decode_agent_obs(_obs)

            cleaned_obs = {}
            for key, value in _obs.items():
                if value.shape[0] == 1:
                    cleaned_obs[key] = np.squeeze(value, axis=0)

            if cleaned_obs["step_typ"] == 1:
                action[i] = self.pol_1(cleaned_obs)

            elif cleaned_obs["step_typ"] == 2:
                action[i] = self.pol_2(cleaned_obs)

            else:
                action[i] = self.pol_3(cleaned_obs)

        return action

    def action_best_n(self, obs, n):
        """
            Args:
                obs(nparray(n_envs, obs_size) or sci.sparse)

            Return:
                action(nparray(n_envs, n))
                score(nparray(n_envs, n))
        """
        assert len(obs.shape) == 2
        action = np.zeros([np.shape(obs)[0], n], dtype=np.int32)
        score = np.zeros([np.shape(obs)[0], n])

        if sci.sparse.issparse(obs):
            obs = obs.toarray()

        for i in range(np.shape(obs)[0]):
            _obs = obs[i]
            _obs = PackingEnv._decode_agent_obs(_obs)

            cleaned_obs = {}
            for key, value in _obs.items():
                if value.shape[0] == 1:
                    cleaned_obs[key] = np.squeeze(value, axis=0)

            if cleaned_obs["step_typ"] == 1:
                action[i], score[i] = self.pol_1(cleaned_obs, n)

            elif cleaned_obs["step_typ"] == 2:
                action[i], score[i] = self.pol_2(cleaned_obs, n)

            else:
                action[i], score[i] = self.pol_3(cleaned_obs, n)

        return action, score

    def action_all_sorted(self, obs):
        """
            Args:
                obs(nparray(n_envs, obs_size) or sci.sparse)

            Return:
                action(nparray(n_envs, n))
                score(nparray(n_envs, n))
        """
        assert len(obs.shape) == 2
        action = []

        if sci.sparse.issparse(obs):
            obs = obs.toarray()

        for i in range(np.shape(obs)[0]):
            _obs = obs[i]
            _obs = PackingEnv._decode_agent_obs(_obs)

            cleaned_obs = {}
            for key, value in _obs.items():
                if value.shape[0] == 1:
                    cleaned_obs[key] = np.squeeze(value, axis=0)

            if cleaned_obs["step_typ"] == 1:
                action.append(self.pol_1(cleaned_obs))

            elif cleaned_obs["step_typ"] == 2:
                action.append(self.pol_2(cleaned_obs))

            else:
                action.append(self.pol_3(cleaned_obs))

        return action
