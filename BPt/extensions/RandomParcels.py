import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator


class RandomParcels(BaseEstimator):

    def __init__(self, geo, n_parcels, medial_wall_inds=None,
                 medial_wall_mask=None, random_state=None):
        ''' This object is used to create random surface parcellations

        Parameters
        -----------
        geo : list of array-like
            geo, or geometry, should be passed as a list of arrays
            (or sets, or lists), where
            each index of the outer list is associated with that vertex index.
            Each list should contain all of that vertex's valid nieghbors.
            So for example, if the vertex at index 0 has nieghbors 5, 8, 9, 10,
            then the first element of passed geo should be [5,8,9,10].

        n_parcels : int
            The number of random parcels to generate.
            Will generate as 1-n_parcels, with 0 reserved for
            non-parcels.

        medial_wall_inds : array-like of int or None, optional
            By passing an array-like containing valid ints
            between 0 and the size of the parcel, this parameter
            can specify optionally index where parcels should not be
            generated.

            This parameter should be left as None if
            passing medial_wall_mask, as only one will be used.

            ::

                default = None

        medial_wall_mask : array-like of bool or None, optional
            Simmilar to medial_wall_inds, you may also specify
            the medial wall, or spots not to fill, via this
            parameter by passing a boolean mask, where 1 or True
            signifies that vertex is part of the medial wall to exclude.

            This parameter should be left as None if
            passing medial_wall_inds, as only one will be used.

            ::

                default = None

        random_state : int, None, or RandomState, optional
            A random seed to be passed to create a numpy RandomState,
            as either a numpy random state, an integer seed, or None
            to generate a random RandomState.

            ::

                default = None
        '''

        # Set passed params
        self.geo = geo
        self.n_parcels = n_parcels
        self.medial_wall_inds = medial_wall_inds
        self.medial_wall_mask = medial_wall_mask
        self.random_state = random_state
        self.mask = None

    def get_parc(self, copy=True):

        if self.mask is None:
            self._generate_parc_from_params()

        if copy:
            return self.mask.copy()
        else:
            return self.mask

    def _generate_parc_from_params(self):

        # Proc by input args
        self._proc_medial_wall()
        self._proc_geo()
        self._proc_random_state()

        # Set up mask, done and flags
        self.sz = len(self._geo)
        self.reset()

        # Init
        self.init_parcels()

        # Then generate
        self.generate_parcels()

    def _proc_geo(self):
        '''Convert geo to list of np arrays. Also set so only
        consider neighbors as non medial wall, if any medial wall passed.'''
        self._geo = []
        for g in self.geo:
            valid_n = list(set(g) - self.m_wall)
            self._geo.append(np.array(valid_n))

    def _proc_medial_wall(self):

        # Proc medial wall inds
        if self.medial_wall_inds is not None:
            self.m_wall = set(list(self.medial_wall_inds))
        elif self.medial_wall_mask is not None:
            self.m_wall = set(list(np.where(self.medial_wall_mask)[0]))
        else:
            self.m_wall = set()

    def _proc_random_state(self):

        if self.random_state is None:
            self.r_state = RandomState()
        elif isinstance(self.random_state, int):
            self.r_state = RandomState(seed=self.random_state)
        else:
            self.r_state = self.random_state

    def reset(self):
        '''Just reset the mask, and set w/ done info'''

        self.mask = np.zeros(self.sz, dtype='int16')
        self.done = self.m_wall.copy()
        self.ready, self.generated = False, False

    def init_parcels(self):

        # Generate the starting locs
        valid = np.setdiff1d(np.arange(self.sz), np.array(list(self.done)))
        self.start_locs = self.r_state.choice(valid, size=self.n_parcels,
                                              replace=False)

        # Set random probs. that each loc is chosen
        self.probs = self.r_state.random(size=self.n_parcels)

    def setup(self):
        '''This should be called before generating parcel,
        so after a mutation has been made, setup needs to
        be called. It also does not hurt to call setup an
        extra time, as nothing random is set.'''

        # Generate corresponding labels w/ each loc
        self.labels = np.arange(1, self.n_parcels+1, dtype='int16')

        # Mask where if == 1, then that parcel is done
        self.finished = np.zeros(self.n_parcels, dtype='bool_')

        # Drop the first points
        self.mask[self.start_locs] = self.labels

        # Set ready flag to True
        self.ready = True

    def get_probs(self):

        return self.probs / np.sum(self.probs)

    def choice(self):
        '''Select a valid label based on probs.'''

        msk = self.finished == 0
        probs = self.probs[msk] / np.sum(self.probs[msk])
        label = self.r_state.choice(self.labels[msk], p=probs)

        return label

    def get_valid_neighbors(self, loc):

        ns = self._geo[loc]
        valid_ns = ns[self.mask[ns] == 0]

        return valid_ns

    def generate_parcels(self):

        if self.ready is False:
            self.setup()

        # Keep looping until every spot is filled
        while (self.finished == 0).any():
            self.add_spot()

        # Set generated flag when done
        self.generated = True

    def add_spot(self):

        # Select which parcel to add to
        label = self.choice()

        # Determine valid starting locations anywhere in exisitng parcel
        current = np.where(self.mask == label)[0]
        valid = set(current) - self.done

        self.proc_spot(valid, label)

    def proc_spot(self, valid, label):

        # If no valid choices, then set this parcel to finished
        if len(valid) == 0:
            self.finished[label-1] = 1
            return

        # Select randomly from the valid starting locs
        loc = self.r_state.choice(tuple(valid))

        # Select a valid + open neighbor
        valid_ns = self.get_valid_neighbors(loc)

        if len(valid_ns) > 0:

            # Select a valid choice, and add it w/ the right label
            choice = self.r_state.choice(valid_ns)
            self.mask[choice] = label

            # If this was the only choice, mark start loc as done
            if len(valid_ns) == 1:
                self.done.add(loc)

        # If there are no valid choices, mark as done
        else:
            self.done.add(loc)

            valid.remove(loc)
            self.proc_spot(valid, label)
