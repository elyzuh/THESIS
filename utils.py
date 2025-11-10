import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon

        fin = open(args.data)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = args.normalize
        self.scale = np.ones(self.m)
        self._normalized(self.normalize)

        # number of training samples, numebr of traninging and validation samples
        self._split(int(args.train * self.n), int((args.train+args.valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()

        #compute denominator of the RSE and RAE
        self.compute_metric(args)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(args.sim_mat, delimiter=','))

        # for i in range(0,len(self.adj)):
        #     for j in range(0,len(self.adj)):
        #         if i == j:
        #             self.adj[i][j]=0
        #         else:
        #             # dense
        #             self.adj[i][j]=1

        # for i in range(0,len(self.adj)):
        #     for j in range(0,len(self.adj)):
        #         self.adj[i][j]=1
        
        # print ("***********")
        # print (self.adj)

        # normalize
        # rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        # self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        # print (self.adj)

        # print (rowsum)
        # print (rowsum[:, np.newaxis].shape)
        # print (rowsum[np.newaxis, :].shape)

        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()

    def compute_metric(self, args):
        #use the normal rmse and mae when args.metric == 0
        if (args.metric == 0):
            self.rse = 1.
            self.rae = 1.
            return

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        #normalized by the maximum value of entire matrix.
        if (normalize == 1):
            self.scale = self.scale * (np.mean(np.abs(self.rawdat))) * 2
            self.dat = self.rawdat / (np.mean(np.abs(self.rawdat)) * 2)

        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))

    def _split(self, train, valid, test):
        # Store the split points as attributes of the class
        self._train_end = train
        self._valid_end = valid

        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

        if (train==valid):
            self.valid = self.test

    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))

        # print ("batchify")

        for i in range(n):
            # the end point of x
            end = idx_set[i] - self.h + 1
            # the start point of x
            start = end - self.P
            X[i,:self.P,:] = torch.from_numpy(self.dat[start:end, :])
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])

            # print (self.dat.shape, n,start,end,idx_set[i])
            # # ?
            # Y[i,:] = torch.from_numpy(self.dat[idx_set[i]:idx_set[i]+self.h, :])

        return [X, Y]

    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data
            start_idx += batch_size

    def get_pinn_batches(self):
        # This method prepares the data in the (t, loc, value) format needed by the PINN.
        
        # 1. Prepare data points where we have ground truth infection values
        train_indices = np.arange(len(self.train[1]))
        Y_train_truth = self.train[1]  # Shape: [num_train_samples, m]
        
        t_data, i_data, loc_data = [], [], []
        
        # The time 't' is the actual row index in the original dataset
        time_indices = np.arange(self.P + self.h - 1, self._train_end)

        # Flatten the data from a grid into a list of (t, loc, I) points
        for i, t in enumerate(time_indices):
            for loc in range(self.m):
                t_data.append(t)
                loc_data.append(loc)
                i_data.append(Y_train_truth[i, loc])

        # Convert to tensors
        t_data_tensor = torch.tensor(t_data, dtype=torch.float32).unsqueeze(1)
        i_data_tensor = torch.tensor(i_data, dtype=torch.float32).unsqueeze(1)
        loc_data_tensor = torch.tensor(loc_data, dtype=torch.float32).unsqueeze(1)

        # 2. Prepare "collocation points" where we will enforce the SEIR equations
        # These are random points spread across the entire training domain
        num_physics_points = 4096 # More points help enforce the physics better
        t_physics = torch.FloatTensor(num_physics_points, 1).uniform_(0, self._train_end)
        loc_physics = torch.FloatTensor(num_physics_points, 1).uniform_(0, self.m - 1e-5) # Use a small epsilon

        if self.cuda:
            return t_data_tensor.cuda(), i_data_tensor.cuda(), loc_data_tensor.cuda(), t_physics.cuda(), loc_physics.cuda()
        return t_data_tensor, i_data_tensor, loc_data_tensor, t_physics, loc_physics

    def get_pinn_test_batches(self):
        # This prepares the test data in the same (t, loc, value) format
        Y_test_truth = self.test[1] # Shape: [num_test_samples, m]
        
        t_test, i_test, loc_test = [], [], []
        
        time_indices = np.arange(self._valid_end, self.n)

        for i, t in enumerate(time_indices):
            for loc in range(self.m):
                t_test.append(t)
                loc_test.append(loc)
                i_test.append(Y_test_truth[i, loc])
        
        t_test_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
        i_test_tensor = torch.tensor(i_test, dtype=torch.float32).unsqueeze(1)
        loc_test_tensor = torch.tensor(loc_test, dtype=torch.float32).unsqueeze(1)

        if self.cuda:
            return t_test_tensor.cuda(), i_test_tensor.cuda(), loc_test_tensor.cuda()
        return t_test_tensor, i_test_tensor, loc_test_tensor