
class csMALA(MetropolisHastings):
    def __init__(self, dataset, batch_size, inv_temp, learn_rate, std, corr_param, ):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.inv_temp = inv_temp
        self.learn_rate = learn_rate
        self.std = std
        self.corr_param = corr_param

    def run(self, theta, T):
        pass

    def take_subset(self):
        subset_indx = npr.binomial(n=1,p=self.batch_size / self.N, size=self.N)
        return self.dataset[subset_indx == 1]