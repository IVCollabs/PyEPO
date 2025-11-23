import pyepo

from torch import nn

NBR_PROCESSES=1

class LossMethods():
    def __init__(self, optmodel, dataset_train):
        self.optmodel = optmodel
        self.dataset_train = dataset_train

    def ptbl1(self, cp, w):
        ptb = pyepo.func.perturbedOpt(self.optmodel, n_samples=3, sigma=1.0, processes=NBR_PROCESSES)
        # set loss
        l1 = nn.L1Loss()
        # perturbed optimizer
        we = ptb(cp)
        # loss
        loss = l1(we, w)
        return loss
    
    def ilmel1(self, cp, w):
        # init i-mle solver
        imle = pyepo.func.implicitMLE(self.optmodel, n_samples=10, sigma=1.0, lambd=10, processes=NBR_PROCESSES)
        # set loss
        l1 = nn.L1Loss()
        # perturbed optimizer
        we = imle(cp)
        # loss
        loss = l1(we, w)
        return loss

    def ailmel1(self, cp, w):
        # init ai-mle solver
        aimle = pyepo.func.adaptiveImplicitMLE(self.optmodel, n_samples=10, sigma=1.0, processes=NBR_PROCESSES)
        # set loss
        l1 = nn.L1Loss()
        # perturbed optimizer
        we = aimle(cp)
        # loss
        loss = l1(we, w)
        return loss

    def dbbl1(self, cp, c, z):
        # init dbb solver
        dbb = pyepo.func.blackboxOpt(self.optmodel, lambd=20, processes=NBR_PROCESSES)
        # set loss
        l1 = nn.L1Loss()
        # black-box optimizer
        wp = dbb(cp)
        # objective value
        zp = (wp * c).sum(1).view(-1, 1)
        # loss
        loss = l1(zp, z)
        return loss

    def nidl1(self, cp, c, z):
        # init nid solver
        nid = pyepo.func.negativeIdentity(self.optmodel, processes=NBR_PROCESSES)
        # set loss
        l1 = nn.L1Loss()
        # black-box optimizer
        wp = nid(cp)
        # objective value
        zp = (wp * c).sum(1).view(-1, 1)
        # loss
        loss = l1(zp, z)
        return loss

    def get_loss_dict(self):
        method_dict = {
            'spo+' : pyepo.func.SPOPlus(self.optmodel, processes=1),
            '2s' : nn.MSELoss(),
            'pg' : pyepo.func.perturbationGradient(self.optmodel, sigma=0.1, two_sides=False, processes=NBR_PROCESSES),
            'ptb' : self.ptbl1,
            'pfy' : pyepo.func.perturbedFenchelYoung(self.optmodel, n_samples=10, sigma=0.5, processes=NBR_PROCESSES),
            'imle' :  self.ilmel1,
            'aimle' :  self.ailmel1,
            'dbb' : self.dbbl1,
            'nid' : self.nidl1,
            'nce' : pyepo.func.NCE(self.optmodel, processes=NBR_PROCESSES, solve_ratio=0.05, dataset=self.dataset_train),
            'cmap' : pyepo.func.contrastiveMAP(self.optmodel, processes=NBR_PROCESSES, solve_ratio=0.05, dataset=self.dataset_train),
            'ltr' : pyepo.func.listwiseLTR(self.optmodel, processes=NBR_PROCESSES, solve_ratio=0.05, dataset=self.dataset_train),
        }
        return method_dict