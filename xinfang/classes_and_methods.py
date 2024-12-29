import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyepo
import time
import torch

from torch import nn

class LinearRegression(nn.Module):

    def __init__(self, num_feat, num_edge):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, num_edge)

    def forward(self, x):
        out = self.linear(x)
        return out

# loader_train and loader_test are global variables; no need to pass as arguments
def trainModel(reg, loss_func, method_name, opt_model, loader_train, loader_test, num_epochs=10, lr=1e-2):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    reg.train()
    # init log
    loss_log = []
    #The initial regret is computed on the test set using pyepo.metric.regret().
    loss_log_regret = [pyepo.metric.regret(reg, opt_model, loader_test)]
    # init elpased time
    elapsed = 0
    for epoch in range(num_epochs):
        print("Start Epoch {:2}".format(epoch+1))
        # start timing
        tick = time.time()
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            
            cp = reg(x)
            if method_name == "spo+":
                loss = loss_func(cp, c, w, z)        
            if method_name == "ptb" or method_name == "pfy" or method_name == "nce" or method_name == "cmap":
                loss = loss_func(cp, w)
            if method_name == "dbb" or method_name == "nid":
                loss = loss_func(cp, c, z)
            #if method_name == "ltr": #learning to rank
            if method_name == "2s" or method_name == "ltr":
                loss = loss_func(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss.mean().backward() #$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            optimizer.step()
            # record time
            tock = time.time()
            elapsed += tock - tick
            # log
            loss = loss.mean() #$$$$$$$$$$$$$$$$$$$$$$$$
            loss_log.append(loss.item())
        regret = pyepo.metric.regret(reg, opt_model, loader_test)
        loss_log_regret.append(regret)
        print("Epoch {:2},  Loss (Train): {:9.4f},  Regret (Validation): {:7.4f}%".format(epoch+1, loss.item(), regret*100))
    print("Total Elapsed Time: {:.2f} Sec.".format(elapsed))
    return loss_log, loss_log_regret 
    #loss_log is loss for training set and loss_log_regret is for testing set

def visLearningCurve(loss_log, loss_log_regret, path, show = False):
    # create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,4))

    # draw plot for training loss
    ax1.plot(loss_log, color="c", lw=1)
    ax1.tick_params(axis="both", which="major", labelsize=12)
    ax1.set_xlabel("Iters", fontsize=16)
    ax1.set_ylabel("Loss", fontsize=16)
    ax1.set_title("Learning Curve on Training Set", fontsize=16)

    # draw plot for regret on test
    ax2.plot(loss_log_regret, color="royalblue", ls="--", alpha=0.7, lw=1)
    ax2.set_xticks(range(0, len(loss_log_regret), 2))
    ax2.tick_params(axis="both", which="major", labelsize=12)
    # Set y-axis limit dynamically based on the max value of loss_log_regret
    max_regret = max(loss_log_regret)+0.1
    ax2.set_ylim(0, max_regret)
   
    # Format y-axis labels as percentages
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))  # xmax=1 if data is normalized, adjust as needed
        
    ax2.set_xlabel("Epochs", fontsize=16)
    ax2.set_ylabel("Regret (%)", fontsize=16)
    ax2.set_title("Learning Curve on Test Set", fontsize=16)

    if show:
        plt.show()

    # Saves the fig in the output path
    fig.savefig(path + 'learning_curve.png', dpi=300, bbox_inches='tight')