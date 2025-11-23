import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyepo
import time
import torch
import math

from torch import nn

class LinearRegression(nn.Module):
    def __init__(self, num_feat, num_edge):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_edge)

    def forward(self, x):
        return self.linear(x)


# loader_train and loader_test are global variables; no need to pass as arguments
def trainModel(reg, loss_func, method_name, opt_model, loader_train, loader_test, num_epochs=10, lr=1e-2):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    reg.train()
    # init log
    loss_log = []
    #The initial regret is computed on the test set using pyepo.metric.regret().
    #loss_log_regret = [pyepo.metric.regret(reg, opt_model, loader_test)]
    loss_log_regret = [(pyepo.metric.regret(reg, opt_model, loader_test), time.time())]

    
    mse_log = [pyepo.metric.mse.MSE(reg,  loader_test)]
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
            if method_name in ["ptb", "pfy", "imle", "aimle", "nce", "cmap"]:
                loss = loss_func(cp, w)
            if method_name in ["dbb", "nid"]:
                loss = loss_func(cp, c, z)
            #if method_name == "ltr": #learning to rank
            if method_name in ["pg", "ltr", "2s"]:
                loss = loss_func(cp, c)
            # backward pass
            optimizer.zero_grad()
            loss.mean().backward() 
            optimizer.step()
            # record time
            tock = time.time()
            elapsed += tock - tick
            # log
            loss = loss.mean() 
            loss_log.append(loss.item())
        regret = pyepo.metric.regret(reg, opt_model, loader_test)
        #loss_log_regret.append(regret)
        loss_log_regret.append((regret, time.time()))

                
        mse_value=pyepo.metric.mse.MSE(reg,  loader_test)
        mse_log.append(mse_value)
        print("Epoch {:2},  MSE (Vali): {:9.4f},  Decision Regret (Vali): {:7.4f}%".format(epoch+1, mse_value, regret*100))
       
    
    print("Total Elapsed Time: {:.2f} Sec.".format(elapsed))
    return loss_log_regret, mse_log
    #loss_log is loss for training set and loss_log_regret is for testing set

def plot_mse_and_regret_side_by_side(df_all, SEED, num, METHODS, OUTPUT_PATH):
    """
    Creates a side-by-side plot of two methods
    """
    method_1 = METHODS[0]
    method_2 = METHODS[1]

    # Extract data
    epochs = df_all.index

    mse_1  = df_all[f"mse_{method_1}"]
    mse_2 = df_all[f"mse_{method_2}"]

    regret_1  = df_all[f"decision_regret_{method_1}"] * 100   # convert to %
    regret_2 = df_all[f"decision_regret_{method_2}"] * 100

    min_1  = regret_1.min()
    min_2 = regret_2.min()

    # --- Create Figure (Two Subplots) ---
    plt.figure(figsize=(14, 5))

    # ---------------------------------------------------
    # 1) MSE subplot
    # ---------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mse_2, marker="o", linestyle="--", linewidth=2, label=f"{method_2} MSE")
    plt.plot(epochs, mse_1, marker="s", linestyle="--", linewidth=2, label=f"{method_1} MSE")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(f"MSE: {method_1} vs {method_2}", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False)

    # ---------------------------------------------------
    # 2) Decision Regret subplot
    # ---------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(
        epochs, regret_2,
        marker="o", linestyle="--", linewidth=2,
        label=f"{method_2} Regret (min: {min_2:.2f}%)"
    )
    plt.plot(
        epochs, regret_1,
        marker="s", linestyle="--", linewidth=2,
        label=f"{method_1} Regret (min: {min_1:.2f}%)"
    )

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Decision Regret (%)", fontsize=12)
    plt.title("Decision Regret: {method_1} vs {method_2}", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False)

    plt.tight_layout()

    # Save figure
    fname = f"{OUTPUT_PATH}plot_mse_regret_side_by_side_seed{SEED}_location{num}.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    print(f"Saved side-by-side {method_1} vs {method_2} Regret figure → {fname}")


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

def haversine(coord1, coord2):
    R = 3959.0  # Radius of the Earth in miles
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Distance in miles
    distance = R * c
    return distance