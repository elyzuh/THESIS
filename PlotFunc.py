import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import os

####################################################
# ---- plot figures
####################################################
def PlotTrends(InputRealLocationTimeData, RealLocationTimeData, PredictedLocationTimeData, save_dir, horizon):
    # print (InputRealLocationTimeData)
    # location, number of sample, time length
    # print (InputRealLocationTimeData.shape)

    # print (RealLocationTimeData)
    # print (RealLocationTimeData.shape)

    LocationNumber = RealLocationTimeData.shape[0]
    InputTimeLength = InputRealLocationTimeData.shape[2]
    NumberOfSamples = RealLocationTimeData.shape[1]

    for i in range(0,LocationNumber):
        fig, ax = plt.subplots(figsize=(15, 6))
        count = 0

        for n in range(0, NumberOfSamples):
            # if n != 0:
            #     continue

            x = [t+count for t in range(0,InputTimeLength)]
            y = InputRealLocationTimeData[i][n]
            x2 = [InputTimeLength+count-1, InputTimeLength+count+horizon-1]
            y2 = [InputRealLocationTimeData[i][n][-1], RealLocationTimeData[i][n]]
            x3 = [InputTimeLength+count-1, InputTimeLength+count+horizon-1]
            y3 = [InputRealLocationTimeData[i][n][-1], PredictedLocationTimeData[i][n]]

            # print (x)
            # print (y)
            # print (x2)
            # print (y2)
            # print (x3)
            # print (y3)

            if count == 0:
                ax.plot(x2,y2,color='tab:blue', label="Ground Truth - Output", marker='.', markersize=5, linestyle='dashed', alpha=0.5)
                ax.plot(x3,y3,color='red', label="Predicted - Output", marker='.', markersize=5, linestyle='dashed')
                ax.plot(x,y,color='navy', label="Ground Truth - Input", marker='.', markersize=5, alpha=0.5)
                ax.axvline(x[-1], color='gray', lw=0.25)
                # ax.vlines(x[-1], linestyles='dashed', colors='red', lw=0.5)
            else:
                ax.plot(x2,y2,color='tab:blue', marker='.', markersize=5, linestyle='dashed', alpha=0.5)
                ax.plot(x3,y3,color='red', marker='.', markersize=5, linestyle='dashed')
                ax.plot(x,y,color='navy', marker='.', markersize=5, alpha=0.5)
                ax.axvline(x[-1], color='gray', lw=0.25)

            count = count+1

        ax.set_xlim([-1, InputTimeLength+NumberOfSamples+horizon-1])
        plt.legend()
        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza activity level")
        FigureType="Input_Prediction_Groundtruth"
        plt.savefig(save_dir+FigureType+"-"+str(i+1)+".pdf",transparent=True,bbox_inches='tight')
        plt.close()

def PlotPredictionTrends(RealLocationTimeData, PredictedLocationTimeData, save_dir):
    LocationNumber = RealLocationTimeData.shape[0]
    TimeLength = RealLocationTimeData.shape[1]

    for i in range(0,LocationNumber):
        x = [count for count in range(0,TimeLength)]
        y = RealLocationTimeData[i]
        y2 = PredictedLocationTimeData[i]

        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(x,y,color='navy',label="Ground Truth", marker='.', markersize=5)
        ax.plot(x,y2,color='red',label="Prediction", linestyle='dashed', marker='.', markersize=5)

        for t in x:
            ax.axvline(t, color='gray', lw=0.25)

        # set the limits
        ax.set_xlim([-1, TimeLength])
        # ax.set_ylim([0, 1])
        plt.legend()
        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("Influenza activity level")
        FigureType="PredictionAndGroundtruth"
        plt.savefig(save_dir+FigureType+"-"+str(i+1)+".pdf",transparent=True,bbox_inches='tight')
        plt.close()

def PlotParameters(BetaList, GammaList, save_dir):
    LocationNumber = BetaList.shape[0]
    TimeLength = BetaList.shape[1]

    for i in range(0, LocationNumber):
        x = [count for count in range(0,TimeLength)]
        Beta = BetaList[i]
        Gamma = GammaList[i]
        R0 = Beta/Gamma

        # ---- Predict the beta and gamma
        fig, ax = plt.subplots(figsize=(15, 6))
        ax2 = ax.twinx()
        ax.plot(x, Beta,color='slateblue',label="Beta", marker='.', markersize=5)
        ax2.plot(x, Gamma,color='plum',label="Gamma", marker='.', markersize=5)
        # ax2.plot(x,R0,color='black',label="$R_0$", marker='.', markersize=5)

        for t in x:
            ax.axvline(t, color='gray', lw=0.25)

        # set the limits
        ax.set_xlim([-1, TimeLength])
        # ax.set_ylim([0, 1])
        ax.legend(loc=2)
        ax2.legend(loc=1)

        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("Beta")
        ax2.set_ylabel("Gamma")

        FigureType="PredictedEpiParameters"
        plt.savefig(save_dir+FigureType+"-"+str(i+1)+".pdf",transparent=True,bbox_inches='tight')
        plt.close()
        
        # ---- Predict the R0
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(x, R0,color='black',label="$R_0$", marker='.', markersize=5)
        for t in x:
            ax.axvline(t, color='gray', lw=0.25)
        ax.set_xlim([-1, TimeLength])
        ax.legend(loc=1)
        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("$R_0$")
        FigureType="PredictedR0"
        plt.savefig(save_dir+FigureType+"-"+str(i+1)+".pdf",transparent=True,bbox_inches='tight')
        plt.close()

def PlotEachMatrix(Matrix, Type, SaveName, save_dir):
    TimeLength = Matrix.shape[0]

    if Type == "Next Generation Matrix" or "Human Mobility Matrix":
        cmap='Spectral_r'
    else:
        # cmap='Spectral_r'
        cmap='RdBu_r'
        # cmap='viridis'
        # cmap='summer'

    SelectedTimes = [0, 1, 2, 3]

    for t in range(0, TimeLength):
        if t not in SelectedTimes:
            break
        fig, ax = plt.subplots(figsize=(15, 6))
        # vmin=minValue, vmax=maxValue
        # ax = sns.heatmap(data=Matrix,fmt=".2f",cmap=cmap,annot=False,
        #                     square=True, cbar=False, xticklabels=False, yticklabels=False)
        ax = sns.heatmap(data=Matrix[t],fmt=".2f",cmap=cmap,annot=False,
                            square=True, cbar=True, xticklabels=False, yticklabels=False)
        
        # xticklabels=2, yticklabels=2
        # ax = sns.heatmap(HeatMapList[i],fmt=".2f",cmap='RdBu_r',annot=False,vmin=minValue,vmax=maxValue)
        ax.xaxis.tick_top()
        # ax.set_title(Type)
        # plt.xticks(fontsize=3,rotation = 60)
        # plt.yticks(fontsize=3,rotation = 60)
        plt.savefig(save_dir + SaveName + "t-" + str(t) +".pdf", transparent=True, bbox_inches='tight')
        plt.close()

def PlotAllMatrices(Matrix, Type, SaveName, save_dir):
    TimeLength = Matrix.shape[0]

    if Type == "Next Generation Matrix" or "Human Mobility Matrix":
        cmap='Spectral_r'
    else:
        cmap='RdBu_r'
        # cmap='Spectral_r'
        # cmap='viridis'
        # cmap='summer'

    columnNumber = math.ceil(np.sqrt(TimeLength))
    rowNumber = math.ceil(TimeLength/columnNumber)
    r=0
    c=0

    vmin=np.min(Matrix)
    vmax=np.max(Matrix)

    figAll, axAll = plt.subplots(ncols=columnNumber, nrows=rowNumber,figsize=(columnNumber*6,rowNumber*5))

    for t in range(0, TimeLength):
        axEach=axAll[r,c]
        
        sns.heatmap(data=Matrix[t],fmt=".2f", cmap=cmap, annot=False,
                            square=True, cbar=True, xticklabels=False, yticklabels=False,
                            ax = axEach, vmin = vmin, vmax = vmax)
        
        axEach.xaxis.tick_top()
        # xticklabels=2, yticklabels=2
        # ax.set_title(Type)
        # plt.xticks(fontsize=3,rotation = 60)
        # plt.yticks(fontsize=3,rotation = 60)

        if c==columnNumber-1:
            c=0
            r=r+1
        else:
            c=c+1

    for fc in range(c,columnNumber):
        for fr in range(r,rowNumber):
            figAll.delaxes(axAll[fr][fc])
    
    plt.savefig(save_dir + SaveName + "-AllTime" +".pdf", transparent=True, bbox_inches='tight')
    plt.close()

    # Add this function to the bottom of PlotFunc.py

def PlotSigma(SigmaList, save_dir):
    LocationNumber = SigmaList.shape[0]
    TimeLength = SigmaList.shape[1]

    for i in range(0, LocationNumber):
        x = [count for count in range(0,TimeLength)]
        y = SigmaList[i]

        fig, ax = plt.subplots(figsize=(15, 6))
        # Using a distinct color (forestgreen) for Sigma
        ax.plot(x, y, color='forestgreen', label="Sigma (Incubation Rate)", marker='.', markersize=5)

        for t in x:
            ax.axvline(t, color='gray', lw=0.25)

        # set the limits
        ax.set_xlim([-1, TimeLength])
        plt.legend()
        ax.set_title("Location " + str(i+1))
        ax.set_xlabel("Week")
        ax.set_ylabel("Sigma")
        
        FigureType = "PredictedSigma"
        plt.savefig(save_dir + FigureType + "-" + str(i+1) + ".pdf", transparent=True, bbox_inches='tight')
        plt.close()

def PlotMatrix(Matrix, Type, SaveName, save_dir,
               symmetric=True,
               annotate=False):
    """
    Plot publication-quality epidemiological heatmaps.

    Parameters
    ----------
    Matrix : np.ndarray
        2D matrix (NGM, mobility, adjacency, etc.)
    Type : str
        Descriptive title (e.g., 'Next Generation Matrix')
    SaveName : str
        Output filename (without extension)
    save_dir : str
        Directory to save figure
    symmetric : bool
        If True, center color scale at zero (for interaction matrices)
    annotate : bool
        If True, write values inside cells (NOT recommended for large matrices)
    """

    Matrix = np.asarray(Matrix)

    plt.figure(figsize=(6, 5))

    # --- Colormap selection
    if Type in ["Next Generation Matrix", "Human Mobility Matrix"]:
        cmap = 'Spectral_r'
    else:
        cmap = 'RdBu_r'

    # --- Color scaling
    if symmetric:
        vmax = np.max(np.abs(Matrix))
        vmin = -vmax
    else:
        vmin = np.min(Matrix)
        vmax = np.max(Matrix)

    ax = sns.heatmap(
        Matrix,
        cmap=cmap,
        square=True,
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.4,
        linecolor='lightgray',
        annot=annotate,
        fmt=".2f",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={
            'shrink': 0.8,
            'pad': 0.02
        }
    )

    # --- Epidemiology-style formatting
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', which='both', length=0)

    # --- Optional title (commented for paper figures)
    # plt.title(Type, fontsize=13, pad=12)

    # --- Save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, f"{SaveName}.pdf"),
        bbox_inches='tight',
        transparent=True,
        dpi=300
    )
    plt.close()

    