import torch
import torch.nn as nn

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score
import numpy as np

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    # print ("--------- Evaluate")
    counter = 0
    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]

        if modelName == "SEIRmodel":
            # Unpack 5 values (res, EpiOutput, Beta, Gamma, Sigma)
            output, EpiOutput, _, _, _ = model(X);
        else:
            output = model(X);

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict, output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)

        counter = counter + 1
        
        if torch.__version__ < '0.4.0':
            total_loss += evaluateL2(output * scale , Y * scale).data[0]
            total_loss_l1 += evaluateL1(output * scale , Y * scale).data[0]
        else:
            total_loss += evaluateL2(output * scale , Y * scale).item()
            total_loss_l1 += evaluateL1(output * scale , Y * scale).item()

        n_samples += (output.size(0) * loader.m);

        rmselist=[]
        outputValue = (output * scale).T
        RealValue = (Y * scale).T
        maelist=[]
        for i in range(0,len(outputValue)):
            LengthOfTime=len(outputValue[i])
            rmselist.append(math.sqrt(evaluateL2(outputValue[i], RealValue[i]).item()/LengthOfTime))

    rse = math.sqrt(total_loss / n_samples)
    rae = (total_loss_l1/n_samples)

    predict = predict.data.numpy();
    Ytest = test.data.numpy();

    correlation = 0;
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation;

def train(loader, data, model, criterion, optim,
          batch_size, modelName, Lambda, lambda_t=0.0):
    """
    Training step with explicit PINN loss decomposition:
    - data loss
    - epidemiological (SEIR) loss
    - temporal smoothness regularization
    """
    model.train()

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    epi_loss_sum = 0.0
    n_samples = 0

    for inputs in loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        model.zero_grad()

        # --------------------------------------------
        # Forward pass
        if modelName == "SEIRmodel":
            output, EpiOutput, Beta, Gamma, Sigma = model(X)
        else:
            output = model(X)

        scale = loader.scale.expand(output.size(0), loader.m)

        # --------------------------------------------
        # Data loss
        data_loss = criterion(output * scale, Y * scale)

        # --------------------------------------------
        # Epidemiological (PINN) loss
        epi_loss = torch.tensor(0.0, device=output.device)
        if modelName == "SEIRmodel":
            epi_loss = criterion(EpiOutput * scale, Y * scale)

        # --------------------------------------------
        # Temporal smoothness regularization (PINN-inspired)
        temporal_loss = torch.tensor(0.0, device=output.device)

        if modelName == "SEIRmodel" and lambda_t > 0:
            # enforce smooth evolution of epidemic parameters
            if Beta is not None and Beta.size(0) > 1:
                temporal_loss += torch.mean((Beta[1:] - Beta[:-1]) ** 2)
            if Gamma is not None and Gamma.size(0) > 1:
                temporal_loss += torch.mean((Gamma[1:] - Gamma[:-1]) ** 2)
            if Sigma is not None and Sigma.size(0) > 1:
                temporal_loss += torch.mean((Sigma[1:] - Sigma[:-1]) ** 2)

        # --------------------------------------------
        # Total loss
        total_loss = (
            data_loss
            + Lambda * epi_loss
            + lambda_t * temporal_loss
        )

        # --------------------------------------------
        total_loss.backward()
        optim.step()

        # --------------------------------------------
        # Accumulate statistics
        batch_samples = output.size(0) * loader.m
        total_loss_sum += total_loss.item()
        data_loss_sum += data_loss.item()
        epi_loss_sum += epi_loss.item() if modelName == "SEIRmodel" else 0.0
        n_samples += batch_samples

    # --------------------------------------------
    # Normalize losses
    total_loss_avg = total_loss_sum / n_samples
    data_loss_avg = data_loss_sum / n_samples
    epi_loss_avg = epi_loss_sum / n_samples

    return total_loss_avg, data_loss_avg, epi_loss_avg


def GetPrediction(loader, data, model, evaluateL2, evaluateL1, batch_size, modelName):
    model.eval();
    Y_predict = None;
    Y_true = None;
    X_true = None

    # print ("--------- Get prediction")
    counter = 0
    if modelName == "SEIRmodel":
        BetaList = None
        GammaList = None
        SigmaList = None # Renamed from NGMList

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        
        if modelName == "SEIRmodel":
            # UPDATED unpack for SEIR model
            output, EpiOutput, Beta, Gamma, Sigma = model(X);
        else:
            output = model(X);
        
        counter = counter+1

        if Y_predict is None:
            Y_predict = output.cpu()
            Y_true = Y.cpu()
            X_true = X.cpu()

            if modelName == "SEIRmodel":
                BetaList = Beta.cpu()
                GammaList = Gamma.cpu()
                SigmaList = Sigma.cpu() 
        else:
            Y_predict = torch.cat((Y_predict,output.cpu()))
            Y_true = torch.cat((Y_true, Y.cpu()))
            X_true = torch.cat((X_true, X.cpu()))

            if modelName == "SEIRmodel":
                BetaList = torch.cat((BetaList, Beta.cpu()))
                GammaList = torch.cat((GammaList, Gamma.cpu()))
                SigmaList = torch.cat((SigmaList, Sigma.cpu()))

    scale = loader.scale
    
    # Time * location
    Y_predict = (Y_predict * scale)
    Y_true = (Y_true * scale)
    X_true = (X_true * scale)
    
    Y_predict = Y_predict.detach().numpy()
    Y_true = Y_true.detach().numpy()
    X_true = X_true.detach().numpy()

    if modelName == "SEIRmodel":
        BetaList = BetaList.detach().numpy()
        GammaList = GammaList.detach().numpy()
        SigmaList = SigmaList.detach().numpy()

    if modelName == "SEIRmodel":
        return X_true, Y_predict, Y_true, BetaList, GammaList, SigmaList
    else:
        return X_true, Y_predict, Y_true