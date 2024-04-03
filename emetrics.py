import numpy as np
import gc
import subprocess

def get_aupr(Y, P):
    if hasattr(Y, 'A'): Y = Y.A
    if hasattr(P, 'A'): P = P.A
    Y = np.where(Y>0, 1, 0)
    Y = Y.ravel()
    P = P.ravel()
    f = open("temp.txt", 'w')
    for i in range(Y.shape[0]):
        f.write("%f %d\n" %(P[i], Y[i]))
    f.close()
    f = open("foo.txt", 'w')
    subprocess.call(["java", "-jar", "auc.jar", "temp.txt", "list"], stdout=f)
    f.close()
    f = open("foo.txt")
    lines = f.readlines()
    aucpr = float(lines[-2].split()[-1])
    f.close()
    return aucpr
# 98545=5*19709
def get_cindex(Y, P, num_batches=5):
    Y = np.asarray(Y)
    P = np.asarray(P)

    batch_size = len(Y) // num_batches
    cindex_sum = 0
    total_pairs = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(Y))

        Y_batch = Y[start_idx:end_idx]
        P_batch = P[start_idx:end_idx]

        diff_Y = Y_batch[:, np.newaxis] - Y
        diff_P = P_batch[:, np.newaxis] - P
        # print(Y_batch.shape)
        # print(diff_Y.shape)
        indicator = np.where(diff_Y > 0, 1, 0)

        pair = np.sum(indicator)
        total_pairs += pair

        summ = np.sum((diff_Y > 0) * (diff_P > 0) + 0.5 * (diff_Y == 0) * (diff_P == 0))
        cindex_sum += summ

    if total_pairs != 0:
        return cindex_sum / total_pairs
    else:
        return 0

'''
def get_cindex(Y, P):
    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ/pair
    else:
        return 0
'''
def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred-y_pred_mean) * (y_obs-y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))
    y_pred_sq = sum((y_pred-y_pred_mean) * (y_pred-y_pred_mean))

    return mult / float(y_obs_sq*y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs-(k*y_pred)) * (y_obs-(k*y_pred)))
    down= sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))

    return 1 - (upp/float(down))

def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1-np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def get_MSE(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum((y_obs-y_pred) * (y_obs-y_pred)) / len(y_obs)

def get_RMSE(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return np.sqrt(sum((y_obs-y_pred) * (y_obs-y_pred)) / len(y_obs))

def spearmanr(y_obs, y_pred):
    diff_pred, diff_obs = y_pred-np.mean(y_pred), y_obs-np.mean(y_obs)
    return np.sum(diff_pred*diff_obs) / np.sqrt(np.sum(diff_pred**2) * np.sum(diff_obs**2))