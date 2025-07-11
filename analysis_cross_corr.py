import numpy as np
import sys
import iaaft
from scipy.signal import find_peaks

data_list = [404,310,978] # label of samples
window_size = 1200        # window size
window_step = 10          # window step
lag_max = 200             # maximum lag value
prominence = 0.10         # threhsold of prominence
tolerance=10              # tolerance of symmetry of tau_xy and tau_ys 

I = int(sys.argv[1])

fp0 =open('cross_max_values_%.3d_ws_%.4d_delay_%.3d_p_%.2f.dat' % (data_list[I],window_size,lag_max,prominence),'w')
MUA = np.loadtxt('data_MUA_%.3d.dat' % data_list[I])
ENV = np.loadtxt('data_ENV_%.3d.dat' % data_list[I])



def preprocess_time_series(X, Y):
    """
    Preprocess two time series by ensuring they have the same length
    and handling missing values.

    Parameters:
    X (np.ndarray): First time series.
    Y (np.ndarray): Second time series.

    Returns:
    tuple: Preprocessed X and Y.
    """
    # Ensure both series are 1-D
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    
    # Find the minimum length
    min_length = min(len(X), len(Y))
    
    # Truncate both series to the minimum length
    X = X[:min_length]
    Y = Y[:min_length]
    
    # Handle missing values by removing any pairs with NaNs
    valid_indices = ~np.isnan(X) & ~np.isnan(Y)
    X = X[valid_indices]
    Y = Y[valid_indices]
    
    return X, Y

def create_lagged_variables(X,Y, lag,lag_max):
    """
    Create lagged versions of the time series.

    Parameters:
    X (np.ndarray): Preprocessed first time series.
    Y (np.ndarray): Preprocessed second time series.
    lag (int): Number of lag steps.

    Returns:
    tuple: Arrays of Y_t, Y_{t-lag}, and X_{t-lag}
    """

    # Ensure there are enough data points
    if len(X) <= lag or len(Y) <= lag:
        raise ValueError("Time series too short for the specified lag.")
    
    X_t = X[lag_max:-lag_max]
    Y_t = Y[lag_max+lag:-lag_max+lag]    
    if len(X_t) != len(Y_t):
        raise ValueError("lag should be updated.")
    return X_t,Y_t

# Function to evaluate cross_correlation
def evaluate_cross_correlation(X, Y, lag,lag_max):
    X_pre, Y_pre = preprocess_time_series(X, Y)
    # Step 2: Create lagged variables
    X_t, Y_t = create_lagged_variables(X_pre, Y_pre, lag=lag,lag_max=lag_max)
    cross = np.corrcoef(X_t, Y_t)[0, 1]
    return cross

# Function to extract overlapping windows
def extract_windows(data, window_size, window_step):
    windows = []
    for start in range(0, len(data) - window_size + 1, window_step):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)

def find_peak_near_zero(cc, lags, prominence_threshold=0.1):
    """
    Find the peak (local maximum) closest to zero lag.

    Parameters:
    - cc: cross-correlation array
    - lags: corresponding lag array
    - prominence_threshold: minimum prominence to consider a peak

    Returns:
    - peak_lag: lag value of the peak closest to zero
    - peak_val: cross-correlation value at that peak
    """
    # Ensure inputs are numpy arrays
    cc = np.asarray(cc)
    lags = np.asarray(lags)

    # Find all peaks with given prominence
    peaks, properties = find_peaks(cc, prominence=prominence_threshold)

    if len(peaks) == 0:
        print("No significant peaks found.")
        return None, None

    # Get lags and values of peaks
    peak_lags = lags[peaks]
    peak_vals = cc[peaks]

    # Find the peak whose lag is closest to zero
    idx = np.argmin(np.abs(peak_lags))
    peak_lag = peak_lags[idx]
    peak_val = peak_vals[idx]

    print(f"Peak closest to zero: lag = {peak_lag}, cc = {peak_val}")

    return peak_lag, peak_val

def check_cross_corr(peak_val_xy, peak_lag_xy, peak_val_yx, peak_lag_yx, tolerance=10):
    """
    Selects the (C, Tau) pair based on the proximity of lags and the higher C value.

    Parameters:
    peak_val_xy (float): Cross-correlation value C_XY at lag peak_lag_xy.
    peak_lag_xy (int): Lag value for peak C_XY.
    peak_val_yx (float): Cross-correlation value C_YX at lag peak_lag_yx.
    peak_lag_yx (int): Lag value for peak C_YX.
    tolerance (int): Maximum difference between lags to consider them close.

    Returns:
    (float, int) or None: Returns (C, Tau) with higher C, or None if lags are too different.
    """

    if abs(peak_lag_xy - peak_lag_yx) <= tolerance:
        if peak_val_xy >= peak_val_yx:
            return peak_val_xy, peak_lag_xy
        else:
            return peak_val_yx, peak_lag_yx
    else:
        return None  # Lags are too different



# Extract the windows from the data
windows_MUA = extract_windows(MUA, window_size, window_step)
windows_ENV = extract_windows(ENV, window_size, window_step)

# Surrogated dataset #1
# Estimate 1000 surrogates from shuffled windows
Ns = 1000
Surr_max = np.zeros(Ns)   
Surr_min = np.zeros(Ns)   

for j in range(Ns):
    surr_cc = []
    Xs = windows_MUA[np.random.randint(0, len(windows_MUA))]
    Ys = windows_ENV[np.random.randint(0, len(windows_ENV))]
    for l in range(-lag_max,lag_max):
        CROSS_XY = evaluate_cross_correlation(Xs,Ys, lag=l,lag_max=lag_max)
        surr_cc.append(CROSS_XY)
    surr_cc = np.array(surr_cc)
    Surr_max[j] = np.max(surr_cc)
    Surr_min[j] = np.min(surr_cc)

mu_2_max = np.mean(Surr_max)
std_2_max = np.std(Surr_max)
mu_2_min = np.mean(Surr_min)
std_2_min = np.std(Surr_min)

for i in range(len(windows_ENV)):
    X = windows_MUA[i]
    Y = windows_ENV[i]
    
    # Surrogated dataset #2
    # Estimate 100 IAAFT surrogates from the original window
    ns = 100
    surr_max = np.zeros(ns)
    surr_min = np.zeros(ns)

    XS = iaaft.surrogates(x=X, ns=ns, verbose=False,seed = 1000*i)
    YS = iaaft.surrogates(x=Y, ns=ns, verbose=False,seed = 9999*i)
    for j in range(ns):
        surr_cc = []
        Xs = XS[j,:]
        Ys = YS[j,:]
        for l in range(-lag_max,lag_max):
            CROSS_XY = evaluate_cross_correlation(Xs,Ys, lag=l,lag_max=lag_max)
            surr_cc.append(CROSS_XY)
        surr_cc = np.array(surr_cc)
        surr_max[j] = np.max(surr_cc)
        surr_min[j] = np.min(surr_cc)

    mu_1_max  = np.mean(surr_max)
    std_1_max = np.std(surr_max)
    mu_1_min  = np.mean(surr_min)
    std_1_min = np.std(surr_min)


    #COMPUTE CROSS-CORR
    cc_xy = []
    lags_xy = []
    cc_yx = []
    lags_yx = []
    for l in range(-lag_max,lag_max):
        CROSS_XY = evaluate_cross_correlation(X, Y, lag=l,lag_max=lag_max)
        CROSS_YX = evaluate_cross_correlation(Y, X, lag=l,lag_max=lag_max)
        cc_xy.append(CROSS_XY)
        lags_xy.append(l)
        cc_yx.append(CROSS_YX)
        lags_yx.append(-l)

    peak_lag_xy, peak_val_xy= find_peak_near_zero(cc_xy,lags_xy,prominence)
    peak_lag_yx, peak_val_yx= find_peak_near_zero(cc_yx,lags_yx,prominence)

    result = check_cross_corr(peak_val_xy, peak_lag_xy, peak_val_yx, peak_lag_yx,tolerance)

    if result is None:
        fp0.write('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))
        print('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))
    else:
        C,tau = result
        if C < 0.5:
            fp0.write('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))
            print('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))
        else:
            a1 = int(C > mu_1_max+std_1_max)
            a2 = int(C > mu_2_max+std_2_max)
            if a1 == 1 and a2 == 1:
                fp0.write('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,1))
                print('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,1))
            else:
                fp0.write('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))
                print('%d %d %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %d\n' % (i*window_step+lag_max,peak_lag_xy, peak_val_xy, peak_lag_yx, peak_val_yx,mu_1_max,std_1_max,mu_1_min,std_1_min,mu_2_max,std_2_max,mu_2_min,std_2_min,0))


#    print(I,i*window_step+lag_max,selected_tau,selected_max,mu,std,Mu,Std,a,b,output)
#    fp0.write('%d %lf %lf %d %lf %lf %lf %lf %d %d\n' % (i*window_step+lag_max,selected_max, selected_tau, output,mu,std,Mu,Std,a,b))
