# -*- coding: utf-8 -*-
"""
Final Analysis v4.1

Lukas Kostal, 8.1.2024, ICL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.constants as sc
import scipy.signal as ss
import scipy.sparse as sp
import scipy.interpolate as si
import scipy.optimize as so
import scipy.odr as odr


# Gaussian function for fitting
def gauss(x, *param):
    n = int(len(param) / 3)
    if n == 0:
        raise Exception("insufficient number of parameters")

    y = np.zeros(len(x))
    for i in range(0, n):
        A = param[i*3]
        mu = param[i*3 + 1]
        sig = param[i*3 + 2]

        y += A * np.exp(-(x - mu)**2 / (2 * sig**2))

    return y


# Lorentzian function for fitting
def lorentz(x, *param):
    n = int(len(param) / 3)
    if n == 0:
        raise Exception("insufficient number of parameters")

    y = np.zeros(len(x))
    for i in range(0, n):
        A = param[i*3]
        x0 = param[i*3 + 1]
        gam = param[i*3 + 2]

        y += A / (2 * np.pi) * gam / ((x - x0)**2 + (gam / 2)**2)
    return y


# Lorentzian function with linear baseline for fitting
def lorentz_lin(x, *param):
    n = int(len(param) / 5)
    if n == 0:
        raise Exception("insufficient number of parameters")

    y = np.zeros(len(x))
    for i in range(0, n):
        A = param[i*5]
        x0 = param[i*5 + 1]
        gam = param[i*5 + 2]
        m = param[i*5 + 3]
        c = param[i*5 + 4]

        y += m * x + c
        y += A / (2 * np.pi) * gam / ((x - x0)**2 + (gam / 2)**2)

    return y


# function for fitting signal amplitude against power
def amp_fit(x, a, sat):
    s = x / sat
    y = a * s / 2 / (s + 1)
    return y

# function for fitting peak width against power
def gam_fit(x, gam, sat):
    y = gam * np.sqrt(1 + 2 * x / sat)
    return y


# function to calculate reduced chi squared statistic
def get_chi(o, sig, e, dof):
    with np.errstate(all='ignore'):
        chi_arr = (o - e)**2 / (sig)**2

    chi_arr = chi_arr[~np.isinf(chi_arr)]
    chi_arr = chi_arr[~np.isnan(chi_arr)]
    chi = np.sum(chi_arr) / (len(chi_arr) - dof)

    return chi


# function to get power to be analysed and pump beam power in uW
def get_power(mes_num):

    # load power measurement file
    df = pd.read_csv(f'Data/Mes{mes_num}_pwr.csv', comment='#')
    power = df.iloc[:, 0].to_numpy()
    P = df.iloc[:, 1].to_numpy()
    r = df.iloc[0, 3] / df.iloc[0, 2]

    # get pump power and associated error in uW
    P_pmp = r * P
    Perr_pmp = np.sqrt(3) * 0.02 * P_pmp

    # get probe power and associated error in uW
    P_prb = df.iloc[0, 4]
    Perr_prb = P_prb * 0.02

    return power, P_pmp, Perr_pmp, P_prb, Perr_prb


# function to load measurement data
def get_data(mes_num, power):

    # load Doppler-free, Doppler broadened, Fabry-Perot Etalon data
    data_dfs = np.loadtxt(f'Data/Mes{mes_num}_dfs_{power}.csv', delimiter=',', unpack=True, skiprows=1)
    data_dbs = np.loadtxt(f'Data/Mes{mes_num}_dbs_{power}.csv', delimiter=',', unpack=True, skiprows=1)
    data_fpe = np.loadtxt(f'Data/Mes{mes_num}_fpe_{power}.csv', delimiter=',', unpack=True, skiprows=1)

    # slice data into arrays
    t = data_dfs[0, :]
    trig = data_dfs[2, :]
    V_dfs = data_dfs[1, :]
    V_dbs = data_dbs[1, :]
    V_fpe = data_fpe[3, :]

    # offset measurement time to start at 0s
    t -= t[0]

    # offset trigger signal to 0V and normalise
    trig = trig - np.amin(trig) - np.ptp(trig) / 2
    trig /= np.ptp(trig) / 2

    return t, trig, V_dfs, V_dbs, V_fpe


# function to slice measurement data to spectrum
def get_spec(t, trig, V_dfs, V_dbs, V_fpe, motion='fwd'):

    # differentiate trigger signal and get indices of rising and falling edges
    idx_rise, _ = ss.find_peaks(np.diff(trig), prominence=1)
    idx_fall, _ = ss.find_peaks(-np.diff(trig), prominence=1)

    # array of indices at which to slice spectra
    idx_slc = np.concatenate((idx_rise, idx_fall))
    idx_slc = np.sort(idx_slc)
    idx_slc = idx_slc[:-1] + np.diff(idx_slc) / 2
    idx_slc = idx_slc.astype(int)

    # get number of spectra in the measurement
    n_spec = len(idx_slc) - 1

    # lowest number of samples in spectra
    n_samp = np.amin(np.diff(idx_slc))

    # array to hold sciled spectra
    V_dfs_arr = np.zeros((n_spec, n_samp))
    V_dbs_arr = np.zeros((n_spec, n_samp))
    V_fpe_arr = np.zeros((n_spec, n_samp))
    t_arr = np.zeros((n_spec, n_samp))

    # boolean to indicate if starts with rising edge for flipping
    rise = idx_rise[0] < idx_fall[0]

    # loop over all spectra and slice waveform
    for i in range(0, n_spec):
        V_dfs_arr[i, :] = V_dfs[idx_slc[i]:idx_slc[i] + n_samp]
        V_dbs_arr[i, :] = V_dbs[idx_slc[i]:idx_slc[i] + n_samp]
        V_fpe_arr[i, :] = V_fpe[idx_slc[i]:idx_slc[i] + n_samp]
        t_arr[i, :] = t[idx_slc[i]:idx_slc[i] + n_samp] - t[idx_slc[i]]

        # if starts with rising edge flip starting at 0
        if (rise == True) and (i % 2 == 0):
            V_dfs_arr[i] = np.flip(V_dfs_arr[i])
            V_dbs_arr[i] = np.flip(V_dbs_arr[i])
            V_fpe_arr[i] = np.flip(V_fpe_arr[i])

        # if starts with falling edge flip starting at 1
        if (rise == False) and (i % 2 == 1):
            V_dfs_arr[i] = np.flip(V_dfs_arr[i])
            V_dbs_arr[i] = np.flip(V_dbs_arr[i])
            V_fpe_arr[i] = np.flip(V_fpe_arr[i])

    # get boolean of motion corresponding to piezo going forwards
    motion_bool = motion == 'fwd'

    if rise == motion_bool:
        init_idx = 0
    elif rise != motion_bool:
        init_idx = 1

    # average each 2nd spectrum starting at init_idx
    # corresponds to piezo moving forwards or reverse
    V_dbs = np.mean(V_dbs_arr[init_idx:-1:2, :], axis=0)
    V_dfs = np.mean(V_dfs_arr[init_idx:-1:2, :], axis=0)
    V_fpe = np.mean(V_fpe_arr[init_idx:-1:2, :], axis=0)
    t = np.mean(t_arr[init_idx:-1:2, :], axis=0)

    return t, V_dfs, V_dbs, V_fpe


# function for calculating baseline using assymetric least squares smoothing
# lambda is smoothing parameter, p is assymetric weighting parameter
def get_baseline(y, lmbd=1e4, p=5e-4, n=10):
    # get no of datapoints
    ny = len(y)

    # create 2nd order difference matrix
    d_mat = sp.diags([1,-2,1], [0,-1,-2], shape=(ny, ny-2))
    d_mat = lmbd * d_mat * np.transpose(d_mat)

    # create weighting vector
    w_vec = np.zeros(ny)

    # create weighting matrix from weighting vectors
    w_mat = sp.spdiags(w_vec, 0, ny, ny)

    for i in range(0, n):
        # update the difference matrix with wight vector
        w_mat.setdiag(w_vec)

        # penalised least sqaure matrix
        pls_mat = w_mat + d_mat

        # calculate new baseline point
        z = sp.linalg.spsolve(pls_mat, w_vec * y)

        # update the weighting vector
        w_vec = p * (y > z) + (1 - p) * (y < z)

    return z


# function to get frequency calibration with linear interpolation
def get_fcal(t, A_fpe, l, lerr, f0=0):

    # free spectral range of the FPE note take 8 peaks so 9 gaps
    fsr = sc.c / (2 * 9 * l)
    fsr_err = sc.c / (2 * 9 * l) * lerr / l

    # detect peaks in the absorbance spectrum
    pks, _ = ss.find_peaks(A_fpe, width=10, distance=10, prominence=0.2)

    # find array of indices of peak edges at FWHM
    wds = ss.peak_widths(A_fpe, pks, rel_height=0.5)
    wds = np.array(wds, int)

    # gwt approximate FWHM for fitting
    fwhm = t[wds[3, :]] - t[wds[2, :]]

    # arrays to hold data for each FPE peak
    t_pks = np.zeros(len(pks))
    terr_pks = np.zeros(len(pks))
    fin_pks = np.zeros(len(pks))

    # loop over all detected FPE peaks
    for i in range(0, len(pks)):

        # get initial and final index for slicing FPE peak
        idx_i = np.argmin(np.abs(t - (t[pks[i]] - fwhm[i])))
        idx_f = np.argmin(np.abs(t - (t[pks[i]] + fwhm[i])))

        # slice the scan time and FPE amplitude
        t_fit = t[idx_i : idx_f]
        A_fit = A_fpe[idx_i : idx_f]

        # array of initial guess
        ig = np.array([np.amax(A_fit), t_fit[np.argmax(A_fit)], fwhm[i]])

        # perform Lorentzian and get expected uncertainty on fit parameters
        popt, pcov = so.curve_fit(lorentz, t_fit, A_fit, p0=ig)
        perr = np.sqrt(np.diag(pcov))

        # write the fit parameters and uncertainties to arrays
        t_pks[i] = popt[1]
        terr_pks[i] = perr[1]
        fin_pks[i] = popt[2]

    # array of peak frequencies by multiplying interference order by FSR in Hz
    f_pks = np.arange(0, len(pks)) * fsr

    # array of min and max peak frequencies due to error in FSR
    f_pks_min = np.arange(0, len(pks)) * (fsr - fsr_err)
    f_pks_max = np.arange(0, len(pks)) * (fsr + fsr_err)

    # perform linear interpolation on peak frequency against scan time
    li = si.interp1d(t_pks, f_pks, fill_value='extrapolate')

    # min and max linear interpolation due to error in frequency and fit
    li_min = si.interp1d(t_pks + terr_pks, f_pks_min, fill_value='extrapolate')
    li_max = si.interp1d(t_pks - terr_pks, f_pks_max, fill_value='extrapolate')

    # perform frequency calibration and expected error in Hz
    f = li(t)
    ferr = li_max(t) - li_min(t)

    # shift relative frequency to start at f0
    f += f0 - f[0]

    # get average peak fwhm and error on mean and convert them to FWHM
    fwhm = np.mean(fin_pks) * np.ptp(f) / np.ptp(t)
    fwhm_err = np.ptp(fin_pks) * np.ptp(f) / np.ptp(t)

    # calcualte finesse from fwhm
    fin = fsr / fwhm
    fin_err = np.sqrt((fsr_err / fsr)**2 + (fwhm_err / fwhm)**2)

    return f, ferr, fin, fin_err


# function to get error on amplitude as the peak to peak of the baseline noise
def get_noi(f, A, f_noi):

    # get index for frequency past which the baseline is taken
    idx_noi = np.argmin(np.abs(f - f_noi))

    # get noise level as ptp of the baseline
    noi = np.ptp(A[idx_noi:])

    return noi


# --------------------------------------------------------------------------- #

# measurement number to analyse
mes_num = 10

# phase of motion of piezo to analyse
motion = 'fwd'

# length of FPE cavity and associated error in m
l = 194 * 1e-3
lerr = 2 * 1e-3

# --------------------------------------------------------------------------- #

# get measurement power and power of pump beam in uW
power, P_pmp, Perr_pmp, P_prb, Perr_prb = get_power(mes_num)

# get total incident power and associated error in uW
P_tot = P_pmp + P_prb
Perr_tot = np.sqrt(Perr_pmp**2 + Perr_prb**2)

# number of power measurements
nP = len(P_tot)

# arrays to hold data for each spectrum
f_arr = np.zeros(nP, dtype=object)
A_arr = np.zeros(nP, dtype=object)
ferr_arr = np.zeros(nP, dtype=object)
Aerr_arr = np.zeros(nP, dtype=object)

# array to hold peak to peak value used for normalisation
ptp_arr = np.zeros(nP)

# loop over all measured powers
for i in range(0, nP):

    # load the data and slice into arrays
    t, trig, V_dfs, V_dbs, V_fpe = get_data(mes_num, power[i])

    # slice the spectra and average
    t, V_dfs, V_dbs, V_fpe = get_spec(t, trig, V_dfs, V_dbs, V_fpe, motion)

    # get hyperfine resolved spectrum
    A = V_dbs - V_dfs

    # get normalised absorbance from hyperfine resolved spectrum
    ptp_arr[i] = np.ptp(A)
    A = (np.amax(A) - A) / np.ptp(A)

    # get normalised Fabry-Perot Etalon spectrum
    A_fpe = (V_fpe - np.amin(V_fpe)) / np.ptp(V_fpe)

    # perform frequency calibration in GHz
    f, ferr, fin, fin_err = get_fcal(t, A_fpe, l, lerr)

    # get expected error in signal as std of noise past 8GHz
    noi = get_noi(f, A, 8e9)

    # write calibrated frequency and expected error to arrays in GHz
    f_arr[i] = f * 1e-9
    ferr_arr[i] = ferr * 1e-9

    # write normalosed absorbance and associated error to arrays
    A_arr[i] = A
    Aerr_arr[i] = np.ones(len(A)) * noi

# array of titles for plotting Doppler-free spectra
tit_arr = np.array([r'$^{87}$Rb $f=2$', r'$^{85}$Rb $f=3$', r'$^{85}$Rb $f=2$', r'$^{87}$Rb $f=1$'])

# load file with initial frequencies in GHz for slicing Doppler broadened peaks
df = pd.read_csv(f'Load/Mes{mes_num}_slice.csv', comment='#')
fi_arr = df.iloc[:, 1:5].to_numpy()
# array with width in GHz for slicing Doppler broadened peaks
fw_arr = np.array([0.7, 0.4, 0.3, 0.6])

# arrays to hold amplitude and width of peaks for different powers
amp_arr = np.zeros(nP)
gam_arr = np.zeros(nP)
amp_err = np.zeros(nP)
gam_err = np.zeros(nP)

# arrays to hold relative frequency of hyperfine peaks and associated error
f_pks = np.zeros((4, 3))
ferr_pks = np.zeros((4, 3))

# loop over all measured powers
for i in range(0, nP):

    # get baseline over entire spectrum for subtracting
    A_bl = get_baseline(A_arr[i], lmbd=1e3, p=1e-9)

    # get plots and detuning frequencies from measurement at power p9
    if i==8:

        # pick out arrays of data for plotting
        f_plt = f_arr[i]
        A_plt = A_arr[i]

        # plot entire Doppler-free spectrum
        plt.figure(1)
        plt.title(r'Rb $D_2$ Line Doppler-free Spectrum')
        plt.xlabel(r'relative frequency $\nu$ ($GHz$)')
        plt.ylabel(r'Doppler-free signal $A$ ($a.u.$)')
        plt.rc('grid', linestyle=':', color='black', alpha=0.8)
        plt.grid()

        plt.plot(f_plt, A_plt, c='b')

        # set axis limit and aspect ratio
        plt.xlim(0, 8)
        plt.ylim(0, 1.2)
        ax = plt.gca()
        rat = np.ptp(ax.get_xlim()) / np.ptp(ax.get_ylim())
        ax.set_aspect(rat * 0.6)

        plt.savefig('Output/spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()

    # loop over all 4 Doppler broadened peaks
    for j in range(0, 4):

        # get initial and final index for slicing Doppler broadened peaks
        idx_i = np.argmin(np.abs(f_arr[i] - fi_arr[i, j]))
        idx_f = np.argmin(np.abs(f_arr[i] - (fi_arr[i, j] + fw_arr[j])))

        # slice spectrum into 4 Doppler broadened peaks
        f_crp = f_arr[i][idx_i:idx_f]
        ferr_crp = ferr_arr[i][idx_i:idx_f]
        A_crp = A_arr[i][idx_i:idx_f]
        Aerr_crp = Aerr_arr[i][idx_i:idx_f]

        # get sliced spectrum with removed baseline
        A_rbs = A_crp - A_bl[idx_i:idx_f]

        # use the 87Rb f=2 initial state for pump power analysis
        if j==0:

            # use the last Doppler-free peak corresponding to f'=3 for pump power analysis
            idx_fit = -1

            # multiple of FWHM to specify interval for fitting
            fact = 2

            # find Doppler-free peaks and get index of peak to analyse
            pks, _ = ss.find_peaks(A_crp, width=10, height=1e-2, prominence=2e-2)
            pkfit = pks[idx_fit]

            # correction since at lowest power find_peaks detects noise
            if i==13:
                idx_fit = -11

            # find array of indices of peak edges at FWHM
            wds = ss.peak_widths(A_crp, pks, rel_height=0.5)
            wds = np.array(wds, int)

            # get FWHM of peak to fit
            fwhm = f_crp[wds[3, idx_fit]] - f_crp[wds[2, idx_fit]]

            # initial and final index for slicing peak to fit
            idx_i = np.argmin(np.abs(f_crp - (f_crp[pkfit] - fact * fwhm)))
            idx_f = np.argmin(np.abs(f_crp - (f_crp[pkfit] + fact * fwhm)))

            # slice peak for fitting
            f_fit = f_crp[idx_i : idx_f]
            ferr_fit = ferr_crp[idx_i : idx_f]
            A_fit = A_crp[idx_i : idx_f]
            Aerr_fit = Aerr_crp[idx_i : idx_f]

            # array of initial values for fitting
            ig = np.array([A_crp[pkfit], f_crp[pkfit], fwhm, 1e-2, 1e-2])

            # perform Lorentzian fit with linear baseline and get error on fitting parameters
            popt, pcov = so.curve_fit(lorentz_lin, f_fit, A_fit, p0=ig, sigma=Aerr_fit, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))

            # calculate peak amplitude and associated error and write to array
            amp_arr[i] = popt[0] / (2 * np.pi * popt[2])
            amp_err[i] = popt[0] / (2 * np.pi * popt[2]) * np.sqrt((perr[0] / popt[0])**2 + (perr[2] / popt[2])**2)

            # get peak FWHM from fit parameter and associated error and write to array
            gam_arr[i] = popt[2]
            gam_err[i] = perr[2]


        # get plots and detuning frequencies from measurement at power p9
        if i==8:

            # multiple of FWHM to specify interval for fitting
            fact = 1

            # find all Doppler-free peaks
            pks, _ = ss.find_peaks(A_crp, width=8, height=5e-2, prominence=2e-3)

            # correction since in 4th Doppler broadened peak 1st crossover peak is not detected
            if j==3:
                pks = np.array([pks[0], pks[0] + (pks[1] - pks[0]) / 2, pks[1], pks[2], pks[3], pks[4]], dtype=int)

            # pick out Doppler-free peaks not crossovers
            pks_fit = np.array([True, False, True, False, False, True])
            cpks = pks[~pks_fit]
            pks = pks[pks_fit]

            # find array of indices of peak edges for FWHM
            wds = ss.peak_widths(A_crp, pks, rel_height=0.5)
            wds = np.array(wds, int)

            # loop over all Doppler-free peaks to find relative frequency
            for k in range(0, 3):

                # get index of Doppler-free peak to fit
                pkfit = pks[k]

                # get FWHM of peak to fit
                fwhm = f_crp[wds[3, idx_fit]] - f_crp[wds[2, idx_fit]]

                # initial and final index for slicing peak to fit
                idx_i = np.argmin(np.abs(f_crp - (f_crp[pkfit] - fact * fwhm)))
                idx_f = np.argmin(np.abs(f_crp - (f_crp[pkfit] + fact * fwhm)))

                # slice peak for fitting
                f_fit = f_crp[idx_i : idx_f]
                ferr_fit = ferr_crp[idx_i : idx_f]
                A_fit = A_crp[idx_i : idx_f]
                Aerr_fit = Aerr_crp[idx_i : idx_f]

                # array of initial values for fitting
                ig = np.array([A_crp[pkfit], f_crp[pkfit], fwhm, 1e-2, 1e-2])

                # fit Lorentzian with linear baseline and get error on fitting parameters
                popt, pcov = so.curve_fit(lorentz_lin, f_fit, A_fit, p0=ig, sigma=Aerr_fit, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))

                # write relative frequency and associated error in GHz to array
                f_pks[j, k] = popt[1]
                ferr_pks[j, k] = perr[1]

            # shift relative frequency to start at 0 and convert to MHz
            f_plt = (f_crp - np.amin(f_crp))
            A_plt = A_crp

            # plot Doppler-free spectra for each individual set of final hyperfine levels
            plt.figure(1)
            plt.title('Doppler-free Spectrum \t' + tit_arr[j] + r'$\rightarrow f^\prime$')
            plt.xlabel(r'relative frequency $\nu$ ($GHz$)')
            plt.ylabel(r'Doppler-free signal $A$ ($a.u.$)')
            plt.rc('grid', linestyle=':', color='black', alpha=0.8)
            plt.grid()

            plt.plot(f_plt, A_plt, c='b')
            plt.plot(f_plt[pks], A_plt[pks], 'x', c='r', label='transition')
            plt.plot(f_plt[cpks], A_plt[cpks], 'x', c='g', label='crossover')

            plt.legend(loc=1)

            # set aspect ratio
            ax = plt.gca()
            rat = np.ptp(ax.get_xlim()) / np.ptp(ax.get_ylim())
            ax.set_aspect(rat * 0.4)

            plt.savefig(f'Output/spectrum_{j}.png', dpi=300, bbox_inches='tight')
            plt.show()

# get detuning frequencies and associated error in GHz
f_det = np.diff(f_pks, axis=1)
ferr_det = np.array([ferr_pks[:, 0]**2 + ferr_pks[:, 1]**2, ferr_pks[:, 1]**2 + ferr_pks[:, 2]**2])
ferr_det = np.transpose(np.sqrt(ferr_det))

# function for fitting peak amplitude against power using ODR
amp_odr = lambda beta, x : amp_fit(x, beta[0], beta[1])

# perform fit of peak amplitude using ODR
amp_mod = odr.Model(amp_odr)
amp_dat = odr.RealData(P_tot, amp_arr*ptp_arr, sx=Perr_tot , sy=amp_err)
amp_odr = odr.ODR(amp_dat, amp_mod, beta0=[1e-3, 100])
amp_out = amp_odr.run()
pamp = np.array(amp_out.beta)
camp = np.array(amp_out.cov_beta)
eamp = np.array(np.diag(camp))

# function for fitting peak FWHM against power using ODR
gam_odr = lambda beta, x : gam_fit(x, beta[0], beta[1])

# perform fit of peak FWHM using ODR
gam_mod = odr.Model(gam_odr)
gam_dat = odr.RealData(P_tot, gam_arr, sx=Perr_tot, sy=gam_err)
gam_odr = odr.ODR(gam_dat, gam_mod, beta0=[1e-3, 100])
gam_out = gam_odr.run()
pgam = np.array(gam_out.beta)
cgam = np.array(gam_out.cov_beta)
egam = np.sqrt(np.diag(cgam))

# array of expected values for peak amplitude and FWHM
amp_exp = amp_fit(P_tot, *pamp)
gam_exp = gam_fit(P_tot, *pgam)

# calculate reduced chi-sqaured values for peak ampltude and FWHM
amp_chi = get_chi(amp_arr*ptp_arr, amp_err*ptp_arr, amp_exp, 2)
gam_chi = get_chi(gam_arr, gam_err, gam_exp, 2)

# print numerical results
print('peak amplitude curve fit parameters:')
print(f'Psat  = {pamp[1]:.4g} ± {eamp[1]:.4g}')
print(f'chi   = {amp_chi:.4g}')
print()
print('peak width curve fit parameters:')
print(f'Psat  = {pgam[1]:.4g} ± {egam[1]:.4g}')
print(f"Gamma = {pgam[0]*1e3:.4g} ± {egam[0]*1e3:.4g} MHz")
print(f'chi   = {gam_chi:.4g}')
print()
print('frequenct differences between hyperfine states:')
print(f'measurement power: {power[8]}')
print(f'Ppump {P_pmp[8]:.4g} ± {Perr_pmp[8]:.4g} uW')

for i in range(0, len(f_det[:,0])):
    for j in range(0, len(f_det[0,:])):
        print(f'f [{i:.0f},{j:,.0f}] = {f_det[i,j]*1e3:.5g} ± {ferr_det[i,j]*1e3:.4g} MHz')

# array of total power for plotting fitted curves
P_plt = np.arange(0, np.amax(P_tot))

# plot peak aplitude against total incident power
plt.figure(1)
plt.title('Peak Amplitude against Incident Power')
plt.xlabel(r'total incident power $P_{tot}$ ($\mu W$)')
plt.ylabel(r'peak amplitude $A_0$ ($a.u.$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.errorbar(x=P_tot, y=amp_arr * ptp_arr, xerr=Perr_tot, yerr=amp_err * ptp_arr, fmt='x', capsize=4, c='b', label='data')
plt.plot(P_plt, amp_fit(P_plt, *pamp), c='r', label='fit')

plt.legend(loc=4)

plt.savefig('Output/amplitude.png', dpi=300, bbox_inches='tight')
plt.show()

# plot peak FWHM against total incident power
plt.figure(2)
plt.title('Broadened Peak FWHM against Incident Power')
plt.xlabel(r'total incident power $P_{tot}$ ($\mu W$)')
plt.ylabel(r'broadened FWHM $\Delta$ ($MHz$)')
plt.rc('grid', linestyle=':', color='black', alpha=0.8)
plt.grid()

plt.errorbar(x=P_tot[:-2], y=gam_arr[:-2]*1e3, xerr=Perr_tot[:-2], yerr=gam_err[:-2]*1e3, fmt='x', capsize=4, c='b', label='data')
plt.plot(P_plt, gam_fit(P_plt, *pgam)*1e3, c='r', label='fit')

plt.ylim(2, 13)

plt.legend(loc=4)

plt.savefig('Output/width.png', dpi=300, bbox_inches='tight')
plt.show()
