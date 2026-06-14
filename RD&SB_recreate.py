#!/usr/bin/env python3
"""
RD&SB_recreate.py
Cavity ringdown and sideband calibration analysis for a 38 kHz IR laser.

CSV input: column 1 = time (µs), column 2 = signal (mV).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

VC = 38_000  # Hz – 38 kHz IR laser carrier frequency

# ---------------------------------------------------------------------------
# Curve models
# ---------------------------------------------------------------------------

def _exp_decay(t, I0, tau, Boffset):
    return I0 * np.exp(-t / tau) + Boffset


def _lorentzian(t, A, t0, gamma):
    return A / (1.0 + ((t - t0) / (gamma / 2.0)) ** 2)


def _three_lorentzians(t, A1, t1, g1, A2, t2, g2, A3, t3, g3):
    return (
        _lorentzian(t, A1, t1, g1)
        + _lorentzian(t, A2, t2, g2)
        + _lorentzian(t, A3, t3, g3)
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(path):
    col_name = ''
    try:
        df = pd.read_csv(path, header=None, comment='#')
        pd.to_numeric(df.iloc[0, 0])   # raises if first cell is a string header
    except (ValueError, TypeError):
        df = pd.read_csv(path, comment='#')
        col_name = str(df.columns[0]).lower()

    t = df.iloc[:, 0].to_numpy(float)
    s = df.iloc[:, 1].to_numpy(float)

    # Normalise time to µs so all downstream code stays consistent
    if '_ms' in col_name or col_name == 'ms':
        t = t * 1_000.0          # ms → µs
    elif ('_s' in col_name or col_name == 's') and '_ms' not in col_name:
        t = t * 1_000_000.0      # s → µs

    return t, s


# ---------------------------------------------------------------------------
# Ringdown mode
# ---------------------------------------------------------------------------

def ringdown_mode(t, s):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t, s, s=8, alpha=0.5, label='Data')

    # Estimate the dominant oscillation period from the data via FFT
    # (more reliable than assuming the carrier period, which may differ
    # from the observed beat/oscillation frequency in the signal)
    dt_med = np.median(np.diff(t))
    freqs  = np.fft.rfftfreq(len(s), d=dt_med)
    power  = np.abs(np.fft.rfft(s - np.mean(s))) ** 2
    dom_freq = freqs[np.argmax(power[1:]) + 1]      # skip DC bin
    T_est    = 1.0 / dom_freq                        # estimated period (same units as t)
    min_dist = max(int(T_est / dt_med * 0.5), 2)    # ~50 % of one period in samples

    prom_thresh = (np.max(s) - np.min(s)) * 0.1
    peak_idx, _ = find_peaks(s, distance=min_dist, prominence=prom_thresh)
    if len(peak_idx) < 3:
        peak_idx, _ = find_peaks(s, distance=max(min_dist // 2, 1), prominence=prom_thresh * 0.3)

    if len(peak_idx) < 2:
        print('Could not find enough peaks for envelope fit.')
        plt.close()
        return

    t_pk = t[peak_idx]
    s_pk = s[peak_idx]

    ax.scatter(t_pk, s_pk, s=40, color='red', zorder=5, label='Envelope peaks (fit points)')

    span = t[-1] - t[0]

    # Fix Boffset from the late-time signal mean (last 40 % of the trace),
    # where the ringdown has mostly decayed to the noise floor.
    # Fitting it as a free parameter lets it drift and prevents the
    # envelopes from closing properly at the tail.
    late_mask = t >= (t[0] + span * 0.6)
    Boffset   = float(np.mean(s[late_mask])) if late_mask.any() else 0.0
    s_pk_c    = s_pk - Boffset   # centred peak amplitudes for fitting

    I0_g  = float(s_pk_c[0])
    tau_g = span / 3.0

    try:
        popt, _ = curve_fit(
            lambda t, I0, tau: I0 * np.exp(-t / tau),
            t_pk, s_pk_c,
            p0=[I0_g, tau_g],
            bounds=([0, 1e-9], [np.inf, span * 20]),
            maxfev=20_000,
        )
    except RuntimeError as exc:
        print(f'Ringdown fit failed: {exc}')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Signal (mV)')
        ax.set_title('Cavity Ringdown')
        plt.tight_layout()
        plt.show()
        return

    I0, tau = popt
    t_fit = np.linspace(t[0], t[-1], 2000)
    env   = I0 * np.exp(-t_fit / tau)

    ax.plot(t_fit, env + Boffset, 'r-',  lw=2, label=f'Fit  τ = {tau:.4g} µs')
    ax.plot(t_fit, -(env + Boffset),     'b--', lw=2, label='Mirrored fit (−(I₀·e^(−t/τ) + B))')

    Q = 2.0 * np.pi * VC * tau * 1e-6   # tau µs → s

    if Q < 1e3:
        msg = 'Data Invalid — Q < 10³'
    else:
        msg = f'Q = {Q:.3e}'

    print(f'\nRingdown:  τ = {tau:.4g} µs  |  {msg}')

    ax.text(0.97, 0.95, msg, transform=ax.transAxes, ha='right', va='top',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Signal (mV)')
    ax.set_title('Cavity Ringdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Sideband mode
# ---------------------------------------------------------------------------

def sideband_mode(t, s):
    fm = float(input('Enter the modulation frequency fm (Hz): '))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(t, s, s=8, alpha=0.5, label='Data')

    # --- locate the three peaks (main + two sidebands) -----------------
    min_dist = max(len(t) // 15, 3)
    peaks = np.array([], dtype=int)
    for pct in (65, 50, 35, 20):
        peaks, _ = find_peaks(s, height=np.percentile(s, pct), distance=min_dist)
        if len(peaks) >= 3:
            break

    if len(peaks) < 3:
        print(f'Only found {len(peaks)} peak(s). Cannot fit three Lorentzians.')
        plt.close()
        return

    # Keep the 3 tallest, sorted left → right by time
    top3 = peaks[np.argsort(s[peaks])[-3:]]
    top3 = top3[np.argsort(t[top3])]

    tp = t[top3]
    Ap = s[top3]
    span = t[-1] - t[0]
    g0 = span / 20.0

    # Tallest of the three is the main peak
    main_i = int(np.argmax(Ap))

    p0 = []
    for i in range(3):
        g = g0 if i == main_i else g0 / 2.0
        p0.extend([Ap[i], tp[i], g])

    lower = [0.0, t[0],  1e-9] * 3
    upper = [np.inf, t[-1], span] * 3

    try:
        popt, _ = curve_fit(
            _three_lorentzians, t, s, p0=p0,
            bounds=(lower, upper), maxfev=30_000,
        )
    except RuntimeError as exc:
        print(f'Sideband fit failed: {exc}')
        plt.close()
        return

    A1, t1, g1, A2, t2, g2, A3, t3, g3 = popt

    # Sort fitted triplets by time position; identify main (tallest) and sidebands
    fitted = sorted(
        [(t1, abs(A1), abs(g1)), (t2, abs(A2), abs(g2)), (t3, abs(A3), abs(g3))],
        key=lambda x: x[0],
    )
    main_i2 = int(np.argmax([f[1] for f in fitted]))
    sb_i2 = [i for i in range(3) if i != main_i2]

    t_sbL, A_sbL, g_sbL = fitted[sb_i2[0]]
    t_sbR, A_sbR, g_sbR = fitted[sb_i2[1]]
    t_main, A_main, g_main = fitted[main_i2]

    delta_t = abs(t_sbR - t_sbL)   # µs – sideband separation
    C = 2.0 * fm / delta_t         # Hz / µs – calibration factor

    # --- true baseline: median of data in outer regions ----------------
    # (outside each sideband, away from the main peak)
    margin = 0.1 * delta_t
    left_mask  = t < (t_sbL - margin)
    right_mask = t > (t_sbR + margin)
    base_L = float(np.median(s[left_mask]))  if left_mask.any()  else 0.0
    base_R = float(np.median(s[right_mask])) if right_mask.any() else 0.0
    true_base = (base_L + base_R) / 2.0

    # --- numerical FWHM of main peak above true base -------------------
    t_dense = np.linspace(t[0], t[-1], 50_000)
    s_dense = _three_lorentzians(t_dense, *popt)

    peak_val = float(s_dense[np.argmin(np.abs(t_dense - t_main))])
    half_lev = true_base + (peak_val - true_base) / 2.0

    above = s_dense > half_lev
    crossings = np.where(np.diff(above.astype(int)))[0]
    t_cr = t_dense[crossings]
    lc = t_cr[t_cr < t_main]
    rc = t_cr[t_cr > t_main]
    FWHM = (rc[0] - lc[-1]) if (len(lc) and len(rc)) else g_main  # µs

    delta_v = FWHM * C   # Hz
    Q = VC / delta_v

    # --- plot ----------------------------------------------------------
    t_fit = np.linspace(t[0], t[-1], 2000)
    ax.plot(t_fit, _three_lorentzians(t_fit, *popt), 'r-', lw=2,
            label='Three-Lorentzian fit')
    ax.axhline(true_base, color='gray', ls=':', lw=1.2, alpha=0.8,
               label=f'True base = {true_base:.3g} mV')
    ax.axhline(half_lev, color='purple', ls=':', lw=1.2, alpha=0.8,
               label=f'Half-max = {half_lev:.3g} mV')

    msg = f'Q = {Q:.3e}'
    print(
        f'\nSideband analysis:'
        f'\n  Sideband separation  δt = {delta_t:.4f} µs'
        f'\n  Calibration factor    C = {C:.4e} Hz/µs'
        f'\n  True base               = {true_base:.4f} mV'
        f'\n  FWHM (time domain)      = {FWHM:.4f} µs'
        f'\n  Cavity linewidth      Δν = {delta_v:.4f} Hz'
        f'\n  Q factor                = {Q:.4e}'
    )

    ax.text(0.97, 0.95, msg, transform=ax.transAxes, ha='right', va='top',
            fontsize=13, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax.set_xlabel('Time (µs)')
    ax.set_ylabel('Signal (mV)')
    ax.set_title('Sideband Scan')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = input('CSV file path: ').strip().strip('"')

    if not os.path.isfile(path):
        sys.exit(f'File not found: {path}')

    t, s = _load(path)

    mode = input("Mode — enter 'Ringdown' or 'Sideband': ").strip().lower()
    if mode.startswith('r'):
        ringdown_mode(t, s)
    elif mode.startswith('s'):
        sideband_mode(t, s)
    else:
        sys.exit("Unknown mode. Please enter 'Ringdown' or 'Sideband'.")


if __name__ == '__main__':
    main()
