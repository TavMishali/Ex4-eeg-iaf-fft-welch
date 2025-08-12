import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fft import fft

# Neural data-analysis - Exercise #4
#
# Place your code this scaffold (marked with YOUR_CODE). The commented code
# lines serve as suggestions and clues for your solution.

## parameters
# define parameters here and avoid magic numbers
file_path = './DATA_DIR/'  # change path - TODO
fs = 256  # units in [Hz]
elecNum = 18  # Electrode number
alpha_range = (4, 14)  # frequency range in Hz for plotting

def edfread(fname, assignToVariables=False, targetSignals=None):
    # Open the EDF file
    try:
        fid = open(fname, 'rb')  # chars till 5600
    except IOError as e:
        raise IOError(f"Error opening file: {e}")

    # HEADER
    hdr = {}
    hdr['ver'] = int(fid.read(8).decode('ascii').strip())
    hdr['patientID'] = fid.read(80).decode('ascii').strip()
    hdr['recordID'] = fid.read(80).decode('ascii').strip()
    hdr['startdate'] = fid.read(8).decode('ascii').strip()
    hdr['starttime'] = fid.read(8).decode('ascii').strip()
    hdr['bytes'] = int(fid.read(8).decode('ascii').strip())
    reserved = fid.read(44)
    hdr['records'] = int(fid.read(8).decode('ascii').strip())
    hdr['duration'] = float(fid.read(8).decode('ascii').strip())
    hdr['ns'] = int(fid.read(4).decode('ascii').strip())

    hdr['label'] = [fid.read(16).decode('ascii').strip() for _ in range(hdr['ns'])]

    # Handle targetSignals (subset of signals to read)
    if targetSignals is None:
        targetSignals = list(range(hdr['ns']))  # Read all signals by default
    elif isinstance(targetSignals, (str, list)):
        targetSignals = [hdr['label'].index(signal) for signal in targetSignals]

    hdr['transducer'] = [fid.read(80).decode('ascii').strip() for _ in range(hdr['ns'])]
    hdr['units'] = [fid.read(8).decode('ascii').strip() for _ in range(hdr['ns'])]
    hdr['physicalMin'] = np.array([float(fid.read(8).decode('ascii').strip()) for _ in range(hdr['ns'])])
    hdr['physicalMax'] = np.array([float(fid.read(8).decode('ascii').strip()) for _ in range(hdr['ns'])])
    hdr['digitalMin'] = np.array([float(fid.read(8).decode('ascii').strip()) for _ in range(hdr['ns'])])
    hdr['digitalMax'] = np.array([float(fid.read(8).decode('ascii').strip()) for _ in range(hdr['ns'])])
    hdr['prefilter'] = [fid.read(80).decode('ascii').strip() for _ in range(hdr['ns'])]
    hdr['samples'] = np.array([int(fid.read(8).decode('ascii').strip()) for _ in range(hdr['ns'])])

    # Read reserved bytes for each signal
    for _ in range(hdr['ns']):
        fid.read(32).decode('ascii').strip()

    hdr['label'] = [hdr['label'][i] for i in targetSignals]

    print("Step 1 of 2: Reading requested records. (This may take a few minutes)...")

    if assignToVariables or len(targetSignals) > 0:
        # Scaling factors for the signals
        scalefac = (hdr['physicalMax'] - hdr['physicalMin']) / (hdr['digitalMax'] - hdr['digitalMin'])
        dc = hdr['physicalMax'] - scalefac * hdr['digitalMax']

        # Read records
        record = np.zeros((len(targetSignals), hdr['samples'][0] * hdr['records']))

        for recnum in range(hdr['records']):
            for ii in range(hdr['ns']):
                if ii in targetSignals:
                    num_samples = hdr['samples'][ii]
                    data = np.fromfile(fid, dtype=np.int16, count=num_samples)
                    data = data * scalefac[ii] + dc[ii]
                    record[ii, recnum * num_samples: (recnum + 1) * num_samples] = data
                else:
                    fid.seek(hdr['samples'][ii] * 2, 1)  # Skip samples we don't need

    print("Step 2 of 2: Parsing data...")

    if assignToVariables:
        locals().update({hdr['label'][i]: record[i, :] for i in range(len(hdr['label']))})
        return hdr, None

    fid.close()
    return hdr, record

def load_edf_channel19(edf_path):
    _,data = edfread(edf_path)
    return data[elecNum]

# Process each subject's folder
subject_data = {}
for subFolder in sorted(os.listdir(file_path)):
    subFolderPath = os.path.join(file_path, subFolder)
    ## 1. Data handling
    subject_num_match = re.search(r"S(\d+)", subFolder)
    if not subject_num_match:
        continue  # skip if not a valid subject folder

    subject_num = int(subject_num_match.group(1))
    subject_data[subject_num] = {}

    if not os.path.isdir(subFolderPath):
        continue  # skip if not a directory

    for file in os.listdir(subFolderPath):
        full_path = ""
        if file.endswith(".edf"):
            full_path = os.path.join(subFolderPath, file)
            if "EO" in file:
                subject_data[subject_num]['EO'] = load_edf_channel19(full_path)
            elif "EC" in file:
                subject_data[subject_num]['EC'] = load_edf_channel19(full_path)


def compute_fft_power(signal, fs):
    N = len(signal)
    fft_vals = fft(signal)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    # Take one-sided spectrum
    one_sided = freqs > 0
    power = (np.abs(fft_vals) ** 2) / (fs * N)  # normalization
    return freqs[one_sided], power[one_sided]

def compute_welch_power(signal, fs):
    freqs, psd = welch(signal, fs=fs, nperseg=512, noverlap=256, window='hamming', scaling='density')
    return freqs, psd

def find_iaf(freqs, power_ec, power_eo, alpha_range=(6, 14)):
    diff = power_ec - power_eo
    mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    alpha_freqs = freqs[mask]
    alpha_diff = diff[mask]
    iaf_index = np.argmax(alpha_diff)
    iaf_freq = alpha_freqs[iaf_index]
    return iaf_freq, alpha_freqs, alpha_diff

def plot_fft_power_spectra(subject_id, ec_data, eo_data):
    f_ec, p_ec = compute_fft_power(ec_data, fs)
    f_eo, p_eo = compute_fft_power(eo_data, fs)
    mask = (f_ec >= alpha_range[0]) & (f_ec <= alpha_range[1])
    plt.figure()
    plt.plot(f_ec[mask], p_ec[mask], label='EC')
    plt.plot(f_eo[mask], p_eo[mask], label='EO')
    plt.title(f'Subject {subject_id} - FFT Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power [AU]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_fft_difference_spectrum(subject_id, ec_data, eo_data):
    f_ec, p_ec = compute_fft_power(ec_data, fs)
    f_eo, p_eo = compute_fft_power(eo_data, fs)

    iaf_freq, alpha_freqs, alpha_diff = find_iaf(f_ec, p_ec, p_eo, alpha_range)
    plt.figure()
    plt.plot(alpha_freqs, alpha_diff, label='EC - EO')
    plt.axvline(x=iaf_freq, color='r', linestyle='--', label=f'IAF = {iaf_freq:.2f} Hz')
    plt.title(f'Subject {subject_id} - FFT Difference Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Difference [AU]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_welch_power_spectra(subject_id, ec_data, eo_data):
    f_ec, p_ec = compute_welch_power(ec_data, fs)
    f_eo, p_eo = compute_welch_power(eo_data, fs)
    mask = (f_ec >= alpha_range[0]) & (f_ec <= alpha_range[1])
    plt.figure()
    plt.plot(f_ec[mask], p_ec[mask], label='EC')
    plt.plot(f_eo[mask], p_eo[mask], label='EO')
    plt.title(f'Subject {subject_id} - Welch Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power [AU]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_welch_difference_spectrum(subject_id, ec_data, eo_data):
    f_ec, p_ec = compute_welch_power(ec_data, fs)
    f_eo, p_eo = compute_welch_power(eo_data, fs)

    iaf_freq, alpha_freqs, alpha_diff = find_iaf(f_ec, p_ec, p_eo, alpha_range)
    plt.figure()
    plt.plot(alpha_freqs, alpha_diff, label='EC - EO')
    plt.axvline(x=iaf_freq, color='r', linestyle='--', label=f'IAF = {iaf_freq:.2f} Hz')
    plt.title(f'Subject {subject_id} - Welch Difference Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Difference [AU]')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

for subject_id, data in subject_data.items():
    ec = data['EC']
    eo = data['EO']

    plot_fft_power_spectra(subject_id, ec, eo)
    plot_welch_power_spectra(subject_id, ec, eo)

    plot_fft_difference_spectrum(subject_id, ec, eo)
    plot_welch_difference_spectrum(subject_id, ec, eo)
