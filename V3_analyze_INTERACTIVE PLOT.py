# %% Analysis_utils
import eeg_eyetracking_parser as eet
from eeg_eyetracking_parser import _eeg_preprocessing as eep
import mne; mne.set_log_level(False)
from mne.time_frequency import tfr_morlet
from datamatrix import DataMatrix, convert as cnv, operations as ops, \
    series as srs, functional as fnc
import numpy as np
from scipy.stats import linregress
from pathlib import Path
from matplotlib import pyplot as plt
import time_series_test as tst
import seaborn as sns
from datamatrix import SeriesColumn
from joblib import dump,load

SUBJECTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]#31, 32, 41, 42, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92
EEG_EPOCH = -0.1, 0.9
CHECKPOINT = '27042024'
PUPIL_EPOCH = -0.1, 0.9
ERG_PEAK1 = .047
ERG_PEAK2 = .076
MIN_BLINK_LATENCY = .1
YLIM = -11e-6, 12e-6
STIMULUS_TRIGGER = 2
FREQS = np.arange(4, 30, 1)
MORLET_MARGIN = .5
Z_THRESHOLD = 3

# Maps [-1, 1] intensity to cd/m2
# array([4.334, 8.34256363, 16.05869128, 30.91154911, 59.50197634, 114.536])
INTENSITY_CDM2 = {
    0: 4.334,
    1: 8.34,
    2: 16.05,
    3: 30.91,
    4: 59.50,
    5: 114.53
}
# Ordinally coded intensity
INTENSITY_ORD = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
}
mne.io.pick._PICK_TYPES_DATA_DICT['misc'] = True
FOLDER_SVG = Path('SVG')
FOLDER_PNG = Path('PNG')
# if not FOLDER_SVG.exists():
#     FOLDER_SVG.mkdir()
# if not FOLDER_PNG.exists():
#     FOLDER_PNG.mkdir()
# FOLDER_TOPOMAPS = FOLDER_SVG / 'topomaps'
# if not FOLDER_TOPOMAPS.exists():
#     FOLDER_TOPOMAPS.mkdir()

# Monkeypatch the preprocessor to avoid EOGs from being subtracted and instead
# marking them as separate EOG channels
def custom_eog_channels(raw, *args, **kwargs):
    raw.set_channel_types(
        dict(VEOGB='eog', VEOGT='eog', HEOGL='eog', HEOGR='eog'))
eep.create_eog_channels = custom_eog_channels

def intensity_cdm2(i):
    return {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5
    }[i]

def area_to_mm(au):
    """Converts in arbitrary units to millimeters of diameter. This is specific
    to the recording set-up.

    Parameters
    ----------
    au: float

    Returns
    -------
    float
    """
    return -0.9904 + 0.1275 * au ** .5

def z_by_freq(col):
    """Performs z-scoring across trials, channels, and time points but
    separately for each frequency.

    Parameters
    ----------
    col: MultiDimensionalColumn

    Returns
    -------
    MultiDimensionalColumn
    """
    zcol = col[:]
    for i in range(zcol.shape[2]):
        zcol._seq[:, :, i] = (
            (zcol._seq[:, :, i] - np.nanmean(zcol._seq[:, :, i]))
            / np.nanstd(zcol._seq[:, :, i])
        )
    return zcol

def read_subject(subject_nr):
    return eet.read_subject(
        subject_nr, eeg_preprocessing=[
            'drop_unused_channels',
            'rereference_channels',
            'annotate_emg',
            'create_eog_channels',
            'set_montage',
            'band_pass_filter',
            'autodetect_bad_channels',
            'interpolate_bads'])

# @fnc.memoize(persistent=True, key=f'../checkpoints/{CHECKPOINT}.dm')
def get_merged_data():
    dm = DataMatrix()
    for subject_nr in SUBJECTS:
        raw, events, metadata = read_subject(subject_nr)
        sdm = cnv.from_pandas(metadata)
        sdm.blink_latency = -1
        trial_nr = -1
        stim_onset = None
        for a in raw.annotations:
            if a['description'] == '2':
                trial_nr += 1
                stim_onset = a['onset']
            if stim_onset is None:
                continue
            if a['description'] == 'BLINK':
                blink_latency = a['onset'] - stim_onset
                sdm.blink_latency[trial_nr] = blink_latency
                stim_onset = None
        sdm.pupil = cnv.from_mne_epochs(
            eet.PupilEpochs(raw, eet.epoch_trigger(events, STIMULUS_TRIGGER),
                            tmin=PUPIL_EPOCH[0], tmax=PUPIL_EPOCH[1],
                            metadata=metadata, baseline=None),
            ch_avg=True)
        sdm.eog = cnv.from_mne_epochs(
            mne.Epochs(raw, eet.epoch_trigger(events, STIMULUS_TRIGGER),
                       tmin=EEG_EPOCH[0], tmax=EEG_EPOCH[1],
                       picks='eog', metadata=metadata))
        sdm.erp = cnv.from_mne_epochs(
            mne.Epochs(raw, eet.epoch_trigger(events, STIMULUS_TRIGGER),
                       tmin=EEG_EPOCH[0], tmax=EEG_EPOCH[1],
                       picks='eeg', metadata=metadata))
        # Get time-frequency analyses
        epochs = mne.Epochs(raw, eet.epoch_trigger(events, STIMULUS_TRIGGER),
                       tmin=EEG_EPOCH[0] - MORLET_MARGIN,
                       tmax=EEG_EPOCH[0] + MORLET_MARGIN, picks='eog',
                       metadata=metadata)
        morlet = tfr_morlet(
            epochs, freqs=FREQS, n_cycles=2, n_jobs=-1,
            return_itc=False, use_fft=True, average=False,
            decim=5, picks=np.arange(len(epochs.info['ch_names'])))
        morlet.crop(0, EEG_EPOCH[1])
        sdm.eog_tfr = cnv.from_mne_tfr(morlet)
        sdm.eog_tfr = z_by_freq(sdm.eog_tfr)[:, ...]
        # The subject number is the first digit, the session number the second
        # sdm.session_nr = sdm.subject_nr % 10
        # sdm.subject_nr = sdm.subject_nr // 10
        sdm.erg = sdm.eog[:, ...]
        sdm.erg_upper = sdm.eog[:, ('VEOGB', 'VEOGT')][:, ...]
        sdm.erg_lower = sdm.eog[:, ('HEOGL', 'HEOGR')][:, ...]
        sdm.laterg = sdm.eog[:, ('VEOGB', 'HEOGL')][:, ...] - \
            sdm.eog[:, ('VEOGT', 'HEOGR')][:, ...]
        sdm.erp_occipital = sdm.erp[:, ('O1', 'Oz', 'O2')][:, ...]
        sdm.laterp_occipital = sdm.erp[:, 'O1'] - sdm.erp[:, 'O2']
        # We first convert pupil size to millimeters, and then take the mean
        # over the first 150 ms (below the response latency), The slope is also
        # calculated over this initial 150 ms.
        sdm.pupil = sdm.pupil @ area_to_mm
        sdm.mean_pupil = sdm.pupil[:, 0:150][:, ...]
        x = np.arange(150)
        for row in sdm:
            result = linregress(x, row.pupil[:150])
            row.pupil_slope = result.slope
        # We recode pupil size as surface area (as opposed to diamter),
        # baseline size, and z-scored values. We also make a binary split
        # on the slope indicating whether the pupil was constricting or
        # dilating.
        sdm.mean_pupil_area = sdm.mean_pupil ** 2
        sdm.bl_pupil = srs.baseline(sdm.pupil, sdm.pupil, 0, 50)
        sdm.z_pupil = ops.z(sdm.mean_pupil)
        sdm.z_pupil_slope = ops.z(sdm.pupil_slope)
        sdm.pupil_dilation = 'Constricting'
        sdm.pupil_dilation[sdm.pupil_slope > 0] = 'Dilating'
        sdm.z_erg = ops.z(sdm.erg[:, 90:110][:, ...])
        dm <<= sdm
    dm = dm.z_erg != np.nan
    dm = dm.mean_pupil != np.nan
    dm = dm.z_erg < Z_THRESHOLD
    dm = dm.z_erg > -Z_THRESHOLD
    dm = dm.z_pupil < Z_THRESHOLD
    dm = dm.z_pupil > -Z_THRESHOLD
    dm = dm.z_pupil_slope < Z_THRESHOLD
    dm = dm.z_pupil_slope > -Z_THRESHOLD
    # dm = dm.target == 0
    # dm.intensity_cdm2 = dm.backgroundLevel @ (lambda i: INTENSITY_CDM2[i])
    # dm.intensity_ord = dm.backgroundLevel @ (lambda i: INTENSITY_ORD[i])
    # dm.influx_cdm2 = dm.intensity_cdm2 * dm.mean_pupil ** 2
    dm.has_blink = 0
    dm.has_blink[dm.blink_latency >= 0] = 1
    return dm

# dm=load('dm_eeg_eyetracking.joblib')
plt.close('all')
# %% Load data
dm = get_merged_data()
print(f'before blink removal: {len(dm)}')
dm = (dm.blink_latency < 0) | (dm.blink_latency > .5)
print(f'after blink removal: {len(dm)}')
dm.bin_pupil = -1
dm.bin_pupil_mm = 0
for i, bdm in enumerate(ops.bin_split(dm.z_pupil, 2)):
    dm.bin_pupil[bdm] = i
    dm.bin_pupil_mm[bdm] = bdm.mean_pupil.mean
dm=dm.training!='yes'

dm.min_erg = srs.reduce(dm.erg[:, 130:165], min)
dm.max_erg = srs.reduce(dm.erg[:, 160:210], max)
dm.min_erp_occipital = srs.reduce(dm.erp_occipital[:, 160:210], min)

dm.erg45_index = 0
dm.erg45_peak = 0
for subject_nr, intensity, sdm in ops.split(dm.subject_nr, dm.backgroundLevel):
   erg45_index = np.argmin(sdm.erg.mean[130:165]) + 130
   dm.erg45_index[sdm] = erg45_index
   dm.erg45_peak[sdm] = sdm.erg[:, erg45_index - 2:erg45_index + 3][:, ...]
   
dm.erg75_index = 0
dm.erg75_peak = 0
for subject_nr, intensity, sdm in ops.split(dm.subject_nr, dm.backgroundLevel):
   erg75_index = np.argmax(sdm.erg.mean[160:210]) + 160
   dm.erg75_index[sdm] = erg75_index
   dm.erg75_peak[sdm] = sdm.erg[:, erg75_index - 2:erg75_index + 3][:, ...]
   
dm.ocp_index = 0
dm.ocp_peak = 0
for subject_nr, intensity, sdm in ops.split(dm.subject_nr, dm.backgroundLevel):
   ocp_index = np.argmin(sdm.erg.mean[160:210]) + 160
   dm.ocp_index[sdm] = ocp_index
   dm.ocp_peak[sdm] = sdm.erg[:, ocp_index - 2:ocp_index + 3][:, ...]

dump(dm, 'dm_eeg_eyetracking.joblib')

fdm=dm
# %% 
# del dm.erp  # e memory



# %% Load data# %% Plot Pupil constriction
"""
Plot pupil constriction after stimulus onset as a function of stimulus
intensity for full-field flashes.
"""
plt.close('all')
plt.figure(figsize=(8, 8))
# fdm = (dm.runBD=='False')# & (dm.probe_dur==200)
tst.plot(fdm, dv='pupil', x0=-.1, sampling_freq=1000, hues='jet',
         hue_factor='backgroundLevel',
         legend_kwargs={'title': 'Background Intensity'})
plt.ylabel('Pupil size (mm)')
plt.xlabel('Time since flash onset (s)')
plt.savefig(FOLDER_SVG / 'pupil-by-intensity.svg')
plt.savefig(FOLDER_PNG / 'pupil-by-intensity.png')
plt.show()
# plt.close('all')
# %% # Effects of intensity
"""
# Time-frequency plots
"""
plt.close('all')
# fdm = (dm.runBD=='False')
plt.figure(figsize=(8, 8))
Y_FREQS = np.array([0, 4, 9, 25])
plt.imshow(fdm.eog_tfr[...], aspect='auto')
plt.yticks(Y_FREQS, FREQS[Y_FREQS])
# plt.xticks(np.arange(0, 30, 3), np.arange(0, .15, .015))
plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.savefig(FOLDER_SVG / 'time-frequency.svg')
plt.savefig(FOLDER_PNG / 'time-frequency.png')
plt.show()
# plt.close('all')
# %% Plot Effects of brightness (ERG and EEG signals)
"""
Plot the ERG and EEG signals after stimulus onset as a function of stimulus
intensity for full-field flashes.
"""
plt.close('all')
max_duration=200
plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.title(f'a) Full field ERG by intensity')
plt.ylim(*YLIM)
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='red', linestyle='-')
plt.axvline(ERG_PEAK1, color='green', linestyle=':')
plt.axvline(ERG_PEAK2, color='green', linestyle=':')
tst.plot(fdm, dv='erg', hue_factor='backgroundLevel', x0=-.1,
         sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Brightness'})
plt.xticks([])
# Update the right limit of x axis to tgv/100 + 100 and add a vertical red line at tgv/100
plt.xlim(left=-0.05, right=max_duration/1000 + 0.15)
plt.axvline(max_duration/1000, color='red', linestyle='-')  # Add red vertical line at tgv/100
plt.axhline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.subplot(212)
plt.title(f'b) Full field EEG by intensity')
plt.ylim(*YLIM)
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='red', linestyle='-')
plt.axvline(ERG_PEAK1, color='green', linestyle=':')
plt.axvline(ERG_PEAK2, color='green', linestyle=':')
tst.plot(fdm, dv='erp_occipital', hue_factor='backgroundLevel',
         x0=-.1, sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Brightness'})
# Update the right limit of x axis to tgv/100 + 100 and add a vertical red line at tgv/100
plt.xlim(left=-0.05, right=max_duration/1000 + 0.15)
plt.axvline(max_duration/1000, color='red', linestyle='-')  # Add red vertical line at tgv/100
plt.axhline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.xlabel('Time since stimulus onset (s)')
plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter as needed to fit the legend
plt.savefig(FOLDER_SVG / f'erg-and-eeg-by-intensity-ALL.svg')
plt.savefig(FOLDER_PNG / f'erg-and-eeg-by-intensity-ALL.png')
plt.show()
# plt.close('all')
# %% Plot Effects of brightness SEPARATELY (ERG and EEG signals)
plt.close('all')
for tgv, tdm in ops.split(fdm.probe_dur): #first one is the values, second is the datamatrix
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    plt.title(f'a) Full field ERG by intensity')
    plt.ylim(*YLIM)
    plt.axhline(0, color='black', linestyle=':')
    plt.axvline(0, color='red', linestyle='-')
    plt.axvline(ERG_PEAK1, color='green', linestyle=':')
    plt.axvline(ERG_PEAK2, color='green', linestyle=':')
    tst.plot(tdm, dv='erg', hue_factor='backgroundLevel', x0=-.1,
             sampling_freq=1000, hues='jet',
             legend_kwargs={'title': 'Brightness'})
    plt.xticks([])
    # Update the right limit of x axis to tgv/100 + 100 and add a vertical red line at tgv/100
    plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
    plt.axvline(tgv/1000, color='red', linestyle='-')  # Add red vertical line at tgv/100
    plt.axhline(0, color='black', linestyle=':')
    plt.ylabel('Voltage (µv)')
    plt.subplot(212)
    plt.title(f'b) Full field EEG by intensity')
    plt.ylim(*YLIM)
    plt.axhline(0, color='black', linestyle=':')
    plt.axvline(0, color='red', linestyle='-')
    plt.axvline(ERG_PEAK1, color='green', linestyle=':')
    plt.axvline(ERG_PEAK2, color='green', linestyle=':')
    tst.plot(tdm, dv='erp_occipital', hue_factor='backgroundLevel',
             x0=-.1, sampling_freq=1000, hues='jet',
             legend_kwargs={'title': 'Brightness'})
    # Update the right limit of x axis to tgv/100 + 100 and add a vertical red line at tgv/100
    plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
    plt.axvline(tgv/1000, color='red', linestyle='-')  # Add red vertical line at tgv/100
    plt.axhline(0, color='black', linestyle=':')
    plt.ylabel('Voltage (µv)')
    plt.xlabel('Time since stimulus onset (s)')
    plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the rect parameter as needed to fit the legend
    plt.savefig(FOLDER_SVG / f'erg-and-eeg-by-intensity-{tgv}.svg')
    plt.savefig(FOLDER_PNG / f'erg-and-eeg-by-intensity-{tgv}.png')
    plt.show()

if True:
    plt.figure(figsize=(8, 8))
    plt.title(f'a) Full field ERG by intensity')
    # plt.ylim(*YLIM)
    plt.axhline(0, color='black', linestyle=':')
    plt.axvline(0, color='red', linestyle='-')
    plt.axvline(ERG_PEAK1*1000, color='green', linestyle=':')
    plt.axvline(ERG_PEAK2*1000, color='green', linestyle=':')
    tst.plot(fdm, dv='pupil', hue_factor='backgroundLevel', x0=-100,
             sampling_freq=1, hues='jet',
             legend_kwargs={'title': 'Brightness'})
    # Update the right limit of x axis to tgv/100 + 100 and add a vertical red line at tgv/100
    plt.axvline(0.2, color='red', linestyle='-')  # Add red vertical line at tgv/100
    plt.axhline(0, color='black', linestyle=':')
    plt.ylabel('Voltage (µv)')
# plt.close('all')
# %% Plot Correlations with pupil size
"""
Plot the ERG and EEG signals after stimulus onset as a function of pupil size
(five bins) for full-field flashes.
"""
plt.close('all')
# First calculate pupil bins
fdm.bin_pupil = -1
fdm.bin_pupil_mm = 0
tgv=200
for i, bdm in enumerate(ops.bin_split(fdm.z_pupil, 2)):
    fdm.bin_pupil[bdm] = i
    fdm.bin_pupil_mm[bdm] = bdm.mean_pupil.mean
# Then plot
plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.title(f'a) Full field ERG by pupil size (binned)')
plt.ylim(*YLIM)
plt.axvline(ERG_PEAK1, color='green', linestyle=':')
plt.axvline(ERG_PEAK2, color='green', linestyle=':')
plt.axhline(0, color='red', linestyle=':')
plt.axvline(0, color='red', linestyle=':')
# plt.axvline(.04, color='black', linestyle=':')
# plt.axvline(.06, color='black', linestyle=':')
# plt.axvline(.08, color='black', linestyle=':')
# plt.axvline(.1, color='black', linestyle=':')
tst.plot(fdm, dv='erg', hue_factor='bin_pupil_mm', x0=-.1,
         sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Pupil size'})
plt.xticks([])
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
plt.subplot(212)
plt.title(f'b) Full field EEG by pupil size (binned)')
plt.ylim(*YLIM)
plt.axvline(ERG_PEAK1, color='green', linestyle=':')
plt.axvline(ERG_PEAK2, color='green', linestyle=':')
plt.axhline(0, color='red', linestyle=':')
plt.axvline(0, color='red', linestyle=':')
# plt.axhline(0, color='black', linestyle=':')
# plt.axvline(0, color='black', linestyle=':')
# plt.axvline(.04, color='black', linestyle=':')
# plt.axvline(.06, color='black', linestyle=':')
# plt.axvline(.08, color='black', linestyle=':')
# plt.axvline(.1, color='black', linestyle=':')
tst.plot(fdm, dv='erp_occipital', hue_factor='bin_pupil_mm',x0=-.1,
         sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Pupil size'})
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.xlabel('Time since flash onset (s)')
plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
plt.savefig(FOLDER_SVG / 'erg-and-eeg-by-pupil-size-bin.svg')
plt.savefig(FOLDER_PNG / 'erg-and-eeg-by-pupil-size-bin.png')
# plt.close('all')
# %% Effects of pupil-size change (dilating vs constricting)
plt.close('all')
"""
Plot the ERG and EEG signals after stimulus onset as a function of pupil-size
change (dilating vs constricting) for full-field flashes.
"""
plt.figure(figsize=(8, 8))
plt.subplot(211)
plt.title(f'a) Full field ERG by pupil-size change')
plt.ylim(*YLIM)
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.axvline(.04, color='black', linestyle=':')
plt.axvline(.06, color='black', linestyle=':')
plt.axvline(.08, color='black', linestyle=':')
plt.axvline(.1, color='black', linestyle=':')
tst.plot(fdm, dv='erg', hue_factor='pupil_dilation', x0=-.1,
         sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Pupil-size change'})
plt.xticks([])
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
plt.subplot(212)
plt.title(f'b) Full field EEG by pupil-size change')
plt.ylim(*YLIM)
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.axvline(.04, color='black', linestyle=':')
plt.axvline(.06, color='black', linestyle=':')
plt.axvline(.08, color='black', linestyle=':')
plt.axvline(.1, color='black', linestyle=':')
tst.plot(fdm, dv='erp_occipital', hue_factor='pupil_dilation', x0=-.1,
         sampling_freq=1000, hues='jet',
         legend_kwargs={'title': 'Pupil-size change'})
plt.axhline(0, color='black', linestyle=':')
plt.axvline(0, color='black', linestyle=':')
plt.ylabel('Voltage (µv)')
plt.xlabel('Time since flash onset (s)')
plt.xlim(left=-0.05, right=tgv/1000 + 0.1)
plt.savefig(FOLDER_SVG / 'erg-and-eeg-by-pupil-size-change.svg')
plt.savefig(FOLDER_PNG / 'erg-and-eeg-by-pupil-size-change.png')
# plt.close('all')
# %% The relationship between pupil-size change and pupil size
plt.close('all')
plt.figure(figsize=(8, 8))
tst.plot(dm, dv='pupil', hue_factor='pupil_dilation')
plt.savefig(FOLDER_SVG / 'relationship-between-pupil-size-change-and-pupil-size.svg')
plt.savefig(FOLDER_PNG / 'relationship-between-pupil-size-change-and-pupil-size.png')
# plt.close('all')
# %% Plot voltage by pupil size and intensity for specific time points
# fdm.intensity_cdm2 = fdm.backgroundLevel @ (lambda i: INTENSITY_CDM2[i])
# fdm.intensity_ord = fdm.backgroundLevel @ (lambda i: INTENSITY_ORD[i])

# fdm.erg45 = fdm.erg[:, 90:110][:, ...]
# fdm.erg75 = fdm.erg[:, 110:130][:, ...]
# fdm.erg100 = fdm.erg[:, 150:200][:, ...]
# plt.figure(figsize=(12, 4))
# plt.subplots_adjust(wspace=0)
# plt.subplot(131)
# plt.title('a) 40 - 60 ms')
# sns.pointplot(x='intensity_cdm2', hue='bin_pupil_mm', y='erg45', data=fdm,
#               palette='flare')
# plt.legend(title='Pupil size (bin)')
# plt.xlabel('Intensity (cd/m2)')
# plt.ylabel('Voltage (µv)')
# plt.ylim(*YLIM)
# plt.subplot(132)
# plt.title('b) 60 - 80 ms')
# sns.pointplot(x='intensity_cdm2', hue='bin_pupil_mm', y='erg75', data=fdm,
#               palette='flare')
# plt.ylim(*YLIM)
# plt.legend(title='Pupil size (bin)')
# plt.xlabel('Intensity (cd/m2)')
# plt.yticks([])
# plt.subplot(133)
# plt.title('b) 100 - 150 ms')
# sns.pointplot(x='intensity_cdm2', hue='bin_pupil_mm', y='erg100', data=fdm,
#               palette='flare')
# plt.ylim(*YLIM)
# plt.legend(title='Pupil size (bin)')
# plt.xlabel('Intensity (cd/m2)')
# plt.yticks([])
# plt.savefig(FOLDER_SVG / 'erg-by-pupil-size-bin-and-intensity.svg')
# plt.show()

# """
# Plot variability by pupil size
# """
# vdm = DataMatrix(length=fdm.intensity.count * fdm.bin_pupil.count
#                  * fdm.subject_nr.count)
# vdm.pupil_std = SeriesColumn(depth=fdm.erg.depth - 50)
# for row, (intensity, bin_pupil, subject_nr, sdm) in zip(vdm,
#       ops.split(fdm.intensity, fdm.bin_pupil, fdm.subject_nr)):
#    row.intensity = intensity
#    row.bin_pupil = bin_pupil
#    row.subject_nr = subject_nr
#    row.pupil_std = sdm.erg.std[50:]

# for intensity, idm in ops.split(vdm.intensity):
#    tst.plot(idm, dv='pupil_std', hue_factor='bin_pupil')
#    plt.show()
# tst.plot(vdm, dv='pupil_std', hue_factor='bin_pupil')

# # vresult = tst.lmer_permutation_test(vdm,
# #    'pupil_std ~ bin_pupil + intensity', groups='subject_nr',
# #    suppress_convergence_warnings=True)
