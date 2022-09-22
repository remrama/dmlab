"""Sleeb
"""


################# -------------------
################# Utils
################# -------------------

def eeg_channels_sidecar(**kwargs):
    sidecar = {
        "name": "See BIDS spec",
        "type": "See BIDS spec",
        "units": "See BIDS spec",
        "description": "See BIDS spec",
        "sampling_frequency": "See BIDS spec",
        "reference": "See BIDS spec",
        "low_cutoff": "See BIDS spec",
        "high_cutoff": "See BIDS spec",
        "notch": "See BIDS spec",
        "status": "See BIDS spec",
        "status_description": "See BIDS spec",
        "RespirationHardware": "tbd", # seems like a good thing to add??
    }
    sidecar.update(kwargs)
    return sidecar


def events_sidecar(**kwargs):
    sidecar = {
        "onset": {
            "LongName": "Onset (in seconds) of the event",
            "Description": "Onset (in seconds) of the event"
        },
        "duration": {
            "LongName": "Duration of the event (measured from onset) in seconds",
            "Description": "Duration of the event (measured from onset) in seconds"
        },
        "value": {
            "LongName": "Marker/trigger value associated with the event",
            "Description": "Marker/trigger value associated with the event"
        },
        "description": {
            "LongName": "Value description",
            "Description": "Readable explanation of value markers column",
        },
        "StimulusPresentation": {
            "OperatingSystem": "Linux Ubuntu 18.04.5",
            "SoftwareName": "Psychtoolbox",
            "SoftwareRRID": "SCR_002881",
            "SoftwareVersion": "3.0.14",
            "Code": "doi:10.5281/zenodo.3361717"
        }
    }
    sidecar.update(kwargs)
    # Make sure stim presentation info is last (only one that's not a column).
    sidecar["StimulusPresentation"] = sidecar.pop("StimulusPresentation")
    return sidecar


def eeg_data_sidecar(
        task_name,
        task_description,
        task_instructions,
        reference_channel,
        ground_channel,
        sampling_frequency,
        recording_duration,
        n_eeg_channels,
        n_eog_channels,
        n_ecg_channels,
        n_emg_channels,
        n_misc_channels,
        **kwargs
    ):
    sidecar = {
        "TaskName": task_name,
        "TaskDescription": task_description,
        "Instructions": task_instructions,
        "InstitutionName": "Northwestern University",
        "Manufacturer": "Neuroscan",
        "ManufacturersModelName": "tbd",
        "CapManufacturer": "tbd",
        "CapManufacturersModelName": "tbd",
        "PowerLineFrequency": 60,
        "EEGPlacementScheme": "10-20",
        "EEGReference": f"single electrode placed on {reference_channel}",
        "EEGGround": f"single electrode placed on {ground_channel}",
        "SamplingFrequency": sampling_frequency,
        "EEGChannelCount": n_eeg_channels,
        "EOGChannelCount": n_eog_channels,
        "ECGChannelCount": n_ecg_channels,
        "EMGChannelCount": n_emg_channels,
        "MiscChannelCount": n_misc_channels,
        "TriggerChannelCount": 0,
        "SoftwareFilters": "tbd",
        "HardwareFilters": {
            "tbd": {
                "tbd": "tbd",
                "tbd": "tbd"
            }
        },
        "RecordingType": "continuous",
        "RecordingDuration": recording_duration,
    }
    sidecar.update(kwargs)
    return sidecar