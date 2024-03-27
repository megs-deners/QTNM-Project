#!/usr/bin/python3

'''
CheckPythonOutputs.py

Checks the outputted times and voltages are ok
'''

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.interpolate import interp1d


from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data

from matplotlib import cm
from skimage import img_as_ubyte  # For data type conversion


plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Helvetica"
})
#filenames = ["/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ea43bd30-5573-4cec-8ff5-b91233cb8797.h5"]
filenames = ["/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_04974749-daa1-48f2-b069-82728a64cd84.h5"]
filenames = ["/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_04974749-daa1-48f2-b069-82728a64cd84.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0685f1d0-a971-451d-baa5-1ec769d86518.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_07b33234-56b8-4d35-9a64-b934bf09ae61.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_07d30df8-624c-4b88-a08d-0cc5521d76c4.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0924d056-3d18-4531-97f7-030f0b3f18e3.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0b75f396-fcf0-47b2-81ca-a7f59d9b1309.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0c54d141-6b04-49a6-ba59-7833a4a84ff3.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0d693e15-e59f-4b70-badf-bd249c7dafca.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_0db06467-e94d-49df-9f7f-e12fed7d26ee.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_123bf35b-3113-4dab-bcb9-8ea52677bad1.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_128faeef-1f55-4d04-a9bf-25d921b49e9a.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_1503d88b-4cc5-47b4-9d0a-289d07027f17.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_154c21bf-6449-4359-bf4b-b609feac941d.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_185f99b7-ce52-4b85-a2dd-a3288fea4418.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_1a284827-2fe0-4e03-824e-0a3cea7c1ccc.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_1ad8cfdc-b6bc-4a08-b4d8-2df4ac305fec.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_2019ec22-b4a3-409e-bdf8-d0e139d0db91.h5","/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_20753901-4765-41ff-90a4-bbc096b1d117.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_20935849-7759-4cf9-883c-b62b94aa9d08.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_23073b4c-b210-459b-bec3-5a369d7b27b8.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_2572929f-3137-4018-b937-f914038867af.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_27c2a0ac-4141-438e-a35c-f69581065d11.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_31377ca4-8980-4fd9-8805-1418d2f99a9c.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_317c14e8-4efb-4e35-8027-21ef5342e637.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_3557f989-d441-4495-86ce-a69eedd0391d.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_3726bbe4-db10-4730-9dd0-08d9d652b714.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_3bffdf62-a39d-4eab-8ebd-8332fa33ca54.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_3dc2e827-96f1-4896-9ddf-1a0553e6d312.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_3e74e4e3-ca69-4515-ad44-224797b20dea.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_42dad62f-500b-4d90-8517-ecb8c52f9c7c.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_48048e8b-d3c4-4b1f-9f2e-a0d0cef5151a.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_4839b3ff-1612-476a-b184-b48b02451f7a.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_495a019a-b729-4af6-829c-545c3e230f83.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_52d6c655-194b-4d86-a308-a4be1b2c515e.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_537b0a37-d21d-4730-8cf1-d9893e5a207e.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_5775e3f8-2601-4775-becb-5baaf8ebd260.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_598c62af-2ed9-4e0b-85e9-6a704571b016.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_5fad7697-11b9-48aa-9af2-2ad67b1abcb7.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_60e894a3-c1b1-4f58-9ec6-fafc9ac0f5e0.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_61dca7af-fb3b-409f-8951-4069d79abdfd.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_666f5ff7-aaef-476c-85e6-6611d80e91bc.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_68edcc38-4dae-43be-9f5a-7c2e9a8a5cb6.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_6999db27-d569-4cee-a4dc-72972ad58078.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_6b99f609-e840-4c36-8f02-5bf613d355f6.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_6f8ab3c7-4aac-450f-bbcf-4f1f25a938f7.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_723dc303-50ae-46a9-bb53-9c0992aecfd0.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_73d892ec-23a8-49e6-a02e-30967654bbc4.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_742924dd-bde4-4c0c-a634-4e8316c50b4f.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_760a3c5b-3932-45ae-89cd-7961da5e5dc7.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_76502277-ef28-4f92-9d44-0bd4b9cd3379.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_7f3c7b22-0738-4e65-86d7-f5669f7aed22.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_85abd7cf-68ac-4414-b40e-16dc816f7e43.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_8ccbff0a-23f8-4a81-b2fc-88ada5443f16.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_92093d01-7ffd-4432-92a1-bae52d36ac10.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_93c9a61e-4b65-4614-b1eb-110494e152d9.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_947e7ce8-187b-47c6-93f3-865606329a96.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_95aa0c39-5dd8-4613-b5ef-63a954fc7af5.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9a4530bb-1e37-4025-8870-f36197038f5a.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9a94a097-a395-49cc-9747-5c67168718e9.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9bcec984-3b7a-4859-9458-54d36095de2d.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9bd037d3-c540-4443-8719-2a9838963e03.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9d2dca68-e10a-4017-bf11-fe88afedd7c7.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9f1fe5b9-ce3a-455b-9882-7f8f91d12b87.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_9f9c6a80-54d9-4e50-a5c0-a17b8a748a63.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_a0725190-cfea-416e-b0c7-d7063bca8f4a.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_a6b0b2d1-f1bd-4969-b6ef-d117fea05e5e.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_a756f05d-4f28-4a24-be1f-bda2ce6b1ab4.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ab413b3d-6b00-44d9-bb81-3fdbc6d880da.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_add3b03a-13da-449d-ac87-37f6f75667b2.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_b65e609e-9b7c-48fe-b36f-357dd6362e2d.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_b70d9f63-2e37-46ea-8339-09c32693c1c9.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_b7597944-0f59-4e63-a2a8-5cbf8ff7b8c8.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_b7695125-45e5-4d3d-9c09-58d67e562ed6.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_b9a8c7c0-ce14-4ed6-9289-4ea8e20c991c.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ba121996-d762-4035-b60f-682b192910c1.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_bb04161a-df41-46ee-8cc1-0cd49065d77a.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_bb3649c0-95c6-4d0a-bb36-8954f4ebdb97.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_bcc6ef16-0197-45ae-87e9-60796f74ff8c.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_bf04921c-2150-4cd4-a559-b1f7670c4484.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c00e2ebc-a06e-422e-83cf-900ed4991d83.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c0982cc6-1aab-439d-9442-d0b6cf75d838.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c0c0b09c-adc8-4d75-a376-f7c3280375a4.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c0d51da8-52e5-42da-bf35-4150ec915f62.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c1177430-81de-4539-8f33-71a24ad3ac13.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c1936298-fb2e-4d9f-b7bf-877e2a34c8e4.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c227e652-aace-4361-b2bb-2eaf813f86ee.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c335a2ce-09c2-4adc-9b19-96a5c682e479.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c3415d90-c4ef-4a79-8347-8a38168e3305.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c365cee5-d880-4c6f-a272-dff0eaa89b14.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_c4ce5725-e0cf-4b9c-aeac-3cb8fdddeab2.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ca0273e0-f4fb-4d20-a4ba-3bedcfc9a44b.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ca79d805-70be-4fac-a8f3-c9c7f884713c.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_cad838c7-2936-4187-bde5-f57d508c6aa9.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_cb941a41-1796-44c2-9859-fe118767c588.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ce41a232-7949-4497-9a23-e9e76ed23254.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_d17beeaf-2a7c-4384-8e87-7d13c5af33c1.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_d2c8f962-045e-47be-aacd-82fa4fa01c7f.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_d3ad44bb-2c09-4de6-9a2c-873427f07eca.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_d4eb4ebf-e4e2-4103-94c4-53949c52cefa.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_e58d69f2-7d44-4e44-88a3-1e671eb8b8d4.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_e794d408-ccbc-4342-9902-288f8d4655f8.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_e88a6bf3-6588-4510-84d7-87c17407c267.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_e88f6123-5e33-493e-b04e-913403ce5d8e.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_e90c11e1-e2a2-4f37-9aff-c6a1d7be7f04.h5",
"/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_ea43bd30-5573-4cec-8ff5-b91233cb8797.h5"]
# Check if file can be opened command+control+3 or 4 to remove

startFs = []
Energies = []
pitches = []
axFreqs = []
mainslopes = []
mainpowers = []
sideslopes  = []
sidepowers = []

fig, ax = plt.subplots()


for filename in filenames:
    if not os.access(filename, os.R_OK):
        print("File %s not readable" % filename)

    #File can be opened
    else:
        # Open HDF5 file
        f = h5py.File(filename, "r")
        if f:
            # Check if file is open
            print("File is open")

            # Open the group
            group = f["Data"]
            # Check group is accessible
            if group:
                print("Group is accessible")
                
                # Loop over datasets in group
                for dsetName in group:
                    print("Dataset is %s" % dsetName)
                    dset = group.get(dsetName)
                            
                    # Check dataset is open
                    if dset:
                        # Get info about attributes of dataset
                        axFreq = 0.0
                        pitchAngle = 0.0
                        rad = 0.0
                        startF = 0.0
                        deltaT = 0.0
                        for k in dset.attrs.keys():
                            if (k == "Axial frequency (Hertz)"):
                                axFreq = dset.attrs[k]
                            elif (k == "Pitch angle (degrees)"):
                                pitchAngle = dset.attrs[k]
                            elif (k == "Radial offset (metres)"):
                                rad = dset.attrs[k]
                            elif (k == "Start frequency (hertz)"):
                                startF = dset.attrs[k]
                            elif (k == "Time step (seconds)"):
                                deltaT = dset.attrs[k]
                            elif (k == "Start frequency, downmixed (hertz)"):
                                startFdown = dset.attrs[k]
                            elif (k == "Start E (eV)"):
                                startE = dset.attrs[k]
                            elif (k == "i_coil (amps)"):
                                icoil = dset.attrs[k]
                            elif (k == "r_coil (metres)"):
                                rcoil = dset.attrs[k]
                            elif (k == "r_wg (metres)"):
                                rwg = dset.attrs[k]
                            elif (k == "x start (metres)"):
                                xstart = dset.attrs[k]
                            elif (k == "y start (metres)"):
                                ystart = dset.attrs[k]
                            elif (k == "z start (metres)"):
                                zstart = dset.attrs[k]
                            elif (k == "z_max (metres)"):
                                zmax = dset.attrs[k]
                            elif (k == "B_bkg (tesla)"):
                                Bbkg = dset.attrs[k]
                            
                            else:
                                print("Unrecognised attribute")

                        # Print dataset attributes
                        print(f'Theta = {pitchAngle:.2f} degrees, radius = {rad * 1e3:.3f} mm')
                        print(f'Dataset size, shape = {dset.size}, {dset.shape}')
                        print(f'Start Position = ({xstart:.5f},{ystart:.5f},{zstart})')
                        print(f'Start Energy = {startE} eV')
                        print(f'Start freq = {startFdown}')
                        print(f'True start freq = {startF}')
                        print(f'Z_max = {zmax}')

                        B = 0.6966
                        c = 299792458
                        L = 0.2178
                        fs = 1e9
                        num_per_seg = 2**15
                        # Define constants
                        ele_charge = 1.60217663e-19 # C
                        eps = 8.85418782*1e-12 # m^-3 kg^-1 s^4 A^2
                        mass = 9.1093837*1e-31 #kg
#(L/np.tan(pitchAngle/360*2*np.pi))
                        if pitchAngle >= 50:
                            #energy = ((ele_charge*B*c**2)/(2*np.pi*startF))*(1+zmax**2/(2*L**2)) - mass*c**2
                            energy = (ele_charge*B*c**2)/(2*np.pi*startF) - mass*c**2
                            energy_keV = energy/(1.60218e-19) * 1e-3
                            print (f'The corrected energies are: {energy_keV}')
                            Energies.append(energy_keV)

                        else:
                            pass
                        startFs.append(startF)

                        # Compute spectrogram
                        freq, time, spectrogram = signal.spectrogram(np.array(dset), fs=fs, window='hann',nperseg=num_per_seg, noverlap=0, nfft=None, 
detrend='constant', return_onesided=True, scaling='density', 
axis=-1, mode='psd')

                        # Plot power density spectrum
                        freqs, psd = signal.welch(np.array(dset), fs=fs, window='hann',nperseg=num_per_seg, noverlap=0, nfft=None,
detrend='constant', return_onesided=True, scaling='density', 
axis=-1)
                        

                        peaks, _ = signal.find_peaks(psd, height=0.05*1e-22)

                        # Find highest peak

                        # Find the index from the maximum peak
                        max_peak_index = peaks[np.argmax(psd[peaks])]

                        # Find the x value from that index
                        freqs_max = freqs[max_peak_index]
                        psd_max = psd[max_peak_index]

                        # Work out theoretical slope
                        exp_slope = (np.sum(spectrogram)*startF*1e-6)/(mass*c**2+startE*ele_charge*1e3)
                        

                        # fit main peak with gaussian
                        # Define a Gaussian function
                        def gaussian(x, A, mu, sigma):
                            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

                        fit_results = []
                        # Calculate Full Width at Half Maximum (FWHM) from peak data
                        half_max = psd_max / 2
                        left_idx = np.where(psd[:max_peak_index] <= half_max)[0][-1]
                        right_idx = np.where(psd[max_peak_index:] <= half_max)[0][0] + max_peak_index
                        fwhm = freqs[right_idx] - freqs[left_idx]

                        # Initial guess for parameters: Amplitude = height of peak, Mean = frequency at peak, Sigma = FWHM / (2 * sqrt(2 * ln(2)))
                        guess_params = [psd[max_peak_index], freqs[max_peak_index], fwhm / (2 * np.sqrt(2 * np.log(2)))]
        
                        try:
                            # Fit the Gaussian using curve_fit
                            params, _ = curve_fit(gaussian, freqs, psd, p0=guess_params)
                            fit_results.append(params)
                        except RuntimeError:
                            print(f"Fit failed for peak at frequency {freqs[peak]}")

                        # Extract peak frequencies and widths from the Gaussian fit
                        peak_frequency = [params[1] for params in fit_results]
                        peak_width = [params[2] for params in fit_results]
                        left_boundary = [params[1] - 4 * params[2] for params in fit_results]  # mean - 4 * sigma
                        right_boundary = [params[1] + 4 * params[2] for params in fit_results]

                        # Calculate frequency resolution (delta_f)
                        delta_f = fs / num_per_seg

                        # Convert frequency range to bins
                        lower_bin = float(left_boundary[0]) / delta_f
                        upper_bin = float(right_boundary[0]) / delta_f

                        # Round the limits to nearest integer bins
                        lower_bin = int(np.round(lower_bin))
                        upper_bin = int(np.round(upper_bin))
                    
                        if left_boundary < startFdown < right_boundary:
                            mainslopes.append(exp_slope)
                            # work out power main
                            main_power_band = np.sum(spectrogram[lower_bin:upper_bin:1,:])
                            mainpowers.append(main_power_band)
                            # Plot the original power spectra
##                            plt.figure()
##                            plt.plot(freqs*1e-6, psd, label='Power Spectra')

                            # Plot the Gaussian curve
                            x_values = np.linspace(startFdown / 1e6 - 1, startFdown / 1e6 + 1, 1000)  # Generate x values for smoother curve
                            if fit_results:  # Check if there are fit results
                                params = fit_results[0]  # Assume only one set of fit parameters
                                gaussian_curve = gaussian(x_values, *params)
##                                plt.plot(x_values, gaussian_curve, label='Gaussian Fit', color='red')

                                # Add labels and legend
                                # Plot vertical lines for closest left and right boundaries
##                                plt.xlim(round(startFdown / 1e6 - 1), round(startFdown / 1e6 + 1))
##                                plt.axvline(x=left_boundary[0] * 1e-6, color='r', linestyle='--', label='Closest Left Boundary')
##                                plt.axvline(x=right_boundary[0] * 1e-6, color='g', linestyle='--', label='Closest Right Boundary')
##                                plt.xlabel('Frequency (MHz)')
##                                plt.ylabel('Power Spectral Density (W/Hz)')
##                                plt.legend()
##                                plt.show(block=False)

                            


                        else:
                            sideslopes.append(exp_slope)
                            main_power_band = np.sum(spectrogram[lower_bin:upper_bin:1,:])
                            sidepowers.append(main_power_band)

                                                    
                      
                
        else:
            print("File cannot be opened")

        # Close the file
        f.close()

        plt.show(block=False)

# Find percentage of data with highest power in main band
        
num_bins = 300

# best fit of data
(mu, sigma) = norm.fit(Energies)

# the histogram of the data
n, bins, patches = ax.hist(Energies, num_bins, density=True)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
   np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
print (Energies)
y = norm.pdf( bins, mu, sigma)
#ax.plot(bins, y, '--')
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of normal distribution sample')

# Tweak spacing to prevent clipping of ylabel
#fig.tight_layout()
#plt.show(block=False)
##plt.figure()
##plt.scatter(sideslopes, sidepowers, color='green', label = 'Sidebands')
##plt.scatter(mainslopes, mainpowers, color='blue', label = 'Mainbands')
##plt.xlabel('Slopes (MHz/ms)')
##plt.ylabel('Power Spectral Density (W/Hz)')
##plt.legend()
plt.show()


# Find peaks in power spectra and fit function to them. Locate the lhs value of the peak to get the cyclotron frequency
# Locate the spectral lines in spectrogram and fit them to function, locate the main track and

# Wait main track is at carrier frequency or what?
# What is going on with my noise?
# Am i meant to clean the noise first before obtaining the values
# How to do for ones where the carrier frequency has 0 power
# Whaaaaaatttttt how do i get a spectrum



