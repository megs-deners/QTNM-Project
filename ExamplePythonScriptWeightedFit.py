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
#filenames = ["/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_04974749-daa1-48f2-b069-82728a64cd84.h5"]
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
### Check if file can be opened command+control+3 or 4 to remove

power_spec_density = []
slopesmain = []
slopes1 = []
slopes2 = []
startFdowns = []
powers_main = []
main_band_psd = []
side1_band_psd = []
side2_band_psd = []
pitchmain = []
pitch1 = []
pitch2 = []
startFs = []
Energies = []
pitches = []
axFreqsmain = []
axFreq1 = []
axFreq2 = []
radmain=[]
rad1=[]
rad2=[]
startFmain=[]
startF1=[]
startF2=[]
freqss=[]
Energiesfit=[]
freqdifferences=[]
plt.figure()

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
                            elif (k == "'z_max (metres)"):
                                zmax = dset.attrs[k]
                            
                            else:
                                print("Unrecognised attribute")

                        # Print dataset attributes
                        print(f'Theta = {pitchAngle:.2f} degrees, radius = {rad * 1e3:.3f} mm')
                        print(f'Dataset size, shape = {dset.size}, {dset.shape}')
                        print(f'Start Position = ({xstart:.5f},{ystart:.5f},{zstart})')
                        print(f'Start Energy = {startE} eV')
                        print(f'Start freq = {startFdown}')
                              

                        # Create an array of time values
                        times = np.linspace(0, (dset.size - 1.0) * deltaT * 1e6, dset.size)


                        # Replace or remove invalid characters in the filename
                        sanitized_filename = filename.replace("/", "_").replace(".h5", "")
                        # Save the plot

                        k = 1.38e-23
                        
                        B = 500*1e6
                        
                       

                        # Function to add thermal Gaussian noise to the signal
                        def add_thermal_noise(signal, noise_temp):
                            # Calculate the power of the original signal
                            #signal_power = np.mean(signal**2)

                            # Calculate the noise power
                            #noise_power = noise_temp*(k*fs/num_per_seg) 
                            
                            # Find std
                            sigma = np.sqrt(noise_temp*k*B)
                            
                            # Generate Gaussian noise with the calculated power
                            noise = np.random.normal(0, sigma, len(signal))
                            
                            # Add noise to the signal
                            noisy_signal = signal + noise
                            
                            return noisy_signal, sigma

                        # Add thermal noise to the signal
                        noisy_data, sigma = add_thermal_noise(np.array(dset),0)
                        

                        fs = 1e9
                        num_per_seg = 2**15
                        
                        # Compute spectrogram
                        freq, time, spectrogram = signal.spectrogram(noisy_data, fs=fs, window='hann',nperseg=num_per_seg, noverlap=0, nfft=None, 
    detrend='constant', return_onesided=True, scaling='density', 
    axis=-1, mode='psd')

                        window_size = 2  # Define the size of the frequency bin window
                        max_power_sum = 0
                        max_power_window = None
                        max_power_window_index = None

                        # Iterate through the spectrogram to calculate power within each window
                        for i in range(0, spectrogram.shape[0] - window_size + 1):
                            # Extract the window of bins
                            window_bins = spectrogram[i:i+window_size, :]
                            
                            # Calculate the sum of power within the window
                            power_sum = np.sum(window_bins)
                            
                            # Check if the current window has higher power than the previous maximum
                            if power_sum > max_power_sum:
                                max_power_sum = power_sum
                                max_power_window = window_bins
                                max_power_window_index = i

                        freq_range_start = freq[max_power_window_index]
                        
                        power_spec_density.append(max_power_sum)

                        # If you want to find the frequency range corresponding to the max power window,
                        # you can find the frequency range covered by the window
                        # For example, if freqs contain the frequency values:
                        # freq_range = (freqs[i], freqs[i + window_size - 1])

                        # Now max_power_window contains the spectrogram of the window with the highest power
                        # You can process it further as needed.

                        

                        # Make all lower powers 0
                        #power_threshold = (np.amax(spectrogram)-np.amin(spectrogram))*0.2  # Adjust the threshold value as needed
                        # Set values below the power threshold to zero
                        #spectrogram[spectrogram < power_threshold] = 1e-40

                        # Define a Gaussian function
                        def gaussian(x, A, mu, sigma):
                            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



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

                        

                        #plt.figure(dpi=200)
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.colorbar(label='Power/Frequency')
                        #plt.scatter(0, freqs_max*1e-6,marker='x', color='blue')
                        #plt.scatter(0, freq_range_start*1e-6,marker='x', color='red')
                        #plt.title(f'Spectrogram for {dsetName} for highest power peak')
                        #plt.ylabel('Frequency band (MHz)')
                        #plt.xlabel('Time window (microseconds)')
                        #plt.ylim(round(startFdown / 1e6 - 1), round(startFdown / 1e6 + 1))
                        #plt.show(block=False)

                        



                        # Define constants
                        ele_charge = 1.60217663*1e-19 # C
                        c = 3*1e8 # m s-1
                        eps = 8.85418782*1e-12 # m^-3 kg^-1 s^4 A^2
                        mass = 9.1093837*1e-31 #kg

                        # Find predicted power
                        elec_speed=np.sqrt((2*startE*ele_charge)/mass) # J kg^-1
                        beta = elec_speed/c
                        P_exp = ((2*np.pi*ele_charge**2*startF**2)/(3*eps*c))*((beta**2*np.sin(math.radians(pitchAngle))**2)/(1-beta**2))
                        print(f'Experimental power = {P_exp}')

                        # Work out theoretical slope
                        gamma = 1/np.sqrt(1-beta**2)
                        exp_slope = P_exp*startF*1e-6/(mass*c**2+startE*ele_charge*1e3)
                        print (f'Theoretical slope is {exp_slope}MHz/ms')
# brahsta energy
                        # Fit a Gaussian to each peak
                        fit_results = []
                        for peak in peaks:

                            # Calculate Full Width at Half Maximum (FWHM) from peak data
                            half_max = psd[peak] / 2
                            left_idx = np.where(psd[:peak] <= half_max)[0][-1]
                            right_idx = np.where(psd[peak:] <= half_max)[0][0] + peak
                            fwhm = freqs[right_idx] - freqs[left_idx]

                            # Initial guess for parameters: Amplitude = height of peak, Mean = frequency at peak, Sigma = FWHM / (2 * sqrt(2 * ln(2)))
                            guess_params = [psd[peak], freqs[peak], fwhm / (2 * np.sqrt(2 * np.log(2)))]
        
                            try:
                                # Fit the Gaussian using curve_fit
                                params, _ = curve_fit(gaussian, freqs, psd, p0=guess_params)
                                fit_results.append(params)
                            except RuntimeError:
                                print(f"Fit failed for peak at frequency {freqs[peak]}")

                        # Extract peak frequencies and widths from the Gaussian fit
                        peak_frequencies = [params[1] for params in fit_results]
                        peak_widths = [params[2] for params in fit_results]
                        left_boundaries = [params[1] - 4 * params[2] for params in fit_results]  # mean - 4 * sigma
                        right_boundaries = [params[1] + 4 * params[2] for params in fit_results]

    # Remove close boundaries

                        threshold = 2e6  # Define the threshold for removing close values

                        # Zip left and right boundaries together for easier comparison
                        boundary_pairs = list(zip(left_boundaries, right_boundaries))

                        # Initialize lists to store filtered boundaries
                        filtered_left_boundaries = []
                        filtered_right_boundaries = []

                        # Iterate through boundary pairs and filter out close values
                        for left, right in reversed(boundary_pairs):
                            if not filtered_left_boundaries or abs(left - filtered_left_boundaries[-1]) > threshold:
                                filtered_left_boundaries.append(left)
                                filtered_right_boundaries.append(right)
                        print(left_boundaries, filtered_left_boundaries)  
                        # Convert filtered_left_boundaries to a NumPy array
                        filtered_left_boundaries = list(reversed(filtered_left_boundaries))
                        filtered_right_boundaries = list(reversed(filtered_right_boundaries))
                    
    # Find left and right boundaries closest to start frequency

                        # Calculate absolute differences between start frequency and boundaries
                        left_differences = [abs(startFdown - boundary) for boundary in filtered_left_boundaries if boundary < startFdown]

                        # Find index of the minimum absolute difference for left and right boundaries
                        closest_left_index = left_differences.index(min(left_differences))

                        # Get the corresponding left and right boundaries
                        closest_left_boundary = filtered_left_boundaries[closest_left_index]
                        closest_right_boundary = filtered_right_boundaries[closest_left_index]
                        # megan tomorrow, find power under central peak using power density spectrum

                        # Calculate frequency resolution (delta_f)
                        delta_f = fs / num_per_seg


                        # Convert frequency range to bins
                        lower_bin = closest_left_boundary / delta_f
                        upper_bin = closest_right_boundary / delta_f

                        # Round the limits to nearest integer bins
                        lower_bin = int(np.round(lower_bin))-1
                        upper_bin = int(np.round(upper_bin))-1

                        # Attempt to fit lines correctly
                        
                        # work out power main
                        main_power_band = np.sum(spectrogram[lower_bin:upper_bin:1,:])

                        # Gather your known data: x, y, and n all in the same order as each other

                        new_time = np.tile(time, len(freqs[lower_bin:upper_bin:1]))
                        new_freqs =  np.repeat(freqs[lower_bin:upper_bin:1], len(time))
                        A = np.column_stack((new_time.flatten(), np.ones(len(new_time))))  # Here are the x values from your histogram
                        b = new_freqs.flatten()  # Here are the y-values from your histogram
                        C = np.diag(((spectrogram[lower_bin:upper_bin:1, :])).flatten())  # Counts from each pixel in your 2D histogram
                        print (A.shape, b.shape, C.shape)
                        print (time.shape, freqs.shape, spectrogram[lower_bin:upper_bin:1, :].shape)

                        # Define polynomial coefficients as p = [slope, y_offset];

                        # Usual least-squares solution
                        # b = A.dot(p)  # Remember, p = [slope, y_offset];
                        # p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # Remember, p = [slope, y_offset];

                        # We want to apply a weighting matrix, so incorporate the weighting matrix
                        # A.T.dot(b) = A.T.dot(C).dot(A).dot(p);  
                        p = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T.dot(C).dot(b))  # Remember, p = [slope, y_offset];
                        y0 = p[0]*new_time+p[1]

                        # Calculate residuals
                        residuals = b - y0

                        # Calculate variance of residuals
                        residual_var = np.sum(residuals**2) / (len(b) - 2)  # 2 is the number of coefficients (slope and y-intercept)

                        # Calculate covariance matrix of regression coefficients
                        cov_matrix = residual_var * np.linalg.inv(A.T.dot(C).dot(A))

                        # Standard error of the y-intercept (second coefficient)
                        se_y_intercept = np.sqrt(cov_matrix[1, 1])

                        # Expected y-intercept within sigma
                        expected_y_intercept = startFdown  # Adjust this based on your expected value

                        # Compare to expected y-intercept within sigma
                        sigma = 1 # Define the number of standard deviations
                        if abs(p[1] - expected_y_intercept) <= sigma * se_y_intercept:
                            print("The startF is within", sigma, "sigma of the expected value.")
                        else:
                            print("The startF is not within", sigma, "sigma of the expected value.")
                        
                        print (f'Main Slope = {p[0]},Intercept = {p[1]}')

                        freqsdiff = p[1] - expected_y_intercept
                        print (freqsdiff)
                        

                        freqss.append(p[1])
                        e = 1.60217663e-19
                        B = 0.6966
                        c = 299792458
                        me = 9.1093837e-31
                        L = 0.2178

                        # Do energy reconstruction using weighted fit

                        energy = (e*B*c**2)/(2*np.pi*(startF-startFdown+p[1])) - me*c**2
                        energy_keV = energy/(1.60218e-19) * 1e-3
                        Energiesfit.append(energy_keV)

##
##    # Analyse first sideband
##                        # Get left and right boundaries of two sets of first sidebands
##                        if closest_left_index - 1 >= 0:
##                            side1_left_boundary_set1 = filtered_left_boundaries[closest_left_index - 1]
##                            side1_right_boundary_set1 = filtered_right_boundaries[closest_left_index - 1]
##
##                            # Convert frequency range to bins
##                            lower_bin = side1_left_boundary_set1 / delta_f
##                            upper_bin = side1_right_boundary_set1 / delta_f
##
##                            # Round the limits to nearest integer bins
##                            lower_bin = int(np.round(lower_bin))
##                            upper_bin = int(np.round(upper_bin))
##                            
##                            # work out power main
##                            main_power_band11 = np.sum(spectrogram[lower_bin:upper_bin:1,:])
##
##                            # Gather your known data: x, y, and n all in the same order as each other
##
##                            new_time = np.tile(time, len(freqs[lower_bin:upper_bin:1]))
##                            new_freqs =  np.repeat(freqs[lower_bin:upper_bin:1], len(time))
##                            A = np.column_stack((new_time.flatten(), np.ones(len(new_time))))  # Here are the x values from your histogram
##                            b = new_freqs.flatten()  # Here are the y-values from your histogram
##                            C = np.diag(((spectrogram[lower_bin:upper_bin:1, :]**2)).flatten())  # Counts from each pixel in your 2D histogram
##                            print (A.shape, b.shape, C.shape)
##                            print (time.shape, freqs.shape, spectrogram[lower_bin:upper_bin:1, :].shape)
##
##                            # Define polynomial coefficients as p = [slope, y_offset];
##
##                            # Usual least-squares solution
##                            # b = A.dot(p)  # Remember, p = [slope, y_offset];
##                            # p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # Remember, p = [slope, y_offset];
##
##                            # We want to apply a weighting matrix, so incorporate the weighting matrix
##                            # A.T.dot(b) = A.T.dot(C).dot(A).dot(p);  
##                            p = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T.dot(C).dot(b))  # Remember, p = [slope, y_offset];
##                            y11 = p[0]*time+p[1]
##                            
##                            print (f' 1st sideband Slope = {p[0]},Intercept = {p[1]}')
##                            
##                        if closest_left_index + 1 < len(filtered_left_boundaries):
##                            side1_left_boundary_set2 = filtered_left_boundaries[closest_left_index+1]
##                            side1_right_boundary_set2 = filtered_right_boundaries[closest_left_index+1]
##
##                            # Convert frequency range to bins
##                            lower_bin = side1_left_boundary_set2 / delta_f
##                            upper_bin = side1_right_boundary_set2 / delta_f
##
##                            # Round the limits to nearest integer bins
##                            lower_bin = int(np.round(lower_bin))
##                            upper_bin = int(np.round(upper_bin))
##
##                            # work out power main
##                            main_power_side12 = np.sum(spectrogram[lower_bin:upper_bin:1,:])
##
##
##                            # Gather your known data: x, y, and n all in the same order as each other
##
##                            new_time = np.tile(time, len(freqs[lower_bin:upper_bin:1]))
##                            new_freqs =  np.repeat(freqs[lower_bin:upper_bin:1], len(time))
##                            A = np.column_stack((new_time.flatten(), np.ones(len(new_time))))  # Here are the x values from your histogram
##                            b = new_freqs.flatten()  # Here are the y-values from your histogram
##                            C = np.diag(((spectrogram[lower_bin:upper_bin:1, :]**2)).flatten())  # Counts from each pixel in your 2D histogram
##                            print (A.shape, b.shape, C.shape)
##                            print (time.shape, freqs.shape, spectrogram[lower_bin:upper_bin:1, :].shape)
##
##                            # Define polynomial coefficients as p = [slope, y_offset];
##
##                            # Usual least-squares solution
##                            # b = A.dot(p)  # Remember, p = [slope, y_offset];
##                            # p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # Remember, p = [slope, y_offset];
##
##                            # We want to apply a weighting matrix, so incorporate the weighting matrix
##                            # A.T.dot(b) = A.T.dot(C).dot(A).dot(p);  
##                            p = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T.dot(C).dot(b))  # Remember, p = [slope, y_offset];
##                            y12 = p[0]*time+p[1]
##                            
##                            print (f'1st side band Slope = {p[0]},Intercept = {p[1]}')
##                        
##                        
##
##
##    # Analyse second sideband
##                        # Get left and right boundaries of two sets of first sidebands
##                        if closest_left_index + 2 < len(filtered_left_boundaries):
##                            side2_left_boundary_set2 = filtered_left_boundaries[closest_left_index + 2]
##                            side2_right_boundary_set2 = filtered_right_boundaries[closest_left_index + 2]
##
##                            
##                            # Convert frequency range to bins
##                            lower_bin = side2_left_boundary_set2 / delta_f
##                            upper_bin = side2_right_boundary_set2 / delta_f
##
##                            # Round the limits to nearest integer bins
##                            lower_bin = int(np.round(lower_bin))
##                            upper_bin = int(np.round(upper_bin))
##
##                            # work out power main
##                            main_power_side22 = np.sum(spectrogram[lower_bin:upper_bin:1,:])
##
##                            # Gather your known data: x, y, and n all in the same order as each other
##
##                            new_time = np.tile(time, len(freqs[lower_bin:upper_bin:1]))
##                            new_freqs =  np.repeat(freqs[lower_bin:upper_bin:1], len(time))
##                            A = np.column_stack((new_time.flatten(), np.ones(len(new_time))))  # Here are the x values from your histogram
##                            b = new_freqs.flatten()  # Here are the y-values from your histogram
##                            C = np.diag(((spectrogram[lower_bin:upper_bin:1, :]**2)).flatten())  # Counts from each pixel in your 2D histogram
##                            print (A.shape, b.shape, C.shape)
##                            print (time.shape, freqs.shape, spectrogram[lower_bin:upper_bin:1, :].shape)
##
##                            # Define polynomial coefficients as p = [slope, y_offset];
##
##                            # Usual least-squares solution
##                            # b = A.dot(p)  # Remember, p = [slope, y_offset];
##                            # p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # Remember, p = [slope, y_offset];
##
##                            # We want to apply a weighting matrix, so incorporate the weighting matrix
##                            # A.T.dot(b) = A.T.dot(C).dot(A).dot(p);  
##                            p = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T.dot(C).dot(b))  # Remember, p = [slope, y_offset];
##                            y22 = p[0]*time+p[1]
##
##                            print (f'2nd side band Slope = {p[0]},Intercept = {p[1]}')
##                            
##                        if closest_left_index - 2 >= 0:
##                            side2_left_boundary_set1 = filtered_left_boundaries[closest_left_index - 2]
##                            side2_right_boundary_set1 = filtered_right_boundaries[closest_left_index - 2]
##
##                            # Convert frequency range to bins
##                            lower_bin = side2_left_boundary_set1 / delta_f
##                            upper_bin = side2_right_boundary_set1 / delta_f
##
##                            # Round the limits to nearest integer bins
##                            lower_bin = int(np.round(lower_bin))
##                            upper_bin = int(np.round(upper_bin))
##
##                            # work out power main
##                            main_power_side21 = np.sum(spectrogram[lower_bin:upper_bin:1,:])
##                            
##                            # Gather your known data: x, y, and n all in the same order as each other
##
##                            new_time = np.tile(time, len(freqs[lower_bin:upper_bin:1]))
##                            new_freqs =  np.repeat(freqs[lower_bin:upper_bin:1], len(time))
##                            A = np.column_stack((new_time.flatten(), np.ones(len(new_time))))  # Here are the x values from your histogram
##                            b = new_freqs.flatten()  # Here are the y-values from your histogram
##                            C = np.diag(((spectrogram[lower_bin:upper_bin:1, :]**2)).flatten())  # Counts from each pixel in your 2D histogram
##                            print (A.shape, b.shape, C.shape)
##                            print (time.shape, freqs.shape, spectrogram[lower_bin:upper_bin:1, :].shape)
##
##                            # Define polynomial coefficients as p = [slope, y_offset];
##
##                            # Usual least-squares solution
##                            # b = A.dot(p)  # Remember, p = [slope, y_offset];
##                            # p = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(b))  # Remember, p = [slope, y_offset];
##
##                            # We want to apply a weighting matrix, so incorporate the weighting matrix
##                            # A.T.dot(b) = A.T.dot(C).dot(A).dot(p);  
##                            p = np.linalg.inv(A.T.dot(C).dot(A)).dot(A.T.dot(C).dot(b))  # Remember, p = [slope, y_offset];
##                            y21 = p[0]*time+p[1]
##                            
##                            print (f'2nd side band Slope = {p[0]},Intercept = {p[1]}')
##                        else:
##                            pass
##                            # Handle the case where the index is negative
##                            side2_left_boundary_set1 = 0 # or any other appropriate action
##                            side2_right_boundary_set1 = 0  # or any other appropriate action
##                            side1_left_boundary_set1 = 0  # or any other appropriate action
##                            side1_right_boundary_set1 = 0  # or any other appropriate action
##                            side1_left_boundary_set2 = 0  # or any other appropriate action
##                            side1_right_boundary_set2 = 0
##                            side2_left_boundary_set2 = 0  # or any other appropriate action
##                            side2_right_boundary_set2 = 0



    # From Example Python 5 to find slope and power spectral density

                # Calculate power spectral density and allocate to slope list
                        print (freqs_max)

                        if closest_left_boundary < freqs_max < closest_right_boundary:
                            main_band_psd.append(main_power_band)
                            slopesmain.append(exp_slope)
                            pitchmain.append(pitchAngle)
                            print(f'Highest frequency is in main band, with pitch Angle = {pitchAngle}')

                            # find start frequency in first time bin
                            # Find frequency with highest power in the first time bin
                            index_max_power = np.argmax(spectrogram[:, 0])  # Index of max power in first time bin
                            freq_max_power = freq[index_max_power]  # Corresponding frequency

                            e = 1.60217663e-19
                            B = 0.6966
                            c = 299792458
                            me = 9.1093837e-31
                            L = 0.2178


                            energy = (e*B*c**2)/(2*np.pi*freq_max_power) - me*c**2
                            energy_keV = energy/(1.60218e-19) * 1e-3
                            Energies.append(energy_keV)
                            startFs.append(freq_max_power)
                            pitches.append(pitchAngle)
                            axFreqsmain.append(axFreq)
                            startFmain.append(startF)
                            radmain.append(rad)
                            freqdifferences.append(freqsdiff)
                                

                        else:
                            #side2_band_psd.append(main_power_side21)
                            #slopes2.append(exp_slope)
                            #pitch2.append(pitchAngle)
                            #axFreqs2.append(axFreq)
                            print(f'Highest frequency not in first 2 sidebands or main, with pitch Angle = {pitchAngle}')




                        
                        

            # Repeat for 1st sideband:

                        

                        

            # Repeat for 2nd set of sidebands

                        
                        

                # Move back up

                        
                        # Check these are correct
                        #plt.figure()
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.colorbar(label='Power/Frequency')
                        #plt.title(f'Side band for {dsetName} for slope=')
                        #plt.ylabel('Frequency band (MHz)')
                        #plt.xlabel('Time window (microseconds)')
                        #plt.ylim(round(side1_left_boundary_set1 / 1e6 - 1), round(side1_right_boundary_set1 / 1e6 + 1))
                        #plt.scatter(0, side1_left_boundary_set1*1e-6, marker='x', color='red', label='Start Frequency')
                        #plt.plot(time*1e6, y11*1e-6)
                        #plt.scatter(1000, side1_right_boundary_set1*1e-6, marker='x', color='blue', label='Start Frequency')
                        #plt.scatter(0, side1_left_boundary_set2*1e-6, marker='x', color='red', label='Start Frequency')
                        #plt.scatter(1000, side1_right_boundary_set2*1e-6, marker='x', color='blue', label='Start Frequency')
                        #plt.show(block=False)

                        # Check these are correct
                        #plt.figure()
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.colorbar(label='Power/Frequency')
                        #plt.title(f'Main band for {dsetName} for slope=')
                        #plt.ylabel('Frequency band (MHz)')
                        #plt.xlabel('Time window (microseconds)')
                        #plt.ylim(round(closest_left_boundary / 1e6 - 1), round(closest_right_boundary / 1e6 + 1))
                        #plt.scatter(0, closest_left_boundary*1e-6, marker='x', color='red', label='Start Frequency')
                        #plt.scatter(1000, closest_right_boundary*1e-6, marker='x', color='blue', label='Start Frequency')
                        #plt.plot(time, y0)
                        #plt.show(block=False)
                        
                        # Plot fit line on plot
                        #plt.figure(dpi=200)
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.colorbar(label='Power/Frequency')
                        #plt.title(f'THISSS for {dsetName} for slope={slope}')
                        #plt.ylabel('Frequency band (MHz)')
                        #plt.xlabel('Time window (microseconds)')
                        #plt.ylim(round(startFdown / 1e6 - 1), round(startFdown / 1e6 + 1))
                        #plt.scatter(1000, closest_right_boundary*1e-6, marker='x', color='blue', label='Start Frequency')
                        #plt.show(block=False)
                        #plt.figure(dpi=200)
                        #plt.plot(time*1e6, y*1e-6)
                        #plt.ylim(round(startFdown / 1e6 - 1), round(startFdown / 1e6 + 1))
                        #plt.show()

                # Plot spectrogram with slope ends boundaries marked

                        #plt.figure(dpi=200)
                        #plt.imshow(spectrogram, aspect='auto', cmap='bone_r', origin='lower')
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.colorbar(label='Power/Frequency')
                        #plt.title(f'THISSS for {dsetName} for slope={slope}')
                        #plt.ylabel('Frequency band (MHz)')
                        #plt.xlabel('Time window (microseconds)')
                        #plt.ylim(round(startFdown / 1e6 - 1), round(startFdown / 1e6 + 1))
                    
                        #print (freq.shape)
                        # Plot the spectrogram with a marker at the peak frequency
                        #plt.pcolormesh(time*1e6, freq*1e-6, spectrogram, shading='gouraud', cmap='bone_r')
                        #plt.scatter(0, *1e-6, marker='x', color='red', label='Peak Frequency')
                        

                        # Plot the spectrogram with a marker at the start frequency
                        #plt.scatter(1000, closest_right_boundary*1e-6, marker='x', color='blue', label='Start Frequency')
                        #plt.show(block=False)


                        
                
        else:
            print("File cannot be opened")

        # Close the file
        f.close()

        plt.show(block=False)

if len(startFdowns) == len(freqss):
    error = abs(startFdowns-freqss)/startFdowns
    aveg_error = np.mean(error)*100

    print (f'The average percentage error of the fit = {aveg_error}')
print(freqdifferences)
# Threshold value
threshold = 0.25 * 1e6

# Filter values based on the threshold
filtered_freqdifferences = [i*1e-6 for i in freqdifferences if abs(i) <= threshold]

# Plot histogram of start frequency - obtained frequency
plt.figure()
counts, bins, _ = plt.hist(filtered_freqdifferences, bins=100, color='skyblue', edgecolor='black', alpha=0.7, label='Histogram')

# Fit a Gaussian to the histogram data
bin_centers = 0.5 * (bins[:-1] + bins[1:])
popt, _ = curve_fit(gaussian, bin_centers, counts, p0=[0, 1, 1])  # Initial guess for parameters: mu=0, sigma=1, A=1

# Plot the fitted Gaussian
plt.plot(bin_centers, gaussian(bin_centers, *popt), color='red', label=r'Fitted Gaussian: $\mu={:.2f}$, $\sigma={:.2f}$, $A={:.2f}$'.format(*popt))

# Add labels, title, and legend
plt.xlabel(r'$f_{pred} - f_{exp}$ (MHz)')
plt.ylabel('Number of events')
plt.legend()

# Show plot
plt.show(block=False)

num_bins = 100

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(Energiesfit, num_bins, density=True)
print (Energies)

ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of normal distribution sample')


# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()
#plt.savefig(f"energies_{sanitized_filename}.png")



# Find peaks in power spectra and fit function to them. Locate the lhs value of the peak to get the cyclotron frequency
# Locate the spectral lines in spectrogram and fit them to function, locate the main track and

# Wait main track is at carrier frequency or what?
# What is going on with my noise?
# Am i meant to clean the noise first before obtaining the values
# How to do for ones where the carrier frequency has 0 power
# Whaaaaaatttttt how do i get a spectrum



