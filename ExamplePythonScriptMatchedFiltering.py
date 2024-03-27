#!/usr/bin/python3

'''
CheckPythonOutputs.py

Checks the outputted times and voltages are ok
'''

import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import chirp, convolve, correlate
from scipy import signal
import scipy.linalg as sl
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
mpl.rcParams.update(mpl.rcParamsDefault)

#plt.rcParams.update({
    #"text.usetex": True,
    #"font.family": "Helvetica"
#})
#filename = "/unix/qtnm/sjones/outputs/WaveguideSignalGen/v3/out_20935849-7759-4cf9-883c-b62b94aa9d08.h5"
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
#initialise list for % of good reconstructions from matched filtering technique
false = 0
true = 0

for filename in filenames:

    templates=[]
    # Check if file can be opened
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

                        # Generate templates
                        templates.append(np.array(dset))

                    else:
                        pass

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
                            
                            else:
                                print("Unrecognised attribute")

                        # Print dataset attributes
                        print(f'Theta = {pitchAngle:.2f} degrees, radius = {rad * 1e3:.3f} mm')
                        print(f'Dataset size, shape = {dset.size}, {dset.shape}')
                        print(f'Start Position = ({xstart:.5f},{ystart:.5f},{zstart})')
                        print(f'Start Energy = {startE} eV')
                        print(f'Start freq = {startFdown}')
                        print(f'True start freq = {startF}')
                              

                        # Plot the time series data
                        # Create an array of time values
                        times = np.linspace(0, (dset.size - 1.0) * deltaT * 1e6, dset.size)
                        print(f"{times.size} time values")
                        #plt.figure()
                        #plt.plot(times, np.array(dset))
                        #plt.title(f'$\\theta = {pitchAngle:.2f}$ degrees, $r_i = {rad * 1e3:.2f}$ mm, $f_s = {startF / 1e6:.0f}$ MHz for signal ={dsetName}')
                        #plt.ylabel('Voltage [A. U.]')
                        #plt.xlabel(f'Time [$\mu$s]')
                        #plt.xlim(1-0.01,1)
                        # Replace or remove invalid characters in the filename
                        sanitized_filename = filename.replace("/", "_").replace(".h5", "")
                        # Save the plot
                        #plt.savefig(f"output_plot_{sanitized_filename}.png")

                        fs = 1e9
                        num_per_seg = 2**15
                        
                        freq, time, spectrogram = signal.spectrogram(np.array(dset), fs=fs, window='hann',nperseg=num_per_seg, noverlap=0, nfft=None, 
    detrend='constant', return_onesided=True, scaling='spectrum', 
    axis=-1, mode='psd')

                        
                        # Find gradient using equation Esfahani paper
                        slope = np.sum(spectrogram[:,0])*(startF)/((9.11e-31*(3e8)**2+startE*1.602e-19)*1e3)
                        
                        # Find peaks in the last time bin
                        last_time_bin = spectrogram[:, -1]
                        #print (last_time_bin)
                        peaks, _ = signal.find_peaks(last_time_bin, height=0.05*1e-17)
                        last_time = np.linspace(1000,1000,len(peaks))
                        

                        # Find the closest frequency peak to startF
                        closest_peak_diff = np.argmin(np.abs(freq[peaks] - startFdown))
                        closest_peak_freq = closest_peak_diff+startFdown
                        print (closest_peak_freq)
##
##                        def chirping(t, f0, T, f1):
##                            '''Define a linear chirp signal using inputs t - time array, f0 - initial frequency, T - final time, f1- freqeuncy at final time'''
##                            c = (f1-f0)/T
##                            # assuming phase = 0
##                            x = np.sin(2*np.pi*((c/2)*t**2+f0*t))
##                            
##                            return x
##                        
####                        #template = chirping(times, startFdown, times[-1], startFdown+slope)
##                        template = np.array(dset)
##    ##                    [0:int(dset.size*0.2)+1]
##
##                        d, path = fastdtw(template, template, dist=euclidean)
##                        print(d)
                        

                        #plt.figure()
                        #plt.plot(times[0:int(dset.size*0.2)+1], template)
                        #plt.title(f'Template')
                        #plt.ylabel('Voltage [A. U.]')
                        #plt.xlabel(f'Time [$\mu$s]')
                        #plt.xlim(0, 0.01)
                        #plt.show(block=False)
                        

                        k = 1.38e-23
                        
                        B = 500*1e6
                        
                        noise_values = [9]

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


                        def compute_mse(original_signal, filtered_signal):
                            # Compute the squared differences between corresponding elements
                            squared_diff = (original_signal - filtered_signal) ** 2
                            
                            # Compute the mean squared error
                            mse = np.mean(squared_diff)
                            
                            return mse

                        mse = []
                        
                        

                        for n in noise_values:
                            
                            # Add thermal noise to the signal
                            noisy_data, sigma = add_thermal_noise(np.array(dset),n)

                                # Compute the spectrogram
##                                noisyfreq, noisytime, noisyspectrogram = signal.spectrogram(noisy_data, fs=fs, window='hann',nperseg=num_per_seg, noverlap=0, nfft=None, 
##            detrend='constant', return_onesided=True, scaling='spectrum', 
##            axis=-1, mode='psd')  
                                


                            best_corr = -np.inf
                            best_template = None
                            

                            for template in templates:

                                 # Perform matched filtering
                                matched_filtered = convolve(noisy_data, template, mode='same', method='auto')

                                matched_normalised = matched_filtered / (np.dot(sl.norm(noisy_data),sl.norm(template)))

                                max_corr = np.max(matched_normalised)
                                        
                                    # Check if this template gives a better correlation
                                if max_corr > best_corr:
                                    best_corr = max_corr
                                    best_template = template

                            if np.array_equal(np.array(dset),best_template)==True:
                                true+=1

                            elif np.array_equal(np.array(dset),best_template)==False:
                                false+=1
                                    


##                        # determine reconstruction efficiency
##                            # Calculate frequency resolution (delta_f)
##                            delta_f = fs / num_per_seg
##                            # find the bin with the start freqeuncy
##                            index_startFdown = int(startFdown/delta_f)
##                            # Convert frequency range to bins
##                            lower_bin = int((startFdown - 1e6)/delta_f)
##                            upper_bin = int((startFdown + 1e6)/delta_f)
##
##
##                            noisypowerratio = compute_mse(np.array(dset), noisy_data)
##                            matchedpowerratio = compute_mse(np.array(dset), matched_filtered)
##
##                            
##                            if noisypowerratio > matchedpowerratio:
##                                falserecon.append(dsetName)
##                            elif noisypowerratio < matchedpowerratio:
##                                truerecon.append(dsetName)
##
##                            else:
##                                print(f'Unknown reconstruction efficiency for {dsetName}')
##
##                            print (len(falserecon))
##                            
                                
                            
                        

                                            


        else:
            print("File cannot be opened")

        # Close the file
        f.close()

        plt.show(block=False)

print(true,false)
noise_eff = true/(true+false)*100
print (noise_eff)

##plt.figure()
###plt.bar(noise_values, sum_of_noise_efficencies,color='blue')
##plt.xlabel('Noise Values')
##plt.ylabel('Reconstruction Efficiency')
##plt.show(block=False)
##
##
##plt.figure()
##plt.plot(noise_values, mse, '.-')
###plt.xlim(1.2*1e-10,1.25*1e-10)
##plt.xlabel('Noise Temperature (K)')
##plt.ylabel('Mean squared error (%)')
##plt.show()



# is there an % you can usually neglect when zeroing low powers
