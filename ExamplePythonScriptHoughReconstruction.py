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

startFs = []
Energies = []
pitches = []
axFreqs = []

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
                            
                            else:
                                print("Unrecognised attribute")

                        # Print dataset attributes
                        print(f'Theta = {pitchAngle:.2f} degrees, radius = {rad * 1e3:.3f} mm')
                        print(f'Dataset size, shape = {dset.size}, {dset.shape}')
                        print(f'Start Position = ({xstart:.5f},{ystart:.5f},{zstart})')
                        print(f'Start Energy = {startE} eV')
                        print(f'Start freq = {startFdown}')

                        print (np.array(dset).shape)
                        houghfreq = [273.162841796875, 248.291015625, 299.163818359375, 235.2294921875, 241.455078125, 242.49267578125, 240.631103515625, 245.78857421875, 250.091552734375, 241.485595703125, 235.90087890625, 247.406005859375, 241.668701171875, 241.8212890625, 271.6064453125, 244.44580078125, 238.037109375, 249.69482421875, 239.501953125, 243.896484375, 272.52197265625, 249.176025390625, 241.51611328125, 249.8779296875, 240.997314453125, 239.410400390625, 240.020751953125, 242.7978515625, 242.5537109375, 248.321533203125, 238.34228515625, 249.053955078125, 244.140625, 250.244140625, 239.288330078125, 236.907958984375, 239.013671875, 246.4599609375, 250.091552734375, 240.78369140625, 245.2392578125, 241.851806640625, 338.623046875, 239.532470703125, 249.176025390625, 245.208740234375, 242.5537109375, 249.81689453125, 241.39404296875, 246.27685546875, 245.086669921875, 246.063232421875, 240.692138671875, 251.007080078125, 238.4033203125, 233.245849609375, 242.37060546875, 249.81689453125, 240.997314453125, 249.481201171875, 247.4365234375, 269.53125, 240.29541015625, 245.880126953125, 243.0419921875, 300.35400390625, 242.462158203125, 250.244140625, 238.861083984375, 249.725341796875, 338.409423828125, 246.673583984375, 270.69091796875, 270.904541015625, 250.1220703125, 245.17822265625, 271.91162109375, 245.880126953125, 242.401123046875, 245.78857421875, 234.619140625, 240.692138671875, 236.907958984375, 245.7275390625, 247.589111328125, 243.560791015625, 241.63818359375, 236.4501953125, 248.870849609375, 235.748291015625, 299.896240234375, 246.551513671875, 245.147705078125, 243.2861328125, 240.142822265625, 243.804931640625, 242.889404296875, 253.265380859375, 250.335693359375, 243.83544921875, 248.626708984375, 246.124267578125, 242.3095703125, 270.477294921875, 243.865966796875, 244.93408203125, 248.84033203125, 247.86376953125, 272.705078125, 246.826171875, 247.039794921875, 247.55859375, 270.69091796875, 247.650146484375, 239.349365234375, 242.431640625, 250.67138671875, 249.51171875, 244.9951171875, 252.593994140625, 241.851806640625, 249.3896484375, 249.45068359375, 238.4033203125, 238.4033203125, 245.513916015625, 241.973876953125, 242.73681640625, 240.325927734375, 243.865966796875, 245.208740234375, 250.518798828125, 250.091552734375, 240.386962890625, 240.997314453125, 241.851806640625, 245.1171875, 243.22509765625, 246.4599609375, 239.80712890625, 247.161865234375, 240.8447265625, 246.826171875, 246.15478515625, 273.040771484375, 249.08447265625, 248.931884765625, 300.537109375, 249.053955078125, 251.983642578125, 242.645263671875, 249.359130859375, 245.758056640625, 271.392822265625, 251.922607421875, 241.912841796875, 244.415283203125, 251.52587890625, 244.68994140625, 245.66650390625, 239.715576171875, 272.491455078125, 249.664306640625, 269.500732421875, 243.896484375, 272.0947265625, 236.6943359375, 251.708984375, 243.804931640625, 244.903564453125, 239.105224609375, 249.51171875, 247.894287109375, 247.833251953125, 247.6806640625, 249.359130859375, 251.434326171875, 246.368408203125, 236.297607421875, 251.678466796875, 243.621826171875, 244.9951171875, 299.8046875, 241.0888671875, 247.49755859375, 243.865966796875, 246.03271484375, 239.74609375, 243.59130859375, 241.058349609375, 272.064208984375, 244.537353515625, 251.983642578125, 272.064208984375, 239.19677734375, 245.42236328125, 249.664306640625, 239.349365234375, 243.865966796875, 244.537353515625, 243.011474609375, 241.63818359375, 245.2392578125, 245.849609375, 249.81689453125, 241.851806640625, 247.86376953125, 273.040771484375, 243.0419921875, 300.628662109375, 244.964599609375, 242.095947265625, 244.964599609375, 270.416259765625, 273.040771484375, 299.102783203125, 244.384765625, 245.452880859375, 243.316650390625, 246.856689453125, 250.30517578125, 243.0419921875, 271.575927734375, 240.966796875, 243.59130859375, 244.62890625, 240.478515625, 248.47412109375, 246.246337890625, 247.589111328125, 250.244140625, 244.659423828125, 243.95751953125, 236.99951171875, 243.804931640625, 272.186279296875, 250.91552734375, 247.74169921875, 248.59619140625, 247.74169921875, 243.194580078125, 242.401123046875, 244.68994140625, 244.9951171875, 247.314453125, 249.298095703125, 244.56787109375, 274.169921875, 241.3330078125, 241.485595703125, 248.748779296875, 247.49755859375, 242.3095703125, 253.387451171875, 243.011474609375, 246.673583984375, 242.1875, 247.406005859375, 248.138427734375, 251.15966796875, 250.30517578125, 247.0703125, 246.15478515625, 235.260009765625, 248.046875, 274.169921875, 252.62451171875, 245.66650390625, 300.23193359375, 239.74609375, 248.199462890625, 242.584228515625, 272.94921875, 245.849609375, 251.89208984375, 242.98095703125, 242.3095703125, 234.86328125, 245.941162109375, 242.645263671875, 243.65234375, 236.541748046875, 242.156982421875, 247.222900390625, 235.504150390625, 243.34716796875, 272.216796875, 242.034912109375, 246.795654296875, 245.147705078125, 241.302490234375, 238.76953125, 242.889404296875, 244.415283203125, 245.635986328125, 244.62890625, 271.636962890625, 234.9853515625, 241.546630859375, 240.814208984375, 249.908447265625, 247.4365234375, 271.17919921875, 248.870849609375, 243.988037109375, 245.819091796875, 245.4833984375, 238.494873046875, 238.46435546875, 248.10791015625, 243.988037109375, 245.513916015625, 239.92919921875, 240.142822265625, 249.053955078125, 273.37646484375, 244.384765625, 247.86376953125, 241.8212890625, 243.438720703125, 248.291015625, 248.47412109375, 242.67578125, 249.57275390625, 240.478515625, 251.983642578125, 245.05615234375, 246.917724609375, 250.48828125, 249.847412109375, 245.05615234375, 236.907958984375, 271.05712890625, 245.880126953125, 236.724853515625, 245.30029296875, 245.758056640625, 274.4140625, 242.034912109375, 243.621826171875, 242.0654296875, 246.09375, 242.00439453125, 271.820068359375, 243.133544921875, 244.81201171875, 241.63818359375, 249.267578125, 238.1591796875, 239.74609375, 247.86376953125, 244.415283203125, 238.46435546875, 248.016357421875, 240.997314453125, 246.551513671875, 245.78857421875, 299.652099609375, 244.232177734375, 239.715576171875, 245.452880859375, 248.565673828125, 241.8212890625, 240.631103515625, 246.368408203125, 247.802734375, 250.274658203125, 246.368408203125, 249.57275390625, 245.9716796875, 237.3046875, 238.8916015625, 236.14501953125, 236.541748046875, 250.335693359375, 242.706298828125, 242.0654296875, 238.677978515625, 242.645263671875, 273.01025390625, 246.337890625, 242.1875, 241.76025390625, 247.100830078125, 250.946044921875, 249.93896484375, 272.94921875, 237.12158203125, 243.896484375, 273.895263671875, 247.161865234375, 239.8681640625, 269.866943359375, 246.826171875, 271.636962890625, 241.02783203125, 248.84033203125, 245.147705078125, 249.359130859375, 241.0888671875, 250.640869140625, 251.03759765625, 252.044677734375, 272.674560546875, 244.110107421875, 251.708984375, 248.809814453125, 242.0654296875, 245.7275390625, 272.247314453125, 245.7275390625, 251.434326171875, 249.1455078125, 272.52197265625, 237.335205078125, 240.966796875, 248.199462890625, 250.823974609375, 239.8681640625, 246.337890625, 240.05126953125, 245.30029296875, 273.223876953125, 247.528076171875, 245.849609375, 242.61474609375, 244.56787109375, 240.234375, 236.38916015625, 250.335693359375, 251.220703125, 241.69921875, 240.631103515625, 251.373291015625, 252.716064453125, 240.753173828125, 248.53515625, 271.05712890625, 245.17822265625, 246.429443359375, 242.401123046875, 234.31396484375, 271.514892578125, 247.650146484375, 243.59130859375, 236.87744140625, 247.13134765625, 237.36572265625, 240.66162109375, 242.67578125, 248.931884765625, 246.734619140625, 271.30126953125, 247.833251953125, 246.9482421875, 239.2578125, 272.979736328125, 244.20166015625, 240.72265625, 241.790771484375, 238.128662109375, 247.344970703125, 271.392822265625, 239.3798828125, 249.6337890625, 249.32861328125, 247.283935546875, 242.37060546875, 239.471435546875, 243.194580078125, 242.24853515625, 249.969482421875, 273.406982421875, 239.31884765625, 249.420166015625, 240.570068359375, 247.467041015625, 244.2626953125, 241.27197265625, 235.04638671875, 243.988037109375, 249.298095703125, 245.452880859375, 252.0751953125, 250.732421875, 240.966796875, 250.30517578125, 247.894287109375, 243.499755859375, 243.560791015625, 242.584228515625, 244.20166015625, 251.64794921875, 243.011474609375, 242.0654296875, 245.7275390625, 239.44091796875, 241.76025390625, 249.176025390625, 244.049072265625, 244.903564453125, 242.431640625, 247.528076171875, 237.79296875, 247.467041015625, 244.93408203125, 247.4365234375, 242.095947265625, 244.232177734375, 273.3154296875, 250.823974609375, 243.133544921875, 250.8544921875, 249.847412109375, 235.65673828125, 248.077392578125, 250.3662109375, 274.078369140625, 270.416259765625, 244.415283203125, 247.25341796875, 248.10791015625, 236.968994140625, 246.673583984375, 240.41748046875, 241.546630859375, 245.4833984375, 240.997314453125, 242.889404296875, 272.125244140625, 247.711181640625, 250.06103515625, 271.697998046875, 251.129150390625, 245.17822265625, 244.68994140625, 246.124267578125, 247.467041015625, 243.804931640625, 249.847412109375, 245.086669921875, 247.406005859375, 247.955322265625, 242.706298828125, 272.88818359375, 246.09375, 273.193359375, 250.91552734375, 273.98681640625, 242.340087890625, 241.63818359375, 242.950439453125, 244.68994140625, 246.124267578125, 244.964599609375, 244.873046875, 247.650146484375, 247.955322265625, 271.728515625, 247.55859375, 251.190185546875, 249.69482421875, 245.391845703125, 237.945556640625, 241.058349609375, 241.2109375, 244.62890625, 248.291015625, 250.030517578125, 243.988037109375, 243.988037109375, 241.02783203125, 272.15576171875, 243.10302734375, 241.14990234375, 272.64404296875, 250.030517578125, 245.819091796875, 250.48828125, 239.92919921875, 246.734619140625, 247.25341796875, 249.69482421875, 271.54541015625, 300.384521484375, 241.69921875, 246.64306640625, 245.086669921875, 273.86474609375, 250.946044921875, 248.626708984375, 247.0703125, 243.011474609375, 272.796630859375, 238.555908203125, 248.84033203125, 244.781494140625, 251.434326171875, 246.52099609375, 246.2158203125, 249.755859375, 273.0712890625, 246.673583984375, 243.804931640625, 246.551513671875, 246.429443359375, 271.636962890625, 245.2392578125, 235.382080078125, 269.927978515625, 236.02294921875, 271.636962890625, 237.79296875, 250.335693359375, 241.485595703125, 243.408203125, 246.88720703125, 243.682861328125, 251.312255859375, 246.063232421875, 238.555908203125, 247.711181640625, 251.373291015625, 249.93896484375, 242.462158203125, 248.96240234375, 249.69482421875, 247.955322265625, 246.64306640625, 242.49267578125, 245.758056640625, 236.907958984375, 247.894287109375, 251.617431640625, 240.509033203125, 245.941162109375, 246.52099609375, 246.307373046875, 246.368408203125, 242.98095703125, 299.8046875, 248.96240234375, 246.826171875, 274.23095703125, 246.88720703125, 248.53515625, 238.6474609375, 247.039794921875, 242.767333984375, 240.90576171875, 270.01953125, 243.2861328125, 245.849609375, 271.942138671875, 250.762939453125, 238.95263671875, 252.13623046875, 239.92919921875, 241.546630859375, 245.78857421875, 236.419677734375, 238.311767578125, 301.300048828125, 246.39892578125, 248.22998046875, 249.176025390625, 252.197265625, 250.30517578125, 272.735595703125, 240.41748046875, 237.79296875, 235.809326171875, 242.950439453125, 248.53515625, 246.7041015625, 242.767333984375, 248.687744140625, 244.842529296875, 244.5068359375, 240.66162109375, 248.1689453125, 247.528076171875, 270.904541015625, 241.241455078125, 245.4833984375, 239.105224609375, 242.279052734375, 235.015869140625, 246.4599609375, 245.30029296875, 248.626708984375, 240.72265625, 237.884521484375, 241.14990234375, 237.73193359375, 249.664306640625, 239.593505859375, 238.95263671875, 251.8310546875, 235.443115234375, 244.232177734375, 251.007080078125, 246.429443359375, 246.39892578125, 272.064208984375, 248.10791015625, 247.25341796875, 272.186279296875, 273.25439453125, 248.53515625, 243.896484375, 245.330810546875, 248.59619140625, 244.7509765625, 242.1875, 251.007080078125, 247.98583984375, 249.176025390625, 241.729736328125, 246.7041015625, 244.903564453125, 272.064208984375, 244.598388671875, 251.007080078125, 236.236572265625, 247.61962890625, 251.129150390625, 243.377685546875, 234.80224609375, 246.490478515625, 248.565673828125, 244.140625, 242.61474609375, 238.4033203125]
                         
                        e = 1.60217663e-19
                        B = 0.6966
                        c = 299792458
                        me = 9.1093837e-31
                        L = 0.2178

                        print (startF-startFdown)
                        energy = (e*B*c**2)/(2*np.pi*(np.array(houghfreq)*1e6+(startF-startFdown))) - me*c**2
                        energy_keV = energy/(1.60218e-19) * 1e-3
                        Energies.append(energy_keV)
                        startFs.append(startF)
                        pitches.append(pitchAngle)
                        axFreqs.append(axFreq)                    
                      
                
        else:
            print("File cannot be opened")

        # Close the file
        f.close()

        plt.show(block=False)
        
num_bins = 100

fig, ax = plt.subplots()

# best fit of data
#(mu, sigma) = norm.fit(Energies)

# the histogram of the data
n, bins, patches = ax.hist(Energies, num_bins, density=True)

# add a 'best fit' line
#y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
   #np.exp(-0.5 * (1 / sigma * (bins - mu))**2))

#y = norm.pdf( bins, mu, sigma)
#ax.plot(bins, y, '--')
ax.set_xlabel('Energy (keV)')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of normal distribution sample')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show(block=False)

plt.figure()
plt.plot(axFreqs, pitches, 'o')
plt.xlabel('Axial Frequencies')
plt.ylabel('Pitch Angles (degrees)')
plt.show()

# Find peaks in power spectra and fit function to them. Locate the lhs value of the peak to get the cyclotron frequency
# Locate the spectral lines in spectrogram and fit them to function, locate the main track and

# Wait main track is at carrier frequency or what?
# What is going on with my noise?
# Am i meant to clean the noise first before obtaining the values
# How to do for ones where the carrier frequency has 0 power
# Whaaaaaatttttt how do i get a spectrum



