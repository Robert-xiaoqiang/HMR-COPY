exec open('model_load.py').read()
import cv2
import matplotlib
import pylab as plt
import numpy as np
import util
import sys
import os
extensions_img = {".jpg", ".png", ".gif", ".bmp", ".jpeg"}

for filename in os.listdir(sys.argv[1]):
  for ext in extensions_img:
    if filename.endswith(ext):
      # test_image = 'sample_images/'+filename
      test_image = os.path.join(sys.argv[1], filename)
      file_noExt = os.path.splitext(filename)[0]
      print('Now proccessing:', filename)
    
      oriImg = cv2.imread(test_image) # B,G,R order
      param, model_params = config_reader()

      multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in param['scale_search']]

      heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
      paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))


      for m in range(len(multiplier)):
          scale = multiplier[m]
          imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
          imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_params['stride'], model_params['padValue'])        

          input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
          print("Input shape: " + str(input_img.shape))  

          output_blobs = model.predict(input_img)
          print("Output shape (heatmap): " + str(output_blobs[1].shape))

          # extract outputs, resize, and remove padding
          heatmap = np.squeeze(output_blobs[1]) # output 1 is heatmaps
          heatmap = cv2.resize(heatmap, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
          heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
          heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

          paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
          paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
          paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
          paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

          heatmap_avg = heatmap_avg + heatmap / len(multiplier)
          paf_avg = paf_avg + paf / len(multiplier)


      from numpy import ma
      U = paf_avg[:,:,16] * -1
      V = paf_avg[:,:,17]
      X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
      M = np.zeros(U.shape, dtype='bool')
      M[U**2 + V**2 < 0.5 * 0.5] = True
      U = ma.masked_array(U, mask=M)
      V = ma.masked_array(V, mask=M)

      from scipy.ndimage.filters import gaussian_filter
      all_peaks = []
      peak_counter = 0

      for part in range(19-1):
          map_ori = heatmap_avg[:,:,part]
          map = gaussian_filter(map_ori, sigma=3)

          map_left = np.zeros(map.shape)
          map_left[1:,:] = map[:-1,:]
          map_right = np.zeros(map.shape)
          map_right[:-1,:] = map[1:,:]
          map_up = np.zeros(map.shape)
          map_up[:,1:] = map[:,:-1]
          map_down = np.zeros(map.shape)
          map_down[:,:-1] = map[:,1:]

          peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
          peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
          peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
          id = range(peak_counter, peak_counter + len(peaks))
          peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

          all_peaks.append(peaks_with_score_and_id)
          peak_counter += len(peaks)

      json_template = '{"version":1.2,"people":[{"pose_keypoints":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}'

      all_peaks_list = []

      for element in all_peaks:
        while len(element) > 1:
          element.pop()

      for idx, val in enumerate(all_peaks):
        if len(val) < 1:
          all_peaks_list.append([0,0,0,0])
        elif len(val) >= 1:
          all_peaks_list.append(list(val[0]))
        if idx == 17:
          break

      for i in all_peaks_list:
        i.pop()

      all_peaks_list.insert(8, [(x+y)/2.0 for (x, y) in zip(all_peaks_list[8], all_peaks_list[11])])

      for i in range(6):
        all_peaks_list.append([0,0,0])

      all_peaks_flat = [item for sublist in all_peaks_list for item in sublist]

      all_peaks_flat = [float(i) for i in all_peaks_flat]

      o = json.loads(json_template)

      o['people'][0]['pose_keypoints'] = all_peaks_flat


      with open(sys.argv[1] + file_noExt+'.json', 'w') as outfile:
          json.dump(o, outfile)
