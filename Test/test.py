import chainer
from chainer import Variable, serializers
import numpy as np
from PIL import Image
import os
import pickle

from model import*



chanel_num = 3
size_w = 1024
size_h = 768


shot_num = 2
view_point_row_num = 5
view_point_num = view_point_row_num*view_point_row_num
dev_index = [0, 0.005];

is_gpu = True


if os.path.isdir("./Dst") == False:
	os.mkdir("./Dst")

if os.path.isdir("./Dst/ReconstructedImage") == False:
	os.mkdir("./Dst/ReconstructedImage")

if os.path.isdir("./Dst/AcquiredImage") == False:
	os.mkdir("./Dst/AcquiredImage")

if os.path.isdir("./Dst/Mask") == False:
	os.mkdir("./Dst/Mask")


#Load test datas
light_field = []
reconstruct_light_field = []
acquired_image = []

for t in range(view_point_num):
	fname = "Src/LightField/%d.png" %(t+1)
	img_load = Image.open(fname)
	img_load = np.asarray(img_load)
	img_load = img_load.astype(np.float32) / 255.0
	light_field.append(img_load.copy())

for t in range(view_point_num):
	reconstruct_light_field.append(np.zeros((size_h,size_w,3)))

for t in range(shot_num):
	acquired_image.append(np.zeros((size_h,size_w,3)))



#Model initaition
shot = Shot()
reconstruct = Reconstruct()
vdsr = VeryDeepSuperResolution()

serializers.load_npz("./Src/Model/shot.npz", shot)
serializers.load_npz("./Src/Model/reconstruct.npz", reconstruct)
serializers.load_npz("./Src/Model/vdsr.npz", vdsr)

if is_gpu == True:
	shot.to_gpu()
	reconstruct.to_gpu()
	vdsr.to_gpu()
	xp = chainer.cuda.cupy




#Test models
for dev in dev_index:

	if os.path.isdir("./Dst/ReconstructedImage/dev%f"%dev) == False:
		os.mkdir("./Dst/ReconstructedImage/dev%f"%dev)

	#Test models by each chanels
	print("start simulation with %f devietion noise" %dev)
	for ch in range(chanel_num):
		if ch == 0:
		    print('Reconstruct red chanel')
		elif ch == 1:
		    print('Reconstruct green chanel')
		elif ch == 2:
		    print('Reconstruct blue chanel')


		gray_light_field = np.zeros((1, view_point_num, size_h, size_w), np.float32)

		for t in range(view_point_num):
			gray_light_field[0,t,:,:] = light_field[t][:,:,ch]



		noise = view_point_num*dev*np.random.randn(1,shot_num,size_h,size_w)
		if is_gpu == True:
			gray_light_field = chainer.cuda.to_gpu(gray_light_field)
			noise = view_point_num*dev*xp.random.randn(1,shot_num,size_h,size_w)


		gray_light_field = Variable(gray_light_field)
		gray_acquired_image = shot(gray_light_field)
		gray_noise_acquired_image = gray_acquired_image + noise
		gray_reconstruct_light_field = vdsr(reconstruct(gray_noise_acquired_image))


		gray_acquired_image = gray_acquired_image.data
		gray_reconstruct_light_field = gray_reconstruct_light_field.data

		if is_gpu == True:
			gray_acquired_image = chainer.cuda.to_cpu(gray_acquired_image)
			gray_reconstruct_light_field = chainer.cuda.to_cpu(gray_reconstruct_light_field)

		for t in range(view_point_num):
			reconstruct_light_field[t][:,:,ch] = gray_reconstruct_light_field[0,t,:,:]

		for s in range(shot_num):
			acquired_image[s][:,:,ch] = gray_acquired_image[0,s,:,:]






	#Save reconstruct light field
	mse = 0.0
	for t in range(view_point_num):
		img_save = reconstruct_light_field[t]
		img_save[img_save < 0.0] = 0.0
		img_save[img_save > 1.0] = 1.0
		img_err = light_field[t][:,:,:] - img_save[:,:,:]
		img_err = img_err**2
		mse += np.sum(img_err)
		img_save = Image.fromarray(np.uint8(img_save * 255.0))

		img_save.save("Dst/ReconstructedImage/dev%f/%02d_%02d.png" % (dev,t/view_point_row_num, t%view_point_row_num))	

	mse /= size_h*size_w*chanel_num*view_point_num


	print("psnr:%f"%(10*np.log10(1.0 / (mse))))




#Save mask
for s in range(shot_num):
	img_save = chainer.cuda.to_cpu(shot.cn01.W.data[s,:,0,0])
	img_save = img_save.reshape(view_point_row_num,view_point_row_num)

	img_save = Image.fromarray(np.uint8(img_save * 255.0))
	img_save = img_save.resize((256,256))
	img_save.save("./Dst/Mask/" + str('{0:02d}'.format(s+1)) + ".png")


#Save acquired images
for s in range(shot_num):
	img_save = acquired_image[s]/view_point_num
	img_save[img_save < 0.0] = 0.0
	img_save[img_save > 1.0] = 1.0
	img_save = Image.fromarray(np.uint8(img_save * 255.0))
	img_save.save("Dst/AcquiredImage/%02d.png" % (s))







