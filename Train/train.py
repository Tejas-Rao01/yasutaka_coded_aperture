from chainer import cuda, optimizers, Variable, serializers
import numpy as np
from PIL import Image
import os

from model import*


epoch_num = 20
batch_size = 15
chanel_num = 3
image_file_num = 6
block_num = 10
block_set_num = image_file_num*block_num
block_size = 64
light_kind_num = 6
train_data_num = chanel_num*block_set_num*light_kind_num
check_num = train_data_num
check_num = check_num - check_num%15

shots_num = 2
view_point_row_num = 5
view_point_num = view_point_row_num*view_point_row_num
dev = 0.005

aug_data = np.zeros((batch_size, view_point_num, block_size, block_size), np.float32)
cnt = 0
loss_sum = 0.0

xp = cuda.cupy
cuda.get_device(0).use()


if os.path.isdir("./Dst") == False:
	os.mkdir("./Dst")

if os.path.isdir("./Dst/Mask") == False:
	os.mkdir("./Dst/Mask")

if os.path.isdir("./Dst/Model") == False:
	os.mkdir("./Dst/Model")




#Load train datas
load_light_field = np.zeros((block_set_num*chanel_num, view_point_num, block_size, block_size), np.uint8)
start_index_R = 0;
start_index_G = image_file_num*block_num
start_index_B = image_file_num*block_num*2

print("train with %f devietion noise" %dev)
for n in range(image_file_num):
	if(n % 10 == 0):
		print("load %04d" % (n))

	fname = "Src/%03d.png" % (n)
	img_load = Image.open(fname)
	img_load = np.asarray(img_load)

	for b in range(block_num):
		for t in range(view_point_num):
			load_light_field[n*block_num+b +start_index_R,t,:,:] = img_load[block_size*b:block_size*b+block_size,block_size*t:block_size*t+block_size,0]
			load_light_field[n*block_num+b +start_index_G,t,:,:] = img_load[block_size*b:block_size*b+block_size,block_size*t:block_size*t+block_size,1]
			load_light_field[n*block_num+b +start_index_B,t,:,:] = img_load[block_size*b:block_size*b+block_size,block_size*t:block_size*t+block_size,2]





#Model initaition
shot = Shot()
shot.to_gpu()
shot_optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08)
shot_optimizer.setup(shot)

reconstruct = Reconstruct()
reconstruct.to_gpu()
reconstruct_optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08)
reconstruct_optimizer.setup(reconstruct)

vdsr = VeryDeepSuperResolution()
vdsr.to_gpu()
vdsr_optimizer = optimizers.Adam(alpha=0.0001, beta1=0.9, beta2=0.999, eps=1e-08)
vdsr_optimizer.setup(vdsr)


#Physical limitations of masks
shot.cn01.W.data[shot.cn01.W.data < 0.0] = 0.0
shot.cn01.W.data[shot.cn01.W.data > 1.0] = 1.0
shot.cn01.b.data[:] = 0.0





#Train loop
for i in range(1, epoch_num+1):

	sff = np.random.permutation(train_data_num)

	for n in range(0, train_data_num, batch_size):
		#Data augmentation by intensity levels
		for j in range(batch_size):
			aug_data[j,:,:,:] = ((1-(sff[n+j]%light_kind_num)/10)*load_light_field[int(sff[n+j]/light_kind_num)]/255).astype(np.float32)
		
		gray_light_field = cuda.to_gpu(aug_data)
		gray_light_field = Variable(gray_light_field)


		#Train models		
		shot.cleargrads()
		reconstruct.cleargrads()
		vdsr.cleargrads()


		gray_acquired_image = shot(gray_light_field)
		gray_noise_acquired_image = gray_acquired_image + view_point_num*dev*xp.random.randn(batch_size,shots_num,block_size,block_size)
		gray_makeshift_light_field = reconstruct(gray_noise_acquired_image)
		gray_reconstruct_light_field = vdsr(gray_makeshift_light_field)

		loss = F.mean_squared_error(gray_reconstruct_light_field, gray_light_field)

		loss.backward()
		shot_optimizer.update()
		reconstruct_optimizer.update()
		vdsr_optimizer.update()


		shot.cn01.W.data[shot.cn01.W.data < 0.0] = 0.0
		shot.cn01.W.data[shot.cn01.W.data > 1.0] = 1.0
		shot.cn01.b.data[:] = 0.0

		print("epoch:%04d/%04d, data:%06d/%06d, psnr:%.6f" % (i, epoch_num, n, train_data_num, 10*xp.log10(1.0/loss.data)))

		loss_sum += loss.data	
		cnt += 1



		#Save masks and calculate average psnr
		if(n%check_num == 0):

			mask_txt_name = "./Dst/Mask/mask_txt_%03depoch-%06ddata.txt" % (i, n)
			f_mask = open(mask_txt_name, "w")

			for s in range(shots_num):
				img_save = cuda.to_cpu(shot.cn01.W.data[s,:,0,0])
				img_save = img_save.reshape(view_point_row_num,view_point_row_num)
				f_mask.write("%s\n" % img_save)	
				img_save = Image.fromarray(np.uint8(img_save * 255.0))
				img_save = img_save.resize((256,256))
				img_save.save("./Dst/Mask/epoch" + str('{0:04d}'.format(i))+"-batch"+str('{0:06d}'.format(n)) + "done_" + str('{0:02d}'.format(s+1)) + ".png")
	
			f_mask.close()


			print("@@@    average psnr:%.6f    @@@" % (10*xp.log10(1.0 / (loss_sum / cnt))))
			f_psnr = open("./Dst/average_psnr.txt", "a")
			f_psnr.write("epoch:%04d/%04d, data:%06d/%06d, psnr:%.6f\n" % (i, epoch_num, n, train_data_num, 10*xp.log10(1.0 / (loss_sum / cnt))))
			f_psnr.close()

			cnt = 0
			loss_sum = 0.0
	


	#Save models every epoch
	shot.to_cpu()
	serializers.save_npz("./Dst/Model/shot%04d.npz" %i, shot)
	shot.to_gpu()

	reconstruct.to_cpu()
	serializers.save_npz("./Dst/Model/reconstruct%04d.npz" %i, reconstruct)
	reconstruct.to_gpu()

	vdsr.to_cpu()
	serializers.save_npz("./Dst/Model/vdsr%04d.npz" %i, vdsr)
	vdsr.to_gpu()














