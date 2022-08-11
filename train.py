import torch
from torch.utils.data.dataloader import DataLoader
from Loader_batch import *
import cv2
from Environment import Cusenv
from PPO import ActorCritic, Buffer, train_epoch
from Data import *
from torch.distributions import Normal
from torch.distributions import Categorical

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    # torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

Train_Folder = '/home/rongkai/pixelRL/BSD68/gray/train/'
Test_Folder = '/home/rongkai/pixelRL/BSD68/gray/test/'
Crop_Size = 70

input_channel = 1
action_dim = 11
batch_size = 64
lr_actor = 0.0001  # learning rate for actor network
lr_critic = 0.0001  # learning rate for critic network
gamma = 0.95  # discount factor
K_epochs = 15  # update policy for K epochs in one PPO update
eps_clip = 0.2  # clip parameter for PPO
has_continuous_action_space = True  # False for discrete actions
action_var = 0.4  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.9  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.2  # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

print_interval = 1
test_interval = 100
save_interval = 100
decay_interval = 15

dataset = Dataset(Train_Folder, Crop_Size)
dataloader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=12,
                        pin_memory=True,
                        drop_last=True)

test_data = load_images_from_folder(Test_Folder)

env = Cusenv()

ppo_agent = ActorCritic(input_channel, action_dim).to(device)
print(torch.cuda.device_count())
if torch.cuda.device_count() > 1:
    print('using multiple GPUs')
    ppo_agent = torch.nn.DataParallel(ppo_agent)
else:
    pass

buffer = Buffer(batch_size)

optimizer = torch.optim.Adam([{'params': ppo_agent.parameters(), 'lr': lr_actor}])

i = 0

for n_epi in range(5001):  #
    score = 0.0
    # Data preprocessing (augmentationm, crop, resize to (B,C,H,W), normalize)
    #if n_epi % decay_interval == 0:
        #if action_var<min_action_std:
            #pass
        #else:
            #action_var = action_var*action_std_decay_rate
            #action_var = round(action_var,3)
            #print('narrow action_var to', action_var)
    for data in dataloader:
        data = data[:, np.newaxis, :, :]
        raw_x = np.array(data).astype(np.float32) / 255
        # Generate noise
        raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
        # Reset env (here is just add noise to image)
        s = env.reset(raw_x, raw_n)
        done = False
        t_info = 0
        while not done:
            state = torch.from_numpy(s).float().cuda()
            with torch.no_grad():
                prob, _ = ppo_agent(state)
                dist = Categorical(prob)
            a = dist.sample()
            prob_a = dist.log_prob(a)
            action = a.cpu().numpy()
            logprob_a = prob_a.cpu().numpy()
            s_prime, r, done = env.step(action, t_info)
            buffer.put_data((s, action, r, s_prime, logprob_a, done))
            t_info += 1
            s = s_prime
            score += r
        train_epoch(ppo_agent, buffer, optimizer, batch_size, gamma, K_epochs, eps_clip, action_var)
        print("# of episode :{}, avg score : {:.3f}".format(n_epi, np.mean(score) * 255))
        score = 0.0
        #if has_continuous_action_space:
            #trainer.decay_action_std(action_std_decay_rate, min_action_std)
    if n_epi % test_interval == 0 and n_epi != 0:
        test_result = 0
        test_id = 0
        img_id = 0
        input_psnr = 0
        for im in test_data:
            data = im[np.newaxis, np.newaxis, :, :]
            raw_x = data / 255
            raw_n = np.random.normal(0, 25, raw_x.shape).astype(raw_x.dtype) / 255
            s = env.reset(raw_x, raw_n)
            I = np.maximum(0, raw_x)
            I = np.minimum(1, I)
            N = np.maximum(0, raw_x + raw_n)
            N = np.minimum(1, N)
            I = (I[0] * 255 + 0.5).astype(np.uint8)
            N = (N[0] * 255 + 0.5).astype(np.uint8)
            I = np.transpose(I, (1, 2, 0))
            N = np.transpose(N, (1, 2, 0))
            cv2.imwrite('result_dis/' + str(test_id) + '_input.png', N)
            psnr1_cv = cv2.PSNR(N, I)
            done = False
            t = 0
            while not done:
                with torch.no_grad():
                    #a_test, _ = ppo_agent(torch.from_numpy(s).float().cuda())
                    prob, _ = ppo_agent(torch.from_numpy(s).float().cuda())
                # prob = prob.permute(0,2,3,1).detach().cpu()
                # m = Categorical(prob)
                _, a_test = torch.max(prob,-1)
                a_test = a_test.cpu().numpy()
                s_prime, r, done = env.step(a_test, t)
                if done:
                    with open('output.txt', 'a') as f:
                        print('test image', img_id, 'process', t, 'steps', file=f)
                s = s_prime
                t += 1
            img_id += 1
            p = np.maximum(0, s)
            p = np.minimum(1, p)
            p = (p[0] * 255 + 0.5).astype(np.uint8)
            p = np.transpose(p, (1, 2, 0))
            cv2.imwrite('result_dis/' + str(test_id) + '_output.png', p)
            psnr2_cv = cv2.PSNR(p, I)
            test_result += psnr2_cv
            with open("output.txt", "a") as f:
                print('test: PSNR_CV before:', psnr1_cv, 'PSNR_CV after:', psnr2_cv, file=f)
            test_id += 1
        with open("output_overall.txt", "a") as f:
            print('test', n_epi / test_interval, 'Overall performance:', test_result / len(test_data), file=f)

    if n_epi % save_interval == 0 and n_epi != 0:
        print("--------------------------------------------------------------------------------------------")
        print("saving model")
        torch.save(ppo_agent,'pretrained_dis/PPO_model_50patch_t=10_{}.pt'.format(n_epi))
        print("model saved")
        print("--------------------------------------------------------------------------------------------")
        # torch.save(ppo_agent, 'PPO_model_{}.pt'.format(n_epi))
        score = 0.0
