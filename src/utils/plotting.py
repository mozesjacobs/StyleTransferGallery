import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import torch
import imageio
import math
import os

# deleted all plotting code since this is just a vector


def canvas2rgb_array(canvas):
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """Adapted from: https://stackoverflow.com/a/21940031/959926"""
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    ncols, nrows = canvas.get_width_height()
    scale = int(round(math.sqrt(buf.size / 3 / nrows / ncols)))
    return buf.reshape(scale * nrows, scale * ncols, 3)

def stacked_fig_full(truth, prior, error, post, num):
    fig, axes = plt.subplots(num, 4, figsize=(20, 20))
    for i in range(num):
        for j in range(4):
            # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)
    for i in range(num):
        axes[i][0].imshow(truth[i], cmap='gray')
        axes[i][1].imshow(prior[i], cmap='gray')
        axes[i][2].imshow(error[i], cmap='gray')
        axes[i][3].imshow(post[i], cmap='gray')
    axes[0][0].set_title("Ground Truth")
    axes[0][1].set_title("Prediction")
    axes[0][2].set_title("Error")
    axes[0][3].set_title("Correction")
    plt.subplots_adjust(wspace=0)
    plt.close()
    return fig

def make_gifs_together_full(truth, prior, post, T, num, fpath, duration=0.33):
    os.system("mkdir -p " + fpath)
    truth = truth.cpu().detach().numpy().squeeze()
    truth = np.swapaxes(truth, 0, 1)
    prior = np.array([curr.cpu().detach().numpy().squeeze() for curr in prior])
    post = np.array([curr.cpu().detach().numpy().squeeze() for curr in post])
    error = np.array([np.abs(truth[i] - prior[i]) for i in range(T)])
    images = []
    for i in range(T):
        the_fig = stacked_fig_full(truth[i], prior[i], error[i], post[i], num)
        data = canvas2rgb_array(the_fig.canvas)
        images.append(data)
    imageio.mimsave(fpath[0:-1] + ".gif", images, duration=duration)

def stacked_fig(truth, prior, error, post):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for i in range(4):
        # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    axes[0].imshow(truth, cmap='gray')
    axes[1].imshow(prior, cmap='gray')
    axes[2].imshow(error, cmap='gray')
    axes[3].imshow(post, cmap='gray')
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Prediction")
    axes[2].set_title("Error")
    axes[3].set_title("Correction")
    plt.subplots_adjust(wspace=0)
    plt.close()
    return fig

def make_gifs_together(truth, prior, post, T, num, fpath, duration=0.33):
    os.system("mkdir -p " + fpath)
    truth = truth.cpu().detach().numpy().squeeze()
    truth = np.swapaxes(truth, 0, 1)
    prior = np.array([curr.cpu().detach().numpy().squeeze() for curr in prior])
    post = np.array([curr.cpu().detach().numpy().squeeze() for curr in post])
    error = np.array([np.abs(truth[i] - prior[i]) for i in range(T)])
    for j in range(num):
        images = []
        for i in range(T):
            the_fig = stacked_fig(truth[i][j], prior[i][j], error[i][j], post[i][j])
            data = canvas2rgb_array(the_fig.canvas)
            images.append(data)
        imageio.mimsave(fpath + "stacked_" + str(j) + ".gif", images, duration=duration)

def plot_losses_marked(outer_recon, inner_recon, kld, marks,
                       file_name="Losses Marked",
                       titles=["Posterior Recon", "Prior Recon", "KLD"]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(outer_recon)
    axes[0].set_title(titles[0])
    axes[1].plot(inner_recon)
    axes[1].set_title(titles[1])
    axes[2].plot(kld)
    axes[2].set_title(titles[2])
    if marks is not None:
        for i in range(3):
            for t in marks:
                axes[i].axvline(x=t)
    plt.savefig(file_name)
    plt.show()
    plt.close()


def plot_losses(total_losses, img_losses, time_losses, file_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(total_losses)
    axes[0].set_title('Total Losses')
    axes[1].plot(img_losses)
    axes[1].set_title('BCE Loss')
    axes[2].plot(time_losses)
    axes[2].set_title('KLD')
    plt.savefig(file_name)
    plt.show()


def plot_imgs(x, predictions, corrections, img_size, chosen_imgs, timesteps=20,
              file_name=None, board=None, title=None):
    for i in chosen_imgs:
        fig, axes = plt.subplots(3, timesteps, figsize=(20, 6))
        for t in range(timesteps):
            img = x[i][t].cpu().detach().numpy().reshape(img_size, img_size)
            axes[0][t].imshow(img, cmap='gray')
        for t in range(timesteps):
            img = predictions[t][i].cpu().detach().numpy().reshape(img_size, img_size) 
            axes[1][t].imshow(img, cmap='gray')
        for t in range(timesteps):
            img = corrections[t][i].cpu().detach().numpy().reshape(img_size, img_size)
            axes[2][t].imshow(img, cmap='gray')
        if file_name is not None:
            plt.savefig(file_name + '_' + str(i))
        if title is not None:
            plt.title(title)
        if board is None:
            plt.show()
        else:
            board.add_figure(title, fig)
        plt.close()


def make_animations(seqs, batch_size, T, img_size, axes, inverted, cpu=False):
    assert batch_size >= 10
    all_ims = []
    for i in range(10):
        curr_seq = []
        for j in range(T):
            if inverted:
                frame = seqs[j][i].reshape(img_size, img_size)
                if cpu:
                    im = axes[i].imshow(frame, animated=True, cmap='gray')   
                else:
                    im = axes[i].imshow(frame.cpu().detach().numpy(), animated=True, cmap='gray')   
            else:
                frame = seqs[i][j].reshape(img_size, img_size)
                if cpu:
                    im = axes[i].imshow(frame, animated=True, cmap='gray')   
                else:
                    im = axes[i].imshow(frame.cpu(), animated=True, cmap='gray')   
            curr_seq.append([im])
        all_ims.append(curr_seq)
    return all_ims


def plot_gif(x, batch_size, T, img_size, file_name, inverted):
    temp = []
    for curr in x:
        temp.append(curr.cpu().detach().numpy())
    x = temp
    fig, axes = plt.subplots(1, 10)
    x_gifs = make_animations(x, batch_size, T, img_size, axes, inverted, True)
    animations = []
    for i in range(len(x_gifs)):
        animations.append(ArtistAnimation(fig, x_gifs[i],
                                          interval=50, blit=True, repeat_delay=1000))
    plt.show()
    return animations


def plot_imgs_grid(x, T, img_size, inverted, num=10, file_name=None, title=None):
    fig, axes = plt.subplots(num, T, figsize=(20, 20))
    for i in range(num):
        for j in range(T):
            if inverted:
                if x[0].size(-1) == 3:
                    frame = x[j][i].reshape(img_size, img_size, 3)
                else:
                    frame = x[j][i].reshape(img_size, img_size)
            else:
                if x[0].size(-1) == 3:
                    frame = x[i][j].reshape(img_size, img_size, 3)
                else:
                    frame = x[i][j].reshape(img_size, img_size)
            im = axes[i, j].imshow(frame.cpu().detach().numpy(), cmap='gray')
            
    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
    #plt.show()
    plt.close()
    
def plot_imgs_grid_inference(x_ground, x_prior, x_post, T, img_size,
                             num=10, file_name=None, title=None):
    fig, axes = plt.subplots(int(num * 3), T, figsize=(20, 20))
    # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    for i in range(num * 3):
        for j in range(T):
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)
    
    
    count = 0
    for i in range(num):
        for j in range(T):
            if x_ground[0].size(-1) == 3:
                f1 = x_ground[i][j].reshape(img_size, img_size, 3)
                f2 = x_prior[j][i].reshape(img_size, img_size, 3)
                f3 = x_post[j][i].reshape(img_size, img_size, 3)
            else:
                f1 = x_ground[i][j].reshape(img_size, img_size)
                f2 = x_prior[j][i].reshape(img_size, img_size)
                f3 = x_post[j][i].reshape(img_size, img_size)
            axes[int(i * 3), j].imshow(f1.cpu().detach().numpy(), cmap='gray')
            axes[int(i * 3 + 1), j].imshow(f2.cpu().detach().numpy(), cmap='gray')
            axes[int(i * 3 + 2), j].imshow(f3.cpu().detach().numpy(), cmap='gray')
            
    plt.subplots_adjust(wspace=0)
    
    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
    #plt.show()
    plt.close()

def plot_imgs_grid_sample_context(x_ground, x_prior, x_post, T, img_size,
                                  num=10, file_name=None, title=None):
    fig, axes = plt.subplots(int(num * 3), T, figsize=(20, 20))
    # https://stackoverflow.com/questions/20057260/how-to-remove-gaps-between-subplots-in-matplotlib
    for i in range(num * 3):
        for j in range(T):
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].get_yaxis().set_visible(False)
    
    
    count = 0
    for i in range(num):
        for j in range(T):
            if x_ground[0].size(-1) == 3:
                f1 = x_ground[i][j].reshape(img_size, img_size, 3)
                f2 = x_prior[j][i].reshape(img_size, img_size, 3)
                if len(x_post) > j:
                    f3 = x_post[j][i].reshape(img_size, img_size, 3)
            else:
                f1 = x_ground[i][j].reshape(img_size, img_size)
                f2 = x_prior[j][i].reshape(img_size, img_size)
                if len(x_post) > j:
                    f3 = x_post[j][i].reshape(img_size, img_size)
            axes[int(i * 3), j].imshow(f1.cpu().detach().numpy(), cmap='gray')
            axes[int(i * 3 + 1), j].imshow(f2.cpu().detach().numpy(), cmap='gray')
            if len(x_post) > j:
                axes[int(i * 3 + 2), j].imshow(f3.cpu().detach().numpy(), cmap='gray')
            
    plt.subplots_adjust(wspace=0)
    
    if title is not None:
        plt.title(title)
    if file_name is not None:
        plt.savefig(file_name)
    plt.close()