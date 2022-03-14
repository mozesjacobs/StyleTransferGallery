import argparse

args = None

def parse_args():
    
    parser = argparse.ArgumentParser(description="Template")

    # anything that affects the name of the saved folders (for checkpoints, experiments, tensorboard)
    parser.add_argument('-sess', '--session_name', default="march_9_mlp4", type=str, help="session name (KG, HyperKG, HyperKG_RNN)")
    parser.add_argument('-EX', '--experiments', default='./experiments/', type=str, help="Output folder for experiments")
    parser.add_argument('-MF', '--model_folder', default='./trained_models/', type=str, help="Output folder for experiments")
    parser.add_argument('-TB', '--tensorboard_folder', default='./tb_runs/', type=str, help="Output folder for tensorboard")
    
    # model parameters
    parser.add_argument('-NG', '--decoder_channels', default=10, type=int, help="Factor for dim of channels inside decoder")
    parser.add_argument('-R', '--z_channels', default=100, type=int, help="Number of channels in r latent")
    parser.add_argument('-RW', '--z_w', default=8, type=int, help="Width of r latent")
    parser.add_argument('-RH', '--z_h', default=8, type=int, help="Height of r latent")
    parser.add_argument('-RL', '--recon_loss', default="MSE", type=str, help="Which reconstruction loss function to use (BCE, MSE)")
    parser.add_argument('-H', '--hidden_dim', default=10, type=int, help="Dimension of LSTM output")

    # inference parameters
    parser.add_argument('-lmda_r2', '--regularization_r2', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-lmda_r', '--regularization_r', default=5e-4, type=float, help="Learning rate")
    parser.add_argument('-lr_r', '--lr_r', default=1e-3, type=float, help="Learning rate r during inference")
    parser.add_argument('-lr_r2', '--lr_r2', default=5e-4, type=float, help="Learning rate r2 during inference")
    parser.add_argument('-ctr', '--converge_threshold_r', default=1e-2, type=float, help="Convergence threshold for r during inference")
    parser.add_argument('-ctr2', '--converge_threshold_r2', default=1e-2, type=float, help="Covergence threshold for r2 during inference")
    parser.add_argument('-IMT', '--inf_max_t', default=350, type=int, help="Max number of timesteps to infer")

    # training / testing
    parser.add_argument('-D', '--device', default=0, type=int, help="Which device to use")
    parser.add_argument('-E', '--epochs', default=50, type=int, help="Number of Training Epochs")
    parser.add_argument('-B', '--batch_size', default=50, type=int, help="Batch size")    
    parser.add_argument('-T', '--timesteps', default=10, type=int, help="How many timesteps to run on")
    parser.add_argument('-CM', '--c_max', default=0.0, type=float, help="Max beta value for KL")
    parser.add_argument('-CW', '--c_weight_update', default=1.0, type=float, help="KLD epoch weight update")
    parser.add_argument('-CSW', '--c_specific_weight', default=None, type=float, help="KLD epoch weight (if None, uses defaults with weight_update")    
    parser.add_argument('-I', '--checkpoint_interval', default=1, type=int, help="Saves the model every checkpoint_interval intervals")
    parser.add_argument('-CL', '--clip', default=1.0, type=float, help="Gradient clip value")
    parser.add_argument('-GF', '--gamma_factor', default=0.9995, type=float, help="Learning rate decay factor")
    parser.add_argument('-C', '--load_cp', default=0, type=int, help="If 1, loads previous checkpoint")
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('-lmda', '--regularization', default=1e-4, type=float, help="L2 penalty")
    parser.add_argument('-IE', '--initial_e', default=0, type=int, help="Initial Number of Epochs")
    parser.add_argument('-BM', '--beta_max', default=1.0, type=float, help="Max beta value for KL")
    parser.add_argument('-W', '--weight_update', default=0.01, type=float, help="KLD epoch weight update")
    parser.add_argument('-SW', '--specific_weight', default=None, type=float, help="KLD epoch weight (if None, uses defaults with weight_update")

    # data
    parser.add_argument('-F', '--frame_size', default=64, type=int, help="Dimensions of the frames")
    parser.add_argument('-CH', '--channels', default=1, type=int, help="Number of channels in image frame")
    parser.add_argument('-SEQ', '--seq_len', default=20, type=int, help="Number of frames in a sequence")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # experiment arguments  
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    # mp4 animation tests
    parser.add_argument("--do_all", default=0, type=int, help="Run all tests")
    parser.add_argument("--ani", default=0, type=int, help="Animation tests")
    parser.add_argument("--ani_a", default=0, type=int, help="Animation tests (pred, corr, original)")
    parser.add_argument("--ani_b", default=0, type=int, help="Animation tests (sampling)")
    
    # misc
    parser.add_argument("--gif_frame_duration", default=0.25, type=float, help="Duration of each frame in the gif")
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help="If 1, prints the name of the experiments output folder")
    
    # grids / graphs
    parser.add_argument("--noise", default=0, type=int, help="Noise test")
    parser.add_argument("--num_plot_noise", default=3, type=int, help="Number to plot for test")
    parser.add_argument("--bounce", default=0, type=int, help="Bounce test")
    parser.add_argument("--missing", default=0, type=int, help="Missing test")
    parser.add_argument("--ood", default=0, type=int, help="OOD test")
    parser.add_argument('--ood_digits', default=[0], type=list, help="OOD Digits")
    parser.add_argument("--num_plot_ood", default=4, type=int, help="Number to plot for test")
    parser.add_argument("--sample_grid", default=0, type=int, help="Sample grid test")
    parser.add_argument("--sample_context_grid", default=0, type=int, help="Sample with context frames grid test")
    parser.add_argument('--num_context', default=5, type=int, help="Number of context frames to use")
    parser.add_argument("--num_sample_grid", default=3, type=int, help="Number of samples to plot in sample grid test")
    parser.add_argument("--inference_grid", default=0, type=int, help="Inference grid test")
    parser.add_argument("--num_inference_grid", default=5, type=int, help="Number of samples to plot in inference grid test")
    parser.add_argument("--disentangle_grid", default=0, type=int, help="Disentangle grid test")
    parser.add_argument("--num_disentangle_grid", default=4, type=int, help="Number of samples to plot in disentangle grid test")
    
    # tests specifically for the f identity vectr
    parser.add_argument("--f_over_seq", default=0, type=int, help="f over sequence test")
    parser.add_argument("--f_clustering", default=0, type=int, help="f clustering test")
    parser.add_argument("--f_clustering_efficiency", default=0, type=int, help="f clustering efficiency test")
    parser.add_argument("--mess_up_i", default=0, type=int, help="mess up f test")
    
    # tests specifically for r2
    parser.add_argument("--r2_over_seq", default=0, type=int, help="r2 over sequence test")
    parser.add_argument("--mess_up_r2", default=0, type=int, help="mess up r2 test")
    parser.add_argument("--r2_clustering", default=0, type=int, help="r2 clustering test")
    parser.add_argument("--r2_novel_action", default=0, type=int, help="r2 novel action test")
    
    # other
    parser.add_argument("--multiple_perturbation", default=0, type=int, help="multiple perturbation test")
    parser.add_argument("--sprites_change_action", default=0, type=int, help="sprites change action test")
    parser.add_argument("--sprites_change_identity", default=0, type=int, help="sprites change identity test")
    parser.add_argument("--transfer_learning", default=0, type=int, help="transfer learning test")
    parser.add_argument("--mess_up_r2_i", default=0, type=int, help="mess up r2 and i test")
    return parser.parse_args() 

def run_args():
    global args
    if args is None:
        args = parse_args()


run_args()


# 
