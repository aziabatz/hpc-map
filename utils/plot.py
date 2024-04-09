import matplotlib.pyplot as plt
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def plot(td, env, trained, rew_trained, untrained = None, rew_untrained = None):

    # Plotting
    for i, td in enumerate(td):
        fig, axs = plt.subplots(1,2, figsize=(20,10))
        if untrained is not None and rew_untrained is not None:
            env.render(td, untrained[i], ax=axs[0])
            axs[0].set_title(f"Untrained | Cost (Less is better) = {-rew_untrained[i].item():.3f}")
            log.info(f"Actions for untrained model-{i}: {untrained[i]}")
            log.info(f"Reward for untrained model-{i}: {-rew_untrained[i].item():.3f}")
        env.render(td, trained[i], ax=axs[1])
        axs[1].set_title(r"Trained $\pi_\theta$" + f"| Cost (Less is better) = {-rew_trained[i].item():.3f}")
        log.info(f"Actions for trained model-{i}: {trained[i]}")
        log.info(f"Reward for trained model-{i}: {-rew_trained[i].item():.3f}")
    plt.show()