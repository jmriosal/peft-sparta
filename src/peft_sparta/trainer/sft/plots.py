import os
import matplotlib.pyplot as plt

def plot_stats(stats, config, loss_baseline, fname_prefix=''):

    fpath_prefix = os.path.join(config['output_dir'], fname_prefix)

    has_val = bool(stats['val_loss'])

    train_loss = stats['train_loss']
    if has_val:
        val_loss = {'x': stats['val_step'], 'y': stats['val_loss']}
        plot(train_loss, w = config['logging_freq'],
             baseline = loss_baseline,
             ts2 = val_loss, ts_name = 'train', ts2_name = 'val',
             anchor = val_loss['y'][0], # pre-trained performace at x=0
             title = 'Loss',
             save_file = fpath_prefix + 'loss.pdf')
        if stats.get('val_acc'):
            plot(stats['val_acc'], x_start = 0, # w = 1,
                 baseline = 1.0,
                 title = 'Validation Accuracy',
                 save_file = fpath_prefix + 'val_acc.pdf')
    else:
        plot(train_loss, w = config['logging_freq'],
             baseline = loss_baseline,
             title = 'Training Loss',
             save_file = fpath_prefix + 'train_loss.pdf')

    plot(stats['lr'],
         title = 'Learning Rate',
         save_file = fpath_prefix + 'learning_rate.pdf')

    if config['max_grad_norm']:
        max_grad_norm = config['max_grad_norm']
        plot(stats['grad_norm'], # w = 1, 
             baseline = max_grad_norm if max_grad_norm < float('inf') else 0, 
             title = 'Gradient Norm',  
             save_file = fpath_prefix + 'grad_norm.pdf')



def plot(ts, x_start=1, w=None, baseline=None, title=None, save_file=None,
         ts2=None, ts_name=None, ts2_name=None, anchor=None):
    fig, ax = plt.subplots(figsize=(8,5))
    if w: # rolling avg window
        y = [sum(ts[i:i+w])/w for i in range(len(ts) - w + 1)]
        x = list(range(x_start + w - 1, x_start + len(ts))) # last batch-step of each window
    else:
        y = ts
        x = list(range(x_start , x_start + len(ts)))
    if anchor:
        assert x_start == 1
        x, y = [0] + x, [anchor] + y # line segment from x=0 to x=w
    ax.plot(x, y, label=ts_name);
    if baseline:
        ax.hlines(baseline, xmin=0, xmax=x[-1], color='r')
    if ts2:
        ax.plot(ts2['x'], ts2['y'], label=ts2_name)
        ax.legend();
    ax.set_ylim(bottom=0);
    ax.set_title(title);
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
