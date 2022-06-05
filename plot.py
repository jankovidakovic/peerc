import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math

from matplotlib.ticker import FormatStrFormatter

if __name__ == '__main__':
    # adapter_bottleneck

    adapter_x = math.log(1488196, 10)
    f1s_adapter = np.array([0.7238, 0.7389, 0.7283, 0.7238, 0.7367, 0.7315, 0.7072, 0.7154, 0.726, 0.74])

    baseline_x = math.log(125239300, 10)
    f1s_baseline = np.array(
        [0.7189473684210526, 0.7007007007007007, 0.7348008385744234, 0.7320598864223026, 0.7172909184197024,
         0.719712525667351, 0.7344992050874405, 0.7176591375770021, 0.7266223811957077, 0.7206931702344547])

    bitfit_x = math.log(696580, 10)
    f1s_bitfit = np.array(
        [0.6896896896896898, 0.6756238003838771, 0.6858513189448441, 0.6755980861244019, 0.6695652173913045,
         0.6907317073170731, 0.6817042606516291, 0.6980470706059088, 0.7012345679012344, 0.709873417721519])

    fig = plt.figure()
    ax = fig.add_subplot()
    boxplot = ax.boxplot([f1s_bitfit, f1s_baseline, f1s_adapter], positions=[bitfit_x, baseline_x, adapter_x])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'$10^{%d}$'))
    # ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(r'$%.2f_{\mu}$'))

    # plt.boxplot([f1s_bitfit, f1s_baseline, f1s_adapter], positions=[bitfit_x, baseline_x, adapter_x])
    # plt.boxplot([f1s_bitfit, f1s_baseline, f1s_adapter], positions=[0.2, 0.1, 0.3])

    adapter_market = plt.scatter(np.repeat(adapter_x, len(f1s_adapter)), f1s_adapter, marker='^', color='black', label='Adapter', alpha=0.3, facecolors='none')
    baseline_marker = plt.scatter(np.repeat(baseline_x, len(f1s_baseline)), f1s_baseline, marker='x', color='black', label='Full Fine-tuning', alpha=0.3)
    bitfit_marker = plt.scatter(np.repeat(bitfit_x, len(f1s_bitfit)), f1s_bitfit, marker='s', color='black', label='BitFit',alpha=0.3, facecolors='none')

    ax.set_ylabel(r'$F1_{\mu}$')
    ax.set_xlabel('trainable parameters')

    ax.get_xaxis().set_ticks(range(5, 10))
    ax.get_xaxis().grid(True, which='major')
    ax.get_yaxis().grid(True, which='major')
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))
    # ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.xaxis.get_major_formatter().set_scientific(True)


    plt.legend(handles=[adapter_market, baseline_marker, bitfit_marker])
    plt.show()
    print("test")
