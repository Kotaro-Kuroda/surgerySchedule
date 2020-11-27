import distribution_csv
import os
import matplotlib.pyplot as plt
import numpy as np
home = os.environ['HOME']
path = home + '/Documents/data-july/operations_with_jisseki_remake.csv'
file_path = home + '/Documents/surgerySchedule/distribution.csv'


def log_normal(x, mu, sigma):
    return 1 / (np.sqrt(2 * np.pi * sigma) * x) * np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma))


def draw_figure(g, time_dict, distribution_dict, kind):
    mu = distribution_dict[g, 'mu']
    sigma = distribution_dict[g, 'sigma']
    time_list = time_dict[g]
    plt.hist(time_list, density=True)
    x = np.arange(0.01, max(time_list) + 20.01, 0.01)
    y = log_normal(x, mu, sigma)
    plt.plot(x, y)
    plt.title(str(g) + str(kind))
    plt.savefig('/Users/kurodakotaro/Documents/image/distribution/' + str(g) + '_' + kind + '.png')
    plt.show()


def main():
    csv_distribution = distribution_csv.CsvDistribution(path, file_path)
    list_group = csv_distribution.get_group_list()
    preparation_distribution = csv_distribution.get_group_distribution('preparation')
    surgery_distribution = csv_distribution.get_group_distribution('surgery')
    cleaning_distribution = csv_distribution.get_group_distribution('cleaning')
    preparation_dict = csv_distribution.get_time_list('preparation')
    surgery_dict = csv_distribution.get_time_list('surgery')
    cleaning_dict = csv_distribution.get_time_list('cleaning')
    for g in list_group:
        if g != '関節外科':
            draw_figure(g, preparation_dict, preparation_distribution, 'preparation')
            draw_figure(g, surgery_dict, surgery_distribution, 'surgery')
            draw_figure(g, cleaning_dict, cleaning_distribution, 'cleaning')


if __name__ == '__main__':
    main()
