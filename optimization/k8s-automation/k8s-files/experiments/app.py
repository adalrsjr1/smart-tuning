import matplotlib.pyplot as plt
import jmeter
import jmeter_summary
import mongo_metrics
import mongo_configs


def plot(name, _jmeter_folder_, _mongo_folder_, _jmeter_id_, _timestep_, _interval_, _expected_avg_):
    _jmeter_ = 'volume/jmeter/'
    _mongo_ = 'volume/mongo/'

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(top=0.6)

    # ax0 = jmeter.plot(ax=axs[0, 0], title=f'throughput measured at client',
    #                   filepath=_jmeter_ + _jmeter_folder_ + '/raw_data_' + _jmeter_id_ + '.jtl',
    #                   timestep=_timestep_,
    #                   interval=_interval_,
    #                   expected_avg=_expected_avg_ * 2)

    # ax1 = jmeter_summary.plot(axs[0, 1],
    #                           filename=_jmeter_ + _jmeter_folder_ + 'acmeair.stats.' + _jmeter_id_,
    #                           title='errors overtime', expected_avg=_expected_avg_)

    ax2 = mongo_metrics.plot(axs[1, 0],
                             filepath=_mongo_ + _mongo_folder_ + 'mongo_metrics.json',
                             title='throughput per pod', expected_avg=_expected_avg_)

    ax3 = mongo_configs.plot(axs[1, 1],
                             filepath=_mongo_ + _mongo_folder_ + 'mongo_workloads.json',
                             title='configurations over time',
                             expected_avg=_expected_avg_)

    fig.suptitle('tuning interval: {}min, experiment interval: {}h, req/s expectation: {}, sampling: {}min'.format(
        _timestep_ // 60, _interval_ // 3600, _expected_avg_, _timestep_ // 60), y=1)
    fig.tight_layout()
    plt.show()
    # plt.savefig(str(name))

if __name__ == '__main__':
    # jmeter_folder =    ['20200515-043107/', '20200515-150051/', '20200515-231509/', '20200515-190929/', '20200516-013602/', '20200518-022204/', '20200518-165703/', '20200519-044613/', '20200519-195932/', '20200519-225006/']
    # mongo_folder =     ['20200515-063218/', '20200515-190326/', '20200516-011810/', '20200515-231120/', '20200518-014210/', '20200518-162720/', '20200519-043903/', '20200519-174229/', '20200519-223838/', '20200520-011626/']
    # jmeter_id =        ['2020051543107', '20200515150051', '20200515231509', '20200515190929', '2020051613602', '2020051822204', '20200518165703', '2020051944613', '20200519195932', '20200519225006']
    # timestep =         [300, 600, 300, 600, 3600, 3600, 3600, 3600, 900, 900]
    # interval =         [2 * 3600, 4 * 3600, 2 * 3600, 4 * 3600, 48 * 3600, 14 * 3600, 12 * 3600, 12*3600, 2*3600, 2 * 3600]
    # expected_avg =     [1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 1000, 1000]
    #
    # name = 0
    # for _jmeter_folder_, _mongo_folder_, _jmeter_id_, _timestep_, _interval_, _expected_avg_ in zip(
    #     jmeter_folder, mongo_folder, jmeter_id, timestep, interval, expected_avg
    # ):
    #     if name > 8:
    #         plot(name, _jmeter_folder_, _mongo_folder_, _jmeter_id_, _timestep_, _interval_, _expected_avg_)
    #     name += 1

    # plot('', '', '20200520-164010/', '', 900, 135 * 60, 2000)
    plot('', '', '20200520-204704/', '', 900, 120 * 60, 2000)
    plot('', '', '20200520-230338/', '', 900, 120 * 60, 2000)

