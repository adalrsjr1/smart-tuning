import matplotlib.pyplot as plt
import jmeter
import jmeter_summary
import mongo_metrics
import mongo_configs


def plot(n_clients, name, _jmeter_folder_, _mongo_folder_, _jmeter_id_, _timestep_, _interval_, _expected_avg_):
    _jmeter_ = 'volume/jmeter/'
    _mongo_ = 'volume/mongo/'

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(top=0.6)

    ax0 = jmeter.plot(ax=axs[0, 0], title=f'throughput measured at client',
                      filepath=_jmeter_ + 'prod/' + _jmeter_folder_ + '/raw_data_' + _jmeter_id_ + '.jtl',
                      timestep=_timestep_,
                      interval=_interval_,
                      expected_avg=_expected_avg_ * 2, label='prod')

    if n_clients == 2:
        ax0 = jmeter.plot(ax=ax0, title=f'throughput measured at client',
                          filepath=_jmeter_ + 'train/' + _jmeter_folder_ + '/raw_data_' + _jmeter_id_ + '.jtl',
                          timestep=_timestep_,
                          interval=_interval_,
                          expected_avg=_expected_avg_ * 2, label='train')

    ax1 = jmeter_summary.plot(axs[0, 1],
                              filename=_jmeter_ + 'prod/' + _jmeter_folder_ + 'acmeair.stats.' + _jmeter_id_,
                              title='errors overtime', expected_avg=_expected_avg_, label='prod')

    if n_clients == 2:
        ax1 = jmeter_summary.plot(ax1,
                                  filename=_jmeter_ + 'train/' + _jmeter_folder_ + 'acmeair.stats.' + _jmeter_id_,
                                  title='errors overtime', expected_avg=_expected_avg_, label='train')

    ax2 = mongo_metrics.plot(axs[1, 0],
                             filepath=_mongo_ + _mongo_folder_ + 'mongo_metrics.json',
                             title='throughput per pod', expected_avg=_expected_avg_)

    ax3 = mongo_configs.plot(axs[1, 1],
                             filepath=_mongo_ + _mongo_folder_ + 'mongo_workloads.json',
                             title='configurations over time',
                             expected_avg=_expected_avg_)

    fig.suptitle('tuning interval: {}min, experiment interval: {}h'.format(
        _timestep_ // 60, _interval_ // 3600), y=1)
    fig.tight_layout()
    if name:
        plt.savefig(str(name))
    else:
        plt.show()
    #

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

    # # sampling .6667 / min conn 4 / 3 it
    # plot('', '', '20200520-164010/', '', 900, 135 * 60, 1000)
    # # sampling 1.0 / min conn 4 / 0 it
    # plot('', '', '20200520-204704/', '', 900, 135 * 60, 1000)
    # # sampling 1.0 / min conn 1 / 0 it
    # plot('', '', '20200520-230338/', '', 900, 135 * 60, 1000)
    # # sampling 1.0 / min conn 1 / 0 it
    # plot('', '20200520-205006/', '20200521-011404/', '20200520205006', 900, 120 * 60, 1000)
    # # only http parameters
    # plot('', '20200521-012617/', '20200521-033713/', '2020052112617', 900, 120 * 60, 1000)
    # max threads
    # plot('', '20200522-014432/', '20200522-035512/', '2020052214432', 900, 130 * 60, 1000)
    # xmx / mongo conn prod4 train100
    # plot('', '20200523-022434/', '20200523-044328/', '2020052322434', 900, 130 * 60, 2000)
    # -xmx 2h 10% to update - sampling 66% [ok]
    # plot('', '20200523-161514/', '20200523-181854/', '20200523161514', 900, 130 * 60, 2000)

    # #-xmx 4h 0% to update - sampling 66% [ok]
    # plot(2, '4h-s066-2c', '20200523-182342/', '20200523-223255/', '20200523182342', 900, 240 * 60, 2000)
    # # -xmx 4h 0% to update - sampling 100% - two services
    # plot(2, '4h-s100-2c', '20200524-194620/', '20200524-235332/', '20200524194620', 900, 240 * 60, 2000)
    #
    # # -xmx 4h 0% to update - sampling 66% - single service
    # plot(1, '4h-s066-1c', '20200523-223727/', '20200524-025037/', '20200523223727', 900, 240 * 60, 2000)
    # # -xmx 4h 0% to update - sampling 100% - single service
    # plot(1, '4h-s100-1c', '20200524-152813/', '20200524-193347/', '20200524152813', 900, 240 * 60, 2000)

    # plot(2, '', '20200525-174655/', '20200525-191638/', '20200525174655', 900, 90, 2000)
    # threads (4,100), http (4, 100), mongo(1, 30)
    # plot(2, '4h-s033-2c-multi', '20200531-183030/', '20200531-225554/', '20200531183030', 900, 3600*4, 8000)
    # mongo(1, 30)
    # plot(2, '4h-s033-2c-mongo', '20200531-230123/', '20200601-010756/', '20200531230123', 900, 3600*2, 8000)
    # threads (4,100), http (4, 100), mongo(1, 30)
    # plot(2, '4h-s033-2c-multi2', '20200602-160050/', '20200602-201422/', '20200602160050', 900, 3600*4, 8000)
    # threads (4,100), http (4, 100), mongo(1, 30) no load balancer
    # plot(1, '2h-s033-2c-multi-noLB', '20200602-202617/', '20200602-223113/', '20200602202617', 900, 3600*2, 8000)
    plot(2, '4h-s033-2c-multi3', '20200602-223445/', '20200603-024429/', '20200602223445', 900, 3600*4, 8000)

