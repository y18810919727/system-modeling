import mongoengine
import datetime
from data.db_models import GmsMonitor
import pandas as pd
import os
from torch.utils.data import Dataset


def mongodb_connect():
    mongoengine.connect('nfca_db', host='192.168.0.37', port=27017, username='nfca', password='nfca')


def queryset2df(query_data):
    """
    queryset转 dataframe
    :return:
    """
    dic = {"time": [], "value": []}
    for q in query_data:
        dic["time"].append(q.time)
        dic["value"].append(q.Monitoring_value)
    df = pd.DataFrame(dic)
    return df


class SoutheastOreDataset(Dataset):
    """
    东南矿体数据集
    """

    def __init__(self, time_range=None):
        # 进料浓度、出料浓度、进料流量、底流流量、泥层压力
        self.point = {
            1: [5, 7, 11, 17, 67],
            2: [6, 8, 12, 18, 68],
            "name": ["feed_c", "out_c", "feed_f", "out_f", "pressure"]
        }

        self.data = {1: [], 2: []}

        # interpolation config
        self.inter_sep = 10
        self.inter_method = 'linear'

        mongodb_connect()
        if not os.path.exists('data/south_con') or not os.listdir('data/south_con'):
            self.gene_data(1, time_range)
            self.gene_data(2, time_range)
        else:
            # TODO: 长期计划：add config文件，注明Dataset时间段、是否被插值
            self.read_csv()

        # merge both thickener data
        self.merge_data = {name: self.data[1][inx].append(self.data[2][inx])
                           for inx, name in enumerate(self.point['name'])}
    #   TODO: 更新二号浓密机的fill_round

    def get_filling_range(self, key, time_range=None):
        """
        筛选满足规则的数据段
        :return: [(start_time1, end_time1),(start_time2, end_time2)]
        """
        # 以底流浓度为标准：分割相邻点间隔时间或浓度<40持续时间大于180s
        if time_range is not None:
            data_c = list(GmsMonitor.objects(
                time__gte=time_range[0], time__lt=time_range[1]).only(
                'point_id', 'time', 'Monitoring_value').filter(
                point_id=self.point[key][1]).filter(Monitoring_value__gte=40).order_by("time"))
        else:
            data_c = list(GmsMonitor.objects().only(
                'point_id', 'time', 'Monitoring_value').filter(
                point_id=self.point[key][1]).filter(Monitoring_value__gte=40).order_by("time"))
        c_count = len(data_c)
        start_time = None
        c_range = []
        for i in range(c_count - 1):
            delta_time = data_c[i + 1].time - data_c[i].time
            if start_time is None and delta_time.seconds < 180:
                start_time = data_c[i].time
            elif start_time is not None and delta_time.seconds > 180:
                c_range.append((start_time, data_c[i].time))
                start_time = None
        c_range.append((start_time, data_c[-1].time))

        # 以底流流量为标准：舍弃长度小于1hours的序列；若相邻点间隔时间或流量<5持续时间大于180s，将其切成两段序列；
        c_f_range = []
        for t in c_range:
            if (t[1] - t[0]).seconds > 3600:
                data_f = list(
                    GmsMonitor.objects(time__gte=t[0], time__lt=t[1]).filter(point_id=self.point[key][3]).filter(
                        Monitoring_value__gte=5).order_by("time"))
                f_count = len(data_f)
                start_time = t[0]
                for i in range(f_count - 1):
                    delta_time = data_f[i + 1].time - data_f[i].time
                    if delta_time.seconds > 180:
                        c_f_range.append((start_time, data_f[i].time))
                        start_time = data_f[i + 1].time
                c_f_range.append((start_time, t[1]))

            c_f_range = list(filter(lambda t: (t[1] - t[0]).seconds > 3600, c_f_range))

        return c_f_range

    def zScoreNormalization(self, df):
        mean = df['value'].mean()
        std = df['value'].std()
        df['value'] = df['value'].apply(lambda x: (x - mean) / std)
        return df

    def get_time_end(self, th_id, time_range):
        time_list = []
        for point_id in self.point[th_id]:
            time_list.append(
                GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1]).filter(point_id=point_id)[0].time)
        return min(time_list)

    def save_csv(self, th_id, point_id, point_df):
        round_count = int(max(point_df['fill_round'])) + 1
        path = './data/south_con/'
        file_name = "{key}-{name}-{round_count}-{data_count}.csv". \
            format(key=th_id,
                   name=self.point["name"][
                       self.point[th_id].index(point_id)],
                   round_count=round_count,
                   data_count=point_df.shape[0])
        if not os.path.exists(path):
            os.makedirs(path)
        point_df.to_csv(path + file_name)

    def read_csv(self):
        # TODO:更新csv格式
        for filename in os.listdir('data/south_con'):
            th_id, name, round_count, data_count = filename[:-4].split('-')
            self.data[th_id] = [0] * len(self.point['name'])
            self.data[th_id][self.point[th_id][self.point['name'].index(name)]] = pd.read_csv(filename)

    def gene_data(self, th_id, time_range=None):
        time_list = self.get_filling_range(th_id, time_range)
        print("{count}个时间段".format(count=len(time_list)))
        for point_id in self.point[th_id]:
            point_df = pd.DataFrame()
            for inx, t in enumerate(time_list):
                end_time = self.get_time_end(th_id, t)
                df_data = queryset2df(
                    GmsMonitor.objects(
                        time__gte=t[0], time__lt=end_time).filter(point_id=point_id).order_by("time"))
                df_data['fill_round'] = inx

                # interpolate
                # TODO: reshample是Series类内的成员函数，需要以时间戳为索引，dataframe会报错
                df_data['value'] = df_data['value'].resample(
                    str(self.inter_sep) + 'S').interpolate(self.inter_method)

                point_df = point_df.append(df_data, ignore_index=True)

            self.zScoreNormalization(point_df)
            self.data[th_id].append(point_df)
            self.save_csv(th_id, point_id, point_df)

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    test_range = (datetime.datetime(2021, 4, 1, 0, 0, 0), datetime.datetime(2021, 6, 1, 0, 0, 0))
    dataset = SoutheastOreDataset(test_range)
