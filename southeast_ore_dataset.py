import mongoengine
import datetime
from data.db_models import GmsMonitor
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
import copy
from common import cal_time


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

    def __init__(self, data_dir, step_time, time_range=None):
        # 进料浓度、出料浓度、进料流量、底流流量、泥层压力
        self.point = {
            1: [5, 7, 11, 17, 67],
            2: [6, 8, 12, 18, 68],
            "name": ["feed_c", "out_c", "feed_f", "out_f", "pressure"]
        }

        self.raw_data = {1: [], 2: []}
        self.data_dir = os.path.join(data_dir, 'data/south_con/')

        # interpolation config
        self.inter_sep = 10
        self.inter_method = 'quadratic'

        # 更新二号浓密机的fill_round
        self.already_filled_round = 0
        # Network in/out config
        self.in_columns = ["out_f", "pressure"]
        self.out_column = "out_c"
        self.in_length = int(60 / self.inter_sep) * step_time[0]  # 30min
        self.out_length = int(60 / self.inter_sep) * step_time[1]  # 10min
        self.window_step = int(60 / self.inter_sep) * step_time[2]  # 5min

        mongodb_connect()
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            self.gene_data(1, time_range)
            self.gene_data(2, time_range)
        else:
            # TODO 长期计划：add config文件，注明Dataset时间段、是否被插值
            self.read_csv()

        # merge both thickeners data
        self.data = pd.concat(self.raw_data, ignore_index=True)

        round_split = [0]
        fill_round_list = list(self.data['fill_round'])
        for i in range(1, len(fill_round_list)):
            if fill_round_list[i - 1] != fill_round_list[i]:
                round_split.append(i)

        self.split_pos = []
        for i in range(1, len(round_split)):
            this_fill_length = round_split[i] - round_split[i - 1]
            for j in range(self.in_length, this_fill_length - self.out_length, self.window_step):
                self.split_pos.append(j)

    @cal_time
    def linear_interpolation(self, data, start_time, end_time=None):
        """
        对数据进行线性插值
        """
        current_time = start_time
        # datafream转list
        data = data.to_dict('records')
        i = 0
        data_len = len(data)
        return_data = {'time': [], 'value': []}
        if current_time < data[0]['time']:
            current_time = data[0]['time']
        while (data[data_len - 1]['time'] >= current_time and i < data_len - 1):
            if data[i]['time'] <= current_time:
                if current_time < data[i + 1]['time']:
                    inter_value = data[i]['value'] + (
                            (current_time - data[i]['time']) / (data[i + 1]['time'] - data[i]['time'])) * (
                                          data[i + 1]['value'] - data[i]['value'])
                    return_data['time'].append(current_time)
                    return_data['value'].append(inter_value)
                    current_time = current_time + datetime.timedelta(seconds=self.inter_sep)
                    if end_time is not None and current_time > end_time:
                        return pd.DataFrame(return_data)
                else:
                    i = i + 1
            else:
                print('error')

    @cal_time
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
                    GmsMonitor.objects(time__gte=t[0], time__lt=t[1]).only(
                        'point_id', 'time', 'Monitoring_value').filter(point_id=self.point[key][3]).filter(
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

    @cal_time
    def zScoreNormalization(self, th_id, df):
        for point_id in self.point[th_id]:
            mean = df[point_id].mean()
            std = df[point_id].std()
            df[point_id] = df[point_id].apply(lambda x: (x - mean) / std)
            if point_id == 7 or point_id == 8:
                print(f'浓密机{th_id}出料浓度的标准差为{std},均值为{mean}')
        return df

    def get_time_end(self, th_id, time_range):
        time_list = []
        for point_id in self.point[th_id]:
            time_list.append(
                GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1]).only(
                    'point_id', 'time', 'Monitoring_value').filter(point_id=point_id).first().time)
        return min(time_list)

    def get_time_start(self, th_id, time_range):
        time_list = []
        for point_id in self.point[th_id]:
            time_list.append(
                GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1]).only(
                    'point_id', 'time', 'Monitoring_value').filter(point_id=point_id).order_by(
                    "time").first().time)
        return max(time_list)

    @cal_time
    def save_csv(self, th_id, point_df, time):
        round_count = int(max(point_df['fill_round'])) + 1
        path = self.data_dir
        file_name = "{key}-{round_count}-{data_count}.csv". \
            format(key=th_id,
                   round_count=round_count,
                   data_count=point_df.shape[0])
        if not os.path.exists(path):
            os.makedirs(path)
        point_df.to_csv(path + file_name)

    def read_csv(self):
        for filename in os.listdir(self.data_dir):
            th_id, round_count, data_count = filename[:-4].split('-')
            self.raw_data[int(th_id)] = pd.read_csv(os.path.join(self.data_dir, filename))
            print(f"已读取浓密机{th_id}的{round_count}段数据，共计{data_count}条")

    def gene_data(self, th_id, time_range=None):
        time_list = self.get_filling_range(th_id, time_range)
        print("{count}个时间段".format(count=len(time_list)))
        all_df = pd.DataFrame()
        for inx, t in enumerate(time_list):
            df_list = []
            for point_id in self.point[th_id]:
                start_time = self.get_time_start(th_id, t)
                end_time = self.get_time_end(th_id, t)
                df_data = queryset2df(
                    GmsMonitor.objects(
                        time__gte=time_range[0], time__lt=time_range[1]).only(
                        'point_id', 'time', 'Monitoring_value').filter(
                        point_id=point_id).order_by("time"))
                # interpolate
                # helper = pd.DataFrame({'time': pd.date_range(start_time, end_time,
                #                                              freq=str(self.inter_sep)+"S")})
                # df_data = pd.merge(df_data, helper, on='time', how='outer')
                # df_data = df_data.set_index("time")
                # df_data.sort_index(ascending=True)
                # df_data = df_data[~df_data.index.duplicated()]
                # df_data = df_data.interpolate(self.inter_method)
                df_data = self.linear_interpolation(df_data, start_time=start_time, end_time=end_time)

                df_data = df_data.rename(columns={'value': point_id})
                df_list.append(df_data)

            df_merge = df_list[0]
            for i in range(len(df_list) - 1):
                df_merge = df_merge.merge(df_list[i + 1], on='time')
            df_merge['fill_round'] = inx + self.already_filled_round
            all_df = all_df.append(df_merge)

        all_df = self.zScoreNormalization(th_id, all_df)
        all_df.rename(columns={self.point[th_id][i]: self.point['name'][i] for i in range(len(self.point['name']))},
                      inplace=True)
        self.raw_data[th_id] = all_df
        self.save_csv(th_id, all_df, time_range)
        self.already_filled_round = len(time_list)

    def __getitem__(self, item):
        item_in = np.array(
            self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][self.in_columns],
            dtype=np.float32)
        item_out = np.array(
            [self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][self.out_column]],
            dtype=np.float32).T
        return item_in, item_out

    def __len__(self):
        return len(self.split_pos)

    @cal_time
    def get_part_dataset(self, start, end):
        transcript = copy.deepcopy(self)
        transcript.split_pos = transcript.split_pos[int(len(self.split_pos) * start):int(len(self.split_pos) * end)]
        return transcript

    @cal_time
    def get_split_dataset(self, dataset_split: list):
        """

        Args:
            dataset_split: train:test:valid eg. [0.6,0.2,0.2]

        Returns:
            Tuple(train_set,test_set,valid_set)
        """
        dataset_split = [sum(dataset_split[0:i]) for i in range(1, len(dataset_split) + 1)]
        # eg.[0.6,0.8,1]
        assert dataset_split[2] == 1
        return (self.get_part_dataset(0, dataset_split[0]),
                self.get_part_dataset(dataset_split[0], dataset_split[1]),
                self.get_part_dataset(dataset_split[1], dataset_split[2])
                )


if __name__ == '__main__':
    test_range = (datetime.datetime(2021, 4, 1, 0, 0, 0), datetime.datetime(2021, 4, 2, 0, 0, 0))
    dataset = SoutheastOreDataset(data_dir=os.getcwd(), step_time=[30, 10, 5], time_range=test_range)
