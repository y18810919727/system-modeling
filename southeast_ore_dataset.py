import datetime
import pandas as pd
import numpy as np
# import hydra
import os
from torch.utils.data import Dataset
import copy
from common import cal_time, SimpleLogger
from matplotlib import pyplot as plt
from model.informer.utils.tools import StandardScaler


def mongodb_connect():
    import mongoengine
    mongoengine.connect('nfca_db', host='dgx.server.ustb-ai3d.cn', port=27018, username='nfca', password='nfca')


def queryset2df(query_data):
    """
    queryset转 dataframe
    :return:
    """
    dic = {"time": [], "value": []}
    for q in query_data:
        dic["time"].append(q['time'])
        dic["value"].append(q['Monitoring_value'])
    #  存在数据库多条目同一时间同一值、多条目同一时间不同值的情况，仅保留首条数据
    df = pd.DataFrame(dic).drop_duplicates(subset=['time'])
    return df


# TODO: 切换成默认是未归一化的模式
class SoutheastOreDataset(Dataset):
    """
    东南矿体数据集
    """

    def __init__(self, data_dir, step_time, in_name, out_name, logging, time_range=None,
                 data_from_csv=True, ctrl_solution=0):
        """

        Args:
            data_dir:
            step_time:
            in_name:
            out_name:
            logging:
            time_range:
            data_from_csv:
            ctrl_solution: 0：什么也不做
                           1：泥层压强项增加out_length时延
                           2：泥层压强从模型输入更为模型输出
        """
        # 进料浓度、出料浓度、进料流量、底流流量、泥层压力
        self.point = {
            1: [5, 7, 11, 17, 67],
            2: [6, 8, 12, 18, 68],
            "name": ["feed_c", "out_c", "feed_f", "out_f", "pressure"]
        }
        self.ctrl_solution = ctrl_solution
        self.unfilter_data = {}
        self.raw_data = {1: pd.DataFrame(), 2: pd.DataFrame()}
        self.data_dir = os.path.join(data_dir, 'data/south_con/')
        self.logging = logging

        # interpolation config
        self.inter_sep = 10

        # 更新二号浓密机的fill_round
        self.already_filled_round = 0
        # Network in/out config
        # 因为使用omegaconf传入的list config类型是ListConfig, 他不是List的子类(isinstance==False)
        in_name = in_name[:] if len(list(in_name)[0]) > 1 else [in_name]
        out_name = out_name[:] if len(list(out_name)[0]) > 1 else [out_name]
        self.in_columns = in_name
        self.out_columns = out_name
        self.in_length = int(60 / self.inter_sep) * step_time[0]  # 30min
        self.out_length = int(60 / self.inter_sep) * step_time[1]  # 10min
        self.window_step = int(60 / self.inter_sep) * step_time[2]  # 5min

        if self.ctrl_solution == 1:
            self.in_columns.remove('pressure')
        elif self.ctrl_solution == 2:
            self.in_columns.remove('pressure')
            self.out_columns.append('pressure')

        if not data_from_csv:
            # if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            mongodb_connect()
            self.gene_data(1, time_range)
            self.gene_data(2, time_range)

        else:
            # TODO 长期计划：add config文件，注明Dataset时间段、是否被插值
            # from common import detect_download
            # access_key = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), 'data', 'AccessKey.csv'))
            # _ = detect_download(
            #     pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), self.data_dir, 'export_urls.csv')),
            #     self.data_dir,
            #     'http://oss-cn-beijing.aliyuncs.com',
            #     'southeast-thickener',
            #     access_key['AccessKey ID'][0],
            #     access_key['AccessKey Secret'][0]
            # )
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
            shifted_in_length = self.in_length + (self.out_length if self.ctrl_solution == 1 else 0)
            for j in range(shifted_in_length, this_fill_length - self.out_length, self.window_step):
                self.split_pos.append(round_split[i - 1] + j)

        if data_from_csv:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data.values,
                            inpt=[self.point['name'].index(i) for i in self.in_columns],
                            outpt=[self.point['name'].index(i) for i in self.out_columns])
            self.data = pd.DataFrame(self.scaler.transform(self.data.values),
                                     columns=['feed_c', 'out_c', 'feed_f', 'out_f', 'pressure', 'fill_round'])

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
        while data[data_len - 1]['time'] >= current_time and i < data_len - 1:
            if data[i]['time'] <= current_time:
                if current_time < data[i + 1]['time']:
                    inter_value = data[i]['value'] + (
                            (current_time - data[i]['time']) / (data[i + 1]['time'] - data[i]['time'])) * (
                                          data[i + 1]['value'] - data[i]['value'])
                    return_data['time'].append(current_time)
                    return_data['value'].append(inter_value)
                    current_time = current_time + datetime.timedelta(seconds=self.inter_sep)
                    if current_time > end_time:
                        return pd.DataFrame(return_data)
                else:
                    i = i + 1
            else:
                self.logging('error')
        return pd.DataFrame(return_data)

    @cal_time
    def get_filling_range(self, key):
        """
        筛选满足规则的数据段
        :return: [(start_time1, end_time1),(start_time2, end_time2)]
        """
        # 以底流浓度为标准：分割「相邻点间隔时间」或「浓度<40持续时间」大于180s
        data_c = self.unfilter_data[self.point[key][1]].loc[lambda x: x['value'] > 40]['time']
        c_count = len(data_c)
        start_time = None
        c_range = []
        delta_time = data_c.diff()
        for i in range(c_count - 1):
            if start_time is None and delta_time[i + 1].seconds < 180:
                start_time = data_c[i]
            elif start_time is not None and delta_time[i + 1].seconds > 180:
                c_range.append((start_time, data_c[i]))
                start_time = None
        c_range.append((start_time, data_c[-1]))

        # 以底流流量为标准：
        #       1. 序列长度大于1hours；
        #       2. 若「相邻点间隔时间」或「流量<5持续时间」大于180s，将其切成两段序列；
        #       3. 切割后的序列中，「流量>5」的个数大于180个（标准采样频率为5s，即至少有1/4的有效值)
        c_f_range = []
        for t in c_range:
            if (t[1] - t[0]).seconds > 3600:
                data_f = self.unfilter_data[self.point[key][3]][t[0]:t[1]].loc[lambda x: x['value'] > 5]['time']
                f_count = len(data_f)
                if f_count < 180:
                    continue
                start_time = data_f[0]
                start_inx = 0
                delta_time = data_f.diff()
                for i in range(f_count - 1):
                    if delta_time[i + 1].seconds > 180:
                        if (i - start_inx) > 180:
                            c_f_range.append((start_time, data_f[i]))
                        start_time = data_f[i + 1]
                        start_inx = i + 1
                c_f_range.append((start_time, t[1]))

        # 以底流浓度导数为标准，删除开头和结尾，win_size尺寸的滑动窗口内变化大于max_general_dt的区间
        c_f_c_range = []
        for t in c_f_range:
            data_c_3 = self.unfilter_data[self.point[key][1]][t[0]:t[1]]
            c_delta = data_c_3['value'].diff()
            WIN_SIZE = 10
            MAX_GENERAL_DT = 5
            WATCHING_WINS = len(c_delta) / 2
            aggr_delta = [c_delta[i:i + WIN_SIZE].sum() for i in range(1, len(c_delta) - WIN_SIZE)]
            for i in range(len(aggr_delta)):
                if max(aggr_delta[i:i + WATCHING_WINS]) < MAX_GENERAL_DT:
                    start_time = data_c_3['time'][i]
                    break
            else:
                continue
            for i in range(len(aggr_delta) - 1, 0, -1):
                if max(aggr_delta[i - WATCHING_WINS:i]) < MAX_GENERAL_DT:
                    end_time = data_c_3['time'][i + WIN_SIZE]
                break
            else:
                continue
            c_f_c_range.append((start_time, end_time))

        c_f_c_range = list(filter(lambda t: (t[1] - t[0]).seconds > 3600, c_f_c_range))
        return c_f_c_range

    def zScoreNormalization(self, th_id, df):

        for inx, point_id in enumerate(self.point[th_id]):
            mean = df[point_id].mean()
            std = df[point_id].std()
            df[point_id] = df[point_id].apply(lambda x: (x - mean) / std)
            self.logging(
                f'#{th_id} thickener {self.point["name"][inx]} std is {round(std, 4)}, mean is {round(mean, 4)}')
        return df

    def get_time_end(self, th_id, time_range):
        time_list = []
        for point_id in self.point[th_id]:
            time_list.append(
                self.unfilter_data[point_id].loc[time_range[0]:time_range[1]]['time'][-1])
        return min(time_list)

    def get_time_start(self, th_id, time_range):
        time_list = []
        for point_id in self.point[th_id]:
            time_list.append(
                self.unfilter_data[point_id].loc[time_range[0]:time_range[1]]['time'][0])
        return max(time_list)

    @cal_time
    def save_csv(self, th_id, point_df, unnormalized=False):
        round_count = int(max(point_df['fill_round'])) + 1
        path = self.data_dir
        file_name = "{key}-{round_count}-{data_count}.csv". \
            format(key=th_id,
                   round_count=round_count,
                   data_count=point_df.shape[0])
        if unnormalized:
            file_name = file_name[:-4] + "-unnormalized.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        point_df.to_csv(path + file_name)

    def read_csv(self):
        for filename in os.listdir(self.data_dir):
            if filename.count('-') != 3:
                continue
            th_id, round_count, data_count, _ = filename[:-4].split('-')
            self.raw_data[int(th_id)] = pd.read_csv(os.path.join(self.data_dir, filename), usecols=range(2, 8))
            self.logging(f"get thickener#{th_id} {round_count} round filling, a total of {data_count} records")

    @staticmethod
    def see_duplicate_item(df):
        """ 输入Dataframe（含time、value列），返回重复的条目"""
        tar_df = df.reset_index(drop=True)
        time_repeat_mask = tar_df.groupby('time').count() > 1
        inx = time_repeat_mask[time_repeat_mask['value'] == True].index
        repeat_df = tar_df[tar_df['time'].isin(inx)]
        return repeat_df

    @cal_time
    def gene_data(self, th_id, time_range=None, unnormalized_save=True):
        from data.db_models import GmsMonitor
        if time_range is not None:
            for i in self.point[th_id]:
                self.unfilter_data[i] = (
                    queryset2df(GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1], point_id=i)
                                .only('time', 'Monitoring_value').as_pymongo())
                        .set_index(['time'], drop=False)
                        .iloc[::-1].sort_index(ascending=True))
        else:
            for i in self.point[th_id]:
                self.unfilter_data[i] = (
                    queryset2df(GmsMonitor.objects(point_id=i).only('time', 'Monitoring_value').as_pymongo())
                        .set_index(['time'], drop=False)
                        .iloc[::-1].sort_index(ascending=True))

        time_list = self.get_filling_range(th_id)
        self.logging("thickener#{th_id} total {count} time frame".format(th_id=th_id, count=len(time_list)))
        all_df = pd.DataFrame()
        for inx, t in enumerate(time_list):
            df_list = []
            for point_id in self.point[th_id]:
                start_time = self.get_time_start(th_id, t)
                end_time = self.get_time_end(th_id, t)
                df_data = self.unfilter_data[point_id].loc[start_time:end_time]
                df_data = self.linear_interpolation(df_data, start_time=start_time, end_time=end_time)

                df_data = df_data.rename(columns={'value': point_id})
                df_list.append(df_data)

            df_merge = df_list[0]
            for i in range(len(df_list) - 1):
                df_merge = df_merge.merge(df_list[i + 1], on='time')
            df_merge['fill_round'] = inx + self.already_filled_round
            all_df = all_df.append(df_merge)
        #     TODO: 未归一化数据标签未rename，无法合并
        if unnormalized_save:
            self.save_csv(th_id, all_df, unnormalized=True)
        all_df = self.zScoreNormalization(th_id, all_df)
        all_df.rename(columns={self.point[th_id][i]: self.point['name'][i] for i in range(len(self.point['name']))},
                      inplace=True)
        self.raw_data[th_id] = all_df
        self.save_csv(th_id, all_df)
        self.already_filled_round = len(time_list)

    def __getitem__(self, item):
        if self.ctrl_solution == 1:
            item_in_1 = np.array(
                self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][
                    self.in_columns], dtype=np.float32)
            item_in_2 = np.array(
                self.data[self.split_pos[item] - self.in_length - self.out_length:self.split_pos[item]][[
                    'pressure']], dtype=np.float32)
            item_in = np.concatenate((item_in_1, item_in_2), axis=1)

            item_out = np.array(
                self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][
                    self.out_columns], dtype=np.float32)
        else:
            item_in = np.array(
                self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][
                    self.in_columns], dtype=np.float32)
            item_out = np.array(
                self.data[self.split_pos[item] - self.in_length:self.split_pos[item] + self.out_length][
                    self.out_columns], dtype=np.float32)
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
                self.get_part_dataset(dataset_split[1], dataset_split[2]),
                self.scaler
                )


if __name__ == '__main__':
    test_range = (datetime.datetime(2021, 4, 1, 0, 0, 0), datetime.datetime(2021, 4, 10, 0, 0, 0))
    dataset0 = SoutheastOreDataset(data_dir=os.getcwd(), step_time=[30, 10, 5], time_range=test_range,
                                   in_name=["out_f", "pressure"], out_name="out_c",
                                   logging=SimpleLogger(os.path.join('tmp', 'test.out')),
                                   data_from_csv=True)
