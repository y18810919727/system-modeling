import mongoengine
import datetime
from db_model.models import GmsMonitor
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset


class SoutheastOreDataset(Dataset):
    """
    东南矿体数据集
    """
    def __init__(self):
        # 进料浓度、出料浓度、进料流量、底流流量、泥层压力
        self.point = {
            1: [5, 7, 11, 17, 67],
            2: [6, 8, 12, 18, 68],
            "name": ["feed_c", "out_c", "feed_f", "out_f", "pressure"]
        }
        # self.time_range = time_range
        self.mongodb_connect()

    def mongodb_connect(self):
        mongoengine.connect('nfca_db', host='192.168.0.37', port=27017, username='nfca', password='nfca')

    def filter(self, key, time_range):
        """
        筛选满足规则的数据段
        :return: [](start_time, end_time)
        """
        data_c = list(GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1]).filter(point_id=self.point[key][1]).filter(Monitoring_value__gte=40).order_by("time"))
        c_count = len(data_c)
        start_time = None
        time_list = []
        for i in range(c_count-1):
            diff_time = data_c[i + 1].time - data_c[i].time
            if start_time is None and diff_time.seconds < 120:
                start_time = data_c[i].time
            elif start_time is not None and diff_time.seconds > 120:
                time_list.append((start_time, data_c[i].time))
                start_time = None
        time_list.append((start_time, data_c[i].time))

        time_list_new = []
        for t in time_list:
            if (t[1] - t[0]).seconds > 3600:
                data_f = list(GmsMonitor.objects(time__gte=t[0], time__lt=t[1]).filter(point_id=self.point[key][3]).filter(Monitoring_value__gte=5).order_by("time"))
                f_count = len(data_f)
                if f_count <= 24:
                    time_list_new.append(t)
                else:
                    start_time = t[0]
                    for i in range(f_count-1):
                        diff_time = data_f[i + 1].time - data_f[i].time
                        if diff_time.seconds > 180:
                            time_list_new.append((start_time, data_f[i].time))
                            start_time = data_f[i + 1].time
                    time_list_new.append((start_time, t[1]))

        return time_list_new


    def queryset2df(self, query_data):
        """
        queryset转 datafream
        :return:
        """
        dic = {"time": [], "value": []}
        for q in query_data:
            dic["time"].append(q.time)
            dic["value"].append(q.Monitoring_value)
        df = pd.DataFrame(dic)
        return df

    def get_time_end(self, key, time_range):
        time_list = []
        for id in self.point[key]:
            time_list.append(GmsMonitor.objects(time__gte=time_range[0], time__lt=time_range[1]).filter(point_id=id)[0].time)
        return min(time_list)


    def get_data(self, key, time_range):
        time_list = filter(key, time_range)
        print("{count}个时间段".format(count=len(time_list)))
        file_id = 1
        for t in time_list:
            if (t[1] - t[0]).seconds > 3600:
                end_time = self.get_time_end(key, t)
                for id in self.point[key]:
                    df_data = self.queryset2df(GmsMonitor.objects(time__gte=t[0], time__lt=end_time).filter(point_id=id).order_by("time"))
                    path = './csv_file/{key}-{name}/'.format(key=key, name=self.point["name"](self.point[key].index(id)))
                    file_name = "{file_id}-{count}.csv".format(file_id=file_id, count=df_data.shape[0])
                    if not os.path.exists(path):
                        os.makedirs(path)
                    df_data.to_csv(path+file_name)
                file_id = file_id + 1


if __name__ == '__main__':
    time_range = (datetime.datetime(2021, 4, 1, 0, 0, 0), datetime.datetime(2021, 6, 1, 0, 0, 0))
    dataset = SoutheastOreDataset()
    dataset.get_data(1, time_range)
    dataset.get_data(2, time_range)
