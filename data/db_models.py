import mongoengine

app_name = 'nfcadb'


class Point(mongoengine.Document):
    """点位管理表

    point_id: 点位编号
    opc_tag: 点位tag
    describe: 点位描述
    device_description_chinese：点位描述中文
    serial: 点位所属仪器
    instrument: 点位所属仪器名称
    device_name_chinese: 点位所属仪器名称中文
    belong_co: 点位所属公司
    status: 点位状态(normal/abnormal)
    value_min: 正常工作最小值
    value_max: 正常工作最大值
    unit: 监测值单位
    last_monitor_time: 最近数据采集时间

    ---
    以下为充填任务管理模块优化新加
    > https://gitee.com/USTB1001/nfcaindustrialintelligent/issues/I1AZIM?from=project-issue
    ---
    thickener_id: 浓密机选择(1/2)
    mixer_id: 搅拌机选择(1/2)
    """
    point_id = mongoengine.IntField(max_value=1024)
    opc_tag = mongoengine.StringField(max_length=150)
    describe = mongoengine.StringField(max_length=150)
    device_description_chinese = mongoengine.StringField(max_length=1000)
    serial = mongoengine.StringField(max_length=50)
    instrument = mongoengine.StringField(max_length=100)
    device_name_chinese = mongoengine.StringField(max_length=500)
    belong_co = mongoengine.StringField(max_length=30)
    status = mongoengine.StringField(max_length=10)
    value_min = mongoengine.FloatField(max_value=2048)
    value_max = mongoengine.FloatField(max_value=2048)
    unit = mongoengine.StringField(max_length=10)
    last_monitor_time = mongoengine.DateTimeField()
    thickener_id = mongoengine.IntField(max_value=1024)
    mixer_id = mongoengine.IntField(max_value=1024)


class GmsControlSettingNow(mongoengine.Document):
    """当前控制系统设置表

    point_id: 点位编号
    time：写入时间
    auto_setting： 当前目标值是否是智能推荐的，（否=手动设置
    control_mode： 0/1/2 关闭/美卓控制/智能控制
    curr_control_value: 当前设定值(由上层控制器写入，顶层控制器为预设值)
    final_control_value: 最终目标值

    """
    point_id = mongoengine.IntField(max_value=1024)
    time = mongoengine.DateTimeField()
    auto_setting = mongoengine.IntField(max_value=1024)
    control_mode = mongoengine.IntField(max_value=1024)
    curr_control_value = mongoengine.FloatField(max_value=10240000)
    final_control_value = mongoengine.FloatField(max_value=10240000)


class GmsControl(mongoengine.Document):
    """历史控制数据表

    point_id: 点位编号
    time：写入时间
    auto_setting： 当前目标值是否是智能推荐的，（否=手动设置
    control_mode： 0/1/2 关闭/美卓控制/智能控制
    curr_control_value: 当前设定值(由上层控制器写入，顶层控制器为预设值)
    final_control_value: 最终目标值
    """
    point_id = mongoengine.IntField(max_value=1024)
    time = mongoengine.DateTimeField()
    auto_setting = mongoengine.IntField(max_value=1024)
    control_mode = mongoengine.IntField(max_value=1024)
    curr_control_value = mongoengine.FloatField(max_value=10240000)
    final_control_value = mongoengine.FloatField(max_value=10240000)


class RecommendLog(mongoengine.Document):
    """推荐日志表
    point: ReferenceField(Point)外键
    point_id: 操作点位
    time: 操作时间
    control_value: 修改后参数值
    """
    point = mongoengine.ReferenceField(Point)
    point_id = mongoengine.IntField(max_value=1024)
    time = mongoengine.DateTimeField()
    control_value = mongoengine.FloatField(max_value=2048)


class WarningLog(mongoengine.Document):
    """报警日志表
    point: ReferenceField(Point)外键
    point_id: 报警点位
    instrument: 点位所属仪器名称
    time: 报警时间
    principal: 充填负责人
    value_min: 正常工作最小值
    value_max: 正常工作最大值
    Monitoring_value: 监测值
    """
    point = mongoengine.ReferenceField(Point)
    point_id = mongoengine.IntField(max_value=1024)
    instrument = mongoengine.StringField(max_length=100)
    time = mongoengine.DateTimeField()
    principal = mongoengine.IntField(max_value=1024)
    value_min = mongoengine.FloatField(max_value=2048)
    value_max = mongoengine.FloatField(max_value=2048)
    Monitoring_value = mongoengine.FloatField(max_value=10240000)
    fill_id = mongoengine.IntField(max_value=1024)

    meta = {
        'ordering': ['-time']
    }


class StopeManage(mongoengine.Document):
    """采场管理表

    serial: 采场编号
    creator: 创建人
    time: 创建时间
    fill_volume: 目前填充量
    fill_height: 目前充填高度
    flow: 充填流量
    status: 采场状态(1/2/3/4/5, 1:填充中 2:暂停中 3:空闲中 4:初始化中 5:初始化失败)
    location: 采区类型(1/2, 1表示南采区，2表示北采区)
    name: 采场名称
    fill_round: 充填轮次

    model: 采场模型文件
    origin_model: 原始模型文件
    simplified_model: 简化模型文件
    overall_height: 采场总高度
    average_height: 平均每片高度
    overall_volume: 采场总体积
    segment_volume: 各片体积
    is_deleted: 是否被删除
    """
    serial = mongoengine.IntField(max_value=65536)
    creator = mongoengine.StringField(max_length=20)
    time = mongoengine.DateTimeField()
    fill_volume = mongoengine.FloatField(max_value=65536, default=0.0)
    fill_height = mongoengine.FloatField(max_value=65536, default=0.0)
    flow = mongoengine.FloatField(max_value=65536, default=0.0)
    status = mongoengine.IntField(max_value=1024, default=4)
    location = mongoengine.IntField(max_value=1024)
    name = mongoengine.StringField(max_length=20)
    fill_round = mongoengine.IntField(max_value=1024, default=0)
    model = mongoengine.FileField()
    origin_model = mongoengine.FileField()
    simplified_model = mongoengine.FileField()
    overall_height = mongoengine.FloatField(max_value=1024)
    average_height = mongoengine.FloatField(max_value=1024)
    overall_volume = mongoengine.FloatField(max_value=65536, default=1.0)
    segment_volume = mongoengine.ListField(mongoengine.FloatField(max_value=65536))
    is_deleted = mongoengine.BooleanField(default=False)


class TunnelManage(mongoengine.Document):
    """
    巷道管理表

    serial: 巷道编号
    creator: 创建人
    name: 巷道名称
    time: 创建时间
    location: 采区类型(1/2, 1表示采区，2表示北采区)
    is_deleted: 是否被删除
    origin_model: 原始模型文件
    simplified_model: 简化模型文件
    """
    serial = mongoengine.IntField(max_value=65536)
    creator = mongoengine.StringField(max_length=20)
    name = mongoengine.StringField(max_length=20)
    time = mongoengine.DateTimeField()
    location = mongoengine.IntField(max_value=1024)
    is_deleted = mongoengine.BooleanField(default=False)
    origin_model = mongoengine.FileField()
    simplified_model = mongoengine.FileField()


class TunnelManage(mongoengine.Document):
    """
    巷道管理表

    serial: 巷道编号
    creator: 创建人
    name: 巷道名称
    time: 创建时间
    location: 采区类型(1/2, 1表示采区，2表示北采区)
    is_deleted: 是否被删除
    origin_model: 原始模型文件
    simplified_model: 简化模型文件
    """
    serial = mongoengine.IntField(max_value=65536)
    creator = mongoengine.StringField(max_length=20)
    name = mongoengine.StringField(max_length=20)
    time = mongoengine.DateTimeField()
    location = mongoengine.IntField(max_value=1024)
    is_deleted = mongoengine.BooleanField(default=False)
    origin_model = mongoengine.FileField()
    simplified_model = mongoengine.FileField()


class ConfigParm(mongoengine.Document):
    """配置参数表

    intelligent_recommend: 智能推荐(是/否)
    a_fill_serial: A系统的充填采场编号
    a_fill_round: A系统的充填轮次
    a_fill_id: A系统的充填作业ID(外键)
    b_fill_serial: B系统的充填采场编号
    b_fill_round: B系统的充填轮次
    b_fill_id: B系统的充填作业ID(外键)
    """
    intelligent_recommend = mongoengine.BooleanField()
    a_fill_serial = mongoengine.IntField(max_value=1024)
    a_fill_round = mongoengine.IntField(max_value=16)
    a_fill_id = mongoengine.IntField()
    b_fill_serial = mongoengine.IntField(max_value=1024)
    b_fill_round = mongoengine.IntField(max_value=16)
    b_fill_id = mongoengine.IntField()


class BackfillRecord(mongoengine.Document):
    """充填记录表
    表中所有记录针对的是采场的某一次充填记录，故start_*、end_*指的都是总采场的变量

    fill_id: 充填作业ID(主键，唯一区分)
    serial: 采场编号
    account: 创建人
    time: 采场创建时间
    fill_round: 充填轮次
    start_time: 起始时间
    start_volume: 充填前充填方量
    start_height: 充填前高度
    start_percent: 充填前进度百分比

    end_time: 结束时间
    end_volume: 充填后充填方量
    end_height: 充填后高度
    end_percent: 充填后进度百分比
    fill_time: 充填时长
    volume_increase: 充填方量增量  # NOTE:应为current_volume，意思是当前采场总填充量
    hight_increase: 充填高度增量
    percent_increase: 充填进度百分比增量

    cement_used: 水泥消耗
    csr_set: 灰砂比设定值(cement-sand ratio)
    csr_actual: 实际灰砂比
    csr_deviation: 水泥添加误差

    flocculant_used: 絮凝剂消耗
    ftr_set: 絮凝剂消耗比设定值(flocculant-tailings ratio)

    ftr_actual: 实际絮凝剂消耗比
    ftr_deviation: 絮凝剂添加误差

    tailings_weight: 尾砂干重

    uc_mean: 底流浓度均值(Underflow concentration)
    uc_std: 底流浓度标准差
    uc_25: 底流浓度25%分位点
    uc_75: 底流浓度75%分位点
    fill_mean: 充填浓度均值(Underflow concentration)
    fill_std: 充填浓度标准差
    fill_25: 充填浓度25%分位点
    fill_75: 充填浓度75%分位点
    strength: 强度检测
    active_state: 激活状态
    system: 使用的系统ID

    ---
    以下为充填任务管理模块优化新加
    > https://gitee.com/USTB1001/nfcaindustrialintelligent/issues/I1AZIM?from=project-issue
    ---
    fill_name: 充填任务名称
    thickener_id: 浓密机选择(1/2)
    mixer_id: 搅拌机选择(1/2)
    fill_status: 当前任务状态(1/2/3 - online/pause/stop)
    is_deleted: 是否被删除(y/n)

    """
    fill_id = mongoengine.IntField()
    serial = mongoengine.IntField(max_value=1024)
    account = mongoengine.StringField(max_length=1024)
    time = mongoengine.DateTimeField()
    fill_round = mongoengine.IntField(max_value=100)
    start_time = mongoengine.DateTimeField()
    start_volume = mongoengine.FloatField()
    start_height = mongoengine.FloatField()
    start_percent = mongoengine.FloatField(max_value=100)
    end_time = mongoengine.DateTimeField()
    end_volume = mongoengine.FloatField()
    end_height = mongoengine.FloatField()
    end_percent = mongoengine.FloatField(max_value=100)
    fill_time = mongoengine.FloatField()
    volume_increase = mongoengine.FloatField()
    hight_increase = mongoengine.FloatField()
    percent_increase = mongoengine.FloatField(max_value=100)
    cement_used = mongoengine.FloatField()
    csr_start_set = mongoengine.FloatField(max_value=1024)
    csr_end_set = mongoengine.FloatField(max_value=1024)
    csr_set = mongoengine.FloatField(max_value=1024)
    csr_actual = mongoengine.FloatField(max_value=1024)
    csr_deviation = mongoengine.FloatField()
    flocculant_used = mongoengine.FloatField()
    ftr_set = mongoengine.FloatField(max_value=1024)
    ftr_actual = mongoengine.FloatField(max_value=1024)
    ftr_deviation = mongoengine.FloatField()
    tailings_weight = mongoengine.FloatField()
    uc_mean = mongoengine.FloatField()
    uc_std = mongoengine.FloatField()
    uc_25 = mongoengine.FloatField()
    uc_75 = mongoengine.FloatField()
    fill_mean = mongoengine.FloatField()
    fill_std = mongoengine.FloatField()
    fill_25 = mongoengine.FloatField()
    fill_75 = mongoengine.FloatField()
    strength = mongoengine.FloatField()

    active_state = mongoengine.BooleanField(defult=True)
    system = mongoengine.IntField(max_value=3)

    uc_count = mongoengine.IntField()
    fill_count = mongoengine.IntField()
    uc_square_mean = mongoengine.FloatField()
    fill_square_mean = mongoengine.FloatField()

    fill_name = mongoengine.StringField()
    thickener_id = mongoengine.IntField(max_value=1024)
    mixer_id = mongoengine.IntField(max_value=1024)
    fill_status = mongoengine.IntField(max_value=4)
    is_deleted = mongoengine.BooleanField(defult=False)


class MissionOperation(mongoengine.Document):
    """任务操作表（原设备模式表）

    user_id: 当前用户id
    username: 当前用户名字
    time: 修改时间
    ---
    以下为充填任务管理模块优化新加
    > https://gitee.com/USTB1001/nfcaindustrialintelligent/issues/I1AZIM?from=project-issue
    ---
    fill_id: 操作的任务ID(充填记录表中fill_id的外键，外键直接获取为Mongodb自带的"_id"，fill_id需要间接获取)
    content: 修改的内容(一个json，用于存储被修改的内容)
    content举例：
        "{
            "被修改的字段1":{
                "old":yyy,
                "new":xxx
            },
            "被修改的字段2":{
                "old":yyy,
                "new":xxx
            }
        }"
    """
    user_id = mongoengine.IntField(max_value=1024)
    username = mongoengine.StringField(max_length=16)
    time = mongoengine.DateTimeField()
    fill_id = mongoengine.ReferenceField(BackfillRecord)
    content = mongoengine.DictField()


class GmsNow(mongoengine.Document):
    """现在时刻数据表

    point_id: 点位编号
    point: ReferenceField(Point)外键
    instrument: 点位所属仪器名称
    time: 监测时间
    Monitoring_value: 监测值
    alarm: 是否报警

    ---
    以下为充填任务管理模块优化新加
    > https://gitee.com/USTB1001/nfcaindustrialintelligent/issues/I1AZIM?from=project-issue
    ---
    fill_id: 操作的任务ID(充填记录表中fill_id的外键，外键直接获取为Mongodb自带的"_id"，fill_id需要间接获取)
    """
    point_id = mongoengine.IntField(max_value=1024)
    point = mongoengine.ReferenceField(Point)
    instrument = mongoengine.StringField(max_length=100)
    time = mongoengine.DateTimeField()
    Monitoring_value = mongoengine.FloatField(max_value=2048)
    alarm = mongoengine.BooleanField()
    state = mongoengine.StringField(max_length=20)
    fill_id = mongoengine.ReferenceField(BackfillRecord)


class GmsMonitor(mongoengine.Document):
    """历史数据表

    point: ReferenceField(Point)外键
    point_id: 点位编号
    instrument: 点位所属仪器名称
    time: 监测时间
    Monitoring_value: 监测值
    alarm: 是否报警

    ---
    以下为充填任务管理模块优化新加
    > https://gitee.com/USTB1001/nfcaindustrialintelligent/issues/I1AZIM?from=project-issue
    ---
    fill_id: 操作的任务ID(充填记录表中fill_id的外键，外键直接获取为Mongodb自带的"_id"，fill_id需要间接获取)
    """
    point = mongoengine.ReferenceField(Point)
    point_id = mongoengine.IntField(max_value=1024)
    instrument = mongoengine.StringField(max_length=100)
    time = mongoengine.DateTimeField()
    Monitoring_value = mongoengine.FloatField(max_value=10240000)
    alarm = mongoengine.BooleanField()
    fill_id = mongoengine.ReferenceField(BackfillRecord)
    state = mongoengine.StringField(max_length=100)

    # meta = {
    #     'ordering': ['-time']
    # }


class GmsDenEven(mongoengine.Document):
    """浓度均匀度数据表

    instrument_serial: 设备编号
    density: 浓度
    evenness: 均匀度
    time: 监测时间
    """
    instrument_serial = mongoengine.IntField(max_value=1024)
    density = mongoengine.FloatField(max_value=100)
    evenness = mongoengine.FloatField(max_value=100)
    time = mongoengine.DateTimeField()


class GmsDenEvenNow(mongoengine.Document):
    """最新浓度均匀度数据表

    instrument_serial: 设备编号
    density: 浓度
    evenness: 均匀度
    time: 监测时间
    """
    instrument_serial = mongoengine.IntField(max_value=1024)
    density = mongoengine.FloatField(max_value=100)
    evenness = mongoengine.FloatField(max_value=100)
    time = mongoengine.DateTimeField()


class BackfillLog(mongoengine.Document):
    '''填充日志

    fill_id: 填充任务id
    time: 记录时间
    fill_height: 已填充高度
    fill_volume: 已填充体积
    fill_percent: 填充百分比
    remaining_time: 剩余填充时间
    alarm: 报警内容
    '''
    fill_id = mongoengine.ReferenceField(BackfillRecord)
    time = mongoengine.DateTimeField()
    fill_height = mongoengine.FloatField()
    fill_volume = mongoengine.FloatField()
    fill_percent = mongoengine.FloatField()
    remaining_time = mongoengine.FloatField()
    flow = mongoengine.FloatField()
    alarm = mongoengine.StringField()
