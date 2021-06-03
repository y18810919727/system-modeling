

# Create your models here.
class MyModel:
    necessary_attr = None

    def my_to_dict(self, without=None):
        content = {}
        for key, value in vars(self).items():
            content[key] = value
        if without is None:
            without = []
        for w in without:
            if hasattr(content, w):
                del content[w]
        del content['_state']
        return content

    def my_update(self, update_dict: dict, necessary_attr=None):
        if necessary_attr is None:
            if self.necessary_attr is None:
                pass
            else:
                necessary_attr = []
                for n in self.necessary_attr:
                    necessary_attr.append(n.attname)
        key_list = []
        for key, value in update_dict.items():
            key_list.append(key)
        missing_attr = list(set(necessary_attr).difference(set(key_list)))
        error_info = ''
        for m in missing_attr:
            error_info = error_info + m + ' '
        if len(missing_attr) != 0:  # 差集为空
            raise Exception('missing:' + error_info)
        for key, value in update_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.save()

    # 只用于简单的删除，删除的其他逻辑还得自己写
    def my_delete(self):
        setattr(self, 'yn', False)
        self.save()

    class Meta:
        abstract = True

    # 返回self._meta.fields中没有的，但是又是需要的字段名的列表
    # 形如['name','type']
    def getMtMField(self):
        pass

    # 返回需要在json中忽略的字段名的列表
    # 形如['password']
    def getIgnoreList(self):
        pass

    def isAttrInstance(self, attr, clazz):
        return isinstance(getattr(self, attr), clazz)

    def getDict(self):
        fields = []
        for field in self._fields:
            if field is not 'id' and 'fill_id':
                fields.append(field)

        d = {}
        import datetime
        for attr in fields:
            if isinstance(getattr(self, attr), datetime.datetime):
                d[attr] = getattr(self, attr).strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(getattr(self, attr), datetime.date):
                d[attr] = getattr(self, attr).strftime('%Y-%m-%d')
            # 特殊处理datetime的数据
            elif isinstance(getattr(self, attr), MyModel):
                d[attr] = getattr(self, attr).getDict()
            # 递归生成BaseModel类的dict
            elif self.isAttrInstance(attr, int) or self.isAttrInstance(attr, float) \
                    or self.isAttrInstance(attr, str):
                d[attr] = getattr(self, attr)
            else:
                d[attr] = getattr(self, attr)

        mAttr = self.getMtMField()
        if mAttr is not None:
            for m in mAttr:
                if hasattr(self, m):
                    attlist = getattr(self, m).all()
                    l = []
                    for attr in attlist:
                        if isinstance(attr, MyModel):
                            l.append(attr.getDict())
                        else:
                            dic = attr.__dict__
                            if '_state' in dic:
                                dic.pop('_state')
                            l.append(dic)
                    d[m] = l
        # 由于ManyToMany类不能存在于_meat.fields，因而子类需要在getMtMFiled中返回这些字段
        if 'basemodel_ptr' in d:
            d.pop('basemodel_ptr')

        ignoreList = self.getIgnoreList()
        if ignoreList is not None:
            for m in ignoreList:
                if d.get(m) is not None:
                    d.pop(m)
        # 移除不需要的字段
        return d

    def toJSON(self):
        import json
        return json.dumps(self.getDict(), ensure_ascii=False).encode('utf-8').decode()
